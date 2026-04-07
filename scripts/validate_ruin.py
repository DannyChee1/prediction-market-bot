#!/usr/bin/env python3
"""
3g — Fee-adjusted gambler's ruin (Quant Guild #28).

Question:  Given the strategy's win rate, average win, average loss,
AND the fees that have already been paid on each fill, what is the
probability of going broke before reaching some target bankroll?

Method:    Gambler's-ruin probability for a biased random walk:

    P(ruin) = (1 - r^i) / (1 - r^N)         if r ≠ 1
            = 1 - i/N                       if r = 1

where:
    r = q / p              (loss-to-win odds ratio of EQUAL-stake bet)
    p = empirical fee-adjusted win prob
    q = 1 - p
    i = current bankroll in "units" (= bankroll / avg_loss_amount)
    N = ruin barrier

For a *binary* prediction market the fee is already in `cost_usd`
(it's the entry price plus the per-share Polymarket fee), so the
empirical avg_win and avg_loss come out fee-adjusted automatically.
We compute (W, L) directly from the trade dump and feed them in.

Result interpretation:
  * r > 1  → strategy is unfavorable (more losses-by-amount than wins)
              → ruin is asymptotically certain
  * r < 1  → strategy is favorable
              → ruin probability decays with starting capital
  * r = 1  → fair game
              → ruin probability is purely 1 - i/N

Reads:     validation_runs/<tag>/trades_test.parquet
Writes:    validation_runs/<tag>/gamblers_ruin.json
Prints:    a summary table

Usage:
    python scripts/validate_ruin.py validation_runs/postfix_btc_5m
    python scripts/validate_ruin.py --target-multiple 10 \
        validation_runs/postfix_btc
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def gamblers_ruin_probability(p: float, q: float, units_initial: int,
                              units_target: int) -> float:
    """Classic gambler's ruin. Returns P(ruin before reaching target)."""
    if p <= 0:
        return 1.0
    if q <= 0:
        return 0.0
    r = q / p
    i = units_initial
    N = units_target
    if i <= 0:
        return 1.0
    if i >= N:
        return 0.0
    if abs(r - 1.0) < 1e-12:
        return 1.0 - i / N
    # Numerically: use log to avoid overflow at large N
    # P(ruin) = (r^N - r^i) / (r^N - 1)   (alternative form)
    if r > 1.0 and N > 100:
        # log domain
        log_r = math.log(r)
        log_num = log_r * N + math.log1p(-math.exp(log_r * (i - N)))
        log_den = log_r * N + math.log1p(-math.exp(-log_r * N))
        return math.exp(log_num - log_den)
    return (r**N - r**i) / (r**N - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag_dir", type=Path)
    ap.add_argument("--target-multiple", type=float, default=2.0,
                    help="Target bankroll = target_multiple × initial. "
                         "Default 2.0 (i.e. 'double your money').")
    args = ap.parse_args()

    tag_dir = args.tag_dir.resolve()
    trades_path = tag_dir / "trades_test.parquet"
    if not trades_path.exists():
        print(f"ERROR: {trades_path} not found", file=sys.stderr)
        sys.exit(1)

    trades = pd.read_parquet(trades_path)
    if trades.empty:
        print(f"ERROR: empty trade dump", file=sys.stderr)
        sys.exit(1)

    # Read initial bankroll from metrics.json
    metrics_path = tag_dir / "metrics.json"
    initial_bk = 10_000.0
    if metrics_path.exists():
        with open(metrics_path) as f:
            initial_bk = float(json.load(f)["args"]["bankroll"])

    pnls = trades["pnl"].to_numpy(dtype=np.float64)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    n = len(pnls)
    n_wins = len(wins)
    n_losses = len(losses)

    if n_wins == 0 or n_losses == 0:
        print("ERROR: need at least one win and one loss", file=sys.stderr)
        sys.exit(1)

    p_win = n_wins / n
    q_lose = 1.0 - p_win
    avg_win = float(wins.mean())
    avg_loss = float(-losses.mean())  # positive

    # Fees already baked into cost_usd, so avg_win/avg_loss are net of
    # entry fees. Polymarket has no resolution fee on these markets.
    # If you ever add a resolution-time fee, subtract it from avg_win
    # and add it to avg_loss here.

    # Bankroll in "loss units": how many average losses can the
    # initial bankroll absorb before zero?
    units_initial = max(int(round(initial_bk / avg_loss)), 1)

    # Per-trade EV in dollars
    ev_per_trade = p_win * avg_win - q_lose * avg_loss
    # "favorable r": Quant Guild defines this as r = (q*L) / (p*W).
    # r < 1 ⇒ favorable, r > 1 ⇒ unfavorable.
    r_favor = (q_lose * avg_loss) / (p_win * avg_win)

    print(f"Loaded {n} trades from {trades_path.name}")
    print(f"  Initial bankroll:    ${initial_bk:,.2f}")
    print(f"  p_win:               {p_win:.4f}")
    print(f"  avg win  ($, net):   {avg_win:+.2f}")
    print(f"  avg loss ($, net):   {-avg_loss:+.2f}")
    print(f"  Per-trade EV:        ${ev_per_trade:+.2f}")
    print(f"  Favorability r = qL/pW: {r_favor:.4f}  "
          f"({'FAVORABLE' if r_favor < 1 else 'UNFAVORABLE'})")
    print(f"  Initial bankroll = {units_initial} loss-units")
    print()

    if r_favor >= 1.0:
        print("  ⚠ r >= 1: strategy is unfavorable per dollar staked.")
        print("    Gambler's ruin is asymptotically certain at any sizing")
        print("    that is approximately equal-stakes.")
        ruin_results = {}
        for target in (1.5, 2.0, 5.0, 10.0):
            units_target = int(round(initial_bk * target / avg_loss))
            p_ruin = gamblers_ruin_probability(
                p_win, q_lose, units_initial, units_target
            )
            ruin_results[f"target_{target}x"] = {
                "units_initial": units_initial,
                "units_target": units_target,
                "p_ruin": p_ruin,
            }
            print(f"    Target {target:.1f}× (${initial_bk*target:,.0f}): "
                  f"P(ruin) = {p_ruin:.4f}")
    else:
        print("  ✓ r < 1: strategy is favorable per dollar staked.")
        print("  Gambler's-ruin probability vs target bankroll:")
        ruin_results = {}
        for target in (1.5, 2.0, 5.0, 10.0):
            units_target = int(round(initial_bk * target / avg_loss))
            p_ruin = gamblers_ruin_probability(
                p_win, q_lose, units_initial, units_target
            )
            ruin_results[f"target_{target}x"] = {
                "units_initial": units_initial,
                "units_target": units_target,
                "p_ruin": p_ruin,
            }
            print(f"    Target {target:.1f}× (${initial_bk*target:,.0f}): "
                  f"P(ruin) = {p_ruin:.6f}")
    print()

    # Persist
    out = {
        "n_trades": int(n),
        "p_win": float(p_win),
        "avg_win_usd": float(avg_win),
        "avg_loss_usd": float(avg_loss),
        "ev_per_trade_usd": float(ev_per_trade),
        "r_favorability": float(r_favor),
        "favorable": bool(r_favor < 1.0),
        "initial_bankroll_usd": float(initial_bk),
        "units_initial": int(units_initial),
        "ruin_by_target": ruin_results,
    }
    out_path = tag_dir / "gamblers_ruin.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  → wrote {out_path}")


if __name__ == "__main__":
    main()
