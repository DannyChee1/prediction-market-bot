#!/usr/bin/env python3
"""
3b — Bootstrap Kelly CI (Quant Guild #36 — Kelly + #28 — Gambler's Ruin).

Question:  What is the *uncertainty* on our optimal Kelly fraction given
the finite trade history we have? If the lower 5%-tile of bootstrapped
Kelly is below `kelly_fraction × point_estimate`, we are over-betting at
the lower bound and exposed to ruin in the worst-case parameter draw.

Method:    Resample the trade outcomes with replacement N times. For
each resample compute the empirical optimal Kelly fraction:

    f* = (p × W − q × L) / (W × L)

where p = empirical win rate, q = 1-p, W = avg win in % of stake,
L = avg loss in % of stake. Report 5/50/95 percentiles of f*.

Compared against the LIVE setting:
    kelly_fraction × max_kelly_per_trade

If the lower bound of bootstrapped Kelly is below the live setting,
print a warning. The notebook recommends the lower 5th percentile as
the *deployable* Kelly, not the point estimate.

Reads:     validation_runs/<tag>/trades_test.parquet
Writes:    validation_runs/<tag>/bootstrap_kelly.json
Prints:    a summary table

Usage:
    python scripts/validate_bootstrap_kelly.py validation_runs/postfix_btc_5m
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── Empirical Kelly from a trade list ────────────────────────────────────────


def _empirical_kelly(pnl_pcts: np.ndarray) -> float:
    """Return the empirical optimal Kelly fraction from a PnL-pct series.

    pnl_pcts is per-trade return as a fraction of the stake (e.g.
    +0.5 = 50% win, -1.0 = total loss). Uses the full Kelly formula
    f* = (p·W − q·L) / (W·L) where W and L are absolute average win
    and average |loss| respectively.

    Returns 0 if the formula is undefined (no wins, no losses, or
    negative-EV strategy).
    """
    if len(pnl_pcts) == 0:
        return 0.0
    wins = pnl_pcts[pnl_pcts > 0]
    losses = pnl_pcts[pnl_pcts < 0]
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    p = len(wins) / len(pnl_pcts)
    q = 1.0 - p
    W = float(wins.mean())
    L = float(-losses.mean())  # average loss as a positive number
    if W <= 0 or L <= 0:
        return 0.0
    f_star = (p * W - q * L) / (W * L)
    return max(0.0, f_star)  # negative Kelly = don't bet, clamp to 0


def bootstrap_kelly(
    pnl_pcts: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(pnl_pcts)

    point_kelly = _empirical_kelly(pnl_pcts)
    p_win = float((pnl_pcts > 0).mean())
    avg_win = float(pnl_pcts[pnl_pcts > 0].mean()) if (pnl_pcts > 0).any() else 0.0
    avg_loss = float(-pnl_pcts[pnl_pcts < 0].mean()) if (pnl_pcts < 0).any() else 0.0

    boot_kellys = np.empty(n_boot, dtype=np.float64)
    boot_p_wins = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = pnl_pcts[idx]
        boot_kellys[i] = _empirical_kelly(sample)
        boot_p_wins[i] = float((sample > 0).mean())

    return {
        "n_trades": int(n),
        "n_boot": int(n_boot),
        "point_kelly": float(point_kelly),
        "p_win": p_win,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "kelly_p05": float(np.percentile(boot_kellys, 5)),
        "kelly_p25": float(np.percentile(boot_kellys, 25)),
        "kelly_p50": float(np.percentile(boot_kellys, 50)),
        "kelly_p75": float(np.percentile(boot_kellys, 75)),
        "kelly_p95": float(np.percentile(boot_kellys, 95)),
        "kelly_mean": float(boot_kellys.mean()),
        "p_win_p05": float(np.percentile(boot_p_wins, 5)),
        "p_win_p95": float(np.percentile(boot_p_wins, 95)),
        "frac_kelly_zero": float((boot_kellys <= 0).mean()),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Bootstrap Kelly fraction CI from a trade dump")
    ap.add_argument("tag_dir", type=Path)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--live-kelly-fraction", type=float, default=0.25,
                    help="The kelly_fraction multiplier currently used in "
                         "DiffusionSignal (default 0.25 = quarter-Kelly)")
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

    # pnl_pct is what fraction of the *stake* the trade returned (signed)
    if "pnl_pct" not in trades.columns:
        # fall back: derive from pnl/cost_usd
        trades["pnl_pct"] = trades["pnl"] / trades["cost_usd"].replace(0, np.nan)
        trades.dropna(subset=["pnl_pct"], inplace=True)

    pnl_pcts = trades["pnl_pct"].to_numpy(dtype=np.float64)

    print(f"Loaded {len(pnl_pcts)} trades from {trades_path.name}")
    print(f"  Empirical p_win:    {(pnl_pcts > 0).mean():.4f}")
    print(f"  Empirical avg win:  {pnl_pcts[pnl_pcts > 0].mean():+.4f} (per stake)")
    print(f"  Empirical avg loss: {pnl_pcts[pnl_pcts < 0].mean():+.4f} (per stake)")
    print(f"  Running {args.n_boot} bootstrap resamples...")

    result = bootstrap_kelly(pnl_pcts, args.n_boot, args.seed)
    result["live_kelly_fraction"] = args.live_kelly_fraction
    result["live_effective_kelly"] = (
        args.live_kelly_fraction * result["point_kelly"]
    )

    print()
    print("=" * 64)
    print("  BOOTSTRAP KELLY CI")
    print("=" * 64)
    print(f"  N trades:          {result['n_trades']}")
    print(f"  Point Kelly f*:    {result['point_kelly']:.4f}  "
          f"(empirical optimum, full Kelly)")
    print(f"  Bootstrap CI:")
    print(f"    5%-ile:          {result['kelly_p05']:.4f}")
    print(f"    25%-ile:         {result['kelly_p25']:.4f}")
    print(f"    50%-ile:         {result['kelly_p50']:.4f}")
    print(f"    75%-ile:         {result['kelly_p75']:.4f}")
    print(f"    95%-ile:         {result['kelly_p95']:.4f}")
    print(f"  P(Kelly = 0 in resample): {result['frac_kelly_zero']:.4f}")
    print()
    print(f"  Live multiplier:   kelly_fraction × f* = "
          f"{args.live_kelly_fraction} × {result['point_kelly']:.4f} = "
          f"{result['live_effective_kelly']:.4f}")
    print()

    # Verdict
    safe_kelly_lower = result["kelly_p05"]
    live_eff = result["live_effective_kelly"]
    if safe_kelly_lower <= 0:
        print("  ⚠ WARNING: lower 5%-ile bootstrap Kelly is ≤ 0.")
        print("    This means in 5% of plausible alternative samples the")
        print("    strategy has no edge at all. Reduce position sizing.")
    elif live_eff > safe_kelly_lower:
        ratio = live_eff / safe_kelly_lower
        print(f"  ⚠ Live effective Kelly ({live_eff:.4f}) exceeds the "
              f"lower-CI safe Kelly ({safe_kelly_lower:.4f}) by "
              f"{ratio:.2f}×.")
        print(f"    Notebook #36 recommendation: cap kelly_fraction so that")
        print(f"    kelly_fraction × f* ≤ lower 5%-ile, i.e. set "
              f"kelly_fraction ≤ {safe_kelly_lower / result['point_kelly']:.4f}")
    else:
        print(f"  ✓ Live effective Kelly ({live_eff:.4f}) is within the "
              f"lower-CI safe range ({safe_kelly_lower:.4f}).")
        print(f"    Current kelly_fraction = {args.live_kelly_fraction} is "
              "conservative enough.")

    out_path = tag_dir / "bootstrap_kelly.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print()
    print(f"  → wrote {out_path}")


if __name__ == "__main__":
    main()
