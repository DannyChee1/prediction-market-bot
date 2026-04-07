#!/usr/bin/env python3
"""
Oracle Lag Backtest

Tests whether Chainlink oracle staleness is a predictive edge on Polymarket BTC Up/Down markets.

Methodology:
  - Chainlink updates on-chain only when price moves >0.5% (CHAINLINK_TRIGGER_FRAC)
    or 1hr heartbeat. Consecutive identical chainlink_price seconds = oracle is "stale".
  - We proxy oracle lag as: staleness_streak = consecutive seconds with unchanged CL price
    at the decision point (default: 700s remaining = ~3m into a 15m window).
  - Signal: when oracle is stale AND the last CL price move was UP (or DOWN), bet that
    the market will resolve in the same direction (oracle is lagging behind real price).
  - We compare this signal vs baseline (bet randomly / always UP) across all windows.

Usage:
    python lag_backtest.py
    python lag_backtest.py --asset btc --timeframe 15m
    python lag_backtest.py --asset eth --timeframe 5m
    python lag_backtest.py --decision-tau 300 --min-staleness 30
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR    = Path("data")
VIG         = 0.02          # Polymarket vig (2% each side = ~4% round trip)
BET_USD     = 10.0          # fixed bet size per trade
DECISION_TAU = 700          # seconds remaining when we "decide" (early in window)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_windows(data_subdir: str, decision_tau: int = DECISION_TAU) -> list[dict]:
    """Load all parquet files and group into per-window summaries."""
    d = DATA_DIR / data_subdir
    if not d.exists():
        raise FileNotFoundError(f"No data at {d} — run live_trader.py with --record first.")

    files = sorted(d.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {d}.")

    print(f"Loading {len(files)} parquet files from {d} ...")
    frames = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f))
        except Exception as e:
            print(f"  Skipping {f.name}: {e}")

    if not frames:
        raise RuntimeError("All parquet files failed to load.")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("ts_ms").reset_index(drop=True)
    print(f"  {len(df):,} total rows across all windows")

    # Group by window start timestamp
    df["window_key"] = df["window_start_ms"].astype(str) + "_" + df["market_slug"]
    windows = []
    for wkey, grp in df.groupby("window_key"):
        grp = grp.sort_values("ts_ms").reset_index(drop=True)
        w = _extract_window(grp, decision_tau)
        if w is not None:
            windows.append(w)

    print(f"  {len(windows)} valid windows extracted")
    return windows


def _extract_window(grp: pd.DataFrame, decision_tau: int = DECISION_TAU) -> dict | None:
    """Extract features from a single window's row group."""
    if len(grp) < 30:
        return None

    slug = grp["market_slug"].iloc[0]
    start_ms  = int(grp["window_start_ms"].iloc[0])
    end_ms    = int(grp["window_end_ms"].iloc[0])
    duration_s = (end_ms - start_ms) / 1000

    # Need at least 80% of expected rows
    expected_rows = duration_s
    if len(grp) < expected_rows * 0.80:
        return None

    start_price = grp["window_start_price"].dropna().iloc[0] if grp["window_start_price"].notna().any() else None
    if start_price is None:
        return None

    # End price: last Chainlink reading near window end
    # Old parquets used "chainlink_btc"; new ones use "chainlink_price"
    if "chainlink_price" in grp.columns:
        cl_col = "chainlink_price"
    elif "chainlink_btc" in grp.columns:
        cl_col = "chainlink_btc"
    else:
        return None
    end_price_rows = grp[grp["time_remaining_s"] <= 10][cl_col].dropna()
    if end_price_rows.empty:
        end_price_rows = grp[cl_col].dropna()
    if end_price_rows.empty:
        return None
    end_price = float(end_price_rows.iloc[-1])

    outcome_up = int(end_price >= float(start_price))

    # Find the decision row closest to DECISION_TAU
    dec_rows = grp[(grp["time_remaining_s"] - DECISION_TAU).abs() <= 30]
    if dec_rows.empty:
        dec_rows = grp[grp["time_remaining_s"] <= DECISION_TAU + 60]
        if dec_rows.empty:
            return None
        dec_row = dec_rows.iloc[0]
    else:
        dec_row = dec_rows.iloc[(dec_rows["time_remaining_s"] - DECISION_TAU).abs().argsort().iloc[0]]

    dec_idx  = dec_row.name
    dec_ts   = int(dec_row["ts_ms"])
    dec_cl   = dec_row[cl_col] if pd.notna(dec_row[cl_col]) else None
    if dec_cl is None:
        return None

    # Compute staleness: consecutive seconds before decision with unchanged CL price
    before_dec = grp[grp.index <= dec_idx].copy()
    cl_vals = before_dec[cl_col].values
    staleness = 0
    for i in range(len(cl_vals) - 1, 0, -1):
        if pd.isna(cl_vals[i]) or pd.isna(cl_vals[i-1]):
            break
        if cl_vals[i] == cl_vals[i-1]:
            staleness += 1
        else:
            break

    # Direction of last CL price change before decision
    cl_changes = before_dec[cl_col].dropna()
    cl_diffs = cl_changes.diff().dropna()
    nonzero = cl_diffs[cl_diffs != 0]
    last_cl_direction = None
    if not nonzero.empty:
        last_cl_direction = 1 if nonzero.iloc[-1] > 0 else -1

    # Binance mid at decision (if available)
    binance_mid = None
    if "binance_mid" in grp.columns and pd.notna(dec_row.get("binance_mid")):
        binance_mid = float(dec_row["binance_mid"])

    return {
        "slug":              slug,
        "start_ms":          start_ms,
        "end_ms":            end_ms,
        "duration_s":        duration_s,
        "start_price":       float(start_price),
        "end_price":         end_price,
        "outcome_up":        outcome_up,
        "dec_cl":            float(dec_cl),
        "dec_ts":            dec_ts,
        "staleness_s":       staleness,
        "last_cl_direction": last_cl_direction,
        "binance_mid":       binance_mid,
        "n_rows":            len(grp),
    }


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate(windows: list[dict], min_staleness: int) -> dict:
    """Simulate betting on stale-oracle windows and return PnL stats."""
    trades = []
    for w in windows:
        if w["staleness_s"] < min_staleness:
            continue
        if w["last_cl_direction"] is None:
            continue
        # Signal: bet in the direction CL last moved (stale = lagging the real price)
        bet_up = (w["last_cl_direction"] == 1)
        won = (bet_up == bool(w["outcome_up"]))
        # Simple binary payout: win = $1/share * shares, lose = $0
        # Assuming entry at ~0.50 (midpoint), 2% vig → effective entry ~0.51
        entry_price = 0.51  # approximate with vig
        shares = BET_USD / entry_price
        payout = shares * (1.0 if won else 0.0)
        pnl = payout - BET_USD
        trades.append({
            "slug":       w["slug"],
            "start_ms":   w["start_ms"],
            "staleness":  w["staleness_s"],
            "bet_up":     bet_up,
            "outcome_up": w["outcome_up"],
            "won":        won,
            "pnl":        pnl,
        })

    if not trades:
        return {"n": 0, "win_rate": float("nan"), "net_pnl": 0.0, "sharpe": float("nan")}

    pnls = np.array([t["pnl"] for t in trades])
    n = len(pnls)
    win_rate = float(np.mean([t["won"] for t in trades]))
    net_pnl  = float(pnls.sum())
    sharpe   = float(pnls.mean() / pnls.std()) * math.sqrt(n) if pnls.std() > 0 else float("nan")

    return {
        "n":        n,
        "win_rate": win_rate,
        "net_pnl":  net_pnl,
        "sharpe":   sharpe,
    }


def baseline(windows: list[dict]) -> dict:
    """Baseline: always bet UP, no oracle filter."""
    trades = []
    for w in windows:
        entry_price = 0.51
        shares = BET_USD / entry_price
        won = bool(w["outcome_up"])
        pnl = shares * (1.0 if won else 0.0) - BET_USD
        trades.append({"won": won, "pnl": pnl})

    pnls = np.array([t["pnl"] for t in trades])
    n = len(pnls)
    win_rate = float(np.mean([t["won"] for t in trades]))
    net_pnl  = float(pnls.sum())
    sharpe   = float(pnls.mean() / pnls.std()) * math.sqrt(n) if pnls.std() > 0 else float("nan")
    return {"n": n, "win_rate": win_rate, "net_pnl": net_pnl, "sharpe": sharpe}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--asset",          default="btc",  choices=["btc", "eth", "sol", "xrp"])
    p.add_argument("--timeframe",      default="15m",  choices=["15m", "5m"])
    p.add_argument("--decision-tau",   type=int, default=DECISION_TAU,
                   help="Seconds remaining at decision point (default 700)")
    p.add_argument("--min-staleness",  type=int, default=None,
                   help="Only test one staleness threshold instead of sweeping")
    args = p.parse_args()

    decision_tau = args.decision_tau

    subdir = f"{args.asset}_{args.timeframe}"
    windows = load_windows(subdir, decision_tau)

    if not windows:
        print("No valid windows. Exiting.")
        return

    print(f"\n{'='*70}")
    print(f"ORACLE LAG BACKTEST — {subdir.upper()}")
    print(f"  Decision tau:  {decision_tau}s remaining")
    print(f"  Bet size:      ${BET_USD:.0f} per window")
    print(f"  Total windows: {len(windows)}")

    # Outcome distribution
    n_up = sum(w["outcome_up"] for w in windows)
    print(f"  Outcome UP:    {n_up}/{len(windows)} ({n_up/len(windows)*100:.1f}%)")

    # Baseline
    bl = baseline(windows)
    print(f"\nBaseline (always UP):  n={bl['n']}  win={bl['win_rate']:.3f}  "
          f"pnl=${bl['net_pnl']:.2f}  sharpe={bl['sharpe']:.2f}")

    # Staleness sweep
    thresholds = [args.min_staleness] if args.min_staleness else [5, 10, 20, 30, 60, 120, 300]

    print(f"\n{'staleness_thresh':>17} {'n_windows':>10} {'win_rate':>9} {'net_pnl':>9} {'sharpe':>8}")
    print("-" * 60)

    best_sharpe = -999
    best_threshold = None
    for thresh in thresholds:
        r = simulate(windows, thresh)
        if r["n"] == 0:
            print(f"  ≥{thresh:>4}s          {'0':>10}  {'  —':>9}  {'  —':>9}  {'  —':>8}")
            continue
        flag = " ← best" if r["sharpe"] > best_sharpe else ""
        if r["sharpe"] > best_sharpe:
            best_sharpe = r["sharpe"]
            best_threshold = thresh
        print(f"  ≥{thresh:>4}s          {r['n']:>10}  {r['win_rate']:>9.3f}  "
              f"${r['net_pnl']:>8.2f}  {r['sharpe']:>8.2f}{flag}")

    # Binance mid availability check
    has_binance = sum(1 for w in windows if w["binance_mid"] is not None)
    print(f"\nNote: {has_binance}/{len(windows)} windows have binance_mid data.")
    if has_binance == 0:
        print("      Run live_trader.py to collect new parquet data with binance_mid column.")

    # Stopping criterion
    print(f"\n{'='*70}")
    print("STOPPING CRITERION")
    if best_threshold is not None and best_sharpe >= 0.5:
        print(f"  Best Sharpe: {best_sharpe:.2f} at ≥{best_threshold}s staleness")
        if best_sharpe >= 1.0:
            print("  STATUS: Edge confirmed — Sharpe ≥ 1.0. Consider live deployment.")
        else:
            print("  STATUS: Marginal edge (0.5 ≤ Sharpe < 1.0). Collect more data.")
    else:
        print(f"  Best Sharpe: {best_sharpe:.2f}")
        print("  STATUS: STOP — Sharpe < 0.5 across all thresholds. Oracle lag edge not confirmed.")


if __name__ == "__main__":
    main()
