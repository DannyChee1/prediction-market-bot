#!/usr/bin/env python3
"""Oracle lag empirical test.

The user asked: "Polymarket RTDS lags Binance by 1.23 seconds —
how does it work? Is this the same as oracle lag? We tested with
dashboard.py and saw ~0.02% lag."

Two distinct measurements:
  1. TIME lag — how many seconds Polymarket's chainlink rebroadcast is
     behind real-time Binance. Measured at ~1.23s constant in
     tasks/findings/feed_latency_2026-04-08.md.
  2. PRICE gap — how much (binance_mid - chainlink_price) / chainlink
     differs at any wall-clock moment. The user remembers ~0.02%.

These ARE the same phenomenon:
  - Binance reports the price NOW
  - Chainlink (via RTDS) reports the price ~1.23s ago
  - At any wall-clock moment, both feeds are visible
  - The gap = how much BTC moved in the last 1.23s
  - Average BTC vol per second is ~$5-15 → 1.23s ≈ $6-18 → on $73000 ≈ 0.008-0.025%
  - So 0.02% is exactly what we'd expect from a 1.23s lag in normal vol

This script:
  1. Reads recent btc_5m parquets
  2. Computes (binance_mid - chainlink_price) / chainlink_price for every row
  3. Reports the distribution
  4. Quantifies HOW OFTEN the gap exceeds tradeable thresholds (0.05%, 0.1%, 0.2%, 0.5%)
  5. Confirms the time-lag → price-gap relationship by checking BTC's
     1.23s realized move

Usage: python analysis/oracle_lag_test.py
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

ROOT = Path("/Users/dannychee/Desktop/prediction-market-bot")
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402


def main():
    data_dir = ROOT / "data" / "btc_5m"
    files = sorted(data_dir.glob("*.parquet"))[-300:]  # last 300 windows
    print(f"Loading {len(files)} most recent btc_5m parquets...", file=sys.stderr)

    gaps_pct = []   # (binance - chainlink) / chainlink × 100
    gaps_abs = []   # binance - chainlink in $
    delta_per_sec_abs = []  # |binance(t) - binance(t-1)| in $ — for the "what BTC does in 1s" baseline

    rows_total = 0
    rows_used = 0

    for f in files:
        try:
            df = pd.read_parquet(f, columns=["ts_ms", "binance_mid", "chainlink_price"])
        except Exception:
            continue
        df = df.dropna(subset=["binance_mid", "chainlink_price"])
        df = df[(df["binance_mid"] > 0) & (df["chainlink_price"] > 0)]
        if len(df) < 5:
            continue
        df = df.sort_values("ts_ms").reset_index(drop=True)
        rows_total += len(df)

        bn = df["binance_mid"].values
        cl = df["chainlink_price"].values
        ts = df["ts_ms"].values

        gap_pct = (bn - cl) / cl * 100.0
        gap_abs = bn - cl
        gaps_pct.extend(gap_pct.tolist())
        gaps_abs.extend(gap_abs.tolist())
        rows_used += len(bn)

        # 1-second realized binance move
        if len(bn) >= 2:
            for i in range(1, len(bn)):
                dt = (ts[i] - ts[i - 1]) / 1000.0
                if 0.5 <= dt <= 2.0:
                    delta_per_sec_abs.append(abs(bn[i] - bn[i - 1]) / dt)

    n = len(gaps_pct)
    if n == 0:
        print("No data found.", file=sys.stderr)
        return

    gaps_pct_arr = np.array(gaps_pct)
    gaps_abs_arr = np.array(gaps_abs)

    print(f"\n=== Oracle Lag Test — {n:,} rows from {len(files)} parquets ===\n")

    print("PRICE GAP: (binance_mid - chainlink_price) / chainlink × 100  [%]")
    print(f"  mean:           {gaps_pct_arr.mean():+.4f}%")
    print(f"  median:         {np.median(gaps_pct_arr):+.4f}%")
    print(f"  abs mean:       {np.abs(gaps_pct_arr).mean():.4f}%")
    print(f"  abs median:     {np.median(np.abs(gaps_pct_arr)):.4f}%")
    print(f"  std:            {gaps_pct_arr.std():.4f}%")
    print(f"  abs p50/p75/p90/p95/p99/max: "
          f"{np.percentile(np.abs(gaps_pct_arr), [50, 75, 90, 95, 99, 100]).round(4)}%")
    print()
    print("PRICE GAP in $:")
    print(f"  abs median:     ${np.median(np.abs(gaps_abs_arr)):.2f}")
    print(f"  abs p50/p75/p90/p95/p99/max: "
          f"{np.percentile(np.abs(gaps_abs_arr), [50, 75, 90, 95, 99, 100]).round(2)}")
    print()

    # Threshold counts — how often is the gap large enough to trade?
    print("How often does the gap exceed each threshold?")
    print(f"  |gap| > 0.02%  (~$15 on $73k): "
          f"{(np.abs(gaps_pct_arr) > 0.02).mean()*100:6.1f}%")
    print(f"  |gap| > 0.05%  (~$37 on $73k): "
          f"{(np.abs(gaps_pct_arr) > 0.05).mean()*100:6.1f}%")
    print(f"  |gap| > 0.10%  (~$73 on $73k): "
          f"{(np.abs(gaps_pct_arr) > 0.10).mean()*100:6.1f}%")
    print(f"  |gap| > 0.20%  (~$147): "
          f"{(np.abs(gaps_pct_arr) > 0.20).mean()*100:6.1f}%")
    print(f"  |gap| > 0.50%  (~$367): "
          f"{(np.abs(gaps_pct_arr) > 0.50).mean()*100:6.1f}%")
    print(f"  |gap| > 1.00%  (~$735): "
          f"{(np.abs(gaps_pct_arr) > 1.00).mean()*100:6.1f}%")
    print()

    # Confirm relationship: gap should be roughly equal to BTC's 1.23s move
    if delta_per_sec_abs:
        dps = np.array(delta_per_sec_abs)
        print(f"BINANCE 1-second |move| distribution (n={len(dps):,}):")
        print(f"  median: ${np.median(dps):.2f}")
        print(f"  p75:    ${np.percentile(dps, 75):.2f}")
        print(f"  p95:    ${np.percentile(dps, 95):.2f}")
        print(f"  p99:    ${np.percentile(dps, 99):.2f}")
        print()
        median_per_sec = np.median(dps)
        expected_gap = median_per_sec * 1.23
        print(f"  Expected gap from 1.23s lag = median 1s move × 1.23")
        print(f"  = ${median_per_sec:.2f} × 1.23 = ${expected_gap:.2f}")
        print(f"  Actual median |gap| in $:    ${np.median(np.abs(gaps_abs_arr)):.2f}")
        actual = np.median(np.abs(gaps_abs_arr))
        ratio = actual / expected_gap if expected_gap > 0 else float("nan")
        print(f"  Ratio actual/expected: {ratio:.2f}x")
        print()
        if 0.5 < ratio < 2.0:
            print("  ✓ Confirmed: the price gap matches what we'd expect from a 1.23s lag.")
        elif ratio > 2.0:
            print("  ! Gap is larger than 1.23s lag predicts — may be additional clock skew or recording artifact.")
        else:
            print("  ! Gap is smaller than expected — either lag is shorter or BTC vol is lower than measured.")
    print()
    print("INTERPRETATION:")
    print("  The TIME lag (1.23s) and the PRICE gap (~0.02%) are the SAME phenomenon.")
    print("  Binance reports prices NOW; Chainlink (via Polymarket RTDS) reports prices")
    print("  ~1.23s ago. At any wall-clock moment, the gap = how much BTC moved in the")
    print("  past 1.23s.")
    print()
    print("  For a TRADEABLE arbitrage you need the gap to exceed the round-trip cost:")
    print("  taker fee + slippage. On a $0.50 contract that's ~1-2c. The fraction of time")
    print("  the gap exceeds 0.5% is the upper bound on how often pure latency arb can fire.")
    print()
    print("  But: the bot doesn't need to wait for an 'arb' moment — it can fire whenever")
    print("  the cumulative move exceeds a smaller threshold and the book is still stale.")


if __name__ == "__main__":
    main()
