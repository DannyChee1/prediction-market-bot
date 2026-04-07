#!/usr/bin/env python3
"""
Backtest comparison: approaches to fix the p_side 0.40-0.60 bleed zone.

Tests:
  A) Baseline (current params)
  B) edge_threshold=0.12 (higher bar for entry)
  C) edge_threshold=0.15 (even higher bar)
  D) min_edge=0.10 (raise A-S floor)
  E) Higher min_sigma (filter unreliable vol)
"""

import subprocess
import sys
import re
import json
import concurrent.futures
from dataclasses import dataclass


@dataclass
class Result:
    label: str
    market: str
    pnl: float
    sharpe: float
    trades: int
    win_rate: float


# Market-specific optimal defaults from MEMORY.md
MARKET_DEFAULTS = {
    "btc_15m": ["--max-z", "3.0", "--reversion-discount", "0.0", "--gamma-spread", "1.5", "--min-sigma", "3e-05"],
    "btc_5m":  ["--min-sigma", "7e-05", "--gamma-spread", "1.5"],
    "eth_15m": ["--max-z", "3.0", "--reversion-discount", "0.2", "--gamma-spread", "0.75"],
    "eth_5m":  ["--gamma-spread", "1.5"],
}

N_WINDOWS = 80


def run_one(market: str, label: str, extra_args: list[str]) -> Result:
    """Run tick_backtest.py and parse output."""
    cmd = [
        sys.executable, "tick_backtest.py",
        "--market", market,
        "--windows", str(N_WINDOWS),
        "--seed", "42",
        "--max-fills", "1",
        "--as-mode",
        *extra_args,
    ]

    # Add market defaults unless overridden
    defaults = MARKET_DEFAULTS.get(market, [])
    extra_keys = {a for a in extra_args if a.startswith("--")}
    for i in range(0, len(defaults), 2):
        if defaults[i] not in extra_keys:
            cmd.extend([defaults[i], defaults[i + 1]])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout
    except subprocess.TimeoutExpired:
        return Result(label, market, 0, 0, 0, 0)

    pnl = sharpe = 0.0
    trades = 0
    win_rate = 0.0

    for line in output.split("\n"):
        m = re.search(r"Total PnL:\s+\$([-+\d,.]+)", line)
        if m:
            pnl = float(m.group(1).replace(",", ""))

        m = re.search(r"Sharpe \(annualized\):\s+([-\d.]+)", line)
        if m:
            sharpe = float(m.group(1))

        m = re.search(r"Total fills:\s+(\d+)", line)
        if m:
            trades = int(m.group(1))

        m = re.search(r"Wins:\s+\d+\s+\(([\d.]+)%\)", line)
        if m:
            win_rate = float(m.group(1))

    return Result(label, market, pnl, sharpe, trades, win_rate)


def main():
    markets = ["btc_15m", "btc_5m", "eth_15m", "eth_5m"]

    approaches = [
        ("A) Baseline",           {}),
        ("B) edge_thresh=0.12",   {"--edge-threshold": "0.12"}),
        ("C) edge_thresh=0.15",   {"--edge-threshold": "0.15"}),
        ("D) min_edge=0.10",      {"--min-edge": "0.10"}),
        ("E) higher min_sigma",   {}),  # handled per-market
    ]

    # Build all (label, market, args) jobs
    jobs = []
    for label, overrides in approaches:
        for market in markets:
            args = []
            for k, v in overrides.items():
                args.extend([k, v])

            if label.startswith("E)"):
                if "btc" in market:
                    args = ["--min-sigma", "1e-4"]
                else:
                    args = ["--min-sigma", "8e-5"]

            jobs.append((market, label, args))

    print(f"Running {len(jobs)} backtests ({N_WINDOWS} windows each)...\n")

    # Run in parallel (4 workers to not thrash CPU too hard)
    all_results: list[Result] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_one, m, l, a): (l, m) for m, l, a in jobs}
        for fut in concurrent.futures.as_completed(futures):
            label, market = futures[fut]
            try:
                r = fut.result()
                all_results.append(r)
                avg = r.pnl / r.trades if r.trades else 0
                print(f"  {r.label:<30s} {r.market:<10s} PnL=${r.pnl:>+8.2f}  Sharpe={r.sharpe:>6.2f}  "
                      f"trades={r.trades:>3d}  win={r.win_rate:>5.1f}%  avg=${avg:>+.2f}")
            except Exception as exc:
                print(f"  {label:<30s} {market:<10s} FAILED: {exc}")

    # Sort for display
    label_order = [l for l, _ in approaches]
    all_results.sort(key=lambda r: (label_order.index(r.label), markets.index(r.market)))

    # Per-approach per-market table
    print(f"\n{'='*95}")
    print(f"{'Approach':<30s} {'Market':<10s} {'PnL':>10s} {'Sharpe':>8s} {'Trades':>7s} {'Win%':>6s} {'$/trade':>9s}")
    print("-" * 95)
    for r in all_results:
        avg = r.pnl / r.trades if r.trades else 0
        print(f"{r.label:<30s} {r.market:<10s} ${r.pnl:>+8.2f} {r.sharpe:>7.2f} {r.trades:>7d} {r.win_rate:>5.1f}% ${avg:>+7.2f}")

    # Aggregate
    print(f"\n{'='*75}")
    print(f"{'Approach':<30s} {'Total PnL':>12s} {'Avg Sharpe':>12s} {'Trades':>8s} {'Avg $/trade':>12s}")
    print("-" * 75)
    for label in label_order:
        group = [r for r in all_results if r.label == label]
        if not group:
            continue
        tot_pnl = sum(r.pnl for r in group)
        avg_sharpe = sum(r.sharpe for r in group) / len(group)
        tot_trades = sum(r.trades for r in group)
        avg_per = tot_pnl / tot_trades if tot_trades else 0
        print(f"{label:<30s} ${tot_pnl:>+10.2f} {avg_sharpe:>11.2f} {tot_trades:>8d} ${avg_per:>+10.2f}")


if __name__ == "__main__":
    main()
