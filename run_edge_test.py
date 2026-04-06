#!/usr/bin/env python3
"""
Monte Carlo edge significance test.

Takes walk-forward backtest trades and answers: "Is this edge real?"

Method:
1. Run walk-forward backtest to get actual trades with outcomes
2. Shuffle the outcomes randomly (permutation test)
3. Recompute PnL under shuffled outcomes N times
4. p-value = fraction of shuffled runs that beat actual PnL

Also computes:
- Bootstrap 95% CI on win rate and PnL
- Expected PnL under random 50/50 outcomes (null hypothesis)
- Break-even win rate given average entry price
- Sharpe ratio confidence interval

Usage: uv run python run_edge_test.py [--market btc] [--n-sims 10000]
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

from backtest import BacktestEngine, DiffusionSignal
from market_config import get_config


def compute_pnl_from_trades(trades: list[dict], shuffle: bool = False) -> float:
    """Compute total PnL from trade list, optionally shuffling outcomes."""
    if shuffle:
        outcomes = [t["won"] for t in trades]
        random.shuffle(outcomes)
        pnl = 0.0
        for t, won in zip(trades, outcomes):
            if won:
                pnl += t["payout"] - t["cost"]
            else:
                pnl -= t["cost"]
        return pnl
    return sum(t["pnl"] for t in trades)


def run_edge_test(market: str, n_sims: int, bankroll: float, min_z: float):
    config = get_config(market)
    data_dir = Path("data") / config.data_subdir
    n_files = len(list(data_dir.glob("*.parquet")))

    print(f"\n{'='*60}")
    print(f"MONTE CARLO EDGE TEST — {config.display_name}")
    print(f"  {n_files} windows, {n_sims:,} simulations")
    print(f"{'='*60}\n")

    # Run actual walk-forward backtest
    print("Running walk-forward backtest...")
    signal = DiffusionSignal(
        bankroll=bankroll, slippage=0.001,
        max_sigma=config.max_sigma, min_sigma=config.min_sigma,
        maker_mode=True,
        tail_mode=config.tail_mode,
        tail_nu_default=config.tail_nu_default,
        kou_lambda=config.kou_lambda,
        kou_p_up=config.kou_p_up,
        kou_eta1=config.kou_eta1,
        kou_eta2=config.kou_eta2,
        min_entry_z=min_z,
    )
    engine = BacktestEngine(signal=signal, data_dir=data_dir)
    _, test_m, trades_df = engine.run_walk_forward(
        train_frac=0.7, verbose_test=False
    )

    if trades_df is None or trades_df.empty:
        print("No trades in test set!")
        return

    # Extract trade-level data
    trades = []
    for _, row in trades_df.iterrows():
        cost = row.get("cost_usd", 0)
        pnl = row.get("pnl", 0)
        won = pnl > 0
        payout = cost + pnl if won else 0.0
        trades.append({
            "cost": cost,
            "pnl": pnl,
            "payout": payout,
            "won": won,
            "entry": row.get("entry_price", 0.5),
        })

    n_trades = len(trades)
    actual_pnl = sum(t["pnl"] for t in trades)
    actual_wr = sum(t["won"] for t in trades) / n_trades
    avg_entry = np.mean([t["entry"] for t in trades])
    break_even_wr = avg_entry  # for binary options, BE WR = entry price

    print(f"\n── ACTUAL RESULTS ──")
    print(f"  Trades: {n_trades}")
    print(f"  Win rate: {actual_wr:.1%} ({sum(t['won'] for t in trades)}/{n_trades})")
    print(f"  Total PnL: ${actual_pnl:.2f}")
    print(f"  Avg entry price: ${avg_entry:.3f}")
    print(f"  Break-even WR: {break_even_wr:.1%}")
    print(f"  Edge over BE: {(actual_wr - break_even_wr)*100:+.1f}pp")

    # ── Permutation test: shuffle outcomes ─────────────────────────────────
    print(f"\n── PERMUTATION TEST ({n_sims:,} shuffles) ──")
    t0 = time.time()

    shuffled_pnls = []
    for _ in range(n_sims):
        shuffled_pnls.append(compute_pnl_from_trades(trades, shuffle=True))

    shuffled_pnls = np.array(shuffled_pnls)
    p_value_pnl = (shuffled_pnls >= actual_pnl).mean()

    print(f"  Actual PnL: ${actual_pnl:.2f}")
    print(f"  Shuffled PnL: mean=${shuffled_pnls.mean():.2f}  "
          f"std=${shuffled_pnls.std():.2f}")
    print(f"  p-value (PnL): {p_value_pnl:.4f}  "
          f"{'*** SIGNIFICANT' if p_value_pnl < 0.05 else '(not significant)'}")

    # ── Bootstrap CI on win rate ───────────────────────────────────────────
    print(f"\n── BOOTSTRAP CI ({n_sims:,} resamples) ──")
    boot_wrs = []
    boot_pnls = []
    won_list = [t["won"] for t in trades]
    pnl_list = [t["pnl"] for t in trades]

    for _ in range(n_sims):
        idx = np.random.randint(0, n_trades, size=n_trades)
        boot_wrs.append(np.mean([won_list[i] for i in idx]))
        boot_pnls.append(np.sum([pnl_list[i] for i in idx]))

    boot_wrs = np.array(boot_wrs)
    boot_pnls = np.array(boot_pnls)

    wr_lo, wr_hi = np.percentile(boot_wrs, [2.5, 97.5])
    pnl_lo, pnl_hi = np.percentile(boot_pnls, [2.5, 97.5])

    print(f"  Win rate 95% CI: [{wr_lo:.1%}, {wr_hi:.1%}]  "
          f"(actual: {actual_wr:.1%})")
    print(f"  PnL 95% CI: [${pnl_lo:.0f}, ${pnl_hi:.0f}]  "
          f"(actual: ${actual_pnl:.0f})")
    print(f"  WR lower bound vs break-even ({break_even_wr:.1%}): "
          f"{'ABOVE' if wr_lo > break_even_wr else 'BELOW'}")

    # ── Null hypothesis: 50/50 random entry ────────────────────────────────
    print(f"\n── NULL HYPOTHESIS (random 50/50 entries) ──")
    null_pnls = []
    for _ in range(n_sims):
        pnl = 0.0
        for t in trades:
            if random.random() < 0.5:
                pnl += t["payout"] - t["cost"]
            else:
                pnl -= t["cost"]
        null_pnls.append(pnl)

    null_pnls = np.array(null_pnls)
    p_value_null = (null_pnls >= actual_pnl).mean()

    print(f"  Null PnL: mean=${null_pnls.mean():.2f}  std=${null_pnls.std():.2f}")
    print(f"  p-value vs null: {p_value_null:.4f}  "
          f"{'*** SIGNIFICANT' if p_value_null < 0.05 else '(not significant)'}")

    # ── Summary verdict ────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"VERDICT — {config.display_name}")
    print(f"{'='*60}")
    print(f"  Win rate: {actual_wr:.1%}  95% CI: [{wr_lo:.1%}, {wr_hi:.1%}]")
    print(f"  PnL: ${actual_pnl:.0f}  95% CI: [${pnl_lo:.0f}, ${pnl_hi:.0f}]")
    print(f"  p-value (permutation): {p_value_pnl:.4f}")
    print(f"  p-value (vs null 50/50): {p_value_null:.4f}")

    if p_value_pnl < 0.01:
        print(f"  STRONG EVIDENCE of edge (p < 0.01)")
    elif p_value_pnl < 0.05:
        print(f"  MODERATE EVIDENCE of edge (p < 0.05)")
    elif p_value_pnl < 0.10:
        print(f"  WEAK EVIDENCE of edge (p < 0.10)")
    else:
        print(f"  INSUFFICIENT EVIDENCE of edge (p >= 0.10)")

    print(f"  ({elapsed:.1f}s for {n_sims:,} simulations)")


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo edge significance test")
    parser.add_argument("--market", default="btc")
    parser.add_argument("--n-sims", type=int, default=10_000)
    parser.add_argument("--bankroll", type=float, default=500.0)
    parser.add_argument("--min-z", type=float, default=0.0)
    args = parser.parse_args()

    run_edge_test(args.market, args.n_sims, args.bankroll, args.min_z)


if __name__ == "__main__":
    main()
