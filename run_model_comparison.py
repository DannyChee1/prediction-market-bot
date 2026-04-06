#!/usr/bin/env python3
"""
Compare diffusion models on walk-forward backtest:
  1. GBM/Gaussian (current btc_15m default)
  2. GBM/Student-t nu=20 (current btc_5m default)
  3. Kou jump-diffusion
  4. Market-Adaptive (novel: blends GBM + market + choppiness + time)

Run: uv run python run_model_comparison.py [--market btc|btc_5m]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from backtest import (
    BacktestEngine,
    DiffusionSignal,
    build_calibration_table,
)
from market_config import get_config


MODELS = {
    "gaussian": dict(tail_mode="normal"),
    "student_t_20": dict(tail_mode="student_t", tail_nu_default=20.0),
    "kou": dict(
        tail_mode="kou",
        kou_lambda=0.007,
        kou_p_up=0.51,
        kou_eta1=1100.0,
        kou_eta2=1100.0,
    ),
    "market_adaptive": dict(
        tail_mode="market_adaptive",
        market_blend_alpha=0.30,
    ),
}


def run_comparison(market: str, bankroll: float, train_frac: float,
                   maker: bool) -> None:
    config = get_config(market)
    data_dir = Path("data") / config.data_subdir

    if not data_dir.exists():
        print(f"No data at {data_dir}")
        return

    n_files = len(list(data_dir.glob("*.parquet")))
    print(f"\n{'='*70}")
    print(f"MODEL COMPARISON — {config.display_name} ({n_files} windows)")
    print(f"  bankroll=${bankroll:.0f}  train_frac={train_frac}  mode={'MAKER' if maker else 'FOK'}")
    print(f"{'='*70}\n")

    results = {}

    for name, model_kw in MODELS.items():
        t0 = time.time()
        print(f"--- {name} ---")

        signal = DiffusionSignal(
            bankroll=bankroll,
            slippage=0.001,
            max_sigma=config.max_sigma,
            min_sigma=config.min_sigma,
            maker_mode=maker,
            **model_kw,
        )

        engine = BacktestEngine(
            signal=signal,
            data_dir=data_dir,
        )

        train_m, test_m, trades_df = engine.run_walk_forward(
            train_frac=train_frac,
            verbose_test=False,
        )

        elapsed = time.time() - t0
        results[name] = {
            "train": train_m,
            "test": test_m,
            "trades": trades_df,
            "elapsed": elapsed,
        }

        tm = test_m
        pnl = tm.get('total_pnl', tm.get('pnl', 0))
        print(f"  test: PnL=${pnl:.2f}  "
              f"WR={tm.get('win_rate', 0)*100:.1f}%  "
              f"trades={tm.get('n_trades', 0)}  "
              f"sharpe={tm.get('sharpe', 0):.2f}  "
              f"({elapsed:.1f}s)")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"SUMMARY — {config.display_name} TEST SET")
    print(f"{'='*70}")
    header = f"{'Model':<20} {'PnL':>8} {'WR':>6} {'Trades':>7} {'Sharpe':>7} {'MaxDD':>8}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        tm = r["test"]
        pnl = tm.get('total_pnl', tm.get('pnl', 0))
        print(f"{name:<20} "
              f"${pnl:>7.0f} "
              f"{tm.get('win_rate', 0)*100:>5.1f}% "
              f"{tm.get('n_trades', 0):>7d} "
              f"{tm.get('sharpe', 0):>7.2f} "
              f"${tm.get('max_drawdown', 0):>7.2f}")

    # ── Z-bucket breakdown per model ──────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Z-BUCKET BREAKDOWN (test set)")
    print(f"{'='*70}")
    import pandas as pd
    import numpy as np

    for name, r in results.items():
        df = r["trades"]
        if df is None or df.empty or "z" not in df.columns:
            continue
        df = df.copy()
        df["abs_z"] = df["z"].abs()
        df["z_bucket"] = pd.cut(df["abs_z"], bins=[0, 0.3, 0.5, 0.7, 1.0, 5.0])
        g = df.groupby("z_bucket", observed=True).agg(
            n=("pnl", "count"),
            pnl=("pnl", "sum"),
            wr=("won", "mean") if "won" in df.columns else ("pnl", lambda x: (x > 0).mean()),
        )
        print(f"\n  {name}:")
        for idx, row in g.iterrows():
            print(f"    {str(idx):15s}  n={int(row['n']):4d}  "
                  f"PnL=${row['pnl']:>7.2f}  WR={row['wr']*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Compare diffusion models")
    parser.add_argument("--market", default="btc", help="Market key (default: btc)")
    parser.add_argument("--bankroll", type=float, default=500.0)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--maker", action="store_true", default=True)
    args = parser.parse_args()

    run_comparison(args.market, args.bankroll, args.train_frac, args.maker)


if __name__ == "__main__":
    main()
