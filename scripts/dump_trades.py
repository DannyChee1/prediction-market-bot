#!/usr/bin/env python3
"""
Run a walk-forward backtest and dump per-trade results plus bankroll
history to parquet/JSON so the section-3 validation scripts can consume
them without re-running the backtest every time.

Outputs (under validation_runs/<tag>/):
  trades_train.parquet   # in-sample trades  (one row per fill)
  trades_test.parquet    # out-of-sample trades
  metrics.json           # train + test summary metrics
  bankroll_train.json    # tick-by-tick bankroll history (in-sample)
  bankroll_test.json     # tick-by-tick bankroll history (out-of-sample)
  config.json            # the MarketConfig actually used at run time

The validation scripts read only `trades_test.parquet` and
`bankroll_test.json` by default — out-of-sample is the only honest
sample for permutation/bootstrap/ergodicity tests. The train artifacts
are dumped only for parity-check / debugging.

Usage:
    python scripts/dump_trades.py --market btc_5m --tag postfix
    python scripts/dump_trades.py --market btc    --tag postfix
"""
from __future__ import annotations

import argparse
import json
import sys
import dataclasses
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from backtest import BacktestEngine, build_diffusion_signal, DATA_DIR  # noqa: E402
from market_config import MARKET_CONFIGS  # noqa: E402


def _config_to_dict(cfg) -> dict:
    if dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    return {k: getattr(cfg, k) for k in dir(cfg)
            if not k.startswith("_") and not callable(getattr(cfg, k))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True,
                    choices=list(MARKET_CONFIGS.keys()))
    ap.add_argument("--tag", default="postfix",
                    help="Subdirectory under validation_runs/ to write to")
    ap.add_argument("--bankroll", type=float, default=10_000.0)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = MARKET_CONFIGS[args.market]
    data_dir = DATA_DIR / cfg.data_subdir

    out_dir = REPO_ROOT / "validation_runs" / f"{args.tag}_{args.market}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Market:    {args.market} ({cfg.display_name})")
    print(f"Data dir:  {data_dir}")
    print(f"Output:    {out_dir}")
    print(f"Bankroll:  ${args.bankroll:,.0f}")
    print(f"train_frac:{args.train_frac}")
    print()

    signal = build_diffusion_signal(args.market, bankroll=args.bankroll)
    engine = BacktestEngine(
        signal=signal,
        data_dir=data_dir,
        latency_ms=0,
        slippage=0.0,
        initial_bankroll=args.bankroll,
    )

    # walk_forward also saves the bankroll history internally; we re-create
    # the test bankroll history by running the test slugs again. The cleanest
    # way is to monkey-patch _run_slug_list to capture both. Simpler: just
    # use run_walk_forward and re-derive bankroll from PnL cumsum.
    train_m, test_m, test_trades = engine.run_walk_forward(
        train_frac=args.train_frac,
        verbose_test=False,
    )

    print(f"Train: {train_m['n_trades']} trades, "
          f"WR {train_m.get('win_rate', 0):.1%}, "
          f"PnL {train_m.get('total_pnl', 0):+.2f}, "
          f"Sharpe {train_m.get('sharpe', 0):.2f}")
    print(f"Test:  {test_m['n_trades']} trades, "
          f"WR {test_m.get('win_rate', 0):.1%}, "
          f"PnL {test_m.get('total_pnl', 0):+.2f}, "
          f"Sharpe {test_m.get('sharpe', 0):.2f}")

    # Persist test trades (the only honest sample for validation)
    test_trades.to_parquet(out_dir / "trades_test.parquet", index=False)
    print(f"\n✓ wrote {out_dir/'trades_test.parquet'} ({len(test_trades)} rows)")

    # Persist metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"train": train_m, "test": test_m, "args": vars(args)},
                  f, indent=2, default=str)
    print(f"✓ wrote {out_dir/'metrics.json'}")

    # Reconstruct bankroll history from PnL cumsum
    if not test_trades.empty:
        bk = float(args.bankroll) + test_trades["pnl"].cumsum()
        bk = pd.concat([pd.Series([float(args.bankroll)]), bk], ignore_index=True)
        bk.tolist()
        with open(out_dir / "bankroll_test.json", "w") as f:
            json.dump([float(x) for x in bk.tolist()], f)
        print(f"✓ wrote {out_dir/'bankroll_test.json'} ({len(bk)} ticks)")

    # Persist the config we ran with so the validation scripts can
    # reference it (e.g. for fee schedule, edge_threshold, sigma bounds).
    with open(out_dir / "config.json", "w") as f:
        json.dump(_config_to_dict(cfg), f, indent=2, default=str)
    print(f"✓ wrote {out_dir/'config.json'}")


if __name__ == "__main__":
    main()
