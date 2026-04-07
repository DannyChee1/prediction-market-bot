#!/usr/bin/env python3
"""
Ablate Section-2 improvements one at a time to find which (if any)
hurt backtest performance.

For each market we run:
  baseline    — bug fixes only (sigma_estimator='yz', no regime, raw OBI,
                old mid_momentum behavior)
  +ewma       — sigma_estimator='ewma'
  +regime     — regime classifier active
  +ewma+regime — both

We hold the bug fixes constant (Kou drift removed, sigma bounds widened)
because those have already been verified to help.

The Kalman OBI and mid_momentum parity fix are NOT ablated here because
they don't have config flags — they would require code reverts to
disable. Their effect is therefore in every run below as a baseline.

Results are printed as a side-by-side table and saved to
validation_runs/ablation_<market>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backtest import BacktestEngine, build_diffusion_signal, DATA_DIR  # noqa: E402
from market_config import MARKET_CONFIGS  # noqa: E402


def run_one(market: str, sigma_est: str, use_regime: bool,
            bankroll: float = 10_000.0, train_frac: float = 0.7) -> dict:
    """Run a single backtest with the chosen ablation toggles.

    To override sigma_estimator without mutating market_config we
    monkey-patch the DiffusionSignal *after* `build_diffusion_signal`
    constructs it. The constructor's validation has already run so this
    is safe (we go through the same valid set: yz / rv / ewma).
    """
    cfg = MARKET_CONFIGS[market]
    data_dir = DATA_DIR / cfg.data_subdir
    signal = build_diffusion_signal(
        market, bankroll=bankroll,
        use_regime_classifier=use_regime,
    )
    if sigma_est not in ("yz", "rv", "ewma"):
        raise ValueError(f"bad sigma_est {sigma_est!r}")
    signal.sigma_estimator = sigma_est
    if not use_regime:
        signal.regime_classifier = None

    engine = BacktestEngine(
        signal=signal,
        data_dir=data_dir,
        latency_ms=0,
        slippage=0.0,
        initial_bankroll=bankroll,
    )
    train_m, test_m, _ = engine.run_walk_forward(
        train_frac=train_frac, verbose_test=False
    )
    return {
        "test_n_trades": int(test_m.get("n_trades", 0)),
        "test_win_rate": float(test_m.get("win_rate", 0)),
        "test_total_pnl": float(test_m.get("total_pnl", 0)),
        "test_sharpe": float(test_m.get("sharpe", 0)),
        "test_max_dd_pct": float(test_m.get("max_dd_pct", 0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True,
                    choices=list(MARKET_CONFIGS.keys()))
    args = ap.parse_args()

    print(f"Ablating Section-2 improvements on {args.market}...")
    print(f"  (Kalman OBI and mid_momentum parity fixes are baked in)")
    print()

    runs = [
        ("baseline (yz, no regime)",  "yz",   False),
        ("+ewma                  ",   "ewma", False),
        ("+regime                ",   "yz",   True),
        ("+ewma +regime          ",   "ewma", True),
    ]

    results = {}
    for label, sigma_est, use_regime in runs:
        print(f"  Running: {label.strip()}...")
        r = run_one(args.market, sigma_est, use_regime)
        results[label.strip()] = r
        print(f"    n={r['test_n_trades']:4d}  WR={r['test_win_rate']:.1%}  "
              f"PnL={r['test_total_pnl']:+9.2f}  "
              f"Sharpe={r['test_sharpe']:+.3f}  "
              f"MaxDD={r['test_max_dd_pct']:.1%}")

    print()
    print("=" * 76)
    print(f"  ABLATION TABLE — {args.market}")
    print("=" * 76)
    print(f"  {'configuration':<30}  {'n':>4}  {'WR':>6}  {'PnL':>10}  "
          f"{'Sharpe':>8}  {'MaxDD':>7}")
    for label, _, _ in runs:
        r = results[label.strip()]
        print(f"  {label.strip():<30}  {r['test_n_trades']:>4}  "
              f"{r['test_win_rate']:>5.1%}  "
              f"{r['test_total_pnl']:>+10.2f}  "
              f"{r['test_sharpe']:>+8.3f}  "
              f"{r['test_max_dd_pct']:>6.1%}")
    print()

    out_path = REPO_ROOT / "validation_runs" / f"ablation_{args.market}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  → wrote {out_path}")


if __name__ == "__main__":
    main()
