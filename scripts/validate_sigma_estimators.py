#!/usr/bin/env python3
"""
2a — A/B test of σ estimators (Quant Guild #47 — ARCH/GARCH).

Question:  Does Yang-Zhang on 5s micro-bars actually win against simple
realized variance, EWMA, or GARCH(1,1) for 1-step-ahead σ forecasting?
The current `_compute_vol` uses Yang-Zhang, but the YZ estimator was
designed for daily bars with overnight gaps and the `var_oc` (open-to-
prev-close) component is meaningless on a continuously-traded feed.

Method:
  1. Walk all parquet windows for the chosen market in time order.
  2. Split into train (first 70%) and test (last 30%).
  3. On the train set, fit GARCH(1,1) (omega, alpha, beta) by QML.
  4. On the test set, at every sample point compute σ̂(t+1) under
     each estimator using only data up to t. Compare against the
     realised σ over the next N seconds (the "actual" forward σ).
  5. Report 1-step forecast MSE and bias per estimator.
  6. Recommend whichever has the lowest MSE.

Reads:     data/<market>/*.parquet
Writes:    validation_runs/sigma_estimators/<market>.json
Prints:    a comparison table with the recommended winner

Usage:
    python scripts/validate_sigma_estimators.py --market btc_5m
    python scripts/validate_sigma_estimators.py --market btc
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
sys.path.insert(0, str(REPO_ROOT))

from backtest import (  # noqa: E402
    _compute_vol_deduped,
    MIN_FINAL_REMAINING_S,
    MAX_START_GAP_S,
)
from market_config import MARKET_CONFIGS  # noqa: E402
from scripts.sigma_estimators import (  # noqa: E402
    realized_variance_per_s,
    ewma_sigma_per_s,
    garch11_sigma_per_s,
    fit_garch11,
)

DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "validation_runs" / "sigma_estimators"


def load_complete_windows(market: str, max_files: int | None = None):
    """Yield (prices, ts_list) tuples for each complete window."""
    cfg = MARKET_CONFIGS[market]
    data_dir = DATA_DIR / cfg.data_subdir
    files = sorted(data_dir.glob("*.parquet"))
    if max_files:
        files = files[:max_files]
    for f in files:
        df = pd.read_parquet(f)
        if df.empty:
            continue
        if "chainlink_btc" in df.columns and "chainlink_price" not in df.columns:
            df.rename(columns={"chainlink_btc": "chainlink_price"}, inplace=True)
        if "chainlink_price" not in df.columns:
            continue
        if "window_end_ms" in df.columns:
            if df["ts_ms"].iloc[-1] < df["window_end_ms"].iloc[0]:
                continue
        else:
            if df["time_remaining_s"].iloc[-1] > MIN_FINAL_REMAINING_S:
                continue
        if ("window_start_ms" in df.columns and "window_end_ms" in df.columns
                and "time_remaining_s" in df.columns):
            window_dur_s = (df["window_end_ms"].iloc[0]
                            - df["window_start_ms"].iloc[0]) / 1000
            if df["time_remaining_s"].iloc[0] < window_dur_s - MAX_START_GAP_S:
                continue
        yield (df["chainlink_price"].tolist(), df["ts_ms"].tolist())


def evaluate_window(prices, ts_list, garch_params, lookback_s=90,
                    forecast_horizon_s=60, sample_every=15):
    """For one window, generate 1-step forecasts at multiple sample points
    and compare to the actually-realised σ over the next forecast_horizon_s.

    Returns a list of dicts with one entry per sample point.
    """
    rows = []
    n = len(prices)
    if n < lookback_s + forecast_horizon_s:
        return rows
    for idx in range(lookback_s, n - forecast_horizon_s, sample_every):
        # History up to idx (inclusive)
        hp = prices[max(0, idx - lookback_s):idx + 1]
        ht = ts_list[max(0, idx - lookback_s):idx + 1]

        sigma_yz = _compute_vol_deduped(hp, ht)
        sigma_rv = realized_variance_per_s(hp, ht)
        sigma_ewma = ewma_sigma_per_s(hp, ht, lambda_=0.94)
        sigma_garch = garch11_sigma_per_s(hp, ht, **garch_params)

        # Realised forward σ over the next forecast_horizon_s
        fp = prices[idx:idx + forecast_horizon_s + 1]
        ft = ts_list[idx:idx + forecast_horizon_s + 1]
        sigma_fwd = realized_variance_per_s(fp, ft)
        if sigma_fwd <= 0:
            continue

        rows.append({
            "yz": sigma_yz, "rv": sigma_rv,
            "ewma": sigma_ewma, "garch": sigma_garch,
            "actual": sigma_fwd,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True,
                    choices=list(MARKET_CONFIGS.keys()))
    ap.add_argument("--max-files", type=int, default=None,
                    help="Cap on parquet files to process (for fast iteration)")
    ap.add_argument("--lookback-s", type=int, default=90)
    ap.add_argument("--forecast-horizon-s", type=int, default=60)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.market} windows...")
    windows = list(load_complete_windows(args.market, args.max_files))
    n_windows = len(windows)
    if n_windows == 0:
        print("No windows found", file=sys.stderr)
        sys.exit(1)

    split = int(n_windows * 0.7)
    train_windows = windows[:split]
    test_windows = windows[split:]
    print(f"  {n_windows} windows: {len(train_windows)} train / "
          f"{len(test_windows)} test")

    # Fit GARCH on the concatenated train returns
    print("\nFitting GARCH(1,1) on train returns...")
    all_prices = []
    all_ts = []
    starts = []
    cursor = 0
    for p, t in train_windows:
        starts.append(cursor)
        all_prices.extend(p)
        all_ts.extend(t)
        cursor += len(p)
    garch = fit_garch11(all_prices, all_ts, window_starts=starts)
    print(f"  omega={garch['omega']:.6e}  alpha={garch['alpha']:.4f}  "
          f"beta={garch['beta']:.4f}  nll={garch['nll']:.2f}  "
          f"n_obs={garch['n_obs']}")

    garch_params = {"omega": garch["omega"], "alpha": garch["alpha"],
                    "beta": garch["beta"]}

    # Evaluate on test windows
    print("\nEvaluating estimators on test windows...")
    all_rows = []
    for p, t in test_windows:
        all_rows.extend(evaluate_window(
            p, t, garch_params,
            lookback_s=args.lookback_s,
            forecast_horizon_s=args.forecast_horizon_s,
        ))
    n_obs = len(all_rows)
    if n_obs == 0:
        print("No usable test observations", file=sys.stderr)
        sys.exit(1)
    print(f"  {n_obs} forecast/actual pairs")

    df = pd.DataFrame(all_rows)
    df = df[df["actual"] > 0]
    estimators = ["yz", "rv", "ewma", "garch"]
    metrics = {}
    for est in estimators:
        # Filter zero-σ rows for this estimator
        sub = df[df[est] > 0]
        if len(sub) == 0:
            metrics[est] = {"n": 0}
            continue
        err = sub[est] - sub["actual"]
        mse = float((err ** 2).mean())
        rmse = math.sqrt(mse)
        bias = float(err.mean())
        # Relative MSE: MSE / mean(actual^2) — scale-invariant
        rel_mse = mse / float((sub["actual"] ** 2).mean())
        # Pearson correlation between forecast and actual
        if sub[est].std() > 0 and sub["actual"].std() > 0:
            corr = float(np.corrcoef(sub[est], sub["actual"])[0, 1])
        else:
            corr = 0.0
        metrics[est] = {
            "n": int(len(sub)),
            "mse": mse,
            "rmse": rmse,
            "bias": bias,
            "rel_mse": rel_mse,
            "corr": corr,
        }

    print()
    print("=" * 72)
    print("  σ ESTIMATOR COMPARISON  (1-step forecast vs realised forward σ)")
    print("=" * 72)
    print(f"  {'estimator':<12}  {'n':>7}  {'rmse':>11}  {'bias':>11}  "
          f"{'rel_mse':>9}  {'corr':>8}")
    sorted_ests = sorted(estimators, key=lambda e: metrics[e].get(
        "rel_mse", float("inf")))
    for est in sorted_ests:
        m = metrics[est]
        if m.get("n", 0) == 0:
            print(f"  {est:<12}  {'-':>7}  {'-':>11}  {'-':>11}  "
                  f"{'-':>9}  {'-':>8}")
            continue
        print(f"  {est:<12}  {m['n']:>7}  {m['rmse']:>11.2e}  "
              f"{m['bias']:>+11.2e}  {m['rel_mse']:>9.4f}  {m['corr']:>8.4f}")

    winner = sorted_ests[0]
    yz_rel = metrics["yz"]["rel_mse"]
    win_rel = metrics[winner]["rel_mse"]
    improvement = (yz_rel - win_rel) / yz_rel if yz_rel > 0 else 0
    print()
    print(f"  Winner: {winner.upper()}  "
          f"({improvement*100:+.1f}% relative MSE vs Yang-Zhang)")
    if winner == "yz":
        print("  → YZ is the best forecaster on this data; keep using it.")
    elif improvement < 0.05:
        print(f"  → {winner.upper()} is marginally better than YZ. Not worth")
        print(f"    swapping unless other constraints favor it.")
    else:
        print(f"  → {winner.upper()} is materially better than YZ. Consider")
        print(f"    wiring it into _compute_vol via a config field.")

    out = {
        "market": args.market,
        "n_train_windows": len(train_windows),
        "n_test_windows": len(test_windows),
        "n_test_observations": n_obs,
        "garch_fit": garch,
        "metrics": metrics,
        "winner": winner,
    }
    out_path = OUT_DIR / f"{args.market}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  → wrote {out_path}")


if __name__ == "__main__":
    main()
