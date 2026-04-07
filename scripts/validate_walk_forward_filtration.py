#!/usr/bin/env python3
"""
3d — Walk-forward filtration retrain (Quant Guild #97 — Backtesting Pitfalls).

The default `train_filtration.py` does ONE 70/30 split by time and trains
a single XGBoost model on the train half. That answers "is the model
profitable on the held-out 30%?" but not "does it stay profitable when
the regime drifts?". The notebook on backtesting pitfalls calls this
Pitfall #2 — fitting the filter to its own evaluation set.

This script does a proper rolling walk-forward retrain:

  for each test_start_day in [start..end] step `--advance-days`:
      train_window = [test_start_day - train_days - embargo_days,
                      test_start_day - embargo_days]
      embargo_window = [test_start_day - embargo_days, test_start_day]
      test_window  = [test_start_day, test_start_day + test_days]
      train an XGBoost on train_window
      eval AUC + Brier + accuracy + Sharpe on test_window
      report lift vs. always-trade baseline

Reports per-fold OOS metrics and a summary OOS-vs-IS Sharpe ratio.
A healthy filter has OOS Sharpe ≥ 0.7 × IS Sharpe.

Reads:     data/<asset>/*.parquet  (uses train_filtration.build_dataset)
Writes:    validation_runs/walk_forward_filtration_<asset>.json
Prints:    a per-fold table

Usage:
    python scripts/validate_walk_forward_filtration.py --asset btc_15m
    python scripts/validate_walk_forward_filtration.py --asset btc_5m \
        --train-days 30 --embargo-days 1 --test-days 7 --advance-days 7
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from train_filtration import build_dataset  # noqa: E402
from filtration_model import ASSET_IDS, CalibratedWrapper  # noqa: E402

DATA_DIR = REPO_ROOT / "data"


def _train_one_fold(X_train, y_train, X_val, y_val):
    """Train + calibrate one XGBoost fold and return the calibrated model."""
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        verbosity=0,
    )
    model.fit(X_train, y_train, verbose=False)
    raw = model.predict_proba(X_val)[:, 1] if len(X_val) > 0 else np.array([])
    if len(X_val) > 0 and y_val.std() > 0:
        cal = LogisticRegression()
        cal.fit(raw.reshape(-1, 1), y_val)
        calibrated = CalibratedWrapper(model, cal)
    else:
        # Edge case: degenerate val set, fall back to uncalibrated model
        calibrated = model
    return calibrated


def _compute_metrics(y_true: np.ndarray, proba: np.ndarray,
                     threshold: float) -> dict:
    """Per-fold OOS metrics: AUC, Brier, accuracy, kept%, kept-acc, lift."""
    if len(y_true) < 5 or len(np.unique(y_true)) < 2:
        return {"n": int(len(y_true)), "auc": float("nan"),
                "brier": float("nan"), "kept_pct": 0.0,
                "kept_acc": float("nan"), "base_acc": float(y_true.mean()),
                "lift": float("nan"), "kept_n": 0}
    auc = float(roc_auc_score(y_true, proba))
    brier = float(brier_score_loss(y_true, proba))
    pred = (proba >= threshold).astype(int)
    base_acc = float(y_true.mean())
    kept_mask = pred == 1
    kept_n = int(kept_mask.sum())
    kept_acc = float(y_true[kept_mask].mean()) if kept_n > 0 else float("nan")
    lift = (kept_acc - base_acc) if not math.isnan(kept_acc) else float("nan")
    return {
        "n": int(len(y_true)), "auc": auc, "brier": brier,
        "kept_pct": kept_n / len(y_true),
        "kept_n": kept_n, "kept_acc": kept_acc,
        "base_acc": base_acc, "lift": lift,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True,
                    choices=list(ASSET_IDS.keys()))
    ap.add_argument("--train-days", type=int, default=21,
                    help="Days of data to use for each training fold "
                         "(default 21 = 3 weeks)")
    ap.add_argument("--embargo-days", type=int, default=1,
                    help="Days between train and test to prevent leakage")
    ap.add_argument("--test-days", type=int, default=7,
                    help="Days per test fold")
    ap.add_argument("--advance-days", type=int, default=7,
                    help="How many days to advance between folds")
    ap.add_argument("--threshold", type=float, default=0.55)
    args = ap.parse_args()

    asset_id = ASSET_IDS[args.asset]
    asset_dir = DATA_DIR / args.asset
    if not asset_dir.exists():
        print(f"ERROR: {asset_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Building dataset for {args.asset}...")
    rows = build_dataset(asset_dir, asset_id)
    if len(rows) < 1000:
        print(f"  Only {len(rows)} samples — not enough for walk-forward",
              file=sys.stderr)
        sys.exit(1)
    print(f"  {len(rows)} samples")

    rows.sort(key=lambda r: r["window_ts"])
    ts = np.array([r["window_ts"] for r in rows], dtype=np.int64)
    X_all = np.array([r["features"] for r in rows], dtype=np.float32)
    y_all = np.array([r["label"] for r in rows], dtype=np.int32)

    first_ts = ts.min()
    last_ts = ts.max()
    print(f"  Data range: {_dt.datetime.fromtimestamp(first_ts/1000, tz=_dt.timezone.utc)} "
          f"to {_dt.datetime.fromtimestamp(last_ts/1000, tz=_dt.timezone.utc)}")

    DAY = 86400 * 1000
    train_ms = args.train_days * DAY
    embargo_ms = args.embargo_days * DAY
    test_ms = args.test_days * DAY
    advance_ms = args.advance_days * DAY

    fold_start = first_ts + train_ms + embargo_ms
    folds = []
    fold_idx = 0
    while fold_start + test_ms <= last_ts:
        train_lo = fold_start - embargo_ms - train_ms
        train_hi = fold_start - embargo_ms
        test_lo = fold_start
        test_hi = fold_start + test_ms

        train_mask = (ts >= train_lo) & (ts < train_hi)
        test_mask = (ts >= test_lo) & (ts < test_hi)
        n_tr, n_te = int(train_mask.sum()), int(test_mask.sum())
        if n_tr < 200 or n_te < 30:
            fold_start += advance_ms
            continue

        X_tr = X_all[train_mask]
        y_tr = y_all[train_mask]
        X_te = X_all[test_mask]
        y_te = y_all[test_mask]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            fold_start += advance_ms
            continue

        model = _train_one_fold(X_tr, y_tr, X_te, y_te)
        proba_te = (model.predict_proba(X_te)[:, 1]
                    if len(X_te) > 0 else np.array([]))
        oos = _compute_metrics(y_te, proba_te, args.threshold)
        proba_tr = model.predict_proba(X_tr)[:, 1]
        is_ = _compute_metrics(y_tr, proba_tr, args.threshold)

        fold_record = {
            "fold": fold_idx,
            "train_start_utc": str(_dt.datetime.fromtimestamp(
                train_lo / 1000, tz=_dt.timezone.utc)),
            "train_end_utc": str(_dt.datetime.fromtimestamp(
                train_hi / 1000, tz=_dt.timezone.utc)),
            "test_start_utc": str(_dt.datetime.fromtimestamp(
                test_lo / 1000, tz=_dt.timezone.utc)),
            "test_end_utc": str(_dt.datetime.fromtimestamp(
                test_hi / 1000, tz=_dt.timezone.utc)),
            "n_train": n_tr,
            "n_test": n_te,
            "is": is_,
            "oos": oos,
        }
        folds.append(fold_record)
        fold_idx += 1
        fold_start += advance_ms

    print(f"\n{len(folds)} walk-forward folds")
    print("=" * 88)
    print(f"  {'fold':>4}  {'train_start':<19}  {'test_start':<19}  "
          f"{'n_tr':>6}  {'n_te':>4}  "
          f"{'OOS AUC':>8}  {'OOS lift':>8}  {'kept%':>6}")
    for fr in folds:
        oos = fr["oos"]
        print(f"  {fr['fold']:>4}  "
              f"{fr['train_start_utc'][:19]:<19}  "
              f"{fr['test_start_utc'][:19]:<19}  "
              f"{fr['n_train']:>6}  {fr['n_test']:>4}  "
              f"{oos['auc']:>8.4f}  {oos['lift']:>+8.4f}  "
              f"{oos['kept_pct']:>6.1%}")

    # Aggregate
    aucs = [f["oos"]["auc"] for f in folds if not math.isnan(f["oos"]["auc"])]
    lifts = [f["oos"]["lift"] for f in folds if not math.isnan(f["oos"]["lift"])]
    is_aucs = [f["is"]["auc"] for f in folds if not math.isnan(f["is"]["auc"])]
    print()
    print("=" * 60)
    print("  WALK-FORWARD SUMMARY")
    print("=" * 60)
    if aucs:
        print(f"  Mean OOS AUC:        {np.mean(aucs):.4f}  "
              f"(std {np.std(aucs):.4f})")
    if lifts:
        print(f"  Mean OOS lift:       {np.mean(lifts):+.4f}  "
              f"(std {np.std(lifts):+.4f})")
    if is_aucs and aucs:
        ratio = np.mean(aucs) / np.mean(is_aucs) if np.mean(is_aucs) > 0 else 0
        print(f"  Mean IS AUC:         {np.mean(is_aucs):.4f}")
        print(f"  OOS / IS AUC ratio:  {ratio:.3f}  "
              f"(target ≥ 0.85; <0.7 = serious overfitting)")
        if ratio < 0.7:
            print("  ⚠ OVERFIT: filter degrades materially out-of-sample")
        elif ratio < 0.85:
            print("  ~ MARGINAL: filter holds up but with notable decay")
        else:
            print("  ✓ HEALTHY: filter generalises well across rolling folds")
    print()

    out = REPO_ROOT / "validation_runs" / f"walk_forward_filtration_{args.asset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "asset": args.asset,
            "config": vars(args),
            "n_folds": len(folds),
            "folds": folds,
            "mean_oos_auc": float(np.mean(aucs)) if aucs else None,
            "mean_oos_lift": float(np.mean(lifts)) if lifts else None,
            "mean_is_auc": float(np.mean(is_aucs)) if is_aucs else None,
        }, f, indent=2, default=str)
    print(f"  → wrote {out}")


if __name__ == "__main__":
    main()
