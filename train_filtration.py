#!/usr/bin/env python3
"""
Train the XGBoost filtration model.

Two label modes:

  --target classification (default, legacy)
    Label: at a given decision point, was the z-score signal direction correct?
      - z > 0 (model says UP)  and outcome UP   → correct (1)
      - z > 0 (model says UP)  and outcome DOWN → incorrect (0)
      - z < 0 (model says DOWN) and outcome DOWN → correct (1)
      - z < 0 (model says DOWN) and outcome UP  → incorrect (0)
    Trains an XGBClassifier with logistic calibration. predict_proba returns
    P(direction correct).

  --target regression
    Label: realized PnL per dollar invested if we had taken the trade at the
    current ask, given the resolved outcome.
      - won:  (1.0 - cost - fee) / cost   (typically +0.5..+2.0 range)
      - lost: -1.0                        (lose entire stake)
    Trains an XGBRegressor (no calibration, the output is already a PnL).
    Use this with --filtration-mode size_mult in backtest — the predicted
    EV is mapped directly to a Kelly multiplier.

Both modes skip rows where |z| < MIN_Z_SIGNAL (no directional signal to label).

Walk-forward split: train on first 70%, validate on last 30% by time.
Never use future data in features.

Usage:
    python3 train_filtration.py                           # legacy classification
    python3 train_filtration.py --target regression       # PnL regression
    python3 train_filtration.py --threshold 0.58          # custom gate
    python3 train_filtration.py --target regression --output filtration_model_pnl.pkl
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, brier_score_loss, classification_report, roc_auc_score,
)
from sklearn.calibration import CalibratedClassifierCV

from filtration_model import (
    ASSET_IDS, FEATURE_NAMES, FiltrationModel, extract_features,
)


def _poly_fee(p: float) -> float:
    """Polymarket taker fee — duplicated here to keep this script standalone.
    Stays in lockstep with backtest_core.poly_fee."""
    return 0.02 * p * (1.0 - p)

warnings.filterwarnings("ignore")

DATA_DIR  = Path("data")
MODEL_PATH = Path("filtration_model.pkl")

VOL_LOOKBACK_S   = 90
BASELINE_VOL_S   = 300
MIN_Z_SIGNAL     = 0.10   # skip near-zero z (no signal to label)
TAU_CHECKPOINTS  = [750, 600, 450, 300, 150, 60]
TRAIN_FRAC       = 0.70   # walk-forward split


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def compute_sigma(prices: list[float], timestamps: list[int]) -> float:
    changes = []
    for i, p in enumerate(prices):
        if p > 0 and (not changes or p != changes[-1][1]):
            changes.append((timestamps[i], p))
    if len(changes) < 3:
        return 0.0
    log_rets = []
    for j in range(1, len(changes)):
        dt = (changes[j][0] - changes[j-1][0]) / 1000.0
        if dt > 0:
            lr = math.log(changes[j][1] / changes[j-1][1])
            log_rets.append(lr / math.sqrt(dt))
    if len(log_rets) < 2:
        return 0.0
    return float(np.std(log_rets, ddof=1))


def compute_buy_pressure(df: pd.DataFrame, idx: int, lookback: int = 60) -> float:
    """Fraction of trades in last `lookback` rows that were BUY."""
    lo = max(0, idx - lookback)
    sides = df["last_trade_side_up"].iloc[lo:idx+1].dropna()
    if len(sides) == 0:
        return 0.5
    buys = (sides == "BUY").sum()
    return float(buys) / len(sides)


def compute_mid_momentum(df: pd.DataFrame, idx: int, lookback: int = 60) -> float:
    """Change in mid_up over last `lookback` rows."""
    lo = max(0, idx - lookback)
    mids = df["mid_up"].iloc[lo:idx+1].dropna()
    if len(mids) < 2:
        return 0.0
    return float(mids.iloc[-1] - mids.iloc[0])


def build_dataset(subdir: Path, asset_id: int,
                  target: str = "classification") -> list[dict]:
    """Extract labeled feature rows from all windows in a subdirectory.

    target="classification" — label is 1 if z direction matched outcome
    target="regression"     — label is realized PnL per dollar at current ask
    """
    rows = []
    files = sorted(subdir.glob("*.parquet"))

    for f in files:
        df = pd.read_parquet(f)
        if df.empty:
            continue

        price_col = ("chainlink_price" if "chainlink_price" in df.columns
                     else "chainlink_btc" if "chainlink_btc" in df.columns
                     else None)
        if price_col is None:
            continue
        if "last_trade_side_up" not in df.columns:
            continue

        start_px = df["window_start_price"].dropna()
        if start_px.empty:
            continue
        start_px = float(start_px.iloc[0])
        if start_px == 0:
            continue

        final_px = float(df[price_col].iloc[-1])
        outcome_up = 1 if final_px >= start_px else 0

        prices  = df[price_col].tolist()
        ts_list = df["ts_ms"].tolist()
        tau_all = df["time_remaining_s"].tolist()
        window_ts = ts_list[0]  # for walk-forward split

        for target_tau in TAU_CHECKPOINTS:
            best_idx = min(range(len(tau_all)),
                           key=lambda i: abs(tau_all[i] - target_tau))
            actual_tau = tau_all[best_idx]
            if abs(actual_tau - target_tau) > 30:
                continue

            # Vol for z-score
            lo = max(0, best_idx - VOL_LOOKBACK_S)
            sigma = compute_sigma(prices[lo:best_idx+1], ts_list[lo:best_idx+1])
            if sigma <= 0:
                continue

            # Baseline vol for regime ratio
            lo_base = max(0, best_idx - BASELINE_VOL_S)
            sigma_base = compute_sigma(prices[lo_base:best_idx+1],
                                       ts_list[lo_base:best_idx+1])
            vol_regime_ratio = (sigma / sigma_base
                                if sigma_base > 0 else 1.0)

            current_px = prices[best_idx]
            delta = (current_px - start_px) / start_px
            z_raw = delta / (sigma * math.sqrt(actual_tau))
            z = max(-1.5, min(1.5, z_raw))

            # Skip near-zero z (ambiguous signal)
            if abs(z) < MIN_Z_SIGNAL:
                continue

            signal_up = z > 0

            # Market features at decision row
            row = df.iloc[best_idx]
            spread_up   = float(row.get("spread_up",   0.0) or 0.0)
            spread_down = float(row.get("spread_down", 0.0) or 0.0)
            imb_up      = float(row.get("imbalance5_up",   0.0) or 0.0)
            imb_down    = float(row.get("imbalance5_down", 0.0) or 0.0)

            if pd.isna(spread_up) or pd.isna(spread_down):
                continue

            # Compute the trade we WOULD have placed and its realized PnL.
            # The "won" flag is the classification label; pnl_per_dollar is
            # the regression label.
            won = (signal_up == bool(outcome_up))
            if target == "regression":
                # Need ask price for the side we'd have bet
                if signal_up:
                    cost = float(row.get("best_ask_up", 0.0) or 0.0)
                else:
                    cost = float(row.get("best_ask_down", 0.0) or 0.0)
                # Skip rows with no usable ask (would have been FLAT live)
                if cost <= 0.0 or cost >= 1.0:
                    continue
                fee = _poly_fee(cost)
                if won:
                    # Win: receive $1, paid `cost` + fee per share
                    pnl_per_dollar = (1.0 - cost - fee) / cost
                else:
                    # Loss: forfeit the entire stake (cost) plus fee
                    pnl_per_dollar = -1.0 - (fee / cost)
                label: float = pnl_per_dollar
            else:
                label = int(won)

            buy_pressure   = compute_buy_pressure(df, best_idx)
            mid_momentum   = compute_mid_momentum(df, best_idx)
            hour_of_day    = pd.Timestamp(ts_list[best_idx], unit="ms", tz="UTC").hour
            is_weekend     = int(pd.Timestamp(ts_list[best_idx], unit="ms", tz="UTC")
                                 .weekday() >= 5)

            features = extract_features(
                z=z,
                sigma=sigma,
                tau=actual_tau,
                spread_up=spread_up,
                spread_down=spread_down,
                imbalance5_up=imb_up,
                imbalance5_down=imb_down,
                buy_pressure=buy_pressure,
                vol_regime_ratio=vol_regime_ratio,
                mid_up_momentum=mid_momentum,
                hour_of_day=hour_of_day,
                is_weekend=is_weekend,
                asset_id=asset_id,
            )

            rows.append({
                "features": features,
                "label": label,
                "window_ts": window_ts,
                "asset": subdir.name,
                "tau": target_tau,
                "z": z,
            })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Confidence threshold for trading (default 0.55)")
    parser.add_argument("--target", choices=["classification", "regression"],
                        default="classification",
                        help="classification: P(direction correct) — legacy. "
                             "regression: predicted PnL per dollar — pairs with "
                             "--filtration-mode size_mult.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output pkl path. Defaults to filtration_model.pkl "
                             "(or filtration_model_pnl.pkl for regression).")
    args = parser.parse_args()

    is_regression = args.target == "regression"
    output_path = Path(args.output) if args.output else (
        Path("filtration_model_pnl.pkl") if is_regression else MODEL_PATH
    )

    print(f"Building dataset (target={args.target})...")
    all_rows = []
    for asset_name, asset_id in ASSET_IDS.items():
        d = DATA_DIR / asset_name
        if not d.exists():
            continue
        rows = build_dataset(d, asset_id, target=args.target)
        all_rows.extend(rows)
        if rows:
            if is_regression:
                mean_label = sum(r['label'] for r in rows) / len(rows)
                print(f"  {asset_name}: {len(rows)} samples  "
                      f"(mean PnL/$: {mean_label:+.4f})")
            else:
                pos_rate = sum(r['label'] for r in rows) / len(rows)
                print(f"  {asset_name}: {len(rows)} samples  "
                      f"(positive rate: {pos_rate:.1%})")

    print(f"\nTotal: {len(all_rows)} samples")

    # Walk-forward split by timestamp (not random — must respect time order)
    all_rows.sort(key=lambda r: r["window_ts"])
    split_idx = int(len(all_rows) * TRAIN_FRAC)
    train_rows = all_rows[:split_idx]
    val_rows   = all_rows[split_idx:]

    X_train = np.array([r["features"] for r in train_rows], dtype=np.float32)
    X_val   = np.array([r["features"] for r in val_rows],   dtype=np.float32)
    if is_regression:
        y_train = np.array([r["label"] for r in train_rows], dtype=np.float32)
        y_val   = np.array([r["label"] for r in val_rows],   dtype=np.float32)
    else:
        y_train = np.array([r["label"] for r in train_rows], dtype=np.int32)
        y_val   = np.array([r["label"] for r in val_rows],   dtype=np.int32)

    print(f"\nTrain: {len(X_train)} samples | Val: {len(X_val)} samples")
    if is_regression:
        print(f"Train mean PnL/$: {y_train.mean():+.4f}  std: {y_train.std():.4f}")
        print(f"Val   mean PnL/$: {y_val.mean():+.4f}  std: {y_val.std():.4f}")
    else:
        print(f"Train positive rate: {y_train.mean():.1%} | Val: {y_val.mean():.1%}")

    # ── Train XGBoost ─────────────────────────────────────────────────────────
    print("\nTraining XGBoost...")
    if is_regression:
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="rmse",
            verbosity=0,
        )
    else:
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,             # shallow trees = less overfitting
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=20,     # require 20 samples per leaf
            reg_alpha=0.1,           # L1 regularization
            reg_lambda=1.0,          # L2 regularization
            eval_metric="logloss",
            verbosity=0,
        )
    model.fit(X_train, y_train, verbose=False)

    if is_regression:
        from filtration_model import RegressionWrapper
        wrapped = RegressionWrapper(model)
    else:
        # Calibrate probabilities (Platt scaling) so predict_proba is reliable
        # Use prefit mode: model is already trained, just fit the calibration layer
        print("Calibrating probabilities...")
        from sklearn.linear_model import LogisticRegression
        raw_proba = model.predict_proba(X_val)[:, 1]
        cal_model = LogisticRegression()
        cal_model.fit(raw_proba.reshape(-1, 1), y_val)
        from filtration_model import CalibratedWrapper
        wrapped = CalibratedWrapper(model, cal_model)

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred_score = wrapped.predict_proba(X_val)[:, 1]

    print("\n" + "=" * 60)
    print(f"VALIDATION RESULTS (target={args.target})")
    print("=" * 60)

    if is_regression:
        from sklearn.metrics import mean_squared_error, r2_score
        rmse = math.sqrt(mean_squared_error(y_val, y_pred_score))
        r2 = r2_score(y_val, y_pred_score)
        print(f"  RMSE:        {rmse:.4f}  (lower=better)")
        print(f"  R²:          {r2:.4f}  (1.0=perfect, 0.0=no better than mean)")
        print(f"  Mean pred:   {y_pred_score.mean():+.4f}  vs actual {y_val.mean():+.4f}")
        print(f"  Pred std:    {y_pred_score.std():.4f}  vs actual {y_val.std():.4f}")
        # Bin trades by predicted EV decile and look at realized PnL
        import numpy as _np
        deciles = _np.percentile(y_pred_score, [10, 20, 30, 40, 50, 60, 70, 80, 90])
        print(f"\n  Realized PnL by predicted-EV decile:")
        print(f"  {'decile':>8}  {'n':>5}  {'pred_ev':>8}  {'actual_pnl':>11}")
        for d in range(10):
            lo = deciles[d - 1] if d > 0 else -float('inf')
            hi = deciles[d] if d < 9 else float('inf')
            mask = (y_pred_score > lo) & (y_pred_score <= hi)
            if mask.sum() < 20:
                continue
            pred_mean = y_pred_score[mask].mean()
            actual_mean = y_val[mask].mean()
            print(f"  {d+1:>8}  {mask.sum():>5}  {pred_mean:>+8.4f}  {actual_mean:>+11.4f}")
    else:
        y_pred = (y_pred_score >= args.threshold).astype(int)
        print(f"  AUC-ROC:     {roc_auc_score(y_val, y_pred_score):.4f}  (0.5=random, 1.0=perfect)")
        print(f"  Brier score: {brier_score_loss(y_val, y_pred_score):.4f}  (lower=better, 0.25=random)")
        print(f"  Accuracy at threshold={args.threshold}: {accuracy_score(y_val, y_pred):.4f}")
        trades_kept = y_pred.sum()
        print(f"  Trades kept: {trades_kept}/{len(y_val)} = {trades_kept/len(y_val):.1%}")
        print("\n  Classification report (at threshold):")
        print(classification_report(y_val, y_pred, target_names=["wrong", "correct"],
                                     digits=3))

    # Feature importance
    if hasattr(wrapped, "feature_importances_"):
        importances = wrapped.feature_importances_
        ranked = sorted(zip(FEATURE_NAMES, importances),
                        key=lambda x: -x[1])
        print("  Top 10 feature importances:")
        for name, imp in ranked[:10]:
            bar = "█" * int(imp * 200)
            print(f"    {name:25s} {imp:.4f}  {bar}")

    # ── Save ──────────────────────────────────────────────────────────────────
    FiltrationModel.save(wrapped, output_path)
    print(f"\nModel saved → {output_path}")
    if is_regression:
        print(f"Use with: --filtration-mode size_mult --filtration-threshold 0.0")
        print(f"  (regressor predicts EV directly; threshold=0.0 means 'positive EV')")
    else:
        print(f"Use with: FiltrationModel.load(threshold={args.threshold})")


if __name__ == "__main__":
    main()
