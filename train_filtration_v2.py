#!/usr/bin/env python3
"""
Train the v2 filtration model on LIVE-ONLY parquets.

Context:
  - The existing filtration_model.pkl was trained on ~94% backfilled
    parquets where bid_depth5_up is NaN and size_bid_up is a dummy
    constant (100.0). Result: all 4 OBI features got 0% XGBoost
    importance because the training data had no real depth signal.
  - This script filters to live parquets only (562 btc_5m + 219 btc_15m
    as of 2026-04-11) and uses the full 90-feature set from features.py.
  - Sample is thin (~47h of btc_5m + ~55h of btc_15m) so this is an
    EXPLORATORY baseline, not a ship-ready model. Ship criterion per
    tasks/plans/track_b_ml_signal_2026-04-11.md: AUC > 0.55 AND >2000
    live windows AND top features pass physical-sense check.

Method:
  1. Scan parquets, filter to live-only using parquet_kind.filter_live
  2. For each live window, iterate rows and build ctx state incrementally
  3. Sample feature vectors at multiple decision-time taus (240, 180,
     120, 60 seconds remaining). Each sample is one training row.
  4. Label each sample by the FINAL window outcome (binance_mid at
     tau=0 vs window_start_price). 1 = UP won, 0 = DOWN won.
  5. Walk-forward 80/20 split (chronological, no leakage).
  6. Train XGBoost classifier with logistic calibration.
  7. Save as filtration_model_v2_YYYYMMDD.pkl (NOT overwriting v1).
  8. Report: AUC, log-loss, top feature importances, calibration plot
     summary, comparison vs v1 on the same held-out set.

Usage:
  .venv/bin/python3.11 train_filtration_v2.py --market btc_5m
  .venv/bin/python3.11 train_filtration_v2.py --market btc --max-windows 500
  .venv/bin/python3.11 train_filtration_v2.py --market both --save
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import math

from backtest_core import Snapshot, _compute_vol_deduped, norm_cdf
from features import compute_features, feature_names
from filtration_model import CalibratedWrapper
from market_config import get_config
from parquet_kind import filter_live, classify

# Late-window leakage features (2026-04-11 audit finding):
# When the bot samples at tau=60 on a 300s window, the following features
# encode "where the window has been" in a way that mechanically reveals the
# outcome. Per-tau AUC breakdown:
#   tau=240: AUC 0.76 (legitimate signal)
#   tau=60:  AUC 0.995 (leakage territory)
# Dropping these forces the model to rely on predictive, not mnemonic,
# features. Retrain AUC should stay ≥ 0.65 at tau=240 if the other features
# carry real information.
LEAKAGE_FEATURES = {
    "elapsed_frac_x_signed_z",   # (1 − tau/full) × sign(z) × |z| — encodes elapsed time × direction
    "dd_from_peak_z",             # drawdown from the window's peak z — the peak itself IS late info
    "peak_delta_seen",            # max |delta| seen in the window so far
    "signed_z_range",             # max_z − min_z — reveals how far the window has swung
    "run_length_current",         # consecutive same-direction seconds — path memory
    "n_zero_crossings",           # number of times the path crossed the start line
    "n_zero_crossings_normalized",
    "n_trades_fired_already",     # just tells us how many times the bot already traded
}

# Decision-time taus at which to sample feature vectors. For 5m markets,
# tau=240 is 60s elapsed (early), tau=60 is 240s elapsed (late). Each
# tau gives one training sample per window.
DECISION_TAUS_5M = [240, 180, 120, 60]
DECISION_TAUS_15M = [840, 720, 600, 480, 360, 240, 120]


def _compute_sigma_and_p_gbm(
    hist: list[float],
    ts_hist: list[int],
    effective_price: float,
    window_start_price: float,
    tau_s: float,
    vol_lookback_s: float,
    min_sigma: float,
    max_z: float,
) -> tuple[float, float]:
    """Compute the σ and p_gbm the signal would have computed at this tick.

    This mirrors signal_diffusion.DiffusionSignal._smoothed_sigma + _p_model
    without the Kalman filter (which needs persistent state). We use raw
    Yang-Zhang σ directly — the difference between raw and Kalman-smoothed
    is < 20% on typical data and this is exploratory training anyway.

    Returns (sigma_per_s, p_gbm). Both are always finite; sigma is floored
    at min_sigma and p_gbm is bounded by max_z.
    """
    # Window the history to the last vol_lookback_s seconds
    if not hist or not ts_hist:
        return min_sigma, 0.5
    end_ts = ts_hist[-1]
    cutoff = end_ts - int(vol_lookback_s * 1000)
    sl_p = []
    sl_t = []
    for i in range(len(ts_hist) - 1, -1, -1):
        if ts_hist[i] < cutoff:
            break
        sl_p.append(hist[i])
        sl_t.append(ts_hist[i])
    sl_p.reverse()
    sl_t.reverse()
    if len(sl_p) < 3:
        raw_sigma = min_sigma
    else:
        try:
            raw_sigma = _compute_vol_deduped(sl_p, sl_t)
        except Exception:
            raw_sigma = 0.0
        if raw_sigma <= 0:
            raw_sigma = min_sigma

    sigma_per_s = max(raw_sigma, min_sigma)

    # p_gbm from the same formula the signal uses
    if window_start_price > 0 and effective_price > 0 and tau_s > 0 and sigma_per_s > 0:
        delta_log = math.log(effective_price / window_start_price)
        z_raw = delta_log / (sigma_per_s * math.sqrt(tau_s))
        z_capped = max(-max_z, min(max_z, z_raw))
        p_gbm = norm_cdf(z_capped)
    else:
        p_gbm = 0.5

    return sigma_per_s, p_gbm


def extract_window_samples(
    parquet_path: Path,
    decision_taus: list[int],
    window_duration_s: float,
    vol_lookback_s: float,
    min_sigma: float,
    max_z: float,
) -> list[tuple[dict, int, int, int, int]]:
    """Extract training samples from one window.

    Yields one sample per decision tau as:
        (features, label, ts_ms, target_tau_s, actual_tau_s)

    Label = 1 if the window closed UP (last binance_mid > window_start_price),
    else 0.

    Critical: for each sample we recompute the σ and p_gbm the signal
    would have produced at that tick. Without this, features that depend
    on sigma_per_s (z-score, delta_per_typical_move) or p_gbm (pm_p_residual)
    are nonsense.
    """
    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        return []
    if len(df) < 10:
        return []

    # Determine the final outcome from the last row.
    last_row = df.iloc[-1]
    end_price = None
    if "binance_mid" in df.columns and pd.notna(last_row.get("binance_mid")):
        end_price = float(last_row["binance_mid"])
    if end_price is None and pd.notna(last_row.get("chainlink_price")):
        end_price = float(last_row["chainlink_price"])
    if end_price is None:
        return []

    wsp_raw = df["window_start_price"].iloc[0]
    if wsp_raw is None or (isinstance(wsp_raw, float) and math.isnan(wsp_raw)) or pd.isna(wsp_raw):
        return []
    try:
        window_start_price = float(wsp_raw)
    except (TypeError, ValueError):
        return []
    if window_start_price <= 0:
        return []

    label = 1 if end_price > window_start_price else 0

    # Iterate through rows, building ctx state and history.
    ctx: dict = {
        "inventory_up": 0,
        "inventory_down": 0,
        "_window_start_ms": int(df["window_start_ms"].iloc[0]),
    }
    hist: list[float] = []
    ts_hist: list[int] = []
    samples: list[tuple[dict, int, int, int, int]] = []
    target_taus = sorted(decision_taus, reverse=True)
    next_target_idx = 0

    for _, row in df.iterrows():
        if next_target_idx >= len(target_taus):
            break
        snap = Snapshot.from_row(row)
        if snap is None:
            continue

        # Update history from binance_mid (preferred) or chainlink fallback
        bn = row.get("binance_mid")
        if bn is None or pd.isna(bn):
            bn = row.get("chainlink_price")
        if bn is not None and not pd.isna(bn) and bn > 0:
            bn = float(bn)
            hist.append(bn)
            ts_hist.append(int(row["ts_ms"]))
            ctx["_binance_mid"] = bn

        ctx["_book_age_ms"] = 0.0  # backtest: assume fresh

        if len(hist) < 5:
            continue

        tau = snap.time_remaining_s
        while next_target_idx < len(target_taus) and tau <= target_taus[next_target_idx]:
            target_tau = int(target_taus[next_target_idx])
            # Recompute what the live signal would see at this tick:
            #   sigma_per_s from Yang-Zhang on the history window
            #   p_gbm from delta_log / (sigma * sqrt(tau))
            eff_price = ctx.get("_binance_mid", snap.chainlink_price)
            sigma_per_s, p_gbm = _compute_sigma_and_p_gbm(
                hist, ts_hist, eff_price, snap.window_start_price, tau,
                vol_lookback_s, min_sigma, max_z,
            )
            # Stash in ctx so features.py reads the right values
            ctx["_sigma_per_s"] = sigma_per_s
            ctx["_p_model_raw"] = p_gbm

            feats = compute_features(
                snap, ctx,
                history=hist, ts_history=ts_hist,
                window_duration_s=window_duration_s,
                sigma_per_s=sigma_per_s,
                p_gbm=p_gbm,
            )
            samples.append((feats, label, int(row["ts_ms"]), target_tau, int(round(tau))))
            next_target_idx += 1

    return samples


def build_training_set(
    data_dir: Path,
    decision_taus: list[int],
    window_duration_s: float,
    vol_lookback_s: float,
    min_sigma: float,
    max_z: float,
    max_windows: int | None = None,
) -> pd.DataFrame:
    """Iterate live parquets, extract samples, return a DataFrame."""
    all_paths = sorted(data_dir.glob("*.parquet"))
    live_paths = filter_live(all_paths, verbose=True)
    if max_windows is not None and len(live_paths) > max_windows:
        live_paths = live_paths[:max_windows]
        print(f"  capped to first {max_windows} live parquets")

    print(f"  extracting samples from {len(live_paths)} windows...")
    t0 = time.time()

    rows: list[dict] = []
    feature_cols = feature_names()
    for i, p in enumerate(live_paths, 1):
        samples = extract_window_samples(
            p, decision_taus, window_duration_s,
            vol_lookback_s, min_sigma, max_z,
        )
        for feats, label, ts_ms, target_tau, actual_tau in samples:
            row = {name: float(feats.get(name, 0.0)) for name in feature_cols}
            row["__label__"] = label
            row["__ts_ms__"] = ts_ms
            row["__slug__"] = p.stem
            row["__tau_target__"] = target_tau
            row["__tau_actual__"] = actual_tau
            rows.append(row)
        if i % 100 == 0:
            print(f"    {i}/{len(live_paths)} windows, {len(rows)} samples so far "
                  f"({time.time()-t0:.1f}s elapsed)")

    df = pd.DataFrame(rows)
    print(f"  built training set: {len(df)} rows × {len(feature_cols)} features "
          f"in {time.time()-t0:.1f}s")
    return df


def train_and_evaluate(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    test_frac: float = 0.2,
) -> tuple[CalibratedWrapper, dict, pd.DataFrame]:
    """Walk-forward train XGBoost + logistic calibration, return model + metrics."""
    try:
        import xgboost as xgb
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
    except ImportError as exc:
        raise RuntimeError(f"Missing dependency: {exc}. Install with: uv pip install xgboost scikit-learn")

    # Chronological split — no random shuffle
    df_sorted = train_df.sort_values("__ts_ms__").reset_index(drop=True)
    n = len(df_sorted)
    split = int(n * (1 - test_frac))
    train_part = df_sorted.iloc[:split]
    test_part = df_sorted.iloc[split:]
    print(f"  train: {len(train_part)} rows  test: {len(test_part)} rows")

    X_train = train_part[feature_cols].values.astype(np.float32)
    y_train = train_part["__label__"].values.astype(np.int32)
    X_test = test_part[feature_cols].values.astype(np.float32)
    y_test = test_part["__label__"].values.astype(np.int32)

    # XGBoost classifier. Conservative hyperparameters for small data.
    base = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    base.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Calibration head on the raw XGB probabilities
    raw_train = base.predict_proba(X_train)[:, 1]
    cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
    cal.fit(raw_train.reshape(-1, 1), y_train)

    # Evaluate
    raw_test = base.predict_proba(X_test)[:, 1]
    cal_test = cal.predict_proba(raw_test.reshape(-1, 1))[:, 1]

    metrics = {
        "n_train": len(train_part),
        "n_test": len(test_part),
        "base_auc": float(roc_auc_score(y_test, raw_test)),
        "cal_auc": float(roc_auc_score(y_test, cal_test)),
        "base_logloss": float(log_loss(y_test, raw_test, labels=[0, 1])),
        "cal_logloss": float(log_loss(y_test, cal_test, labels=[0, 1])),
        "base_brier": float(brier_score_loss(y_test, raw_test)),
        "cal_brier": float(brier_score_loss(y_test, cal_test)),
        "test_base_rate": float(y_test.mean()),
        "train_base_rate": float(y_train.mean()),
    }

    eval_df = test_part[[
        "__market__", "__label__", "__ts_ms__", "__slug__",
        "__tau_target__", "__tau_actual__",
    ]].copy()
    eval_df["__raw_pred__"] = raw_test
    eval_df["__cal_pred__"] = cal_test

    wrapper = CalibratedWrapper(base, cal)
    return wrapper, metrics, eval_df


def report_feature_importance(wrapper: CalibratedWrapper, feature_cols: list[str], top_n: int = 30):
    """Print feature importances in ranked order."""
    imps = wrapper.feature_importances_
    total = float(imps.sum()) if imps.sum() > 0 else 1.0
    ranked = sorted(enumerate(imps), key=lambda x: -x[1])
    print(f"\n  Top {top_n} features by XGBoost importance:")
    print(f"  {'rank':>4}  {'feature':<35} {'importance':>12}  {'% total':>8}")
    print("  " + "-" * 62)
    for rank, (idx, imp) in enumerate(ranked[:top_n], 1):
        name = feature_cols[idx] if idx < len(feature_cols) else f"idx{idx}"
        pct = imp / total * 100
        print(f"  {rank:>4}  {name:<35} {imp:>12.4f}  {pct:>7.2f}%")
    # Count dead features
    dead = sum(1 for _, imp in ranked if imp < 1e-6)
    print(f"  ({dead} features with importance < 1e-6, i.e. unused by the trees)")


def report_segment_metrics(eval_df: pd.DataFrame, pred_col: str = "__raw_pred__") -> None:
    """Print held-out metrics by market and target tau."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    def _row(group: pd.DataFrame) -> tuple[float, float, float]:
        y = group["__label__"].to_numpy()
        p = group[pred_col].to_numpy()
        auc = float("nan")
        if len(np.unique(y)) >= 2:
            auc = float(roc_auc_score(y, p))
        ll = float(log_loss(y, p, labels=[0, 1]))
        brier = float(brier_score_loss(y, p))
        return auc, ll, brier

    by_market = []
    for market, group in eval_df.groupby("__market__", sort=True):
        auc, ll, brier = _row(group)
        by_market.append({
            "market": market,
            "n": len(group),
            "base_rate": float(group["__label__"].mean()),
            "auc": auc,
            "logloss": ll,
            "brier": brier,
        })
    market_df = pd.DataFrame(by_market)
    if not market_df.empty:
        print("\n=== Held-Out Metrics By Market (raw XGB) ===")
        print(f"  {'market':<8} {'n':>5} {'pos%':>7} {'auc':>8} {'logloss':>10} {'brier':>8}")
        print("  " + "-" * 54)
        for _, row in market_df.sort_values("market").iterrows():
            auc_txt = f"{row['auc']:.3f}" if pd.notna(row["auc"]) else "n/a"
            print(
                f"  {row['market']:<8} {int(row['n']):>5} {100*row['base_rate']:>6.1f}% "
                f"{auc_txt:>8} {row['logloss']:>10.4f} {row['brier']:>8.4f}"
            )

    by_tau = []
    for (market, tau), group in eval_df.groupby(["__market__", "__tau_target__"], sort=True):
        auc, ll, brier = _row(group)
        by_tau.append({
            "market": market,
            "tau_target": int(tau),
            "n": len(group),
            "base_rate": float(group["__label__"].mean()),
            "auc": auc,
            "logloss": ll,
            "brier": brier,
        })
    tau_df = pd.DataFrame(by_tau)
    if not tau_df.empty:
        print("\n=== Held-Out Metrics By Market × Target Tau (raw XGB) ===")
        print(f"  {'market':<8} {'tau':>5} {'n':>5} {'pos%':>7} {'auc':>8} {'logloss':>10} {'brier':>8}")
        print("  " + "-" * 60)
        tau_df = tau_df.sort_values(["market", "tau_target"], ascending=[True, False])
        for _, row in tau_df.iterrows():
            auc_txt = f"{row['auc']:.3f}" if pd.notna(row["auc"]) else "n/a"
            print(
                f"  {row['market']:<8} {int(row['tau_target']):>5} {int(row['n']):>5} "
                f"{100*row['base_rate']:>6.1f}% {auc_txt:>8} {row['logloss']:>10.4f} {row['brier']:>8.4f}"
            )


def main():
    ap = argparse.ArgumentParser(description="Train filtration v2 on live-only parquets")
    ap.add_argument("--market", default="btc_5m",
                    choices=["btc_5m", "btc", "both"],
                    help="which market to train (btc = 15m; 'both' trains a combined model)")
    ap.add_argument("--max-windows", type=int, default=None,
                    help="cap the number of live windows (default: use all)")
    ap.add_argument("--save", action="store_true",
                    help="save the trained model (default: dry run, don't save)")
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="held-out test fraction (chronological tail, default 0.2)")
    args = ap.parse_args()

    if args.market == "both":
        markets = ["btc_5m", "btc"]
    else:
        markets = [args.market]

    from backtest import DATA_DIR  # avoid import cycle at top of file
    all_training_dfs = []
    for m in markets:
        cfg = get_config(m)
        data_dir = DATA_DIR / cfg.data_subdir
        print(f"\n=== Building training set for {m} ===")
        taus = DECISION_TAUS_5M if "_5m" in m else DECISION_TAUS_15M
        # vol_lookback mirrors the per-market value set in live_trader.py
        vol_lookback = 300.0 if "_5m" in m else 240.0
        df = build_training_set(
            data_dir, taus, cfg.window_duration_s,
            vol_lookback_s=vol_lookback,
            min_sigma=cfg.min_sigma,
            max_z=cfg.max_z,
            max_windows=args.max_windows,
        )
        if len(df) == 0:
            print(f"  No samples for {m}, skipping")
            continue
        df["__market__"] = m
        all_training_dfs.append(df)

    if not all_training_dfs:
        print("No training data. Exiting.")
        return

    full_df = pd.concat(all_training_dfs, ignore_index=True)
    feature_cols = [f for f in feature_names() if f not in LEAKAGE_FEATURES]
    print(f"\n=== Training on {len(full_df)} samples, {len(feature_cols)} features "
          f"({len(LEAKAGE_FEATURES)} leakage features dropped) ===")
    print(f"  label balance: {full_df['__label__'].mean():.3f} (1=UP won)")
    print(f"  dropped: {sorted(LEAKAGE_FEATURES)}")

    wrapper, metrics, eval_df = train_and_evaluate(
        full_df, feature_cols, test_frac=args.test_frac
    )

    print(f"\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    if metrics["cal_logloss"] > metrics["base_logloss"] and metrics["cal_brier"] > metrics["base_brier"]:
        print("  note: logistic calibration underperformed the raw XGB scores on the held-out tail")

    report_feature_importance(wrapper, feature_cols, top_n=30)
    report_segment_metrics(eval_df, pred_col="__raw_pred__")

    if args.save:
        date_tag = datetime.now().strftime("%Y%m%d")
        out = ROOT / f"filtration_model_v2_{date_tag}.pkl"
        with open(out, "wb") as f:
            pickle.dump(wrapper, f)
        print(f"\n  Saved model to {out}")
    else:
        print("\n  [DRY RUN] Model not saved. Pass --save to write to disk.")


if __name__ == "__main__":
    main()
