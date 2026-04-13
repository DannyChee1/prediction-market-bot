#!/usr/bin/env python3
"""
Train a dedicated BTC early-tau filtration model on full live windows only.

This script is intentionally narrower than train_filtration_v2.py:

  - market: one BTC timeframe at a time (`btc_5m` or `btc`)
  - windows: full live windows only (no backfill, no partials)
  - taus: early checkpoints only (market-specific defaults)
  - labels: aligned to the contemplated GBM trade, not generic window-UP

Two target modes are supported:

  --target direction
      Classification target. At each early tau, choose the side the raw GBM
      edge would have traded (best of BUY_UP / BUY_DOWN after fee). Label 1 if
      that contemplated trade resolved in the money, else 0.

  --target edge_share
      Regression target. Same contemplated trade, but label is realized edge
      per share:
          resolved_value - ask_price - fee
      where resolved_value is 1.0 if the chosen side won, else 0.0.
      This is a bounded "cheapness" target and is better aligned with
      "buying positions cheaper than they were actually worth" than PnL/$.

  --target value_weighted_direction
      Same binary label as --target direction, but the XGBoost loss is weighted
      by abs(realized edge/share). This keeps the robust classification setup
      while forcing the model to care more about expensive mistakes and large
      cheap wins.

Usage:
  .venv/bin/python train_filtration_btc5_early.py --market btc_5m
  .venv/bin/python train_filtration_btc5_early.py --market btc --target edge_share --taus 840
  .venv/bin/python train_filtration_btc5_early.py --market btc_5m --target direction --save
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

from backtest import DATA_DIR
from backtest_core import Snapshot, poly_fee
from experimental_filtration import default_tau_tolerance_s
from features import compute_features, feature_names
from filtration_model import RegressionWrapper
from market_config import get_config
from parquet_kind import filter_live
from train_filtration_v2 import LEAKAGE_FEATURES, _compute_sigma_and_p_gbm


TRADE_FEATURE_NAMES = [
    "trade_is_buy_up",
    "trade_p_side_gbm",
    "trade_ask_price",
    "trade_fee",
    "trade_eff_cost",
    "trade_mid_price",
    "trade_ask_minus_mid",
    "trade_spread_side",
    "trade_raw_edge_gbm",
    "trade_alt_edge_gbm",
    "trade_edge_gap_gbm",
]


def _default_taus_for_market(market: str) -> list[int]:
    if market == "btc":
        return [840, 720]
    return [240, 180]


def _parse_taus(raw: str | None, market: str) -> list[int]:
    if raw is None or not raw.strip():
        return _default_taus_for_market(market)
    vals = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise ValueError("taus cannot be empty")
    return sorted(set(vals), reverse=True)


def _safe_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _trade_overlay(snapshot: Snapshot, p_gbm: float) -> dict[str, float] | None:
    ask_up = _safe_float(snapshot.best_ask_up)
    ask_down = _safe_float(snapshot.best_ask_down)
    bid_up = _safe_float(snapshot.best_bid_up)
    bid_down = _safe_float(snapshot.best_bid_down)
    if ask_up is None or ask_down is None:
        return None
    if ask_up <= 0.0 or ask_up >= 1.0 or ask_down <= 0.0 or ask_down >= 1.0:
        return None

    fee_up = poly_fee(ask_up)
    fee_down = poly_fee(ask_down)
    eff_up = ask_up + fee_up
    eff_down = ask_down + fee_down
    if eff_up <= 0.0 or eff_up >= 1.0 or eff_down <= 0.0 or eff_down >= 1.0:
        return None

    edge_up = p_gbm - eff_up
    edge_down = (1.0 - p_gbm) - eff_down
    if edge_up >= edge_down:
        side = "BUY_UP"
        ask = ask_up
        fee = fee_up
        eff_cost = eff_up
        bid = bid_up if bid_up is not None else 0.0
        p_side = p_gbm
        raw_edge = edge_up
        alt_edge = edge_down
        resolved_if_up = 1.0
    else:
        side = "BUY_DOWN"
        ask = ask_down
        fee = fee_down
        eff_cost = eff_down
        bid = bid_down if bid_down is not None else 0.0
        p_side = 1.0 - p_gbm
        raw_edge = edge_down
        alt_edge = edge_up
        resolved_if_up = 0.0

    mid = None
    if bid > 0.0 and ask > 0.0:
        mid = (bid + ask) / 2.0

    return {
        "trade_side": side,
        "trade_resolved_if_up": resolved_if_up,
        "trade_is_buy_up": 1.0 if side == "BUY_UP" else 0.0,
        "trade_p_side_gbm": float(p_side),
        "trade_ask_price": float(ask),
        "trade_fee": float(fee),
        "trade_eff_cost": float(eff_cost),
        "trade_mid_price": float(mid) if mid is not None else 0.0,
        "trade_ask_minus_mid": float(ask - mid) if mid is not None else 0.0,
        "trade_spread_side": float(ask - bid) if bid > 0.0 else 0.0,
        "trade_raw_edge_gbm": float(raw_edge),
        "trade_alt_edge_gbm": float(alt_edge),
        "trade_edge_gap_gbm": float(raw_edge - alt_edge),
    }


def extract_trade_samples(
    parquet_path: Path,
    decision_taus: list[int],
    *,
    window_duration_s: float,
    vol_lookback_s: float,
    min_sigma: float,
    max_z: float,
    min_raw_edge: float,
) -> list[dict]:
    """Extract contemplated-trade training rows from one full window."""
    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        return []
    if len(df) < 10:
        return []

    last_row = df.iloc[-1]
    end_price = None
    if "binance_mid" in df.columns and pd.notna(last_row.get("binance_mid")):
        end_price = float(last_row["binance_mid"])
    if end_price is None and pd.notna(last_row.get("chainlink_price")):
        end_price = float(last_row["chainlink_price"])
    if end_price is None:
        return []

    wsp_raw = df["window_start_price"].iloc[0]
    if wsp_raw is None or pd.isna(wsp_raw):
        return []
    try:
        window_start_price = float(wsp_raw)
    except (TypeError, ValueError):
        return []
    if window_start_price <= 0:
        return []

    outcome_up = 1 if end_price > window_start_price else 0

    ctx: dict = {
        "inventory_up": 0,
        "inventory_down": 0,
        "_window_start_ms": int(df["window_start_ms"].iloc[0]),
    }
    hist: list[float] = []
    ts_hist: list[int] = []
    rows: list[dict] = []
    target_taus = sorted(decision_taus, reverse=True)
    next_target_idx = 0

    for _, row in df.iterrows():
        if next_target_idx >= len(target_taus):
            break
        snap = Snapshot.from_row(row)
        if snap is None:
            continue

        bn = row.get("binance_mid")
        if bn is None or pd.isna(bn):
            bn = row.get("chainlink_price")
        if bn is not None and not pd.isna(bn) and bn > 0:
            bn = float(bn)
            hist.append(bn)
            ts_hist.append(int(row["ts_ms"]))
            ctx["_binance_mid"] = bn

        ctx["_book_age_ms"] = 0.0

        if len(hist) < 5:
            continue

        tau = snap.time_remaining_s
        while next_target_idx < len(target_taus) and tau <= target_taus[next_target_idx]:
            target_tau = int(target_taus[next_target_idx])
            sigma_per_s, p_gbm = _compute_sigma_and_p_gbm(
                hist,
                ts_hist,
                ctx.get("_binance_mid", snap.chainlink_price),
                snap.window_start_price,
                tau,
                vol_lookback_s,
                min_sigma,
                max_z,
            )
            ctx["_sigma_per_s"] = sigma_per_s
            ctx["_p_model_raw"] = p_gbm

            feats = compute_features(
                snap,
                ctx,
                history=hist,
                ts_history=ts_hist,
                window_duration_s=window_duration_s,
                sigma_per_s=sigma_per_s,
                p_gbm=p_gbm,
            )
            overlay = _trade_overlay(snap, p_gbm)
            if overlay is not None and overlay["trade_raw_edge_gbm"] > min_raw_edge:
                resolved_value = float(outcome_up) if overlay["trade_side"] == "BUY_UP" else float(1 - outcome_up)
                eff_cost = overlay["trade_eff_cost"]
                ask_price = overlay["trade_ask_price"]
                edge_share = resolved_value - eff_cost
                pnl_per_dollar = edge_share / ask_price if ask_price > 0 else 0.0

                sample = {name: float(feats.get(name, 0.0)) for name in feature_names()}
                sample.update({name: float(overlay.get(name, 0.0)) for name in TRADE_FEATURE_NAMES})
                sample["__label_direction__"] = int(resolved_value > 0.5)
                sample["__label_edge_share__"] = float(edge_share)
                sample["__label_pnl_per_dollar__"] = float(pnl_per_dollar)
                sample["__raw_gbm_edge__"] = float(overlay["trade_raw_edge_gbm"])
                sample["__p_side_gbm__"] = float(overlay["trade_p_side_gbm"])
                sample["__ts_ms__"] = int(row["ts_ms"])
                sample["__tau_target__"] = target_tau
                sample["__tau_actual__"] = int(round(tau))
                sample["__slug__"] = parquet_path.stem
                sample["__trade_side__"] = overlay["trade_side"]
                rows.append(sample)
            next_target_idx += 1

    return rows


def build_training_set(
    data_dir: Path,
    decision_taus: list[int],
    *,
    window_duration_s: float,
    vol_lookback_s: float,
    min_sigma: float,
    max_z: float,
    min_raw_edge: float,
    max_windows: int | None = None,
) -> pd.DataFrame:
    paths = sorted(data_dir.glob("*.parquet"))
    live_paths = filter_live(paths, verbose=True)
    if max_windows is not None and len(live_paths) > max_windows:
        live_paths = live_paths[:max_windows]
        print(f"  capped to first {max_windows} live parquets")

    print(f"  extracting contemplated trades from {len(live_paths)} windows...")
    t0 = time.time()
    rows: list[dict] = []
    for i, p in enumerate(live_paths, 1):
        rows.extend(
            extract_trade_samples(
                p,
                decision_taus,
                window_duration_s=window_duration_s,
                vol_lookback_s=vol_lookback_s,
                min_sigma=min_sigma,
                max_z=max_z,
                min_raw_edge=min_raw_edge,
            )
        )
        if i % 100 == 0:
            print(f"    {i}/{len(live_paths)} windows, {len(rows)} samples so far ({time.time()-t0:.1f}s elapsed)")
    df = pd.DataFrame(rows)
    print(f"  built dataset: {len(df)} rows in {time.time()-t0:.1f}s")
    return df


def _feature_columns() -> list[str]:
    base = [f for f in feature_names() if f not in LEAKAGE_FEATURES]
    return base + TRADE_FEATURE_NAMES


def _split_time(df: pd.DataFrame, test_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("__ts_ms__").reset_index(drop=True)
    split = int(len(ordered) * (1.0 - test_frac))
    return ordered.iloc[:split].copy(), ordered.iloc[split:].copy()


def _report_score_deciles(
    eval_df: pd.DataFrame,
    score_col: str,
    *,
    top_n: int = 10,
) -> None:
    if eval_df.empty:
        return
    tmp = eval_df[[score_col, "__label_edge_share__", "__label_pnl_per_dollar__", "__label_direction__"]].copy()
    q = min(top_n, len(tmp))
    if q < 2:
        return
    try:
        tmp["bucket"] = pd.qcut(tmp[score_col], q=q, labels=False, duplicates="drop")
    except ValueError:
        return
    tmp["bucket"] = tmp["bucket"].astype(int)
    grouped = (
        tmp.groupby("bucket", sort=True)
        .agg(
            n=(score_col, "size"),
            mean_score=(score_col, "mean"),
            mean_edge_share=("__label_edge_share__", "mean"),
            mean_pnl_per_dollar=("__label_pnl_per_dollar__", "mean"),
            hit_rate=("__label_direction__", "mean"),
        )
        .reset_index()
        .sort_values("bucket", ascending=False)
    )
    print("\n=== Validation Score Deciles (highest score first) ===")
    print(f"  {'bucket':>6} {'n':>5} {'score':>10} {'edge/sh':>10} {'pnl/$':>10} {'hit%':>8}")
    print("  " + "-" * 60)
    for _, row in grouped.iterrows():
        print(
            f"  {int(row['bucket']):>6} {int(row['n']):>5} {row['mean_score']:>10.4f} "
            f"{row['mean_edge_share']:>10.4f} {row['mean_pnl_per_dollar']:>10.4f} "
            f"{100*row['hit_rate']:>7.1f}%"
        )

    top_bucket = grouped.iloc[0]
    bottom_bucket = grouped.iloc[-1]
    overall_edge = float(tmp["__label_edge_share__"].mean())
    print(
        f"\n  overall mean edge/share: {overall_edge:+.4f} | "
        f"top bucket: {top_bucket['mean_edge_share']:+.4f} | "
        f"bottom bucket: {bottom_bucket['mean_edge_share']:+.4f}"
    )


def _report_tau_summary(eval_df: pd.DataFrame, score_col: str, target: str) -> None:
    if eval_df.empty:
        return
    print("\n=== Validation By Target Tau ===")
    if target in {"direction", "value_weighted_direction"}:
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

        print(f"  {'tau':>5} {'n':>5} {'hit%':>8} {'auc':>8} {'logloss':>10} {'brier':>8}")
        print("  " + "-" * 54)
        for tau, group in eval_df.groupby("__tau_target__", sort=True):
            y = group["__label_direction__"].to_numpy()
            p = group[score_col].to_numpy()
            auc = float("nan")
            if len(np.unique(y)) >= 2:
                auc = float(roc_auc_score(y, p))
            auc_txt = f"{auc:.3f}" if np.isfinite(auc) else "n/a"
            print(
                f"  {int(tau):>5} {len(group):>5} {100*y.mean():>7.1f}% "
                f"{auc_txt:>8} {log_loss(y, p, labels=[0, 1]):>10.4f} "
                f"{brier_score_loss(y, p):>8.4f}"
            )
    else:
        print(f"  {'tau':>5} {'n':>5} {'pred':>10} {'edge/sh':>10} {'pnl/$':>10} {'hit%':>8}")
        print("  " + "-" * 58)
        for tau, group in eval_df.groupby("__tau_target__", sort=True):
            print(
                f"  {int(tau):>5} {len(group):>5} {group[score_col].mean():>10.4f} "
                f"{group['__label_edge_share__'].mean():>10.4f} "
                f"{group['__label_pnl_per_dollar__'].mean():>10.4f} "
                f"{100*group['__label_direction__'].mean():>7.1f}%"
            )


def train_direction(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    sample_weight_mode: str = "none",
):
    import xgboost as xgb
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["__label_direction__"].to_numpy(dtype=np.int32)
    X_eval = eval_df[feature_cols].to_numpy(dtype=np.float32)
    y_eval = eval_df["__label_direction__"].to_numpy(dtype=np.int32)
    sample_weight = None
    if sample_weight_mode == "abs_edge":
        sample_weight = np.abs(train_df["__label_edge_share__"].to_numpy(dtype=np.float32))
        sample_weight = np.maximum(sample_weight, 1e-3)

    model = xgb.XGBClassifier(
        n_estimators=250,
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
    fit_kwargs = {"eval_set": [(X_eval, y_eval)], "verbose": False}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(X_train, y_train, **fit_kwargs)
    score = model.predict_proba(X_eval)[:, 1]

    metrics = {
        "n_train": len(train_df),
        "n_test": len(eval_df),
        "train_hit_rate": float(y_train.mean()),
        "test_hit_rate": float(y_eval.mean()),
        "auc": float(roc_auc_score(y_eval, score)) if len(np.unique(y_eval)) >= 2 else float("nan"),
        "logloss": float(log_loss(y_eval, score, labels=[0, 1])),
        "brier": float(brier_score_loss(y_eval, score)),
        "test_mean_edge_share": float(eval_df["__label_edge_share__"].mean()),
        "test_mean_pnl_per_dollar": float(eval_df["__label_pnl_per_dollar__"].mean()),
    }
    if sample_weight is not None:
        metrics["train_weight_mean"] = float(np.mean(sample_weight))
        metrics["train_weight_std"] = float(np.std(sample_weight))
    out = eval_df.copy()
    out["__score__"] = score
    return model, metrics, out


def train_edge_share(train_df: pd.DataFrame, eval_df: pd.DataFrame, feature_cols: list[str]):
    import xgboost as xgb

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["__label_edge_share__"].to_numpy(dtype=np.float32)
    X_eval = eval_df[feature_cols].to_numpy(dtype=np.float32)
    y_eval = eval_df["__label_edge_share__"].to_numpy(dtype=np.float32)

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)
    score = model.predict(X_eval)

    corr = float(np.corrcoef(score, y_eval)[0, 1]) if len(score) >= 2 and np.std(score) > 0 and np.std(y_eval) > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((score - y_eval) ** 2)))
    metrics = {
        "n_train": len(train_df),
        "n_test": len(eval_df),
        "train_mean_edge_share": float(y_train.mean()),
        "test_mean_edge_share": float(y_eval.mean()),
        "train_mean_pnl_per_dollar": float(train_df["__label_pnl_per_dollar__"].mean()),
        "test_mean_pnl_per_dollar": float(eval_df["__label_pnl_per_dollar__"].mean()),
        "rmse": rmse,
        "corr": corr,
        "pred_mean": float(np.mean(score)),
        "pred_std": float(np.std(score)),
        "test_hit_rate": float(eval_df["__label_direction__"].mean()),
    }
    out = eval_df.copy()
    out["__score__"] = score
    return RegressionWrapper(model), metrics, out


def report_feature_importance(model, feature_cols: list[str], top_n: int = 25) -> None:
    if not hasattr(model, "feature_importances_"):
        return
    imps = np.asarray(model.feature_importances_)
    total = float(imps.sum()) if imps.sum() > 0 else 1.0
    ranked = sorted(enumerate(imps), key=lambda x: -x[1])
    print(f"\n=== Top {top_n} Features ===")
    print(f"  {'rank':>4} {'feature':<35} {'imp':>10} {'%':>7}")
    print("  " + "-" * 62)
    for rank, (idx, imp) in enumerate(ranked[:top_n], 1):
        name = feature_cols[idx] if idx < len(feature_cols) else f"idx{idx}"
        print(f"  {rank:>4} {name:<35} {imp:>10.4f} {100*imp/total:>6.2f}%")


def main():
    ap = argparse.ArgumentParser(description="Train a BTC early-tau filter on full live windows only")
    ap.add_argument("--market", choices=["btc_5m", "btc"], default="btc_5m")
    ap.add_argument("--taus", default=None,
                    help="comma-separated target taus; defaults to 240,180 for btc_5m and 840,720 for btc")
    ap.add_argument("--target", choices=["direction", "edge_share", "value_weighted_direction"], default="edge_share")
    ap.add_argument("--min-raw-edge", type=float, default=0.0,
                    help="only keep rows where the contemplated raw GBM edge exceeds this threshold")
    ap.add_argument("--tau-tolerance", type=float, default=None,
                    help="saved bundle metadata: live gate only applies within this many seconds of a trained tau")
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    taus = _parse_taus(args.taus, args.market)
    cfg = get_config(args.market)
    data_dir = DATA_DIR / cfg.data_subdir
    vol_lookback = 300.0 if args.market == "btc_5m" else 240.0

    print(f"=== Building {args.market} early dataset (taus={taus}, target={args.target}) ===")
    df = build_training_set(
        data_dir,
        taus,
        window_duration_s=cfg.window_duration_s,
        vol_lookback_s=vol_lookback,
        min_sigma=cfg.min_sigma,
        max_z=cfg.max_z,
        min_raw_edge=args.min_raw_edge,
        max_windows=args.max_windows,
    )
    if df.empty:
        print("No rows extracted. Exiting.")
        return

    feature_cols = _feature_columns()
    train_df, eval_df = _split_time(df, args.test_frac)
    print(f"\n=== Dataset ===")
    print(f"  rows: {len(df)}  train: {len(train_df)}  test: {len(eval_df)}")
    print(f"  taus: {taus}")
    print(f"  raw gbm edge mean: train={train_df['__raw_gbm_edge__'].mean():+.4f} test={eval_df['__raw_gbm_edge__'].mean():+.4f}")
    print(f"  realized edge/share mean: train={train_df['__label_edge_share__'].mean():+.4f} test={eval_df['__label_edge_share__'].mean():+.4f}")
    print(f"  realized pnl/$ mean: train={train_df['__label_pnl_per_dollar__'].mean():+.4f} test={eval_df['__label_pnl_per_dollar__'].mean():+.4f}")
    print(f"  contemplated side hit rate: train={train_df['__label_direction__'].mean():.3f} test={eval_df['__label_direction__'].mean():.3f}")

    if args.target == "direction":
        model, metrics, scored = train_direction(train_df, eval_df, feature_cols)
    elif args.target == "value_weighted_direction":
        model, metrics, scored = train_direction(
            train_df,
            eval_df,
            feature_cols,
            sample_weight_mode="abs_edge",
        )
    else:
        model, metrics, scored = train_edge_share(train_df, eval_df, feature_cols)

    print(f"\n=== Metrics ({args.target}) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    base_model = model._base if hasattr(model, "_base") else model
    report_feature_importance(base_model, feature_cols)
    _report_tau_summary(scored, "__score__", args.target)
    _report_score_deciles(scored, "__score__")

    if args.save:
        tag = datetime.now().strftime("%Y%m%d")
        market_tag = args.market.replace("_", "")
        out = ROOT / f"filtration_{market_tag}_early_{args.target}_{tag}.pkl"
        bundle = {
            "model": model,
            "feature_cols": feature_cols,
            "taus": taus,
            "target": args.target,
            "market": args.market,
            "min_raw_edge": args.min_raw_edge,
            "tau_tolerance_s": (
                float(args.tau_tolerance)
                if args.tau_tolerance is not None
                else default_tau_tolerance_s(args.market)
            ),
            "trained_at": datetime.now().isoformat(),
        }
        with open(out, "wb") as f:
            pickle.dump(bundle, f)
        print(f"\nSaved bundle to {out}")
    else:
        print("\n[DRY RUN] Bundle not saved. Pass --save to write to disk.")


if __name__ == "__main__":
    main()
