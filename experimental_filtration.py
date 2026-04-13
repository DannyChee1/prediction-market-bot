"""Helpers for the experimental live filtration bundles.

These bundles are distinct from the legacy 29-feature filtration_model.pkl:
they use the newer `features.py` feature set plus contemplated-trade overlay
features. The live signal loads them only when explicitly opted in.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest_core import poly_fee
from features import compute_features


def default_tau_tolerance_s(market: str) -> float:
    return 90.0 if market == "btc" else 30.0


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


def trade_overlay(snapshot, p_gbm: float) -> dict[str, float] | None:
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
    else:
        side = "BUY_DOWN"
        ask = ask_down
        fee = fee_down
        eff_cost = eff_down
        bid = bid_down if bid_down is not None else 0.0
        p_side = 1.0 - p_gbm
        raw_edge = edge_down
        alt_edge = edge_up

    mid = None
    if bid > 0.0 and ask > 0.0:
        mid = (bid + ask) / 2.0

    return {
        "trade_side": side,
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


def load_bundle(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, dict):
        raise TypeError(f"experimental filtration bundle must be a dict, got {type(bundle).__name__}")
    if "model" not in bundle or "feature_cols" not in bundle:
        raise ValueError("experimental filtration bundle missing required keys: model, feature_cols")
    market = str(bundle.get("market", "btc_5m"))
    bundle.setdefault("market", market)
    bundle.setdefault("target", "edge_share")
    bundle.setdefault("taus", [])
    bundle.setdefault("min_raw_edge", 0.0)
    bundle.setdefault("tau_tolerance_s", default_tau_tolerance_s(market))
    return bundle


def score_bundle(
    bundle: dict[str, Any],
    snapshot,
    ctx: dict,
    *,
    sigma_per_s: float,
    p_gbm: float,
    tau: float,
    window_duration_s: float,
) -> tuple[float | None, str]:
    taus = [int(t) for t in bundle.get("taus", [])]
    tol = float(bundle.get("tau_tolerance_s", default_tau_tolerance_s(str(bundle.get("market", "btc_5m")))))
    if taus and min(abs(float(tau) - float(t)) for t in taus) > tol:
        return None, "tau_window"

    overlay = trade_overlay(snapshot, p_gbm)
    if overlay is None:
        return None, "no_trade_overlay"
    if overlay["trade_raw_edge_gbm"] <= float(bundle.get("min_raw_edge", 0.0)):
        return None, "raw_edge_below_min"

    feats = compute_features(
        snapshot,
        ctx,
        history=ctx.get("price_history", []),
        ts_history=ctx.get("ts_history", []),
        window_duration_s=window_duration_s,
        sigma_per_s=sigma_per_s,
        p_gbm=p_gbm,
    )
    row = dict(feats)
    row.update(overlay)
    feature_cols = list(bundle["feature_cols"])
    X = np.array([[float(row.get(name, 0.0)) for name in feature_cols]], dtype=np.float32)
    model = bundle["model"]
    target = str(bundle.get("target", "edge_share"))
    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(X)[0, 1])
    elif target == "edge_share" and hasattr(model, "predict"):
        score = float(model.predict(X)[0])
    else:
        raise TypeError(f"bundle model type {type(model).__name__} is not scoreable")
    return score, "ok"
