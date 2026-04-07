#!/usr/bin/env python3
"""
DiffusionSignal worker for dashboard.py.

Runs under Python 3.11+, keeps per-market signal state in memory, and
returns JSON diagnostics for each evaluation request.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import DATA_DIR, DiffusionSignal, Snapshot, build_calibration_table, _compute_vol_deduped
from market_config import get_config


def _build_sigma_calibration(data_subdir: str) -> float | None:
    cal_dir = DATA_DIR / data_subdir
    sigmas = []
    files = sorted(cal_dir.glob("*.parquet"))[-200:]
    for path in files:
        df = pd.read_parquet(path)
        if df.empty:
            continue
        pcol = "chainlink_price" if "chainlink_price" in df.columns else "chainlink_btc"
        if pcol not in df.columns:
            continue
        prices = df[pcol].tolist()
        ts_list = df["ts_ms"].tolist() if "ts_ms" in df.columns else None
        sigma = _compute_vol_deduped(prices, ts_list)
        if sigma > 0:
            sigmas.append(float(sigma))
    if not sigmas:
        return None
    return max(float(np.median(sigmas)), 1e-7)


def _signal_kwargs(mkey: str, settings: dict) -> tuple[dict, dict]:
    config = get_config(mkey)
    base_market = mkey.replace("_5m", "")
    is_5m = "_5m" in mkey

    signal_kw: dict = {}
    if base_market == "btc":
        signal_kw = dict(
            max_z=3.0 if not is_5m else 1.0,
            reversion_discount=0.0,
        )
    elif base_market == "eth":
        signal_kw = dict(
            max_z=3.0 if not is_5m else 0.7,
            edge_threshold=0.12,
            reversion_discount=0.20 if not is_5m else 0.0,
            momentum_lookback_s=15,
            momentum_majority=0.7,
            spread_edge_penalty=0.2,
        )
    elif base_market in ("sol", "xrp"):
        signal_kw = dict(
            edge_threshold=0.12,
            reversion_discount=0.0,
            momentum_lookback_s=15,
            momentum_majority=0.7,
            spread_edge_penalty=0.2,
        )

    signal_kw["window_duration"] = config.window_duration_s
    signal_kw["maker_mode"] = True
    signal_kw["max_bet_fraction"] = float(settings["max_bet_fraction"])
    signal_kw["kelly_fraction"] = float(settings["kelly_fraction"])
    signal_kw["edge_threshold"] = (
        float(settings["edge_threshold"])
        if settings.get("edge_threshold") is not None
        else signal_kw.get("edge_threshold", 0.12)
    )
    signal_kw["momentum_majority"] = 0.0
    signal_kw["spread_edge_penalty"] = signal_kw.get("spread_edge_penalty", 0.0)
    signal_kw["max_sigma"] = config.max_sigma
    signal_kw["min_sigma"] = config.min_sigma
    signal_kw["toxicity_threshold"] = 0.75
    signal_kw["toxicity_edge_mult"] = 1.5
    signal_kw["down_edge_bonus"] = 0.05
    signal_kw["regime_z_scale"] = bool(settings["regime_z_scale"])
    signal_kw["vpin_threshold"] = 0.95
    signal_kw["vpin_edge_mult"] = 1.5
    signal_kw["vpin_window"] = 20
    signal_kw["vpin_bar_s"] = 60.0
    signal_kw["oracle_lag_threshold"] = 0.002
    # The dashboard handles oracle lag as a paper-trade confirmation gate,
    # so the model itself should not also widen thresholds for lag.
    signal_kw["oracle_lag_mult"] = float(settings.get("oracle_lag_mult", 0.0))
    signal_kw["obi_weight"] = 0.03
    signal_kw["tail_mode"] = config.tail_mode
    signal_kw["tail_nu_default"] = config.tail_nu_default
    signal_kw["kou_lambda"] = config.kou_lambda
    signal_kw["kou_p_up"] = config.kou_p_up
    signal_kw["kou_eta1"] = config.kou_eta1
    signal_kw["kou_eta2"] = config.kou_eta2
    signal_kw["market_blend"] = config.market_blend
    signal_kw["max_book_age_ms"] = config.max_book_age_ms
    signal_kw["slippage"] = float(settings["slippage"])
    signal_kw["inventory_skew"] = 0.02
    signal_kw["maker_warmup_s"] = 30.0 if is_5m else 200.0
    signal_kw["maker_withdraw_s"] = 20.0 if is_5m else 60.0
    signal_kw["as_mode"] = False
    signal_kw["gamma_inv"] = 0.15
    signal_kw["gamma_spread"] = 2.0 if (base_market == "btc" and is_5m) else 1.5
    signal_kw["min_edge"] = 0.05
    signal_kw["tox_spread"] = 0.05
    signal_kw["vpin_spread"] = 0.05
    signal_kw["lag_spread"] = 0.08
    signal_kw["edge_step"] = 0.01
    signal_kw["contract_vol_lookback_s"] = 60
    signal_kw["min_entry_z"] = float(settings.get("min_entry_z", 0.0)) or config.min_entry_z
    signal_kw["min_entry_price"] = float(settings.get("min_entry_price", 0.0)) or config.min_entry_price

    if base_market == "btc":
        signal_kw["vamp_mode"] = "filter"
    elif base_market in ("eth", "sol", "xrp"):
        signal_kw["vamp_mode"] = "filter"
        signal_kw["vamp_filter_threshold"] = 0.07

    meta = {
        "calibrated": False,
        "config_name": config.display_name,
        "signal_edge_threshold": signal_kw["edge_threshold"],
        "maker_warmup_s": signal_kw["maker_warmup_s"],
        "maker_withdraw_s": signal_kw["maker_withdraw_s"],
        "window_duration": signal_kw["window_duration"],
    }

    if settings.get("calibrated"):
        cal_dir = DATA_DIR / config.data_subdir
        cal_table = build_calibration_table(cal_dir, vol_lookback_s=90)
        signal_kw["calibration_table"] = cal_table
        signal_kw["cal_prior_strength"] = 50.0
        signal_kw["cal_max_weight"] = 0.70
        meta["calibrated"] = True
        if settings.get("regime_z_scale"):
            sigma_cal = _build_sigma_calibration(config.data_subdir)
            if sigma_cal is not None:
                signal_kw["sigma_calibration"] = sigma_cal
                meta["sigma_calibration"] = sigma_cal

    return signal_kw, meta


def _snapshot_from_payload(payload: dict) -> Snapshot:
    def _levels(name: str) -> tuple[tuple[float, float], ...]:
        return tuple((float(px), float(sz)) for px, sz in payload.get(name, []))

    return Snapshot(
        ts_ms=int(payload["ts_ms"]),
        market_slug=str(payload["market_slug"]),
        time_remaining_s=float(payload["time_remaining_s"]),
        chainlink_price=float(payload["chainlink_price"]),
        window_start_price=float(payload["window_start_price"]),
        best_bid_up=float(payload["best_bid_up"]) if payload.get("best_bid_up") is not None else None,
        best_ask_up=float(payload["best_ask_up"]) if payload.get("best_ask_up") is not None else None,
        best_bid_down=float(payload["best_bid_down"]) if payload.get("best_bid_down") is not None else None,
        best_ask_down=float(payload["best_ask_down"]) if payload.get("best_ask_down") is not None else None,
        size_bid_up=float(payload["size_bid_up"]) if payload.get("size_bid_up") is not None else None,
        size_ask_up=float(payload["size_ask_up"]) if payload.get("size_ask_up") is not None else None,
        size_bid_down=float(payload["size_bid_down"]) if payload.get("size_bid_down") is not None else None,
        size_ask_down=float(payload["size_ask_down"]) if payload.get("size_ask_down") is not None else None,
        ask_levels_up=_levels("ask_levels_up"),
        ask_levels_down=_levels("ask_levels_down"),
        bid_levels_up=_levels("bid_levels_up"),
        bid_levels_down=_levels("bid_levels_down"),
    )


runtimes: dict[str, dict] = {}


def _ensure_runtime(mkey: str, settings: dict) -> dict:
    runtime = runtimes.get(mkey)
    if runtime is not None and runtime.get("settings") == settings:
        return runtime

    signal_kw, meta = _signal_kwargs(mkey, settings)
    runtime = {
        "signal": DiffusionSignal(bankroll=0.0, **signal_kw),
        "ctx": {},
        "meta": meta,
        "settings": dict(settings),
        "window_start_ms": None,
    }
    runtimes[mkey] = runtime
    return runtime


def _handle_evaluate(req: dict) -> dict:
    mkey = req["market_key"]
    settings = req["settings"]
    runtime = _ensure_runtime(mkey, settings)
    signal = runtime["signal"]
    ctx = runtime["ctx"]

    window_start_ms = req.get("window_start_ms")
    if runtime.get("window_start_ms") != window_start_ms:
        ctx = {}
        ctx["_window_start_ms"] = window_start_ms
        runtime["ctx"] = ctx
        runtime["window_start_ms"] = window_start_ms

    signal.bankroll = float(req["bankroll"])

    binance_mid = req.get("binance_mid")
    if binance_mid is not None:
        ctx["_binance_mid"] = float(binance_mid)
    else:
        ctx.pop("_binance_mid", None)
    ctx["_trade_bars"] = req.get("trade_bars", [])
    ctx["_trade_total_bars"] = req.get("trade_total_bars", 0)
    ctx["_trade_side_history"] = req.get("trade_sides", [])

    snap = _snapshot_from_payload(req["snapshot"])
    up_dec, down_dec = signal.decide_both_sides(snap, ctx)

    return {
        "ok": True,
        "meta": runtime["meta"],
        "signal": {
            "sigma_per_s": ctx.get("_sigma_per_s"),
            "z": ctx.get("_z"),
            "z_raw": ctx.get("_z_raw"),
            "p_display": ctx.get("_p_display"),
            "p_model_raw": ctx.get("_p_model_raw"),
            "p_model_trade": ctx.get("_p_model_trade"),
            "edge_up": ctx.get("_edge_up"),
            "edge_down": ctx.get("_edge_down"),
            "dyn_threshold_up": ctx.get("_dyn_threshold_up"),
            "dyn_threshold_down": ctx.get("_dyn_threshold_down"),
            "toxicity": ctx.get("_toxicity"),
            "vpin": ctx.get("_vpin"),
            "oracle_lag": ctx.get("_oracle_lag"),
            "regime_z_factor": ctx.get("_regime_z_factor"),
            "up_dec": {
                "action": up_dec.action,
                "edge": up_dec.edge,
                "size_usd": up_dec.size_usd,
                "reason": up_dec.reason,
            },
            "down_dec": {
                "action": down_dec.action,
                "edge": down_dec.edge,
                "size_usd": down_dec.size_usd,
                "reason": down_dec.reason,
            },
        },
    }


def main():
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
            cmd = req.get("cmd")
            if cmd == "evaluate":
                resp = _handle_evaluate(req)
            else:
                resp = {"ok": False, "error": f"unknown command: {cmd}"}
        except Exception as exc:
            resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
