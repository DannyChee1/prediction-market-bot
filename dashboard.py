#!/usr/bin/env python3
"""
Live Dashboard: BTC Polymarket Up/Down Windows

Displays active BTC 5m and 15m windows with:
  - Polymarket link, start/end price, resolution
  - Live DiffusionSignal state + one-shot paper trade log
  - Precise oracle lag tracker (Binance mid vs Chainlink)

Usage:
    python3 dashboard.py
    python3 dashboard.py --bankroll 500 --edge-threshold 0.03
    python3 dashboard.py --calibrated --regime-z-scale
    python3 dashboard.py --port 8080

Open: http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import ssl
import time as _time
import warnings
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import requests
import uvicorn
import websockets
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from market_api import find_market
from market_config import get_config
from dashboard_signal_worker import _handle_evaluate

warnings.filterwarnings("ignore", category=UserWarning)

# ── Rust feeds (optional — use if polybot_core is compiled) ──────────────────
try:
    import polybot_core as _pbc
    _RUST = True
except ImportError:
    _pbc = None  # type: ignore
    _RUST = False

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    SSL_CTX.check_hostname = False
    SSL_CTX.verify_mode = ssl.CERT_NONE

GAMMA_API    = "https://gamma-api.polymarket.com"
CLOB_WS      = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
RTDS_WS      = "wss://ws-live-data.polymarket.com"
POLY_URL     = "https://polymarket.com/event"
BINANCE_WS   = "wss://data-stream.binance.vision/ws/btcusdt@bookTicker"

CHAINLINK_TRIGGER_FRAC = 0.005   # 0.5% deviation triggers oracle update
CHAINLINK_HEARTBEAT_S  = 3600    # hourly fallback update
LAG_HIST_MAXLEN        = 240     # 4 min of per-second lag readings
LAG_RAW_MAXLEN         = 512     # recent raw lag samples for velocity estimation
LAG_VELOCITY_LOOKBACK_S = 3.0
LAG_ACCURACY_SAMPLE_GAP_S = 2.0
LAG_ACCURACY_HORIZON_S = 10.0
PAPER_LOG_MAXLEN       = 512
STALE_WINDOW_GRACE_S   = 15.0

PAPER_TRADES_LOG = "dashboard_paper_trades.jsonl"
LAG_LOG          = "dashboard_lag_history.jsonl"

# ── Shared live state ─────────────────────────────────────────────────────────

oracle: dict = {
    "binance_bid":        None,   # float
    "binance_ask":        None,
    "binance_mid":        None,
    "binance_ts_ms":      None,
    "chainlink_price":    None,
    "chainlink_ts_ms":    None,   # timestamp of RTDS update message
    "last_cl_changed_ts": None,   # wall-clock time (s) when CL price last CHANGED
    "last_cl_price_seen": None,   # last distinct CL price (to detect changes)
    "cl_change_history":  collections.deque(maxlen=512),
    "chainlink_history":  collections.deque(maxlen=4096),
    # deque of (wall_clock_s, lag_frac) — lag = (binance_mid - cl) / cl
    "lag_history":        collections.deque(maxlen=LAG_HIST_MAXLEN),
    "lag_recent":         collections.deque(maxlen=LAG_RAW_MAXLEN),
    "_last_lag_hist_ts":  0.0,
    # Lag velocity: slope of lag over last N readings (frac/s, positive = lag growing)
    "lag_velocity":       None,
    # Live accuracy: ring-buffer of (ts, lag_frac, cl_price) for checking later.
    # We resolve each entry after LAG_ACCURACY_HORIZON_S.
    "_pending_checks":    collections.deque(maxlen=1000),
    "_last_accuracy_sample_ts": 0.0,
    # Accuracy buckets keyed by |lag| threshold fraction.
    "accuracy":           {
        0.0002: {"n": 0, "moved": 0, "correct": 0},   # >=0.02%
        0.0005: {"n": 0, "moved": 0, "correct": 0},   # >=0.05%
        0.0010: {"n": 0, "moved": 0, "correct": 0},   # >=0.10%
        0.0020: {"n": 0, "moved": 0, "correct": 0},   # >=0.20%
        0.0030: {"n": 0, "moved": 0, "correct": 0},   # >=0.30%
    },
    "rust_feeds": False,  # whether Rust polybot_core feeds are actively in use
    "rust_feed_mode": None,
    "_lag_pending": [],   # accumulated lag samples pending JSONL flush
}

# Rust feed handles (set at startup if _RUST)
_rust_price_feed   = None
_rust_binance_feed = None

# market_key -> per-window state dict
markets: dict[str, dict] = {}
market_runtimes: dict[str, dict] = {}
paper_trade_log = collections.deque(maxlen=PAPER_LOG_MAXLEN)

# CLI config (set by main)
cfg: dict = {
    "bankroll":          1000.0,
    "edge_threshold":    None,
    "kelly_fraction":    0.25,
    "max_bet_fraction":  0.05,
    "paper_edge_buffer_5m": 0.0,
    "paper_min_confidence_15m": 0.75,
    "paper_lag_threshold_15m": 0.0002,
    "paper_lag_edge_bonus_15m": 0.01,
    "paper_require_growing_lag_15m": True,
    "slippage":          0.0,
    "calibrated":        False,
    "regime_z_scale":    False,
}


# ── Math helpers ──────────────────────────────────────────────────────────────

def _lag_pct(binance_mid: Optional[float], cl: Optional[float]) -> Optional[float]:
    if not binance_mid or not cl or cl == 0:
        return None
    return (binance_mid - cl) / cl


def _paper_lag_check(
    side: str,
    signed_lag: Optional[float],
    lag_velocity: Optional[float],
    lag_threshold: float,
) -> dict:
    if signed_lag is None:
        return {
            "active": False,
            "aligned": None,
            "growing": None,
            "confirmed": False,
            "label": "no lag",
        }

    active = abs(signed_lag) >= lag_threshold if lag_threshold > 0 else signed_lag != 0.0
    side_sign = 1.0 if side == "UP" else -1.0
    aligned = (side_sign * signed_lag) > 0 if active else None
    growing = None
    if active and lag_velocity is not None:
        growing = (lag_velocity * signed_lag) > 0

    label = f"{signed_lag * 100:+.3f}%"
    return {
        "active": active,
        "aligned": aligned,
        "growing": growing,
        "confirmed": bool(active and aligned and growing),
        "label": label,
    }


def _write_paper_trade_jsonl(record: dict) -> None:
    """Append one resolved-window record to the paper trades log."""
    try:
        with open(PAPER_TRADES_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def _flush_lag_to_disk() -> None:
    """Write accumulated lag samples to JSONL and clear the pending buffer."""
    pending = oracle["_lag_pending"]
    if not pending:
        return
    try:
        with open(LAG_LOG, "a") as f:
            for rec in pending:
                f.write(json.dumps(rec) + "\n")
    except Exception:
        pass
    oracle["_lag_pending"] = []


def _block_15m_ms(ts_ms: Optional[int]) -> Optional[int]:
    if not ts_ms:
        return None
    return (ts_ms // 900_000) * 900_000


def _record_chainlink_history(price: Optional[float], ts_ms: Optional[int]):
    if price is None:
        return
    if ts_ms is None:
        ts_ms = int(_time.time() * 1000)
    hist = oracle["chainlink_history"]
    if hist and hist[-1][0] == ts_ms and hist[-1][1] == price:
        return
    if hist and hist[-1][1] == price and abs(hist[-1][0] - ts_ms) < 250:
        return
    hist.append((int(ts_ms), float(price)))


def _find_chainlink_price_near(ts_ms: Optional[int], tolerance_ms: int = 30_000) -> Optional[float]:
    if ts_ms is None:
        return None
    hist = oracle["chainlink_history"]
    if not hist:
        return None
    best_ts, best_px = min(hist, key=lambda item: abs(item[0] - ts_ms))
    if abs(best_ts - ts_ms) > tolerance_ms:
        return None
    return float(best_px)


def _finalize_market_resolution(m: Optional[dict], force: bool = False) -> bool:
    if not m or m.get("resolved") is not None:
        return bool(m and m.get("resolved") is not None)

    meta = m.get("meta", {})
    now_ms = int(_time.time() * 1000)
    end_ms = meta.get("end_ms")
    if end_ms is None:
        return False
    if not force and now_ms < int(end_ms):
        return False

    start_price = meta.get("start_price")
    if start_price is None:
        start_price = _find_chainlink_price_near(meta.get("start_ms"))
        if start_price is not None:
            meta["start_price"] = start_price
    if start_price is None:
        return False

    end_price = meta.get("resolved_end_price")
    if end_price is None:
        end_price = _find_chainlink_price_near(end_ms, tolerance_ms=45_000)
        if end_price is None:
            # Fallback: use current oracle price only if it was received near window end
            cl_ts_ms = oracle.get("chainlink_ts_ms")
            if cl_ts_ms is not None and end_ms is not None and abs(cl_ts_ms - end_ms) <= 45_000:
                end_price = oracle.get("chainlink_price")
        if end_price is None:
            return False
        meta["resolved_end_price"] = float(end_price)

    m["resolved"] = "UP" if float(end_price) >= float(start_price) else "DOWN"
    pt = m.get("paper_trade")
    if pt is not None and "final_pnl" not in pt:
        won = (pt["side"] == m["resolved"])
        cost = float(pt.get("cost_usd", 0.0))
        shares = float(pt.get("shares", 0.0))
        payout = shares * (1.0 if won else 0.0)
        pt["payout"] = round(payout, 2)
        pt["won"] = won
        pt["final_pnl"] = round(payout - cost, 2)
        pt["final_pnl_pct"] = round((payout - cost) / cost * 100, 1) if cost > 0 else 0
    return True


def _paper_realized_pnl() -> float:
    return round(
        sum(row["pnl_usd"] for row in paper_trade_log if row.get("traded")),
        2,
    )


def _paper_open_cost() -> float:
    total = 0.0
    for m in markets.values():
        pt = m.get("paper_trade")
        if pt and "final_pnl" not in pt:
            total += float(pt.get("cost_usd", 0.0))
    return round(total, 2)


def _paper_open_mark_value() -> float:
    total = 0.0
    for m in markets.values():
        pt = m.get("paper_trade")
        if not pt or "final_pnl" in pt:
            continue
        sig = m.get("signal", {})
        side = pt.get("side")
        shares = float(pt.get("shares", 0.0))
        if shares <= 0:
            continue
        mark = None
        if side == "UP":
            bid = _best_bid(m["book"]["up_bids"])
            ask = _best_ask(m["book"]["up_asks"])
            if bid is not None and ask is not None:
                mark = (bid + ask) / 2.0
        elif side == "DOWN":
            bid = _best_bid(m["book"]["down_bids"])
            ask = _best_ask(m["book"]["down_asks"])
            if bid is not None and ask is not None:
                mark = (bid + ask) / 2.0
        if mark is None and side == "UP":
            mark = sig.get("up_bid")
        if mark is None and side == "DOWN":
            mark = sig.get("down_bid")
        if mark is not None:
            total += shares * mark
    return round(total, 2)


def _paper_available_cash() -> float:
    cash = cfg["bankroll"] + _paper_realized_pnl() - _paper_open_cost()
    return round(max(0.0, cash), 2)


def _best_levels(levels: dict, reverse: bool) -> tuple[tuple[float, float], ...]:
    if not levels:
        return ()
    ordered = sorted(levels.items(), key=lambda item: item[0], reverse=reverse)
    return tuple((float(px), float(sz)) for px, sz in ordered[:5])


async def _cancel_task(task: Optional[asyncio.Task]) -> None:
    """Cancel a background task and swallow cancellation cleanly."""
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


def _make_rust_binance_feed(symbol: str):
    mode = (os.getenv("BINANCE_FEED_MODE", "json") or "json").strip().lower()
    api_key = (os.getenv("BINANCE_SBE_API_KEY", "") or "").strip()
    if mode == "sbe":
        if not api_key:
            raise RuntimeError("BINANCE_SBE_API_KEY is required when BINANCE_FEED_MODE=sbe")
        try:
            return _pbc.BinanceFeed(symbol, mode, api_key)
        except TypeError as exc:
            raise RuntimeError(
                "installed polybot_core extension is JSON-only; rebuild it before using BINANCE_FEED_MODE=sbe"
            ) from exc
    return _pbc.BinanceFeed(symbol)


def _fresh_binance_mid() -> Optional[float]:
    mid = oracle.get("binance_mid")
    ts_ms = oracle.get("binance_ts_ms")
    if mid is None:
        return None
    if ts_ms is None:
        return float(mid)
    age_ms = int(_time.time() * 1000) - int(ts_ms)
    if age_ms > 10_000:
        return None
    return float(mid)


def poly_fee(p: float, maker: bool = False) -> float:
    if maker:
        return 0.0
    return 0.02 * p * (1.0 - p)


def walk_book(
    ask_levels: tuple[tuple[float, float], ...],
    desired_shares: float,
    slippage: float,
) -> tuple[float, float, float]:
    if not ask_levels or desired_shares <= 0:
        return (0.0, 0.0, 0.0)

    filled = 0.0
    total_cost = 0.0
    for price, size in ask_levels:
        if filled >= desired_shares:
            break
        take = min(size, desired_shares - filled)
        total_cost += take * (price + poly_fee(price) + slippage)
        filled += take
    if filled <= 0:
        return (0.0, 0.0, 0.0)
    return (filled, total_cost, total_cost / filled)




def _signal_worker_settings() -> dict:
    return {
        "edge_threshold": cfg["edge_threshold"],
        "kelly_fraction": cfg["kelly_fraction"],
        "max_bet_fraction": cfg["max_bet_fraction"],
        "oracle_lag_mult": 0.0,
        "slippage": cfg["slippage"],
        "calibrated": cfg["calibrated"],
        "regime_z_scale": cfg["regime_z_scale"],
        "min_entry_z": cfg.get("min_entry_z", 0.0),
        "min_entry_price": cfg.get("min_entry_price", 0.10),
    }


def _get_market_runtime(mkey: str) -> dict:
    runtime = market_runtimes.get(mkey)
    if runtime is not None:
        return runtime

    runtime = {
        "trade_state": {
            "bars": collections.deque(maxlen=200),
            "sides": collections.deque(maxlen=300),
            "current_bar": {"buy_vol": 0.0, "sell_vol": 0.0, "start_ts": 0.0},
            "bar_duration_s": 60.0,
            "total_bars": 0,
        },
    }
    market_runtimes[mkey] = runtime
    return runtime


def _snapshot_from_dashboard_market(mkey: str, market: dict) -> Optional[dict]:
    meta = market["meta"]
    start_price = meta.get("start_price")
    if start_price is None:
        start_price = _find_chainlink_price_near(meta.get("start_ms"))
        if start_price is not None:
            meta["start_price"] = start_price

    chainlink_price = oracle.get("chainlink_price")
    if chainlink_price is None or start_price is None:
        return None

    up_bid = _best_bid(market["book"]["up_bids"])
    up_ask = _best_ask(market["book"]["up_asks"])
    down_bid = _best_bid(market["book"]["down_bids"])
    down_ask = _best_ask(market["book"]["down_asks"])

    return {
        "ts_ms": int(_time.time() * 1000),
        "market_slug": meta["slug"],
        "time_remaining_s": float(meta.get("tau", 0.0)),
        "chainlink_price": float(chainlink_price),
        "window_start_price": float(start_price),
        "best_bid_up": up_bid,
        "best_ask_up": up_ask,
        "best_bid_down": down_bid,
        "best_ask_down": down_ask,
        "size_bid_up": market["book"]["up_bids"].get(up_bid) if up_bid is not None else None,
        "size_ask_up": market["book"]["up_asks"].get(up_ask) if up_ask is not None else None,
        "size_bid_down": market["book"]["down_bids"].get(down_bid) if down_bid is not None else None,
        "size_ask_down": market["book"]["down_asks"].get(down_ask) if down_ask is not None else None,
        "ask_levels_up": _best_levels(market["book"]["up_asks"], reverse=False),
        "ask_levels_down": _best_levels(market["book"]["down_asks"], reverse=False),
        "bid_levels_up": _best_levels(market["book"]["up_bids"], reverse=True),
        "bid_levels_down": _best_levels(market["book"]["down_bids"], reverse=True),
    }


def _record_resolved_window(mkey: str, force: bool = False):
    m = markets.get(mkey)
    if not m or m.get("history_logged"):
        return
    if not _finalize_market_resolution(m, force=force):
        return

    meta = m["meta"]
    pt = m.get("paper_trade")
    sig = m.get("signal", {})
    start_ms = meta.get("start_ms")
    end_ms = meta.get("end_ms")
    start_price = meta.get("start_price")
    end_price = meta.get("resolved_end_price")
    if end_price is None:
        end_price = sig.get("cl") or oracle.get("chainlink_price")

    record = {
        "logged_at_ms": int(_time.time() * 1000),
        "start_ms": start_ms,
        "end_ms": end_ms,
        "block_15m_ms": _block_15m_ms(start_ms),
        "timeframe": meta.get("timeframe"),
        "slug": meta.get("slug"),
        "url": meta.get("url"),
        "resolved": m.get("resolved"),
        "start_price": round(float(start_price), 2) if start_price else None,
        "end_price": round(float(end_price), 2) if end_price else None,
        "traded": bool(pt),
        "trade_side": pt["side"] if pt else None,
        "bet_usd": round(float(pt.get("cost_usd", 0.0)), 2) if pt else 0.0,
        "entry_price": round(float(pt["entry_price"]), 4) if pt else None,
        "entry_quote": round(float(pt.get("entry_quote", 0.0)), 4) if pt and pt.get("entry_quote") is not None else None,
        "shares": round(float(pt.get("shares", 0.0)), 4) if pt else 0.0,
        "payout_usd": round(float(pt.get("payout", 0.0)), 2) if pt else 0.0,
        "pnl_usd": round(float(pt.get("final_pnl", 0.0)), 2) if pt else 0.0,
        "pnl_pct": round(float(pt.get("final_pnl_pct", 0.0)), 1) if pt else 0.0,
        "won": pt.get("won") if pt else None,
        # Signal diagnostics at entry (for Brier score / edge-bucket analysis)
        "p_at_entry": round(float(pt["p_at_entry"]), 4) if pt and pt.get("p_at_entry") is not None else None,
        "z_at_entry": round(float(pt["z_at_entry"]), 4) if pt and pt.get("z_at_entry") is not None else None,
        "edge_at_entry": round(float(pt["edge_at_entry"]), 4) if pt and pt.get("edge_at_entry") is not None else None,
        "threshold_at_entry": round(float(pt["threshold_at_entry"]), 4) if pt and pt.get("threshold_at_entry") is not None else None,
        "confidence_at_entry": round(float(pt["confidence_at_entry"]), 4) if pt and pt.get("confidence_at_entry") is not None else None,
        "confidence_target": round(float(pt["confidence_target"]), 4) if pt and pt.get("confidence_target") is not None else None,
        "oracle_lag_signed_at_entry": round(float(pt["oracle_lag_signed_at_entry"]), 6) if pt and pt.get("oracle_lag_signed_at_entry") is not None else None,
        "oracle_lag_abs_at_entry": round(float(pt["oracle_lag_abs_at_entry"]), 6) if pt and pt.get("oracle_lag_abs_at_entry") is not None else None,
        "oracle_lag_velocity_at_entry": round(float(pt["oracle_lag_velocity_at_entry"]), 8) if pt and pt.get("oracle_lag_velocity_at_entry") is not None else None,
        "oracle_lag_active_at_entry": pt.get("oracle_lag_active_at_entry") if pt else None,
        "oracle_lag_aligned_at_entry": pt.get("oracle_lag_aligned_at_entry") if pt else None,
        "oracle_lag_growing_at_entry": pt.get("oracle_lag_growing_at_entry") if pt else None,
        "oracle_lag_confirmed_at_entry": pt.get("oracle_lag_confirmed_at_entry") if pt else None,
        "oracle_lag_bonus_at_entry": round(float(pt["oracle_lag_bonus_at_entry"]), 4) if pt and pt.get("oracle_lag_bonus_at_entry") is not None else None,
    }
    paper_trade_log.append(record)
    _write_paper_trade_jsonl(record)
    m["history_logged"] = True


# ── Oracle lag updater (runs every tick) ─────────────────────────────────────

def _update_lag_history():
    cl  = oracle["chainlink_price"]
    now = _time.time()

    # Skip lag computation if Binance feed is stale (>60s without update).
    b_ts_ms = oracle.get("binance_ts_ms")
    if b_ts_ms is not None and (now - b_ts_ms / 1000) > 60:
        oracle["lag_velocity"] = None
        return

    lag = _lag_pct(oracle["binance_mid"], cl)

    if lag is not None:
        oracle["lag_recent"].append((now, lag))

        last_hist_ts = oracle["_last_lag_hist_ts"]
        if now - last_hist_ts >= 1.0 or not oracle["lag_history"]:
            oracle["lag_history"].append((now, lag))
            oracle["_last_lag_hist_ts"] = now
            oracle["_lag_pending"].append({
                "ts_s": round(now, 3),
                "lag_pct": round(lag * 100, 6),
                "binance_mid": oracle.get("binance_mid"),
                "chainlink_price": oracle.get("chainlink_price"),
            })

        # Lag velocity: linear regression slope over the last few seconds.
        recent = [
            h for h in oracle["lag_recent"]
            if now - h[0] <= LAG_VELOCITY_LOOKBACK_S
        ]
        if len(recent) >= 4 and recent[-1][0] - recent[0][0] >= 0.75:
            xs = np.array([h[0] for h in recent])
            ys = np.array([h[1] for h in recent])
            xs -= xs[0]
            if xs[-1] > 0:
                oracle["lag_velocity"] = float(np.polyfit(xs, ys, 1)[0])
        else:
            oracle["lag_velocity"] = None

        # Record for accuracy check no faster than every N seconds.
        last_sample_ts = oracle["_last_accuracy_sample_ts"]
        if (
            abs(lag) >= 0.0002
            and cl is not None
            and now - last_sample_ts >= LAG_ACCURACY_SAMPLE_GAP_S
        ):
            oracle["_pending_checks"].append((now, lag, cl))
            oracle["_last_accuracy_sample_ts"] = now
    else:
        oracle["lag_velocity"] = None

    # Resolve pending accuracy checks once they age past the horizon.
    pending = oracle["_pending_checks"]
    while pending and now - pending[0][0] >= LAG_ACCURACY_HORIZON_S:
        ts_then, lag_then, cl_then = pending.popleft()
        horizon_end = ts_then + LAG_ACCURACY_HORIZON_S
        first_move = next(
            (
                (chg_ts, chg_price)
                for chg_ts, chg_price in oracle["cl_change_history"]
                if ts_then < chg_ts <= horizon_end
            ),
            None,
        )
        moved = first_move is not None
        predicted_up = lag_then > 0
        correct = bool(
            moved
            and cl_then is not None
            and ((first_move[1] > cl_then) == predicted_up)
        )
        for thresh, bucket in oracle["accuracy"].items():
            if abs(lag_then) >= thresh:
                bucket["n"] += 1
                bucket["moved"] += int(moved)
                bucket["correct"] += int(correct)

    # Detect distinct Chainlink price change
    if cl is not None and cl != oracle["last_cl_price_seen"]:
        oracle["last_cl_changed_ts"] = now
        oracle["last_cl_price_seen"] = cl
        oracle["cl_change_history"].append((now, cl))


# ── Binance bookTicker WebSocket ──────────────────────────────────────────────

async def _rust_feed_poll_loop(cancel: asyncio.Event):
    """Poll Rust atomic feeds every 50ms (20Hz). Much lower latency than Python WS."""
    global _rust_price_feed, _rust_binance_feed
    if not _RUST:
        return
    cl_symbol = get_config("btc").chainlink_symbol
    binance_symbol = get_config("btc").binance_symbol or "btcusdt"
    _rust_price_feed = _pbc.PriceFeed(cl_symbol)
    _rust_binance_feed = _make_rust_binance_feed(binance_symbol)
    oracle["rust_feeds"] = True
    oracle["rust_feed_mode"] = (os.getenv("BINANCE_FEED_MODE", "json") or "json").strip().lower()
    await asyncio.sleep(1.0)   # let feeds connect

    # Track last wall-clock time we received a NEW Chainlink price (distinct cl_ts).
    # Used to detect silent subscription drops where PING/PONG keeps TCP alive but
    # no price data arrives (the 30s read timeout in Rust never fires in that case).
    _CL_STALE_TIMEOUT_S = 90.0
    _last_cl_ts_seen: float = 0.0
    _last_cl_received_wall: float = _time.monotonic()

    while not cancel.is_set():
        await asyncio.sleep(0.05)
        cl = _rust_price_feed.price()
        cl_ts = _rust_price_feed.last_update_ts()
        mid = _rust_binance_feed.mid()
        mid_ts = _rust_binance_feed.last_update_ts()

        if cl is not None:
            # Detect whether the Rust feed actually received a new update.
            if cl_ts != _last_cl_ts_seen:
                _last_cl_ts_seen = cl_ts
                _last_cl_received_wall = _time.monotonic()

            # Auto-reconnect if subscription silently dropped (PING keeps socket
            # alive but no price updates flow). Chainlink updates at most every
            # ~30s on deviation; if nothing for 90s the feed is stale.
            stale_s = _time.monotonic() - _last_cl_received_wall
            if stale_s > _CL_STALE_TIMEOUT_S:
                print(f"  [PriceFeed] no CL update for {stale_s:.0f}s — reconnecting")
                _rust_price_feed = _pbc.PriceFeed(cl_symbol)
                _last_cl_received_wall = _time.monotonic()
                await asyncio.sleep(1.0)
                continue

            oracle["chainlink_price"] = cl
            if cl_ts > 0:
                oracle["chainlink_ts_ms"] = int(cl_ts * 1000)
            _record_chainlink_history(oracle["chainlink_price"], oracle.get("chainlink_ts_ms"))

        if mid is not None:
            oracle["binance_mid"] = mid
            if mid_ts > 0:
                oracle["binance_ts_ms"] = int(mid_ts * 1000)
            # Derive approximate bid/ask from mid (Rust only exposes mid)
            oracle["binance_bid"] = mid
            oracle["binance_ask"] = mid
        if cl is not None or mid is not None:
            _update_lag_history()


async def _binance_bookticker_ws(cancel: asyncio.Event):
    """Stream BTCUSDT best bid/ask from Binance; update oracle mid in real time.
    Skipped when Rust feeds are active (they handle this faster)."""
    if _RUST:
        return
    backoff = 2
    while not cancel.is_set():
        try:
            async with websockets.connect(
                BINANCE_WS, ssl=SSL_CTX, ping_interval=20
            ) as ws:
                backoff = 2
                async for raw in ws:
                    if cancel.is_set():
                        break
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    bid = float(msg.get("b", 0))
                    ask = float(msg.get("a", 0))
                    if bid > 0 and ask > 0:
                        oracle["binance_bid"]   = bid
                        oracle["binance_ask"]   = ask
                        oracle["binance_mid"]   = (bid + ask) / 2.0
                        oracle["binance_ts_ms"] = int(_time.time() * 1000)
                        _update_lag_history()
        except Exception:
            if cancel.is_set():
                return
            await asyncio.sleep(min(backoff, 30))
            backoff = min(backoff * 2, 60)


# ── Chainlink RTDS WebSocket ──────────────────────────────────────────────────

async def _chainlink_rtds_ws(cancel: asyncio.Event):
    """Stream Chainlink BTC/USD price from Polymarket RTDS.
    Skipped when Rust PriceFeed is active."""
    if _RUST:
        return
    backoff = 2
    while not cancel.is_set():
        try:
            async with websockets.connect(
                RTDS_WS, ssl=SSL_CTX, ping_interval=None
            ) as ws:
                backoff = 2
                await ws.send(json.dumps({
                    "action": "subscribe",
                    "subscriptions": [{
                        "topic": "crypto_prices_chainlink",
                        "type": "*",
                    }],
                }))

                async def _hb():
                    try:
                        while not cancel.is_set():
                            await asyncio.sleep(5)
                            await ws.send("PING")
                    except Exception:
                        pass

                hb = asyncio.create_task(_hb())
                try:
                    async for raw in ws:
                        if cancel.is_set():
                            break
                        if raw in ("PONG", ""):
                            continue
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        payload = msg.get("payload", {})
                        if payload.get("symbol") != "btc/usd":
                            continue

                        data_arr = payload.get("data")
                        if isinstance(data_arr, list):
                            for entry in data_arr:
                                p = entry.get("value")
                                ts = entry.get("timestamp")
                                if p is not None:
                                    oracle["chainlink_price"] = float(p)
                                    oracle["chainlink_ts_ms"] = int(ts) if ts else None
                                    _record_chainlink_history(oracle["chainlink_price"], oracle.get("chainlink_ts_ms"))
                                    _update_lag_history()
                            continue

                        price = payload.get("value")
                        ts    = payload.get("timestamp")
                        if price is not None:
                            oracle["chainlink_price"] = float(price)
                            oracle["chainlink_ts_ms"] = int(ts) if ts else None
                            _record_chainlink_history(oracle["chainlink_price"], oracle.get("chainlink_ts_ms"))
                            _update_lag_history()
                finally:
                    hb.cancel()

        except Exception:
            if cancel.is_set():
                return
            await asyncio.sleep(min(backoff, 30))
            backoff = min(backoff * 2, 60)


# ── Polymarket CLOB WebSocket (per window) ────────────────────────────────────

async def _clob_ws(mkey: str, up_token: str, down_token: str, cancel: asyncio.Event):
    """Subscribe to UP/DOWN book for one window and update markets[mkey]."""
    token_map = {up_token: "up", down_token: "down"}
    backoff = 2

    while not cancel.is_set():
        try:
            async with websockets.connect(
                CLOB_WS, ssl=SSL_CTX, ping_interval=None
            ) as ws:
                backoff = 2
                await ws.send(json.dumps({
                    "assets_ids":            [up_token, down_token],
                    "type":                  "market",
                    "custom_feature_enabled": True,
                }))

                async def _hb():
                    try:
                        while not cancel.is_set():
                            await asyncio.sleep(10)
                            await ws.send("PING")
                    except Exception:
                        pass

                hb = asyncio.create_task(_hb())
                try:
                    async for raw in ws:
                        if cancel.is_set():
                            break
                        if raw in ("PONG", ""):
                            continue
                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        msgs = payload if isinstance(payload, list) else [payload]
                        for msg in msgs:
                            if not isinstance(msg, dict):
                                continue
                            etype    = msg.get("event_type")
                            asset_id = msg.get("asset_id")
                            side     = token_map.get(asset_id)

                            if etype == "book" and side and mkey in markets:
                                bids = {float(l["price"]): float(l["size"])
                                        for l in msg.get("bids", [])}
                                asks = {float(l["price"]): float(l["size"])
                                        for l in msg.get("asks", [])}
                                book = markets[mkey]["book"]
                                book[f"{side}_bids"] = bids
                                book[f"{side}_asks"] = asks

                            elif etype == "price_change" and mkey in markets:
                                for ch in msg.get("price_changes", []):
                                    s = token_map.get(ch.get("asset_id"))
                                    if not s:
                                        continue
                                    p, sz = float(ch["price"]), float(ch["size"])
                                    book = markets[mkey]["book"]
                                    target = book[f"{s}_bids"] if ch["side"] == "BUY" else book[f"{s}_asks"]
                                    if sz == 0:
                                        target.pop(p, None)
                                    else:
                                        target[p] = sz
                            elif etype == "last_trade_price" and mkey in markets:
                                try:
                                    size = float(msg.get("size", 0))
                                    trade_side = msg.get("side", "").upper()
                                    if size > 0 and trade_side in ("BUY", "SELL"):
                                        trade_state = markets[mkey]["trade_state"]
                                        trade_state["sides"].append(trade_side)
                                        bar = trade_state["current_bar"]
                                        now_ts = _time.time()
                                        if bar["start_ts"] == 0:
                                            bar["start_ts"] = now_ts
                                        if trade_side == "BUY":
                                            bar["buy_vol"] += size
                                        else:
                                            bar["sell_vol"] += size
                                        bar_dur = trade_state.get("bar_duration_s", 60.0)
                                        if now_ts - bar["start_ts"] >= bar_dur:
                                            trade_state["bars"].append(
                                                (bar["buy_vol"], bar["sell_vol"])
                                            )
                                            trade_state["total_bars"] = trade_state.get("total_bars", 0) + 1
                                            bar["buy_vol"] = 0.0
                                            bar["sell_vol"] = 0.0
                                            bar["start_ts"] = now_ts
                                except (TypeError, ValueError, KeyError):
                                    pass
                finally:
                    hb.cancel()

        except Exception:
            if cancel.is_set():
                return
            await asyncio.sleep(min(backoff, 30))
            backoff = min(backoff * 2, 60)


# ── Window discovery (per market key) ────────────────────────────────────────

def _find_market_sync(config):
    """Use the same market lookup path as the live trader."""
    return find_market(config)


async def _window_discovery_loop(mkey: str, cancel: asyncio.Event):
    """Continuously discover and manage the active window for one market key."""
    config   = get_config(mkey)
    loop     = asyncio.get_event_loop()
    clob_cancel: Optional[asyncio.Event] = None
    clob_task: Optional[asyncio.Task]    = None
    current_slug: Optional[str]          = None

    while not cancel.is_set():
        event, market = await loop.run_in_executor(
            None, _find_market_sync, config
        )

        if not event or not market:
            if mkey in markets:
                now_ms = int(_time.time() * 1000)
                end_ms = markets[mkey]["meta"]["end_ms"]
                markets[mkey]["meta"]["tau"] = max(0.0, (end_ms - now_ms) / 1000.0)
                _tick_signal(mkey)

                if now_ms >= end_ms + int(STALE_WINDOW_GRACE_S * 1000):
                    _record_resolved_window(mkey)
                    if clob_cancel:
                        clob_cancel.set()
                    await _cancel_task(clob_task)
                    markets.pop(mkey, None)
                    current_slug = None
                    clob_cancel = None
                    clob_task = None
                    await asyncio.sleep(2)
                    continue

                await asyncio.sleep(2)
                continue

            await asyncio.sleep(30)
            continue

        slug = event["slug"]
        if slug == current_slug:
            # Same window still active — just update time
            if mkey in markets:
                end_ms = markets[mkey]["meta"]["end_ms"]
                now_ms = int(_time.time() * 1000)
                markets[mkey]["meta"]["tau"] = max(0.0, (end_ms - now_ms) / 1000.0)
                _tick_signal(mkey)
            await asyncio.sleep(10)
            continue

        # New window detected — cancel old CLOB WS
        if mkey in markets:
            _record_resolved_window(mkey, force=True)
        if clob_cancel:
            clob_cancel.set()
        await _cancel_task(clob_task)

        # Parse tokens
        import json as _json
        outcomes = market["outcomes"]
        tokens   = market["clobTokenIds"]
        if isinstance(outcomes, str):
            outcomes = _json.loads(outcomes)
        if isinstance(tokens, str):
            tokens = _json.loads(tokens)
        outcomes_l = [o.lower() for o in outcomes]
        up_idx   = outcomes_l.index("up")   if "up"  in outcomes_l else 0
        down_idx = outcomes_l.index("down") if "down" in outcomes_l else 1
        up_token   = tokens[up_idx]
        down_token = tokens[down_idx]

        end   = datetime.fromisoformat(market["endDate"].replace("Z", "+00:00"))
        start = datetime.fromisoformat(market["eventStartTime"].replace("Z", "+00:00"))
        end_ms   = int(end.timestamp() * 1000)
        start_ms = int(start.timestamp() * 1000)

        runtime = _get_market_runtime(mkey)
        start_price = (
            _find_chainlink_price_near(start_ms)
            or oracle.get("chainlink_price")
        )

        markets[mkey] = {
            "meta": {
                "slug":        slug,
                "url":         f"{POLY_URL}/{slug}",
                "start_ms":    start_ms,
                "end_ms":      end_ms,
                "start_price": start_price,
                "timeframe":   config.display_name,
                "tau":         max(0.0, (end_ms - int(_time.time() * 1000)) / 1000.0),
            },
            "book": {
                "up_bids":   {},
                "up_asks":   {},
                "down_bids": {},
                "down_asks": {},
            },
            "signal":        {},
            "trade_state":   runtime["trade_state"],
            "price_history": collections.deque(maxlen=900),
            "paper_trade":   None,
            "resolved":      None,
            "history_logged": False,
        }

        current_slug = slug
        clob_cancel  = asyncio.Event()
        clob_task    = asyncio.create_task(
            _clob_ws(mkey, up_token, down_token, clob_cancel)
        )
        print(f"  [Dashboard] {config.display_name}: {slug}")
        await asyncio.sleep(10)


# ── Signal tick ───────────────────────────────────────────────────────────────

def _best_ask(asks: dict) -> Optional[float]:
    return min(asks) if asks else None

def _best_bid(bids: dict) -> Optional[float]:
    return max(bids) if bids else None


def _tick_signal(mkey: str):
    """Evaluate DiffusionSignal and maybe open a one-shot paper trade."""
    m = markets.get(mkey)
    if not m:
        return

    _get_market_runtime(mkey)
    meta = m["meta"]
    cl = oracle["chainlink_price"]
    start = meta.get("start_price")
    tau = meta.get("tau", 0.0)

    if cl is not None:
        now_ms = int(_time.time() * 1000)
        hist = m["price_history"]
        if not hist or hist[-1][1] != cl:
            hist.append((now_ms, cl))
            _record_chainlink_history(cl, oracle.get("chainlink_ts_ms") or now_ms)

    if start is None:
        start = _find_chainlink_price_near(meta.get("start_ms"))
        if start is not None:
            meta["start_price"] = start

    base_signal = {
        "engine":          "DiffusionSignal",
        "calibrated":      bool(cfg["calibrated"]),
        "cl":              cl,
        "start":           start,
        "tau":             tau,
        "sigma":           None,
        "z":               None,
        "z_raw":           None,
        "p_up":            None,
        "p_down":          None,
        "p_trade_up":      None,
        "p_trade_down":    None,
        "up_bid":          _best_bid(m["book"]["up_bids"]),
        "up_ask":          _best_ask(m["book"]["up_asks"]),
        "down_bid":        _best_bid(m["book"]["down_bids"]),
        "down_ask":        _best_ask(m["book"]["down_asks"]),
        "edge_up":         None,
        "edge_down":       None,
        "threshold_up":    None,
        "threshold_down":  None,
        "toxicity":        None,
        "vpin":            None,
        "oracle_lag":      None,
        "oracle_lag_signed": None,
        "oracle_lag_velocity": None,
        "confidence":      None,
        "confidence_target": None,
        "lag_confirms_up": False,
        "lag_confirms_down": False,
        "lag_bonus":       0.0,
        "regime_z_factor": None,
        "up_action":       "FLAT",
        "down_action":     "FLAT",
        "up_reason":       None,
        "down_reason":     None,
        "selected_side":   None,
        "paper_gate":      None,
        "signal_bankroll": None,
    }

    if tau <= 0:
        if start is not None and meta.get("start_price") is None:
            meta["start_price"] = start
        if cl is not None and meta.get("resolved_end_price") is None:
            meta["resolved_end_price"] = cl
        _finalize_market_resolution(m, force=True)
        if m.get("resolved") is None:
            m["signal"] = base_signal
            return
        _record_resolved_window(mkey)
        m["signal"] = base_signal
        return

    snapshot = _snapshot_from_dashboard_market(mkey, m)
    if snapshot is None or snapshot["window_start_price"] <= 0:
        m["signal"] = base_signal
        return

    fresh_mid = _fresh_binance_mid()
    available_cash = _paper_available_cash()
    try:
        worker_resp = _handle_evaluate({
            "market_key": mkey,
            "settings": _signal_worker_settings(),
            "bankroll": available_cash,
            "window_start_ms": meta.get("start_ms"),
            "snapshot": snapshot,
            "binance_mid": fresh_mid,
            "trade_bars": list(m["trade_state"]["bars"]),
            "trade_total_bars": m["trade_state"].get("total_bars", 0),
            "trade_sides": list(m["trade_state"].get("sides", [])),
        })
    except Exception as exc:
        m["signal"] = {**base_signal, "paper_gate": f"signal error: {exc}"}
        return

    signal_meta = worker_resp["meta"]
    signal_state = worker_resp["signal"]
    up_dec = signal_state["up_dec"]
    down_dec = signal_state["down_dec"]
    p_display = signal_state.get("p_display")
    p_trade = signal_state.get("p_model_trade")
    p_raw = signal_state.get("p_model_raw")
    signed_lag = _lag_pct(fresh_mid, cl)
    lag_velocity = oracle.get("lag_velocity")
    lag_threshold_15m = float(cfg["paper_lag_threshold_15m"])
    up_lag = _paper_lag_check("UP", signed_lag, lag_velocity, lag_threshold_15m)
    down_lag = _paper_lag_check("DOWN", signed_lag, lag_velocity, lag_threshold_15m)
    chosen_side = None
    chosen_dec = None
    candidates = []
    if up_dec["action"] != "FLAT":
        candidates.append(("UP", up_dec))
    if down_dec["action"] != "FLAT":
        candidates.append(("DOWN", down_dec))
    if candidates:
        chosen_side, chosen_dec = max(candidates, key=lambda item: item[1]["edge"])

    m["signal"] = {
        **base_signal,
        "calibrated":      bool(signal_meta.get("calibrated")),
        "sigma":           signal_state.get("sigma_per_s"),
        "z":               signal_state.get("z"),
        "z_raw":           signal_state.get("z_raw"),
        "p_up":            p_display,
        "p_down":          (1.0 - p_display) if p_display is not None else None,
        "p_trade_up":      p_trade if p_trade is not None else p_raw,
        "p_trade_down":    (1.0 - p_trade) if p_trade is not None else ((1.0 - p_raw) if p_raw is not None else None),
        "edge_up":         signal_state.get("edge_up"),
        "edge_down":       signal_state.get("edge_down"),
        "threshold_up":    signal_state.get("dyn_threshold_up"),
        "threshold_down":  signal_state.get("dyn_threshold_down"),
        "toxicity":        signal_state.get("toxicity"),
        "vpin":            signal_state.get("vpin"),
        "oracle_lag":      signal_state.get("oracle_lag"),
        "oracle_lag_signed": signed_lag,
        "oracle_lag_velocity": lag_velocity,
        "lag_confirms_up": up_lag["confirmed"],
        "lag_confirms_down": down_lag["confirmed"],
        "regime_z_factor": signal_state.get("regime_z_factor"),
        "up_action":       up_dec["action"],
        "down_action":     down_dec["action"],
        "up_reason":       up_dec["reason"],
        "down_reason":     down_dec["reason"],
        "selected_side":   chosen_side,
        "signal_bankroll": available_cash,
    }

    if m["paper_trade"] is not None or m["resolved"] is not None:
        return

    elapsed = float(signal_meta.get("window_duration", 0.0)) - snapshot["time_remaining_s"]
    maker_warmup_s = float(signal_meta.get("maker_warmup_s", 0.0))
    maker_withdraw_s = float(signal_meta.get("maker_withdraw_s", 0.0))
    if elapsed < maker_warmup_s:
        m["signal"]["paper_gate"] = (
            f"maker warmup ({elapsed:.0f}s < {maker_warmup_s:.0f}s)"
        )
        return
    if snapshot["time_remaining_s"] < maker_withdraw_s:
        m["signal"]["paper_gate"] = (
            f"maker withdraw ({snapshot['time_remaining_s']:.0f}s < {maker_withdraw_s:.0f}s)"
        )
        return
    if chosen_side is None or chosen_dec is None:
        m["signal"]["paper_gate"] = up_dec["reason"] or down_dec["reason"]
        return

    budget_usd = min(float(chosen_dec["size_usd"]), available_cash * 0.995)
    if budget_usd <= 0:
        m["signal"]["paper_gate"] = "no available paper cash"
        return

    ask_levels = snapshot["ask_levels_up"] if chosen_side == "UP" else snapshot["ask_levels_down"]
    entry_quote = snapshot["best_ask_up"] if chosen_side == "UP" else snapshot["best_ask_down"]
    if entry_quote is None:
        m["signal"]["paper_gate"] = "missing ask"
        return
    entry_eff = entry_quote + poly_fee(entry_quote) + cfg["slippage"]
    if entry_eff <= 0 or entry_eff >= 1:
        m["signal"]["paper_gate"] = f"invalid taker price ({entry_eff:.4f})"
        return

    desired_shares = budget_usd / entry_eff
    filled_shares, total_cost, avg_price = walk_book(ask_levels, desired_shares, cfg["slippage"])
    if filled_shares <= 0 or total_cost <= 0 or avg_price <= 0:
        m["signal"]["paper_gate"] = "book walk failed"
        return

    p_at_entry = m["signal"]["p_trade_up"] if chosen_side == "UP" else m["signal"]["p_trade_down"]
    threshold_at_entry = (
        m["signal"]["threshold_up"] if chosen_side == "UP" else m["signal"]["threshold_down"]
    )
    confidence_target = None
    lag_ctx = up_lag if chosen_side == "UP" else down_lag
    lag_edge_bonus = 0.0
    if not mkey.endswith("_5m"):
        confidence_target = float(cfg["paper_min_confidence_15m"])
        m["signal"]["confidence"] = p_at_entry
        m["signal"]["confidence_target"] = confidence_target
        if p_at_entry is None:
            m["signal"]["paper_gate"] = "missing chosen-side confidence"
            return
        if confidence_target > 0 and p_at_entry < confidence_target:
            m["signal"]["paper_gate"] = (
                f"confidence gate ({p_at_entry*100:.1f}% < {confidence_target*100:.1f}%)"
            )
            return

        if lag_ctx["active"]:
            if not lag_ctx["aligned"]:
                m["signal"]["paper_gate"] = (
                    f"oracle lag veto ({lag_ctx['label']} disagrees with {chosen_side})"
                )
                return
            if cfg["paper_require_growing_lag_15m"] and lag_ctx["growing"] is not True:
                if lag_ctx["growing"] is None:
                    growth_label = "velocity unavailable"
                else:
                    growth_label = "lag reverting"
                m["signal"]["paper_gate"] = f"oracle lag growth gate ({growth_label})"
                return
            lag_edge_bonus = float(cfg["paper_lag_edge_bonus_15m"])

    paper_edge_buffer = cfg["paper_edge_buffer_5m"] if mkey.endswith("_5m") else 0.0
    required_edge = max(0.0, (threshold_at_entry or 0.0) + paper_edge_buffer - lag_edge_bonus)
    m["signal"]["lag_bonus"] = lag_edge_bonus
    if chosen_dec["edge"] < required_edge:
        m["signal"]["paper_gate"] = (
            f"edge gate ({chosen_dec['edge']*100:.2f}% < {required_edge*100:.2f}%)"
        )
        return

    m["paper_trade"] = {
        "side":               chosen_side,
        "entry_price":        round(avg_price, 4),
        "entry_quote":        round(entry_quote, 4),
        "cost_usd":           round(total_cost, 2),
        "kelly_usd":          round(total_cost, 2),
        "shares":             round(filled_shares, 4),
        "entry_tau":          tau,
        "entry_ts":           _time.time(),
        "p_at_entry":         p_at_entry,
        "confidence_at_entry": p_at_entry,
        "confidence_target":  confidence_target,
        "z_at_entry":         m["signal"]["z"],
        "edge_at_entry":      chosen_dec["edge"],
        "threshold_at_entry": threshold_at_entry,
        "oracle_lag_signed_at_entry": signed_lag,
        "oracle_lag_abs_at_entry": abs(signed_lag) if signed_lag is not None else None,
        "oracle_lag_velocity_at_entry": lag_velocity,
        "oracle_lag_active_at_entry": lag_ctx["active"],
        "oracle_lag_aligned_at_entry": lag_ctx["aligned"],
        "oracle_lag_growing_at_entry": lag_ctx["growing"],
        "oracle_lag_confirmed_at_entry": lag_ctx["confirmed"],
        "oracle_lag_bonus_at_entry": lag_edge_bonus,
        "decision_reason":    chosen_dec["reason"],
    }

# ── Periodic lag JSONL flush ──────────────────────────────────────────────────

async def _flush_lag_periodically(cancel: asyncio.Event):
    """Flush accumulated lag samples to JSONL every 60 seconds."""
    try:
        while not cancel.is_set():
            await asyncio.sleep(60)
            _flush_lag_to_disk()
    except asyncio.CancelledError:
        _flush_lag_to_disk()


# ── Periodic signal update loop ───────────────────────────────────────────────

async def _signal_loop(cancel: asyncio.Event):
    """Tick signals every second for all active windows."""
    while not cancel.is_set():
        await asyncio.sleep(1)
        # Always sample lag history on a steady 1s clock, regardless of WS event rate
        _update_lag_history()
        for mkey in list(markets.keys()):
            if mkey in markets:
                # Update tau
                m = markets[mkey]
                end_ms = m["meta"]["end_ms"]
                m["meta"]["tau"] = max(0.0, (end_ms - int(_time.time() * 1000)) / 1000.0)
                _tick_signal(mkey)


# ── State serializer ──────────────────────────────────────────────────────────

def _build_state() -> dict:
    now = _time.time()

    # Oracle lag details
    cl  = oracle["chainlink_price"]
    mid = oracle["binance_mid"]
    lag = _lag_pct(mid, cl)

    last_cl_changed = oracle["last_cl_changed_ts"]
    cl_age_s = round(now - last_cl_changed, 1) if last_cl_changed else None

    # Lag history for sparkline (last 60 readings)
    lag_hist_raw = list(oracle["lag_history"])[-120:]
    lag_hist = [round(v * 100, 4) for _, v in lag_hist_raw]  # convert to %

    accuracy_rows = []
    for thresh, bucket in oracle["accuracy"].items():
        n = bucket["n"]
        moved = bucket["moved"]
        correct = bucket["correct"]
        accuracy_rows.append({
            "threshold_pct": round(thresh * 100, 4),
            "samples": n,
            "move_rate": round(moved / n * 100, 1) if n else None,
            "hit_rate": round(correct / n * 100, 1) if n else None,
            "directional_rate": round(correct / moved * 100, 1) if moved else None,
        })

    current_bucket = None
    if lag is not None:
        eligible = [
            row for row in accuracy_rows
            if abs(lag) * 100 >= row["threshold_pct"] and row["samples"] > 0
        ]
        if eligible:
            current_bucket = eligible[-1]

    oracle_data = {
        "binance_bid":    oracle["binance_bid"],
        "binance_ask":    oracle["binance_ask"],
        "binance_mid":    round(mid, 2) if mid else None,
        "chainlink":      round(cl, 2)  if cl  else None,
        "lag_pct":        round(lag * 100, 4) if lag is not None else None,
        "lag_frac":       lag,
        "trigger_pct":    CHAINLINK_TRIGGER_FRAC * 100,
        "trigger_progress": min(1.0, abs(lag) / CHAINLINK_TRIGGER_FRAC) if lag is not None else 0,
        "lag_direction":  ("UP" if lag > 0 else "DOWN") if lag else None,
        "cl_age_s":       cl_age_s,
        "lag_hist":       lag_hist,
        "lag_velocity_bps_s": round(oracle["lag_velocity"] * 10000, 2)
            if oracle["lag_velocity"] is not None else None,
        "feed_mode":      (
            f"Rust atomics ({str(oracle.get('rust_feed_mode') or 'json').upper()})"
            if oracle["rust_feeds"] else
            "Python websockets"
        ),
        "accuracy_horizon_s": LAG_ACCURACY_HORIZON_S,
        "accuracy_rows":  accuracy_rows,
        "current_bucket": current_bucket,
        "pending_checks": len(oracle["_pending_checks"]),
        # Distance remaining before CL triggers
        "dist_to_trigger": round((CHAINLINK_TRIGGER_FRAC - abs(lag)) * 100, 4)
            if lag is not None else None,
    }

    # Per-market state
    available_cash = _paper_available_cash()
    market_states = {}
    for mkey, m in markets.items():
        sig = m.get("signal", {})
        pt  = m.get("paper_trade")
        meta = m["meta"]

        # Current mark-to-market P&L for open paper trade
        mtm_pnl     = None
        mtm_pnl_pct = None
        if pt and "final_pnl" not in pt:
            cost = float(pt.get("cost_usd", 0.0))
            shares = float(pt.get("shares", 0.0))
            # Mark at midpoint bid/ask for the held side
            up_bid   = _best_bid(m["book"]["up_bids"])
            down_bid = _best_bid(m["book"]["down_bids"])
            mid_price = None
            if pt["side"] == "UP" and up_bid and sig.get("up_ask") is not None:
                mid_price = (up_bid + sig["up_ask"]) / 2
            elif pt["side"] == "DOWN" and down_bid and sig.get("down_ask") is not None:
                mid_price = (down_bid + sig["down_ask"]) / 2
            if mid_price is not None:
                mtm_value = shares * mid_price
                mtm_pnl   = round(mtm_value - cost, 2)
                mtm_pnl_pct = round(mtm_pnl / cost * 100, 1) if cost > 0 else 0

        market_states[mkey] = {
            "meta": {
                "slug":      meta["slug"],
                "url":       meta["url"],
                "timeframe": meta["timeframe"],
                "start_price": round(meta["start_price"], 2) if meta.get("start_price") else None,
                "current_price": round(sig["cl"], 2) if sig.get("cl") else None,
                "tau_s":      int(meta.get("tau", 0)),
                "resolved":   m.get("resolved"),
                "end_ms":     meta["end_ms"],
            },
            "signal": {
                "engine":          sig.get("engine"),
                "calibrated":      sig.get("calibrated"),
                "z":               round(sig["z"], 4) if sig.get("z") is not None else None,
                "z_raw":           round(sig["z_raw"], 4) if sig.get("z_raw") is not None else None,
                "p_up":            round(sig["p_up"] * 100, 2) if sig.get("p_up") is not None else None,
                "p_down":          round(sig["p_down"] * 100, 2) if sig.get("p_down") is not None else None,
                "p_trade_up":      round(sig["p_trade_up"] * 100, 2) if sig.get("p_trade_up") is not None else None,
                "p_trade_down":    round(sig["p_trade_down"] * 100, 2) if sig.get("p_trade_down") is not None else None,
                "sigma":           round(sig["sigma"], 8) if sig.get("sigma") is not None else None,
                "up_bid":          round(sig["up_bid"], 4) if sig.get("up_bid") is not None else None,
                "up_ask":          round(sig["up_ask"], 4) if sig.get("up_ask") is not None else None,
                "down_bid":        round(sig["down_bid"], 4) if sig.get("down_bid") is not None else None,
                "down_ask":        round(sig["down_ask"], 4) if sig.get("down_ask") is not None else None,
                "edge_up":         round(sig["edge_up"] * 100, 2) if sig.get("edge_up") is not None else None,
                "edge_down":       round(sig["edge_down"] * 100, 2) if sig.get("edge_down") is not None else None,
                "threshold_up":    round(sig["threshold_up"] * 100, 2) if sig.get("threshold_up") is not None else None,
                "threshold_down":  round(sig["threshold_down"] * 100, 2) if sig.get("threshold_down") is not None else None,
                "toxicity":        round(sig["toxicity"], 3) if sig.get("toxicity") is not None else None,
                "vpin":            round(sig["vpin"], 3) if sig.get("vpin") is not None else None,
                "oracle_lag":      round(sig["oracle_lag"] * 100, 3) if sig.get("oracle_lag") is not None else None,
                "oracle_lag_signed": round(sig["oracle_lag_signed"] * 100, 3) if sig.get("oracle_lag_signed") is not None else None,
                "oracle_lag_velocity_bps_s": round(sig["oracle_lag_velocity"] * 10000, 2) if sig.get("oracle_lag_velocity") is not None else None,
                "confidence":      round(sig["confidence"] * 100, 1) if sig.get("confidence") is not None else None,
                "confidence_target": round(sig["confidence_target"] * 100, 1) if sig.get("confidence_target") is not None else None,
                "lag_confirms_up": bool(sig.get("lag_confirms_up")),
                "lag_confirms_down": bool(sig.get("lag_confirms_down")),
                "lag_bonus":       round(sig["lag_bonus"] * 100, 2) if sig.get("lag_bonus") is not None else None,
                "regime_z_factor": round(sig["regime_z_factor"], 3) if sig.get("regime_z_factor") is not None else None,
                "up_action":       sig.get("up_action"),
                "down_action":     sig.get("down_action"),
                "up_reason":       sig.get("up_reason"),
                "down_reason":     sig.get("down_reason"),
                "selected_side":   sig.get("selected_side"),
                "paper_gate":      sig.get("paper_gate"),
                "signal_bankroll": round(sig["signal_bankroll"], 2) if sig.get("signal_bankroll") is not None else None,
            },
            "paper_trade": {
                "side":             pt["side"],
                "entry_price":      round(pt["entry_price"], 4),
                "entry_quote":      round(pt["entry_quote"], 4) if pt.get("entry_quote") is not None else None,
                "cost_usd":         round(pt["cost_usd"], 2),
                "kelly_usd":        round(pt["cost_usd"], 2),
                "shares":           round(pt["shares"], 4),
                "entry_tau":        int(pt["entry_tau"]),
                "p_at_entry":       round(pt["p_at_entry"] * 100, 1) if pt.get("p_at_entry") is not None else None,
                "confidence_at_entry": round(pt["confidence_at_entry"] * 100, 1) if pt.get("confidence_at_entry") is not None else None,
                "confidence_target": round(pt["confidence_target"] * 100, 1) if pt.get("confidence_target") is not None else None,
                "z_at_entry":       round(pt["z_at_entry"], 3) if pt.get("z_at_entry") is not None else None,
                "edge_at_entry":    round(pt["edge_at_entry"] * 100, 2) if pt.get("edge_at_entry") is not None else None,
                "threshold_at_entry": round(pt["threshold_at_entry"] * 100, 2) if pt.get("threshold_at_entry") is not None else None,
                "oracle_lag_signed_at_entry": round(pt["oracle_lag_signed_at_entry"] * 100, 3) if pt.get("oracle_lag_signed_at_entry") is not None else None,
                "oracle_lag_velocity_bps_s": round(pt["oracle_lag_velocity_at_entry"] * 10000, 2) if pt.get("oracle_lag_velocity_at_entry") is not None else None,
                "oracle_lag_confirmed_at_entry": pt.get("oracle_lag_confirmed_at_entry"),
                "oracle_lag_bonus_at_entry": round(pt["oracle_lag_bonus_at_entry"] * 100, 2) if pt.get("oracle_lag_bonus_at_entry") is not None else None,
                "decision_reason":  pt.get("decision_reason"),
                "mtm_pnl":          mtm_pnl,
                "mtm_pnl_pct":      mtm_pnl_pct,
                "final_pnl":        pt.get("final_pnl"),
                "final_pnl_pct": pt.get("final_pnl_pct"),
            } if pt else None,
        }

    return {
        "oracle": oracle_data,
        "markets": market_states,
        "paper_available_cash": available_cash,
        "paper_open_mark_value": _paper_open_mark_value(),
        "ts": now,
    }


def _build_trade_log_state() -> dict:
    rows = sorted(
        list(paper_trade_log),
        key=lambda row: ((row.get("start_ms") or 0), row.get("timeframe") or ""),
        reverse=True,
    )
    traded_rows = [row for row in rows if row["traded"]]
    realized_pnl = _paper_realized_pnl()
    total_staked = round(sum(row["bet_usd"] for row in traded_rows), 2)
    total_payout = round(sum(row["payout_usd"] for row in traded_rows), 2)
    wins = sum(1 for row in traded_rows if row["pnl_usd"] > 0)
    losses = sum(1 for row in traded_rows if row["pnl_usd"] < 0)
    open_trades = sum(
        1 for m in markets.values()
        if m.get("paper_trade") is not None and "final_pnl" not in m["paper_trade"]
    )
    open_mark_value = _paper_open_mark_value()
    available_cash = _paper_available_cash()
    paper_equity = round(available_cash + open_mark_value, 2)
    unrealized_pnl = round(paper_equity - cfg["bankroll"] - realized_pnl, 2)

    return {
        "summary": {
            "kelly_base_bankroll": round(cfg["bankroll"], 2),
            "realized_pnl": realized_pnl,
            "paper_equity": paper_equity,
            "unrealized_pnl": unrealized_pnl,
            "available_cash": available_cash,
            "open_mark_value": open_mark_value,
            "windows_total": len(rows),
            "windows_traded": len(traded_rows),
            "windows_no_trade": len(rows) - len(traded_rows),
            "win_rate": round(wins / len(traded_rows) * 100, 1) if traded_rows else None,
            "wins": wins,
            "losses": losses,
            "total_staked": total_staked,
            "total_payout": total_payout,
            "open_trades": open_trades,
            "windows_5m": sum(1 for row in rows if "5m" in (row.get("timeframe") or "")),
            "windows_15m": sum(1 for row in rows if "15m" in (row.get("timeframe") or "")),
        },
        "rows": rows,
    }


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI()


@app.get("/api/state")
async def api_state():
    return JSONResponse(_build_state())


@app.get("/api/trades")
async def api_trades():
    return JSONResponse(_build_trade_log_state())


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(HTML_TEMPLATE)


# ── HTML template ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BTC Polymarket Dashboard</title>
<style>
  :root {
    --bg: #0d0d0d; --bg2: #141414; --bg3: #1a1a1a;
    --border: #2a2a2a; --border2: #333;
    --text: #e0e0e0; --dim: #888; --dimmer: #555;
    --green: #00c853; --red: #ff1744; --yellow: #ffd600;
    --orange: #ff6d00; --blue: #2979ff; --purple: #aa00ff;
    --mono: 'SF Mono', 'Fira Code', 'Consolas', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--mono);
         font-size: 13px; line-height: 1.5; padding: 12px; }
  h2 { font-size: 11px; text-transform: uppercase; letter-spacing: 2px;
       color: var(--dim); margin-bottom: 8px; }
  .topbar {
    display: flex; justify-content: space-between; align-items: center;
    gap: 10px; margin-bottom: 12px; flex-wrap: wrap;
  }
  .nav-actions { display: flex; gap: 8px; flex-wrap: wrap; }
  .nav-btn {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 10px; border-radius: 4px; text-decoration: none;
    border: 1px solid var(--border2); background: var(--bg2); color: var(--text);
    font-size: 11px; cursor: pointer; font-family: var(--mono);
  }
  .nav-btn:hover { border-color: var(--blue); color: var(--blue); }
  .nav-btn.tab-active { border-color: var(--blue); color: var(--blue); background: #0a1a3a; }
  .tab-panel { display: none; }
  .tab-panel.tab-active { display: block; }
  .grid { display: grid; gap: 12px; }
  .panel { background: var(--bg2); border: 1px solid var(--border);
           border-radius: 4px; padding: 14px; }
  .panel-title { font-size: 10px; text-transform: uppercase; letter-spacing: 1.5px;
                 color: var(--dimmer); margin-bottom: 12px; }

  /* Oracle panel */
  #oracle-panel { grid-column: 1 / -1; }
  .oracle-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .oracle-prices { display: flex; flex-direction: column; gap: 6px; }
  .price-row { display: flex; justify-content: space-between; align-items: baseline; }
  .price-label { color: var(--dim); font-size: 11px; }
  .price-val { font-size: 15px; font-weight: bold; }
  .lag-section { display: flex; flex-direction: column; gap: 8px; }
  .lag-big { font-size: 22px; font-weight: bold; letter-spacing: -0.5px; }
  .lag-up   { color: var(--green); }
  .lag-down { color: var(--red); }
  .lag-zero { color: var(--dim); }

  /* Progress bar */
  .progress-track {
    height: 8px; background: var(--bg3); border-radius: 4px;
    overflow: hidden; position: relative;
  }
  .progress-fill {
    height: 100%; border-radius: 4px;
    transition: width 0.3s ease, background-color 0.3s ease;
  }
  .prog-0  { background: var(--green); }
  .prog-50 { background: var(--yellow); }
  .prog-80 { background: var(--orange); }
  .prog-100 { background: var(--red); animation: pulse 0.5s infinite alternate; }
  @keyframes pulse { from { opacity: 1; } to { opacity: 0.4; } }
  .progress-label { display: flex; justify-content: space-between; margin-top: 3px; }
  .progress-label span { font-size: 10px; color: var(--dim); }

  /* Sparkline */
  .sparkline-wrap { margin-top: 4px; }
  .sparkline-wrap svg { width: 100%; height: 32px; }

  /* Oracle details table */
  .oracle-detail { display: grid; grid-template-columns: auto 1fr; gap: 2px 12px; }
  .oracle-detail .k { color: var(--dim); font-size: 11px; white-space: nowrap; }
  .oracle-detail .v { font-size: 11px; }
  .feed-badge {
    display: inline-block; padding: 1px 6px; border-radius: 999px;
    background: var(--bg3); border: 1px solid var(--border2);
  }
  .accuracy-table { width: 100%; border-collapse: collapse; margin-top: 6px; }
  .accuracy-table th, .accuracy-table td { padding: 4px 6px; font-size: 10px; }
  .accuracy-table th {
    color: var(--dim); font-weight: normal; text-transform: uppercase;
    letter-spacing: 0.5px; border-bottom: 1px solid var(--border);
  }
  .accuracy-table td { text-align: right; }
  .accuracy-table th:first-child, .accuracy-table td:first-child { text-align: left; }
  .accuracy-table tr:nth-child(even) td { background: var(--bg3); }
  .acc-good { color: var(--green); }
  .acc-mid  { color: var(--yellow); }
  .acc-bad  { color: var(--red); }
  .summary-grid {
    display: grid; grid-template-columns: repeat(6, minmax(0, 1fr));
    gap: 10px; margin-bottom: 10px;
  }
  .summary-card {
    background: var(--bg3); border: 1px solid var(--border);
    border-radius: 4px; padding: 10px;
  }
  .summary-lbl {
    font-size: 9px; color: var(--dimmer); text-transform: uppercase;
    letter-spacing: 0.7px; margin-bottom: 4px;
  }
  .summary-val { font-size: 14px; }
  .summary-note { font-size: 10px; color: var(--dim); margin-bottom: 10px; }
  .log-wrap { overflow-x: auto; }
  .log-table { width: 100%; min-width: 1180px; border-collapse: collapse; }
  .log-table th, .log-table td { padding: 6px 8px; font-size: 10px; }
  .log-table th {
    color: var(--dim); font-weight: normal; text-transform: uppercase;
    letter-spacing: 0.5px; border-bottom: 1px solid var(--border);
    background: var(--bg2);
  }
  .log-table tr:nth-child(even) td { background: var(--bg3); }
  .log-link { color: var(--blue); text-decoration: none; }
  .log-link:hover { text-decoration: underline; }
  .tf-badge {
    display: inline-block; padding: 1px 6px; border-radius: 999px;
    background: var(--bg3); border: 1px solid var(--border2);
  }

  /* Market cards */
  .markets-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  @media (max-width: 1100px) {
    .summary-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  }
  @media (max-width: 900px) {
    .markets-row { grid-template-columns: 1fr; }
    .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  }
  .market-card { display: flex; flex-direction: column; gap: 10px; }
  .market-header { display: flex; justify-content: space-between; align-items: flex-start; }
  .market-title { font-size: 14px; font-weight: bold; }
  .market-link { font-size: 10px; color: var(--blue); text-decoration: none; }
  .market-link:hover { text-decoration: underline; }
  .tau-badge { font-size: 10px; padding: 2px 6px; border-radius: 3px;
               background: var(--bg3); color: var(--dim); }
  .tau-low { background: #1a0000; color: var(--red); }
  .tau-mid { background: #1a1000; color: var(--yellow); }

  .price-summary { display: flex; gap: 16px; flex-wrap: wrap; }
  .ps-item { display: flex; flex-direction: column; }
  .ps-label { font-size: 9px; color: var(--dimmer); text-transform: uppercase; }
  .ps-val { font-size: 13px; }
  .move-up   { color: var(--green); }
  .move-down { color: var(--red); }

  /* Signal table */
  .signal-table { width: 100%; border-collapse: collapse; }
  .signal-table td { padding: 3px 6px; font-size: 12px; }
  .signal-table td:first-child { color: var(--dim); width: 110px; }
  .signal-table tr:nth-child(even) td { background: var(--bg3); }

  /* Edge bars */
  .edge-row { display: flex; align-items: center; gap: 8px; }
  .edge-bar-wrap { flex: 1; height: 6px; background: var(--bg3); border-radius: 3px; overflow: hidden; }
  .edge-bar { height: 100%; border-radius: 3px; background: var(--green); }
  .edge-neg { background: var(--red); }
  .edge-label { font-size: 11px; min-width: 60px; text-align: right; }

  /* Paper trade box */
  .paper-trade { border: 1px solid var(--border2); border-radius: 3px;
                 padding: 10px; background: var(--bg3); }
  .pt-header { display: flex; justify-content: space-between; margin-bottom: 6px; }
  .pt-title { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; color: var(--dim); }
  .pt-side-up   { color: var(--green); font-weight: bold; }
  .pt-side-down { color: var(--red); font-weight: bold; }
  .pt-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px 16px; }
  .pt-item { display: flex; flex-direction: column; }
  .pt-lbl { font-size: 9px; color: var(--dimmer); text-transform: uppercase; }
  .pt-val { font-size: 12px; }
  .pnl-pos { color: var(--green); }
  .pnl-neg { color: var(--red); }
  .pnl-neu { color: var(--dim); }

  /* Resolution badge */
  .resolved-up   { color: var(--green); font-weight: bold; font-size: 13px; }
  .resolved-down { color: var(--red); font-weight: bold; font-size: 13px; }
  .no-trade { color: var(--dimmer); font-size: 11px; font-style: italic; }

  /* Separator */
  .sep { border-top: 1px solid var(--border); margin: 4px 0; }

  /* Status bar */
  #status { position: fixed; bottom: 0; left: 0; right: 0;
            background: var(--bg2); border-top: 1px solid var(--border);
            padding: 4px 12px; font-size: 10px; color: var(--dimmer);
            display: flex; justify-content: space-between; }
  .dot-live { color: var(--green); }
</style>
</head>
<body>
<div class="topbar">
  <div style="font-size:10px;color:var(--dim)">
    Live oracle monitor + DiffusionSignal paper overlay
  </div>
  <div class="nav-actions">
    <button class="nav-btn tab-active" onclick="switchTab('live')">Live View</button>
    <button class="nav-btn" onclick="switchTab('logs')">Trade Logs</button>
  </div>
</div>

<div class="tab-panel tab-active" id="tab-live">
<div class="grid" id="live-view">

  <!-- Oracle Lag Panel -->
  <div class="panel" id="oracle-panel">
    <div class="panel-title">Oracle Lag Tracker &mdash; Binance Mid vs Chainlink (BTC/USD)</div>
    <div class="oracle-grid">

      <!-- Left: prices + lag gauge -->
      <div>
        <div class="oracle-prices" style="margin-bottom:12px">
          <div class="price-row">
            <span class="price-label">Binance Mid</span>
            <span class="price-val" id="o-binance-mid">---</span>
          </div>
          <div class="price-row">
            <span class="price-label">Chainlink Oracle</span>
            <span class="price-val" id="o-chainlink">---</span>
          </div>
          <div class="sep"></div>
          <div class="price-row" style="margin-top:4px">
            <span class="price-label">Lag (Binance &minus; CL)</span>
            <span class="lag-big" id="o-lag-pct">---</span>
          </div>
          <div id="o-lag-direction" style="font-size:11px;color:var(--dim);text-align:right"></div>
        </div>

        <!-- Progress bar to trigger -->
        <div style="margin-bottom:6px;font-size:10px;color:var(--dim)">
          Distance to 0.5000% trigger
        </div>
        <div class="progress-track">
          <div class="progress-fill" id="o-prog-fill" style="width:0%"></div>
        </div>
        <div class="progress-label">
          <span id="o-prog-left">0.0000%</span>
          <span id="o-prog-mid"></span>
          <span style="color:var(--dimmer)">0.5000%</span>
        </div>

        <!-- Sparkline -->
        <div class="sparkline-wrap">
          <svg id="sparkline" viewBox="0 0 300 32" preserveAspectRatio="none">
            <line x1="0" y1="16" x2="300" y2="16" stroke="#333" stroke-width="0.5"/>
          </svg>
        </div>
        <div style="font-size:9px;color:var(--dimmer);text-align:right">← 2 min lag history</div>
      </div>

      <!-- Right: detail table -->
      <div class="lag-section">
        <div class="oracle-detail" id="o-detail">
          <span class="k">Bid / Ask</span>       <span class="v" id="o-bidask">---</span>
          <span class="k">Feed mode</span>        <span class="v" id="o-feed-mode">---</span>
          <span class="k">Lag absolute</span>    <span class="v" id="o-lag-abs">---</span>
          <span class="k">Lag velocity</span>    <span class="v" id="o-lag-velocity">---</span>
          <span class="k">Trigger at</span>       <span class="v">±0.5000%</span>
          <span class="k">Remaining</span>       <span class="v" id="o-remain">---</span>
          <span class="k">CL last changed</span> <span class="v" id="o-cl-age">---</span>
          <span class="k">10s hit @ curr lag</span><span class="v" id="o-current-hit">---</span>
          <span class="k">Hypothesis</span>      <span class="v" id="o-implied">---</span>
          <span class="k">If CL updates now</span><span class="v" id="o-if-updates">---</span>
        </div>

        <div class="sep" style="margin-top:8px"></div>
        <div style="font-size:10px;color:var(--dimmer);line-height:1.8">
          Chainlink triggers on <strong style="color:var(--text)">±0.5%</strong> spot deviation
          or every <strong style="color:var(--text)">3600s</strong> heartbeat.<br>
          When lag &gt; 0.4%: oracle update is <strong style="color:var(--yellow)">imminent</strong>.<br>
          Lag direction is a hypothesis, not a guarantee. Validate it with the live hit-rate table.
        </div>

        <div class="sep" style="margin-top:8px"></div>
        <div style="font-size:10px;color:var(--dimmer);line-height:1.6">
          Live validation over <span id="o-accuracy-window">10</span>s horizon.
          <strong style="color:var(--text)">Move</strong> = Chainlink changed at all.
          <strong style="color:var(--text)">Hit</strong> = Chainlink moved in the lag direction.
        </div>
        <table class="accuracy-table">
          <thead>
            <tr>
              <th>|lag|</th>
              <th>N</th>
              <th>Move</th>
              <th>Hit</th>
              <th>Dir|Move</th>
            </tr>
          </thead>
          <tbody id="o-accuracy-body">
            <tr>
              <td colspan="5" style="text-align:left;color:var(--dimmer)">Collecting live samples…</td>
            </tr>
          </tbody>
        </table>
      </div>

    </div>
  </div>

  <!-- Market Cards -->
  <div class="markets-row" id="markets-row">
    <div class="panel" style="color:var(--dim);font-size:11px">
      Connecting to Polymarket feeds&hellip;
    </div>
  </div>

</div><!-- end .grid -->
</div><!-- end #tab-live -->

<div class="tab-panel" id="tab-logs">
  <div class="panel" id="trade-logs">
    <div class="panel-title">Paper Trade Logs</div>
    <div style="font-size:10px;color:var(--dimmer);line-height:1.6;margin:0 0 12px 0">
      One paper trade max per window. Entry timing and sizing come from <strong style="color:var(--text)">DiffusionSignal</strong>.
      Fills are modeled as a one-shot taker buy for easier validation, not as maker queue management.
      Use the <strong style="color:var(--text)">15m block</strong> column to line up the three 5m windows against each 15m window.
    </div>

    <div class="summary-grid" id="trade-summary">
      <div class="summary-card"><div class="summary-lbl">Kelly Base</div><div class="summary-val">---</div></div>
      <div class="summary-card"><div class="summary-lbl">Available Cash</div><div class="summary-val">---</div></div>
      <div class="summary-card"><div class="summary-lbl">Realized P&amp;L</div><div class="summary-val">---</div></div>
      <div class="summary-card"><div class="summary-lbl">Unrealized P&amp;L</div><div class="summary-val">---</div></div>
      <div class="summary-card"><div class="summary-lbl">Paper Equity</div><div class="summary-val">---</div></div>
      <div class="summary-card"><div class="summary-lbl">Win Rate</div><div class="summary-val">---</div></div>
      <div class="summary-card"><div class="summary-lbl">Total Staked</div><div class="summary-val">---</div></div>
    </div>
    <div class="summary-note" id="trade-summary-note">Waiting for resolved windows&hellip;</div>

    <div class="log-wrap">
      <table class="log-table">
        <thead>
          <tr>
            <th>Start UTC</th>
            <th>15m Block</th>
            <th>TF</th>
            <th>Outcome</th>
            <th>Trade</th>
            <th>Bet</th>
            <th>Entry</th>
            <th>Payout</th>
            <th>P&amp;L</th>
            <th>Start Px</th>
            <th>End Px</th>
            <th>Window</th>
          </tr>
        </thead>
        <tbody id="trade-log-body">
          <tr>
            <td colspan="12" style="text-align:left;color:var(--dimmer)">No resolved windows yet&hellip;</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</div><!-- end #tab-logs -->

<div id="status">
  <span><span class="dot-live" id="live-dot">●</span> Live &mdash; polling every 500ms</span>
  <span id="ts-label"></span>
</div>

<script>
let lastBlink = 0;

function switchTab(name) {
  document.querySelectorAll('.tab-panel').forEach(el => el.classList.remove('tab-active'));
  document.querySelectorAll('.nav-btn').forEach(el => el.classList.remove('tab-active'));
  document.getElementById('tab-' + name).classList.add('tab-active');
  const btns = document.querySelectorAll('.nav-btn');
  btns[name === 'live' ? 0 : 1].classList.add('tab-active');
}

function fmt(v, dec=2) { return v != null ? v.toFixed(dec) : '---'; }
function fmtPrice(v) { return v != null ? '$' + v.toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2}) : '---'; }
function fmtTau(s) {
  if (s == null || s <= 0) return 'EXPIRED';
  const m = Math.floor(s / 60), sec = s % 60;
  return m + ':' + String(sec).padStart(2,'0');
}
function colorClass(v, pos='pnl-pos', neg='pnl-neg', neu='pnl-neu') {
  if (v == null) return neu;
  return v > 0 ? pos : (v < 0 ? neg : neu);
}
function accClass(v) {
  if (v == null) return '';
  return v >= 55 ? 'acc-good' : (v >= 45 ? 'acc-mid' : 'acc-bad');
}
function fmtMoney(v) {
  return v != null ? '$' + Number(v).toFixed(2) : '---';
}
function fmtUtc(ms) {
  if (ms == null) return '---';
  return new Date(ms).toISOString().replace('T', ' ').slice(0, 16);
}

function setHtml(id, html) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = html;
}
function setClass(el, cls) {
  if (typeof el === 'string') el = document.getElementById(el);
  if (el) el.className = el.className.replace(/\blag-\w+|\bprog-\d+/g,'').trim() + ' ' + cls;
}

function renderOracle(o) {
  if (!o) return;

  setHtml('o-binance-mid', fmtPrice(o.binance_mid));
  setHtml('o-chainlink',   fmtPrice(o.chainlink));
  setHtml('o-bidask', o.binance_bid ? `$${o.binance_bid.toFixed(2)} / $${o.binance_ask.toFixed(2)}` : '---');
  setHtml('o-feed-mode', o.feed_mode ? `<span class="feed-badge">${o.feed_mode}</span>` : '---');
  setHtml('o-lag-velocity', o.lag_velocity_bps_s != null
    ? `${o.lag_velocity_bps_s > 0 ? '+' : ''}${o.lag_velocity_bps_s.toFixed(2)} bp/s`
    : '---');

  const lag = o.lag_pct;
  let lagEl = document.getElementById('o-lag-pct');
  if (lag != null) {
    const sign = lag > 0 ? '+' : '';
    lagEl.textContent = sign + lag.toFixed(4) + '%';
    lagEl.className = 'lag-big ' + (lag > 0 ? 'lag-up' : (lag < 0 ? 'lag-down' : 'lag-zero'));

    const dir = o.lag_direction;
    setHtml('o-lag-direction', dir ? `▲ Binance ${dir === 'UP' ? 'ABOVE' : 'BELOW'} Chainlink → hypothesis: next CL move <strong>${dir}</strong>` : '');

    // Progress bar
    const pct = Math.min(100, o.trigger_progress * 100);
    const fill = document.getElementById('o-prog-fill');
    if (fill) {
      fill.style.width = pct.toFixed(1) + '%';
      fill.className = 'progress-fill ' + (pct >= 100 ? 'prog-100' : pct >= 80 ? 'prog-80' : pct >= 50 ? 'prog-50' : 'prog-0');
    }
    setHtml('o-prog-left', Math.abs(lag).toFixed(4) + '%');
    setHtml('o-prog-mid',  pct.toFixed(1) + '% of trigger');
    setHtml('o-lag-abs',   Math.abs(lag).toFixed(4) + '%');
    setHtml('o-remain',    o.dist_to_trigger != null
      ? (o.dist_to_trigger > 0
          ? o.dist_to_trigger.toFixed(4) + '% more needed'
          : '<span style="color:var(--red);font-weight:bold">THRESHOLD EXCEEDED</span>')
      : '---');
    setHtml('o-implied',   dir
      ? `<span class="${dir==='UP'?'pnl-pos':'pnl-neg'}">▲ ${dir} bias</span>`
      : '---');
    setHtml('o-if-updates', o.chainlink != null && o.binance_mid != null
      ? `$${o.binance_mid.toFixed(2)} (${(lag>0?'+':'')}${lag.toFixed(4)}%)` : '---');
  } else {
    lagEl.textContent = '---';
    lagEl.className = 'lag-big lag-zero';
  }

  setHtml('o-cl-age', o.cl_age_s != null ? o.cl_age_s.toFixed(1) + 's ago' : '---');
  setHtml('o-accuracy-window', o.accuracy_horizon_s != null ? o.accuracy_horizon_s.toFixed(0) : '10');

  const curr = o.current_bucket;
  setHtml('o-current-hit', curr
    ? `<span class="${accClass(curr.hit_rate)}">${curr.hit_rate != null ? curr.hit_rate.toFixed(1)+'%' : '---'}</span> hit / `
      + `${curr.move_rate != null ? curr.move_rate.toFixed(1)+'%' : '---'} move `
      + `(${curr.samples} obs)`
    : (o.pending_checks ? `Collecting (${o.pending_checks} pending)` : 'Collecting live samples'));

  const accRows = o.accuracy_rows || [];
  const accBody = document.getElementById('o-accuracy-body');
  if (accBody) {
    if (accRows.length === 0 || accRows.every(r => !r.samples)) {
      accBody.innerHTML = '<tr><td colspan="5" style="text-align:left;color:var(--dimmer)">Collecting live samples…</td></tr>';
    } else {
      accBody.innerHTML = accRows.map(r => `
        <tr>
          <td>&ge;${r.threshold_pct.toFixed(2)}%</td>
          <td>${r.samples}</td>
          <td class="${accClass(r.move_rate)}">${r.move_rate != null ? r.move_rate.toFixed(1)+'%' : '---'}</td>
          <td class="${accClass(r.hit_rate)}">${r.hit_rate != null ? r.hit_rate.toFixed(1)+'%' : '---'}</td>
          <td class="${accClass(r.directional_rate)}">${r.directional_rate != null ? r.directional_rate.toFixed(1)+'%' : '---'}</td>
        </tr>
      `).join('');
    }
  }

  // Sparkline
  const hist = o.lag_hist || [];
  if (hist.length > 1) {
    const W = 300, H = 32, PAD = 2;
    const min = Math.min(...hist), max = Math.max(...hist);
    const range = max - min || 0.001;
    const pts = hist.map((v, i) => {
      const x = PAD + (i / (hist.length - 1)) * (W - 2*PAD);
      const y = H - PAD - ((v - min) / range) * (H - 2*PAD);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
    // Zero line
    const zero_y = H - PAD - ((0 - min) / range) * (H - 2*PAD);
    const svg = document.getElementById('sparkline');
    if (svg) {
      svg.innerHTML = `
        <line x1="0" y1="${zero_y.toFixed(1)}" x2="${W}" y2="${zero_y.toFixed(1)}"
              stroke="#333" stroke-width="0.5" stroke-dasharray="2,2"/>
        <polyline points="${pts}" fill="none" stroke="${lag > 0 ? '#00c853' : '#ff1744'}" stroke-width="1.5"/>
      `;
    }
  }
}

function renderTradeLog(data) {
  if (!data) return;
  const s = data.summary || {};
  const pnlClass = colorClass(s.realized_pnl);
  const unrealizedClass = colorClass(s.unrealized_pnl);
  const cashDelta = (s.available_cash != null && s.kelly_base_bankroll != null)
    ? (s.available_cash - s.kelly_base_bankroll) : null;
  const cashClass = colorClass(cashDelta);
  const equityDelta = (s.paper_equity != null && s.kelly_base_bankroll != null)
    ? (s.paper_equity - s.kelly_base_bankroll) : null;
  const equityClass = colorClass(equityDelta);

  const summary = document.getElementById('trade-summary');
  if (summary) {
    summary.innerHTML = `
      <div class="summary-card"><div class="summary-lbl">Kelly Base</div><div class="summary-val">${fmtMoney(s.kelly_base_bankroll)}</div></div>
      <div class="summary-card"><div class="summary-lbl">Available Cash</div><div class="summary-val ${cashClass}">${fmtMoney(s.available_cash)}</div></div>
      <div class="summary-card"><div class="summary-lbl">Realized P&amp;L</div><div class="summary-val ${pnlClass}">${fmtMoney(s.realized_pnl)}</div></div>
      <div class="summary-card"><div class="summary-lbl">Unrealized P&amp;L</div><div class="summary-val ${unrealizedClass}">${fmtMoney(s.unrealized_pnl)}</div></div>
      <div class="summary-card"><div class="summary-lbl">Paper Equity</div><div class="summary-val ${equityClass}">${fmtMoney(s.paper_equity)}</div></div>
      <div class="summary-card"><div class="summary-lbl">Win Rate</div><div class="summary-val ${accClass(s.win_rate)}">${s.win_rate != null ? s.win_rate.toFixed(1) + '%' : '---'}</div></div>
      <div class="summary-card"><div class="summary-lbl">Total Staked</div><div class="summary-val">${fmtMoney(s.total_staked)}</div></div>
    `;
  }

  setHtml('trade-summary-note',
    `${s.windows_5m ?? 0} resolved 5m windows | ${s.windows_15m ?? 0} resolved 15m windows | `
    + `${s.windows_traded ?? 0} traded windows | ${s.windows_no_trade ?? 0} no-trade windows | `
    + `${s.open_trades ?? 0} open paper trades | open mark ${fmtMoney(s.open_mark_value)}`
  );

  const body = document.getElementById('trade-log-body');
  if (!body) return;
  const rows = data.rows || [];
  if (rows.length === 0) {
    body.innerHTML = '<tr><td colspan="12" style="text-align:left;color:var(--dimmer)">No resolved windows yet&hellip;</td></tr>';
    return;
  }

  body.innerHTML = rows.map(r => {
    const pnlCls = colorClass(r.pnl_usd);
    const outcomeCls = r.resolved === 'UP' ? 'pnl-pos' : (r.resolved === 'DOWN' ? 'pnl-neg' : 'pnl-neu');
    const tradeLabel = r.traded
      ? `<span class="${r.trade_side === 'UP' ? 'pnl-pos' : 'pnl-neg'}">${r.trade_side}</span>`
      : '<span class="pnl-neu">NO TRADE</span>';
    const entryCell = r.entry_price != null
      ? `${r.entry_price.toFixed(4)}${r.entry_quote != null ? ` <span style="color:var(--dimmer)">(q ${r.entry_quote.toFixed(4)})</span>` : ''}`
      : '---';
    return `
      <tr>
        <td>${fmtUtc(r.start_ms)}</td>
        <td>${fmtUtc(r.block_15m_ms)}</td>
        <td><span class="tf-badge">${r.timeframe || '---'}</span></td>
        <td><span class="${outcomeCls}">${r.resolved || '---'}</span></td>
        <td>${tradeLabel}</td>
        <td>${r.traded ? fmtMoney(r.bet_usd) : '---'}</td>
        <td>${entryCell}</td>
        <td>${r.traded ? fmtMoney(r.payout_usd) : '---'}</td>
        <td><span class="${pnlCls}">${r.traded ? fmtMoney(r.pnl_usd) : '$0.00'}</span></td>
        <td>${r.start_price != null ? fmtPrice(r.start_price) : '---'}</td>
        <td>${r.end_price != null ? fmtPrice(r.end_price) : '---'}</td>
        <td>${r.url ? `<a class="log-link" href="${r.url}" target="_blank" rel="noopener">${r.slug}&nbsp;↗</a>` : (r.slug || '---')}</td>
      </tr>
    `;
  }).join('');
}

function renderMarket(mkey, m) {
  const meta = m.meta, sig = m.signal, pt = m.paper_trade;
  const tau = meta.tau_s;
  const tauClass = tau < 60 ? 'tau-low' : tau < 180 ? 'tau-mid' : '';

  // Price move
  const sp = meta.start_price, cp = meta.current_price;
  const movePct = (sp && cp) ? ((cp - sp) / sp * 100) : null;
  const moveStr = movePct != null
    ? `<span class="${movePct >= 0 ? 'move-up' : 'move-down'}">${movePct >= 0 ? '+' : ''}${movePct.toFixed(3)}%</span>`
    : '---';

  // Resolution badge
  let resolvedBadge = '';
  if (meta.resolved) {
    resolvedBadge = `<span class="resolved-${meta.resolved.toLowerCase()}">● ${meta.resolved}</span>`;
  }

  // Signal rows
  let sigRows = '<span style="color:var(--dimmer);font-size:11px">Waiting for signal&hellip;</span>';
  if (sig.engine) {
    const edgeUpColor = sig.edge_up != null && sig.edge_up > 0 ? 'pnl-pos' : (sig.edge_up != null && sig.edge_up < 0 ? 'pnl-neg' : '');
    const edgeDnColor = sig.edge_down != null && sig.edge_down > 0 ? 'pnl-pos' : (sig.edge_down != null && sig.edge_down < 0 ? 'pnl-neg' : '');
    const selected = sig.selected_side
      ? `<span class="${sig.selected_side === 'UP' ? 'pnl-pos' : 'pnl-neg'}">${sig.selected_side}</span>`
      : '<span class="pnl-neu">NONE</span>';
    const modeLbl = sig.calibrated ? 'Diffusion + calibration' : 'Diffusion only';
    sigRows = `
      <table class="signal-table">
        <tr><td>Mode</td><td>${modeLbl}</td></tr>
        <tr><td>z-score</td><td>${sig.z != null ? sig.z.toFixed(4) : '---'}</td></tr>
        <tr><td>p(UP) trade</td><td>${sig.p_trade_up != null ? sig.p_trade_up.toFixed(2)+'%' : '---'}</td></tr>
        <tr><td>p(DOWN) trade</td><td>${sig.p_trade_down != null ? sig.p_trade_down.toFixed(2)+'%' : '---'}</td></tr>
        <tr><td>σ (per-s)</td><td>${sig.sigma != null ? sig.sigma.toExponential(3) : '---'}</td></tr>
        <tr><td>UP bid / ask</td><td>${sig.up_bid != null ? sig.up_bid.toFixed(4) : '---'} / ${sig.up_ask != null ? sig.up_ask.toFixed(4) : '---'}</td></tr>
        <tr><td>DOWN bid / ask</td><td>${sig.down_bid != null ? sig.down_bid.toFixed(4) : '---'} / ${sig.down_ask != null ? sig.down_ask.toFixed(4) : '---'}</td></tr>
        <tr><td>UP edge / thr</td><td class="${edgeUpColor}">${sig.edge_up != null ? (sig.edge_up>0?'+':'')+sig.edge_up.toFixed(2)+'%' : '---'} / ${sig.threshold_up != null ? sig.threshold_up.toFixed(2)+'%' : '---'}</td></tr>
        <tr><td>DOWN edge / thr</td><td class="${edgeDnColor}">${sig.edge_down != null ? (sig.edge_down>0?'+':'')+sig.edge_down.toFixed(2)+'%' : '---'} / ${sig.threshold_down != null ? sig.threshold_down.toFixed(2)+'%' : '---'}</td></tr>
        <tr><td>Toxicity / VPIN</td><td>${sig.toxicity != null ? sig.toxicity.toFixed(3) : '---'} / ${sig.vpin != null ? sig.vpin.toFixed(3) : '---'}</td></tr>
        <tr><td>Oracle Lag</td><td>${sig.oracle_lag != null ? (sig.oracle_lag > 0 ? '+' : '') + sig.oracle_lag.toFixed(3) + '%' : '---'}</td></tr>
        <tr><td>Lag Sign / Vel</td><td>${sig.oracle_lag_signed != null ? (sig.oracle_lag_signed > 0 ? '+' : '') + sig.oracle_lag_signed.toFixed(3) + '%' : '---'} / ${sig.oracle_lag_velocity_bps_s != null ? (sig.oracle_lag_velocity_bps_s > 0 ? '+' : '') + sig.oracle_lag_velocity_bps_s.toFixed(2) + ' bp/s' : '---'}</td></tr>
        <tr><td>Confidence / Bonus</td><td>${sig.confidence != null ? sig.confidence.toFixed(1)+'%' : '---'} / ${sig.confidence_target != null ? sig.confidence_target.toFixed(1)+'%' : '---'} | lag bonus ${sig.lag_bonus != null ? sig.lag_bonus.toFixed(2)+'%' : '0.00%'}</td></tr>
        <tr><td>z-scale / cash</td><td>${sig.regime_z_factor != null ? sig.regime_z_factor.toFixed(3) : '---'} / ${sig.signal_bankroll != null ? fmtMoney(sig.signal_bankroll) : '---'}</td></tr>
        <tr><td>Chosen side</td><td>${selected}</td></tr>
        <tr><td>Gate</td><td>${sig.paper_gate || sig.up_reason || sig.down_reason || 'ready'}</td></tr>
      </table>`;
  }

  // Paper trade section
  let ptHtml = `<div class="no-trade">${sig.paper_gate || 'No paper trade yet'}</div>`;
  if (pt) {
    const sideClass = pt.side === 'UP' ? 'pt-side-up' : 'pt-side-down';
    const sideArrow = pt.side === 'UP' ? '▲' : '▼';
    const pnl = pt.final_pnl != null ? pt.final_pnl : pt.mtm_pnl;
    const pnlPct = pt.final_pnl_pct != null ? pt.final_pnl_pct : pt.mtm_pnl_pct;
    const pnlLabel = pt.final_pnl != null ? 'P&L (final)' : 'P&L (mark)';
    const pnlColor = colorClass(pnl);
    const pnlStr = pnl != null
      ? `<span class="${pnlColor}">${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)} (${pnlPct >= 0 ? '+' : ''}${pnlPct != null ? pnlPct.toFixed(1) : '---'}%)</span>`
      : '<span class="pnl-neu">---</span>';

    ptHtml = `
      <div class="paper-trade">
        <div class="pt-header">
          <span class="pt-title">Paper Trade</span>
          <span class="${sideClass}">${sideArrow} ${pt.side}</span>
        </div>
        <div class="pt-grid">
          <div class="pt-item"><span class="pt-lbl">Entry Eff / Quote</span><span class="pt-val">${pt.entry_price.toFixed(4)} / ${pt.entry_quote != null ? pt.entry_quote.toFixed(4) : '---'}</span></div>
          <div class="pt-item"><span class="pt-lbl">Bet / Shares</span><span class="pt-val">${fmtMoney(pt.cost_usd)} / ${pt.shares.toFixed(2)}</span></div>
          <div class="pt-item"><span class="pt-lbl">p at Entry</span><span class="pt-val">${pt.p_at_entry != null ? pt.p_at_entry.toFixed(1)+'%' : '---'}</span></div>
          <div class="pt-item"><span class="pt-lbl">Conf / Target</span><span class="pt-val">${pt.confidence_at_entry != null ? pt.confidence_at_entry.toFixed(1)+'%' : '---'} / ${pt.confidence_target != null ? pt.confidence_target.toFixed(1)+'%' : '---'}</span></div>
          <div class="pt-item"><span class="pt-lbl">Edge / Thr</span><span class="pt-val">${pt.edge_at_entry != null ? pt.edge_at_entry.toFixed(2)+'%' : '---'} / ${pt.threshold_at_entry != null ? pt.threshold_at_entry.toFixed(2)+'%' : '---'}</span></div>
          <div class="pt-item"><span class="pt-lbl">Lag / Bonus</span><span class="pt-val">${pt.oracle_lag_signed_at_entry != null ? (pt.oracle_lag_signed_at_entry > 0 ? '+' : '') + pt.oracle_lag_signed_at_entry.toFixed(3) + '%' : '---'} / ${pt.oracle_lag_bonus_at_entry != null ? pt.oracle_lag_bonus_at_entry.toFixed(2)+'%' : '---'}</span></div>
          <div class="pt-item"><span class="pt-lbl">Tau at Entry</span><span class="pt-val">${fmtTau(pt.entry_tau)}</span></div>
          <div class="pt-item"><span class="pt-lbl">${pnlLabel}</span><span class="pt-val">${pnlStr}</span></div>
        </div>
      </div>`;
  }

  return `
    <div class="panel market-card">
      <div class="market-header">
        <div>
          <div class="market-title">${meta.timeframe} ${resolvedBadge}</div>
          <a class="market-link" href="${meta.url}" target="_blank" rel="noopener">
            ${meta.slug}&nbsp;↗
          </a>
        </div>
        <span class="tau-badge ${tauClass}">${fmtTau(tau)} left</span>
      </div>

      <div class="price-summary">
        <div class="ps-item">
          <span class="ps-label">Start</span>
          <span class="ps-val">${fmtPrice(meta.start_price)}</span>
        </div>
        <div class="ps-item">
          <span class="ps-label">Current</span>
          <span class="ps-val">${fmtPrice(meta.current_price)}</span>
        </div>
        <div class="ps-item">
          <span class="ps-label">Move</span>
          <span class="ps-val">${moveStr}</span>
        </div>
      </div>

      <div class="sep"></div>
      ${sigRows}
      <div class="sep"></div>
      ${ptHtml}
    </div>`;
}

async function poll() {
  try {
    const needTrades = !window.__lastTradesFetch || (Date.now() - window.__lastTradesFetch) > 3000;
    const [stateResp, tradesResp] = await Promise.all([
      fetch('/api/state'),
      needTrades ? fetch('/api/trades') : Promise.resolve(null),
    ]);
    const data = await stateResp.json();
    if (tradesResp) {
      renderTradeLog(await tradesResp.json());
      window.__lastTradesFetch = Date.now();
    }

    renderOracle(data.oracle);

    const row = document.getElementById('markets-row');
    const mkeys = Object.keys(data.markets || {});
    if (mkeys.length === 0) {
      row.innerHTML = '<div class="panel" style="color:var(--dim);font-size:11px">No active windows found yet&hellip;</div>';
    } else {
      row.innerHTML = mkeys.map(k => renderMarket(k, data.markets[k])).join('');
    }

    // Status bar
    const now = new Date(data.ts * 1000);
    setHtml('ts-label', now.toISOString().replace('T', ' ').slice(0, 19) + ' UTC');

    // Blink live dot
    const dot = document.getElementById('live-dot');
    if (dot) { dot.style.opacity = '0.3'; setTimeout(() => { dot.style.opacity = '1'; }, 100); }

  } catch(e) {
    console.error('poll error', e);
  }
}

setInterval(poll, 500);
poll();
</script>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BTC Polymarket live dashboard")
    parser.add_argument("--bankroll", type=float, default=1000.0,
                        help="Starting paper bankroll in USD (default 1000)")
    parser.add_argument("--edge-threshold", type=float, default=None,
                        help="Override DiffusionSignal edge threshold (default: use live-trader market default)")
    parser.add_argument("--kelly-fraction", type=float, default=0.25,
                        help="Fractional Kelly multiplier (default 0.25 = quarter-Kelly)")
    parser.add_argument("--max-bet-fraction", type=float, default=0.05,
                        help="Maximum fraction of bankroll per paper trade (default 0.05 = 5%%)")
    parser.add_argument("--kelly-cap", type=float, default=None,
                        help="Backward-compatible alias for --max-bet-fraction")
    parser.add_argument("--paper-edge-buffer-5m", type=float, default=0.0,
                        help="Extra paper-trade edge required for 5m windows only (default 0.0)")
    parser.add_argument("--paper-min-confidence-15m", type=float, default=0.75,
                        help="Require chosen-side p_at_entry >= this for 15m paper trades (default 0.75)")
    parser.add_argument("--paper-lag-threshold-15m", type=float, default=0.0002,
                        help="Activate 15m lag confirmation when |Binance-Chainlink|/Chainlink >= this (default 0.0002 = 0.02%%)")
    parser.add_argument("--paper-lag-edge-bonus-15m", type=float, default=0.01,
                        help="Reduce required 15m paper edge by this amount when lag confirms the side (default 0.01)")
    parser.add_argument("--paper-require-growing-lag-15m",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Require active 15m oracle lag to be growing in the trade direction (default: on)")
    parser.add_argument("--slippage", type=float, default=0.0,
                        help="Extra taker slippage per share for paper fills (default 0.0)")
    parser.add_argument("--calibrated", action="store_true",
                        help="Use the calibration table, matching README live/backtest commands")
    parser.add_argument("--regime-z-scale", action="store_true",
                        help="Scale z by sigma_calibration / sigma_live (requires --calibrated for sigma calibration)")
    parser.add_argument("--min-z", type=float, default=0.0,
                        help="Minimum |z-score| to enter a trade (default 0.0 = disabled, "
                             "recommended 0.7 based on walk-forward analysis)")
    parser.add_argument("--min-entry-price", type=float, default=0.10,
                        help="Minimum contract bid price to trade (default 0.10, "
                             "use 0.25 to filter out cheap long-shot contracts)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    max_bet_fraction = args.kelly_cap if args.kelly_cap is not None else args.max_bet_fraction

    cfg["bankroll"] = args.bankroll
    cfg["edge_threshold"] = args.edge_threshold
    cfg["kelly_fraction"] = args.kelly_fraction
    cfg["max_bet_fraction"] = max_bet_fraction
    cfg["paper_edge_buffer_5m"] = args.paper_edge_buffer_5m
    cfg["paper_min_confidence_15m"] = args.paper_min_confidence_15m
    cfg["paper_lag_threshold_15m"] = args.paper_lag_threshold_15m
    cfg["paper_lag_edge_bonus_15m"] = args.paper_lag_edge_bonus_15m
    cfg["paper_require_growing_lag_15m"] = args.paper_require_growing_lag_15m
    cfg["slippage"] = args.slippage
    cfg["calibrated"] = args.calibrated
    cfg["regime_z_scale"] = args.regime_z_scale
    cfg["min_entry_z"] = args.min_z
    cfg["min_entry_price"] = args.min_entry_price

    print(f"  Bankroll:       ${args.bankroll:.2f}")
    if args.edge_threshold is None:
        print("  Edge threshold: live-trader default")
    else:
        print(f"  Edge threshold: {args.edge_threshold*100:.1f}%")
    if args.paper_edge_buffer_5m > 0:
        print(f"  5m edge buffer: +{args.paper_edge_buffer_5m*100:.1f}% (paper only)")
    print(f"  15m confidence: {args.paper_min_confidence_15m*100:.1f}% chosen-side p")
    print(
        f"  15m lag gate:   {args.paper_lag_threshold_15m*100:.3f}% active, "
        f"bonus={args.paper_lag_edge_bonus_15m*100:.1f}%, "
        f"growing={'on' if args.paper_require_growing_lag_15m else 'off'}"
    )
    print(f"  Kelly:         {args.kelly_fraction:.2f}x Kelly, max {max_bet_fraction*100:.1f}%")
    print(f"  Calibration:   {'on' if args.calibrated else 'off'}")
    print(f"  Regime z-scale:{' on' if args.regime_z_scale else ' off'}")
    print(f"  Slippage:      {args.slippage:.4f}")
    print(f"  Feed mode:      {'Rust atomics' if _RUST else 'Python websockets'}")
    print(f"  Dashboard:      http://localhost:{args.port}")

    cancel = asyncio.Event()

    @asynccontextmanager
    async def lifespan(_app):
        oracle["rust_feeds"] = False
        feed_tasks = (
            [asyncio.create_task(_rust_feed_poll_loop(cancel))]
            if _RUST else
            [
                asyncio.create_task(_binance_bookticker_ws(cancel)),
                asyncio.create_task(_chainlink_rtds_ws(cancel)),
            ]
        )
        tasks = feed_tasks + [
            asyncio.create_task(_signal_loop(cancel)),
            asyncio.create_task(_window_discovery_loop("btc",    cancel)),
            asyncio.create_task(_window_discovery_loop("btc_5m", cancel)),
            asyncio.create_task(_flush_lag_periodically(cancel)),
        ]
        yield
        cancel.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        _flush_lag_to_disk()  # final flush of any remaining lag samples

    app.router.lifespan_context = lifespan

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="warning",
        lifespan="on",
    )


if __name__ == "__main__":
    main()
