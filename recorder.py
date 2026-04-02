#!/usr/bin/env python3
"""
Polymarket Market Recorder (multi-timeframe)

Captures 1-second snapshots to parquet for backtesting:
  - Full top-5 book depth for Up and Down outcomes
  - Chainlink price (streaming)
  - Window metadata and time remaining
  - Trade ticks (last_trade_price events)

Outputs one parquet file per window under ./data/<market>/

Usage:
    py -3 recorder.py                  # BTC 15m + 5m
    py -3 recorder.py --market eth     # ETH 15m + 5m
    py -3 recorder.py --market btc_5m  # BTC 5m only
    py -3 recorder.py --debug
    py -3 recorder.py --top-n 10       # capture top-10 levels
"""

import argparse
import asyncio
import collections
import json
import os
import ssl
import sys
import time as _time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import websockets

from market_config import (
    MarketConfig, MARKET_CONFIGS, DEFAULT_MARKET,
    get_config, get_paired_configs,
)

try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    SSL_CTX.check_hostname = False
    SSL_CTX.verify_mode = ssl.CERT_NONE

# ── Config ───────────────────────────────────────────────────────────────────
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
RTDS_WS = "wss://ws-live-data.polymarket.com"
DATA_DIR = Path("data")

DEBUG = False
TOP_N = 5


# ── Order book ───────────────────────────────────────────────────────────────
class OrderBook:
    """Maintains a price-level order book from WS events."""

    __slots__ = ("bids", "asks")

    def __init__(self):
        self.bids: dict[float, float] = {}  # price -> size
        self.asks: dict[float, float] = {}

    def on_snapshot(self, bid_levels: list, ask_levels: list):
        self.bids = {
            float(l["price"]): float(l["size"]) for l in bid_levels
        }
        self.asks = {
            float(l["price"]): float(l["size"]) for l in ask_levels
        }

    def on_price_change(self, price: str, size: str, side: str):
        book = self.bids if side == "BUY" else self.asks
        p, s = float(price), float(size)
        if s == 0:
            book.pop(p, None)
        else:
            book[p] = s

    @property
    def best_bid(self) -> float | None:
        return max(self.bids) if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return min(self.asks) if self.asks else None

    def top_bids(self, n: int) -> list[tuple[float, float]]:
        """Top N bids, highest price first."""
        return sorted(self.bids.items(), key=lambda x: -x[0])[:n]

    def top_asks(self, n: int) -> list[tuple[float, float]]:
        """Top N asks, lowest price first."""
        return sorted(self.asks.items(), key=lambda x: x[0])[:n]


# ── Shared RTDS price state ─────────────────────────────────────────────────
# Shared across all timeframes (same asset price feed)
shared_price: dict = {
    "chainlink_price": None,
    "price_history": collections.deque(maxlen=600),
}

# ── Shared Binance trade flow state ─────────────────────────────────────────
# aggTrade stream accumulates buy/sell volume; sampler snaps+resets each tick.
binance_trade_state: dict = {
    # accumulator (reset each second by sampler)
    "acc_buy_vol":  0.0,
    "acc_sell_vol": 0.0,
    "acc_n":        0,
    "acc_vwap_num": 0.0,   # sum(price * qty) for buy+sell
    # last snapped values (what goes into the parquet row)
    "snap_buy_vol":  0.0,
    "snap_sell_vol": 0.0,
    "snap_n":        0,
    "snap_vwap":     None,
}

# ── Shared Deribit implied-vol state ────────────────────────────────────────
deribit_state: dict = {
    "dvol": None,   # annualised 30-day implied vol % (e.g. 65.0 = 65%)
}


def _ensure_list(val):
    return json.loads(val) if isinstance(val, str) else val


# ── Market discovery ─────────────────────────────────────────────────────────
def find_market(config: MarketConfig):
    align = config.window_align_m
    now = datetime.now(timezone.utc)
    minute = (now.minute // align) * align
    base = now.replace(minute=minute, second=0, microsecond=0)

    for offset in [0, -align, align, -2*align, 2*align]:
        candidate = base + timedelta(minutes=offset)
        ts = int(candidate.timestamp())
        slug = f"{config.slug_prefix}-{ts}"
        try:
            resp = requests.get(
                f"{GAMMA_API}/events",
                params={"slug": slug},
                timeout=10,
            )
            data = resp.json()
            if not data:
                continue
            event, market = data[0], data[0]["markets"][0]
            end = datetime.fromisoformat(
                market["endDate"].replace("Z", "+00:00")
            )
            start = datetime.fromisoformat(
                market["eventStartTime"].replace("Z", "+00:00")
            )
            if now < end or start > now:
                return event, market
        except Exception:
            continue

    return None, None


# ── Snapshot builder ─────────────────────────────────────────────────────────
def build_row(
    book_up: OrderBook, book_down: OrderBook,
    meta: dict,
    last_trade_up: dict | None, last_trade_down: dict | None,
    window_start_price: float | None,
) -> dict:
    """Sample current state into a flat dict for one parquet row."""
    now_ms = int(_time.time() * 1000)
    remaining_s = max(0, (meta["window_end_ms"] - now_ms) / 1000)

    row = {
        "ts_ms": now_ms,
        "market_slug": meta["market_slug"],
        "condition_id": meta["condition_id"],
        "token_id_up": meta["token_id_up"],
        "token_id_down": meta["token_id_down"],
        "window_start_ms": meta["window_start_ms"],
        "window_end_ms": meta["window_end_ms"],
        "time_remaining_s": round(remaining_s, 3),
        "chainlink_price": shared_price["chainlink_price"],
        "window_start_price": window_start_price,
    }

    # Book data for each side
    for label, book in [("up", book_up), ("down", book_down)]:
        bb = book.best_bid
        ba = book.best_ask
        row[f"best_bid_{label}"] = bb
        row[f"best_ask_{label}"] = ba

        # Sizes at best
        row[f"size_bid_{label}"] = book.bids.get(bb) if bb else None
        row[f"size_ask_{label}"] = book.asks.get(ba) if ba else None

        # Mid / spread
        if bb is not None and ba is not None:
            row[f"mid_{label}"] = round((bb + ba) / 2, 6)
            row[f"spread_{label}"] = round(ba - bb, 6)
        else:
            row[f"mid_{label}"] = None
            row[f"spread_{label}"] = None

        # Top N levels
        top_b = book.top_bids(TOP_N)
        top_a = book.top_asks(TOP_N)

        for i in range(TOP_N):
            if i < len(top_b):
                row[f"bid_px_{label}_{i+1}"] = top_b[i][0]
                row[f"bid_sz_{label}_{i+1}"] = top_b[i][1]
            else:
                row[f"bid_px_{label}_{i+1}"] = None
                row[f"bid_sz_{label}_{i+1}"] = None

            if i < len(top_a):
                row[f"ask_px_{label}_{i+1}"] = top_a[i][0]
                row[f"ask_sz_{label}_{i+1}"] = top_a[i][1]
            else:
                row[f"ask_px_{label}_{i+1}"] = None
                row[f"ask_sz_{label}_{i+1}"] = None

        # Depth and imbalance over top N
        bid_depth = sum(s for _, s in top_b)
        ask_depth = sum(s for _, s in top_a)
        total = bid_depth + ask_depth
        row[f"bid_depth{TOP_N}_{label}"] = round(bid_depth, 4)
        row[f"ask_depth{TOP_N}_{label}"] = round(ask_depth, 4)
        row[f"imbalance{TOP_N}_{label}"] = (
            round(bid_depth / total, 6) if total > 0 else None
        )

    # Last trade info
    for label, lt in [("up", last_trade_up), ("down", last_trade_down)]:
        if lt:
            row[f"last_trade_px_{label}"] = lt.get("price")
            row[f"last_trade_sz_{label}"] = lt.get("size")
            row[f"last_trade_side_{label}"] = lt.get("side")
        else:
            row[f"last_trade_px_{label}"] = None
            row[f"last_trade_sz_{label}"] = None
            row[f"last_trade_side_{label}"] = None

    # Binance trade flow (snapped by sampler before each call)
    bv_buy  = binance_trade_state["snap_buy_vol"]
    bv_sell = binance_trade_state["snap_sell_vol"]
    bv_tot  = bv_buy + bv_sell
    row["binance_buy_vol"]       = round(bv_buy,  6)
    row["binance_sell_vol"]      = round(bv_sell, 6)
    row["binance_net_flow"]      = round(bv_buy - bv_sell, 6)
    row["binance_flow_imbalance"] = (
        round((bv_buy - bv_sell) / bv_tot, 6) if bv_tot > 0 else None
    )
    row["binance_n_trades"]      = binance_trade_state["snap_n"]
    row["binance_vwap"]          = binance_trade_state["snap_vwap"]

    # Deribit 30-day implied vol index
    row["deribit_dvol"] = deribit_state["dvol"]

    return row


# ── Binance aggTrade WebSocket (shared) ──────────────────────────────────────
async def binance_trade_ws(symbol: str, cancel: asyncio.Event):
    """Stream Binance aggTrades and accumulate buy/sell volume each second."""
    url = f"wss://stream.binance.com:9443/ws/{symbol}@aggTrade"
    backoff = 2

    while not cancel.is_set():
        try:
            async with websockets.connect(
                url, ssl=SSL_CTX, ping_interval=20
            ) as ws:
                backoff = 2
                async for raw in ws:
                    if cancel.is_set():
                        break
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    # m=True  → buyer is maker (seller aggressed) → sell flow
                    # m=False → seller is maker (buyer aggressed) → buy flow
                    qty   = float(msg.get("q", 0))
                    price = float(msg.get("p", 0))
                    s = binance_trade_state
                    if msg.get("m", False):
                        s["acc_sell_vol"] += qty
                    else:
                        s["acc_buy_vol"] += qty
                    s["acc_n"] += 1
                    s["acc_vwap_num"] += price * qty
        except Exception as exc:
            if cancel.is_set():
                return
            if DEBUG:
                print(f"  [Binance] {type(exc).__name__}: {exc}")
            await asyncio.sleep(min(backoff, 30))
            backoff = min(backoff * 2, 60)


# ── Deribit DVOL WebSocket (shared, BTC/ETH only) ────────────────────────────
async def deribit_ws(asset: str, cancel: asyncio.Event):
    """Stream Deribit volatility index (annualised 30-day IV %)."""
    if asset not in {"btc", "eth"}:
        return   # Deribit DVOL only exists for BTC and ETH

    url     = "wss://www.deribit.com/ws/api/v2"
    channel = f"deribit_volatility_index.{asset}_usd"
    backoff = 2

    while not cancel.is_set():
        try:
            async with websockets.connect(
                url, ssl=SSL_CTX, ping_interval=None
            ) as ws:
                backoff = 2
                await ws.send(json.dumps({
                    "jsonrpc": "2.0", "id": 1,
                    "method": "public/subscribe",
                    "params": {"channels": [channel]},
                }))

                async def heartbeat():
                    try:
                        while not cancel.is_set():
                            await asyncio.sleep(15)
                            await ws.send(json.dumps({
                                "jsonrpc": "2.0", "id": 9999,
                                "method": "public/test",
                                "params": {},
                            }))
                    except Exception:
                        pass

                hb = asyncio.create_task(heartbeat())
                try:
                    async for raw in ws:
                        if cancel.is_set():
                            break
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        data = msg.get("params", {}).get("data", {})
                        if isinstance(data, dict) and "volatility" in data:
                            deribit_state["dvol"] = float(data["volatility"])
                            if DEBUG:
                                print(f"  [Deribit] DVOL={data['volatility']:.2f}")
                finally:
                    hb.cancel()

        except Exception as exc:
            if cancel.is_set():
                return
            if DEBUG:
                print(f"  [Deribit] {type(exc).__name__}: {exc}")
            await asyncio.sleep(min(backoff, 30))
            backoff = min(backoff * 2, 60)


# ── CLOB WebSocket (per-window, per-timeframe) ──────────────────────────────
async def clob_ws(
    up_token: str, down_token: str,
    book_up: OrderBook, book_down: OrderBook,
    trade_state: dict,
    cancel: asyncio.Event,
):
    """CLOB WS for one window — manages its own books and trade ticks."""
    token_map = {up_token: "up", down_token: "down"}
    book_map = {"up": book_up, "down": book_down}
    backoff = 2

    while not cancel.is_set():
        try:
            async with websockets.connect(
                CLOB_WS, ssl=SSL_CTX, ping_interval=None
            ) as ws:
                backoff = 2
                await ws.send(
                    json.dumps(
                        {
                            "assets_ids": [up_token, down_token],
                            "type": "market",
                            "custom_feature_enabled": True,
                        }
                    )
                )

                async def heartbeat():
                    try:
                        while not cancel.is_set():
                            await asyncio.sleep(10)
                            await ws.send("PING")
                    except Exception:
                        pass

                hb = asyncio.create_task(heartbeat())
                try:
                    async for raw in ws:
                        if cancel.is_set():
                            break
                        if raw == "PONG" or not raw:
                            continue
                        if DEBUG:
                            print(f"  [CLOB] {raw[:200]}")

                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        msgs = (
                            payload if isinstance(payload, list) else [payload]
                        )
                        for msg in msgs:
                            if not isinstance(msg, dict):
                                continue
                            etype = msg.get("event_type")
                            asset_id = msg.get("asset_id")
                            side = token_map.get(asset_id)

                            if etype == "book" and side:
                                book_map[side].on_snapshot(
                                    msg.get("bids", []),
                                    msg.get("asks", []),
                                )

                            elif etype == "price_change":
                                for ch in msg.get("price_changes", []):
                                    s = token_map.get(ch.get("asset_id"))
                                    if s:
                                        book_map[s].on_price_change(
                                            ch["price"],
                                            ch["size"],
                                            ch["side"],
                                        )

                            elif etype == "last_trade_price" and side:
                                trade = {
                                    "price": float(msg.get("price", 0)),
                                    "size": float(msg.get("size", 0)),
                                    "side": msg.get("side"),
                                }
                                trade_state[f"last_trade_{side}"] = trade

                finally:
                    hb.cancel()

        except Exception as exc:
            if cancel.is_set():
                return
            if DEBUG:
                print(f"  [CLOB] {type(exc).__name__}: {exc}")
            await asyncio.sleep(min(backoff, 30))
            backoff = min(backoff * 2, 60)


# ── RTDS WebSocket (shared across timeframes) ───────────────────────────────
async def rtds_ws(cancel: asyncio.Event, config: MarketConfig):
    """Persistent RTDS websocket — shared across all timeframes."""
    backoff = 2

    while not cancel.is_set():
        try:
            async with websockets.connect(
                RTDS_WS, ssl=SSL_CTX, ping_interval=None
            ) as ws:
                backoff = 2
                await ws.send(
                    json.dumps(
                        {
                            "action": "subscribe",
                            "subscriptions": [
                                {
                                    "topic": "crypto_prices_chainlink",
                                    "type": "*",
                                }
                            ],
                        }
                    )
                )

                async def heartbeat():
                    try:
                        while not cancel.is_set():
                            await asyncio.sleep(5)
                            await ws.send("PING")
                    except Exception:
                        pass

                hb = asyncio.create_task(heartbeat())
                try:
                    async for raw in ws:
                        if cancel.is_set():
                            break
                        if raw == "PONG" or not raw:
                            continue

                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        payload = msg.get("payload", {})
                        symbol = payload.get("symbol")
                        if symbol is None or symbol != config.chainlink_symbol:
                            continue

                        # Initial batch — buffer all entries
                        data_arr = payload.get("data")
                        if isinstance(data_arr, list) and data_arr:
                            for entry in data_arr:
                                p = entry.get("value")
                                ts_ms = entry.get("timestamp")
                                if p is not None:
                                    shared_price["chainlink_price"] = float(p)
                                    if ts_ms is not None:
                                        shared_price["price_history"].append(
                                            (int(ts_ms), float(p))
                                        )
                            continue

                        # Single streaming update
                        price = payload.get("value")
                        ts_ms = payload.get("timestamp")
                        if price is not None:
                            shared_price["chainlink_price"] = float(price)
                            if ts_ms is not None:
                                shared_price["price_history"].append(
                                    (int(ts_ms), float(price))
                                )

                finally:
                    hb.cancel()

        except Exception as exc:
            if cancel.is_set():
                return
            if DEBUG:
                print(f"  [RTDS] {type(exc).__name__}: {exc}")
            await asyncio.sleep(min(backoff, 30))
            backoff = min(backoff * 2, 60)


# ── Snapshot sampler (per-window) ────────────────────────────────────────────
async def sampler(
    rows: list[dict],
    book_up: OrderBook, book_down: OrderBook,
    meta: dict, trade_state: dict,
    window_start_price_ref: list,
    cancel: asyncio.Event,
    interval: float = 1.0,
):
    """Append a snapshot row every `interval` seconds."""
    while not cancel.is_set():
        await asyncio.sleep(interval)
        if cancel.is_set():
            break

        # Snap and reset Binance trade flow accumulator atomically
        s = binance_trade_state
        bv_buy  = s["acc_buy_vol"]
        bv_sell = s["acc_sell_vol"]
        bv_n    = s["acc_n"]
        bv_vnum = s["acc_vwap_num"]
        s["acc_buy_vol"]  = 0.0
        s["acc_sell_vol"] = 0.0
        s["acc_n"]        = 0
        s["acc_vwap_num"] = 0.0
        s["snap_buy_vol"]  = bv_buy
        s["snap_sell_vol"] = bv_sell
        s["snap_n"]        = bv_n
        s["snap_vwap"] = (
            round(bv_vnum / (bv_buy + bv_sell), 6)
            if (bv_buy + bv_sell) > 0 else None
        )

        rows.append(build_row(
            book_up, book_down, meta,
            trade_state.get("last_trade_up"),
            trade_state.get("last_trade_down"),
            window_start_price_ref[0],
        ))


# ── Parquet flush ────────────────────────────────────────────────────────────
def flush_parquet(rows: list[dict], slug: str, config: MarketConfig):
    if not rows:
        return None
    out_dir = DATA_DIR / config.data_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    path = out_dir / f"{slug}.parquet"

    # Append if file already exists (e.g., recorder restarted mid-window)
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)
        except Exception:
            pass

    # Write to temp then rename for atomic flush
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False, engine="pyarrow")
    tmp.replace(path)
    return path


# ── Window lifecycle (per-timeframe) ─────────────────────────────────────────
async def run_window(config: MarketConfig):
    """Run a single recording window for one timeframe."""
    label = config.display_name

    print(f"  [{label}] Searching for active market...")
    event, market = find_market(config)

    if not event or not market:
        print(f"  [{label}] No market found. Retrying in 30s...")
        await asyncio.sleep(30)
        return

    outcomes = _ensure_list(market["outcomes"])
    tokens = _ensure_list(market["clobTokenIds"])
    # Case-insensitive outcome lookup
    outcomes_lower = [o.lower() for o in outcomes]
    try:
        up_idx = outcomes_lower.index("up")
    except ValueError:
        up_idx = outcomes_lower.index("yes") if "yes" in outcomes_lower else 0
    try:
        down_idx = outcomes_lower.index("down")
    except ValueError:
        down_idx = outcomes_lower.index("no") if "no" in outcomes_lower else 1
    up_token = tokens[up_idx]
    down_token = tokens[down_idx]

    end = datetime.fromisoformat(market["endDate"].replace("Z", "+00:00"))
    start = datetime.fromisoformat(
        market["eventStartTime"].replace("Z", "+00:00")
    )
    slug = event["slug"]

    # Per-window state (not shared)
    book_up = OrderBook()
    book_down = OrderBook()
    trade_state = {"last_trade_up": None, "last_trade_down": None}

    meta = {
        "market_slug": slug,
        "condition_id": market.get("conditionId", ""),
        "token_id_up": up_token,
        "token_id_down": down_token,
        "window_start_ms": int(start.timestamp() * 1000),
        "window_end_ms": int(end.timestamp() * 1000),
    }

    print(f"  [{label}] Recording: {event['title']}")
    print(f"  [{label}] Window: {start.strftime('%H:%M:%S')} -> "
          f"{end.strftime('%H:%M:%S')} UTC")

    # Wait until eventStartTime + 5s so RTDS buffer has the start price
    now = datetime.now(timezone.utc)
    target = start + timedelta(seconds=5)
    wait_s = (target - now).total_seconds()
    if 0 < wait_s <= 120:
        print(f"  [{label}] Waiting {wait_s:.0f}s for start price...")
        await asyncio.sleep(wait_s)

    # Look up exact Chainlink price at eventStartTime from RTDS buffer
    start_ts_ms = int(start.timestamp() * 1000)
    start_price_exact = None
    price_history = shared_price["price_history"]
    if price_history:
        best_entry = None
        best_diff = float("inf")
        for ts_ms, px in price_history:
            diff = abs(ts_ms - start_ts_ms)
            if diff < best_diff:
                best_diff = diff
                best_entry = (ts_ms, px)
        if best_entry and best_diff <= 2000:
            start_price_exact = best_entry[1]

    # Use a mutable ref so sampler sees updates
    window_start_price_ref = [None]
    if start_price_exact is not None:
        window_start_price_ref[0] = start_price_exact
        offset_ms = best_entry[0] - start_ts_ms
        print(f"  [{label}] Start: ${start_price_exact:,.2f} "
              f"(Chainlink @ eventStart{offset_ms:+d}ms)")
    else:
        cp = shared_price["chainlink_price"]
        window_start_price_ref[0] = cp
        if cp:
            print(f"  [{label}] Start: ${cp:,.2f} (RTDS fallback)")
        else:
            print(f"  [{label}] Start: waiting for RTDS...")

    rows: list[dict] = []
    cancel = asyncio.Event()

    tasks = [
        asyncio.create_task(
            clob_ws(up_token, down_token, book_up, book_down,
                    trade_state, cancel)
        ),
        asyncio.create_task(
            sampler(rows, book_up, book_down, meta, trade_state,
                    window_start_price_ref, cancel)
        ),
    ]

    # Progress printer
    async def progress():
        while not cancel.is_set():
            await asyncio.sleep(30)
            if cancel.is_set():
                break
            now_t = datetime.now(timezone.utc)
            rem = max(0, (end - now_t).total_seconds())
            cp = shared_price["chainlink_price"]
            price_str = f"${cp:,.2f}" if cp else "---"
            bb = book_up.best_bid
            ba = book_up.best_ask
            book_str = f"Up {bb:.2f}/{ba:.2f}" if bb and ba else "Up ---/---"
            print(
                f"  [{label} {now_t.strftime('%H:%M:%S')}] "
                f"{len(rows):>5} rows | {rem:>5.0f}s left | "
                f"{price_str} | {book_str}"
            )

    # Periodic flusher
    async def periodic_flush():
        while not cancel.is_set():
            await asyncio.sleep(60)
            if rows:
                flush_parquet(rows, slug, config)

    tasks.append(asyncio.create_task(progress()))
    tasks.append(asyncio.create_task(periodic_flush()))

    # Wait until window ends + 5s grace
    now = datetime.now(timezone.utc)
    try:
        await asyncio.sleep(max(0, (end - now).total_seconds()) + 5)
    except asyncio.CancelledError:
        pass
    finally:
        cancel.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        path = flush_parquet(rows, slug, config)
        if path:
            print(f"  [{label}] Saved {len(rows)} rows -> {path}")
        else:
            print(f"  [{label}] No rows captured.")


async def _window_loop(config: MarketConfig):
    """Infinite loop recording windows for one timeframe."""
    while True:
        await run_window(config)


async def main(configs: list[MarketConfig]):
    """Run all timeframe recorders concurrently with shared RTDS."""
    base_config = configs[0]

    # Derive asset name for Deribit (btc/eth/sol/xrp) from chainlink_symbol
    # chainlink_symbol is e.g. "btc/usd" → asset = "btc"
    asset = base_config.chainlink_symbol.split("/")[0].lower()

    shared_cancel = asyncio.Event()

    # Persistent shared feeds
    shared_tasks = [
        asyncio.create_task(rtds_ws(shared_cancel, base_config)),
        asyncio.create_task(
            binance_trade_ws(base_config.binance_symbol, shared_cancel)
        ),
        asyncio.create_task(deribit_ws(asset, shared_cancel)),
    ]

    try:
        # Run all timeframe loops concurrently
        await asyncio.gather(
            *[_window_loop(c) for c in configs]
        )
    finally:
        shared_cancel.set()
        for t in shared_tasks:
            t.cancel()
        await asyncio.gather(*shared_tasks, return_exceptions=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record Polymarket market data to parquet"
    )
    parser.add_argument(
        "--market",
        default=DEFAULT_MARKET,
        choices=list(MARKET_CONFIGS),
        help="Market to record — 'btc' records both 15m+5m (default: btc)",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of book levels to capture (default: 5)",
    )
    args = parser.parse_args()
    DEBUG = args.debug
    TOP_N = args.top_n

    paired = get_paired_configs(args.market)
    configs = [config for _, config in paired]
    names = ", ".join(c.display_name for c in configs)
    print(f"  Recording: {names}")

    try:
        asyncio.run(main(configs))
    except KeyboardInterrupt:
        print("\n  Exiting.")
