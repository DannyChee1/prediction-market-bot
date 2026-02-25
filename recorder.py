#!/usr/bin/env python3
"""
Polymarket Market Recorder

Captures 1-second snapshots to parquet for backtesting:
  - Full top-5 book depth for Up and Down outcomes
  - Chainlink price (streaming)
  - Window metadata and time remaining
  - Trade ticks (last_trade_price events)

Outputs one parquet file per window under ./data/<market>/

Run with native Windows Python (has pyarrow):
    py -3 recorder.py                  # BTC (default)
    py -3 recorder.py --market eth     # ETH
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

from market_config import MarketConfig, MARKET_CONFIGS, DEFAULT_MARKET, get_config

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


# ── Shared state ─────────────────────────────────────────────────────────────
book_up = OrderBook()
book_down = OrderBook()
chainlink_price: float | None = None
window_start_price: float | None = None
price_history: collections.deque = collections.deque(maxlen=600)  # (ts_ms, price)

# Metadata set per window
meta = {
    "market_slug": "",
    "condition_id": "",
    "token_id_up": "",
    "token_id_down": "",
    "window_start_ms": 0,
    "window_end_ms": 0,
}

# Trade tick accumulator (flushed into each snapshot row)
last_trade_up: dict | None = None
last_trade_down: dict | None = None


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
def build_row() -> dict:
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
        "chainlink_price": chainlink_price,
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
    global last_trade_up, last_trade_down
    for label, lt in [("up", last_trade_up), ("down", last_trade_down)]:
        if lt:
            row[f"last_trade_px_{label}"] = lt.get("price")
            row[f"last_trade_sz_{label}"] = lt.get("size")
            row[f"last_trade_side_{label}"] = lt.get("side")
        else:
            row[f"last_trade_px_{label}"] = None
            row[f"last_trade_sz_{label}"] = None
            row[f"last_trade_side_{label}"] = None

    return row


# ── CLOB WebSocket ───────────────────────────────────────────────────────────
async def clob_ws(up_token: str, down_token: str, cancel: asyncio.Event):
    global last_trade_up, last_trade_down
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
                                if side == "up":
                                    last_trade_up = trade
                                else:
                                    last_trade_down = trade

                finally:
                    hb.cancel()

        except Exception as exc:
            if cancel.is_set():
                return
            if DEBUG:
                print(f"  [CLOB] {type(exc).__name__}: {exc}")
            await asyncio.sleep(min(backoff, 30))
            backoff = min(backoff * 2, 60)


# ── RTDS WebSocket ───────────────────────────────────────────────────────────
async def rtds_ws(cancel: asyncio.Event, config: MarketConfig):
    """Persistent RTDS websocket — runs across windows, buffers (ts_ms, price)."""
    global chainlink_price
    backoff = 2

    while not cancel.is_set():
        try:
            async with websockets.connect(
                RTDS_WS, ssl=SSL_CTX, ping_interval=None
            ) as ws:
                backoff = 2
                # Unfiltered subscription streams continuously;
                # filtered goes silent after first batch.
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

                        # Initial batch — buffer all entries with timestamps
                        data_arr = payload.get("data")
                        if isinstance(data_arr, list) and data_arr:
                            for entry in data_arr:
                                p = entry.get("value")
                                ts_ms = entry.get("timestamp")
                                if p is not None:
                                    chainlink_price = float(p)
                                    if ts_ms is not None:
                                        price_history.append(
                                            (int(ts_ms), float(p))
                                        )
                            continue

                        # Single streaming update
                        price = payload.get("value")
                        ts_ms = payload.get("timestamp")
                        if price is not None:
                            chainlink_price = float(price)
                            if ts_ms is not None:
                                price_history.append(
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


# ── Snapshot sampler ─────────────────────────────────────────────────────────
async def sampler(
    rows: list[dict], cancel: asyncio.Event, interval: float = 1.0
):
    """Append a snapshot row every `interval` seconds."""
    while not cancel.is_set():
        await asyncio.sleep(interval)
        if cancel.is_set():
            break
        rows.append(build_row())


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
            pass  # corrupted file, overwrite it

    # Write to temp then rename for atomic flush
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False, engine="pyarrow")
    tmp.replace(path)
    return path


# ── Window lifecycle ─────────────────────────────────────────────────────────
async def run_window(config: MarketConfig):
    global chainlink_price, window_start_price
    global last_trade_up, last_trade_down

    print(f"  Searching for active {config.display_name} market...")
    event, market = find_market(config)

    if not event or not market:
        print(f"  No {config.display_name} market found. Retrying in 30s...")
        await asyncio.sleep(30)
        return

    outcomes = _ensure_list(market["outcomes"])
    tokens = _ensure_list(market["clobTokenIds"])
    up_token = tokens[outcomes.index("Up")]
    down_token = tokens[outcomes.index("Down")]

    end = datetime.fromisoformat(market["endDate"].replace("Z", "+00:00"))
    start = datetime.fromisoformat(
        market["eventStartTime"].replace("Z", "+00:00")
    )
    slug = event["slug"]

    # Reset per-window state
    book_up.__init__()
    book_down.__init__()
    window_start_price = None
    last_trade_up = None
    last_trade_down = None

    meta.update(
        {
            "market_slug": slug,
            "condition_id": market.get("conditionId", ""),
            "token_id_up": up_token,
            "token_id_down": down_token,
            "window_start_ms": int(start.timestamp() * 1000),
            "window_end_ms": int(end.timestamp() * 1000),
        }
    )

    print(f"  Recording: {event['title']}")
    print(f"  Window:    {start.strftime('%H:%M:%S')} -> {end.strftime('%H:%M:%S')} UTC")
    print(f"  Slug:      {slug}")

    # Wait until eventStartTime + 5s so RTDS buffer has the start price
    now = datetime.now(timezone.utc)
    target = start + timedelta(seconds=5)
    wait_s = (target - now).total_seconds()
    if 0 < wait_s <= 120:
        print(f"  Waiting {wait_s:.0f}s for start price...")
        await asyncio.sleep(wait_s)

    # Look up exact Chainlink price at eventStartTime from RTDS buffer
    # (matches live_trader.py logic — Polymarket's "Price to Beat")
    start_ts_ms = int(start.timestamp() * 1000)
    start_price_exact = None
    if price_history:
        best_entry = None
        best_diff = float("inf")
        for ts_ms, px in price_history:
            diff = abs(ts_ms - start_ts_ms)
            if diff < best_diff:
                best_diff = diff
                best_entry = (ts_ms, px)
        if best_entry and best_diff <= 2000:  # within 2 seconds
            start_price_exact = best_entry[1]

    if start_price_exact is not None:
        window_start_price = start_price_exact
        offset_ms = best_entry[0] - start_ts_ms
        print(f"  Start price: ${start_price_exact:,.2f} (Chainlink @ eventStart{offset_ms:+d}ms)")
    else:
        window_start_price = chainlink_price
        if chainlink_price:
            print(f"  Start price: ${chainlink_price:,.2f} (RTDS fallback)")
        else:
            print("  Start price: waiting for RTDS...")

    rows: list[dict] = []
    cancel = asyncio.Event()

    # RTDS is persistent (started in main), only start CLOB + sampler here
    tasks = [
        asyncio.create_task(clob_ws(up_token, down_token, cancel)),
        asyncio.create_task(sampler(rows, cancel)),
    ]

    # Progress printer
    async def progress():
        while not cancel.is_set():
            await asyncio.sleep(10)
            if cancel.is_set():
                break
            now = datetime.now(timezone.utc)
            rem = max(0, (end - now).total_seconds())
            bb = book_up.best_bid
            ba = book_up.best_ask
            price_str = f"${chainlink_price:,.2f}" if chainlink_price else "---"
            book_str = (
                f"Up {bb:.2f}/{ba:.2f}" if bb and ba else "Up ---/---"
            )
            print(
                f"  [{now.strftime('%H:%M:%S')}] "
                f"{len(rows):>5} rows | {rem:>6.1f}s left | "
                f"{price_str} | {book_str}"
            )

    # Periodic flusher — write every 60s so we don't lose data on crash
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

        # Final flush
        path = flush_parquet(rows, slug, config)
        if path:
            print(f"  Saved {len(rows)} rows -> {path}")
        else:
            print("  No rows captured.")


_current_window_task: asyncio.Task | None = None


async def main(config: MarketConfig):
    global _current_window_task

    # Start persistent RTDS websocket (runs across windows)
    rtds_cancel = asyncio.Event()
    rtds_task = asyncio.create_task(rtds_ws(rtds_cancel, config))

    try:
        while True:
            _current_window_task = asyncio.current_task()
            await run_window(config)
    finally:
        rtds_cancel.set()
        rtds_task.cancel()
        await asyncio.gather(rtds_task, return_exceptions=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record Polymarket market data to parquet"
    )
    parser.add_argument(
        "--market",
        default=DEFAULT_MARKET,
        choices=list(MARKET_CONFIGS),
        help="Market to record (default: btc)",
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

    _config = get_config(args.market)
    print(f"  Market: {_config.display_name} ({_config.slug_prefix})")

    try:
        asyncio.run(main(_config))
    except KeyboardInterrupt:
        # run_window's finally block handles the flush
        print("\n  Exiting.")
