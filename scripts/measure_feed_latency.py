#!/usr/bin/env python3
"""
F1 Phase 0a — Feed latency measurement campaign.

Standalone, read-only instrumentation that captures per-feed timing
data without touching the live trader. Run alongside the live bot
for 24-48h to gather honest p50/p99/p99.9 staleness statistics.

What it captures
================

For each feed, every received message gets a row:
  {ts_local_ms, feed, server_event_ms, gap_to_prev_ms, age_ms, price}

- ts_local_ms     : monotonic-corrected local recv time
- server_event_ms : server-side timestamp (RTDS payload.timestamp;
                    None for feeds without one)
- gap_to_prev_ms  : ms since the previous message on the same feed
- age_ms          : current local now - server_event_ms (the real
                    staleness metric we want to use as a gate)
- price           : the value, just for sanity

Output goes to feed_latency.jsonl in the working directory, rotated
daily. Use scripts/analyze_feed_latency.py to compute the histogram
summary after a campaign.

Usage
-----

    uv run python scripts/measure_feed_latency.py --symbol btc/usd
    uv run python scripts/measure_feed_latency.py --symbol btc/usd --binance btcusdt
    uv run python scripts/measure_feed_latency.py --duration 86400  # 24h

This is read-only — does NOT affect the live trader. Spawn it in a
separate terminal alongside `live_trader.py --market btc`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import ssl
import sys
import time as _time
from datetime import datetime, timezone
from pathlib import Path

try:
    import websockets
except ImportError:
    print("ERROR: pip install websockets")
    sys.exit(1)

RTDS_WS = "wss://ws-live-data.polymarket.com"
BINANCE_WS = "wss://data-stream.binance.vision/ws/{symbol}@bookTicker"
COINBASE_WS = "wss://ws-feed.exchange.coinbase.com"
KRAKEN_WS = "wss://ws.kraken.com/v2"

SSL_CTX = ssl.create_default_context()


def now_ms() -> int:
    return int(_time.time() * 1000)


def write_record(out: Path, rec: dict) -> None:
    """Append one JSONL row. Atomic enough for our purposes — single
    writer, no concurrent access. Don't fsync; we want low overhead."""
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ─────────────────────────────────────────────────────────────────────
# Per-feed measurement loops
# ─────────────────────────────────────────────────────────────────────


async def measure_rtds(symbol: str, out: Path, cancel: asyncio.Event):
    """RTDS Chainlink feed — has server-side payload.timestamp.
    Both rebroadcast tax (server_event_ms vs local recv) and
    inter-message gap distribution can be measured here.
    """
    backoff = 2
    last_recv = None
    while not cancel.is_set():
        try:
            async with websockets.connect(
                RTDS_WS, ssl=SSL_CTX, ping_interval=20, ping_timeout=10
            ) as ws:
                backoff = 2
                await ws.send(json.dumps({
                    "action": "subscribe",
                    "subscriptions": [{
                        "topic": "crypto_prices_chainlink",
                        "type": "*",
                    }],
                }))
                print(f"  [RTDS] connected, filtering {symbol}")
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
                    if payload.get("symbol") != symbol:
                        continue

                    # Extract price + server timestamp from either
                    # data array format or direct value format.
                    price = None
                    server_ms = None
                    data_arr = payload.get("data")
                    if isinstance(data_arr, list) and data_arr:
                        last = data_arr[-1]
                        price = last.get("value")
                        server_ms = (last.get("timestamp")
                                     or payload.get("timestamp"))
                    else:
                        price = payload.get("value")
                        server_ms = payload.get("timestamp")

                    if price is None:
                        continue
                    recv = now_ms()
                    gap = (recv - last_recv) if last_recv is not None else None
                    last_recv = recv

                    age = (recv - int(server_ms)) if server_ms else None
                    write_record(out, {
                        "ts_local_ms": recv,
                        "feed": "rtds_chainlink",
                        "symbol": symbol,
                        "server_event_ms": int(server_ms) if server_ms else None,
                        "gap_to_prev_ms": gap,
                        "age_ms": age,
                        "price": float(price),
                    })
        except Exception as exc:
            print(f"  [RTDS] {type(exc).__name__}: {exc} → reconnect in {backoff}s")
            await asyncio.sleep(min(backoff, 30))
            backoff = min(backoff * 2, 60)


async def measure_binance(symbol: str, out: Path, cancel: asyncio.Event):
    """Binance @bookTicker — does NOT include event_time in the standard
    payload. We log local recv only and inter-message gap. The age_ms
    field is None for this feed.
    """
    url = BINANCE_WS.format(symbol=symbol.lower())
    backoff = 2
    last_recv = None
    while not cancel.is_set():
        try:
            async with websockets.connect(
                url, ssl=SSL_CTX, ping_interval=20, ping_timeout=10
            ) as ws:
                backoff = 2
                print(f"  [Binance] connected to {symbol}@bookTicker")
                async for raw in ws:
                    if cancel.is_set():
                        break
                    try:
                        msg = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    bid = msg.get("b")
                    ask = msg.get("a")
                    if bid is None or ask is None:
                        continue
                    try:
                        mid = (float(bid) + float(ask)) / 2.0
                    except (TypeError, ValueError):
                        continue
                    recv = now_ms()
                    gap = (recv - last_recv) if last_recv is not None else None
                    last_recv = recv
                    write_record(out, {
                        "ts_local_ms": recv,
                        "feed": "binance_bookticker",
                        "symbol": symbol,
                        "server_event_ms": None,  # not in @bookTicker payload
                        "gap_to_prev_ms": gap,
                        "age_ms": None,
                        "price": mid,
                    })
        except Exception as exc:
            print(f"  [Binance] {type(exc).__name__}: {exc} → reconnect in {backoff}s")
            await asyncio.sleep(min(backoff, 30))
            backoff = min(backoff * 2, 60)


async def measure_binance_trade(symbol: str, out: Path, cancel: asyncio.Event):
    """Binance @trade stream — DOES include `E` (event time in ms).
    Use this to measure server→here latency for Binance even though
    the bot's actual trading uses @bookTicker. Same exchange, same
    co-location, similar latency profile.
    """
    url = f"wss://data-stream.binance.vision/ws/{symbol.lower()}@trade"
    backoff = 2
    last_recv = None
    while not cancel.is_set():
        try:
            async with websockets.connect(
                url, ssl=SSL_CTX, ping_interval=20, ping_timeout=10
            ) as ws:
                backoff = 2
                print(f"  [Binance@trade] connected to {symbol}@trade")
                async for raw in ws:
                    if cancel.is_set():
                        break
                    try:
                        msg = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    event_ms = msg.get("E")
                    price = msg.get("p")
                    if event_ms is None or price is None:
                        continue
                    recv = now_ms()
                    gap = (recv - last_recv) if last_recv is not None else None
                    last_recv = recv
                    age = recv - int(event_ms)
                    write_record(out, {
                        "ts_local_ms": recv,
                        "feed": "binance_trade",
                        "symbol": symbol,
                        "server_event_ms": int(event_ms),
                        "gap_to_prev_ms": gap,
                        "age_ms": age,
                        "price": float(price),
                    })
        except Exception as exc:
            print(f"  [Binance@trade] {type(exc).__name__}: {exc} → reconnect in {backoff}s")
            await asyncio.sleep(min(backoff, 30))
            backoff = min(backoff * 2, 60)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="btc/usd",
                        help="RTDS chainlink symbol (default btc/usd)")
    parser.add_argument("--binance", default="btcusdt",
                        help="Binance bookTicker symbol (default btcusdt)")
    parser.add_argument("--no-binance-trade", action="store_true",
                        help="Skip the @trade-based Binance feed")
    parser.add_argument("--duration", type=int, default=0,
                        help="Run for N seconds (0 = forever, default)")
    parser.add_argument("--output", type=str, default="feed_latency.jsonl",
                        help="Output JSONL path")
    args = parser.parse_args()

    out = Path(args.output)
    print(f"  Writing to {out.absolute()}")
    print(f"  RTDS symbol: {args.symbol}")
    print(f"  Binance symbol: {args.binance}")
    if args.duration > 0:
        print(f"  Duration: {args.duration}s")
    else:
        print(f"  Duration: forever (Ctrl+C to stop)")
    print()

    cancel = asyncio.Event()

    async def stop_after(n: int):
        await asyncio.sleep(n)
        print(f"\n  Duration reached, stopping...")
        cancel.set()

    tasks = [
        asyncio.create_task(measure_rtds(args.symbol, out, cancel)),
        asyncio.create_task(measure_binance(args.binance, out, cancel)),
    ]
    if not args.no_binance_trade:
        tasks.append(asyncio.create_task(
            measure_binance_trade(args.binance, out, cancel)
        ))
    if args.duration > 0:
        tasks.append(asyncio.create_task(stop_after(args.duration)))

    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except KeyboardInterrupt:
        print("\n  Interrupted, shutting down...")
        cancel.set()
        await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  Done.")
