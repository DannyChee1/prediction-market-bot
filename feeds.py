"""WebSocket feeds: CLOB order book and RTDS Chainlink price streams."""

from __future__ import annotations

import asyncio
import collections
import json
import time as _time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import websockets

from market_api import SSL_CTX
from market_config import MarketConfig
from recorder import OrderBook
from backtest import Snapshot

if TYPE_CHECKING:
    from tracker import LiveTradeTracker

# ── WS endpoints ─────────────────────────────────────────────────────────────
CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
RTDS_WS = "wss://ws-live-data.polymarket.com"


# ── Live snapshot builder ────────────────────────────────────────────────────

def snapshot_from_live(
    book_up: OrderBook, book_down: OrderBook,
    price: float | None, window_start_price: float | None,
    window_end: datetime, market_slug: str,
) -> Snapshot | None:
    if price is None or window_start_price is None:
        return None

    now = datetime.now(timezone.utc)
    time_remaining_s = max(0.0, (window_end - now).total_seconds())

    ba_up = book_up.best_ask
    ba_down = book_down.best_ask

    if ba_up is None or ba_down is None:
        return None

    bb_up = book_up.best_bid
    bb_down = book_down.best_bid

    return Snapshot(
        ts_ms=int(_time.time() * 1000),
        market_slug=market_slug,
        time_remaining_s=time_remaining_s,
        chainlink_price=price,
        window_start_price=window_start_price,
        best_bid_up=bb_up,
        best_ask_up=ba_up,
        best_bid_down=bb_down,
        best_ask_down=ba_down,
        size_bid_up=book_up.bids.get(bb_up) if bb_up else None,
        size_ask_up=book_up.asks.get(ba_up) if ba_up else None,
        size_bid_down=book_down.bids.get(bb_down) if bb_down else None,
        size_ask_down=book_down.asks.get(ba_down) if ba_down else None,
        ask_levels_up=tuple(book_up.top_asks(5)),
        ask_levels_down=tuple(book_down.top_asks(5)),
        bid_levels_up=tuple(book_up.top_bids(5)),
        bid_levels_down=tuple(book_down.top_bids(5)),
    )


# ── CLOB order book WebSocket ────────────────────────────────────────────────

async def clob_ws(
    up_token: str, down_token: str,
    book_up: OrderBook, book_down: OrderBook,
    flat_state: dict, cancel: asyncio.Event,
    debug: bool = False,
    trade_state: dict | None = None,
):
    token_map = {up_token: "up", down_token: "down"}
    book_map = {"up": book_up, "down": book_down}
    backoff = 2

    while not cancel.is_set():
        try:
            async with websockets.connect(
                CLOB_WS, ssl=SSL_CTX, ping_interval=None
            ) as ws:
                backoff = 2
                await ws.send(json.dumps({
                    "assets_ids": [up_token, down_token],
                    "type": "market",
                    "custom_feature_enabled": True,
                }))

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
                        if debug:
                            t = datetime.now().strftime("%H:%M:%S")
                            print(f"\n  [CLOB {t}] {raw[:300]}")

                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        msgs = payload if isinstance(payload, list) else [payload]
                        for msg in msgs:
                            if not isinstance(msg, dict):
                                continue
                            etype = msg.get("event_type")
                            asset_id = msg.get("asset_id")
                            side = token_map.get(asset_id)

                            if etype == "book" and side:
                                book_map[side].on_snapshot(
                                    msg.get("bids", []), msg.get("asks", []))
                                bb = book_map[side].best_bid
                                ba = book_map[side].best_ask
                                flat_state[f"{side}_best_bid"] = str(bb) if bb else None
                                flat_state[f"{side}_best_ask"] = str(ba) if ba else None

                            elif etype == "price_change":
                                for ch in msg.get("price_changes", []):
                                    s = token_map.get(ch.get("asset_id"))
                                    if s:
                                        book_map[s].on_price_change(
                                            ch["price"], ch["size"], ch["side"])
                                        bb = book_map[s].best_bid
                                        ba = book_map[s].best_ask
                                        flat_state[f"{s}_best_bid"] = str(bb) if bb else None
                                        flat_state[f"{s}_best_ask"] = str(ba) if ba else None

                            elif etype == "last_trade_price" and trade_state is not None:
                                try:
                                    size = float(msg.get("size", 0))
                                    trade_side = msg.get("side", "").upper()
                                    if size > 0 and trade_side in ("BUY", "SELL"):
                                        bar = trade_state["current_bar"]
                                        now_ts = _time.time()
                                        if bar["start_ts"] == 0:
                                            bar["start_ts"] = now_ts
                                        if trade_side == "BUY":
                                            bar["buy_vol"] += size
                                        else:
                                            bar["sell_vol"] += size
                                        # Rotate bar when elapsed >= bar_duration_s
                                        bar_dur = trade_state.get("bar_duration_s", 60.0)
                                        if now_ts - bar["start_ts"] >= bar_dur:
                                            trade_state["bars"].append(
                                                (bar["buy_vol"], bar["sell_vol"])
                                            )
                                            bar["buy_vol"] = 0.0
                                            bar["sell_vol"] = 0.0
                                            bar["start_ts"] = now_ts
                                except (TypeError, ValueError, KeyError):
                                    pass

                finally:
                    hb.cancel()

        except Exception as exc:
            if cancel.is_set():
                return
            if debug:
                print(f"\n  [CLOB] error: {type(exc).__name__}: {exc}")
            await asyncio.sleep(min(backoff, 30))
            backoff = min(backoff * 2, 60)


# ── RTDS Chainlink price WebSocket ───────────────────────────────────────────

async def rtds_ws(price_state: dict, cancel: asyncio.Event,
                  config: MarketConfig, tracker: "LiveTradeTracker",
                  debug: bool = False):
    backoff = 2
    STALE_DATA_TIMEOUT = 60

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
                print(f"  [RTDS] connected, filtering for {config.chainlink_symbol}")

                async def heartbeat():
                    try:
                        while not cancel.is_set():
                            await asyncio.sleep(5)
                            await ws.send("PING")
                    except Exception:
                        pass

                async def stale_watchdog():
                    try:
                        while not cancel.is_set():
                            await asyncio.sleep(10)
                            if tracker.last_price_update_ts > 0:
                                age = _time.time() - tracker.last_price_update_ts
                                if age > STALE_DATA_TIMEOUT:
                                    print(f"\n  [RTDS] stale data watchdog: "
                                          f"no update for {age:.0f}s, forcing reconnect")
                                    await ws.close()
                                    return
                    except Exception:
                        pass

                hb = asyncio.create_task(heartbeat())
                wd = asyncio.create_task(stale_watchdog())
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

                        data_arr = payload.get("data")
                        if isinstance(data_arr, list) and data_arr:
                            p = data_arr[-1].get("value")
                            ts_ms = data_arr[-1].get("timestamp") or payload.get("timestamp")
                            if p is not None:
                                price_state["price"] = float(p)
                                tracker.last_price_update_ts = _time.time()
                                if ts_ms is not None:
                                    price_state["price_history"].append((int(ts_ms), float(p)))
                                if price_state["window_start_price"] is None:
                                    price_state["window_start_price"] = float(p)
                            continue

                        p = payload.get("value")
                        ts_ms = payload.get("timestamp")
                        if p is not None:
                            price_state["price"] = float(p)
                            tracker.last_price_update_ts = _time.time()
                            if ts_ms is not None:
                                price_state["price_history"].append((int(ts_ms), float(p)))
                            if price_state["window_start_price"] is None:
                                price_state["window_start_price"] = float(p)

                finally:
                    hb.cancel()
                    wd.cancel()

        except Exception as exc:
            if cancel.is_set():
                return
            print(f"\n  [RTDS] reconnecting: {type(exc).__name__}: {exc}")
            await asyncio.sleep(min(backoff, 30))
            backoff = min(backoff * 2, 60)
