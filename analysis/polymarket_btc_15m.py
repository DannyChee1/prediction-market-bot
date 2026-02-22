#!/usr/bin/env python3
"""
Polymarket BTC 15-Minute Market Monitor

Connects to:
  1. CLOB Market WS (wss://ws-subscriptions-clob.polymarket.com/ws/market)
     -> book snapshots + live best_bid_ask / price_change updates
  2. RTDS WS (wss://ws-live-data.polymarket.com)
     -> Chainlink BTC/USD price stream

Gamma API is used once per window solely to resolve the current
market's token IDs (the CLOB WS requires assets_ids to subscribe).

Usage:
    python polymarket_btc_15m.py
    python polymarket_btc_15m.py --debug
"""

import argparse
import asyncio
import json
import ssl
import sys
from datetime import datetime, timezone, timedelta

import requests
import websockets

# ── SSL ──────────────────────────────────────────────────────────────────────
try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    SSL_CTX.check_hostname = False
    SSL_CTX.verify_mode = ssl.CERT_NONE

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Endpoints ────────────────────────────────────────────────────────────────
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
RTDS_WS = "wss://ws-live-data.polymarket.com"

DEBUG = False

# ── State ────────────────────────────────────────────────────────────────────
state = {
    "up_best_bid": None,
    "up_best_ask": None,
    "down_best_bid": None,
    "down_best_ask": None,
    "btc_price": None,
    "market_title": None,
    "window_start": None,
    "window_end": None,
    "up_token_id": None,
    "down_token_id": None,
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def _to_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _ensure_list(val):
    """Gamma API returns some fields as JSON strings; parse them."""
    if isinstance(val, str):
        return json.loads(val)
    return val


def _best_from_book(levels: list, side: str):
    """Best bid = max price, best ask = min price."""
    if not levels:
        return None
    prices = [float(l["price"]) for l in levels]
    return str(max(prices)) if side == "bid" else str(min(prices))


# ── Market discovery (Gamma API, once per window) ────────────────────────────
def _try_slug(slug: str):
    try:
        resp = requests.get(
            f"{GAMMA_API}/events", params={"slug": slug}, timeout=10
        )
        data = resp.json()
        if data:
            return data[0], data[0]["markets"][0]
    except Exception:
        pass
    return None


def find_market():
    """
    Resolve the current BTC 15-min market.
    Slug convention: btc-updown-15m-{unix_ts_of_window_start}
    """
    now = datetime.now(timezone.utc)
    minute = (now.minute // 15) * 15
    window_start = now.replace(minute=minute, second=0, microsecond=0)

    for offset in [0, -15, 15, -30, 30]:
        candidate = window_start + timedelta(minutes=offset)
        ts = int(candidate.timestamp())
        result = _try_slug(f"btc-updown-15m-{ts}")
        if result:
            event, market = result
            end = datetime.fromisoformat(
                market["endDate"].replace("Z", "+00:00")
            )
            start = datetime.fromisoformat(
                market["eventStartTime"].replace("Z", "+00:00")
            )
            if now < end or start > now:
                return event, market

    # Broad fallback
    try:
        resp = requests.get(
            f"{GAMMA_API}/events",
            params={
                "active": "true",
                "closed": "false",
                "tag_slug": "up-or-down",
                "limit": 100,
            },
            timeout=15,
        )
        candidates = []
        for e in resp.json():
            if "btc-updown-15m" not in e.get("slug", ""):
                continue
            m = e["markets"][0]
            end = datetime.fromisoformat(
                m["endDate"].replace("Z", "+00:00")
            )
            start = datetime.fromisoformat(
                m["eventStartTime"].replace("Z", "+00:00")
            )
            if now < end:
                candidates.append((e, m, start))
        if candidates:
            candidates.sort(key=lambda x: x[2])
            for e, m, s in candidates:
                end = datetime.fromisoformat(
                    m["endDate"].replace("Z", "+00:00")
                )
                if s <= now < end:
                    return e, m
            return candidates[0][0], candidates[0][1]
    except Exception:
        pass

    return None, None


# ── Display ──────────────────────────────────────────────────────────────────
def display():
    lines = ["\033[2J\033[H"]
    lines.append("=" * 62)
    lines.append(f"  {state['market_title'] or 'Searching for market...'}")
    lines.append("=" * 62)
    lines.append("")

    if state["window_end"]:
        now = datetime.now(timezone.utc)
        remaining = (state["window_end"] - now).total_seconds()
        if remaining > 0:
            m, s = int(remaining // 60), int(remaining % 60)
            bar_len = 30
            filled = max(0, min(bar_len, int((remaining / 900) * bar_len)))
            lines.append(
                f"  Time Remaining:  {m:02d}:{s:02d}  "
                f"[{'#' * filled}{'-' * (bar_len - filled)}]"
            )
        else:
            lines.append(
                "  Time Remaining:  EXPIRED  (rolling to next window...)"
            )
    else:
        lines.append("  Time Remaining:  --:--")

    if state["window_start"] and state["window_end"]:
        fmt = "%H:%M:%S UTC"
        lines.append(
            f"  Window:          {state['window_start'].strftime(fmt)}"
            f" -> {state['window_end'].strftime(fmt)}"
        )
    lines.append("")

    if state["btc_price"] is not None:
        lines.append(f"  Chainlink BTC/USD:  ${state['btc_price']:>12,.2f}")
    else:
        lines.append("  Chainlink BTC/USD:     waiting...")
    lines.append("")

    def fp(p):
        if p is None:
            return "   ---   "
        try:
            return f"  {float(p):.4f}  "
        except (ValueError, TypeError):
            return f"  {str(p):>7}  "

    lines.append("  +----------+-----------+-----------+")
    lines.append("  | Outcome  |  Best Bid |  Best Ask |")
    lines.append("  +----------+-----------+-----------+")
    lines.append(
        f"  |    Up    |{fp(state['up_best_bid'])}|{fp(state['up_best_ask'])}|"
    )
    lines.append(
        f"  |   Down   |{fp(state['down_best_bid'])}|{fp(state['down_best_ask'])}|"
    )
    lines.append("  +----------+-----------+-----------+")
    lines.append("")

    up_ask = _to_float(state["up_best_ask"])
    dn_ask = _to_float(state["down_best_ask"])
    up_bid = _to_float(state["up_best_bid"])
    dn_bid = _to_float(state["down_best_bid"])

    if up_ask is not None:
        lines.append(f"  P(Up)   from ask:   {up_ask:>6.2%}")
    if dn_ask is not None:
        lines.append(f"  P(Down) from ask:   {dn_ask:>6.2%}")
    if up_ask is not None and dn_ask is not None:
        lines.append(f"  Overround (vig):    {up_ask + dn_ask - 1:>+6.2%}")
    if up_bid is not None and dn_bid is not None:
        lines.append(f"  Spread (1-bids):    {1 - up_bid - dn_bid:>+6.2%}")

    lines.append("")
    lines.append("  Ctrl+C to exit")
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


# ── CLOB Market WebSocket ────────────────────────────────────────────────────
async def clob_ws(up_token: str, down_token: str, cancel: asyncio.Event):
    """
    Subscribe to CLOB market channel for both Up/Down tokens.
    Receives: book (snapshot), price_change, best_bid_ask, last_trade_price.
    """
    token_map = {up_token: "up", down_token: "down"}
    backoff = 2

    while not cancel.is_set():
        try:
            async with websockets.connect(
                CLOB_WS, ssl=SSL_CTX, ping_interval=None
            ) as ws:
                backoff = 2

                # Subscribe per docs: assets_ids, type, custom_feature_enabled
                sub = {
                    "assets_ids": [up_token, down_token],
                    "type": "market",
                    "custom_feature_enabled": True,
                }
                await ws.send(json.dumps(sub))
                if DEBUG:
                    print(f"  [CLOB] subscribed: {up_token[:20]}... / {down_token[:20]}...")

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
                        if raw == "PONG":
                            continue
                        if not raw:
                            continue

                        if DEBUG:
                            t = datetime.now().strftime("%H:%M:%S")
                            print(f"\n  [CLOB {t}] {raw[:300]}")

                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        # WS can send a single message or an array
                        msgs = payload if isinstance(payload, list) else [payload]

                        for msg in msgs:
                            if not isinstance(msg, dict):
                                continue

                            etype = msg.get("event_type")
                            asset_id = msg.get("asset_id")
                            side = token_map.get(asset_id)

                            if etype == "book" and side:
                                bids = msg.get("bids", [])
                                asks = msg.get("asks", [])
                                state[f"{side}_best_bid"] = _best_from_book(bids, "bid")
                                state[f"{side}_best_ask"] = _best_from_book(asks, "ask")

                            elif etype == "best_bid_ask" and side:
                                state[f"{side}_best_bid"] = msg.get("best_bid")
                                state[f"{side}_best_ask"] = msg.get("best_ask")

                            elif etype == "price_change":
                                for ch in msg.get("price_changes", []):
                                    s = token_map.get(ch.get("asset_id"))
                                    if not s:
                                        continue
                                    bb = ch.get("best_bid")
                                    ba = ch.get("best_ask")
                                    if bb is not None:
                                        state[f"{s}_best_bid"] = bb
                                    if ba is not None:
                                        state[f"{s}_best_ask"] = ba

                            elif etype == "last_trade_price" and side:
                                pass  # informational only

                finally:
                    hb.cancel()

        except Exception as exc:
            if cancel.is_set():
                return
            if DEBUG:
                print(f"\n  [CLOB] error: {type(exc).__name__}: {exc}")
            await asyncio.sleep(min(backoff, 30))
            backoff = min(backoff * 2, 60)


# ── RTDS WebSocket (Chainlink BTC/USD) ──────────────────────────────────────
async def rtds_ws(cancel: asyncio.Event):
    backoff = 2

    while not cancel.is_set():
        try:
            async with websockets.connect(
                RTDS_WS, ssl=SSL_CTX, ping_interval=None
            ) as ws:
                backoff = 2
                # Subscribe unfiltered -- the filtered variant sends one
                # batch then goes silent; unfiltered streams continuously.
                # We filter for btc/usd client-side.
                sub = {
                    "action": "subscribe",
                    "subscriptions": [
                        {
                            "topic": "crypto_prices_chainlink",
                            "type": "*",
                        }
                    ],
                }
                await ws.send(json.dumps(sub))

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
                        if DEBUG:
                            t = datetime.now().strftime("%H:%M:%S")
                            print(f"\n  [RTDS {t}] {raw[:300]}")

                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        payload = msg.get("payload", {})

                        # Batch format: payload.data = [{timestamp, value}, ...]
                        data_arr = payload.get("data")
                        if isinstance(data_arr, list) and data_arr:
                            price = data_arr[-1].get("value")
                            if price is not None:
                                state["btc_price"] = float(price)
                            continue

                        # Single update: payload.value
                        price = payload.get("value")
                        if price is not None:
                            state["btc_price"] = float(price)

                finally:
                    hb.cancel()

        except Exception as exc:
            if cancel.is_set():
                return
            if DEBUG:
                print(f"\n  [RTDS] error: {type(exc).__name__}: {exc}")
            await asyncio.sleep(min(backoff, 30))
            backoff = min(backoff * 2, 60)


# ── Ticker ───────────────────────────────────────────────────────────────────
async def ticker(cancel: asyncio.Event):
    while not cancel.is_set():
        display()
        await asyncio.sleep(1)


# ── Window lifecycle ─────────────────────────────────────────────────────────
async def run_window():
    print("  Searching for active BTC 15-minute market...")
    event, market = find_market()

    if not event or not market:
        print("  No active BTC 15-minute market found. Retrying in 30s...")
        await asyncio.sleep(30)
        return

    # Parse token IDs -- Gamma API returns these as JSON strings
    outcomes = _ensure_list(market["outcomes"])
    tokens = _ensure_list(market["clobTokenIds"])
    up_token = tokens[outcomes.index("Up")]
    down_token = tokens[outcomes.index("Down")]

    end = datetime.fromisoformat(market["endDate"].replace("Z", "+00:00"))
    start = datetime.fromisoformat(
        market["eventStartTime"].replace("Z", "+00:00")
    )

    state.update(
        {
            "up_best_bid": None,
            "up_best_ask": None,
            "down_best_bid": None,
            "down_best_ask": None,
            "market_title": event["title"],
            "window_start": start,
            "window_end": end,
            "up_token_id": up_token,
            "down_token_id": down_token,
        }
    )
    display()

    cancel = asyncio.Event()
    tasks = [
        asyncio.create_task(clob_ws(up_token, down_token, cancel)),
        asyncio.create_task(rtds_ws(cancel)),
        asyncio.create_task(ticker(cancel)),
    ]

    # Run until window expires + 5s grace
    now = datetime.now(timezone.utc)
    await asyncio.sleep(max(0, (end - now).total_seconds()) + 5)

    cancel.set()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def main():
    while True:
        await run_window()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Polymarket BTC 15-min market monitor"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print raw websocket messages"
    )
    args = parser.parse_args()
    DEBUG = args.debug

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  Exiting.")
