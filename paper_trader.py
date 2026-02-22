#!/usr/bin/env python3
"""
Paper Trading Mode for Polymarket 15-Min Bot

Connects the DiffusionSignal from backtest.py to real-time market data
via WebSocket feeds, simulating trades without placing real orders.

Usage:
    python paper_trader.py                  # BTC (default)
    python paper_trader.py --market eth     # ETH
    python paper_trader.py --bankroll 5000
    python paper_trader.py --latency 250 --slippage 0.002
    python paper_trader.py --resume
    python paper_trader.py --debug
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import ssl
import sys
import time as _time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import requests
import websockets

from recorder import OrderBook
from backtest import (
    Snapshot, Decision, Fill, TradeResult,
    DiffusionSignal, walk_book, poly_fee, BacktestEngine,
)
from market_config import MarketConfig, MARKET_CONFIGS, DEFAULT_MARKET, get_config

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

# ── Endpoints & paths ────────────────────────────────────────────────────────
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
RTDS_WS = "wss://ws-live-data.polymarket.com"

TRADES_LOG = Path("paper/paper_trades.jsonl")   # overridden per-market in main()
STATE_FILE = Path("paper/paper_state.json")    # overridden per-market in main()

DEBUG = False


# ── Helpers (small stateless, copied from polymarket_btc_15m.py) ─────────────

def _ensure_list(val):
    """Gamma API returns some fields as JSON strings; parse them."""
    if isinstance(val, str):
        return json.loads(val)
    return val


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


def find_market(config: MarketConfig):
    """Resolve the current 15-min market for the given config."""
    now = datetime.now(timezone.utc)
    minute = (now.minute // 15) * 15
    window_start = now.replace(minute=minute, second=0, microsecond=0)

    for offset in [0, -15, 15, -30, 30]:
        candidate = window_start + timedelta(minutes=offset)
        ts = int(candidate.timestamp())
        result = _try_slug(f"{config.slug_prefix}-{ts}")
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
            if config.slug_prefix not in e.get("slug", ""):
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


def poll_market_resolution(slug: str, max_attempts: int = 12,
                           delay: float = 5.0) -> int | None:
    """
    Poll the Gamma API for actual market resolution.
    Returns 1 if Up won, 0 if Down won, None if resolution not available.
    """
    for attempt in range(max_attempts):
        try:
            resp = requests.get(
                f"{GAMMA_API}/events", params={"slug": slug}, timeout=10
            )
            data = resp.json()
            if not data:
                continue
            market = data[0]["markets"][0]
            if not market.get("closed"):
                if DEBUG:
                    print(f"  [RESOLVE] attempt {attempt + 1}: not closed yet")
                _time.sleep(delay)
                continue

            outcomes = _ensure_list(market["outcomes"])
            outcome_prices = _ensure_list(market["outcomePrices"])
            up_idx = outcomes.index("Up")
            up_price = float(outcome_prices[up_idx])

            # After resolution: winning side has price=1, losing has price=0
            outcome_up = 1 if up_price > 0.5 else 0
            if DEBUG:
                print(
                    f"  [RESOLVE] {slug}: outcomes={outcomes} "
                    f"prices={outcome_prices} -> "
                    f"{'UP' if outcome_up else 'DOWN'}"
                )
            return outcome_up
        except Exception as exc:
            if DEBUG:
                print(f"  [RESOLVE] attempt {attempt + 1} error: {exc}")
            _time.sleep(delay)

    return None


# ── Bridge: live OrderBook -> backtest Snapshot ──────────────────────────────

def snapshot_from_live(
    book_up: OrderBook,
    book_down: OrderBook,
    btc_price: float | None,
    window_start_price: float | None,
    window_end: datetime,
    market_slug: str,
) -> Snapshot | None:
    """Convert live OrderBook state + price data into a backtest Snapshot."""
    if btc_price is None or window_start_price is None:
        return None

    now = datetime.now(timezone.utc)
    time_remaining_s = max(0.0, (window_end - now).total_seconds())

    bb_up = book_up.best_bid
    ba_up = book_up.best_ask
    bb_down = book_down.best_bid
    ba_down = book_down.best_ask

    # Need at least asks to evaluate signal
    if ba_up is None or ba_down is None:
        return None

    return Snapshot(
        ts_ms=int(_time.time() * 1000),
        market_slug=market_slug,
        time_remaining_s=time_remaining_s,
        chainlink_price=btc_price,
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


# ── Paper Trade Tracker ─────────────────────────────────────────────────────

class PaperTradeTracker:
    """Core state management for simulated trading."""

    def __init__(
        self,
        signal: DiffusionSignal,
        bankroll: float,
        latency_ms: int = 0,
        slippage: float = 0.0,
        cooldown_ms: int = 30_000,
    ):
        self.signal = signal
        self.bankroll = bankroll
        self.latency_ms = latency_ms
        self.slippage = slippage
        self.cooldown_ms = cooldown_ms

        self.ctx: dict = {}
        self.pending_fills: list[Fill] = []
        self.all_results: list[TradeResult] = []
        self.last_fill_ts_ms: int = 0
        self.last_decision: Decision = Decision("FLAT", 0.0, 0.0, "initializing")
        self.pending_order: tuple[int, Decision] | None = None

        # Session stats
        self.windows_seen: int = 0
        self.windows_traded: int = 0
        self.total_fees: float = 0.0
        self.peak_bankroll: float = bankroll
        self.max_drawdown: float = 0.0
        self.max_dd_pct: float = 0.0

    def new_window(self, window_end: datetime):
        """Reset per-window state."""
        self.ctx = {}
        self.pending_fills = []
        self.pending_order = None
        self.last_fill_ts_ms = 0
        self.last_decision = Decision("FLAT", 0.0, 0.0, "new window")
        self.windows_seen += 1

    def evaluate(self, snapshot: Snapshot) -> Decision:
        """Called every 1s: handle latency queue, enforce cooldown, run signal."""
        # Execute pending order after latency
        if self.pending_order is not None:
            exec_ts, decision = self.pending_order
            if snapshot.ts_ms >= exec_ts:
                self._try_fill(snapshot, decision)
                self.pending_order = None

        # Cooldown between bets
        if (self.last_fill_ts_ms > 0
                and snapshot.ts_ms - self.last_fill_ts_ms < self.cooldown_ms):
            self.last_decision = Decision(
                "FLAT", 0.0, 0.0,
                f"cooldown ({(snapshot.ts_ms - self.last_fill_ts_ms) / 1000:.0f}s"
                f" / {self.cooldown_ms / 1000:.0f}s)")
            return self.last_decision

        # Run signal
        decision = self.signal.decide(snapshot, self.ctx)
        self.last_decision = decision

        if decision.action != "FLAT" and decision.size_usd > 0:
            if self.latency_ms <= 0:
                self._try_fill(snapshot, decision)
            else:
                self.pending_order = (
                    snapshot.ts_ms + self.latency_ms, decision
                )

        return decision

    def _try_fill(self, snapshot: Snapshot, decision: Decision):
        """Simulate a fill using walk_book against live L2 book."""
        if decision.action == "BUY_UP":
            side = "UP"
            ask_levels = snapshot.ask_levels_up
            best_ask = snapshot.best_ask_up
        elif decision.action == "BUY_DOWN":
            side = "DOWN"
            ask_levels = snapshot.ask_levels_down
            best_ask = snapshot.best_ask_down
        else:
            return

        if not ask_levels or best_ask is None or best_ask <= 0:
            return

        eff_est = best_ask + poly_fee(best_ask) + self.slippage
        if eff_est <= 0 or eff_est >= 1.0:
            return

        desired_shares = decision.size_usd / eff_est
        filled, total_cost, avg_price = walk_book(
            ask_levels, desired_shares, self.slippage
        )

        if filled <= 0 or total_cost <= 0:
            return

        # Don't fill if we can't afford it
        if total_cost > self.bankroll:
            return

        # Compute average raw price for fee reporting
        raw_total = 0.0
        temp = 0.0
        for px, sz in ask_levels:
            take = min(sz, desired_shares - temp)
            if take <= 0:
                break
            raw_total += take * px
            temp += take
        raw_avg = raw_total / temp if temp > 0 else 0
        fee_avg = avg_price - raw_avg - self.slippage

        # Expected price range from signal context
        rng = self.ctx.get("_expected_range", {})

        fill = Fill(
            market_slug=snapshot.market_slug,
            side=side,
            entry_ts_ms=snapshot.ts_ms,
            time_remaining_s=snapshot.time_remaining_s,
            entry_price=avg_price,
            fee_per_share=fee_avg,
            shares=filled,
            cost_usd=total_cost,
            signal_name=self.signal.name,
            decision_reason=decision.reason,
            btc_at_fill=rng.get("btc_at_fill", snapshot.chainlink_price),
            start_price=rng.get("start_price", snapshot.window_start_price),
            expected_low=rng.get("expected_low", 0.0),
            expected_high=rng.get("expected_high", 0.0),
        )

        self.pending_fills.append(fill)
        self.bankroll -= total_cost
        self.total_fees += fee_avg * filled
        self.last_fill_ts_ms = snapshot.ts_ms
        self.signal.bankroll = self.bankroll

        # Log to JSONL
        self._log_jsonl({
            "type": "fill",
            "ts": datetime.now(timezone.utc).isoformat(),
            "market_slug": fill.market_slug,
            "side": fill.side,
            "entry_price": round(fill.entry_price, 6),
            "shares": round(fill.shares, 2),
            "cost_usd": round(fill.cost_usd, 2),
            "fee_per_share": round(fill.fee_per_share, 6),
            "time_remaining_s": round(fill.time_remaining_s, 1),
            "btc_at_fill": round(fill.btc_at_fill, 2),
            "reason": fill.decision_reason,
            "bankroll_after": round(self.bankroll, 2),
        })

    def resolve_window(
        self, slug: str, final_btc: float | None,
        window_start_price: float | None,
    ):
        """At window close: poll Gamma API for actual resolution, compute PnL."""
        if not self.pending_fills:
            return

        # Poll Gamma API for the real outcome
        print("\n  Polling Gamma API for market resolution...")
        outcome_up = poll_market_resolution(slug)

        if outcome_up is None:
            # Fallback: compute from our own prices (with warning)
            if final_btc is not None and window_start_price is not None:
                outcome_up = 1 if final_btc >= window_start_price else 0
                print(
                    f"  WARNING: API resolution unavailable, using local "
                    f"prices (start=${window_start_price:,.2f} "
                    f"final=${final_btc:,.2f})"
                )
            else:
                # Can't resolve at all — return cost to bankroll
                print("  ERROR: Cannot resolve window, returning cost")
                for fill in self.pending_fills:
                    self.bankroll += fill.cost_usd
                self.pending_fills = []
                return

        self.windows_traded += 1

        window_pnl = 0.0
        n_fills = len(self.pending_fills)

        for fill in self.pending_fills:
            result = BacktestEngine._resolve_fill(
                fill, outcome_up, final_btc or 0.0
            )
            self.all_results.append(result)
            self.bankroll += result.payout
            window_pnl += result.pnl

            self._log_jsonl({
                "type": "resolution",
                "ts": datetime.now(timezone.utc).isoformat(),
                "market_slug": fill.market_slug,
                "side": fill.side,
                "outcome": "UP" if outcome_up else "DOWN",
                "source": "gamma_api",
                "entry_price": round(fill.entry_price, 6),
                "shares": round(fill.shares, 2),
                "payout": round(result.payout, 2),
                "pnl": round(result.pnl, 2),
                "pnl_pct": round(result.pnl_pct, 4),
                "final_btc": round(final_btc, 2) if final_btc else None,
                "bankroll_after": round(self.bankroll, 2),
            })

        # Update drawdown tracking
        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll
        dd = self.peak_bankroll - self.bankroll
        dd_pct = dd / self.peak_bankroll if self.peak_bankroll > 0 else 0
        if dd > self.max_drawdown:
            self.max_drawdown = dd
            self.max_dd_pct = dd_pct

        self.signal.bankroll = self.bankroll

        # Print window summary
        outcome_str = "UP" if outcome_up else "DOWN"
        btc_str = f"${final_btc:,.2f}" if final_btc else "N/A"
        print(f"  Window resolved: {outcome_str} | BTC {btc_str}")
        for r in self.all_results[-n_fills:]:
            tag = "WON" if r.pnl > 0 else "LOST"
            print(
                f"    {r.fill.side} @ {r.fill.entry_price:.4f} "
                f"x {r.fill.shares:.1f}sh -> {tag} ${r.pnl:+.2f}"
            )
        print(
            f"    Window PnL: ${window_pnl:+.2f} | "
            f"Bankroll: ${self.bankroll:,.2f}"
        )

        self.pending_fills = []

    def _log_jsonl(self, record: dict):
        """Append a JSON record to the trades log."""
        try:
            with open(TRADES_LOG, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def save_state(self):
        """Persist session state for restart continuity."""
        wins = [r for r in self.all_results if r.pnl > 0]
        data = {
            "bankroll": round(self.bankroll, 2),
            "windows_seen": self.windows_seen,
            "windows_traded": self.windows_traded,
            "total_trades": len(self.all_results),
            "wins": len(wins),
            "losses": len(self.all_results) - len(wins),
            "total_pnl": round(sum(r.pnl for r in self.all_results), 2),
            "total_fees": round(self.total_fees, 2),
            "peak_bankroll": round(self.peak_bankroll, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_dd_pct": round(self.max_dd_pct, 4),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    @classmethod
    def load_state(cls) -> dict | None:
        """Load saved session state."""
        if not STATE_FILE.exists():
            return None
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            return None


# ── WebSocket Handlers ───────────────────────────────────────────────────────

async def clob_ws(
    up_token: str,
    down_token: str,
    book_up: OrderBook,
    book_down: OrderBook,
    flat_state: dict,
    cancel: asyncio.Event,
):
    """CLOB market WS: book snapshots + price changes -> OrderBook objects."""
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
                if DEBUG:
                    print(
                        f"  [CLOB] subscribed: "
                        f"{up_token[:20]}... / {down_token[:20]}..."
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
                            t = datetime.now().strftime("%H:%M:%S")
                            print(f"\n  [CLOB {t}] {raw[:300]}")

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
                                bb = book_map[side].best_bid
                                ba = book_map[side].best_ask
                                flat_state[f"{side}_best_bid"] = (
                                    str(bb) if bb else None
                                )
                                flat_state[f"{side}_best_ask"] = (
                                    str(ba) if ba else None
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
                                        bb = book_map[s].best_bid
                                        ba = book_map[s].best_ask
                                        flat_state[f"{s}_best_bid"] = (
                                            str(bb) if bb else None
                                        )
                                        flat_state[f"{s}_best_ask"] = (
                                            str(ba) if ba else None
                                        )

                            elif etype == "best_bid_ask" and side:
                                flat_state[f"{side}_best_bid"] = msg.get(
                                    "best_bid"
                                )
                                flat_state[f"{side}_best_ask"] = msg.get(
                                    "best_ask"
                                )

                finally:
                    hb.cancel()

        except Exception as exc:
            if cancel.is_set():
                return
            if DEBUG:
                print(f"\n  [CLOB] error: {type(exc).__name__}: {exc}")
            await asyncio.sleep(min(backoff, 30))
            backoff = min(backoff * 2, 60)


async def rtds_ws(price_state: dict, cancel: asyncio.Event, config: MarketConfig):
    """RTDS WS: Chainlink price stream for the configured market."""
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

                        # Filter for configured symbol only
                        if payload.get("symbol") not in (config.chainlink_symbol, None):
                            continue

                        # Batch format: payload.data = [{value, ...}, ...]
                        data_arr = payload.get("data")
                        if isinstance(data_arr, list) and data_arr:
                            price = data_arr[-1].get("value")
                            if price is not None:
                                price_state["price"] = float(price)
                                if price_state["window_start_price"] is None:
                                    price_state["window_start_price"] = (
                                        float(price)
                                    )
                            continue

                        # Single update: payload.value
                        price = payload.get("value")
                        if price is not None:
                            price_state["price"] = float(price)
                            if price_state["window_start_price"] is None:
                                price_state["window_start_price"] = (
                                    float(price)
                                )

                finally:
                    hb.cancel()

        except Exception as exc:
            if cancel.is_set():
                return
            if DEBUG:
                print(f"\n  [RTDS] error: {type(exc).__name__}: {exc}")
            await asyncio.sleep(min(backoff, 30))
            backoff = min(backoff * 2, 60)


# ── Signal Ticker ────────────────────────────────────────────────────────────

async def signal_ticker(
    tracker: PaperTradeTracker,
    book_up: OrderBook,
    book_down: OrderBook,
    price_state: dict,
    window_end: datetime,
    market_slug: str,
    cancel: asyncio.Event,
    skip_trading: bool = False,
):
    """Run signal evaluation every 1 second (matches recorder's 1Hz sampling)."""
    while not cancel.is_set():
        await asyncio.sleep(1)
        if cancel.is_set():
            break

        snap = snapshot_from_live(
            book_up, book_down,
            price_state.get("price"),
            price_state.get("window_start_price"),
            window_end, market_slug,
        )
        if snap is not None and not skip_trading:
            tracker.evaluate(snap)


# ── Display ──────────────────────────────────────────────────────────────────

def render_display(
    tracker: PaperTradeTracker,
    price_state: dict,
    flat_state: dict,
    market_title: str,
    window_start: datetime,
    window_end: datetime,
    config: MarketConfig,
):
    """Render terminal TUI with market data and paper trading overlay."""
    lines = ["\033[2J\033[H"]
    lines.append("=" * 62)
    lines.append(f"  {config.display_name} Up/Down 15m: {market_title}")
    lines.append("=" * 62)
    lines.append("")

    now = datetime.now(timezone.utc)
    remaining = (window_end - now).total_seconds()

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
            "  Time Remaining:  EXPIRED  (resolving...)"
        )

    lines.append(
        f"  Window:          {window_start.strftime('%H:%M:%S UTC')}"
        f" -> {window_end.strftime('%H:%M:%S UTC')}"
    )
    lines.append("")

    btc = price_state.get("price")
    start_px = price_state.get("window_start_price")
    price_label = f"Chainlink {config.chainlink_symbol.upper()}"
    if btc is not None:
        lines.append(f"  {price_label}:  ${btc:>12,.2f}")
        if start_px is not None:
            delta = btc - start_px
            lines.append(
                f"  Start Price:{' ' * (len(price_label) - 11)}${start_px:>12,.2f}"
                f"  (delta: ${delta:+,.2f})"
            )
    else:
        lines.append(f"  {price_label}:     waiting...")
    lines.append("")

    # Book table
    def fp(key):
        v = flat_state.get(key)
        if v is None:
            return "   ---   "
        try:
            return f"  {float(v):.4f}  "
        except (ValueError, TypeError):
            return f"  {str(v):>7}  "

    lines.append("  +----------+-----------+-----------+")
    lines.append("  | Outcome  |  Best Bid |  Best Ask |")
    lines.append("  +----------+-----------+-----------+")
    lines.append(
        f"  |    Up    |{fp('up_best_bid')}|{fp('up_best_ask')}|"
    )
    lines.append(
        f"  |   Down   |{fp('down_best_bid')}|{fp('down_best_ask')}|"
    )
    lines.append("  +----------+-----------+-----------+")
    lines.append("")

    # ── Paper Trading section ──
    lines.append("  -- Paper Trading (DiffusionSignal) " + "-" * 25)

    dec = tracker.last_decision
    status = dec.action if dec.action != "FLAT" else "FLAT"
    lines.append(f"  Bankroll: ${tracker.bankroll:,.2f}  |  Status: {status}")
    lines.append(f"  Reason:   {dec.reason[:60]}")

    # Price history info
    hist = tracker.ctx.get("price_history", [])
    hist_len = len(hist)
    vol_str = ""
    if hist_len >= 20:
        recent = hist[-20:]
        log_ret = [
            math.log(recent[i] / recent[i - 1])
            for i in range(1, len(recent))
            if recent[i - 1] > 0 and recent[i] > 0
        ]
        if len(log_ret) >= 2:
            vol_str = f"  |  Vol(20s): {float(np.std(log_ret, ddof=1)):.2e}"
    lines.append(f"  History:  {hist_len}s{vol_str}")
    lines.append("")

    # Current window fills
    if tracker.pending_fills:
        for fill in tracker.pending_fills:
            rem_m = int(fill.time_remaining_s) // 60
            rem_s = int(fill.time_remaining_s) % 60
            lines.append(
                f"  This Window:  {fill.side} @ {fill.entry_price:.4f} "
                f"x {fill.shares:.1f}sh [{rem_m}:{rem_s:02d} left]"
            )
    else:
        lines.append("  This Window:  no trades")

    # Last result
    if tracker.all_results:
        r = tracker.all_results[-1]
        tag = "WON" if r.pnl > 0 else "LOST"
        lines.append(
            f"  Last Result:  {r.fill.side} @ {r.fill.entry_price:.4f} "
            f"x {r.fill.shares:.1f}sh -> {tag} ${r.pnl:+.2f}"
        )
    lines.append("")

    # Session stats
    wins = [r for r in tracker.all_results if r.pnl > 0]
    total = len(tracker.all_results)
    total_pnl = sum(r.pnl for r in tracker.all_results)
    win_count = len(wins)
    if total > 0:
        win_str = f"{win_count}/{total} ({win_count / total:.0%})"
    else:
        win_str = "---"
    lines.append(
        f"  Session:  {tracker.windows_traded}/{tracker.windows_seen}"
        f" windows traded  |  Win: {win_str}"
    )
    lines.append(
        f"            PnL: ${total_pnl:+,.2f}"
        f"  |  Fees: ${tracker.total_fees:.2f}"
        f"  |  DD: ${tracker.max_drawdown:.0f} ({tracker.max_dd_pct:.1%})"
    )
    lines.append("")
    lines.append("  Ctrl+C to exit")

    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


async def display_ticker(
    tracker: PaperTradeTracker,
    price_state: dict,
    flat_state: dict,
    market_title: str,
    window_start: datetime,
    window_end: datetime,
    cancel: asyncio.Event,
    config: MarketConfig,
):
    """Update display every 1 second."""
    while not cancel.is_set():
        render_display(
            tracker, price_state, flat_state,
            market_title, window_start, window_end, config,
        )
        await asyncio.sleep(1)


# ── Window Lifecycle ─────────────────────────────────────────────────────────

async def run_window(tracker: PaperTradeTracker, config: MarketConfig,
                     price_state: dict):
    """Run one 15-minute market window.

    price_state is a shared dict continuously updated by the persistent RTDS
    websocket, ensuring accurate start prices across window transitions.
    """
    print(f"  Searching for active {config.display_name} 15-minute market...")
    event, market = find_market(config)

    if not event or not market:
        print(f"  No active {config.display_name} 15-minute market found. Retrying in 30s...")
        await asyncio.sleep(30)
        return

    # Parse token IDs
    outcomes = _ensure_list(market["outcomes"])
    tokens = _ensure_list(market["clobTokenIds"])
    up_token = tokens[outcomes.index("Up")]
    down_token = tokens[outcomes.index("Down")]

    end = datetime.fromisoformat(market["endDate"].replace("Z", "+00:00"))
    start = datetime.fromisoformat(
        market["eventStartTime"].replace("Z", "+00:00")
    )
    slug = event["slug"]
    title = event["title"]

    # Initialize per-window state
    book_up = OrderBook()
    book_down = OrderBook()

    # Snapshot the current live RTDS price as this window's start price
    current_price = price_state.get("price")
    price_state["window_start_price"] = current_price

    flat_state = {
        "up_best_bid": None,
        "up_best_ask": None,
        "down_best_bid": None,
        "down_best_ask": None,
    }

    tracker.new_window(end)

    # Detect if we're joining mid-window on startup — start price will be wrong
    now_check = datetime.now(timezone.utc)
    elapsed_since_start = (now_check - start).total_seconds()
    skip_trading = (tracker.windows_seen == 1 and elapsed_since_start > 10)
    if skip_trading:
        tracker.last_decision = Decision(
            "FLAT", 0.0, 0.0,
            f"WARM-UP: joined {elapsed_since_start:.0f}s into window, feeds warming up"
        )
        print(f"  [WARM-UP] Joined {elapsed_since_start:.0f}s after window start — "
              f"skipping trading, warming up feeds for next window")

    print(f"  Market:   {title}")
    print(
        f"  Window:   {start.strftime('%H:%M:%S')}"
        f" -> {end.strftime('%H:%M:%S')} UTC"
    )
    if current_price is not None:
        print(f"  Start:    ${current_price:,.2f} (live RTDS)")
    else:
        print(f"  Start:    waiting for RTDS...")
    print(f"  Bankroll: ${tracker.bankroll:,.2f}")

    cancel = asyncio.Event()
    tasks = [
        asyncio.create_task(
            clob_ws(up_token, down_token, book_up, book_down,
                     flat_state, cancel)
        ),
        # RTDS runs persistently in run() — NOT per-window
        asyncio.create_task(
            signal_ticker(tracker, book_up, book_down, price_state,
                          end, slug, cancel,
                          skip_trading=skip_trading)
        ),
        asyncio.create_task(
            display_ticker(tracker, price_state, flat_state,
                           title, start, end, cancel, config)
        ),
    ]

    # Run until window expires + 5s grace
    now = datetime.now(timezone.utc)
    await asyncio.sleep(max(0, (end - now).total_seconds()) + 5)

    cancel.set()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # Resolve window using actual Gamma API resolution
    tracker.resolve_window(
        slug,
        price_state.get("price"),
        price_state.get("window_start_price"),
    )
    tracker.save_state()


async def run(tracker: PaperTradeTracker, config: MarketConfig):
    """Main loop: persistent RTDS + per-window trading."""
    price_state: dict = {"price": None, "window_start_price": None}

    rtds_cancel = asyncio.Event()
    rtds_task = asyncio.create_task(
        rtds_ws(price_state, rtds_cancel, config)
    )

    try:
        while True:
            await run_window(tracker, config, price_state)
    finally:
        rtds_cancel.set()
        rtds_task.cancel()
        await asyncio.gather(rtds_task, return_exceptions=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    global DEBUG, TRADES_LOG, STATE_FILE

    parser = argparse.ArgumentParser(
        description="Paper trading for Polymarket Up/Down 15-min markets"
    )
    parser.add_argument(
        "--market", default=DEFAULT_MARKET,
        choices=list(MARKET_CONFIGS),
        help="Market to paper-trade (default: btc)",
    )
    parser.add_argument(
        "--bankroll", type=float, default=10_000.0,
        help="Starting bankroll in USD (default: 10000)",
    )
    parser.add_argument(
        "--latency", type=int, default=0,
        help="Simulated order latency in ms (default: 0)",
    )
    parser.add_argument(
        "--slippage", type=float, default=0.0,
        help="Simulated slippage per share (default: 0.0)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from saved state file",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print raw WebSocket messages",
    )
    args = parser.parse_args()
    DEBUG = args.debug

    config = get_config(args.market)
    # Per-market state files to avoid clobber when running BTC + ETH simultaneously
    TRADES_LOG = Path(f"paper/paper_trades_{config.data_subdir}.jsonl")
    STATE_FILE = Path(f"paper/paper_state_{config.data_subdir}.json")

    bankroll = args.bankroll
    saved = None
    if args.resume:
        saved = PaperTradeTracker.load_state()
        if saved:
            bankroll = saved["bankroll"]
            print(
                f"  Resumed from paper_state.json: "
                f"bankroll=${bankroll:,.2f}"
            )
            print(
                f"  Previous session: {saved.get('total_trades', 0)} trades, "
                f"PnL=${saved.get('total_pnl', 0):+,.2f}"
            )
        else:
            print("  No saved state found, starting fresh.")

    signal = DiffusionSignal(bankroll=bankroll, slippage=args.slippage)
    tracker = PaperTradeTracker(
        signal=signal,
        bankroll=bankroll,
        latency_ms=args.latency,
        slippage=args.slippage,
    )

    # Restore session counters if resuming
    if saved:
        tracker.windows_seen = saved.get("windows_seen", 0)
        tracker.windows_traded = saved.get("windows_traded", 0)
        tracker.total_fees = saved.get("total_fees", 0.0)
        tracker.peak_bankroll = saved.get("peak_bankroll", bankroll)
        tracker.max_drawdown = saved.get("max_drawdown", 0.0)
        tracker.max_dd_pct = saved.get("max_dd_pct", 0.0)

    print(f"\n  Paper Trading Mode -- {config.display_name} -- DiffusionSignal")
    print(
        f"  Bankroll: ${bankroll:,.2f}  |  "
        f"Latency: {args.latency}ms  |  Slippage: {args.slippage}"
    )
    print(f"  Trades log: {TRADES_LOG}  |  State: {STATE_FILE}")
    print()

    try:
        asyncio.run(run(tracker, config))
    except KeyboardInterrupt:
        tracker.save_state()
        print(f"\n  Session saved. Bankroll: ${tracker.bankroll:,.2f}")
        print("  Exiting.")


if __name__ == "__main__":
    main()
