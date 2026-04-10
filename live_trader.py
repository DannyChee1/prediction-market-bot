#!/usr/bin/env python3
"""
Live Trading Bot for Polymarket Up/Down Markets (multi-timeframe)

Runs both 15m and 5m windows concurrently in a single process, sharing
one PriceFeed, one BinanceFeed, and one display thread.

Usage:
    py -3 live_trader.py                          # BTC 15m + 5m
    py -3 live_trader.py --market eth              # ETH 15m + 5m
    py -3 live_trader.py --market sol              # SOL 15m + 5m
    py -3 live_trader.py --market xrp              # XRP 15m + 5m
    py -3 live_trader.py --market btc_5m           # BTC 5m only
    py -3 live_trader.py --bankroll 500            # smaller bankroll
    py -3 live_trader.py --dry-run                 # signal + log, no real orders
    py -3 live_trader.py --debug
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import os
import sys
import threading
import time as _time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from web3 import Web3

from polybot_core import OrderClient, BookFeed, PriceFeed, BinanceFeed, UserFeed

from backtest import (
    Decision, DiffusionSignal, build_calibration_table,
)
from market_config import (
    MARKET_CONFIGS, DEFAULT_MARKET, get_config, get_paired_configs,
)
from market_api import (
    find_market, _ensure_list,
)
from feeds import snapshot_from_book_feed
from display import render_display
from recording import record_sampler
from tracker import LiveTradeTracker
from redemption import POLYGON_RPC

FAST_PRICE_POLL_S = 0.005
FAST_BINANCE_POLL_S = 0.005
FAST_BOOK_POLL_S = 0.005
FAST_SIGNAL_IDLE_S = 0.01
FAST_SIGNAL_MIN_INTERVAL_S = 0.005


# ── Signal Ticker ────────────────────────────────────────────────────────────

async def signal_ticker(
    tracker: LiveTradeTracker,
    book_feed: BookFeed,
    price_state: dict, window_end: datetime,
    market_slug: str, up_token: str, down_token: str,
    cancel: asyncio.Event,
    skip_trading: bool = False,
    trade_state: dict | None = None,
    binance_state: dict | None = None,
    book_state: dict | None = None,
    window_start_price: float | None = None,
    wake_event: asyncio.Event | None = None,
    wake_state: dict | None = None,
    signal_idle_s: float = 0.05,
    signal_min_interval_s: float = 0.025,
):
    last_seen = (-1, -1, -1)
    last_eval_ms = 0
    while not cancel.is_set():
        timed_out = False
        if wake_event is not None:
            try:
                await asyncio.wait_for(wake_event.wait(), timeout=signal_idle_s)
            except asyncio.TimeoutError:
                timed_out = True
            wake_event.clear()
        else:
            await asyncio.sleep(signal_idle_s)
            timed_out = True
        if cancel.is_set():
            break

        now_ms = int(_time.time() * 1000)
        sig = (
            int(price_state.get("seq", 0) or 0),
            int(binance_state.get("seq", 0) or 0) if binance_state is not None else 0,
            int(book_state.get("seq", 0) or 0) if book_state is not None else 0,
        )
        if not timed_out and sig == last_seen:
            continue
        if (not timed_out
                and signal_min_interval_s > 0
                and last_eval_ms > 0
                and (now_ms - last_eval_ms) < int(signal_min_interval_s * 1000)):
            continue
        last_seen = sig
        eval_start_ms = int(_time.time() * 1000)

        try:
            # Clear previous error on successful tick start
            tracker.ctx.pop("_signal_error", None)
            tracker.ctx["_signal_trigger_ts_ms"] = int(
                (wake_state or {}).get("trigger_wall_ts_ms", eval_start_ms)
            )
            tracker.ctx["_signal_trigger_source"] = (wake_state or {}).get("trigger_source")
            tracker.ctx["_signal_trigger_age_ms"] = max(
                0.0,
                eval_start_ms - float((wake_state or {}).get("trigger_wall_ts_ms", eval_start_ms)),
            )
            tracker.ctx["_signal_trigger_feed_age_ms"] = max(
                0.0,
                eval_start_ms - float((wake_state or {}).get("trigger_feed_ts_ms", eval_start_ms)),
            )
            cl_ts = float(price_state.get("last_update_ts", 0.0) or 0.0)
            bn_ts = float(binance_state.get("last_update_ts", 0.0) or 0.0) if binance_state is not None else 0.0
            bk_ts = float(book_state.get("last_update_ts", 0.0) or 0.0) if book_state is not None else 0.0
            tt_ts = float(trade_state.get("last_trade_ts", 0.0) or 0.0) if trade_state is not None else 0.0
            tracker.ctx["_chainlink_age_ms"] = max(0.0, eval_start_ms - cl_ts * 1000.0) if cl_ts > 0 else None
            tracker.ctx["_binance_age_ms"] = max(0.0, eval_start_ms - bn_ts * 1000.0) if bn_ts > 0 else None
            tracker.ctx["_book_age_ms"] = max(0.0, eval_start_ms - bk_ts * 1000.0) if bk_ts > 0 else None
            # Trade-tape freshness: None until the first trade of the
            # session is seen (so the gate doesn't false-fire during
            # warmup before any trades have arrived).
            tracker.ctx["_trade_tape_age_ms"] = (
                max(0.0, eval_start_ms - tt_ts * 1000.0) if tt_ts > 0 else None
            )

            # Inject Binance mid into ctx only if fresh (< 10s old).
            # Stale Binance data (WS dropped) would poison vol/z/oracle-lag.
            if binance_state is not None:
                _bn_ts = binance_state.get("last_update_ts", 0)
                if _bn_ts and (_time.time() - _bn_ts) < 10.0:
                    tracker.ctx["_binance_mid"] = binance_state.get("mid_price")
                else:
                    tracker.ctx.pop("_binance_mid", None)

            # Accumulate price history — but only when the effective
            # price actually changed. On calm markets the signal_ticker
            # wakes every 10ms via `wake_event` timeout even when no
            # feeds ticked; falling through and appending 100 duplicate
            # prices per second used to bias Yang-Zhang σ downward
            # (zero-range OHLC bars contribute zero vol). _compute_vol_deduped
            # filters consecutive duplicates already, but the 5s-bar
            # YZ path does not.
            eff_px = tracker.ctx.get("_binance_mid") or price_state.get("price")
            if eff_px is not None:
                hist = tracker.ctx.setdefault("price_history", [])
                ts_hist = tracker.ctx.setdefault("ts_history", [])
                if not hist or hist[-1] != eff_px:
                    hist.append(eff_px)
                    ts_hist.append(int(_time.time() * 1000))
                    tracker.ctx["_live_history_appended"] = True
                    # Cap history to prevent unbounded growth if a window
                    # runs longer than expected or new_window() isn't called.
                    _MAX_HIST = 2000
                    if len(hist) > _MAX_HIST:
                        del hist[:-_MAX_HIST]
                        del ts_hist[:-_MAX_HIST]

            snap = snapshot_from_book_feed(
                book_feed, up_token, down_token,
                price_state.get("price"),
                window_start_price,
                window_end, market_slug,
            )
            if snap is not None and not skip_trading:
                if trade_state is not None:
                    tracker.ctx["_trade_bars"] = trade_state["bars"]
                    tracker.ctx["_trade_total_bars"] = trade_state.get("total_bars", 0)
                    tracker.ctx["_trade_side_history"] = trade_state.get("sides", [])
                await asyncio.to_thread(tracker.evaluate, snap, up_token, down_token)
            eval_done_ms = int(_time.time() * 1000)
            tracker.ctx["_signal_eval_ms"] = eval_done_ms - eval_start_ms
            trigger_ts_ms = int(tracker.ctx.get("_signal_trigger_ts_ms", eval_start_ms) or eval_start_ms)
            tracker.ctx["_decision_total_ms"] = eval_done_ms - trigger_ts_ms
            tracker.latency_samples.append({
                "signal_trigger_age_ms": tracker.ctx.get("_signal_trigger_age_ms"),
                "signal_trigger_feed_age_ms": tracker.ctx.get("_signal_trigger_feed_age_ms"),
                "signal_eval_ms": tracker.ctx.get("_signal_eval_ms"),
                "decision_total_ms": tracker.ctx.get("_decision_total_ms"),
                "chainlink_age_ms": tracker.ctx.get("_chainlink_age_ms"),
                "binance_age_ms": tracker.ctx.get("_binance_age_ms"),
                "book_age_ms": tracker.ctx.get("_book_age_ms"),
            })
            last_eval_ms = eval_done_ms

        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            tracker.ctx["_signal_error"] = err_msg
            print(f"\n  [SIGNAL_TICKER ERROR] {err_msg}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


# ── Display Thread ───────────────────────────────────────────────────────────

def _display_thread_fn(
    price_state: dict,
    display_sections: list[dict],
    base_config,
    dry_run: bool,
    exit_enabled: bool,
    stop_event: threading.Event,
):
    """Persistent display thread — lives across windows, never killed mid-session.

    Reads from display_sections which are updated by each _window_loop.
    """
    while not stop_event.is_set():
        try:
            render_display(
                price_state,
                display_sections,
                base_config,
                dry_run=dry_run,
                exit_enabled=exit_enabled,
            )
        except Exception as exc:
            try:
                sys.stdout.write(
                    f"\033[2J\033[H\n  [DISPLAY ERROR] "
                    f"{type(exc).__name__}: {exc}\n"
                )
                sys.stdout.flush()
            except Exception:
                pass
        stop_event.wait(1.0)


# ── Rust Feed Polling ────────────────────────────────────────────────────────

def _notify_signal_wakeup(
    wake_event: asyncio.Event | None,
    wake_state: dict | None,
    source: str,
    feed_ts_s: float | None,
):
    if wake_state is not None:
        now_ms = int(_time.time() * 1000)
        wake_state["trigger_source"] = source
        wake_state["trigger_wall_ts_ms"] = now_ms
        wake_state["trigger_feed_ts_ms"] = int(feed_ts_s * 1000) if feed_ts_s else now_ms
    if wake_event is not None:
        wake_event.set()


def _make_binance_feed(symbol: str) -> tuple[BinanceFeed, str]:
    """Build BinanceFeed, preferring SBE only when the extension supports it."""
    mode = (os.getenv("BINANCE_FEED_MODE", "json") or "json").strip().lower()
    api_key = (os.getenv("BINANCE_SBE_API_KEY", "") or "").strip()
    if mode == "sbe":
        if not api_key:
            raise RuntimeError("BINANCE_SBE_API_KEY is required when BINANCE_FEED_MODE=sbe")
        try:
            return BinanceFeed(symbol, mode, api_key), "sbe"
        except TypeError:
            raise RuntimeError(
                "installed polybot_core extension is JSON-only; rebuild it before using BINANCE_FEED_MODE=sbe"
            )
    return BinanceFeed(symbol), "json"

async def _poll_price_feed(price_feed: PriceFeed, price_state: dict,
                           trackers: list[LiveTradeTracker],
                           cancel: asyncio.Event, debug: bool = False,
                           symbol: str = "",
                           wake_event: asyncio.Event | None = None,
                           wake_state: dict | None = None,
                           poll_interval_s: float = 0.02):
    """Bridge Rust PriceFeed → Python price_state dict (shared across trackers).

    Auto-restarts the Rust WS feed if the price stays stale for >60s
    (beyond the Rust-side 30s reconnect timeout).
    """
    last_ts = 0.0
    _stale_warned = False
    _STALE_RESTART_S = 60.0
    while not cancel.is_set():
        await asyncio.sleep(poll_interval_s)
        if cancel.is_set():
            break
        try:
            px = price_feed.price()
            ts = price_feed.last_update_ts()
            if px is not None and ts > last_ts:
                last_ts = ts
                _stale_warned = False
                price_state["price"] = px
                price_state["last_update_ts"] = ts
                price_state["seq"] = int(price_state.get("seq", 0) or 0) + 1
                for t in trackers:
                    t.last_price_update_ts = ts
                ts_ms = int(ts * 1000)
                price_state["price_history"].append((ts_ms, px))
                _notify_signal_wakeup(wake_event, wake_state, "chainlink", ts)
            elif last_ts > 0:
                age = _time.time() - last_ts
                if age > _STALE_RESTART_S and symbol:
                    print(f"  [PriceFeed] stale for {age:.0f}s — restarting WS")
                    price_feed = PriceFeed(symbol)
                    last_ts = 0.0
                    _stale_warned = False
                elif debug and age > 15 and not _stale_warned:
                    print(f"  [PriceFeed] WARNING: no update for {age:.0f}s "
                          f"(last_ts={last_ts:.0f}, px={px}, raw_ts={ts})")
                    _stale_warned = True
        except Exception:
            pass


async def _poll_binance_feed(binance_feed: BinanceFeed, binance_state: dict,
                             cancel: asyncio.Event, symbol: str = "",
                             wake_event: asyncio.Event | None = None,
                             wake_state: dict | None = None,
                             poll_interval_s: float = 0.02):
    """Bridge Rust BinanceFeed → Python binance_state dict.

    Auto-restarts if no update for >60s.
    """
    _last_ts = 0.0
    _STALE_RESTART_S = 60.0
    while not cancel.is_set():
        await asyncio.sleep(poll_interval_s)
        if cancel.is_set():
            break
        try:
            mid = binance_feed.mid()
            ts = binance_feed.last_update_ts()
            if mid is not None and ts > _last_ts:
                _last_ts = ts
                binance_state["mid_price"] = mid
                binance_state["last_update_ts"] = ts
                binance_state["seq"] = int(binance_state.get("seq", 0) or 0) + 1
                _notify_signal_wakeup(wake_event, wake_state, "binance", ts)
            elif _last_ts > 0:
                age = _time.time() - _last_ts
                if age > _STALE_RESTART_S and symbol:
                    print(f"  [BinanceFeed] stale for {age:.0f}s — restarting WS")
                    # Route through _make_binance_feed so SBE mode and the
                    # API-key env var are honored on reconnect (otherwise
                    # the feed silently downgrades to JSON on every restart).
                    try:
                        binance_feed, mode = _make_binance_feed(symbol)
                        print(f"  [BinanceFeed] restarted in {mode} mode")
                    except Exception as mk_exc:
                        print(f"  [BinanceFeed] restart failed: {mk_exc}; "
                              f"falling back to JSON")
                        binance_feed = BinanceFeed(symbol)
                    _last_ts = 0.0
        except Exception:
            pass


async def _poll_book_feed(book_feed: BookFeed, up_token: str, down_token: str,
                          flat_state: dict, trade_state: dict | None,
                          cancel: asyncio.Event,
                          book_state: dict | None = None,
                          wake_event: asyncio.Event | None = None,
                          wake_state: dict | None = None,
                          poll_interval_s: float = 0.02):
    """Bridge Rust BookFeed → Python flat_state + trade_state for VPIN."""
    while not cancel.is_set():
        await asyncio.sleep(poll_interval_s)
        if cancel.is_set():
            break
        try:
            latest_book_ts = 0.0
            latest_bbo = {}
            changed = False
            trade_changed = False
            # Update flat_state BBO for display thread
            for side, token in [("up", up_token), ("down", down_token)]:
                snap = book_feed.snapshot(token)
                bb = snap.best_bid
                ba = snap.best_ask
                latest_book_ts = max(latest_book_ts, float(snap.timestamp or 0.0))
                latest_bbo[f"{side}_best_bid"] = bb
                latest_bbo[f"{side}_best_ask"] = ba
                prev_bb = flat_state.get(f"{side}_best_bid")
                prev_ba = flat_state.get(f"{side}_best_ask")
                flat_state[f"{side}_best_bid"] = str(bb) if bb is not None else None
                flat_state[f"{side}_best_ask"] = str(ba) if ba is not None else None
                if flat_state[f"{side}_best_bid"] != prev_bb or flat_state[f"{side}_best_ask"] != prev_ba:
                    changed = True

            # Drain trade events for VPIN bar accumulation
            if trade_state is not None:
                now_ts = _time.time()
                for size, trade_side in book_feed.drain_trades():
                    changed = True
                    trade_changed = True
                    trade_state["last_trade_ts"] = now_ts
                    sides = trade_state.get("sides")
                    if sides is not None and trade_side in ("BUY", "SELL"):
                        sides.append(trade_side)
                    bar = trade_state["current_bar"]
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
                        trade_state["total_bars"] = (
                            trade_state.get("total_bars", 0) + 1
                        )
                        bar["buy_vol"] = 0.0
                        bar["sell_vol"] = 0.0
                        bar["start_ts"] = now_ts
                # Time-based bar rotation: close an empty/stale bar even
                # when no trades are arriving, so VPIN doesn't end up with
                # a single 20-minute "bar" masquerading as a 60s one in
                # quiet markets.
                bar = trade_state["current_bar"]
                bar_dur = trade_state.get("bar_duration_s", 60.0)
                if bar["start_ts"] != 0 and (now_ts - bar["start_ts"]) >= bar_dur:
                    trade_state["bars"].append(
                        (bar["buy_vol"], bar["sell_vol"])
                    )
                    trade_state["total_bars"] = (
                        trade_state.get("total_bars", 0) + 1
                    )
                    bar["buy_vol"] = 0.0
                    bar["sell_vol"] = 0.0
                    bar["start_ts"] = now_ts
            if book_state is not None:
                prev_ts = float(book_state.get("last_update_ts", 0.0) or 0.0)
                prev_bbo = book_state.get("last_bbo", {})
                if latest_book_ts > prev_ts or latest_bbo != prev_bbo or trade_changed:
                    book_state["last_update_ts"] = (
                        latest_book_ts if latest_book_ts > 0 else _time.time()
                    )
                    book_state["seq"] = int(book_state.get("seq", 0) or 0) + 1
                    book_state["last_bbo"] = latest_bbo
                    changed = True
            if changed:
                _notify_signal_wakeup(
                    wake_event,
                    wake_state,
                    "book",
                    latest_book_ts or _time.time(),
                )
        except Exception:
            pass


# ── Bankroll sync ────────────────────────────────────────────────────────────

def _sync_bankroll(all_trackers: list[LiveTradeTracker],
                   resolved_tracker: LiveTradeTracker,
                   window_pnl: float):
    """After a resolution, propagate the PnL delta to all other trackers.

    Each tracker independently deducts costs from its own bankroll copy.
    Taking max() would wipe out pending cost deductions in other trackers,
    inflating the bankroll. Instead, we propagate only the PnL delta so
    each tracker's own pending deductions are preserved.
    """
    if len(all_trackers) <= 1:
        return
    for t in all_trackers:
        if t is not resolved_tracker:
            t.bankroll += window_pnl
            t.signal.bankroll = t.bankroll


# ── Window Lifecycle ─────────────────────────────────────────────────────────

async def run_window(
    tracker: LiveTradeTracker, config, price_state: dict,
    trade_state: dict | None,
    binance_state: dict | None,
    signal_wakeup: asyncio.Event,
    wake_state: dict,
    section: dict,
    pending_resolve: asyncio.Task | None = None,
    record: bool = True,
) -> tuple[str, float | None, float | None, str] | None:
    """Run a single trading window. Returns (slug, final_price, start_price,
    condition_id) for resolution, or None if no market was found."""

    # Rebuild calibration table + sigma_calibration from latest data (single pass)
    if tracker.cal_data_dir is not None:
        try:
            need_sigmas = tracker.signal.regime_z_scale
            result = build_calibration_table(
                tracker.cal_data_dir, return_sigmas=need_sigmas,
            )
            if need_sigmas:
                cal_table, sigmas = result
            else:
                cal_table = result
            tracker.signal.calibration_table = cal_table
            n_obs = sum(cal_table.counts.values())
            print(f"  [{config.display_name}] Calibration rebuilt: "
                  f"{len(cal_table.table)} cells, {n_obs} obs")

            # Update sigma_calibration from the same file scan
            if need_sigmas and sigmas:
                new_cal = max(float(np.median(sigmas[-200:])), 1e-7)
                old_cal = tracker.signal.sigma_calibration
                tracker.signal.sigma_calibration = new_cal
                if old_cal and abs(new_cal - old_cal) / old_cal > 0.05:
                    print(f"  [{config.display_name}] sigma_cal updated: "
                          f"{old_cal:.2e} -> {new_cal:.2e}")
        except Exception as exc:
            print(f"  [{config.display_name}] WARNING: calibration rebuild failed: {exc}")

    section["status"] = "searching"
    section["market_title"] = f"Searching for {config.display_name}..."
    print(f"  [{config.display_name}] Searching for active market...")
    event, market = await asyncio.to_thread(find_market, config)

    if not event or not market:
        print(f"  [{config.display_name}] No active market found. Retrying in 30s...")
        await asyncio.sleep(30)
        return None

    outcomes = _ensure_list(market["outcomes"])
    tokens = _ensure_list(market["clobTokenIds"])
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
    title = event["title"]

    tracker.condition_id = market.get("conditionId", "")
    if not tracker.condition_id:
        print(f"  [{config.display_name}] WARNING: conditionId missing")

    book_feed = BookFeed([up_token, down_token])

    # Warm up the order client's connection pool and cache tick_size/neg_risk
    # for both tokens. This saves ~100-200ms per order (2 HTTP GETs that
    # were happening on every single trade) and pre-establishes the TLS
    # connection so the first order doesn't pay a cold-start penalty.
    if hasattr(tracker, 'client') and tracker.client is not None:
        try:
            tracker.client.warmup([up_token, down_token])
        except Exception as exc:
            print(f"  [{config.display_name}] OrderClient warmup failed: {exc}")

    # UserFeed for real-time fill/cancel events (requires API creds)
    api_key = os.getenv("POLY_API_KEY", "")
    api_secret = os.getenv("POLY_API_SECRET", "")
    api_passphrase = os.getenv("POLY_PASSPHRASE", "")
    if api_key and api_secret and api_passphrase and not tracker.dry_run:
        try:
            user_feed = UserFeed(api_key, api_secret, api_passphrase,
                                 [up_token, down_token])
            tracker.user_feed = user_feed
            print(f"  [{config.display_name}] UserFeed started (WS fills enabled)")
        except Exception as exc:
            print(f"  [{config.display_name}] UserFeed failed: {exc} (using REST polling)")
            tracker.user_feed = None
    else:
        tracker.user_feed = None

    # Wait until eventStartTime + 5s so the RTDS buffer has the start price.
    # Skip the wait if we're joining mid-window (start already passed).
    now = datetime.now(timezone.utc)
    target = start + timedelta(seconds=5)
    wait_s = (target - now).total_seconds()
    if 0 < wait_s <= 120:
        section["status"] = "waiting"
        section["market_title"] = f"Waiting {wait_s:.0f}s for start..."
        print(f"  [{config.display_name}] Waiting {wait_s:.0f}s for start price...")
        await asyncio.sleep(wait_s)
    elif wait_s < 0:
        elapsed = -wait_s
        print(f"  [{config.display_name}] Joining mid-window ({elapsed:.0f}s after start)")

    # Look up exact Chainlink price at eventStartTime from RTDS buffer.
    # We want the LAST update at-or-before eventStartTime — that's the
    # price Polymarket's contract considers active at the start boundary.
    start_ts_ms = int(start.timestamp() * 1000)
    start_price_exact = None
    history = price_state.get("price_history", [])
    best_entry = None
    if history:
        for ts_ms, px in history:
            if ts_ms <= start_ts_ms:
                best_entry = (ts_ms, px)
            # history is chronological, so last match is the freshest <= start
        if best_entry and (start_ts_ms - best_entry[0]) <= 2000:
            start_price_exact = best_entry[1]

    # Per-window start price (NOT shared — each timeframe has its own)
    if start_price_exact is not None:
        window_start_price = start_price_exact
        offset_ms = best_entry[0] - start_ts_ms
        print(f"  [{config.display_name}] Start: ${start_price_exact:,.2f} "
              f"(Chainlink @ eventStart{offset_ms:+d}ms)")
    else:
        # Mid-window join: we don't have the Chainlink price from window
        # start in our buffer. Fetch the historical BTC price from Binance
        # at the exact start timestamp (1s kline).
        window_start_price = None
        if config.binance_symbol:
            try:
                import requests as _req
                kline_url = (
                    f"https://api.binance.com/api/v3/klines"
                    f"?symbol={config.binance_symbol.upper()}"
                    f"&interval=1s&startTime={start_ts_ms}&limit=1"
                )
                kline_resp = _req.get(kline_url, timeout=5).json()
                if kline_resp and len(kline_resp) > 0:
                    # kline[4] = close price of that 1-second bar
                    window_start_price = float(kline_resp[0][4])
                    print(f"  [{config.display_name}] Start: ${window_start_price:,.2f} "
                          f"(Binance historical, mid-window join)")
            except Exception as exc:
                print(f"  [{config.display_name}] Binance start price lookup failed: {exc}")
        if window_start_price is None:
            window_start_price = price_state.get("price")
            if window_start_price:
                print(f"  [{config.display_name}] Start: ${window_start_price:,.2f} (RTDS fallback — INACCURATE)")
            else:
                print(f"  [{config.display_name}] Start: waiting for RTDS...")

    flat_state = {
        "up_best_bid": None, "up_best_ask": None,
        "down_best_bid": None, "down_best_ask": None,
    }
    book_state = {"seq": 0, "last_update_ts": 0.0, "last_bbo": {}}

    # Wait for any pending resolution from previous window before trading
    # (bankroll must be settled before we size new orders)
    if pending_resolve is not None:
        await pending_resolve

    tracker.new_window(end)

    # Detect if we're joining mid-window on startup.
    # For short windows (5m/15m), skip trading on a late join because
    # the vol estimator needs time to warm up. For 1h windows, we
    # pre-populate price history from Binance (below), so we can
    # trade immediately even on a mid-window join.
    now_check = datetime.now(timezone.utc)
    elapsed_since_start = (now_check - start).total_seconds()
    is_1h_window = config.window_duration_s >= 3600
    skip_trading = (
        tracker.windows_seen == 1
        and elapsed_since_start > 10
        and not is_1h_window  # 1h windows pre-load history, no skip needed
    )
    if skip_trading:
        tracker.last_decision = Decision(
            "FLAT", 0.0, 0.0,
            f"WARM-UP: joined {elapsed_since_start:.0f}s into window"
        )
        print(f"  [{config.display_name}] WARM-UP: joined {elapsed_since_start:.0f}s in — "
              f"skipping trading")

    # Mid-window join: pre-populate price history from Binance so the vol
    # estimator can compute sigma immediately (no 2-min live warmup needed).
    vol_lookback = getattr(tracker.signal, "vol_lookback_s", 90)
    if elapsed_since_start > 30 and config.binance_symbol and is_1h_window:
        lookback = min(int(elapsed_since_start), vol_lookback * 2)
        fetch_start_ms = int((now_check.timestamp() - lookback) * 1000)
        fetch_end_ms = int(now_check.timestamp() * 1000)
        try:
            import requests as _req
            all_klines = []
            cursor = fetch_start_ms
            while cursor < fetch_end_ms:
                url = (
                    f"https://api.binance.com/api/v3/klines"
                    f"?symbol={config.binance_symbol.upper()}"
                    f"&interval=1s&startTime={cursor}&limit=1000"
                )
                batch = _req.get(url, timeout=10).json()
                if not batch:
                    break
                all_klines.extend(batch)
                cursor = int(batch[-1][0]) + 1000
                if len(batch) < 1000:
                    break
            if all_klines:
                hist = tracker.ctx.setdefault("price_history", [])
                ts_hist = tracker.ctx.setdefault("ts_history", [])
                for k in all_klines:
                    px = float(k[4])  # close price
                    ts_ms = int(k[0])
                    if not hist or hist[-1] != px:
                        hist.append(px)
                        ts_hist.append(ts_ms)
                print(f"  [{config.display_name}] Pre-loaded {len(hist)} price ticks "
                      f"from Binance ({lookback}s lookback) — ready to trade")
        except Exception as exc:
            print(f"  [{config.display_name}] Historical price fetch failed: {exc}")

    mode = "DRY RUN" if tracker.dry_run else "LIVE"
    print(f"  [{config.display_name}] [{mode}] {title}")
    print(f"  [{config.display_name}] {start.strftime('%H:%M:%S')} -> "
          f"{end.strftime('%H:%M:%S')} UTC | "
          f"Bankroll: ${tracker.bankroll:,.2f}")

    # Update display section
    section["status"] = "trading"
    section["market_title"] = title
    section["flat_state"] = flat_state
    section["window_start"] = start
    section["window_end"] = end
    section["window_start_price"] = window_start_price

    cancel = asyncio.Event()

    tasks = [
        asyncio.create_task(
            _poll_book_feed(book_feed, up_token, down_token,
                            flat_state, trade_state, cancel,
                            book_state=book_state,
                            wake_event=signal_wakeup,
                            wake_state=wake_state,
                            poll_interval_s=FAST_BOOK_POLL_S)
        ),
        asyncio.create_task(
            signal_ticker(tracker, book_feed, price_state,
                          end, slug, up_token, down_token, cancel,
                          skip_trading=skip_trading,
                          trade_state=trade_state,
                          binance_state=binance_state,
                          book_state=book_state,
                          window_start_price=window_start_price,
                          wake_event=signal_wakeup,
                          wake_state=wake_state,
                          signal_idle_s=FAST_SIGNAL_IDLE_S,
                          signal_min_interval_s=FAST_SIGNAL_MIN_INTERVAL_S)
        ),
    ]

    if record:
        rec_meta = {
            "market_slug": slug,
            "condition_id": tracker.condition_id,
            "token_id_up": up_token,
            "token_id_down": down_token,
            "window_start_ms": int(start.timestamp() * 1000),
            "window_end_ms": int(end.timestamp() * 1000),
        }
        tasks.append(asyncio.create_task(
            record_sampler(
                book_feed, up_token, down_token, rec_meta,
                price_state, window_start_price,
                slug, config.data_subdir, cancel,
                binance_state=binance_state,
            )
        ))

    now = datetime.now(timezone.utc)
    await asyncio.sleep(max(0, (end - now).total_seconds()) + 5)

    cancel.set()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    if tracker.flat_reason_counts:
        tracker._log_flat_summary()
        tracker.flat_reason_counts = {}

    # CRITICAL: do NOT use price_state.get("price") for the final price.
    # That returns the LAST chainlink update, which is up to 5 seconds
    # AFTER window end (because of the +5s sleep above). In those 5s,
    # chainlink can publish a new price that's different from what
    # Polymarket actually settled on (Polymarket uses the price AT
    # window end exactly, via the same Chainlink feed).
    #
    # Mirror the start-price logic from above: walk price_history
    # for the LAST update at-or-before end_ts_ms.
    end_ts_ms = int(end.timestamp() * 1000)
    final_price_exact = None
    history = price_state.get("price_history", [])
    if history:
        for ts_ms, px in history:
            if ts_ms <= end_ts_ms:
                final_price_exact = px
            else:
                break  # history is chronological — stop at first post-end entry
    if final_price_exact is None:
        # No in-window chainlink update — fall back to current price (rare)
        final_price_exact = price_state.get("price")
        print(f"  [{config.display_name}] WARNING: no in-window chainlink "
              f"price for resolution, using current ${final_price_exact}")

    return (slug, final_price_exact, window_start_price, tracker.condition_id)


async def _resolve_background(
    tracker: LiveTradeTracker,
    slug: str,
    final_price: float | None,
    start_price: float | None,
    condition_id: str,
    all_trackers: list[LiveTradeTracker],
):
    """Background resolution: resolve window + sync bankroll.

    Queue processing is handled by _periodic_redeem_loop — NOT here.
    This keeps resolution fast so the next window isn't blocked.
    """
    window_pnl = 0.0
    resolve_succeeded = False
    try:
        window_pnl = await asyncio.to_thread(
            tracker.resolve_window, slug, final_price, start_price, condition_id,
        ) or 0.0
        resolve_succeeded = True
    except Exception as exc:
        # CRITICAL: do NOT default to pnl=0 and sync_bankroll. Real
        # positions are still in tracker.pending_fills; if we sync now,
        # the ledger silently diverges from on-chain truth. Log loudly,
        # leave the pending fills in place, and let the next window's
        # new_window() flag them as unresolved (which it already does).
        import traceback
        tb = traceback.format_exc(limit=3)
        print(f"  [RESOLVE] FAILED for {slug}: {type(exc).__name__}: {exc}")
        print(f"  [RESOLVE] traceback: {tb}")
        print(f"  [RESOLVE] {len(tracker.pending_fills)} pending fills NOT synced")
        try:
            tracker._log({
                "type": "resolution_failed",
                "ts": datetime.now(timezone.utc).isoformat(),
                "slug": slug,
                "error": f"{type(exc).__name__}: {exc}",
                "pending_fills": len(tracker.pending_fills),
            })
        except Exception:
            pass
        # Skip the bankroll sync — leave bankroll where it was so that
        # the operator can retry / reconcile manually rather than
        # silently absorbing a phantom $0 PnL.
        tracker.save_state()
        return
    _sync_bankroll(all_trackers, resolved_tracker=tracker, window_pnl=window_pnl)
    tracker.save_state()
    # Fire-and-forget: kick off queue processing without blocking.
    # The periodic loop also processes the queue, so this is just for speed.
    if len(tracker.redemption_queue) > 0:
        asyncio.create_task(_process_queue_bg(tracker))


async def _process_queue_bg(tracker: LiveTradeTracker):
    """Process redemption queue in background (fire-and-forget)."""
    try:
        await asyncio.to_thread(tracker.process_redemption_queue)
        tracker.save_state()
    except Exception as exc:
        print(f"  [REDEEM] Background queue error: {type(exc).__name__}: {exc}")


async def _periodic_redeem_loop(
    all_trackers: list[LiveTradeTracker],
    interval_s: float = 90.0,
):
    """Periodically process redemption queues for all trackers.

    Runs independently of the trading loop so items get retried even when
    no new trades are happening. This is the PRIMARY redemption driver.
    Deduplicates by queue file path so shared queues (e.g. BTC 15m + 5m)
    are only processed once per cycle.
    """
    while True:
        await asyncio.sleep(interval_s)
        seen_queues: set[str] = set()
        for tracker in all_trackers:
            qpath = str(tracker.redemption_queue._file)
            if qpath in seen_queues:
                continue
            seen_queues.add(qpath)
            if len(tracker.redemption_queue) > 0:
                try:
                    await asyncio.to_thread(tracker.process_redemption_queue)
                    tracker.save_state()
                except Exception as exc:
                    print(f"  [REDEEM] Periodic error: {type(exc).__name__}: {exc}")


async def _window_loop(
    tracker: LiveTradeTracker,
    config,
    price_state: dict,
    binance_state: dict,
    signal_wakeup: asyncio.Event,
    wake_state: dict,
    section: dict,
    all_trackers: list[LiveTradeTracker],
    shutdown: asyncio.Event,
    record: bool = True,
):
    """Main loop for one timeframe — runs forever, handles search/trade/resolve.

    The `shutdown` event lets the main task signal a graceful exit:
    when set, the loop finishes the current window, awaits any pending
    background resolve, and returns. Without this, Ctrl-C hangs because
    asyncio.gather waits for the loop forever and the loop has no exit
    condition.
    """
    # Trade state persists across windows — VPIN needs 20+ min history
    vpin_bar_s = tracker.signal.vpin_bar_s
    trade_state: dict = {
        "bars": collections.deque(maxlen=200),
        "sides": collections.deque(maxlen=300),
        "current_bar": {"buy_vol": 0.0, "sell_vol": 0.0, "start_ts": 0},
        "bar_duration_s": vpin_bar_s,
        "total_bars": 0,
        # Wall-clock seconds of the most recent trade seen on the tape.
        # Used by signal_ticker to populate ctx["_trade_tape_age_ms"] so
        # the `max_trade_tape_age_ms` gate in signal_diffusion._check_stale_features
        # actually fires in live (it was silently inoperative for months).
        "last_trade_ts": 0.0,
    }

    pending_resolve: asyncio.Task | None = None

    while not shutdown.is_set():
        result = await run_window(
            tracker, config, price_state, trade_state,
            binance_state, signal_wakeup, wake_state, section,
            pending_resolve=pending_resolve,
            record=record,
        )

        if result is None:
            continue  # no market found, run_window already waited 30s

        slug, final_price, start_price, condition_id = result

        # ── Post-window: cancel orders, then resolve in background ──
        section["status"] = "resolving"
        section["market_title"] = f"Resolving {config.display_name}..."

        if tracker.open_orders:
            await asyncio.to_thread(tracker._cancel_open_orders)
        if tracker.open_sell_orders:
            await asyncio.to_thread(tracker._cancel_open_sell_orders)

        # Start resolution in background — next run_window will search for
        # the next market concurrently and await this before trading begins.
        # condition_id is captured NOW (before next run_window overwrites it).
        pending_resolve = asyncio.create_task(
            _resolve_background(
                tracker, slug, final_price, start_price,
                condition_id, all_trackers,
            )
        )

    # Graceful shutdown — drain any in-flight resolution before returning
    # so we don't lose state or leave pending fills mid-resolve.
    if pending_resolve is not None and not pending_resolve.done():
        try:
            print(f"  [{config.display_name}] Shutdown: awaiting pending resolve...")
            await asyncio.wait_for(pending_resolve, timeout=15.0)
        except asyncio.TimeoutError:
            print(f"  [{config.display_name}] Shutdown: pending resolve timed out")
        except Exception as exc:
            print(f"  [{config.display_name}] Shutdown: pending resolve error: {exc}")
    print(f"  [{config.display_name}] Window loop exited cleanly")


async def run(
    trackers_and_configs: list[tuple[LiveTradeTracker, str, object]],
    base_config,
    dry_run: bool,
    exit_enabled: bool,
    debug: bool,
    record: bool = True,
):
    """Run all timeframes concurrently with shared feeds and display.

    trackers_and_configs: [(tracker, config_key, config), ...]
    """
    all_trackers = [t for t, _, _ in trackers_and_configs]

    price_state: dict = {
        "price": None,
        "price_history": collections.deque(maxlen=600),
        "seq": 0,
        "last_update_ts": 0.0,
    }
    binance_state: dict = {"seq": 0, "last_update_ts": 0.0}
    signal_wakeup = asyncio.Event()
    wake_state: dict = {}

    # Shared Rust feeds — one PriceFeed + one BinanceFeed for the asset
    price_feed = PriceFeed(base_config.chainlink_symbol)
    print(f"  [PriceFeed] started (filtering for {base_config.chainlink_symbol})")

    binance_feed = None
    binance_mode = "disabled"
    if base_config.binance_symbol:
        binance_feed, binance_mode = _make_binance_feed(base_config.binance_symbol)
        print(f"  [BinanceFeed] started ({base_config.binance_symbol}, mode={binance_mode})")

    # Shared polling loops + global shutdown signal
    feed_cancel = asyncio.Event()
    shutdown = asyncio.Event()
    # Wire SIGINT/SIGTERM to set the shutdown event so window loops exit
    # cleanly. Without this, Ctrl-C asks asyncio.gather to cancel — but
    # the inner `while True:` loops never check for cancellation, so the
    # process hangs until the OS hard-kills it.
    import signal as _signal
    loop = asyncio.get_running_loop()
    def _request_shutdown():
        if not shutdown.is_set():
            print("\n  [SHUTDOWN] Signal received, finishing current windows...")
            shutdown.set()
    for sig_name in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(getattr(_signal, sig_name), _request_shutdown)
        except (NotImplementedError, RuntimeError):
            # Windows / non-main-thread case — fall back to default handler
            pass

    price_poll = asyncio.create_task(
        _poll_price_feed(price_feed, price_state, all_trackers, feed_cancel,
                         debug=debug, symbol=base_config.chainlink_symbol,
                         wake_event=signal_wakeup, wake_state=wake_state,
                         poll_interval_s=FAST_PRICE_POLL_S)
    )
    binance_poll = None
    if binance_feed is not None:
        binance_poll = asyncio.create_task(
            _poll_binance_feed(binance_feed, binance_state, feed_cancel,
                               symbol=base_config.binance_symbol,
                               wake_event=signal_wakeup,
                               wake_state=wake_state,
                               poll_interval_s=FAST_BINANCE_POLL_S)
        )

    # Display sections — one per timeframe, updated by each _window_loop
    display_sections: list[dict] = []
    for tracker, config_key, config in trackers_and_configs:
        section = {
            "tracker": tracker,
            "config": config,
            "flat_state": {
                "up_best_bid": None, "up_best_ask": None,
                "down_best_bid": None, "down_best_ask": None,
            },
            "window_start": None,
            "window_end": None,
            "market_title": "Starting...",
            "status": "searching",
        }
        display_sections.append(section)

    # Persistent display thread — never killed between windows
    display_stop = threading.Event()
    display_thread = threading.Thread(
        target=_display_thread_fn,
        args=(price_state, display_sections, base_config,
              dry_run, exit_enabled, display_stop),
        daemon=True,
    )
    display_thread.start()

    redeem_task = None
    try:
        if not dry_run:
            # Scan trade logs for any unredeemed wins (catches historical misses)
            for tracker, _, _ in trackers_and_configs:
                try:
                    tracker.scan_unredeemed_wins()
                except Exception as exc:
                    print(f"  [STARTUP] Scan error: {exc}")

            # Process any queued redemptions from previous session + scan
            # Dedup by queue file path (BTC 15m + 5m share the same queue)
            seen_startup_queues: set[str] = set()
            for tracker, _, _ in trackers_and_configs:
                qpath = str(tracker.redemption_queue._file)
                if qpath in seen_startup_queues:
                    continue
                seen_startup_queues.add(qpath)
                if len(tracker.redemption_queue) > 0:
                    print(f"  [STARTUP] Processing {len(tracker.redemption_queue)} "
                          f"queued redemption(s)...")
                    try:
                        await asyncio.to_thread(tracker.process_redemption_queue)
                    except Exception as exc:
                        print(f"  [STARTUP] Queue processing error: {exc}")

            # Start periodic redemption processor (retries every 90s, independent of trading)
            redeem_task = asyncio.create_task(
                _periodic_redeem_loop(all_trackers, interval_s=90.0)
            )

        # Run all timeframe loops concurrently
        window_loops = []
        for i, (tracker, config_key, config) in enumerate(trackers_and_configs):
            window_loops.append(
                _window_loop(
                    tracker, config, price_state, binance_state,
                    signal_wakeup, wake_state,
                    display_sections[i], all_trackers,
                    shutdown,
                    record=record,
                )
            )
        await asyncio.gather(*window_loops)
    finally:
        display_stop.set()
        display_thread.join(timeout=2)
        feed_cancel.set()
        price_poll.cancel()
        if binance_poll is not None:
            binance_poll.cancel()
            await asyncio.gather(binance_poll, return_exceptions=True)
        await asyncio.gather(price_poll, return_exceptions=True)
        # Cancel periodic redemption task
        if redeem_task is not None:
            redeem_task.cancel()
            try:
                await redeem_task
            except asyncio.CancelledError:
                pass


# ── Tracker Builder ──────────────────────────────────────────────────────────

def _build_tracker(
    args, config, config_key: str, client, bankroll: float,
    saved: dict | None,
) -> LiveTradeTracker:
    """Build a LiveTradeTracker for one timeframe config."""
    base_market = config_key.replace("_5m", "").replace("_1h", "")
    is_5m = "_5m" in config_key
    is_1h = "_1h" in config_key or config.window_duration_s >= 3600

    # Per-market signal overrides
    # BTC: max_z=0.5 (prevent overconfidence), reversion_discount=0.30
    # ETH: max_z=0.7, reversion_discount=0.20 (15m) / 0.0 (5m)
    # SOL / XRP: defaults
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
    elif base_market == "sol":
        signal_kw = dict(
            edge_threshold=0.12,
            reversion_discount=0.0,
            momentum_lookback_s=15,
            momentum_majority=0.7,
            spread_edge_penalty=0.2,
        )
    elif base_market == "xrp":
        signal_kw = dict(
            edge_threshold=0.12,
            reversion_discount=0.0,
            momentum_lookback_s=15,
            momentum_majority=0.7,
            spread_edge_penalty=0.2,
        )

    # Maker signal overrides
    signal_kw["window_duration"] = config.window_duration_s
    signal_kw["maker_mode"] = True
    signal_kw["max_bet_fraction"] = args.max_bet_fraction
    signal_kw["kelly_fraction"] = args.kelly_fraction
    signal_kw["edge_threshold"] = signal_kw.get("edge_threshold", config.edge_threshold)
    signal_kw["momentum_majority"] = 0.0
    signal_kw["spread_edge_penalty"] = signal_kw.get("spread_edge_penalty", 0.0)
    # F4: oracle lead-lag bias (default 0.0 = disabled, opt-in via flag)
    signal_kw["oracle_lead_bias"] = args.oracle_lead_bias
    cooldown_ms = 30_000
    stale_timeout = 60.0

    # Calibration — enabled for all markets when --calibrated is set
    use_calibration = args.calibrated
    if use_calibration:
        from backtest import DATA_DIR
        cal_data_dir = DATA_DIR / config.data_subdir
        print(f"  [{config.display_name}] Building calibration table...")
        try:
            cal_table = build_calibration_table(cal_data_dir, vol_lookback_s=90)
            n_cells = len(cal_table.table)
            n_obs = sum(cal_table.counts.values())
            print(f"  [{config.display_name}] Calibration: {n_cells} cells, {n_obs} obs")
            signal_kw["calibration_table"] = cal_table
            signal_kw["cal_prior_strength"] = 50.0
            signal_kw["cal_max_weight"] = 0.70
        except Exception as exc:
            print(f"  [{config.display_name}] WARNING: calibration failed: {exc}")

    signal_kw["inventory_skew"] = args.inventory_skew
    signal_kw["maker_warmup_s"] = args.maker_warmup
    signal_kw["maker_withdraw_s"] = args.maker_withdraw
    signal_kw["max_sigma"] = config.max_sigma
    signal_kw["min_sigma"] = config.min_sigma
    signal_kw["toxicity_threshold"] = args.toxicity_threshold
    signal_kw["toxicity_edge_mult"] = args.toxicity_edge_mult
    if args.vol_kill_sigma is not None:
        signal_kw["vol_kill_sigma"] = args.vol_kill_sigma
    signal_kw["down_edge_bonus"] = args.down_edge_bonus
    signal_kw["regime_z_scale"] = args.regime_z_scale
    signal_kw["vpin_threshold"] = args.vpin_threshold
    signal_kw["vpin_edge_mult"] = args.vpin_edge_mult
    signal_kw["vpin_window"] = args.vpin_window
    signal_kw["vpin_bar_s"] = args.vpin_bar_s
    signal_kw["oracle_lag_threshold"] = args.oracle_lag_threshold
    signal_kw["oracle_lag_mult"] = args.oracle_lag_mult
    signal_kw["obi_weight"] = args.obi_weight
    # Tail mode + Kou params: prefer CLI override, fall back to market config
    signal_kw["tail_mode"] = args.tail_mode or config.tail_mode
    signal_kw["tail_nu_default"] = args.tail_nu if args.tail_nu is not None else config.tail_nu_default
    signal_kw["kou_lambda"] = config.kou_lambda
    signal_kw["kou_p_up"] = config.kou_p_up
    signal_kw["kou_eta1"] = config.kou_eta1
    signal_kw["kou_eta2"] = config.kou_eta2
    signal_kw["market_blend"] = config.market_blend
    signal_kw["max_book_age_ms"] = config.max_book_age_ms
    # Per-market entry filters: CLI overrides config if explicitly set
    signal_kw["min_entry_z"] = args.min_z if args.min_z > 0 else config.min_entry_z
    signal_kw["min_entry_price"] = args.min_entry_price if args.min_entry_price != 0.10 else config.min_entry_price
    signal_kw["edge_threshold"] = config.edge_threshold
    # Edge persistence: require edge to hold for N seconds before firing.
    # Defends against fast-spike chasing where the model briefly crosses
    # the edge threshold and then mean-reverts. Hardcoded per timeframe
    # for now: 5s for 5m markets, 10s for 15m markets. See 2026-04-09
    # debugging session — the 00:35:50 BTC fill that motivated this gate.
    if is_1h:
        signal_kw["edge_persistence_s"] = 0.0   # no spike gate needed on 1h
    elif is_5m:
        signal_kw["edge_persistence_s"] = 5.0
    else:
        signal_kw["edge_persistence_s"] = 10.0  # 15m

    # A-S quoting params
    signal_kw["as_mode"] = args.as_mode
    signal_kw["gamma_inv"] = args.gamma_inv
    signal_kw["gamma_spread"] = args.gamma_spread
    # Per-market gamma_spread overrides (CLI default is 1.5)
    if base_market == "eth" and not is_5m:
        signal_kw["gamma_spread"] = 0.75   # ETH 15m
    elif base_market == "btc" and is_5m:
        signal_kw["gamma_spread"] = 2.0    # BTC 5m
    signal_kw["min_edge"] = args.min_edge
    signal_kw["tox_spread"] = args.tox_spread
    signal_kw["vpin_spread"] = args.vpin_spread_as
    signal_kw["lag_spread"] = args.lag_spread
    signal_kw["edge_step"] = args.edge_step
    signal_kw["contract_vol_lookback_s"] = args.contract_vol_lookback

    # Compute sigma_calibration from recorded data when regime-z-scale is on
    if args.regime_z_scale and args.calibrated:
        from backtest import DATA_DIR, _compute_vol_deduped
        cal_data_dir_for_sigma = DATA_DIR / config.data_subdir
        MAX_CAL_FILES = 200
        MIN_SIGMA_CAL = 1e-7
        try:
            sigmas = []
            files = sorted(cal_data_dir_for_sigma.glob("*.parquet"))
            files = files[-MAX_CAL_FILES:]
            for f in files:
                df = pd.read_parquet(f)
                if df.empty:
                    continue
                pcol = "chainlink_price" if "chainlink_price" in df.columns else "chainlink_btc"
                if pcol not in df.columns:
                    continue
                prices = df[pcol].tolist()
                ts_list = df["ts_ms"].tolist() if "ts_ms" in df.columns else None
                s = _compute_vol_deduped(prices, ts_list)
                if s > 0:
                    sigmas.append(s)
            if sigmas:
                sigma_cal = max(float(np.median(sigmas)), MIN_SIGMA_CAL)
                signal_kw["sigma_calibration"] = sigma_cal
                print(f"  [{config.display_name}] sigma_calibration={sigma_cal:.2e} "
                      f"(median of {len(sigmas)} windows)")
            else:
                print(f"  [{config.display_name}] WARNING: no valid sigma data for regime-z-scale")
        except Exception as exc:
            print(f"  [{config.display_name}] WARNING: sigma_calibration failed: {exc}")

    # VAMP
    if base_market == "btc":
        signal_kw["vamp_mode"] = "filter"
    elif base_market == "eth":
        signal_kw["vamp_mode"] = "filter"
        signal_kw["vamp_filter_threshold"] = 0.07
    elif base_market in ("sol", "xrp"):
        signal_kw["vamp_mode"] = "filter"
        signal_kw["vamp_filter_threshold"] = 0.07

    # Withdraw timing: stop placing new buys near window end.
    # Polymarket can be buggy near resolution — prices spike, spreads
    # blow out, and the oracle hasn't finalized yet.
    is_1h = config.window_duration_s >= 3600
    if is_1h:
        maker_withdraw_override = 120.0   # 2 min buffer for 1h
    elif is_5m:
        maker_withdraw_override = 20.0    # 20s for 5m
    else:
        maker_withdraw_override = 60.0    # 60s for 15m

    # 5m overrides
    maker_warmup = args.maker_warmup
    maker_withdraw = maker_withdraw_override
    max_order_age = args.max_order_age
    requote_cooldown = args.requote_cooldown
    exit_min_hold = args.exit_min_hold
    exit_min_remaining = args.exit_min_remaining

    if is_1h:
        # 1h windows: the early_edge_mult inflates the dynamic threshold
        # too aggressively. At tau=1800 (mid-window), default 1.2 pushes
        # thresh from 0.04 to 0.071, filtering legitimate setups.
        # 0.5 gives thresh=0.04*(1+0.5*0.7)=0.054 at mid-window.
        signal_kw["early_edge_mult"] = 0.5
        signal_kw["maker_warmup_s"] = 120.0    # 2 min warmup for vol
        signal_kw["maker_withdraw_s"] = maker_withdraw
        signal_kw["vol_lookback_s"] = 120      # 2 min vol lookback (more data for 1h)
        cooldown_ms = 30_000                    # 30s cooldown between trades
        exit_min_hold = 60.0
        exit_min_remaining = 120.0
    elif is_5m:
        signal_kw["vol_lookback_s"] = 30
        maker_warmup = 30.0
        if base_market in ("eth", "sol", "xrp"):
            signal_kw["early_edge_mult"] = 1.2
        signal_kw["maker_warmup_s"] = maker_warmup
        signal_kw["maker_withdraw_s"] = maker_withdraw
        cooldown_ms = 10_000
        max_order_age = 20.0
        requote_cooldown = 2.0
        exit_min_hold = 10.0
        exit_min_remaining = 20.0

    # Per-market max_trades_per_window (from config), CLI overrides
    max_trades = (args.max_trades_per_window
                  if args.max_trades_per_window is not None
                  else config.max_trades_per_window)

    # Shared trades log and state file (named after base asset, not timeframe)
    trades_log = Path(f"live_trades_{base_market}.jsonl")
    state_file = Path(f"live_state_{base_market}.json")

    signal = DiffusionSignal(bankroll=bankroll, slippage=args.slippage, **signal_kw)
    tracker = LiveTradeTracker(
        client=client,
        signal=signal,
        initial_bankroll=bankroll,
        latency_ms=args.latency,
        slippage=args.slippage,
        cooldown_ms=cooldown_ms,
        max_loss_pct=args.max_loss_pct,
        max_trades_per_window=max_trades,
        same_direction_stacking_only=config.same_direction_stacking_only,
        stale_price_timeout_s=stale_timeout,
        window_duration_s=config.window_duration_s,
        edge_cancel_threshold=args.edge_cancel_threshold,
        max_order_age_s=max_order_age,
        requote_cooldown_s=requote_cooldown,
        max_exposure_pct=args.max_exposure_pct,
        maker_warmup_s=maker_warmup,
        maker_withdraw_s=maker_withdraw,
        exit_enabled=args.early_exit,
        exit_threshold=args.exit_threshold,
        exit_min_hold_s=exit_min_hold,
        exit_min_remaining_s=exit_min_remaining,
        # 1h: force 1 position at a time for buy/sell/rebuy cycles.
        # Multiple concurrent positions on a $100 bankroll = over-exposure.
        max_positions=1 if is_1h else args.max_positions,
        exit_sell_buffer=args.exit_sell_buffer,
        debug=args.debug,
        dry_run=args.dry_run,
        trades_log=trades_log,
        state_file=state_file,
        min_requote_ticks=args.min_requote_ticks,
        dual_side=args.dual_side,
        max_net_exposure=args.max_net_exposure,
        max_gross_exposure=args.max_gross_exposure,
    )
    if use_calibration:
        from backtest import DATA_DIR
        tracker.cal_data_dir = DATA_DIR / config.data_subdir

    if saved:
        tracker.windows_seen = saved.get("windows_seen", 0)
        tracker.windows_traded = saved.get("windows_traded", 0)
        tracker.total_fees = saved.get("total_fees", 0.0)
        tracker.peak_bankroll = saved.get("peak_bankroll", bankroll)
        tracker.max_drawdown = saved.get("max_drawdown", 0.0)
        tracker.max_dd_pct = saved.get("max_dd_pct", 0.0)
        # Restore lifetime scalar totals (P12.4). These are the source
        # of truth for cumulative PnL — they survive truncation of the
        # all_results list during save. Fall back to the legacy total_pnl
        # field for backward compat with older state files that don't
        # have lifetime_* yet.
        tracker.lifetime_pnl = float(saved.get("lifetime_pnl", saved.get("total_pnl", 0.0)))
        tracker.lifetime_wins = int(saved.get("lifetime_wins", saved.get("wins", 0)))
        tracker.lifetime_losses = int(saved.get("lifetime_losses", saved.get("losses", 0)))
        tracker.lifetime_trades = int(saved.get("lifetime_trades", saved.get("total_trades", 0)))
        # Restore trade history so the displayed PnL is lifetime, not
        # session-since-restart. Without this, the bot's display PnL
        # resets to $0 every restart even though `bankroll` reflects
        # cumulative trades — confusing the operator about real returns.
        saved_results = saved.get("all_results", [])
        if saved_results:
            tracker.all_results = LiveTradeTracker._restore_results(saved_results)
            print(f"  [{config.display_name}] Restored {len(tracker.all_results)} prior trades from state")

        # Restore pending_fills so positions aren't silently lost on
        # crash-between-fill-and-resolve. On-chain CTF tokens still
        # exist; without restoring them the bot forgets and the
        # position goes unredeemed. On restart, we log a warning for
        # each restored fill so the operator knows they're being
        # carried forward; the next resolve_window call will handle
        # them normally.
        saved_fills = saved.get("pending_fills", [])
        if saved_fills:
            tracker.pending_fills = list(saved_fills)
            print(f"  [{config.display_name}] Restored {len(saved_fills)} "
                  f"pending fills from state — will resolve in next window")
            for f in saved_fills:
                print(f"    {f.get('side', '?')} {f.get('shares', 0):.1f}sh "
                      f"${f.get('cost_usd', 0):.2f}")

        # Restore open_orders so the bot can cancel stale resting
        # orders from a prior session. On startup, all restored orders
        # should be cancelled immediately (they belong to a dead
        # window). The caller should invoke _cancel_open_orders after
        # building the tracker if there are restored orders. We set
        # a flag so the caller knows to do this.
        saved_orders = saved.get("open_orders", [])
        if saved_orders:
            tracker.open_orders = list(saved_orders)
            tracker._has_restored_orders = True
            print(f"  [{config.display_name}] Restored {len(saved_orders)} "
                  f"open orders — will cancel on next window start")

    return tracker


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Live trading bot for Polymarket Up/Down markets",
        epilog=(
            "Tip: the Rust feed extension prints WebSocket reconnect "
            "events to stderr (e.g. '[BookFeed] read error: ...'). These "
            "are auto-recovered by the 30s stale watchdog and are not a "
            "concern. To keep the display clean, redirect stderr to a "
            "file: `uv run python live_trader.py --market btc 2>err.log`"
        ),
    )
    parser.add_argument(
        "--market", default=DEFAULT_MARKET,
        choices=list(MARKET_CONFIGS),
        help="Market to trade — 'btc' runs both 15m+5m (default: btc)",
    )
    parser.add_argument(
        "--bankroll", type=float, default=None,
        help="Override bankroll (default: use API USDC balance)",
    )
    parser.add_argument(
        "--latency", type=int, default=0,
        help="Simulated order latency in ms (default: 0)",
    )
    parser.add_argument(
        "--slippage", type=float, default=0.0,
        help="Slippage buffer for signal (default: 0.0)",
    )
    parser.add_argument(
        "--max-loss-pct", type=float, default=50.0,
        help="Circuit breaker: max session loss %% (default: 50)",
    )
    parser.add_argument(
        "--max-trades-per-window", type=int, default=None,
        help="Max trades per window (default: 6)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run signal but don't place real orders",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from saved state file",
    )
    parser.add_argument(
        "--edge-cancel-threshold", type=float, default=0.06,
        help="Cancel open orders when edge drops below this (default: 0.06)",
    )
    parser.add_argument(
        "--max-order-age", type=float, default=30.0,
        help="Auto-cancel orders resting longer than this (seconds, default: 30)",
    )
    parser.add_argument(
        "--requote-cooldown", type=float, default=3.0,
        help="Min seconds between cancel+replace on same side (default: 3)",
    )
    parser.add_argument("--max-exposure-pct", type=float, default=20.0,
                        help="Max %% of initial bankroll committed to open orders (default: 20)")
    parser.add_argument("--maker-warmup", type=float, default=200.0,
                        help="Seconds to wait after window opens before trading (default: 200)")
    parser.add_argument("--calibrated", action="store_true",
                        help="Use empirically calibrated probabilities")
    parser.add_argument("--early-exit", action="store_true",
                        help="Enable early exit of positions based on EV")
    parser.add_argument("--exit-threshold", type=float, default=0.03,
                        help="Min EV edge to trigger exit (default: 0.03)")
    parser.add_argument("--exit-min-hold", type=float, default=30.0,
                        help="Min seconds after fill before evaluating exit (default: 30)")
    parser.add_argument("--exit-min-remaining", type=float, default=60.0,
                        help="Don't exit when < this many seconds remain (default: 60)")
    parser.add_argument("--exit-sell-buffer", type=float, default=0.08,
                        help="Buffer above model value for sell price floor (default: 0.08)")
    parser.add_argument("--max-positions", type=int, default=2,
                        help="Max simultaneous filled positions (default: 1)")
    parser.add_argument("--inventory-skew", type=float, default=0.02,
                        help="Edge penalty per same-side position (default 0.02)")
    parser.add_argument("--maker-withdraw", type=float, default=30.0,
                        help="Stop new orders when tau < N seconds (default 30)")
    parser.add_argument("--min-requote-ticks", type=int, default=2,
                        help="Min tick improvement before requoting (default: 2 = $0.02)")
    parser.add_argument("--dual-side", action="store_true", default=False,
                        help="Allow opposite-side positions only when it reduces net exposure")
    parser.add_argument("--max-net-exposure", type=float, default=0.0,
                        help="Max |UP_shares - DOWN_shares| (0=disabled)")
    parser.add_argument("--max-gross-exposure", type=float, default=0.0,
                        help="Max total shares across both sides (0=disabled)")
    parser.add_argument("--toxicity-threshold", type=float, default=0.75,
                        help="Toxicity score above which edge threshold widens (default: 0.75)")
    parser.add_argument("--toxicity-edge-mult", type=float, default=1.5,
                        help="Max edge threshold multiplier at toxicity=1.0 (default: 1.5)")
    parser.add_argument("--vol-kill-sigma", type=float, default=None,
                        help="Absolute sigma ceiling — pause quoting above this (default: None)")
    parser.add_argument("--down-edge-bonus", type=float, default=0.05,
                        help="Fraction to reduce DOWN edge threshold (optimism tax, default: 0.05)")
    parser.add_argument("--regime-z-scale", action="store_true",
                        help="Scale z-scores by sigma_calibration / sigma_ema (requires --calibrated)")
    parser.add_argument("--vpin-threshold", type=float, default=0.95,
                        help="VPIN above which edge threshold widens (default: 0.95)")
    parser.add_argument("--vpin-edge-mult", type=float, default=1.5,
                        help="Max edge threshold multiplier at VPIN=1.0 (default: 1.5)")
    parser.add_argument("--vpin-window", type=int, default=20,
                        help="Number of completed trade bars for VPIN (default: 20)")
    parser.add_argument("--vpin-bar-s", type=float, default=60.0,
                        help="Trade bar duration in seconds (default: 60)")
    parser.add_argument("--oracle-lag-threshold", type=float, default=0.002,
                        help="Binance-Chainlink discrepancy above which edge widens (default: 0.002)")
    parser.add_argument("--oracle-lag-mult", type=float, default=2.0,
                        help="Max edge threshold multiplier at full oracle lag (default: 2.0)")
    parser.add_argument("--obi-weight", type=float, default=0.03,
                        help="Order book imbalance alpha weight on p_model (default: 0.03)")
    parser.add_argument("--kelly-fraction", type=float, default=0.25,
                        help="Kelly fraction for sizing (0.25 = quarter-Kelly, default: 0.25)")
    parser.add_argument("--max-bet-fraction", type=float, default=0.05,
                        help="Max fraction of bankroll per trade (default: 0.05)")
    parser.add_argument("--oracle-lead-bias", type=float, default=0.0,
                        help="F4: bias on p_model from Binance→Chainlink lead-lag. "
                             "When binance_mid > chainlink, bias p_up upward by up to "
                             "this amount (e.g. 0.05 = +5pp at gap = oracle_lag_threshold). "
                             "Default 0.0 = disabled. Recommended: 0.05 after backtest A/B.")
    parser.add_argument("--tail-mode", choices=["student_t", "normal", "kou"],
                        default=None,
                        help="CDF for z→probability (default: from market config)")
    parser.add_argument("--tail-nu", type=float, default=None,
                        help="Student-t nu floor / default (default: from market config)")
    parser.add_argument("--min-z", type=float, default=0.0,
                        help="Minimum |z-score| to enter a trade (default: 0.0)")
    parser.add_argument("--min-entry-price", type=float, default=0.10,
                        help="Minimum contract entry price (default: 0.10)")
    # Avellaneda-Stoikov unified quoting
    parser.add_argument("--as-mode", action="store_true", default=False,
                        help="Enable A-S reservation price quoting")
    parser.add_argument("--gamma-inv", type=float, default=0.15,
                        help="A-S gamma for inventory penalty (default: 0.15)")
    parser.add_argument("--gamma-spread", type=float, default=1.5,
                        help="A-S gamma for base spread (default: 0.75)")
    parser.add_argument("--min-edge", type=float, default=0.05,
                        help="Floor on required edge in A-S mode (default: 0.05)")
    parser.add_argument("--tox-spread", type=float, default=0.05,
                        help="Additive spread from toxicity (default: 0.05)")
    parser.add_argument("--vpin-spread-as", type=float, default=0.05,
                        help="Additive spread from VPIN in A-S mode (default: 0.05)")
    parser.add_argument("--lag-spread", type=float, default=0.08,
                        help="Additive spread from oracle lag (default: 0.08)")
    parser.add_argument("--edge-step", type=float, default=0.01,
                        help="Additive spread per fill (default: 0.01)")
    parser.add_argument("--contract-vol-lookback", type=int, default=60,
                        help="Lookback (s) for contract mid vol (default: 60)")

    parser.add_argument("--no-record", action="store_true",
                        help="Disable integrated parquet recording")
    parser.add_argument("--no-binance", action="store_true",
                        help="Disable Binance feed (use when another process already connects "
                             "to the same symbol, e.g. dashboard + recorder running together)")
    parser.add_argument("--no-5m", action="store_true",
                        help="Disable 5m timeframe (only trade 15m)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Resolve market -> list of (config_key, config) pairs
    paired = get_paired_configs(args.market)
    if args.no_5m:
        paired = [(k, c) for k, c in paired if "_5m" not in k]
    base_config = paired[0][1]  # use first config for shared settings
    base_market = args.market.replace("_5m", "")

    # Print header
    print(f"\n  {'='*62}")
    mode = "DRY RUN" if args.dry_run else "LIVE TRADING"
    order_mode = "MAKER"
    if args.calibrated:
        order_mode += "+CAL"
    if args.early_exit:
        order_mode += "+EXIT"
    timeframes = ", ".join(c.display_name for _, c in paired)
    print(f"  {mode} [{order_mode}] -- {timeframes}")
    print(f"  {'='*62}")

    # Build authenticated client (skipped in dry-run if no key present)
    private_key = os.getenv("PRIVATE_KEY")
    if not args.dry_run and not private_key:
        print("  ERROR: PRIVATE_KEY not set. Create a .env file with your credentials.")
        return
    client = None
    if private_key:
        client = OrderClient(
            host="https://clob.polymarket.com",
            private_key=private_key,
            chain_id=137,
            api_key=os.getenv("POLY_API_KEY", ""),
            api_secret=os.getenv("POLY_API_SECRET", ""),
            api_passphrase=os.getenv("POLY_PASSPHRASE", ""),
            sig_type=int(os.getenv("SIGNATURE_TYPE", "1")),
            funder=os.getenv("POLY_FUNDER") or None,
        )

    # Determine bankroll
    api_balance = None
    try:
        if client is not None:
            api_balance = client.get_balance()
    except Exception as exc:
        api_balance = None
        if args.debug:
            print(f"  [BALANCE] error: {exc}")
    if api_balance is not None:
        print(f"  API USDC balance: ${api_balance:,.2f}")

    bankroll = args.bankroll
    saved = None
    state_file = Path(f"live_state_{base_market}.json")
    if args.resume:
        saved = LiveTradeTracker.load_state(state_file)
        if saved:
            bankroll = saved["bankroll"]
            print(f"  Resumed: bankroll=${bankroll:,.2f}, "
                  f"{saved.get('total_trades', 0)} trades, "
                  f"PnL=${saved.get('total_pnl', 0):+,.2f}")

    if bankroll is None:
        if api_balance is not None:
            bankroll = api_balance
        else:
            bankroll = 10_000.0
            print(f"  WARNING: Could not query balance, using default ${bankroll:,.0f}")

    # Build one tracker per timeframe
    trackers_and_configs: list[tuple[LiveTradeTracker, str, object]] = []
    for config_key, config in paired:
        tracker = _build_tracker(args, config, config_key, client, bankroll, saved)
        trackers_and_configs.append((tracker, config_key, config))
        print(f"  [{config.display_name}] Tracker ready "
              f"(edge={tracker.signal.edge_threshold:.2f}, "
              f"warmup={tracker.maker_warmup_s:.0f}s, "
              f"withdraw={tracker.maker_withdraw_s:.0f}s)")

    # Gas balance check (once, shared — skipped in dry-run without key)
    try:
        w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
        if w3.is_connected() and private_key:
            signer = w3.eth.account.from_key(private_key)
            pol_balance = w3.eth.get_balance(signer.address)
            pol_ether = w3.from_wei(pol_balance, "ether")
            print(f"  Signer POL:      {pol_ether:.4f} POL")
            if pol_ether < 0.01:
                print(f"  WARNING: Low POL balance ({pol_ether:.4f}) — "
                      f"need gas for CTF redemption txs")
        else:
            print(f"  WARNING: Cannot connect to Polygon RPC — "
                  f"auto-redemption may fail")
    except Exception as exc:
        print(f"  WARNING: Gas check failed: {exc}")

    # Print shared settings
    print(f"  Bankroll:        ${bankroll:,.2f}")
    print(f"  Timeframes:      {timeframes}")
    print(f"  Max loss:        {args.max_loss_pct}%")
    print(f"  Max exposure:    {args.max_exposure_pct}%")
    print(f"  Kelly:           {args.kelly_fraction}x Kelly, max {args.max_bet_fraction*100:.0f}% per trade")
    if args.as_mode:
        print(f"  A-S mode:        gamma_inv={args.gamma_inv}, gamma_spread={args.gamma_spread}, min_edge={args.min_edge}")
    if args.early_exit:
        print(f"  Early exit:      ON (threshold={args.exit_threshold})")
    print(f"  VPIN:            thresh={args.vpin_threshold} mult={args.vpin_edge_mult}")
    print(f"  Oracle lag:      thresh={args.oracle_lag_threshold} mult={args.oracle_lag_mult} "
          f"binance={base_config.binance_symbol or 'disabled'}")
    print(f"  Feed cadence:    price={FAST_PRICE_POLL_S*1000:.0f}ms "
          f"binance={FAST_BINANCE_POLL_S*1000:.0f}ms "
          f"book={FAST_BOOK_POLL_S*1000:.0f}ms "
          f"signal_idle={FAST_SIGNAL_IDLE_S*1000:.0f}ms "
          f"signal_min={FAST_SIGNAL_MIN_INTERVAL_S*1000:.0f}ms")
    record = not args.no_record
    if args.no_binance:
        import dataclasses
        base_config = dataclasses.replace(base_config, binance_symbol="")
    print(f"  Recording:       {'ON' if record else 'OFF'}")
    print(f"  Trades log:      live_trades_{base_market}.jsonl")
    print(f"  State file:      {state_file}")
    print()

    all_trackers = [t for t, _, _ in trackers_and_configs]

    try:
        asyncio.run(run(
            trackers_and_configs, base_config,
            dry_run=args.dry_run,
            exit_enabled=args.early_exit,
            debug=args.debug,
            record=record,
        ))
    except KeyboardInterrupt:
        print(f"\n  Shutting down...")
        total_pnl = 0.0
        for tracker, _, _ in trackers_and_configs:
            if tracker.flat_reason_counts:
                tracker._log_flat_summary()
            tracker.cancel_all_orders()
            tracker.save_state()
            total_pnl += sum(r.pnl for r in tracker.all_results)
        print(f"  Session PnL: ${total_pnl:+,.2f}")
        print(f"  Final bankroll: ${all_trackers[0].bankroll:,.2f}")
        print(f"  State saved to {state_file}")
        print("  Exiting.")


if __name__ == "__main__":
    main()
