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

from polybot_core import OrderClient, BookFeed, PriceFeed, BinanceFeed

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
    window_start_price: float | None = None,
):
    while not cancel.is_set():
        await asyncio.sleep(1)
        if cancel.is_set():
            break

        try:
            # Clear previous error on successful tick start
            tracker.ctx.pop("_signal_error", None)

            # Inject Binance mid into ctx only if fresh (< 10s old).
            # Stale Binance data (WS dropped) would poison vol/z/oracle-lag.
            if binance_state is not None:
                _bn_ts = binance_state.get("last_update_ts", 0)
                if _bn_ts and (_time.time() - _bn_ts) < 10.0:
                    tracker.ctx["_binance_mid"] = binance_state.get("mid_price")
                else:
                    tracker.ctx.pop("_binance_mid", None)

            # Accumulate price history on EVERY tick — this must never be
            # gated by snapshot, skip_trading, stale price, or any guard.
            eff_px = tracker.ctx.get("_binance_mid") or price_state.get("price")
            if eff_px is not None:
                hist = tracker.ctx.setdefault("price_history", [])
                ts_hist = tracker.ctx.setdefault("ts_history", [])
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
                await asyncio.to_thread(tracker.evaluate, snap, up_token, down_token)

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

async def _poll_price_feed(price_feed: PriceFeed, price_state: dict,
                           trackers: list[LiveTradeTracker],
                           cancel: asyncio.Event, debug: bool = False,
                           symbol: str = ""):
    """Bridge Rust PriceFeed → Python price_state dict (shared across trackers).

    Auto-restarts the Rust WS feed if the price stays stale for >60s
    (beyond the Rust-side 30s reconnect timeout).
    """
    last_ts = 0.0
    _stale_warned = False
    _STALE_RESTART_S = 60.0
    while not cancel.is_set():
        await asyncio.sleep(0.1)  # fast poll — captures every Chainlink round for accurate start price
        if cancel.is_set():
            break
        try:
            px = price_feed.price()
            ts = price_feed.last_update_ts()
            if px is not None and ts > last_ts:
                last_ts = ts
                _stale_warned = False
                price_state["price"] = px
                for t in trackers:
                    t.last_price_update_ts = ts
                ts_ms = int(ts * 1000)
                price_state["price_history"].append((ts_ms, px))
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
                             cancel: asyncio.Event, symbol: str = ""):
    """Bridge Rust BinanceFeed → Python binance_state dict.

    Auto-restarts if no update for >60s.
    """
    _last_ts = 0.0
    _STALE_RESTART_S = 60.0
    while not cancel.is_set():
        await asyncio.sleep(0.5)
        if cancel.is_set():
            break
        try:
            mid = binance_feed.mid()
            ts = binance_feed.last_update_ts()
            if mid is not None and ts > _last_ts:
                _last_ts = ts
                binance_state["mid_price"] = mid
                binance_state["last_update_ts"] = ts
            elif _last_ts > 0:
                age = _time.time() - _last_ts
                if age > _STALE_RESTART_S and symbol:
                    print(f"  [BinanceFeed] stale for {age:.0f}s — restarting WS")
                    binance_feed = BinanceFeed(symbol)
                    _last_ts = 0.0
        except Exception:
            pass


async def _poll_book_feed(book_feed: BookFeed, up_token: str, down_token: str,
                          flat_state: dict, trade_state: dict | None,
                          cancel: asyncio.Event):
    """Bridge Rust BookFeed → Python flat_state + trade_state for VPIN."""
    while not cancel.is_set():
        await asyncio.sleep(0.2)
        if cancel.is_set():
            break
        try:
            # Update flat_state BBO for display thread
            for side, token in [("up", up_token), ("down", down_token)]:
                snap = book_feed.snapshot(token)
                bb = snap.best_bid
                ba = snap.best_ask
                flat_state[f"{side}_best_bid"] = str(bb) if bb is not None else None
                flat_state[f"{side}_best_ask"] = str(ba) if ba is not None else None

            # Drain trade events for VPIN bar accumulation
            if trade_state is not None:
                for size, trade_side in book_feed.drain_trades():
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
                        trade_state["total_bars"] = (
                            trade_state.get("total_bars", 0) + 1
                        )
                        bar["buy_vol"] = 0.0
                        bar["sell_vol"] = 0.0
                        bar["start_ts"] = now_ts
        except Exception:
            pass


# ── Bankroll sync ────────────────────────────────────────────────────────────

def _sync_bankroll(all_trackers: list[LiveTradeTracker]):
    """After a resolution, propagate the current bankroll to all trackers.

    Uses the minimum bankroll across trackers (conservative — accounts for
    capital committed by other timeframes). Also syncs the signal's bankroll.
    """
    if len(all_trackers) <= 1:
        return
    # Use the minimum bankroll (most conservative — reflects committed capital)
    min_bankroll = min(t.bankroll for t in all_trackers)
    for t in all_trackers:
        t.bankroll = min_bankroll
        t.signal.bankroll = min_bankroll


# ── Window Lifecycle ─────────────────────────────────────────────────────────

async def run_window(
    tracker: LiveTradeTracker, config, price_state: dict,
    trade_state: dict | None,
    binance_state: dict | None,
    section: dict,
    pending_resolve: asyncio.Task | None = None,
    record: bool = True,
) -> tuple[str, float | None, float | None, str] | None:
    """Run a single trading window. Returns (slug, final_price, start_price,
    condition_id) for resolution, or None if no market was found."""

    # Rebuild calibration table from latest data
    if tracker.cal_data_dir is not None:
        try:
            cal_table = build_calibration_table(tracker.cal_data_dir)
            tracker.signal.calibration_table = cal_table
            n_obs = sum(cal_table.counts.values())
            print(f"  [{config.display_name}] Calibration rebuilt: "
                  f"{len(cal_table.table)} cells, {n_obs} obs")
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

    # Wait until eventStartTime + 5s so the RTDS buffer has the start price
    now = datetime.now(timezone.utc)
    target = start + timedelta(seconds=5)
    wait_s = (target - now).total_seconds()
    if 0 < wait_s <= 120:
        section["status"] = "waiting"
        section["market_title"] = f"Waiting {wait_s:.0f}s for start..."
        print(f"  [{config.display_name}] Waiting {wait_s:.0f}s for start price...")
        await asyncio.sleep(wait_s)

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
        window_start_price = price_state.get("price")
        if window_start_price:
            print(f"  [{config.display_name}] Start: ${window_start_price:,.2f} (RTDS fallback)")
        else:
            print(f"  [{config.display_name}] Start: waiting for RTDS...")

    flat_state = {
        "up_best_bid": None, "up_best_ask": None,
        "down_best_bid": None, "down_best_ask": None,
    }

    # Wait for any pending resolution from previous window before trading
    # (bankroll must be settled before we size new orders)
    if pending_resolve is not None:
        await pending_resolve

    tracker.new_window(end)

    # Detect if we're joining mid-window on startup
    now_check = datetime.now(timezone.utc)
    elapsed_since_start = (now_check - start).total_seconds()
    skip_trading = (tracker.windows_seen == 1 and elapsed_since_start > 10)
    if skip_trading:
        tracker.last_decision = Decision(
            "FLAT", 0.0, 0.0,
            f"WARM-UP: joined {elapsed_since_start:.0f}s into window"
        )
        print(f"  [{config.display_name}] WARM-UP: joined {elapsed_since_start:.0f}s in — "
              f"skipping trading")

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
                            flat_state, trade_state, cancel)
        ),
        asyncio.create_task(
            signal_ticker(tracker, book_feed, price_state,
                          end, slug, up_token, down_token, cancel,
                          skip_trading=skip_trading,
                          trade_state=trade_state,
                          binance_state=binance_state,
                          window_start_price=window_start_price)
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

    return (slug, price_state.get("price"), window_start_price, tracker.condition_id)


async def _resolve_background(
    tracker: LiveTradeTracker,
    slug: str,
    final_price: float | None,
    start_price: float | None,
    condition_id: str,
    all_trackers: list[LiveTradeTracker],
):
    """Background resolution: resolve window + process redemption queue + sync bankroll."""
    await asyncio.to_thread(
        tracker.resolve_window, slug, final_price, start_price, condition_id,
    )
    # Process one queued redemption (includes newly enqueued + retries)
    await asyncio.to_thread(tracker.process_redemption_queue)
    _sync_bankroll(all_trackers)
    tracker.save_state()


async def _window_loop(
    tracker: LiveTradeTracker,
    config,
    price_state: dict,
    binance_state: dict,
    section: dict,
    all_trackers: list[LiveTradeTracker],
    record: bool = True,
):
    """Main loop for one timeframe — runs forever, handles search/trade/resolve."""
    # Trade state persists across windows — VPIN needs 20+ min history
    vpin_bar_s = tracker.signal.vpin_bar_s
    trade_state: dict = {
        "bars": collections.deque(maxlen=200),
        "current_bar": {"buy_vol": 0.0, "sell_vol": 0.0, "start_ts": 0},
        "bar_duration_s": vpin_bar_s,
        "total_bars": 0,
    }

    pending_resolve: asyncio.Task | None = None

    while True:
        result = await run_window(
            tracker, config, price_state, trade_state,
            binance_state, section,
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
    }
    binance_state: dict = {}

    # Shared Rust feeds — one PriceFeed + one BinanceFeed for the asset
    price_feed = PriceFeed(base_config.chainlink_symbol)
    print(f"  [PriceFeed] started (filtering for {base_config.chainlink_symbol})")

    binance_feed = None
    if base_config.binance_symbol:
        binance_feed = BinanceFeed(base_config.binance_symbol)
        print(f"  [BinanceFeed] started ({base_config.binance_symbol}@bookTicker)")

    # Shared polling loops
    feed_cancel = asyncio.Event()
    price_poll = asyncio.create_task(
        _poll_price_feed(price_feed, price_state, all_trackers, feed_cancel,
                         debug=debug, symbol=base_config.chainlink_symbol)
    )
    binance_poll = None
    if binance_feed is not None:
        binance_poll = asyncio.create_task(
            _poll_binance_feed(binance_feed, binance_state, feed_cancel,
                               symbol=base_config.binance_symbol)
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

    try:
        # Process any redemption queue items left from a previous session
        for tracker, _, _ in trackers_and_configs:
            if len(tracker.redemption_queue) > 0:
                print(f"  [STARTUP] Processing {len(tracker.redemption_queue)} "
                      f"queued redemption(s)...")
                await asyncio.to_thread(tracker.process_redemption_queue)

        # Run all timeframe loops concurrently
        window_loops = []
        for i, (tracker, config_key, config) in enumerate(trackers_and_configs):
            window_loops.append(
                _window_loop(
                    tracker, config, price_state, binance_state,
                    display_sections[i], all_trackers,
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


# ── Tracker Builder ──────────────────────────────────────────────────────────

def _build_tracker(
    args, config, config_key: str, client, bankroll: float,
    saved: dict | None,
) -> LiveTradeTracker:
    """Build a LiveTradeTracker for one timeframe config."""
    base_market = config_key.replace("_5m", "")
    is_5m = "_5m" in config_key

    # Per-market signal overrides
    signal_kw: dict = {}
    if base_market == "eth":
        signal_kw = dict(
            edge_threshold=0.08,
            reversion_discount=0.10,
            momentum_lookback_s=15,
            momentum_majority=0.7,
            spread_edge_penalty=0.2,
        )
    elif base_market == "sol":
        signal_kw = dict(
            edge_threshold=0.08,
            reversion_discount=0.10,
            momentum_lookback_s=15,
            momentum_majority=0.7,
            spread_edge_penalty=0.2,
        )
    elif base_market == "xrp":
        signal_kw = dict(
            edge_threshold=0.08,
            reversion_discount=0.10,
            momentum_lookback_s=15,
            momentum_majority=0.7,
            spread_edge_penalty=0.2,
        )

    # Maker signal overrides
    signal_kw["window_duration"] = config.window_duration_s
    signal_kw["maker_mode"] = True
    signal_kw["max_bet_fraction"] = 0.02
    signal_kw["edge_threshold"] = signal_kw.get("edge_threshold", 0.08)
    signal_kw["momentum_majority"] = 0.0
    signal_kw["spread_edge_penalty"] = signal_kw.get("spread_edge_penalty", 0.0)
    cooldown_ms = 30_000
    stale_timeout = 60.0

    # Calibration
    if args.calibrated:
        from backtest import DATA_DIR
        cal_data_dir = DATA_DIR / config.data_subdir
        print(f"  [{config.display_name}] Building calibration table...")
        try:
            cal_table = build_calibration_table(cal_data_dir, vol_lookback_s=90)
            n_cells = len(cal_table.table)
            n_obs = sum(cal_table.counts.values())
            print(f"  [{config.display_name}] Calibration: {n_cells} cells, {n_obs} obs")
            signal_kw["calibration_table"] = cal_table
        except Exception as exc:
            print(f"  [{config.display_name}] WARNING: calibration failed: {exc}")
        signal_kw["edge_threshold"] = 0.08
        signal_kw["early_edge_mult"] = 1.2

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

    # 5m overrides
    maker_warmup = args.maker_warmup
    maker_withdraw = args.maker_withdraw
    max_order_age = args.max_order_age
    requote_cooldown = args.requote_cooldown
    exit_min_hold = args.exit_min_hold
    exit_min_remaining = args.exit_min_remaining

    if is_5m:
        signal_kw["vol_lookback_s"] = 30
        signal_kw["edge_threshold"] = 0.10
        maker_warmup = 30.0
        maker_withdraw = 20.0
        if base_market == "eth":
            signal_kw["edge_threshold"] = 0.10
            signal_kw["early_edge_mult"] = 1.2
            signal_kw["reversion_discount"] = 0.10
        elif base_market in ("sol", "xrp"):
            signal_kw["edge_threshold"] = 0.10
            signal_kw["early_edge_mult"] = 1.2
            signal_kw["reversion_discount"] = 0.10
        signal_kw["maker_warmup_s"] = maker_warmup
        signal_kw["maker_withdraw_s"] = maker_withdraw
        cooldown_ms = 10_000
        max_order_age = 40.0
        requote_cooldown = 2.0
        exit_min_hold = 10.0
        exit_min_remaining = 20.0

    max_trades = args.max_trades_per_window if args.max_trades_per_window is not None else 6

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
        max_positions=args.max_positions,
        exit_sell_buffer=args.exit_sell_buffer,
        debug=args.debug,
        dry_run=args.dry_run,
        trades_log=trades_log,
        state_file=state_file,
    )
    if args.calibrated:
        from backtest import DATA_DIR
        tracker.cal_data_dir = DATA_DIR / config.data_subdir

    if saved:
        tracker.windows_seen = saved.get("windows_seen", 0)
        tracker.windows_traded = saved.get("windows_traded", 0)
        tracker.total_fees = saved.get("total_fees", 0.0)
        tracker.peak_bankroll = saved.get("peak_bankroll", bankroll)
        tracker.max_drawdown = saved.get("max_drawdown", 0.0)
        tracker.max_dd_pct = saved.get("max_dd_pct", 0.0)

    return tracker


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Live trading bot for Polymarket Up/Down markets"
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
        "--edge-cancel-threshold", type=float, default=0.02,
        help="Cancel open orders when edge drops below this (default: 0.02)",
    )
    parser.add_argument(
        "--max-order-age", type=float, default=120.0,
        help="Auto-cancel orders resting longer than this (seconds, default: 120)",
    )
    parser.add_argument(
        "--requote-cooldown", type=float, default=3.0,
        help="Min seconds between cancel+replace on same side (default: 3)",
    )
    parser.add_argument("--max-exposure-pct", type=float, default=20.0,
                        help="Max %% of initial bankroll committed to open orders (default: 20)")
    parser.add_argument("--maker-warmup", type=float, default=100.0,
                        help="Seconds to wait after window opens before trading (default: 100)")
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
    parser.add_argument("--max-positions", type=int, default=1,
                        help="Max simultaneous filled positions (default: 1)")
    parser.add_argument("--inventory-skew", type=float, default=0.02,
                        help="Edge penalty per same-side position (default 0.02)")
    parser.add_argument("--maker-withdraw", type=float, default=30.0,
                        help="Stop new orders when tau < N seconds (default 30)")
    parser.add_argument("--toxicity-threshold", type=float, default=0.75,
                        help="Toxicity score above which edge threshold widens (default: 0.75)")
    parser.add_argument("--toxicity-edge-mult", type=float, default=1.5,
                        help="Max edge threshold multiplier at toxicity=1.0 (default: 1.5)")
    parser.add_argument("--vol-kill-sigma", type=float, default=None,
                        help="Absolute sigma ceiling — pause quoting above this (default: None)")
    parser.add_argument("--down-edge-bonus", type=float, default=0.05,
                        help="Fraction to reduce DOWN edge threshold (optimism tax, default: 0.05)")
    parser.add_argument("--regime-z-scale", action="store_true",
                        help="Scale z-scores by sigma_ema / sigma_calibration (requires --calibrated)")
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
    parser.add_argument("--no-record", action="store_true",
                        help="Disable integrated parquet recording")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Resolve market -> list of (config_key, config) pairs
    paired = get_paired_configs(args.market)
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

    # Build authenticated client (shared across all trackers)
    client = OrderClient(
        host="https://clob.polymarket.com",
        private_key=os.getenv("PRIVATE_KEY"),
        chain_id=137,
        api_key=os.getenv("POLY_API_KEY", ""),
        api_secret=os.getenv("POLY_API_SECRET", ""),
        api_passphrase=os.getenv("POLY_PASSPHRASE", ""),
        sig_type=int(os.getenv("SIGNATURE_TYPE", "1")),
        funder=os.getenv("POLY_FUNDER") or None,
    )

    # Determine bankroll
    try:
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

    # Gas balance check (once, shared)
    try:
        w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
        if w3.is_connected():
            private_key = os.getenv("PRIVATE_KEY", "")
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
    if args.early_exit:
        print(f"  Early exit:      ON (threshold={args.exit_threshold})")
    print(f"  VPIN:            thresh={args.vpin_threshold} mult={args.vpin_edge_mult}")
    print(f"  Oracle lag:      thresh={args.oracle_lag_threshold} mult={args.oracle_lag_mult} "
          f"binance={base_config.binance_symbol or 'disabled'}")
    record = not args.no_record
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
