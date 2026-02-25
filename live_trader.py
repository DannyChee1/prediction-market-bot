#!/usr/bin/env python3
"""
Live Trading Bot for Polymarket 15-Min Up/Down Markets

Places GTC limit orders via the Polymarket CLOB API using the
DiffusionSignal from backtest.py. Tracks balance via API and enforces
circuit breakers.

Usage:
    py -3 live_trader.py                          # BTC, $10k bankroll
    py -3 live_trader.py --market eth              # ETH market
    py -3 live_trader.py --bankroll 500            # smaller bankroll
    py -3 live_trader.py --max-loss-pct 5          # stop at 5% loss
    py -3 live_trader.py --dry-run                 # signal + log, no real orders
    py -3 live_trader.py --debug
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from web3 import Web3

from backtest import (
    Decision, DiffusionSignal, build_calibration_table,
)
from market_config import MARKET_CONFIGS, DEFAULT_MARKET, get_config
from market_api import (
    build_clob_client, query_usdc_balance, find_market, _ensure_list,
)
from feeds import snapshot_from_live, clob_ws, rtds_ws
from display import render_display, display_ticker
from tracker import LiveTradeTracker
from recorder import OrderBook
from redemption import POLYGON_RPC

load_dotenv()


# ── Signal Ticker ────────────────────────────────────────────────────────────

async def signal_ticker(
    tracker: LiveTradeTracker,
    book_up: OrderBook, book_down: OrderBook,
    price_state: dict, window_end: datetime,
    market_slug: str, up_token: str, down_token: str,
    cancel: asyncio.Event,
    skip_trading: bool = False,
    trade_state: dict | None = None,
):
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
            if trade_state is not None:
                tracker.ctx["_trade_bars"] = trade_state["bars"]
            tracker.evaluate(snap, up_token, down_token)

        tracker.check_api_balance()


# ── Window Lifecycle ─────────────────────────────────────────────────────────

async def run_window(tracker: LiveTradeTracker, config, price_state: dict,
                     trade_state: dict | None = None):
    """Run a single trading window."""
    # Rebuild calibration table from latest data
    if tracker.cal_data_dir is not None:
        try:
            cal_table = build_calibration_table(tracker.cal_data_dir)
            tracker.signal.calibration_table = cal_table
            n_obs = sum(cal_table.counts.values())
            print(f"  Calibration table rebuilt: {len(cal_table.table)} cells, "
                  f"{n_obs} obs")
        except Exception as exc:
            print(f"  WARNING: calibration rebuild failed: {exc}")

    print(f"  Searching for active {config.display_name} market...")
    event, market = find_market(config)

    if not event or not market:
        print(f"  No active {config.display_name} market found. Retrying in 30s...")
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
    title = event["title"]

    tracker.condition_id = market.get("conditionId", "")
    if not tracker.condition_id:
        print("  WARNING: conditionId missing from market data — auto-redeem disabled for this window")

    book_up = OrderBook()
    book_down = OrderBook()

    # Wait until eventStartTime + 5s so the RTDS buffer has the start price
    now = datetime.now(timezone.utc)
    target = start + timedelta(seconds=5)
    wait_s = (target - now).total_seconds()
    if 0 < wait_s <= 120:
        print(f"  Waiting {wait_s:.0f}s for start price (until {target.strftime('%H:%M:%S')} UTC)...")
        await asyncio.sleep(wait_s)

    # Look up exact Chainlink price at eventStartTime from RTDS buffer
    start_ts_ms = int(start.timestamp() * 1000)
    start_price_exact = None
    history = price_state.get("price_history", [])
    best_entry = None
    if history:
        best_diff = float("inf")
        for ts_ms, px in history:
            diff = abs(ts_ms - start_ts_ms)
            if diff < best_diff:
                best_diff = diff
                best_entry = (ts_ms, px)
        if best_entry and best_diff <= 2000:
            start_price_exact = best_entry[1]

    if start_price_exact is not None:
        price_state["window_start_price"] = start_price_exact
        offset_ms = best_entry[0] - start_ts_ms
        print(f"  Start price: ${start_price_exact:,.2f} (Chainlink @ eventStart{offset_ms:+d}ms)")
    else:
        current_price = price_state.get("price")
        price_state["window_start_price"] = current_price
        print(f"  Start price: ${current_price:,.2f} (RTDS fallback — no exact match in buffer)"
              if current_price else "  Start price: waiting for RTDS...")

    flat_state = {
        "up_best_bid": None, "up_best_ask": None,
        "down_best_bid": None, "down_best_ask": None,
    }

    tracker.new_window(end)

    # Detect if we're joining mid-window on startup
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

    mode = "DRY RUN" if tracker.dry_run else "LIVE"
    print(f"  [{mode}] Market: {title}")
    print(f"  Window:   {start.strftime('%H:%M:%S')} -> {end.strftime('%H:%M:%S')} UTC")
    print(f"  Bankroll: ${tracker.bankroll:,.2f}  |  Min order: {tracker.min_order_shares:.0f} shares")

    cancel = asyncio.Event()
    tasks = [
        asyncio.create_task(
            clob_ws(up_token, down_token, book_up, book_down,
                    flat_state, cancel, debug=tracker.debug,
                    trade_state=trade_state)
        ),
        asyncio.create_task(
            signal_ticker(tracker, book_up, book_down, price_state,
                          end, slug, up_token, down_token, cancel,
                          skip_trading=skip_trading,
                          trade_state=trade_state)
        ),
        asyncio.create_task(
            display_ticker(tracker, price_state, flat_state,
                           title, start, end, cancel, config)
        ),
    ]

    now = datetime.now(timezone.utc)
    await asyncio.sleep(max(0, (end - now).total_seconds()) + 5)

    cancel.set()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    if tracker.flat_reason_counts:
        tracker._log_flat_summary()
        tracker.flat_reason_counts = {}

    if tracker.open_orders:
        tracker._cancel_open_orders()

    tracker.resolve_window(
        slug,
        price_state.get("price"),
        price_state.get("window_start_price"),
    )
    tracker.save_state()


async def run(tracker: LiveTradeTracker, config):
    price_state: dict = {
        "price": None,
        "window_start_price": None,
        "price_history": collections.deque(maxlen=600),
    }

    # Trade state persists across windows — VPIN needs 20+ min history
    vpin_bar_s = tracker.signal.vpin_bar_s
    trade_state: dict = {
        "bars": collections.deque(maxlen=200),
        "current_bar": {"buy_vol": 0.0, "sell_vol": 0.0, "start_ts": 0},
        "bar_duration_s": vpin_bar_s,
    }

    rtds_cancel = asyncio.Event()
    rtds_task = asyncio.create_task(
        rtds_ws(price_state, rtds_cancel, config, tracker,
                debug=tracker.debug)
    )

    try:
        while True:
            await run_window(tracker, config, price_state, trade_state)
    finally:
        rtds_cancel.set()
        rtds_task.cancel()
        await asyncio.gather(rtds_task, return_exceptions=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Live trading bot for Polymarket Up/Down markets"
    )
    parser.add_argument(
        "--market", default=DEFAULT_MARKET, choices=list(MARKET_CONFIGS),
        help="Market to trade (default: btc)",
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
        help="Max trades per window (default: 2)",
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
    parser.add_argument("--max-positions", type=int, default=4,
                        help="Max simultaneous filled positions (default: 4)")
    parser.add_argument("--inventory-skew", type=float, default=0.02,
                        help="Edge penalty per same-side position (default 0.02)")
    parser.add_argument("--maker-withdraw", type=float, default=120.0,
                        help="Stop new orders when tau < N seconds (default 120)")
    parser.add_argument("--toxicity-threshold", type=float, default=0.75,
                        help="Toxicity score above which edge threshold widens (default: 0.75)")
    parser.add_argument("--toxicity-edge-mult", type=float, default=1.5,
                        help="Max edge threshold multiplier at toxicity=1.0 (default: 1.5)")
    parser.add_argument("--vol-kill-sigma", type=float, default=None,
                        help="Absolute sigma ceiling — pause quoting above this (default: None)")
    parser.add_argument("--down-edge-bonus", type=float, default=0.15,
                        help="Fraction to reduce DOWN edge threshold (optimism tax, default: 0.15)")
    parser.add_argument("--regime-z-scale", action="store_true",
                        help="Scale z-scores by sigma_ema / sigma_calibration (requires --calibrated)")
    parser.add_argument("--vpin-threshold", type=float, default=0.50,
                        help="VPIN above which edge threshold widens (default: 0.50)")
    parser.add_argument("--vpin-edge-mult", type=float, default=1.5,
                        help="Max edge threshold multiplier at VPIN=1.0 (default: 1.5)")
    parser.add_argument("--vpin-window", type=int, default=20,
                        help="Number of completed trade bars for VPIN (default: 20)")
    parser.add_argument("--vpin-bar-s", type=float, default=60.0,
                        help="Trade bar duration in seconds (default: 60)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = get_config(args.market)
    trades_log = Path(f"live_trades_{config.data_subdir}.jsonl")
    state_file = Path(f"live_state_{config.data_subdir}.json")

    # Default max trades
    max_trades = args.max_trades_per_window if args.max_trades_per_window is not None else 2

    # Build authenticated client
    print(f"\n  {'='*62}")
    mode = "DRY RUN" if args.dry_run else "LIVE TRADING"
    order_mode = "MAKER"
    if args.calibrated:
        order_mode += "+CAL"
    if args.early_exit:
        order_mode += "+EXIT"
    print(f"  {mode} [{order_mode}] -- {config.display_name} Up/Down")
    print(f"  {'='*62}")

    client = build_clob_client()

    # Determine bankroll
    api_balance = query_usdc_balance(client, debug=args.debug)
    if api_balance is not None:
        print(f"  API USDC balance: ${api_balance:,.2f}")

    bankroll = args.bankroll
    saved = None
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

    # Per-market signal overrides
    signal_kw: dict = {}
    base_market = args.market.replace("_5m", "")
    if base_market == "eth":
        signal_kw = dict(
            edge_threshold=0.15,
            reversion_discount=0.10,
            momentum_lookback_s=15,
            momentum_majority=0.7,
            spread_edge_penalty=0.2,
        )

    # Maker signal overrides
    signal_kw["window_duration"] = config.window_duration_s
    signal_kw["maker_mode"] = True
    signal_kw["max_bet_fraction"] = 0.02
    signal_kw["edge_threshold"] = signal_kw.get("edge_threshold", 0.10)
    signal_kw["momentum_majority"] = 0.0
    signal_kw["spread_edge_penalty"] = signal_kw.get("spread_edge_penalty", 0.0)
    cooldown_ms = 30_000
    stale_timeout = 60.0

    # Calibration
    cal_table = None
    if args.calibrated:
        from backtest import DATA_DIR
        cal_data_dir = DATA_DIR / config.data_subdir
        print(f"  Building calibration table from {cal_data_dir} ...")
        cal_table = build_calibration_table(cal_data_dir, vol_lookback_s=90)
        n_cells = len(cal_table.table)
        n_obs = sum(cal_table.counts.values())
        print(f"  Calibration table: {n_cells} cells, {n_obs} observations")
        signal_kw["calibration_table"] = cal_table
        signal_kw["edge_threshold"] = 0.10
        signal_kw["early_edge_mult"] = 2.0

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

    # Compute sigma_calibration from recorded data when regime-z-scale is on
    if args.regime_z_scale and args.calibrated:
        from backtest import DATA_DIR, _compute_vol_deduped
        cal_data_dir_for_sigma = DATA_DIR / config.data_subdir
        MAX_CAL_FILES = 200      # cap to recent windows to keep startup fast
        MIN_SIGMA_CAL = 1e-7     # floor to prevent z-scaling explosion in quiet regimes
        try:
            sigmas = []
            files = sorted(cal_data_dir_for_sigma.glob("*.parquet"))
            files = files[-MAX_CAL_FILES:]  # use most recent N windows
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
                print(f"  Regime-z-scale: sigma_calibration={sigma_cal:.2e} "
                      f"(median of {len(sigmas)} windows, last {len(files)} files)")
            else:
                print("  WARNING: regime-z-scale enabled but no valid sigma data found")
        except Exception as exc:
            print(f"  WARNING: sigma_calibration computation failed: {exc}")

    # VAMP: BTC uses cost-based, ETH uses filter-based
    if base_market == "btc":
        signal_kw["vamp_mode"] = "cost"
    elif base_market == "eth":
        signal_kw["vamp_mode"] = "filter"
        signal_kw["vamp_filter_threshold"] = 0.03

    # 5m market overrides
    is_5m = "_5m" in args.market
    maker_warmup = args.maker_warmup
    maker_withdraw = args.maker_withdraw
    max_order_age = args.max_order_age
    requote_cooldown = args.requote_cooldown
    exit_min_hold = args.exit_min_hold
    exit_min_remaining = args.exit_min_remaining

    if is_5m:
        if base_market == "btc":
            maker_warmup = 70.0
            maker_withdraw = 30.0
        elif base_market == "eth":
            maker_warmup = 30.0
            maker_withdraw = 20.0
            signal_kw["edge_threshold"] = 0.12
            signal_kw["early_edge_mult"] = 2.5
            signal_kw["reversion_discount"] = 0.10
            max_trades = 2
        signal_kw["maker_warmup_s"] = maker_warmup
        signal_kw["maker_withdraw_s"] = maker_withdraw
        cooldown_ms = 10_000
        max_order_age = 40.0
        requote_cooldown = 2.0
        exit_min_hold = 10.0
        exit_min_remaining = 20.0

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
        debug=args.debug,
        dry_run=args.dry_run,
        trades_log=trades_log,
        state_file=state_file,
    )
    tracker.api_balance = api_balance
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

    # Gas balance check for on-chain redemption
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
                      f"need gas for CTF redemption txs. Send >= 0.01 POL "
                      f"to {signer.address}")
        else:
            print(f"  WARNING: Cannot connect to Polygon RPC ({POLYGON_RPC}) — "
                  f"auto-redemption may fail")
    except Exception as exc:
        print(f"  WARNING: Gas check failed: {exc}")

    print(f"  Bankroll:        ${bankroll:,.2f}")
    print(f"  Mode:            {order_mode}")
    print(f"  Window:          {config.window_duration_s:.0f}s ({'5m' if is_5m else '15m'})")
    print(f"  Max loss:        {args.max_loss_pct}%")
    print(f"  Max trades/win:  {max_trades}")
    print(f"  Max exposure:    {args.max_exposure_pct}%")
    print(f"  Maker warmup:    {maker_warmup:.0f}s")
    print(f"  Maker withdraw:  {maker_withdraw:.0f}s")
    print(f"  Max order age:   {max_order_age:.0f}s")
    if args.early_exit:
        print(f"  Early exit:      ON (threshold={args.exit_threshold}, "
              f"min_hold={exit_min_hold:.0f}s, "
              f"min_remaining={exit_min_remaining:.0f}s)")
        print(f"  Max positions:   {args.max_positions}")
    print(f"  Cooldown:        {cooldown_ms / 1000:.0f}s")
    print(f"  Stale timeout:   {stale_timeout:.0f}s")
    print(f"  VPIN:            thresh={args.vpin_threshold} mult={args.vpin_edge_mult} "
          f"window={args.vpin_window} bar={args.vpin_bar_s:.0f}s")
    print(f"  Trades log:      {trades_log}")
    print(f"  State file:      {state_file}")
    print()

    try:
        asyncio.run(run(tracker, config))
    except KeyboardInterrupt:
        print(f"\n  Shutting down...")
        if tracker.flat_reason_counts:
            tracker._log_flat_summary()
        tracker.cancel_all_orders()
        tracker.save_state()
        total_pnl = sum(r.pnl for r in tracker.all_results)
        print(f"  Session PnL: ${total_pnl:+,.2f}")
        print(f"  Final bankroll: ${tracker.bankroll:,.2f}")
        print(f"  State saved to {state_file}")
        print("  Exiting.")


if __name__ == "__main__":
    main()
