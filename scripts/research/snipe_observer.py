#!/usr/bin/env python3
"""
Snipe Observer: passive monitor of order book around BTC 5m window resolution.

Logs high-frequency snapshots from t-60s through t+120s after each window end.
Zero trading, zero risk. Designed to run 24+ hours unattended.

Output: data/snipe_research/observer_{slug}.jsonl

Usage:
    python scripts/research/snipe_observer.py              # run continuously
    python scripts/research/snipe_observer.py --windows 5  # observe 5 windows then exit
    python scripts/research/snipe_observer.py --debug      # verbose logging
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time as _time
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv()

from polybot_core import BookFeed, PriceFeed, BinanceFeed
from market_api import find_market, GAMMA_API, _ensure_list
from market_config import get_config
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = _PROJECT_ROOT / "data" / "snipe_research"

# Sampling rates
PRE_END_INTERVAL_S = 0.250   # 4 Hz during t-60 to t+0
POST_END_INTERVAL_S = 1.0    # 1 Hz during t+0 to t+120
GAMMA_POLL_INTERVAL_S = 5.0  # Gamma API every 5s in post phase

# Observation window boundaries (relative to market end)
PRE_END_START_S = 60         # start observing 60s before end
POST_END_DURATION_S = 120    # observe for 120s after end

# Feed startup grace period
FEED_WARMUP_S = 5.0

# How many price levels to capture per side
MAX_BOOK_LEVELS = 10

# Logging
LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_shutdown_requested = False


def _handle_signal(signum, _frame):
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    logging.info("Received %s, requesting shutdown after current window...", sig_name)
    _shutdown_requested = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_gamma_poll(slug: str) -> dict:
    """Poll Gamma API for resolution state. Returns dict of gamma fields.

    Never raises -- returns empty/null fields on any error.
    """
    result = {
        "gamma_closed": None,
        "gamma_uma_status": None,
        "gamma_outcome_prices": None,
    }
    try:
        resp = requests.get(
            f"{GAMMA_API}/events",
            params={"slug": slug},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return result
        market = data[0]["markets"][0]
        result["gamma_closed"] = market.get("closed")
        result["gamma_uma_status"] = market.get("umaResolutionStatus", "")
        raw_prices = market.get("outcomePrices")
        if raw_prices is not None:
            result["gamma_outcome_prices"] = _ensure_list(raw_prices)
    except Exception as exc:
        logging.debug("Gamma poll error for %s: %s", slug, exc)
    return result


def _book_levels(snap_bids_or_asks: list[tuple[float, float]]) -> list[list[float]]:
    """Extract up to MAX_BOOK_LEVELS from a BookSnapshot side.

    Returns list of [price, size] pairs (JSON-serializable).
    """
    return [[p, s] for p, s in snap_bids_or_asks[:MAX_BOOK_LEVELS]]


def _find_size_at_price(levels: list[tuple[float, float]], target: float,
                        tol: float = 0.001) -> float:
    """Find aggregate size at a specific price level (within tolerance)."""
    total = 0.0
    for price, size in levels:
        if abs(price - target) < tol:
            total += size
    return total


def _predict_winner(chainlink_price: float | None,
                    window_start_price: float | None) -> tuple[str | None, float | None]:
    """Predict UP/DOWN winner and compute delta.

    Returns (predicted_winner, delta_usd).
    """
    if chainlink_price is None or window_start_price is None:
        return None, None
    delta = chainlink_price - window_start_price
    if delta > 0:
        return "UP", delta
    elif delta < 0:
        return "DOWN", delta
    else:
        return None, 0.0


def _build_row(
    *,
    ts_ms: int,
    phase: str,
    time_to_end_s: float,
    market_slug: str,
    chainlink_price: float | None,
    window_start_price: float | None,
    binance_mid: float | None,
    snap_up,
    snap_down,
    predicted_winner: str | None,
    delta_usd: float | None,
    gamma_state: dict | None,
) -> dict:
    """Assemble a single JSONL row from current feed state."""

    # Winner-side liquidity at key price points
    winner_ask_99 = 0.0
    winner_ask_98 = 0.0
    winner_bid_99 = 0.0
    winner_bid_98 = 0.0

    if predicted_winner == "UP" and snap_up is not None:
        winner_ask_99 = _find_size_at_price(snap_up.asks, 0.99)
        winner_ask_98 = _find_size_at_price(snap_up.asks, 0.98)
        winner_bid_99 = _find_size_at_price(snap_up.bids, 0.99)
        winner_bid_98 = _find_size_at_price(snap_up.bids, 0.98)
    elif predicted_winner == "DOWN" and snap_down is not None:
        winner_ask_99 = _find_size_at_price(snap_down.asks, 0.99)
        winner_ask_98 = _find_size_at_price(snap_down.asks, 0.98)
        winner_bid_99 = _find_size_at_price(snap_down.bids, 0.99)
        winner_bid_98 = _find_size_at_price(snap_down.bids, 0.98)

    row = {
        "ts_ms": ts_ms,
        "phase": phase,
        "time_to_end_s": round(time_to_end_s, 3),
        "market_slug": market_slug,
        # Price data
        "chainlink_price": chainlink_price,
        "window_start_price": window_start_price,
        "delta_usd": round(delta_usd, 4) if delta_usd is not None else None,
        "predicted_winner": predicted_winner,
        "binance_mid": binance_mid,
        # UP side book
        "up_bids": _book_levels(snap_up.bids) if snap_up else [],
        "up_asks": _book_levels(snap_up.asks) if snap_up else [],
        "up_best_bid": snap_up.best_bid if snap_up else None,
        "up_best_ask": snap_up.best_ask if snap_up else None,
        # DOWN side book
        "down_bids": _book_levels(snap_down.bids) if snap_down else [],
        "down_asks": _book_levels(snap_down.asks) if snap_down else [],
        "down_best_bid": snap_down.best_bid if snap_down else None,
        "down_best_ask": snap_down.best_ask if snap_down else None,
        # Winner-side liquidity at key levels
        "winner_ask_99": winner_ask_99,
        "winner_ask_98": winner_ask_98,
        "winner_bid_99": winner_bid_99,
        "winner_bid_98": winner_bid_98,
        # Gamma API state (populated in post_end phase only)
        "gamma_closed": None,
        "gamma_uma_status": None,
        "gamma_outcome_prices": None,
    }

    if gamma_state:
        row["gamma_closed"] = gamma_state.get("gamma_closed")
        row["gamma_uma_status"] = gamma_state.get("gamma_uma_status")
        row["gamma_outcome_prices"] = gamma_state.get("gamma_outcome_prices")

    return row


# ---------------------------------------------------------------------------
# Core observation loop
# ---------------------------------------------------------------------------

def _sample_feeds(
    book_feed: BookFeed,
    price_feed: PriceFeed,
    binance_feed: BinanceFeed,
    up_token: str,
    down_token: str,
):
    """Read all three feeds. Returns (snap_up, snap_down, chainlink, binance_mid).

    Each value may be None if the feed has no data yet.
    """
    snap_up = None
    snap_down = None
    chainlink = None
    binance_mid = None

    try:
        snap_up = book_feed.snapshot(up_token)
    except Exception as exc:
        logging.debug("BookFeed snapshot(up) error: %s", exc)

    try:
        snap_down = book_feed.snapshot(down_token)
    except Exception as exc:
        logging.debug("BookFeed snapshot(down) error: %s", exc)

    try:
        chainlink = price_feed.price()
    except Exception as exc:
        logging.debug("PriceFeed error: %s", exc)

    try:
        binance_mid = binance_feed.mid()
    except Exception as exc:
        logging.debug("BinanceFeed error: %s", exc)

    return snap_up, snap_down, chainlink, binance_mid


def observe_window(
    config,
    book_feed: BookFeed,
    price_feed: PriceFeed,
    binance_feed: BinanceFeed,
    slug: str,
    up_token: str,
    down_token: str,
    end: datetime,
    debug: bool = False,
) -> bool:
    """Observe a single window's resolution boundary.

    All market metadata (slug, tokens, end time) is passed in from the caller
    to stay consistent with the BookFeed's token subscriptions.

    Returns True if observation completed, False on skip.
    """
    now = datetime.now(timezone.utc)
    time_to_end = (end - now).total_seconds()

    logging.info("Observing market: %s (end in %.0fs)", slug, time_to_end)

    # If the market already ended more than POST_END_DURATION_S ago, skip it
    if time_to_end < -POST_END_DURATION_S:
        logging.info("Market %s already resolved, skipping", slug)
        return False

    # 2. Wait until t - PRE_END_START_S
    observe_start = end - timedelta(seconds=PRE_END_START_S)
    wait_s = (observe_start - now).total_seconds()
    if wait_s > 0:
        logging.info("Waiting %.0fs until observation window for %s...", wait_s, slug)
        # Sleep in small increments to allow shutdown checks
        deadline = _time.monotonic() + wait_s
        while _time.monotonic() < deadline:
            if _shutdown_requested:
                logging.info("Shutdown requested during wait")
                return True
            _time.sleep(min(1.0, deadline - _time.monotonic()))

    # 3. Capture window_start_price from PriceFeed at the start of observation.
    #    Ideally this is the price at eventStartTime, but since we may join
    #    mid-window, use the current chainlink as approximation. The "real"
    #    start price is whatever Polymarket locked at eventStartTime -- we
    #    capture it from chainlink at the moment we begin observing. For
    #    delta_usd accuracy, what matters is the relative movement, so
    #    capturing at observation start is acceptable.
    window_start_price = price_feed.price()
    if window_start_price is None:
        # Wait briefly for feed warmup
        warmup_deadline = _time.monotonic() + FEED_WARMUP_S
        while _time.monotonic() < warmup_deadline and window_start_price is None:
            _time.sleep(0.1)
            window_start_price = price_feed.price()
        if window_start_price is None:
            logging.warning("PriceFeed has no data after warmup, using 0 as placeholder")

    logging.info("Observation started for %s (window_start_price=%s)",
                 slug, window_start_price)

    # Prepare output file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    outfile = DATA_DIR / f"observer_{slug}.jsonl"
    rows_written = 0

    # Open file in append mode so we don't clobber data if the script restarts
    # on the same window
    with open(outfile, "a") as f:

        # --- Phase 1: Pre-end high-frequency sampling (t-60s to t+0s) ---
        logging.info("Phase PRE_END: sampling at %.0fms intervals", PRE_END_INTERVAL_S * 1000)
        while not _shutdown_requested:
            now = datetime.now(timezone.utc)
            time_to_end_s = (end - now).total_seconds()

            if time_to_end_s <= 0:
                break  # transition to post-end phase

            snap_up, snap_down, chainlink, binance_mid = _sample_feeds(
                book_feed, price_feed, binance_feed, up_token, down_token,
            )
            predicted_winner, delta_usd = _predict_winner(chainlink, window_start_price)

            row = _build_row(
                ts_ms=int(_time.time() * 1000),
                phase="pre_end",
                time_to_end_s=-abs(time_to_end_s),  # negative = before end
                market_slug=slug,
                chainlink_price=chainlink,
                window_start_price=window_start_price,
                binance_mid=binance_mid,
                snap_up=snap_up,
                snap_down=snap_down,
                predicted_winner=predicted_winner,
                delta_usd=delta_usd,
                gamma_state=None,
            )
            f.write(json.dumps(row) + "\n")
            rows_written += 1

            if debug and rows_written % 40 == 0:
                logging.debug(
                    "  pre_end sample %d: t=%.1fs, cl=%s, bn=%s, winner=%s",
                    rows_written, time_to_end_s,
                    f"{chainlink:.2f}" if chainlink else "None",
                    f"{binance_mid:.2f}" if binance_mid else "None",
                    predicted_winner,
                )

            _time.sleep(PRE_END_INTERVAL_S)

        if _shutdown_requested:
            logging.info("Shutdown during pre_end phase (%d rows written)", rows_written)
            f.flush()
            return True

        # --- Phase 2: Post-end sampling (t+0s to t+120s) ---
        logging.info("Phase POST_END: sampling at %.0fms intervals + Gamma API every %.0fs",
                     POST_END_INTERVAL_S * 1000, GAMMA_POLL_INTERVAL_S)

        last_gamma_poll = 0.0
        gamma_state: dict | None = None
        post_end_deadline = end + timedelta(seconds=POST_END_DURATION_S)

        while not _shutdown_requested:
            now = datetime.now(timezone.utc)
            if now >= post_end_deadline:
                break

            time_to_end_s = (now - end).total_seconds()  # positive = after end

            # Poll Gamma API periodically
            mono_now = _time.monotonic()
            if mono_now - last_gamma_poll >= GAMMA_POLL_INTERVAL_S:
                gamma_state = _safe_gamma_poll(slug)
                last_gamma_poll = mono_now
                if debug:
                    logging.debug(
                        "  Gamma poll: closed=%s, uma=%s, prices=%s",
                        gamma_state.get("gamma_closed"),
                        gamma_state.get("gamma_uma_status"),
                        gamma_state.get("gamma_outcome_prices"),
                    )

            snap_up, snap_down, chainlink, binance_mid = _sample_feeds(
                book_feed, price_feed, binance_feed, up_token, down_token,
            )
            predicted_winner, delta_usd = _predict_winner(chainlink, window_start_price)

            row = _build_row(
                ts_ms=int(_time.time() * 1000),
                phase="post_end",
                time_to_end_s=abs(time_to_end_s),  # positive = after end
                market_slug=slug,
                chainlink_price=chainlink,
                window_start_price=window_start_price,
                binance_mid=binance_mid,
                snap_up=snap_up,
                snap_down=snap_down,
                predicted_winner=predicted_winner,
                delta_usd=delta_usd,
                gamma_state=gamma_state,
            )
            f.write(json.dumps(row) + "\n")
            rows_written += 1

            if debug and rows_written % 10 == 0:
                logging.debug(
                    "  post_end sample %d: t=+%.1fs, cl=%s, winner=%s, closed=%s",
                    rows_written, time_to_end_s,
                    f"{chainlink:.2f}" if chainlink else "None",
                    predicted_winner,
                    gamma_state.get("gamma_closed") if gamma_state else "?",
                )

            _time.sleep(POST_END_INTERVAL_S)

        f.flush()

    logging.info("Window %s complete: %d rows -> %s", slug, rows_written, outfile)
    return True


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Snipe Observer: passive book monitor around BTC 5m resolution",
    )
    parser.add_argument(
        "--windows", type=int, default=0,
        help="Number of windows to observe (0=infinite, default=0)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable verbose debug logging",
    )
    parser.add_argument(
        "--market", default="btc_5m",
        help="Market config key (default: btc_5m)",
    )
    args = parser.parse_args()

    # Logging setup
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format=LOG_FMT, datefmt=LOG_DATEFMT)

    # Signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Load config
    try:
        config = get_config(args.market)
    except KeyError:
        logging.error("Unknown market config: %s", args.market)
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Snipe Observer starting (market=%s, windows=%s)",
                 args.market, args.windows or "infinite")
    logging.info("Output directory: %s", DATA_DIR)

    # Start shared Rust feeds. These run background WS threads that auto-reconnect.
    # We create them once and reuse across windows to avoid repeated connection churn.
    logging.info("Starting feeds...")
    price_feed = PriceFeed(config.chainlink_symbol)
    logging.info("  PriceFeed started (%s)", config.chainlink_symbol)

    binance_feed = BinanceFeed(config.binance_symbol)
    logging.info("  BinanceFeed started (%s)", config.binance_symbol)

    # BookFeed needs token IDs, which change per window. We'll create it
    # per-window since each 5m market has different token IDs.
    # But PriceFeed and BinanceFeed are asset-level and reusable.

    # Wait for initial feed data
    logging.info("Waiting for feed warmup...")
    warmup_deadline = _time.monotonic() + 15.0
    while _time.monotonic() < warmup_deadline:
        if price_feed.price() is not None:
            logging.info("  PriceFeed ready (price=%.2f)", price_feed.price())
            break
        _time.sleep(0.5)
    else:
        logging.warning("PriceFeed did not produce data within 15s -- continuing anyway")

    binance_ok = False
    warmup_deadline = _time.monotonic() + 10.0
    while _time.monotonic() < warmup_deadline:
        if binance_feed.mid() is not None:
            logging.info("  BinanceFeed ready (mid=%.2f)", binance_feed.mid())
            binance_ok = True
            break
        _time.sleep(0.5)
    if not binance_ok:
        logging.warning("BinanceFeed did not produce data within 10s -- continuing anyway")

    # Main window loop
    n_completed = 0
    while not _shutdown_requested:
        if args.windows > 0 and n_completed >= args.windows:
            break

        # Find the current market and create a BookFeed for its tokens
        try:
            event, market = find_market(config)
        except Exception as exc:
            logging.error("find_market error: %s -- retrying in 30s", exc)
            _time.sleep(30)
            continue

        if not event or not market:
            logging.info("No active market found, retrying in 30s...")
            _time.sleep(30)
            continue

        # Extract tokens for BookFeed
        outcomes = _ensure_list(market["outcomes"])
        tokens = _ensure_list(market["clobTokenIds"])
        outcomes_lower = [o.lower() for o in outcomes]
        try:
            up_idx = outcomes_lower.index("up")
            down_idx = outcomes_lower.index("down")
        except ValueError:
            logging.error("Unexpected outcomes %s, retrying in 30s...", outcomes)
            _time.sleep(30)
            continue

        up_token = tokens[up_idx]
        down_token = tokens[down_idx]

        # Create BookFeed for this specific window's tokens
        try:
            book_feed = BookFeed([up_token, down_token])
            logging.info("BookFeed started (up=%s..., down=%s...)",
                         up_token[:12], down_token[:12])
        except Exception as exc:
            logging.error("BookFeed creation failed: %s -- retrying in 30s", exc)
            _time.sleep(30)
            continue

        # Parse market timing
        end = datetime.fromisoformat(market["endDate"].replace("Z", "+00:00"))
        slug = event["slug"]

        # Brief warmup for BookFeed
        bf_deadline = _time.monotonic() + FEED_WARMUP_S
        while _time.monotonic() < bf_deadline:
            try:
                s = book_feed.snapshot(up_token)
                if s and (s.best_bid is not None or s.best_ask is not None):
                    break
            except Exception:
                pass
            _time.sleep(0.25)

        # Observe this window
        completed = observe_window(
            config, book_feed, price_feed, binance_feed,
            slug=slug,
            up_token=up_token,
            down_token=down_token,
            end=end,
            debug=args.debug,
        )

        if completed:
            n_completed += 1
            logging.info("Completed %d/%s windows",
                         n_completed, args.windows or "inf")

        # Small pause between windows to avoid tight-looping on the same market
        if not _shutdown_requested:
            _time.sleep(5)

    logging.info("Snipe Observer shutting down. Observed %d windows total.", n_completed)


if __name__ == "__main__":
    main()
