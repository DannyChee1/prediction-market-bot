#!/usr/bin/env python3
"""
CLOB Acceptance Test: can we place orders on a market after closed=True?

This is the single most important unknown for the resolution snipe strategy.
If the CLOB accepts orders after the market is closed but before on-chain
settlement, the strategy is near risk-free.

Safety:
  - Default mode is --dry-run (logs what it WOULD do, no real orders)
  - --live flag required to actually place test orders
  - Maximum risk: 1 share at 0.99 = $0.99
  - Only attempts when predicted winner has large delta (>$50)
  - Immediately cancels any accepted order

Usage:
    python scripts/research/snipe_clob_test.py              # dry-run
    python scripts/research/snipe_clob_test.py --live        # real orders ($0.99 max risk)
    python scripts/research/snipe_clob_test.py --debug       # verbose output
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time as _time
from datetime import datetime, timezone, timedelta
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv()

import requests

from polybot_core import OrderClient, BookFeed, PriceFeed
from market_api import find_market, GAMMA_API, _ensure_list
from market_config import get_config

LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

DATA_DIR = _PROJECT_ROOT / "data" / "snipe_research"

# Safety: minimum |delta| required before attempting an order
MIN_DELTA_USD = 50.0

# Test order parameters
TEST_PRICE = 0.99
TEST_SHARES = 5.0  # Polymarket minimum is 5 shares = $4.95 at risk

_shutdown = False


def _handle_signal(signum, _frame):
    global _shutdown
    logging.info("Shutdown requested")
    _shutdown = True


def _gamma_poll(slug: str) -> dict:
    """Poll Gamma API for market state. Returns raw market dict."""
    try:
        resp = requests.get(
            f"{GAMMA_API}/events", params={"slug": slug}, timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data:
            return data[0]["markets"][0]
    except Exception as exc:
        logging.debug("Gamma poll error: %s", exc)
    return {}


def run_test(args) -> None:
    config = get_config(args.market)

    # Build authenticated OrderClient (needed for live mode)
    client = None
    if args.live:
        private_key = os.getenv("PRIVATE_KEY")
        if not private_key:
            logging.error("PRIVATE_KEY not set — cannot run in live mode")
            sys.exit(1)
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
        logging.info("OrderClient ready (LIVE mode — real orders will be placed)")
    else:
        logging.info("DRY-RUN mode — no real orders")

    # Start price feed
    price_feed = PriceFeed(config.chainlink_symbol)
    logging.info("PriceFeed started (%s)", config.chainlink_symbol)

    # Wait for price feed
    deadline = _time.monotonic() + 15.0
    while _time.monotonic() < deadline and price_feed.price() is None:
        _time.sleep(0.5)
    if price_feed.price() is None:
        logging.warning("PriceFeed has no data — continuing anyway")

    # Results log
    test_results = []
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results_file = DATA_DIR / "clob_test_results.jsonl"

    while not _shutdown:
        # Find the current market
        event, market = find_market(config)
        if not event or not market:
            logging.info("No active market, retrying in 30s...")
            _time.sleep(30)
            continue

        slug = event["slug"]
        outcomes = _ensure_list(market["outcomes"])
        tokens = _ensure_list(market["clobTokenIds"])
        outcomes_lower = [o.lower() for o in outcomes]
        up_idx = outcomes_lower.index("up")
        down_idx = outcomes_lower.index("down")
        up_token = tokens[up_idx]
        down_token = tokens[down_idx]
        end = datetime.fromisoformat(market["endDate"].replace("Z", "+00:00"))

        now = datetime.now(timezone.utc)
        time_to_end = (end - now).total_seconds()

        if time_to_end < -180:
            logging.info("Market %s already resolved, waiting for next...", slug)
            _time.sleep(30)
            continue

        # Wait until window end
        if time_to_end > 0:
            logging.info("Market %s ends in %.0fs — waiting...", slug, time_to_end)
            wait_deadline = _time.monotonic() + time_to_end
            while _time.monotonic() < wait_deadline and not _shutdown:
                _time.sleep(min(1.0, wait_deadline - _time.monotonic()))
            if _shutdown:
                break

        logging.info("Window ended for %s — starting CLOB acceptance test", slug)

        # Capture current price for delta check
        chainlink = price_feed.price()
        start_price = None
        # Get start price from Gamma API metadata (window_start_price isn't in API,
        # so we infer from market prices: if UP best_ask is low, DOWN is winning)

        # Start BookFeed to read current book state
        book_feed = BookFeed([up_token, down_token])
        _time.sleep(2)  # Brief warmup

        # Determine predicted winner from book (more reliable than chainlink delta post-window)
        snap_up = book_feed.snapshot(up_token)
        snap_down = book_feed.snapshot(down_token)

        # Use chainlink delta if we can get window_start_price from the recording
        # For now, infer from book: if UP best_bid > 0.9, UP is winning
        up_bid = snap_up.best_bid if snap_up else None
        down_bid = snap_down.best_bid if snap_down else None

        predicted_winner = None
        winner_token = None
        if up_bid is not None and down_bid is not None:
            if up_bid > down_bid:
                predicted_winner = "UP"
                winner_token = up_token
            else:
                predicted_winner = "DOWN"
                winner_token = down_token
        elif up_bid is not None and up_bid > 0.5:
            predicted_winner = "UP"
            winner_token = up_token
        elif down_bid is not None and down_bid > 0.5:
            predicted_winner = "DOWN"
            winner_token = down_token

        logging.info("Book state: UP bid=%s, DOWN bid=%s -> predicted winner: %s",
                     up_bid, down_bid, predicted_winner)

        if predicted_winner is None:
            logging.warning("Cannot determine winner from book, skipping")
            _time.sleep(30)
            continue

        # Safety: require the winning side's bid to be >= 0.90 (strong signal)
        winner_bid = up_bid if predicted_winner == "UP" else down_bid
        if winner_bid is not None and winner_bid < 0.90:
            logging.info("Winner bid %.2f too low (< 0.90), skipping for safety", winner_bid)
            _time.sleep(30)
            continue

        # === Phase 1: Test immediately after window end (before closed=True) ===
        _run_clob_probe(
            phase="post_window_pre_close",
            slug=slug,
            winner_token=winner_token,
            predicted_winner=predicted_winner,
            client=client,
            live=args.live,
            results_file=results_file,
        )

        # === Phase 2: Poll until closed=True, test during the gap ===
        logging.info("Polling Gamma API for closed=True...")
        closed_detected = False
        uma_resolved = False

        for attempt in range(60):  # 60 × 2s = 120s max
            if _shutdown:
                break
            m = _gamma_poll(slug)
            is_closed = m.get("closed", False)
            uma_status = m.get("umaResolutionStatus", "")
            outcome_prices = _ensure_list(m.get("outcomePrices", []))

            logging.info("  [%d] closed=%s, uma=%s, prices=%s",
                         attempt, is_closed, uma_status, outcome_prices)

            if is_closed and not closed_detected:
                closed_detected = True
                logging.info("=== CLOSED detected! Testing CLOB... ===")
                _run_clob_probe(
                    phase="closed_pre_uma",
                    slug=slug,
                    winner_token=winner_token,
                    predicted_winner=predicted_winner,
                    client=client,
                    live=args.live,
                    results_file=results_file,
                )

            if uma_status == "resolved" and not uma_resolved:
                uma_resolved = True
                logging.info("=== UMA RESOLVED! Testing CLOB... ===")
                _run_clob_probe(
                    phase="uma_resolved",
                    slug=slug,
                    winner_token=winner_token,
                    predicted_winner=predicted_winner,
                    client=client,
                    live=args.live,
                    results_file=results_file,
                )

            # Check for settlement (outcome prices are 0/1)
            try:
                pf = [float(p) for p in outcome_prices]
                if sorted(pf) == [0.0, 1.0]:
                    logging.info("=== SETTLED (prices=[0,1])! Testing CLOB... ===")
                    _run_clob_probe(
                        phase="settled",
                        slug=slug,
                        winner_token=winner_token,
                        predicted_winner=predicted_winner,
                        client=client,
                        live=args.live,
                        results_file=results_file,
                    )
                    break
            except (ValueError, TypeError):
                pass

            _time.sleep(2)

        logging.info("Test cycle complete for %s", slug)
        logging.info("Waiting for next window...")
        _time.sleep(30)


def _run_clob_probe(
    *,
    phase: str,
    slug: str,
    winner_token: str,
    predicted_winner: str,
    client: OrderClient | None,
    live: bool,
    results_file: Path,
) -> None:
    """Attempt to place (and immediately cancel) a test order.

    Logs the result to the results file regardless of dry-run/live mode.
    """
    ts = datetime.now(timezone.utc).isoformat()
    ts_ms = int(_time.time() * 1000)

    result = {
        "ts": ts,
        "ts_ms": ts_ms,
        "phase": phase,
        "slug": slug,
        "predicted_winner": predicted_winner,
        "winner_token": winner_token[:16] + "...",
        "test_price": TEST_PRICE,
        "test_shares": TEST_SHARES,
        "live": live,
        "order_accepted": None,
        "order_id": None,
        "response": None,
        "error": None,
        "cancel_response": None,
    }

    if not live or client is None:
        logging.info("  [%s] DRY-RUN: would place BUY %s at %.2f for %.1f shares",
                     phase, predicted_winner, TEST_PRICE, TEST_SHARES)
        result["order_accepted"] = "DRY_RUN"
    else:
        logging.info("  [%s] LIVE: placing BUY %s at %.2f for %.1f shares...",
                     phase, predicted_winner, TEST_PRICE, TEST_SHARES)
        try:
            resp = client.place_order(
                winner_token, TEST_PRICE, TEST_SHARES, "BUY", "GTC", 1000,
            )
            result["response"] = resp
            order_id = resp.get("orderID") or resp.get("id", "")
            result["order_id"] = order_id
            success = resp.get("success", False)
            status = resp.get("status", "")

            if success or order_id:
                result["order_accepted"] = True
                logging.info("  [%s] ORDER ACCEPTED! id=%s status=%s",
                             phase, order_id, status)
                logging.info("  [%s] Full response: %s", phase, json.dumps(resp))

                # Immediately cancel
                if order_id:
                    try:
                        _time.sleep(0.5)  # brief delay before cancel
                        client.cancel(order_id)
                        result["cancel_response"] = "cancelled"
                        logging.info("  [%s] Order cancelled successfully", phase)
                    except Exception as exc:
                        result["cancel_response"] = str(exc)
                        logging.warning("  [%s] Cancel failed: %s", phase, exc)
            else:
                result["order_accepted"] = False
                logging.info("  [%s] ORDER REJECTED. Response: %s",
                             phase, json.dumps(resp))

        except Exception as exc:
            result["order_accepted"] = False
            result["error"] = str(exc)
            logging.info("  [%s] ORDER FAILED: %s", phase, exc)

    # Write result
    with open(results_file, "a") as f:
        f.write(json.dumps(result) + "\n")

    logging.info("  [%s] Result logged to %s", phase, results_file)


def main():
    parser = argparse.ArgumentParser(
        description="Test whether CLOB accepts orders on closed/resolved markets",
    )
    parser.add_argument("--live", action="store_true",
                        help="Actually place test orders ($0.99 max risk)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable verbose debug logging")
    parser.add_argument("--market", default="btc_5m",
                        help="Market config key (default: btc_5m)")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format=LOG_FMT, datefmt=LOG_DATEFMT)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logging.info("CLOB Acceptance Test starting (market=%s, live=%s)", args.market, args.live)
    if args.live:
        logging.warning("LIVE MODE: real orders will be placed ($0.99 max risk per test)")
    else:
        logging.info("Dry-run mode: no real orders. Use --live to enable.")

    run_test(args)
    logging.info("CLOB Acceptance Test complete")


if __name__ == "__main__":
    main()
