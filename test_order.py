#!/usr/bin/env python3
"""
Test Order — Place a single limit order on a live Polymarket market and exit.

Discovers the current 15-minute Up/Down market, places a tiny limit order
on the Up outcome far below the current ask (so it sits on the book and
doesn't fill), prints the response, then cancels it and exits.

Usage:
    py -3 test_order.py                          # BTC, limit buy Up @ $0.01
    py -3 test_order.py --market eth              # ETH market
    py -3 test_order.py --side down               # buy Down instead
    py -3 test_order.py --price 0.50 --size 10    # custom price/size
    py -3 test_order.py --market-order --amount 1  # $1 FOK market order
    py -3 test_order.py --no-cancel                # leave order on book
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta

import requests
from dotenv import load_dotenv

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, MarketOrderArgs, OrderType, ApiCreds
from py_clob_client.order_builder.constants import BUY

from market_config import MARKET_CONFIGS, DEFAULT_MARKET, get_config

load_dotenv()

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137


# ── Market discovery (reused from paper_trader) ─────────────────────────────

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


def find_market(config):
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
            if now < end:
                return event, market

    return None, None


def _ensure_list(val):
    return json.loads(val) if isinstance(val, str) else val


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Place a single test order on Polymarket and exit"
    )
    parser.add_argument(
        "--market", default=DEFAULT_MARKET, choices=list(MARKET_CONFIGS),
        help="Market (default: btc)",
    )
    parser.add_argument(
        "--side", default="up", choices=["up", "down"],
        help="Which outcome to buy (default: up)",
    )
    parser.add_argument(
        "--price", type=float, default=0.01,
        help="Limit price per share 0.01-0.99 (default: 0.01 — sits far from market)",
    )
    parser.add_argument(
        "--size", type=float, default=5.0,
        help="Number of shares for limit order (default: 5)",
    )
    parser.add_argument(
        "--market-order", action="store_true",
        help="Send a FOK market order instead of a limit order",
    )
    parser.add_argument(
        "--amount", type=float, default=1.0,
        help="Dollar amount for market order (default: $1)",
    )
    parser.add_argument(
        "--no-cancel", action="store_true",
        help="Don't cancel the order after placing it",
    )
    args = parser.parse_args()

    # ── Load credentials ─────────────────────────────────────────────────
    private_key = os.getenv("PRIVATE_KEY")
    if not private_key:
        print("ERROR: PRIVATE_KEY not set in .env")
        sys.exit(1)

    # Show the signer address derived from this key so you can verify
    from eth_account import Account
    signer_addr = Account.from_key(private_key).address
    print(f"  Signer address: {signer_addr}")

    funder = os.getenv("POLY_FUNDER", "")
    sig_type = int(os.getenv("SIGNATURE_TYPE", "0"))

    if not funder:
        print("  ERROR: POLY_FUNDER not set in .env")
        print("         Go to Polymarket -> Wallet/Deposit -> copy the Polygon address")
        sys.exit(1)

    print(f"  Funder (proxy): {funder}")
    print(f"  Signature type: {sig_type} ({'EOA' if sig_type == 0 else 'PROXY' if sig_type == 1 else 'GNOSIS_SAFE'})")

    # ── Build client ─────────────────────────────────────────────────────
    client_kwargs = dict(
        host=CLOB_HOST,
        key=private_key,
        chain_id=CHAIN_ID,
        signature_type=sig_type,
    )
    if funder:
        client_kwargs["funder"] = funder

    client = ClobClient(**client_kwargs)

    # API credentials: load from .env or derive
    api_key = os.getenv("POLY_API_KEY", "")
    api_secret = os.getenv("POLY_API_SECRET", "")
    passphrase = os.getenv("POLY_PASSPHRASE", "")

    if api_key and api_secret and passphrase:
        print("  Using API credentials from .env")
        creds = ApiCreds(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=passphrase,
        )
        client.set_api_creds(creds)
    else:
        print("  Deriving API credentials (L1 auth)...")
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)
        print(f"  API Key:      {creds.api_key}")
        print(f"  API Secret:   {creds.api_secret}")
        print(f"  Passphrase:   {creds.api_passphrase}")
        print()
        print("  Save these in your .env to skip derivation next time:")
        print(f"    POLY_API_KEY={creds.api_key}")
        print(f"    POLY_API_SECRET={creds.api_secret}")
        print(f"    POLY_PASSPHRASE={creds.api_passphrase}")
        print()

    # ── Verify connection ────────────────────────────────────────────────
    try:
        ok = client.get_ok()
        print(f"  CLOB health check: {ok}")
    except Exception as exc:
        print(f"  WARNING: health check failed: {exc}")

    # ── Find market ──────────────────────────────────────────────────────
    config = get_config(args.market)
    print(f"\n  Searching for {config.display_name} 15-minute market...")
    event, market = find_market(config)

    if not event or not market:
        print("  ERROR: No active market found. Is the market open?")
        sys.exit(1)

    slug = event["slug"]
    title = event["title"]
    outcomes = _ensure_list(market["outcomes"])
    tokens = _ensure_list(market["clobTokenIds"])

    up_idx = outcomes.index("Up")
    down_idx = outcomes.index("Down")
    up_token = tokens[up_idx]
    down_token = tokens[down_idx]

    target_side = args.side.lower()
    token_id = up_token if target_side == "up" else down_token

    end = datetime.fromisoformat(market["endDate"].replace("Z", "+00:00"))
    remaining = (end - datetime.now(timezone.utc)).total_seconds()

    print(f"  Market:    {title}")
    print(f"  Slug:      {slug}")
    print(f"  Remaining: {remaining:.0f}s")
    print(f"  Token Up:  {up_token}")
    print(f"  Token Down:{down_token}")
    print(f"  Target:    {target_side.upper()} (token {token_id[:20]}...)")
    print()

    # ── Place order ──────────────────────────────────────────────────────
    if args.market_order:
        print(f"  Placing FOK MARKET ORDER: BUY {target_side.upper()} ${args.amount:.2f}")
        order_args = MarketOrderArgs(
            token_id=token_id,
            amount=args.amount,
            side=BUY,
        )
        signed_order = client.create_market_order(order_args)
        resp = client.post_order(signed_order, OrderType.FOK)
    else:
        print(f"  Placing GTC LIMIT ORDER: BUY {args.size} {target_side.upper()} @ {args.price}")
        order_args = OrderArgs(
            token_id=token_id,
            price=args.price,
            size=args.size,
            side=BUY,
        )
        signed_order = client.create_order(order_args)
        resp = client.post_order(signed_order, OrderType.GTC)

    print(f"\n  Response:")
    print(f"    {json.dumps(resp, indent=2)}")

    # ── Cancel ───────────────────────────────────────────────────────────
    if not args.no_cancel and not args.market_order:
        order_id = resp.get("orderID") or resp.get("id")
        if order_id:
            print(f"\n  Cancelling order {order_id}...")
            cancel_resp = client.cancel(order_id)
            print(f"    {cancel_resp}")
        else:
            print("\n  No order ID in response, skipping cancel.")

    print("\n  Done.")


if __name__ == "__main__":
    main()
