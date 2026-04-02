#!/usr/bin/env python3
"""Trace bankroll through fills and resolutions."""
import json
from pathlib import Path

BASE = Path(r"C:\Users\d4nny\Desktop\prediction-market-bot")

for asset in ("btc", "eth"):
    f = BASE / f"live_trades_{asset}.jsonl"
    lines = []
    for raw in f.read_text().strip().split("\n"):
        try:
            lines.append(json.loads(raw))
        except:
            pass

    print(f"\n{'='*90}")
    print(f"  {asset.upper()} BANKROLL TRACE")
    print(f"{'='*90}")

    for l in lines:
        t = l.get("type")
        if t == "limit_fill":
            cost = l.get("cost_usd", 0)
            side = l.get("side", "?")
            price = l.get("price", 0)
            ts = l.get("ts", "")[:19]
            print(f"  {ts}  FILL  {side:>4} @ {price:.3f}  cost=${cost:.2f}")
        elif t == "resolution":
            pnl = l.get("pnl", 0)
            payout = l.get("payout", 0)
            cost = l.get("cost_usd", 0)
            bankroll = l.get("bankroll_after", 0)
            side = l.get("side", "?")
            outcome = l.get("outcome", "?")
            slug = l.get("market_slug", "")
            tag = "WIN" if pnl > 0 else "LOSS"
            print(f"  {ts}  {tag:>4}  {side:>4}  cost=${cost:.2f}  "
                  f"payout=${payout:.2f}  pnl=${pnl:+.2f}  "
                  f"bankroll=${bankroll:.2f}  {slug}")
        elif t == "redemption":
            ts = l.get("ts", "")[:19]
            slug = l.get("market_slug", "")
            print(f"  {ts}  REDEEMED  {slug}")
        elif t == "redemption_abandoned":
            ts = l.get("ts", "")[:19]
            slug = l.get("market_slug", "")
            attempts = l.get("attempts", 0)
            print(f"  {ts}  ABANDON   {slug}  (after {attempts} attempts)")
