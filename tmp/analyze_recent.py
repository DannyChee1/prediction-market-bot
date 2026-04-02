#!/usr/bin/env python3
"""Analyze recent live trades for BTC and ETH."""
import json
from pathlib import Path

BASE = Path(r"C:\Users\d4nny\Desktop\prediction-market-bot")

for asset in ("btc", "eth"):
    fpath = BASE / f"live_trades_{asset}.jsonl"
    if not fpath.exists():
        print(f"\n=== {asset.upper()}: no trade log ===")
        continue

    lines = []
    for raw in fpath.read_text().strip().split("\n"):
        if raw.strip():
            try:
                lines.append(json.loads(raw))
            except json.JSONDecodeError:
                continue

    relevant = [l for l in lines if l.get("type") in
                ("limit_fill", "resolution", "flat_summary", "partial_fill")]

    print(f"\n{'='*75}")
    print(f"  {asset.upper()} - RECENT TRADES (last 50 events)")
    print(f"{'='*75}")

    for l in relevant[-50:]:
        e = l["type"]
        if e == "limit_fill":
            ts = l.get("ts", "")[:19]
            side = l.get("side", "?")
            price = l.get("price", 0)
            size = l.get("size", 0)
            cost = l.get("cost_usd", 0)
            sigma = l.get("sigma", "N/A")
            z = l.get("z_score", "N/A")
            slug = l.get("market_slug", l.get("slug", ""))
            tf = l.get("timeframe", "")
            print(f"  {ts}  FILL  {side:>4} @ {price:.3f}  sz={size:.1f}  "
                  f"cost=${cost:.2f}  sigma={sigma}  z={z}  [{tf}] {slug}")
        elif e == "partial_fill":
            ts = l.get("ts", "")[:19]
            side = l.get("side", "?")
            price = l.get("price", 0)
            filled = l.get("filled_size", 0)
            original = l.get("original_size", 0)
            slug = l.get("market_slug", l.get("slug", ""))
            tf = l.get("timeframe", "")
            print(f"  {ts}  PARTIAL  {side:>4} @ {price:.3f}  filled={filled:.1f}/{original:.1f}  [{tf}] {slug}")
        elif e == "resolution":
            ts = l.get("ts", "")[:19]
            outcome = l.get("outcome", "?")
            pnl = l.get("pnl", 0)
            slug = l.get("market_slug", l.get("slug", ""))
            tf = l.get("timeframe", "")
            entry = l.get("avg_entry", l.get("entry_price", 0))
            side = l.get("side_held", l.get("side", "?"))
            shares = l.get("shares", l.get("size", 0))
            print(f"  {ts}  RESOLVED  {side:>4} @ {entry:.3f}  shares={shares:.1f}  "
                  f"outcome={outcome}  pnl=${pnl:+.2f}  [{tf}] {slug}")
        elif e == "flat_summary":
            ts = l.get("ts", "")[:19]
            reason = l.get("reason", "?")
            slug = l.get("market_slug", l.get("slug", ""))
            tf = l.get("timeframe", "")
            print(f"  {ts}  FLAT  reason={reason}  [{tf}] {slug}")

    # Summary stats on last N resolutions
    resolutions = [l for l in lines if l.get("type") == "resolution"]
    last_n = resolutions[-20:]
    if last_n:
        wins = sum(1 for r in last_n if r.get("pnl", 0) > 0)
        total_pnl = sum(r.get("pnl", 0) for r in last_n)
        entries = [r.get("avg_entry", r.get("entry_price", 0)) for r in last_n]
        avg_entry = sum(entries) / len(entries) if entries else 0
        print(f"\n  LAST {len(last_n)} RESOLUTIONS:")
        print(f"  Wins: {wins}/{len(last_n)} ({wins/len(last_n):.0%})")
        print(f"  Total PnL: ${total_pnl:+.2f}")
        print(f"  Avg entry: {avg_entry:.3f}")

        # Break down by timeframe
        for tf in sorted(set(r.get("timeframe", "?") for r in last_n)):
            tf_res = [r for r in last_n if r.get("timeframe", "?") == tf]
            tf_wins = sum(1 for r in tf_res if r.get("pnl", 0) > 0)
            tf_pnl = sum(r.get("pnl", 0) for r in tf_res)
            tf_entries = [r.get("avg_entry", r.get("entry_price", 0)) for r in tf_res]
            tf_avg_e = sum(tf_entries) / len(tf_entries) if tf_entries else 0
            print(f"    [{tf}] {tf_wins}/{len(tf_res)} wins ({tf_wins/len(tf_res):.0%}), "
                  f"PnL=${tf_pnl:+.2f}, avg_entry={tf_avg_e:.3f}")

        # Show losses detail
        losses = [r for r in last_n if r.get("pnl", 0) < 0]
        if losses:
            print(f"\n  LOSSES ({len(losses)}):")
            for r in losses:
                ts = r.get("ts", "")[:19]
                side = r.get("side_held", r.get("side", "?"))
                entry = r.get("avg_entry", r.get("entry_price", 0))
                pnl = r.get("pnl", 0)
                tf = r.get("timeframe", "")
                slug = r.get("market_slug", r.get("slug", ""))
                shares = r.get("shares", r.get("size", 0))
                print(f"    {ts}  {side:>4} @ {entry:.3f}  shares={shares:.1f}  "
                      f"pnl=${pnl:+.2f}  [{tf}] {slug}")

    # Last fills sigma/z info
    fills = [l for l in lines if l.get("type") == "limit_fill"]
    last_fills = fills[-15:]
    if last_fills:
        sigmas = [f.get("sigma") for f in last_fills if f.get("sigma") is not None]
        zscores = [f.get("z_score") for f in last_fills if f.get("z_score") is not None]
        entries = [f.get("price", 0) for f in last_fills]
        print(f"\n  LAST {len(last_fills)} FILLS STATS:")
        if sigmas:
            print(f"  Sigma range: {min(sigmas):.6f} - {max(sigmas):.6f}")
        if zscores:
            print(f"  Z-score range: {min(zscores):.3f} - {max(zscores):.3f}")
        print(f"  Entry range: {min(entries):.3f} - {max(entries):.3f}")
        up_fills = [f for f in last_fills if f.get("side") == "UP"]
        dn_fills = [f for f in last_fills if f.get("side") == "DOWN"]
        print(f"  UP fills: {len(up_fills)}, DOWN fills: {len(dn_fills)}")

    print()
