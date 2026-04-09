#!/usr/bin/env python3
"""
Competitor Analysis: Wee-Playroom (0xf247...5216)

Fetches a Polymarket trader's trade history and analyzes their strategy,
focusing on timing relative to market resolution (resolution-snipe hypothesis).

Usage:
    python scripts/research/analyze_competitor.py
    python scripts/research/analyze_competitor.py --address 0x...
    python scripts/research/analyze_competitor.py --cached   # use cached data
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

CACHE_FILE = Path("/tmp/competitor_trades.json")
POSITIONS_CACHE = Path("/tmp/competitor_positions.json")

DEFAULT_ADDRESS = "0xf247584e41117bbbe4cc06e4d2c95741792a5216"

# ET -> UTC offset for EDT (March-November): +4 hours
EDT_OFFSET_H = 4

MONTH_MAP = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


def fetch_trades(address: str, use_cache: bool = False) -> list[dict]:
    """Fetch all trades from the Polymarket data API."""
    if use_cache and CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            data = json.load(f)
        print(f"[cache] Loaded {len(data)} trades from {CACHE_FILE}")
        return data

    all_trades: list[dict] = []
    for offset in range(0, 3500, 500):
        url = (
            f"https://data-api.polymarket.com/activity"
            f"?user={address}&limit=500&offset={offset}"
        )
        result = subprocess.run(
            ["curl", "-s", url],
            capture_output=True, text=True, timeout=30,
        )
        try:
            chunk = json.loads(result.stdout)
        except json.JSONDecodeError:
            break
        if not isinstance(chunk, list) or not chunk:
            break
        all_trades.extend(chunk)
        print(f"  fetched offset {offset}: {len(chunk)} trades (total {len(all_trades)})")
        if len(chunk) < 500:
            break

    with open(CACHE_FILE, "w") as f:
        json.dump(all_trades, f)
    print(f"Saved {len(all_trades)} trades to {CACHE_FILE}")
    return all_trades


def fetch_positions(address: str, use_cache: bool = False) -> list[dict]:
    """Fetch positions from the Polymarket data API."""
    if use_cache and POSITIONS_CACHE.exists():
        with open(POSITIONS_CACHE) as f:
            return json.load(f)

    url = (
        f"https://data-api.polymarket.com/positions"
        f"?user={address}&limit=200&sizeThreshold=0"
    )
    result = subprocess.run(
        ["curl", "-s", url],
        capture_output=True, text=True, timeout=30,
    )
    data = json.loads(result.stdout)
    with open(POSITIONS_CACHE, "w") as f:
        json.dump(data, f)
    return data


def parse_resolution_time(title: str):
    """Parse market title to extract candle start and end times (UTC).

    Returns (asset_name, candle_start_dt, candle_end_dt) or (None, None, None).
    """
    m = re.search(r"(Bitcoin|Solana|XRP|Ethereum) Up or Down - (.+)", title)
    if not m:
        return None, None, None

    asset = m.group(1)
    date_time_str = m.group(2)

    m2 = re.search(
        r"(\w+)\s+(\d+),?\s*(?:(\d{4}),?\s*)?(\d+)(?::(\d+))?\s*(AM|PM)",
        date_time_str,
    )
    if not m2:
        return None, None, None

    month_name = m2.group(1)
    day = int(m2.group(2))
    year = int(m2.group(3)) if m2.group(3) else 2026
    hour = int(m2.group(4))
    minute = int(m2.group(5) or 0)
    ampm = m2.group(6)

    month = MONTH_MAP.get(month_name, 0)
    if month == 0:
        return None, None, None

    if ampm == "PM" and hour != 12:
        hour += 12
    elif ampm == "AM" and hour == 12:
        hour = 0

    utc_hour = hour + EDT_OFFSET_H
    utc_day = day
    if utc_hour >= 24:
        utc_hour -= 24
        utc_day += 1

    try:
        candle_start = datetime.datetime(
            year, month, utc_day, utc_hour, minute, 0,
            tzinfo=datetime.timezone.utc,
        )
    except ValueError:
        return None, None, None

    # Check for explicit time-range format (e.g. "7:45AM-7:50AM")
    m_range = re.search(
        r"(\d+):(\d+)\s*(AM|PM)\s*-\s*(\d+):(\d+)\s*(AM|PM)",
        date_time_str,
    )
    if m_range:
        end_h = int(m_range.group(4))
        end_m = int(m_range.group(5))
        end_ap = m_range.group(6)
        if end_ap == "PM" and end_h != 12:
            end_h += 12
        elif end_ap == "AM" and end_h == 12:
            end_h = 0
        utc_end_h = end_h + EDT_OFFSET_H
        utc_end_d = utc_day
        if utc_end_h >= 24:
            utc_end_h -= 24
            utc_end_d += 1
        candle_end = datetime.datetime(
            year, month, utc_end_d, utc_end_h, end_m, 0,
            tzinfo=datetime.timezone.utc,
        )
    else:
        # Default: 1-hour candle
        candle_end = candle_start + datetime.timedelta(hours=1)

    return asset, candle_start, candle_end


def enrich_trades(raw_trades: list[dict]) -> list[dict]:
    """Add timing and direction fields to each trade."""
    enriched = []
    for t in raw_trades:
        if t.get("type") != "TRADE":
            continue

        asset, candle_start, candle_end = parse_resolution_time(t["title"])
        if candle_end is None:
            continue

        trade_ts = t["timestamp"]
        res_ts = int(candle_end.timestamp())
        start_ts = int(candle_start.timestamp())
        window_dur = res_ts - start_ts

        seconds_before = res_ts - trade_ts
        seconds_into = trade_ts - start_ts
        pct_elapsed = (seconds_into / window_dur * 100) if window_dur > 0 else 0

        # Effective direction
        outcome = t["outcome"]
        side = t["side"]
        if (outcome == "Up" and side == "BUY") or (outcome == "Down" and side == "SELL"):
            direction = "BULLISH"
        else:
            direction = "BEARISH"

        enriched.append({
            "title": t["title"],
            "asset": asset,
            "outcome": outcome,
            "side": side,
            "price": t["price"],
            "size": t["size"],
            "usdcSize": t["usdcSize"],
            "timestamp": trade_ts,
            "seconds_before": seconds_before,
            "seconds_into": seconds_into,
            "window_duration": window_dur,
            "pct_elapsed": pct_elapsed,
            "direction": direction,
            "candle_end": candle_end.isoformat(),
        })

    return enriched


def timing_bucket(sb: int) -> str:
    if sb < 0:
        return "AFTER"
    elif sb <= 10:
        return "0-10s"
    elif sb <= 30:
        return "10-30s"
    elif sb <= 60:
        return "30-60s"
    elif sb <= 120:
        return "60-120s"
    elif sb <= 300:
        return "2-5min"
    elif sb <= 600:
        return "5-10min"
    elif sb <= 1800:
        return "10-30min"
    elif sb <= 3600:
        return "30-60min"
    else:
        return ">60min"


def print_section(title: str):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def analyze(trades: list[dict], positions: list[dict]):
    """Run full analysis and print results."""

    print(f"Total enriched trades: {len(trades)}")

    # ----------------------------------------------------------------
    # 1. Timing distribution
    # ----------------------------------------------------------------
    print_section("1. TRADE TIMING: SECONDS BEFORE RESOLUTION")

    bucket_order = [
        "0-10s", "10-30s", "30-60s", "60-120s", "2-5min",
        "5-10min", "10-30min", "30-60min", ">60min", "AFTER",
    ]
    buckets: Counter = Counter()
    for t in trades:
        buckets[timing_bucket(t["seconds_before"])] += 1

    for b in bucket_order:
        count = buckets.get(b, 0)
        pct = count / len(trades) * 100 if trades else 0
        bar = "#" * int(pct * 2)
        print(f"  {b:12s}: {count:4d} ({pct:5.1f}%) {bar}")

    sbs = sorted(t["seconds_before"] for t in trades if t["seconds_before"] >= 0)
    if sbs:
        mean_sb = sum(sbs) / len(sbs)
        median_sb = sbs[len(sbs) // 2]
        print(f"\n  Mean:   {mean_sb:.0f}s ({mean_sb / 60:.1f} min)")
        print(f"  Median: {median_sb:.0f}s ({median_sb / 60:.1f} min)")
        print(f"  Range:  [{sbs[0]}s, {sbs[-1]}s]")

    # Fine-grained: last 5 minutes
    last_5m = [t for t in trades if 0 <= t["seconds_before"] <= 300]
    if last_5m:
        print(f"\n  Trades in last 5 minutes: {len(last_5m)}")
        print("  By 30-second bucket:")
        for lo in range(0, 300, 30):
            hi = lo + 30
            count = len([t for t in last_5m if lo <= t["seconds_before"] < hi])
            bar = "#" * count
            print(f"    {lo:3d}-{hi:3d}s: {count:3d} {bar}")

    # ----------------------------------------------------------------
    # 2. Entry price distribution
    # ----------------------------------------------------------------
    print_section("2. ENTRY PRICE DISTRIBUTION")

    buys = [t for t in trades if t["side"] == "BUY"]
    sells = [t for t in trades if t["side"] == "SELL"]
    print(f"  BUY trades:  {len(buys)} (avg price {sum(t['price'] for t in buys) / len(buys):.3f})")
    print(f"  SELL trades: {len(sells)} (avg price {sum(t['price'] for t in sells) / len(sells):.3f})")

    print("\n  BUY price distribution:")
    for lo_f in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        hi_f = lo_f + 0.1 + (0.01 if lo_f >= 0.9 else 0)
        subset = [t for t in buys if lo_f <= t["price"] < hi_f]
        vol = sum(t["usdcSize"] for t in subset)
        bar = "#" * (len(subset) // 3)
        print(f"    {lo_f:.1f}-{min(hi_f, 1.0):.1f}: {len(subset):4d} trades ${vol:8.2f} vol  {bar}")

    # ----------------------------------------------------------------
    # 3. Direction
    # ----------------------------------------------------------------
    print_section("3. DIRECTIONAL ANALYSIS")

    bull = [t for t in trades if t["direction"] == "BULLISH"]
    bear = [t for t in trades if t["direction"] == "BEARISH"]
    print(f"  BULLISH (BUY Up / SELL Down): {len(bull):4d} trades, ${sum(t['usdcSize'] for t in bull):,.2f}")
    print(f"  BEARISH (BUY Down / SELL Up): {len(bear):4d} trades, ${sum(t['usdcSize'] for t in bear):,.2f}")

    print("\n  Direction by timing bucket:")
    for lo, hi, label in [
        (0, 120, "Last 2min"),
        (120, 600, "2-10min"),
        (600, 1800, "10-30min"),
        (1800, 3600, "30-60min"),
    ]:
        b_bull = len([t for t in bull if lo <= t["seconds_before"] < hi])
        b_bear = len([t for t in bear if lo <= t["seconds_before"] < hi])
        total = b_bull + b_bear
        if total:
            print(f"    {label:12s}: {b_bull:3d} BULL ({b_bull / total * 100:.0f}%) | {b_bear:3d} BEAR ({b_bear / total * 100:.0f}%)")

    # ----------------------------------------------------------------
    # 4. Both-sides analysis (market-making signal)
    # ----------------------------------------------------------------
    print_section("4. BOTH-SIDES ANALYSIS (Market-Making Signal)")

    market_groups = defaultdict(list)
    for t in trades:
        market_groups[t["title"]].append(t)

    both_sides_count = 0
    for market, mtrades in market_groups.items():
        outcomes = set(t["outcome"] for t in mtrades)
        sides = set(t["side"] for t in mtrades)
        if len(outcomes) > 1 or len(sides) > 1:
            both_sides_count += 1

    print(f"  Markets where trader acted on BOTH sides: {both_sides_count} / {len(market_groups)}")
    print(f"  (Holding both Up AND Down positions = market-making / delta-neutral)")

    # ----------------------------------------------------------------
    # 5. Per-market breakdown
    # ----------------------------------------------------------------
    print_section("5. PER-MARKET BREAKDOWN (top 15 by volume)")

    sorted_markets = sorted(
        market_groups.items(),
        key=lambda x: -sum(t["usdcSize"] for t in x[1]),
    )

    for market, mtrades in sorted_markets[:15]:
        vol = sum(t["usdcSize"] for t in mtrades)
        n = len(mtrades)
        avg_sb = sum(t["seconds_before"] for t in mtrades) / n
        min_sb = min(t["seconds_before"] for t in mtrades)
        max_sb = max(t["seconds_before"] for t in mtrades)
        n_bull = len([t for t in mtrades if t["direction"] == "BULLISH"])
        n_bear = len([t for t in mtrades if t["direction"] == "BEARISH"])

        print(f"  {market}")
        print(f"    {n} trades | ${vol:.2f} vol | avg {avg_sb:.0f}s ({avg_sb / 60:.1f}m) before | [{min_sb}s - {max_sb}s]")
        print(f"    {n_bull} bull / {n_bear} bear")

    # ----------------------------------------------------------------
    # 6. Position-level P&L
    # ----------------------------------------------------------------
    print_section("6. POSITION P&L (from /positions endpoint)")

    crypto_pos = [p for p in positions if "Up or Down" in p["title"]]
    total_invested = 0.0
    total_current = 0.0
    total_realized = 0.0

    resolved = []
    for p in crypto_pos:
        inv = p["initialValue"]
        cur = p["currentValue"]
        real = p.get("realizedPnl", 0)
        total_invested += inv
        total_current += cur
        total_realized += real
        cp = p["curPrice"]
        is_resolved = cp <= 0.001 or cp >= 0.999
        net = (cur - inv) + real
        status = "DONE" if is_resolved else "OPEN"
        won = "W" if net > 0 else "L"
        print(f"    {status} {won} | {p['title'][:48]:48s} {p['outcome']:5s} | net=${net:7.2f} | avgP={p['avgPrice']:.4f}")
        if is_resolved:
            resolved.append({"title": p["title"], "outcome": p["outcome"], "net": net})

    print(f"\n  Total invested: ${total_invested:,.2f}")
    print(f"  Unrealized P&L: ${total_current - total_invested:,.2f}")
    print(f"  Realized P&L:   ${total_realized:,.2f}")
    print(f"  TOTAL P&L:      ${(total_current - total_invested) + total_realized:,.2f}")

    if resolved:
        winners = [r for r in resolved if r["net"] > 0]
        losers = [r for r in resolved if r["net"] <= 0]
        print(f"\n  Resolved: {len(resolved)} positions ({len(winners)} W, {len(losers)} L)")
        print(f"  Win rate: {len(winners) / len(resolved) * 100:.1f}%")
        if winners:
            print(f"  Avg win:  ${sum(r['net'] for r in winners) / len(winners):,.2f}")
        if losers:
            print(f"  Avg loss: ${sum(r['net'] for r in losers) / len(losers):,.2f}")

    # ----------------------------------------------------------------
    # 7. Snipe detection: look at last-minute clusters
    # ----------------------------------------------------------------
    print_section("7. RESOLUTION SNIPE DETECTION")

    # For each market, count trades in the last 60 seconds
    snipe_markets = []
    for market, mtrades in market_groups.items():
        last_60 = [t for t in mtrades if 0 <= t["seconds_before"] <= 60]
        last_10 = [t for t in mtrades if 0 <= t["seconds_before"] <= 10]
        total_trades = len(mtrades)
        if last_60:
            snipe_markets.append({
                "market": market,
                "total_trades": total_trades,
                "last_60s": len(last_60),
                "last_10s": len(last_10),
                "last_60_vol": sum(t["usdcSize"] for t in last_60),
                "total_vol": sum(t["usdcSize"] for t in mtrades),
                "pct_last_60": len(last_60) / total_trades * 100,
            })

    snipe_markets.sort(key=lambda x: -x["last_60s"])
    if snipe_markets:
        print("  Markets with trades in last 60 seconds:")
        for sm in snipe_markets:
            print(
                f"    {sm['market'][:50]:50s}: "
                f"{sm['last_60s']:3d}/{sm['total_trades']:3d} trades in last 60s "
                f"({sm['pct_last_60']:.0f}%), "
                f"${sm['last_60_vol']:.2f} vol"
            )

        total_last_60 = sum(sm["last_60s"] for sm in snipe_markets)
        total_all = sum(sm["total_trades"] for sm in snipe_markets)
        print(f"\n  Overall: {total_last_60}/{total_all} trades in last 60s ({total_last_60 / total_all * 100:.1f}%)")
    else:
        print("  No trades found in last 60 seconds of any window.")

    # ----------------------------------------------------------------
    # 8. Strategy classification
    # ----------------------------------------------------------------
    print_section("8. STRATEGY CLASSIFICATION")

    # Check for market-making pattern: positions on both sides
    both_side_markets = 0
    for market_base in set(t["title"] for t in trades):
        outcomes = set(t["outcome"] for t in trades if t["title"] == market_base)
        if len(outcomes) == 2:
            both_side_markets += 1

    pct_both = both_side_markets / len(market_groups) * 100 if market_groups else 0

    # Check for late-window concentration
    late_trades = len([t for t in trades if 0 <= t["seconds_before"] <= 300])
    late_pct = late_trades / len(trades) * 100 if trades else 0

    # Check entry price distribution
    extreme_price_buys = len([
        t for t in trades
        if t["side"] == "BUY" and (t["price"] >= 0.85 or t["price"] <= 0.15)
    ])
    extreme_pct = extreme_price_buys / len(buys) * 100 if buys else 0

    print(f"  Markets with both-side positions: {pct_both:.0f}%")
    print(f"  Trades in last 5 min of window:   {late_pct:.1f}%")
    print(f"  BUY trades at extreme prices:     {extreme_pct:.1f}%")
    print()

    if pct_both > 70:
        print("  >> MARKET-MAKING / DELTA-NEUTRAL strategy detected")
        print("     Trader buys BOTH Up and Down throughout the window,")
        print("     capturing spread and adjusting positions as odds move.")
    if late_pct > 20:
        print("  >> LATE-WINDOW CONCENTRATION detected")
        print("     Significant portion of activity in the final minutes.")
    if extreme_pct > 30:
        print("  >> HIGH-CONFIDENCE ENTRIES detected")
        print("     Buying at extreme prices (near 0 or 1) suggests")
        print("     conviction about the outcome.")

    # Is this the resolution snipe?
    total_last_60_trades = len([t for t in trades if 0 <= t["seconds_before"] <= 60])
    total_last_10_trades = len([t for t in trades if 0 <= t["seconds_before"] <= 10])
    print()
    if total_last_60_trades > 20 and total_last_60_trades / len(trades) > 0.02:
        print("  >> PARTIAL RESOLUTION SNIPE PATTERN")
        print(f"     {total_last_60_trades} trades ({total_last_60_trades / len(trades) * 100:.1f}%) in last 60s")
        print(f"     {total_last_10_trades} trades in last 10s")
        print("     However, the majority of trading occurs THROUGHOUT the window,")
        print("     not concentrated at the end.")
    else:
        print("  >> NO resolution snipe pattern detected")
        print("     Trading is distributed throughout the window.")


def main():
    parser = argparse.ArgumentParser(description="Analyze a Polymarket competitor's trades")
    parser.add_argument("--address", default=DEFAULT_ADDRESS, help="Trader's proxy wallet address")
    parser.add_argument("--cached", action="store_true", help="Use cached data from /tmp")
    args = parser.parse_args()

    print(f"=== COMPETITOR ANALYSIS: {args.address[:10]}...{args.address[-4:]} ===")
    print()

    trades_raw = fetch_trades(args.address, use_cache=args.cached)
    positions = fetch_positions(args.address, use_cache=args.cached)

    trades = enrich_trades(trades_raw)
    print(f"Enriched {len(trades)} trades from {len(trades_raw)} raw records")

    analyze(trades, positions)


if __name__ == "__main__":
    main()
