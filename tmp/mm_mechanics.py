"""
Drill into the MECHANICS of how this market maker profits.
Look at individual fill-level activity within specific windows.
"""
from __future__ import annotations
import requests, time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from statistics import median

DATA_API = "https://data-api.polymarket.com"
ADDR = "0x1979ae6b7e6534de9c4539d0c205e582ca637c9d"
DELAY = 0.12

def fetch_activity_windowed(addr, start_ts, end_ts, activity_type="TRADE"):
    all_records = []
    seen = set()
    cursor = start_ts
    while cursor < end_ts:
        offset = 0
        while True:
            params = {
                "user": addr, "type": activity_type,
                "start": cursor, "end": end_ts,
                "limit": 500, "offset": offset,
                "sortBy": "TIMESTAMP", "sortDirection": "ASC",
            }
            try:
                r = requests.get(f"{DATA_API}/activity", params=params, timeout=30)
                if r.status_code == 400 and offset > 0:
                    break
                if r.status_code != 200:
                    return all_records
                data = r.json()
                if not data:
                    return all_records
            except Exception as e:
                print(f"  Error: {e}")
                return all_records
            for rec in data:
                dk = f"{rec.get('transactionHash','')}-{rec.get('asset','')}-{rec.get('timestamp','')}"
                if dk not in seen:
                    seen.add(dk)
                    all_records.append(rec)
            if len(data) < 500:
                return all_records
            offset += 500
            if offset > 3000:
                break
            time.sleep(DELAY)
        if all_records:
            cursor = all_records[-1]["timestamp"] + 1
        else:
            break
        time.sleep(DELAY)
    return all_records

# Fetch a narrower window for detail — last 6 hours
now = datetime.now(timezone.utc)
end_ts = int(now.timestamp())
start_ts = int((now - timedelta(hours=6)).timestamp())

print(f"Fetching last 6h of trades for detailed fill analysis...")
trades = fetch_activity_windowed(ADDR, start_ts, end_ts, "TRADE")
print(f"Got {len(trades):,} trades\n")

# Group by conditionId (market window)
by_cid = defaultdict(list)
for t in trades:
    by_cid[t["conditionId"]].append(t)

# For each market, analyze the fill flow
print("=" * 80)
print("  FILL-LEVEL FLOW ANALYSIS (per market window)")
print("=" * 80)

# Sort markets by total volume
market_vols = []
for cid, tlist in by_cid.items():
    vol = sum(t["usdcSize"] for t in tlist)
    market_vols.append((cid, tlist, vol))
market_vols.sort(key=lambda x: x[2], reverse=True)

# Analyze top 10 highest-volume markets in detail
for rank, (cid, tlist, vol) in enumerate(market_vols[:10], 1):
    slug = tlist[0].get("slug", "")
    parts = slug.split("-")
    asset = parts[0].upper() if parts else "?"
    tf = "?"
    for p in parts:
        if p in ("5m", "15m", "1h", "4h"):
            tf = p
            break

    # Separate by outcome AND by buy/sell
    up_buys = [t for t in tlist if t["outcome"] == "Up" and t["side"] == "BUY"]
    up_sells = [t for t in tlist if t["outcome"] == "Up" and t["side"] == "SELL"]
    dn_buys = [t for t in tlist if t["outcome"] == "Down" and t["side"] == "BUY"]
    dn_sells = [t for t in tlist if t["outcome"] == "Down" and t["side"] == "SELL"]

    up_buy_cost = sum(t["usdcSize"] for t in up_buys)
    up_buy_shares = sum(t["size"] for t in up_buys)
    up_sell_proceeds = sum(t["usdcSize"] for t in up_sells)
    up_sell_shares = sum(t["size"] for t in up_sells)

    dn_buy_cost = sum(t["usdcSize"] for t in dn_buys)
    dn_buy_shares = sum(t["size"] for t in dn_buys)
    dn_sell_proceeds = sum(t["usdcSize"] for t in dn_sells)
    dn_sell_shares = sum(t["size"] for t in dn_sells)

    up_avg_buy = up_buy_cost / up_buy_shares if up_buy_shares > 0 else 0
    up_avg_sell = up_sell_proceeds / up_sell_shares if up_sell_shares > 0 else 0
    dn_avg_buy = dn_buy_cost / dn_buy_shares if dn_buy_shares > 0 else 0
    dn_avg_sell = dn_sell_proceeds / dn_sell_shares if dn_sell_shares > 0 else 0

    # Realized P&L from round-trips (sell - buy on same side)
    up_realized = up_sell_proceeds - (up_avg_buy * up_sell_shares) if up_sell_shares > 0 else 0
    dn_realized = dn_sell_proceeds - (dn_avg_buy * dn_sell_shares) if dn_sell_shares > 0 else 0

    # Net inventory remaining
    up_net_shares = up_buy_shares - up_sell_shares
    dn_net_shares = dn_buy_shares - dn_sell_shares
    up_net_cost = up_buy_cost - up_sell_proceeds
    dn_net_cost = dn_buy_cost - dn_sell_proceeds

    # Combined cost for remaining inventory
    combined_avg = (up_net_cost + dn_net_cost) / max(up_net_shares + dn_net_shares, 1) if (up_net_shares + dn_net_shares) > 0 else 0

    # If resolution happens: one side pays $1/share, other pays $0
    # Best case: max(up_net_shares, dn_net_shares) * $1
    # Worst case: min(up_net_shares, dn_net_shares) * $1
    if_up_wins = up_net_shares * 1.0 - up_net_cost - dn_net_cost
    if_dn_wins = dn_net_shares * 1.0 - up_net_cost - dn_net_cost

    print(f"\n{'─'*80}")
    print(f"  #{rank} {asset} {tf} | {slug}")
    print(f"  Total fills: {len(tlist)} | Volume: ${vol:,.2f}")
    print(f"")
    print(f"  UP side:")
    print(f"    Buys:  {len(up_buys):>3} fills, {up_buy_shares:>8.1f} shares @ avg {up_avg_buy:.4f} = ${up_buy_cost:>8.2f}")
    print(f"    Sells: {len(up_sells):>3} fills, {up_sell_shares:>8.1f} shares @ avg {up_avg_sell:.4f} = ${up_sell_proceeds:>8.2f}")
    print(f"    Net:   {up_net_shares:>8.1f} shares remaining, cost ${up_net_cost:>8.2f}")
    if up_sell_shares > 0:
        print(f"    Realized P&L from sells: ${up_realized:>+8.2f} (avg sell {up_avg_sell:.4f} vs avg buy {up_avg_buy:.4f})")
    print(f"")
    print(f"  DOWN side:")
    print(f"    Buys:  {len(dn_buys):>3} fills, {dn_buy_shares:>8.1f} shares @ avg {dn_avg_buy:.4f} = ${dn_buy_cost:>8.2f}")
    print(f"    Sells: {len(dn_sells):>3} fills, {dn_sell_shares:>8.1f} shares @ avg {dn_avg_sell:.4f} = ${dn_sell_proceeds:>8.2f}")
    print(f"    Net:   {dn_net_shares:>8.1f} shares remaining, cost ${dn_net_cost:>8.2f}")
    if dn_sell_shares > 0:
        print(f"    Realized P&L from sells: ${dn_realized:>+8.2f} (avg sell {dn_avg_sell:.4f} vs avg buy {dn_avg_buy:.4f})")
    print(f"")
    print(f"  COMBINED:")
    print(f"    Total net cost:    ${up_net_cost + dn_net_cost:>8.2f}")
    print(f"    Realized trading:  ${up_realized + dn_realized:>+8.2f}")
    print(f"    If UP wins:        ${if_up_wins:>+8.2f} (up_shares*$1 - total_cost)")
    print(f"    If DOWN wins:      ${if_dn_wins:>+8.2f} (dn_shares*$1 - total_cost)")
    guaranteed = min(if_up_wins, if_dn_wins)
    print(f"    Guaranteed min:    ${guaranteed:>+8.2f}")

    # Show chronological fill flow (first 30 and last 10)
    print(f"\n  Chronological fills (first 20):")
    sorted_fills = sorted(tlist, key=lambda t: t["timestamp"])
    for i, t in enumerate(sorted_fills[:20]):
        ts = datetime.fromtimestamp(t["timestamp"], tz=timezone.utc).strftime("%H:%M:%S")
        print(f"    {ts} {t['side']:>4} {t['outcome']:>4} {t['size']:>7.1f} shares @ {t['price']:.4f} = ${t['usdcSize']:>7.2f}")

    if len(sorted_fills) > 30:
        print(f"    ... ({len(sorted_fills) - 30} more fills) ...")
        for t in sorted_fills[-10:]:
            ts = datetime.fromtimestamp(t["timestamp"], tz=timezone.utc).strftime("%H:%M:%S")
            print(f"    {ts} {t['side']:>4} {t['outcome']:>4} {t['size']:>7.1f} shares @ {t['price']:.4f} = ${t['usdcSize']:>7.2f}")

# ── Aggregate: how much do they sell vs buy? ──────────────────────────

print(f"\n{'='*80}")
print(f"  AGGREGATE BUY vs SELL ANALYSIS (all {len(trades):,} trades)")
print(f"{'='*80}")

total_buys = [t for t in trades if t["side"] == "BUY"]
total_sells = [t for t in trades if t["side"] == "SELL"]

buy_volume = sum(t["usdcSize"] for t in total_buys)
sell_volume = sum(t["usdcSize"] for t in total_sells)
buy_shares = sum(t["size"] for t in total_buys)
sell_shares = sum(t["size"] for t in total_sells)

print(f"  BUY:  {len(total_buys):>6,} trades, ${buy_volume:>12,.2f} volume, {buy_shares:>12,.1f} shares")
print(f"  SELL: {len(total_sells):>6,} trades, ${sell_volume:>12,.2f} volume, {sell_shares:>12,.1f} shares")
print(f"  Sell/Buy ratio (volume): {sell_volume/buy_volume:.1%}")
print(f"  Sell/Buy ratio (shares): {sell_shares/buy_shares:.1%}")

# ── Price at which they buy vs sell ───────────────────────────────────

print(f"\n{'='*80}")
print(f"  PRICE ANALYSIS: BUY vs SELL prices")
print(f"{'='*80}")

buy_prices = [t["price"] for t in total_buys]
sell_prices = [t["price"] for t in total_sells]

if buy_prices:
    print(f"  Buy prices:  avg={sum(buy_prices)/len(buy_prices):.4f}, median={median(buy_prices):.4f}")
if sell_prices:
    print(f"  Sell prices: avg={sum(sell_prices)/len(sell_prices):.4f}, median={median(sell_prices):.4f}")

# Buy price buckets
print(f"\n  Buy price distribution:")
for lo, hi, label in [(0,0.1,"0-10c"),(0.1,0.2,"10-20c"),(0.2,0.3,"20-30c"),(0.3,0.4,"30-40c"),
                       (0.4,0.5,"40-50c"),(0.5,0.6,"50-60c"),(0.6,0.7,"60-70c"),(0.7,0.8,"70-80c"),
                       (0.8,0.9,"80-90c"),(0.9,1.01,"90-100c")]:
    count = sum(1 for p in buy_prices if lo <= p < hi)
    vol = sum(t["usdcSize"] for t in total_buys if lo <= t["price"] < hi)
    if count > 0:
        print(f"    {label}: {count:>5} fills, ${vol:>10,.2f}")

print(f"\n  Sell price distribution:")
for lo, hi, label in [(0,0.1,"0-10c"),(0.1,0.2,"10-20c"),(0.2,0.3,"20-30c"),(0.3,0.4,"30-40c"),
                       (0.4,0.5,"40-50c"),(0.5,0.6,"50-60c"),(0.6,0.7,"60-70c"),(0.7,0.8,"70-80c"),
                       (0.8,0.9,"80-90c"),(0.9,1.01,"90-100c")]:
    count = sum(1 for p in sell_prices if lo <= p < hi)
    vol = sum(t["usdcSize"] for t in total_sells if lo <= t["price"] < hi)
    if count > 0:
        print(f"    {label}: {count:>5} fills, ${vol:>10,.2f}")

# ── Check: Up+Down cost vs $1.00 per market ──────────────────────────

print(f"\n{'='*80}")
print(f"  UP+DOWN COST ANALYSIS (is sum < $1.00?)")
print(f"{'='*80}")

pair_costs = []
for cid, tlist in by_cid.items():
    up_buys = [t for t in tlist if t["outcome"] == "Up" and t["side"] == "BUY"]
    dn_buys = [t for t in tlist if t["outcome"] == "Down" and t["side"] == "BUY"]
    if up_buys and dn_buys:
        up_avg = sum(t["usdcSize"] for t in up_buys) / sum(t["size"] for t in up_buys)
        dn_avg = sum(t["usdcSize"] for t in dn_buys) / sum(t["size"] for t in dn_buys)
        pair_costs.append(up_avg + dn_avg)

if pair_costs:
    under_1 = sum(1 for c in pair_costs if c < 1.0)
    over_1 = sum(1 for c in pair_costs if c >= 1.0)
    print(f"  Markets where avg_buy(Up) + avg_buy(Down) < $1.00: {under_1} ({under_1/len(pair_costs):.1%})")
    print(f"  Markets where avg_buy(Up) + avg_buy(Down) >= $1.00: {over_1} ({over_1/len(pair_costs):.1%})")
    print(f"  Average combined cost: ${sum(pair_costs)/len(pair_costs):.4f}")
    print(f"  Median combined cost:  ${median(pair_costs):.4f}")
    pair_costs.sort()
    print(f"  Min combined cost:     ${pair_costs[0]:.4f}")
    print(f"  Max combined cost:     ${pair_costs[-1]:.4f}")

    # Distribution
    print(f"\n  Combined cost distribution:")
    for lo, hi, label in [(0.80,0.85,"80-85c"),(0.85,0.90,"85-90c"),(0.90,0.95,"90-95c"),
                           (0.95,1.00,"95-100c"),(1.00,1.05,"100-105c"),(1.05,1.10,"105-110c"),
                           (1.10,1.15,"110-115c"),(1.15,1.20,"115-120c"),(1.20,1.50,"120-150c")]:
        count = sum(1 for c in pair_costs if lo <= c < hi)
        if count > 0:
            bar = "#" * min(count, 50)
            print(f"    {label}: {count:>4} markets  {bar}")

print(f"\n{'='*80}")
print(f"  END")
print(f"{'='*80}")
