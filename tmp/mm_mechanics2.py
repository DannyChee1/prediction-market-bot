"""
Drill into fill-level mechanics during ACTIVE trading period (18-30h ago).
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

# Look at a 2-hour window during active trading (20-22h ago)
now = datetime.now(timezone.utc)
end_ts = int((now - timedelta(hours=20)).timestamp())
start_ts = int((now - timedelta(hours=22)).timestamp())

print(f"Fetching trades from {datetime.fromtimestamp(start_ts, tz=timezone.utc):%m-%d %H:%M} to {datetime.fromtimestamp(end_ts, tz=timezone.utc):%m-%d %H:%M} UTC")
trades = fetch_activity_windowed(ADDR, start_ts, end_ts, "TRADE")
print(f"Got {len(trades):,} trades\n")

if not trades:
    # Try different window
    end_ts = int((now - timedelta(hours=24)).timestamp())
    start_ts = int((now - timedelta(hours=26)).timestamp())
    print(f"No trades. Trying {datetime.fromtimestamp(start_ts, tz=timezone.utc):%m-%d %H:%M} to {datetime.fromtimestamp(end_ts, tz=timezone.utc):%m-%d %H:%M} UTC")
    trades = fetch_activity_windowed(ADDR, start_ts, end_ts, "TRADE")
    print(f"Got {len(trades):,} trades\n")

# Group by conditionId
by_cid = defaultdict(list)
for t in trades:
    by_cid[t["conditionId"]].append(t)

# ── Aggregate buy vs sell ─────────────────────────────────────────────
total_buys = [t for t in trades if t["side"] == "BUY"]
total_sells = [t for t in trades if t["side"] == "SELL"]
buy_vol = sum(t["usdcSize"] for t in total_buys)
sell_vol = sum(t["usdcSize"] for t in total_sells)
buy_shares = sum(t["size"] for t in total_buys)
sell_shares = sum(t["size"] for t in total_sells)

print(f"{'='*80}")
print(f"  BUY vs SELL AGGREGATE")
print(f"{'='*80}")
print(f"  BUY:  {len(total_buys):>6,} trades, ${buy_vol:>12,.2f}, {buy_shares:>10,.1f} shares")
print(f"  SELL: {len(total_sells):>6,} trades, ${sell_vol:>12,.2f}, {sell_shares:>10,.1f} shares")
if buy_vol > 0:
    print(f"  Sell/Buy volume: {sell_vol/buy_vol:.1%}")

# ── Detailed flow for top 5 markets ──────────────────────────────────

market_vols = []
for cid, tlist in by_cid.items():
    vol = sum(t["usdcSize"] for t in tlist)
    market_vols.append((cid, tlist, vol))
market_vols.sort(key=lambda x: x[2], reverse=True)

for rank, (cid, tlist, vol) in enumerate(market_vols[:5], 1):
    slug = tlist[0].get("slug", "")
    parts = slug.split("-")
    asset = parts[0].upper() if parts else "?"
    tf = "?"
    for p in parts:
        if p in ("5m", "15m", "1h", "4h"):
            tf = p
            break

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

    up_net_shares = up_buy_shares - up_sell_shares
    dn_net_shares = dn_buy_shares - dn_sell_shares
    up_net_cost = up_buy_cost - up_sell_proceeds
    dn_net_cost = dn_buy_cost - dn_sell_proceeds

    if_up_wins = up_net_shares * 1.0 - up_net_cost - dn_net_cost
    if_dn_wins = dn_net_shares * 1.0 - up_net_cost - dn_net_cost

    print(f"\n{'='*80}")
    print(f"  #{rank} {asset} {tf} | {slug}")
    print(f"  Total fills: {len(tlist)} | Volume: ${vol:,.2f}")
    print(f"")
    print(f"  UP side:")
    print(f"    Buys:  {len(up_buys):>3} fills, {up_buy_shares:>8.1f} shares @ avg {up_avg_buy:.4f} = ${up_buy_cost:>8.2f}")
    print(f"    Sells: {len(up_sells):>3} fills, {up_sell_shares:>8.1f} shares @ avg {up_avg_sell:.4f} = ${up_sell_proceeds:>8.2f}")
    print(f"    Net:   {up_net_shares:>8.1f} shares, cost ${up_net_cost:>8.2f}")
    print(f"")
    print(f"  DOWN side:")
    print(f"    Buys:  {len(dn_buys):>3} fills, {dn_buy_shares:>8.1f} shares @ avg {dn_avg_buy:.4f} = ${dn_buy_cost:>8.2f}")
    print(f"    Sells: {len(dn_sells):>3} fills, {dn_sell_shares:>8.1f} shares @ avg {dn_avg_sell:.4f} = ${dn_sell_proceeds:>8.2f}")
    print(f"    Net:   {dn_net_shares:>8.1f} shares, cost ${dn_net_cost:>8.2f}")
    print(f"")
    print(f"  RESOLUTION P&L:")
    print(f"    If UP wins:   ${if_up_wins:>+8.2f}")
    print(f"    If DOWN wins: ${if_dn_wins:>+8.2f}")
    print(f"    Guaranteed:   ${min(if_up_wins, if_dn_wins):>+8.2f}")

    # Show ALL fills chronologically for the first 2 markets
    if rank <= 3:
        print(f"\n  ALL FILLS chronologically ({len(tlist)}):")
        sorted_fills = sorted(tlist, key=lambda t: t["timestamp"])
        for i, t in enumerate(sorted_fills):
            ts = datetime.fromtimestamp(t["timestamp"], tz=timezone.utc).strftime("%H:%M:%S")
            print(f"    {i+1:>3}. {ts} {t['side']:>4} {t['outcome']:>4} "
                  f"{t['size']:>8.1f} sh @ {t['price']:.4f} = ${t['usdcSize']:>8.2f}")
            if i >= 80:
                print(f"    ... ({len(sorted_fills) - i - 1} more)")
                break

# ── Up+Down cost analysis ─────────────────────────────────────────────

print(f"\n{'='*80}")
print(f"  UP+DOWN COMBINED COST (is buy_up + buy_down < $1.00?)")
print(f"{'='*80}")

pair_costs = []
for cid, tlist in by_cid.items():
    up_buys = [t for t in tlist if t["outcome"] == "Up" and t["side"] == "BUY"]
    dn_buys = [t for t in tlist if t["outcome"] == "Down" and t["side"] == "BUY"]
    if up_buys and dn_buys:
        up_avg = sum(t["usdcSize"] for t in up_buys) / sum(t["size"] for t in up_buys)
        dn_avg = sum(t["usdcSize"] for t in dn_buys) / sum(t["size"] for t in dn_buys)
        total_cost = up_avg + dn_avg
        # Also compute share-weighted: what's the cost per $1 of guaranteed payout?
        up_sh = sum(t["size"] for t in up_buys)
        dn_sh = sum(t["size"] for t in dn_buys)
        min_sh = min(up_sh, dn_sh)  # paired shares
        cost_of_pairs = min_sh * up_avg + min_sh * dn_avg
        payout_of_pairs = min_sh * 1.0  # one side pays $1
        pair_costs.append({
            "cid": cid,
            "slug": tlist[0].get("slug", ""),
            "up_avg": up_avg,
            "dn_avg": dn_avg,
            "sum": total_cost,
            "up_sh": up_sh,
            "dn_sh": dn_sh,
            "min_sh": min_sh,
            "cost_of_pairs": cost_of_pairs,
            "payout_of_pairs": payout_of_pairs,
        })

if pair_costs:
    sums = [p["sum"] for p in pair_costs]
    under = [p for p in pair_costs if p["sum"] < 1.0]
    over = [p for p in pair_costs if p["sum"] >= 1.0]
    print(f"  Under $1.00: {len(under)} markets")
    print(f"  Over $1.00:  {len(over)} markets")
    print(f"  Average sum: ${sum(sums)/len(sums):.4f}")
    print(f"  Median sum:  ${median(sums):.4f}")

    # Show each one
    for p in sorted(pair_costs, key=lambda x: x["sum"]):
        profit_per_pair = p["payout_of_pairs"] - p["cost_of_pairs"]
        marker = "PROFIT" if p["sum"] < 1.0 else "LOSS  "
        print(f"    {marker} {p['slug'][-20:]}: Up@{p['up_avg']:.4f} + Down@{p['dn_avg']:.4f} = {p['sum']:.4f} "
              f"| {p['min_sh']:.0f} pairs -> ${profit_per_pair:+.2f}")

print(f"\n{'='*80}")
print(f"  END")
print(f"{'='*80}")
