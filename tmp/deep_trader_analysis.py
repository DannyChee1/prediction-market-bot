"""
Deep analysis of trader 0x1979ae6b7e6534de9c4539d0c205e582ca637c9d.
Focuses on:
  - Bimodal position sizing (large vs small)
  - Per-size-bucket win rates, timing, edge
  - Dual-side vs single-side patterns
  - Market-making vs directional
"""
from __future__ import annotations
import requests, time, json
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from statistics import median, stdev

DATA_API = "https://data-api.polymarket.com"
ADDR = "0x1979ae6b7e6534de9c4539d0c205e582ca637c9d"
HOURS = 48  # go wider to get more data
DELAY = 0.12

def fetch_activity_windowed(addr, start_ts, end_ts, activity_type="TRADE"):
    """Paginate with cursor advancement to handle >3000 results."""
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

def fetch_positions(addr):
    all_data = []
    offset = 0
    while True:
        params = {"user": addr, "limit": 100, "offset": offset}
        try:
            r = requests.get(f"{DATA_API}/positions", params=params, timeout=30)
            if r.status_code != 200:
                break
            data = r.json()
            if not data:
                break
        except:
            break
        all_data.extend(data)
        if len(data) < 100:
            break
        offset += 100
        time.sleep(DELAY)
    return all_data


# ── Fetch ─────────────────────────────────────────────────────────────

now = datetime.now(timezone.utc)
end_ts = int(now.timestamp())
start_ts = int((now - timedelta(hours=HOURS)).timestamp())

print(f"Analyzing trader {ADDR}")
print(f"Period: last {HOURS}h ({datetime.fromtimestamp(start_ts, tz=timezone.utc):%Y-%m-%d %H:%M} -> {datetime.fromtimestamp(end_ts, tz=timezone.utc):%Y-%m-%d %H:%M} UTC)")
print()

print("Fetching trades...", end=" ", flush=True)
trades = fetch_activity_windowed(ADDR, start_ts, end_ts, "TRADE")
print(f"{len(trades):,}")

print("Fetching redemptions...", end=" ", flush=True)
redemptions = fetch_activity_windowed(ADDR, start_ts, end_ts, "REDEEM")
print(f"{len(redemptions):,}")

print("Fetching positions...", end=" ", flush=True)
positions = fetch_positions(ADDR)
print(f"{len(positions):,}")

# Build position map (conditionId -> position info)
pos_map = {}
for p in positions:
    cid = p.get("conditionId", "")
    if cid:
        pos_map[cid] = p

# ── Group by conditionId ──────────────────────────────────────────────

grouped = defaultdict(list)
for t in trades:
    k = (t["conditionId"], t["outcomeIndex"])
    grouped[k].append(t)

# Also group redemptions
redeemed_by_cid = defaultdict(float)
for r in redemptions:
    redeemed_by_cid[r["conditionId"]] += r.get("usdcSize", 0)

# ── Build position-level stats ────────────────────────────────────────

results = []
for (cid, oi), tlist in grouped.items():
    slug = tlist[0].get("slug", "")
    outcome = tlist[0].get("outcome", "")
    title = tlist[0].get("title", "")

    # Parse market info
    parts = slug.split("-")
    asset = parts[0].upper() if parts else "?"
    timeframe = "?"
    for p in parts:
        if p in ("5m", "15m", "1h", "4h"):
            timeframe = p
            break
    try:
        window_ts = int(parts[-1])
    except:
        window_ts = None

    buys = [t for t in tlist if t["side"] == "BUY"]
    sells = [t for t in tlist if t["side"] == "SELL"]

    bought_shares = sum(t["size"] for t in buys)
    bought_cost = sum(t["usdcSize"] for t in buys)
    sold_shares = sum(t["size"] for t in sells)
    sold_proceeds = sum(t["usdcSize"] for t in sells)

    net_shares = bought_shares - sold_shares
    net_cost = bought_cost - sold_proceeds
    avg_price = bought_cost / bought_shares if bought_shares > 0 else 0

    # Resolution
    pos = pos_map.get(cid)
    resolved = False
    won = None
    pnl = None

    if pos:
        cp = pos.get("curPrice")
        if cp is not None and cp in (0, 1, 0.0, 1.0):
            resolved = True
            won = (cp == 1.0)
            pnl = (net_shares * 1.0 - net_cost) if won else (-net_cost)
        elif cp is not None:
            pnl = net_shares * cp - net_cost

    # Timing
    timestamps = sorted(t["timestamp"] for t in tlist)
    first_ts = timestamps[0]
    last_ts = timestamps[-1]
    duration = last_ts - first_ts

    results.append({
        "cid": cid, "slug": slug, "outcome": outcome, "title": title,
        "asset": asset, "timeframe": timeframe, "window_ts": window_ts,
        "n_trades": len(tlist), "n_buys": len(buys), "n_sells": len(sells),
        "bought_shares": bought_shares, "bought_cost": bought_cost,
        "sold_shares": sold_shares, "sold_proceeds": sold_proceeds,
        "net_shares": net_shares, "net_cost": net_cost, "avg_price": avg_price,
        "resolved": resolved, "won": won, "pnl": pnl,
        "first_ts": first_ts, "last_ts": last_ts, "duration_s": duration,
    })

# ── Position size distribution ────────────────────────────────────────

print(f"\n{'='*72}")
print(f"  POSITION SIZE DISTRIBUTION ({len(results)} positions)")
print(f"{'='*72}")

costs = [r["net_cost"] for r in results if r["net_cost"] > 0]
costs_sorted = sorted(costs, reverse=True)

buckets = [
    ("$0-10", 0, 10),
    ("$10-25", 10, 25),
    ("$25-50", 25, 50),
    ("$50-100", 50, 100),
    ("$100-250", 100, 250),
    ("$250-500", 250, 500),
    ("$500-1K", 500, 1000),
    ("$1K-2K", 1000, 2000),
    ("$2K-5K", 2000, 5000),
    ("$5K+", 5000, 999999),
]

for label, lo, hi in buckets:
    in_bucket = [c for c in costs if lo <= c < hi]
    if not in_bucket:
        continue
    total_in = sum(in_bucket)
    bucket_results = [r for r in results if r["net_cost"] > 0 and lo <= r["net_cost"] < hi]
    resolved_in = [r for r in bucket_results if r["resolved"]]
    wins_in = [r for r in resolved_in if r["won"]]
    wr = len(wins_in) / len(resolved_in) if resolved_in else float('nan')
    pnl_in = sum(r["pnl"] for r in resolved_in if r["pnl"] is not None)

    bar = "#" * min(len(in_bucket), 50)
    wr_str = f"{wr:.0%}" if resolved_in else "n/a"
    print(f"  {label:>8}: {len(in_bucket):>4} positions, ${total_in:>10,.2f} total, WR={wr_str:>4}, PnL=${pnl_in:>+10,.2f}  {bar}")

print(f"\n  Total positions with net cost > 0: {len(costs)}")
if costs:
    print(f"  Median position size: ${median(costs):.2f}")
    print(f"  Mean position size:   ${sum(costs)/len(costs):.2f}")
    print(f"  Top 5 positions:  {['${:.0f}'.format(c) for c in costs_sorted[:5]]}")
    print(f"  Bottom 5 positions: {['${:.2f}'.format(c) for c in costs_sorted[-5:]]}")

# ── Check for dual-side (same conditionId, both UP and DOWN) ──────────

print(f"\n{'='*72}")
print(f"  DUAL-SIDE ANALYSIS (same market, both UP and DOWN)")
print(f"{'='*72}")

by_cid = defaultdict(list)
for r in results:
    by_cid[r["cid"]].append(r)

dual_markets = {cid: rs for cid, rs in by_cid.items() if len(rs) > 1}
single_markets = {cid: rs[0] for cid, rs in by_cid.items() if len(rs) == 1}

print(f"  Markets with BOTH sides: {len(dual_markets)}")
print(f"  Markets with ONE side:   {len(single_markets)}")

if dual_markets:
    print(f"\n  Dual-side details (top 15 by cost):")
    sorted_dual = sorted(dual_markets.items(), key=lambda x: sum(r["bought_cost"] for r in x[1]), reverse=True)[:15]
    for cid, rs in sorted_dual:
        total_cost = sum(r["net_cost"] for r in rs)
        sides = " + ".join(f"{r['outcome']}@{r['avg_price']:.3f}(${r['net_cost']:.2f})" for r in rs)
        print(f"    {rs[0]['asset']:>4} {rs[0]['timeframe']:>3} | {sides} | total=${total_cost:.2f}")

# ── Per-asset breakdown ───────────────────────────────────────────────

print(f"\n{'='*72}")
print(f"  PER-ASSET BREAKDOWN")
print(f"{'='*72}")

by_asset = defaultdict(list)
for r in results:
    by_asset[r["asset"]].append(r)

for asset in sorted(by_asset.keys()):
    rs = by_asset[asset]
    total_cost = sum(r["net_cost"] for r in rs if r["net_cost"] > 0)
    n = len(rs)
    resolved = [r for r in rs if r["resolved"]]
    wins = [r for r in resolved if r["won"]]
    wr = len(wins)/len(resolved) if resolved else float('nan')
    pnl = sum(r["pnl"] for r in resolved if r["pnl"] is not None)
    sizes = [r["net_cost"] for r in rs if r["net_cost"] > 0]
    med = median(sizes) if sizes else 0
    mx = max(sizes) if sizes else 0
    wr_str = f"WR={wr:.0%}" if resolved else "WR=n/a"
    print(f"  {asset:>5}: {n:>4} positions, ${total_cost:>10,.2f} deployed, {wr_str}, PnL=${pnl:>+9,.2f}, median=${med:.2f}, max=${mx:.2f}")

# ── Per-timeframe breakdown ───────────────────────────────────────────

print(f"\n{'='*72}")
print(f"  PER-TIMEFRAME BREAKDOWN")
print(f"{'='*72}")

by_tf = defaultdict(list)
for r in results:
    by_tf[r["timeframe"]].append(r)

for tf in sorted(by_tf.keys()):
    rs = by_tf[tf]
    total_cost = sum(r["net_cost"] for r in rs if r["net_cost"] > 0)
    n = len(rs)
    resolved = [r for r in rs if r["resolved"]]
    wins = [r for r in resolved if r["won"]]
    wr = len(wins)/len(resolved) if resolved else float('nan')
    pnl = sum(r["pnl"] for r in resolved if r["pnl"] is not None)
    sizes = [r["net_cost"] for r in rs if r["net_cost"] > 0]
    med = median(sizes) if sizes else 0
    wr_str = f"WR={wr:.0%}" if resolved else "WR=n/a"
    print(f"  {tf:>4}: {n:>4} positions, ${total_cost:>10,.2f} deployed, {wr_str}, PnL=${pnl:>+9,.2f}, median_size=${med:.2f}")

# ── Strategy patterns: large vs small positions ───────────────────────

print(f"\n{'='*72}")
print(f"  STRATEGY COMPARISON: SMALL vs LARGE positions")
print(f"{'='*72}")

THRESHOLD = 100
small = [r for r in results if 0 < r["net_cost"] < THRESHOLD]
large = [r for r in results if r["net_cost"] >= THRESHOLD]

for label, subset in [("SMALL (<$100)", small), ("LARGE (>=$100)", large)]:
    if not subset:
        print(f"\n  {label}: no positions")
        continue

    costs_s = [r["net_cost"] for r in subset]
    resolved_s = [r for r in subset if r["resolved"]]
    wins_s = [r for r in resolved_s if r["won"]]
    wr_s = len(wins_s)/len(resolved_s) if resolved_s else float('nan')
    pnl_s = sum(r["pnl"] for r in resolved_s if r["pnl"] is not None)

    # Directional preference
    up_count = sum(1 for r in subset if r["outcome"] == "Up")
    down_count = sum(1 for r in subset if r["outcome"] == "Down")

    # Price distribution
    prices = [r["avg_price"] for r in subset if r["avg_price"] > 0]

    # Entry delay: how early in the window they trade
    entry_delays = []
    for r in subset:
        if r["window_ts"]:
            delay = r["first_ts"] - r["window_ts"]
            if 0 <= delay < 3600:
                entry_delays.append(delay)

    # Number of fills per position
    fills_per = [r["n_trades"] for r in subset]

    print(f"\n  {label}:")
    print(f"    Positions:      {len(subset)}")
    print(f"    Total deployed: ${sum(costs_s):,.2f}")
    print(f"    Median size:    ${median(costs_s):.2f}")
    print(f"    Mean size:      ${sum(costs_s)/len(costs_s):.2f}")
    print(f"    Resolved:       {len(resolved_s)}/{len(subset)}")
    if resolved_s:
        print(f"    Win rate:       {wr_s:.1%}")
    else:
        print(f"    Win rate:       n/a")
    print(f"    PnL:            ${pnl_s:+,.2f}")
    print(f"    Up/Down:        {up_count}/{down_count}")
    if prices:
        print(f"    Avg price:      {sum(prices)/len(prices):.4f}")
        print(f"    Median price:   {median(prices):.4f}")
    print(f"    Avg fills/pos:  {sum(fills_per)/len(fills_per):.1f}")
    if entry_delays:
        print(f"    Avg entry delay: {sum(entry_delays)/len(entry_delays):.0f}s after window open")
        print(f"    Med entry delay: {median(entry_delays):.0f}s after window open")

    # Asset distribution within this size bucket
    asset_counts = defaultdict(int)
    for r in subset:
        asset_counts[r["asset"]] += 1
    print(f"    Assets: {dict(asset_counts)}")

    # Timeframe distribution
    tf_counts = defaultdict(int)
    for r in subset:
        tf_counts[r["timeframe"]] += 1
    print(f"    Timeframes: {dict(tf_counts)}")

# ── Trading frequency / timing patterns ───────────────────────────────

print(f"\n{'='*72}")
print(f"  TIMING PATTERNS")
print(f"{'='*72}")

all_ts = sorted(t["timestamp"] for t in trades)
if len(all_ts) > 1:
    gaps = [all_ts[i] - all_ts[i-1] for i in range(1, len(all_ts)) if all_ts[i] > all_ts[i-1]]
    print(f"  Total trades: {len(trades):,}")
    if gaps:
        print(f"  Median gap:   {median(gaps):.1f}s ({median(gaps)/60:.1f}m)")
        print(f"  Mean gap:     {sum(gaps)/len(gaps):.1f}s ({sum(gaps)/len(gaps)/60:.1f}m)")
        print(f"  Min gap:      {min(gaps):.0f}s")

    # Trades per hour histogram
    hour_counts = defaultdict(int)
    for ts in all_ts:
        h = datetime.fromtimestamp(ts, tz=timezone.utc).hour
        hour_counts[h] += 1

    print(f"\n  Trades by hour (UTC):")
    for h in range(24):
        c = hour_counts.get(h, 0)
        bar = "#" * (c // 10) if c > 0 else ""
        if c > 0:
            print(f"    {h:02d}:00  {c:>5}  {bar}")

# ── Top 20 largest positions detail ───────────────────────────────────

print(f"\n{'='*72}")
print(f"  TOP 20 LARGEST POSITIONS")
print(f"{'='*72}")

top20 = sorted(results, key=lambda r: r["net_cost"], reverse=True)[:20]
for i, r in enumerate(top20, 1):
    status = "OPEN"
    if r["resolved"]:
        status = "WIN " if r["won"] else "LOSS"
    pnl_str = f"${r['pnl']:+.2f}" if r["pnl"] is not None else "?"
    ts_str = datetime.fromtimestamp(r["first_ts"], tz=timezone.utc).strftime("%m-%d %H:%M") if r["first_ts"] else ""
    print(f"  {i:>2}. {ts_str}  {r['asset']:>4} {r['timeframe']:>3} {r['outcome']:>4} "
          f"@ {r['avg_price']:.3f}  ${r['net_cost']:>8,.2f} deployed  "
          f"{r['n_trades']:>3} fills  {status} {pnl_str}")

# ── Bottom 20 smallest positions detail ───────────────────────────────

print(f"\n{'='*72}")
print(f"  BOTTOM 20 SMALLEST POSITIONS")
print(f"{'='*72}")

small_pos = [r for r in results if r["net_cost"] > 0]
bot20 = sorted(small_pos, key=lambda r: r["net_cost"])[:20]
for i, r in enumerate(bot20, 1):
    status = "OPEN"
    if r["resolved"]:
        status = "WIN " if r["won"] else "LOSS"
    pnl_str = f"${r['pnl']:+.2f}" if r["pnl"] is not None else "?"
    ts_str = datetime.fromtimestamp(r["first_ts"], tz=timezone.utc).strftime("%m-%d %H:%M") if r["first_ts"] else ""
    print(f"  {i:>2}. {ts_str}  {r['asset']:>4} {r['timeframe']:>3} {r['outcome']:>4} "
          f"@ {r['avg_price']:.3f}  ${r['net_cost']:>8.2f} deployed  "
          f"{r['n_trades']:>3} fills  {status} {pnl_str}")

# ── Edge analysis per bucket ──────────────────────────────────────────

print(f"\n{'='*72}")
print(f"  EDGE ANALYSIS BY POSITION SIZE")
print(f"{'='*72}")

for label, lo, hi in buckets:
    bucket_r = [r for r in results if r["net_cost"] > 0 and lo <= r["net_cost"] < hi and r["resolved"]]
    if len(bucket_r) < 3:
        continue
    wins = [r for r in bucket_r if r["won"]]
    wr = len(wins)/len(bucket_r)
    avg_p = sum(r["avg_price"] for r in bucket_r) / len(bucket_r)
    pnl = sum(r["pnl"] for r in bucket_r if r["pnl"] is not None)
    capital = sum(r["net_cost"] for r in bucket_r)
    roi = pnl/capital if capital > 0 else 0

    # EV = WR * (1/p - 1) - (1-WR)
    ev = wr * (1/avg_p - 1) - (1-wr) if avg_p > 0 else 0

    print(f"  {label:>8}: {len(bucket_r):>3} resolved, WR={wr:.1%}, avg_price={avg_p:.3f}, "
          f"EV/$ = {ev:+.4f}, ROI={roi:+.1%}, PnL=${pnl:+,.2f}")

print(f"\n{'='*72}")
print(f"  END OF DEEP ANALYSIS")
print(f"{'='*72}")
