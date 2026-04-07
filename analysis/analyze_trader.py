"""
Analyze a specific Polymarket trader's recent activity.

Fetches trades and positions from the Polymarket Data API (no auth required),
then computes win rate, PnL, position sizing, market distribution, and
tries to determine trading style (small edge + big bankroll vs strong signal).

Usage:
    python analyze_trader.py [--hours 6] [--address 0x...]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from statistics import median, stdev

import requests

# ── Config ───────────────────────────────────────────────────────────────────

DATA_API = "https://data-api.polymarket.com"

DEFAULT_ADDRESS = "0x1979ae6B7E6534dE9c4539D0c205E582cA637C9D"
DEFAULT_HOURS = 6

# Rate-limit: small delay between paginated requests
REQUEST_DELAY = 0.15


# ── API helpers ──────────────────────────────────────────────────────────────

def fetch_activity(
    address: str,
    start_ts: int,
    end_ts: int,
    activity_type: str = "TRADE",
    limit: int = 500,
    max_offset: int = 3000,
) -> list[dict]:
    """Fetch all activity records of a given type in [start_ts, end_ts].

    The data-api caps offset at ~3000. When we hit that limit we advance
    the start timestamp to just past the last record we received and
    restart pagination from offset 0.
    """
    all_records = []
    cursor_start = start_ts
    seen_txs: set[str] = set()  # dedupe across pagination windows

    while cursor_start < end_ts:
        offset = 0
        while True:
            params = {
                "user": address,
                "type": activity_type,
                "start": cursor_start,
                "end": end_ts,
                "limit": limit,
                "offset": offset,
                "sortBy": "TIMESTAMP",
                "sortDirection": "ASC",
            }
            resp = requests.get(f"{DATA_API}/activity", params=params, timeout=30)
            if resp.status_code == 400 and offset > 0:
                # Hit the offset ceiling — advance cursor_start
                break
            resp.raise_for_status()
            data = resp.json()
            if not data:
                return all_records  # no more data at all

            for rec in data:
                # dedupe by (txHash, asset, timestamp) since same tx can appear once
                dedup_key = f"{rec.get('transactionHash','')}-{rec.get('asset','')}-{rec.get('timestamp','')}"
                if dedup_key not in seen_txs:
                    seen_txs.add(dedup_key)
                    all_records.append(rec)

            if len(data) < limit:
                return all_records  # exhausted all results

            offset += limit
            if offset > max_offset:
                break
            time.sleep(REQUEST_DELAY)

        # Advance cursor_start past the last timestamp we received
        if all_records:
            cursor_start = all_records[-1]["timestamp"] + 1
        else:
            break
        time.sleep(REQUEST_DELAY)

    return all_records


def fetch_positions(address: str, limit: int = 100) -> list[dict]:
    """Fetch all positions (open + resolved) for the address."""
    all_positions = []
    offset = 0

    while True:
        params = {
            "user": address,
            "limit": limit,
            "offset": offset,
        }
        resp = requests.get(f"{DATA_API}/positions", params=params, timeout=30)
        if resp.status_code == 400 and offset > 0:
            break
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_positions.extend(data)
        if len(data) < limit:
            break
        offset += limit
        time.sleep(REQUEST_DELAY)

    return all_positions


def fetch_trades_sample(
    address: str,
    start_ts: int,
    end_ts: int,
    sample_limit: int = 2000,
) -> list[dict]:
    """Fetch a sample of recent trades from the /trades endpoint.

    The /trades endpoint returns different fields than /activity (e.g.
    fee info). We grab a manageable sample to avoid pagination hell
    with very active traders.
    """
    all_trades: list[dict] = []
    offset = 0
    limit = 500

    while len(all_trades) < sample_limit:
        params = {
            "user": address,
            "limit": limit,
            "offset": offset,
        }
        try:
            resp = requests.get(f"{DATA_API}/trades", params=params, timeout=30)
            if resp.status_code != 200:
                break
            data = resp.json()
            if not data:
                break
        except Exception:
            break

        for t in data:
            ts = t.get("timestamp", 0)
            if start_ts <= ts <= end_ts:
                all_trades.append(t)
            if ts < start_ts:
                return all_trades

        if len(data) < limit:
            break
        offset += limit
        time.sleep(REQUEST_DELAY)

    return all_trades


# ── Analysis helpers ─────────────────────────────────────────────────────────

def parse_market_type(slug: str) -> tuple[str, str]:
    """Parse slug like 'btc-updown-15m-1772335800' -> ('BTC', '15m')."""
    parts = slug.split("-")
    asset = parts[0].upper() if parts else "?"
    timeframe = "?"
    for p in parts:
        if p in ("5m", "15m", "1h", "4h"):
            timeframe = p
            break
    return asset, timeframe


def extract_window_ts(slug: str) -> int | None:
    """Extract the unix timestamp from the slug."""
    parts = slug.split("-")
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return None


# ── Main analysis ────────────────────────────────────────────────────────────

def analyze(address: str, hours: float):
    now = datetime.now(timezone.utc)
    end_ts = int(now.timestamp())
    start_ts = int((now - timedelta(hours=hours)).timestamp())

    print(f"{'=' * 72}")
    print(f"  Polymarket Trader Analysis")
    print(f"  Address:  {address}")
    print(f"  Period:   last {hours:.1f} hours")
    print(f"  Window:   {datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"         -> {datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"{'=' * 72}")
    print()

    # ── Fetch data ───────────────────────────────────────────────────────
    print("Fetching trades...", end=" ", flush=True)
    trades = fetch_activity(address, start_ts, end_ts, "TRADE")
    print(f"{len(trades)} trades found")

    print("Fetching redemptions...", end=" ", flush=True)
    redemptions = fetch_activity(address, start_ts, end_ts, "REDEEM")
    print(f"{len(redemptions)} redemptions found")

    print("Fetching positions...", end=" ", flush=True)
    positions = fetch_positions(address)
    print(f"{len(positions)} total positions found")

    print("Fetching trades sample (for fee/maker analysis)...", end=" ", flush=True)
    trades_sample = fetch_trades_sample(address, start_ts, end_ts)
    # Build a set of tx hashes where fee > 0 (taker pays fees, maker doesn't)
    taker_hashes: set[str] = set()
    for t in trades_sample:
        fee = t.get("fee", t.get("feeRatePercentage", 0))
        tx = t.get("transactionHash", "")
        if fee and float(fee) > 0 and tx:
            taker_hashes.add(tx)
    print(f"{len(trades_sample)} sampled, {len(taker_hashes)} taker trades identified")

    if not trades:
        print("\nNo trades found in the specified period. Exiting.")
        return

    # ── Build per-market position map from positions endpoint ─────────
    # conditionId -> position info (has cashPnl, avgPrice, etc.)
    position_map: dict[str, dict] = {}
    for p in positions:
        cid = p.get("conditionId", "")
        if cid:
            position_map[cid] = p

    # ── Group trades by market (conditionId + outcome) ────────────────
    # Each "market window" is a conditionId. The trader buys Up or Down.
    # We group all trades for the same conditionId + outcomeIndex together.
    MarketKey = tuple  # (conditionId, outcomeIndex)

    market_trades: dict[MarketKey, list[dict]] = defaultdict(list)
    for t in trades:
        key = (t["conditionId"], t["outcomeIndex"])
        market_trades[key].append(t)

    # ── Compute per-market stats ──────────────────────────────────────
    # For each market position: total cost, total shares, outcome
    market_results = []

    for (cid, oi), trade_list in market_trades.items():
        slug = trade_list[0].get("slug", "")
        title = trade_list[0].get("title", "")
        outcome = trade_list[0].get("outcome", "")
        asset_name, timeframe = parse_market_type(slug)
        window_ts = extract_window_ts(slug)

        # Separate buys and sells
        buys = [t for t in trade_list if t["side"] == "BUY"]
        sells = [t for t in trade_list if t["side"] == "SELL"]

        total_bought_shares = sum(t["size"] for t in buys)
        total_bought_cost = sum(t["usdcSize"] for t in buys)
        total_sold_shares = sum(t["size"] for t in sells)
        total_sold_proceeds = sum(t["usdcSize"] for t in sells)

        net_shares = total_bought_shares - total_sold_shares
        net_cost = total_bought_cost - total_sold_proceeds

        avg_buy_price = total_bought_cost / total_bought_shares if total_bought_shares > 0 else 0
        avg_sell_price = total_sold_proceeds / total_sold_shares if total_sold_shares > 0 else 0

        # Check resolution from positions endpoint
        pos = position_map.get(cid)
        resolved = False
        won = None
        pnl = None
        cur_price = None

        if pos:
            cur_price = pos.get("curPrice")
            cash_pnl = pos.get("cashPnl", 0)
            # Position is resolved if curPrice is 0 or 1
            if cur_price is not None and cur_price in (0, 1, 0.0, 1.0):
                resolved = True
                won = cur_price == 1.0
                # PnL: if won, shares are worth $1 each; if lost, worth $0
                pnl = (net_shares * 1.0 - net_cost) if won else (0 - net_cost)
            else:
                # Still open or partially resolved
                if cur_price is not None:
                    pnl = net_shares * cur_price - net_cost

        # Check redemptions for this conditionId
        market_redeemed = sum(
            r["usdcSize"] for r in redemptions if r["conditionId"] == cid
        )

        # Maker/taker classification for trades in this market
        taker_count = sum(
            1 for t in trade_list if t.get("transactionHash", "") in taker_hashes
        )
        maker_count = len(trade_list) - taker_count

        # Average price across all trades for classification
        all_prices = [t["price"] for t in trade_list]

        market_results.append({
            "cid": cid,
            "slug": slug,
            "title": title,
            "asset": asset_name,
            "timeframe": timeframe,
            "outcome": outcome,
            "window_ts": window_ts,
            "num_trades": len(trade_list),
            "num_buys": len(buys),
            "num_sells": len(sells),
            "total_bought_shares": total_bought_shares,
            "total_bought_cost": total_bought_cost,
            "total_sold_shares": total_sold_shares,
            "total_sold_proceeds": total_sold_proceeds,
            "net_shares": net_shares,
            "net_cost": net_cost,
            "avg_buy_price": avg_buy_price,
            "avg_sell_price": avg_sell_price,
            "resolved": resolved,
            "won": won,
            "pnl": pnl,
            "cur_price": cur_price,
            "redeemed": market_redeemed,
            "maker_count": maker_count,
            "taker_count": taker_count,
            "all_prices": all_prices,
            "trade_timestamps": sorted([t["timestamp"] for t in trade_list]),
        })

    # ── Aggregate stats ───────────────────────────────────────────────

    total_trades = sum(m["num_trades"] for m in market_results)
    total_buys = sum(m["num_buys"] for m in market_results)
    total_sells = sum(m["num_sells"] for m in market_results)
    total_markets = len(market_results)

    # Resolved markets for win rate
    resolved_markets = [m for m in market_results if m["resolved"]]
    won_markets = [m for m in resolved_markets if m["won"]]
    lost_markets = [m for m in resolved_markets if m["won"] is False]

    # Win rate (unweighted = by count)
    win_rate_count = len(won_markets) / len(resolved_markets) if resolved_markets else 0

    # Win rate weighted by position size (net_cost)
    total_resolved_cost = sum(abs(m["net_cost"]) for m in resolved_markets)
    weighted_wins = sum(abs(m["net_cost"]) for m in won_markets)
    win_rate_weighted = weighted_wins / total_resolved_cost if total_resolved_cost > 0 else 0

    # PnL calculations
    pnl_values = [m["pnl"] for m in resolved_markets if m["pnl"] is not None]
    total_pnl = sum(pnl_values) if pnl_values else 0
    avg_pnl = total_pnl / len(pnl_values) if pnl_values else 0
    median_pnl = median(pnl_values) if pnl_values else 0

    # Total capital deployed
    total_capital_deployed = sum(m["total_bought_cost"] for m in market_results)

    # Position sizes
    position_sizes = [m["net_cost"] for m in market_results if m["net_cost"] > 0]
    avg_position_size = sum(position_sizes) / len(position_sizes) if position_sizes else 0
    median_position_size = median(position_sizes) if position_sizes else 0

    # Average buy prices (across all trades)
    all_buy_prices = []
    for m in market_results:
        for t in m["all_prices"]:
            all_buy_prices.append(t)
    avg_price_overall = sum(all_buy_prices) / len(all_buy_prices) if all_buy_prices else 0

    # Price distribution buckets
    price_buckets = {"0.00-0.20": 0, "0.20-0.40": 0, "0.40-0.60": 0, "0.60-0.80": 0, "0.80-1.00": 0}
    for p in all_buy_prices:
        if p < 0.20:
            price_buckets["0.00-0.20"] += 1
        elif p < 0.40:
            price_buckets["0.20-0.40"] += 1
        elif p < 0.60:
            price_buckets["0.40-0.60"] += 1
        elif p < 0.80:
            price_buckets["0.60-0.80"] += 1
        else:
            price_buckets["0.80-1.00"] += 1

    # Market distribution by asset
    asset_dist: dict[str, dict] = defaultdict(lambda: {
        "count": 0, "trades": 0, "cost": 0, "pnl": 0, "wins": 0, "losses": 0
    })
    for m in market_results:
        a = m["asset"]
        asset_dist[a]["count"] += 1
        asset_dist[a]["trades"] += m["num_trades"]
        asset_dist[a]["cost"] += m["total_bought_cost"]
        if m["resolved"] and m["pnl"] is not None:
            asset_dist[a]["pnl"] += m["pnl"]
            if m["won"]:
                asset_dist[a]["wins"] += 1
            else:
                asset_dist[a]["losses"] += 1

    # Timeframe distribution
    tf_dist: dict[str, dict] = defaultdict(lambda: {
        "count": 0, "trades": 0, "cost": 0, "pnl": 0
    })
    for m in market_results:
        tf = m["timeframe"]
        tf_dist[tf]["count"] += 1
        tf_dist[tf]["trades"] += m["num_trades"]
        tf_dist[tf]["cost"] += m["total_bought_cost"]
        if m["resolved"] and m["pnl"] is not None:
            tf_dist[tf]["pnl"] += m["pnl"]

    # Maker vs taker
    total_maker = sum(m["maker_count"] for m in market_results)
    total_taker = sum(m["taker_count"] for m in market_results)

    # Time between trades (all timestamps sorted)
    all_timestamps = sorted(t["timestamp"] for t in trades)
    trade_gaps = []
    for i in range(1, len(all_timestamps)):
        gap = all_timestamps[i] - all_timestamps[i - 1]
        if gap > 0:
            trade_gaps.append(gap)
    avg_gap = sum(trade_gaps) / len(trade_gaps) if trade_gaps else 0
    median_gap = median(trade_gaps) if trade_gaps else 0

    # Outcome direction analysis: does the trader pick Up or Down?
    up_positions = [m for m in market_results if m["outcome"] == "Up"]
    down_positions = [m for m in market_results if m["outcome"] == "Down"]

    # ── Print results ─────────────────────────────────────────────────

    print(f"\n{'=' * 72}")
    print(f"  TRADE SUMMARY")
    print(f"{'=' * 72}")
    print(f"  Total individual trades:    {total_trades:,}")
    print(f"  Unique market positions:    {total_markets}")
    print(f"  Resolved positions:         {len(resolved_markets)}")
    print(f"  Open/pending positions:     {total_markets - len(resolved_markets)}")
    print(f"  Buy trades:                 {total_buys:,}")
    print(f"  Sell trades:                {total_sells:,}")
    print(f"  Buy/Sell ratio:             {total_buys / max(total_sells, 1):.1f}x")

    print(f"\n{'=' * 72}")
    print(f"  WIN RATE")
    print(f"{'=' * 72}")
    print(f"  Wins:                       {len(won_markets)}")
    print(f"  Losses:                     {len(lost_markets)}")
    print(f"  Win rate (by count):        {win_rate_count:.1%}")
    print(f"  Win rate (size-weighted):   {win_rate_weighted:.1%}")

    print(f"\n{'=' * 72}")
    print(f"  PnL ANALYSIS (resolved positions only)")
    print(f"{'=' * 72}")
    print(f"  Total PnL:                  ${total_pnl:+,.2f}")
    print(f"  Average PnL per position:   ${avg_pnl:+,.2f}")
    print(f"  Median PnL per position:    ${median_pnl:+,.2f}")
    print(f"  Total capital deployed:     ${total_capital_deployed:,.2f}")
    if total_capital_deployed > 0:
        roi = total_pnl / total_capital_deployed
        print(f"  ROI (PnL / capital):        {roi:+.2%}")

    if pnl_values:
        wins_pnl = [p for p in pnl_values if p > 0]
        losses_pnl = [p for p in pnl_values if p < 0]
        avg_win = sum(wins_pnl) / len(wins_pnl) if wins_pnl else 0
        avg_loss = sum(losses_pnl) / len(losses_pnl) if losses_pnl else 0
        print(f"  Average winning PnL:        ${avg_win:+,.2f}")
        print(f"  Average losing PnL:         ${avg_loss:+,.2f}")
        if avg_loss != 0:
            print(f"  Win/Loss ratio (avg PnL):   {abs(avg_win / avg_loss):.2f}x")

    print(f"\n{'=' * 72}")
    print(f"  POSITION SIZING")
    print(f"{'=' * 72}")
    print(f"  Average position (USDC):    ${avg_position_size:,.2f}")
    print(f"  Median position (USDC):     ${median_position_size:,.2f}")
    if position_sizes:
        print(f"  Min position:               ${min(position_sizes):,.2f}")
        print(f"  Max position:               ${max(position_sizes):,.2f}")
        if len(position_sizes) >= 2:
            print(f"  Std dev:                    ${stdev(position_sizes):,.2f}")

    print(f"\n{'=' * 72}")
    print(f"  PRICE DISTRIBUTION (what prices do they buy at?)")
    print(f"{'=' * 72}")
    print(f"  Average trade price:        {avg_price_overall:.4f}")
    total_price_trades = sum(price_buckets.values())
    for bucket, count in price_buckets.items():
        pct = count / total_price_trades if total_price_trades > 0 else 0
        bar = "#" * int(pct * 40)
        print(f"    {bucket}:  {count:>5} ({pct:>5.1%})  {bar}")

    print(f"\n{'=' * 72}")
    print(f"  OUTCOME DIRECTION")
    print(f"{'=' * 72}")
    print(f"  'Up' positions:             {len(up_positions)}")
    print(f"  'Down' positions:           {len(down_positions)}")
    up_cost = sum(m["net_cost"] for m in up_positions if m["net_cost"] > 0)
    down_cost = sum(m["net_cost"] for m in down_positions if m["net_cost"] > 0)
    print(f"  Capital in 'Up':            ${up_cost:,.2f}")
    print(f"  Capital in 'Down':          ${down_cost:,.2f}")

    print(f"\n{'=' * 72}")
    print(f"  MARKET DISTRIBUTION BY ASSET")
    print(f"{'=' * 72}")
    for asset_name in sorted(asset_dist.keys()):
        d = asset_dist[asset_name]
        wr = d["wins"] / (d["wins"] + d["losses"]) if (d["wins"] + d["losses"]) > 0 else 0
        print(
            f"  {asset_name:>4}:  {d['count']:>3} positions, "
            f"{d['trades']:>5} trades, "
            f"${d['cost']:>10,.2f} deployed, "
            f"PnL=${d['pnl']:>+9,.2f}, "
            f"WR={wr:.0%}"
        )

    print(f"\n{'=' * 72}")
    print(f"  TIMEFRAME DISTRIBUTION")
    print(f"{'=' * 72}")
    for tf in sorted(tf_dist.keys()):
        d = tf_dist[tf]
        print(
            f"  {tf:>4}:  {d['count']:>3} positions, "
            f"{d['trades']:>5} trades, "
            f"${d['cost']:>10,.2f} deployed, "
            f"PnL=${d['pnl']:>+9,.2f}"
        )

    print(f"\n{'=' * 72}")
    print(f"  MAKER vs TAKER")
    print(f"{'=' * 72}")
    print(f"  Maker fills:                {total_maker:,}")
    print(f"  Taker fills:                {total_taker:,}")
    if total_maker + total_taker > 0:
        maker_pct = total_maker / (total_maker + total_taker)
        print(f"  Maker %:                    {maker_pct:.1%}")
        if maker_pct > 0.7:
            print(f"  --> Predominantly a MAKER (passive limit orders)")
        elif maker_pct < 0.3:
            print(f"  --> Predominantly a TAKER (market/aggressive orders)")
        else:
            print(f"  --> Mixed maker/taker style")

    print(f"\n{'=' * 72}")
    print(f"  TRADE TIMING")
    print(f"{'=' * 72}")
    print(f"  Average gap between trades: {avg_gap:.1f}s ({avg_gap/60:.1f}m)")
    print(f"  Median gap between trades:  {median_gap:.1f}s ({median_gap/60:.1f}m)")
    if trade_gaps:
        print(f"  Min gap:                    {min(trade_gaps):.0f}s")
        print(f"  Max gap:                    {max(trade_gaps):.0f}s")

    # Active trading windows (how many windows per hour)
    window_slugs = set(m["slug"] for m in market_results)
    windows_per_hour = len(window_slugs) / hours if hours > 0 else 0
    print(f"  Unique market windows:      {len(window_slugs)}")
    print(f"  Windows per hour:           {windows_per_hour:.1f}")

    # ── Edge / signal analysis ────────────────────────────────────────

    print(f"\n{'=' * 72}")
    print(f"  EDGE & SIGNAL ANALYSIS")
    print(f"{'=' * 72}")

    if resolved_markets:
        # Expected value at entry: avg price paid for winning vs losing
        winning_avg_prices = [m["avg_buy_price"] for m in won_markets if m["avg_buy_price"] > 0]
        losing_avg_prices = [m["avg_buy_price"] for m in lost_markets if m["avg_buy_price"] > 0]

        avg_win_entry = sum(winning_avg_prices) / len(winning_avg_prices) if winning_avg_prices else 0
        avg_loss_entry = sum(losing_avg_prices) / len(losing_avg_prices) if losing_avg_prices else 0

        print(f"  Avg entry price (winners):  {avg_win_entry:.4f}")
        print(f"  Avg entry price (losers):   {avg_loss_entry:.4f}")

        # Edge per dollar: PnL / total_capital
        if total_capital_deployed > 0:
            edge_per_dollar = total_pnl / total_capital_deployed
            print(f"  Edge per $1 deployed:       {edge_per_dollar:+.4f} ({edge_per_dollar*100:+.2f}c)")

        # Expected value calculation:
        # EV = WR * (1/avg_price - 1) - (1-WR) * 1
        # where 1/avg_price is the shares per dollar, and winning shares pay $1
        if win_rate_weighted > 0 and avg_price_overall > 0:
            ev_per_trade = win_rate_weighted * (1.0 / avg_price_overall - 1.0) - (1.0 - win_rate_weighted)
            print(f"  Theoretical EV per $1:      {ev_per_trade:+.4f}")

        # Determine style
        print()
        if len(resolved_markets) >= 10:
            if win_rate_weighted > 0.60 and avg_position_size > 500:
                print("  ASSESSMENT: Strong signal + large bankroll")
                print(f"    - High win rate ({win_rate_weighted:.0%}) with substantial sizing (${avg_position_size:,.0f})")
                print(f"    - This trader appears to have a strong predictive edge")
            elif win_rate_weighted > 0.55 and avg_position_size > 1000:
                print("  ASSESSMENT: Moderate signal + very large bankroll")
                print(f"    - Moderate win rate ({win_rate_weighted:.0%}) but massive position sizes (${avg_position_size:,.0f})")
                print(f"    - Likely relies on volume to extract a small edge")
            elif win_rate_weighted > 0.55:
                print("  ASSESSMENT: Small edge + moderate bankroll")
                print(f"    - Win rate of {win_rate_weighted:.0%} suggests a modest but real edge")
                print(f"    - Position size (${avg_position_size:,.0f}) suggests moderate confidence")
            elif win_rate_weighted > 0.50:
                print("  ASSESSMENT: Marginal edge, likely noise")
                print(f"    - Win rate of {win_rate_weighted:.0%} is barely above 50%")
                print(f"    - May not be statistically significant with {len(resolved_markets)} samples")
            else:
                print("  ASSESSMENT: No apparent edge (or currently losing)")
                print(f"    - Win rate of {win_rate_weighted:.0%} is below breakeven")
                print(f"    - Could be going through a drawdown or has no real signal")

            # Additional observations
            if maker_pct > 0.7 and total_maker + total_taker > 0:
                print(f"    - Primarily makes liquidity (maker), avoiding taker fees")
            if windows_per_hour > 3:
                print(f"    - Very active: {windows_per_hour:.0f} markets/hour (bot-like behavior)")
            if len(set(m["asset"] for m in market_results)) >= 3:
                print(f"    - Diversified across {len(set(m['asset'] for m in market_results))} assets")
        else:
            print(f"  ASSESSMENT: Insufficient data ({len(resolved_markets)} resolved positions)")
            print(f"    - Need more resolved trades for reliable edge assessment")

    # ── Recent trades detail (last 20) ────────────────────────────────

    print(f"\n{'=' * 72}")
    print(f"  RECENT POSITIONS (last 20 resolved)")
    print(f"{'=' * 72}")
    recent_resolved = sorted(
        resolved_markets, key=lambda m: m.get("window_ts") or 0, reverse=True
    )[:20]

    for m in recent_resolved:
        win_str = "WIN " if m["won"] else "LOSS"
        ts_str = ""
        if m["window_ts"]:
            ts_str = datetime.fromtimestamp(m["window_ts"], tz=timezone.utc).strftime("%H:%M")
        pnl_str = f"${m['pnl']:+.2f}" if m["pnl"] is not None else "?"
        print(
            f"  {ts_str}  {m['asset']:>4} {m['timeframe']:>3} "
            f"{m['outcome']:>4} @ {m['avg_buy_price']:.3f} "
            f"({m['num_trades']:>2} fills, ${m['net_cost']:>7.2f}) "
            f"-> {win_str} {pnl_str}"
        )

    # ── Largest positions ─────────────────────────────────────────────

    print(f"\n{'=' * 72}")
    print(f"  LARGEST POSITIONS (top 10 by cost)")
    print(f"{'=' * 72}")
    largest = sorted(market_results, key=lambda m: m["net_cost"], reverse=True)[:10]
    for m in largest:
        status = "OPEN"
        if m["resolved"]:
            status = "WIN " if m["won"] else "LOSS"
        pnl_str = f"${m['pnl']:+.2f}" if m["pnl"] is not None else "?"
        ts_str = ""
        if m["window_ts"]:
            ts_str = datetime.fromtimestamp(m["window_ts"], tz=timezone.utc).strftime("%H:%M")
        print(
            f"  {ts_str}  {m['asset']:>4} {m['timeframe']:>3} "
            f"{m['outcome']:>4} @ {m['avg_buy_price']:.3f} "
            f"${m['net_cost']:>8.2f} deployed "
            f"-> {status} {pnl_str}"
        )

    print(f"\n{'=' * 72}")
    print(f"  END OF ANALYSIS")
    print(f"{'=' * 72}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze a Polymarket trader's activity")
    parser.add_argument(
        "--address", "-a",
        default=DEFAULT_ADDRESS,
        help=f"Trader wallet address (default: {DEFAULT_ADDRESS})",
    )
    parser.add_argument(
        "--hours", "-t",
        type=float,
        default=DEFAULT_HOURS,
        help=f"Number of hours to look back (default: {DEFAULT_HOURS})",
    )
    args = parser.parse_args()
    analyze(args.address, args.hours)


if __name__ == "__main__":
    main()
