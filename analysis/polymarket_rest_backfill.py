#!/usr/bin/env python3
"""
Backfill historical BTC up/down 5m / 15m parquet files from Polymarket REST APIs.

Combines three free public sources:
  1. gamma-api.polymarket.com /events?slug=...           — market metadata (cid, tokens, end_date)
  2. data-api.polymarket.com /trades?market={cid}        — full per-market trade tape (paginated)
  3. clob.polymarket.com /prices-history?market={token}  — 2-min mid bars (sanity check / fallback)
  4. api.binance.com /api/v3/klines?symbol=BTCUSDT       — 1-second BTC USD klines (chainlink proxy)

Output parquet matches our existing schema in `recording.py` (75 columns), with
the per-level book depth columns (bid_px_*_N, bid_sz_*_N, etc.) left as NaN
since Polymarket REST does not expose historical L2 depth.

Bid/ask are approximated from the trade tape using the rule:
    BUY trade  = someone hit the ask  → ask_at_that_moment ≤ trade_price
    SELL trade = someone hit the bid  → bid_at_that_moment ≥ trade_price

Usage:
    uv run python analysis/polymarket_rest_backfill.py --slug btc-updown-5m-1775508300
    uv run python analysis/polymarket_rest_backfill.py --range 2026-03-15:2026-04-05 --interval 5m
"""

from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import json
import time
import urllib.request
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
GAMMA = 'https://gamma-api.polymarket.com'
DATA  = 'https://data-api.polymarket.com'
CLOB  = 'https://clob.polymarket.com'
BINANCE = 'https://api.binance.com'

USER_AGENT = 'btc-bot-backfill/0.1'


class HttpStatusError(Exception):
    def __init__(self, code: int, url: str):
        super().__init__(f'HTTP {code} {url}')
        self.code = code


def http_get(url: str, retries: int = 6, sleep_s: float = 0.5):
    """GET with retry. 429 = rate limit, retried with exponential backoff.
    Other 4xx are raised immediately. 5xx and network errors retry.
    """
    import urllib.error
    last_exc = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                # Rate limit — back off aggressively
                time.sleep(sleep_s * (2 ** attempt) + 0.5 * attempt)
                last_exc = exc
                continue
            if 400 <= exc.code < 500:
                # Other 4xx is permanent — don't retry
                raise HttpStatusError(exc.code, url) from exc
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(sleep_s * (2 ** attempt))
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(sleep_s * (2 ** attempt))
    raise RuntimeError(f'http_get({url}): {last_exc}')


@dataclass
class MarketMeta:
    slug: str
    cid: str
    token_up: str
    token_dn: str
    window_start_ms: int
    window_end_ms: int


def fetch_event(slug: str) -> MarketMeta | None:
    """Look up a Polymarket event by slug. Returns None if not found."""
    url = f'{GAMMA}/events?slug={urllib.parse.quote(slug)}'
    arr = http_get(url)
    if not arr:
        return None
    ev = arr[0]
    if not ev.get('markets'):
        return None
    mk = ev['markets'][0]
    cid = mk.get('conditionId')
    if not cid:
        return None
    outcomes = mk.get('outcomes', [])
    tokens = mk.get('clobTokenIds', [])
    if isinstance(outcomes, str):
        outcomes = json.loads(outcomes)
    if isinstance(tokens, str):
        tokens = json.loads(tokens)
    if 'Up' not in outcomes or 'Down' not in outcomes:
        return None
    up_idx = outcomes.index('Up')
    dn_idx = outcomes.index('Down')
    # Window from slug (slug ends with unix seconds)
    window_start_ts = int(slug.split('-')[-1])
    end_date_str = mk.get('endDate', '')
    end_dt = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
    window_end_ts = int(end_dt.timestamp())
    return MarketMeta(
        slug=slug, cid=cid,
        token_up=str(tokens[up_idx]), token_dn=str(tokens[dn_idx]),
        window_start_ms=window_start_ts * 1000,
        window_end_ms=window_end_ts * 1000,
    )


def fetch_trades(cid: str, max_pages: int = 8) -> list[dict]:
    """Paginate /trades?market={cid}. Default max_pages=8 → 4000 trades cap.

    The data-api enforces a hard offset cap (around 3500). Once we hit a
    400, we stop paginating and return whatever we've collected — that's
    enough trades for any 5m window.
    """
    out = []
    for page in range(max_pages):
        url = f'{DATA}/trades?market={cid}&limit=500&offset={page*500}&takerOnly=true'
        try:
            rows = http_get(url)
        except HttpStatusError as exc:
            if exc.code == 400:
                # Past pagination cap. Use what we already have.
                break
            raise
        if not rows:
            break
        out.extend(rows)
        if len(rows) < 500:
            break
    return out


def fetch_binance_klines(start_ms: int, end_ms: int) -> list[list]:
    """1-second BTCUSDT klines from Binance public API."""
    url = (f'{BINANCE}/api/v3/klines?symbol=BTCUSDT&interval=1s'
           f'&startTime={start_ms}&endTime={end_ms}&limit=1000')
    return http_get(url)


def build_parquet(meta: MarketMeta,
                  trades: list[dict] | None = None,
                  klines: list[list] | None = None) -> pd.DataFrame | None:
    """Build a per-second snapshot DataFrame matching our recording.py schema.

    If trades/klines aren't provided, fetches them. Pass them in to avoid
    redundant network calls when the caller already has them.
    """
    if trades is None:
        trades = fetch_trades(meta.cid)
    if klines is None:
        klines = fetch_binance_klines(meta.window_start_ms, meta.window_end_ms)
    if not klines:
        return None

    # Sort trades chronologically
    trades.sort(key=lambda t: t['timestamp'])

    # State: latest known bid/ask for each token, from trade tape
    # ask_up = latest BUY trade on Up   (someone paid this for Up → ask was ≤ this)
    # bid_up = latest SELL trade on Up  (someone sold at this → bid was ≥ this)
    # Same for Down. We use the most recent trade as a point estimate.
    rolling = {
        'last_buy_up': None, 'last_sell_up': None,
        'last_buy_dn': None, 'last_sell_dn': None,
        'last_trade_up': None,   # (price, size, side)
        'last_trade_dn': None,
    }

    rows = []
    trade_idx = 0
    n_trades = len(trades)

    for k in klines:
        ts_ms = int(k[0])
        ts_s = ts_ms // 1000
        binance_open  = float(k[1])
        binance_close = float(k[4])
        # Apply all trades up to and including this second
        while trade_idx < n_trades and trades[trade_idx]['timestamp'] <= ts_s:
            t = trades[trade_idx]
            outcome = t['outcome']
            side = t['side']
            price = float(t['price'])
            size = float(t['size'])
            if outcome == 'Up':
                if side == 'BUY':
                    rolling['last_buy_up'] = price
                else:
                    rolling['last_sell_up'] = price
                rolling['last_trade_up'] = (price, size, side)
            elif outcome == 'Down':
                if side == 'BUY':
                    rolling['last_buy_dn'] = price
                else:
                    rolling['last_sell_dn'] = price
                rolling['last_trade_dn'] = (price, size, side)
            trade_idx += 1

        # Approximate book state at this second
        ask_up = rolling['last_buy_up']
        bid_up = rolling['last_sell_up']
        ask_dn = rolling['last_buy_dn']
        bid_dn = rolling['last_sell_dn']
        # Cross-side fallback: bid_up ≈ 1 - ask_dn, ask_up ≈ 1 - bid_dn
        if bid_up is None and ask_dn is not None:
            bid_up = max(0.001, 1.0 - ask_dn)
        if ask_up is None and bid_dn is not None:
            ask_up = min(0.999, 1.0 - bid_dn)
        if bid_dn is None and ask_up is not None:
            bid_dn = max(0.001, 1.0 - ask_up)
        if ask_dn is None and bid_up is not None:
            ask_dn = min(0.999, 1.0 - bid_up)
        # If we have nothing for either side, skip this row (warmup)
        if bid_up is None or ask_up is None or bid_dn is None or ask_dn is None:
            continue

        # Sanity: clamp bid <= ask
        if bid_up > ask_up:
            mid = (bid_up + ask_up) / 2.0
            bid_up = mid - 0.005
            ask_up = mid + 0.005
        if bid_dn > ask_dn:
            mid = (bid_dn + ask_dn) / 2.0
            bid_dn = mid - 0.005
            ask_dn = mid + 0.005

        mid_up = (bid_up + ask_up) / 2.0
        mid_dn = (bid_dn + ask_dn) / 2.0
        spread_up = ask_up - bid_up
        spread_dn = ask_dn - bid_dn

        lt_up = rolling['last_trade_up']
        lt_dn = rolling['last_trade_dn']

        time_remaining_s = max(0.0, (meta.window_end_ms - ts_ms) / 1000.0)
        rows.append({
            'ts_ms': ts_ms,
            'market_slug': meta.slug,
            'condition_id': meta.cid,
            'token_id_up': meta.token_up,
            'token_id_down': meta.token_dn,
            'window_start_ms': meta.window_start_ms,
            'window_end_ms': meta.window_end_ms,
            'time_remaining_s': time_remaining_s,
            'chainlink_price': binance_close,   # Binance proxy for chainlink
            'window_start_price': float(klines[0][1]),  # binance open at second 0
            'binance_mid': binance_close,
            # UP side
            'best_bid_up': bid_up,
            'best_ask_up': ask_up,
            'size_bid_up': np.nan,
            'size_ask_up': np.nan,
            'mid_up': mid_up,
            'spread_up': spread_up,
            # depth-5 columns: NaN (not available from REST)
            **{f'bid_px_up_{i}': np.nan for i in range(1, 6)},
            **{f'bid_sz_up_{i}': np.nan for i in range(1, 6)},
            **{f'ask_px_up_{i}': np.nan for i in range(1, 6)},
            **{f'ask_sz_up_{i}': np.nan for i in range(1, 6)},
            'bid_depth5_up': np.nan,
            'ask_depth5_up': np.nan,
            'imbalance5_up': np.nan,
            # DOWN side
            'best_bid_down': bid_dn,
            'best_ask_down': ask_dn,
            'size_bid_down': np.nan,
            'size_ask_down': np.nan,
            'mid_down': mid_dn,
            'spread_down': spread_dn,
            **{f'bid_px_down_{i}': np.nan for i in range(1, 6)},
            **{f'bid_sz_down_{i}': np.nan for i in range(1, 6)},
            **{f'ask_px_down_{i}': np.nan for i in range(1, 6)},
            **{f'ask_sz_down_{i}': np.nan for i in range(1, 6)},
            'bid_depth5_down': np.nan,
            'ask_depth5_down': np.nan,
            'imbalance5_down': np.nan,
            # Trade tape
            'last_trade_px_up':   lt_up[0] if lt_up else np.nan,
            'last_trade_sz_up':   lt_up[1] if lt_up else np.nan,
            'last_trade_side_up': lt_up[2] if lt_up else None,
            'last_trade_px_down':   lt_dn[0] if lt_dn else np.nan,
            'last_trade_sz_down':   lt_dn[1] if lt_dn else np.nan,
            'last_trade_side_down': lt_dn[2] if lt_dn else None,
        })

    if not rows:
        return None
    return pd.DataFrame(rows)


def backfill_one(slug: str, out_dir: Path, overwrite: bool = False) -> str:
    """Returns one of: 'written', 'skipped', 'no_event', 'no_data', 'error'.

    Single network round-trip per source: fetch_event → fetch_trades + fetch_binance
    in parallel via threads → build_parquet using already-fetched data.
    """
    out_path = out_dir / f'{slug}.parquet'
    if out_path.exists() and not overwrite:
        return 'skipped'
    try:
        meta = fetch_event(slug)
    except Exception as exc:
        print(f'  {slug}: gamma error {exc}')
        return 'error'
    if meta is None:
        return 'no_event'
    try:
        # Fetch trades and binance klines in parallel — both depend on meta only
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_trades = ex.submit(fetch_trades, meta.cid)
            f_klines = ex.submit(fetch_binance_klines, meta.window_start_ms, meta.window_end_ms)
            trades = f_trades.result()
            klines = f_klines.result()
        df = build_parquet(meta, trades=trades, klines=klines)
    except Exception as exc:
        print(f'  {slug}: build error {exc}')
        return 'error'
    if df is None or len(df) < 60:
        return 'no_data'
    df.to_parquet(out_path, index=False, compression='snappy')
    return 'written'


def enumerate_slugs(start_unix: int, end_unix: int, interval_s: int, prefix: str) -> list[str]:
    """Generate slugs at every `interval_s` boundary in the half-open [start, end) range."""
    out = []
    t = (start_unix // interval_s) * interval_s
    while t < end_unix:
        out.append(f'{prefix}-{t}')
        t += interval_s
    return out


def main():
    p = argparse.ArgumentParser(description='Polymarket REST → parquet backfill')
    p.add_argument('--slug', help='Single slug to backfill (test mode)')
    p.add_argument('--interval', choices=['5m', '15m'], default='5m')
    p.add_argument('--start', help='Start date YYYY-MM-DD (UTC)')
    p.add_argument('--end',   help='End date YYYY-MM-DD (UTC)')
    p.add_argument('--out-dir', help='Output dir; defaults to data/btc_5m or data/btc_15m')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--throttle-ms', type=int, default=0, help='Sleep between batches')
    p.add_argument('--workers', type=int, default=8,
                   help='Concurrent slug workers (each does its own gamma+trades+binance)')
    args = p.parse_args()

    if args.interval == '5m':
        interval_s = 300
        prefix = 'btc-updown-5m'
        default_out = REPO_ROOT / 'data' / 'btc_5m'
    else:
        interval_s = 900
        prefix = 'btc-updown-15m'
        default_out = REPO_ROOT / 'data' / 'btc_15m'
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.slug:
        slugs = [args.slug]
    else:
        if not args.start or not args.end:
            p.error('--start and --end required when --slug is not given')
        start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
        end_dt   = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
        slugs = enumerate_slugs(int(start_dt.timestamp()), int(end_dt.timestamp()),
                                interval_s, prefix)

    print(f'Backfilling {len(slugs)} slugs → {out_dir} (workers={args.workers})')
    counts = {'written': 0, 'skipped': 0, 'no_event': 0, 'no_data': 0, 'error': 0}
    t0 = time.time()

    from concurrent.futures import ThreadPoolExecutor, as_completed
    done_count = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(backfill_one, s, out_dir, args.overwrite): s for s in slugs}
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception as exc:
                print(f'  {futures[fut]}: future error {exc}')
                result = 'error'
            counts[result] += 1
            done_count += 1
            if done_count % 50 == 0 or done_count == len(slugs):
                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (len(slugs) - done_count) / rate if rate > 0 else 0
                print(f'  [{done_count:5d}/{len(slugs)}] {counts}  rate={rate:.1f}/s eta={eta:.0f}s')
    print(f'\nDONE in {time.time()-t0:.0f}s: {counts}')


if __name__ == '__main__':
    main()
