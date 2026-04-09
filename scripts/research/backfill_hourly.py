#!/usr/bin/env python3
"""
Backfill historical 1-hour "Up or Down" parquet files from Polymarket REST APIs.

Generates human-readable slugs (e.g. bitcoin-up-or-down-april-9-2026-4pm-et),
fetches market metadata from Gamma API, trade tape from Data API, and 1-second
Binance klines, then builds per-second parquet files matching the existing
recording.py schema so the backtest engine works without changes.

Usage:
    python scripts/research/backfill_hourly.py --asset bitcoin --days 30
    python scripts/research/backfill_hourly.py --asset bitcoin --days 3 --dry-run
    python scripts/research/backfill_hourly.py --asset bitcoin --start 2026-03-15 --end 2026-04-05
"""

from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import time
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent
GAMMA = "https://gamma-api.polymarket.com"
DATA = "https://data-api.polymarket.com"
BINANCE = "https://api.binance.com"

USER_AGENT = "btc-bot-backfill-hourly/0.1"

ET = ZoneInfo("America/New_York")


# ── HTTP helpers ─────────────────────────────────────────────────────────────

class HttpStatusError(Exception):
    def __init__(self, code: int, url: str):
        super().__init__(f"HTTP {code} {url}")
        self.code = code


def http_get(url: str, retries: int = 6, sleep_s: float = 0.5):
    """GET with retry. 429 = rate limit with exponential backoff.
    Other 4xx raised immediately. 5xx and network errors retry.
    """
    import urllib.error
    last_exc = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                time.sleep(sleep_s * (2 ** attempt) + 0.5 * attempt)
                last_exc = exc
                continue
            if 400 <= exc.code < 500:
                raise HttpStatusError(exc.code, url) from exc
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(sleep_s * (2 ** attempt))
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(sleep_s * (2 ** attempt))
    raise RuntimeError(f"http_get({url}): {last_exc}")


# ── Slug generation ──────────────────────────────────────────────────────────

# Month names for slug construction (Polymarket uses lowercase full names)
_MONTH_NAMES = {
    1: "january", 2: "february", 3: "march", 4: "april",
    5: "may", 6: "june", 7: "july", 8: "august",
    9: "september", 10: "october", 11: "november", 12: "december",
}


def _hour_to_slug_suffix(hour_24: int) -> str:
    """Convert 24h hour to '12am', '1am', ..., '12pm', '1pm', ..., '11pm'."""
    if hour_24 == 0:
        return "12am"
    elif hour_24 < 12:
        return f"{hour_24}am"
    elif hour_24 == 12:
        return "12pm"
    else:
        return f"{hour_24 - 12}pm"


def generate_hourly_slugs(
    asset: str,
    start_date: date,
    end_date: date,
) -> list[tuple[str, datetime, datetime]]:
    """Generate (slug, window_start_utc, window_end_utc) for each hour.

    Parameters
    ----------
    asset : str
        Polymarket asset name: "bitcoin", "solana", "ethereum", "xrp"
    start_date : date
        First day (inclusive) in ET
    end_date : date
        Last day (inclusive) in ET

    Returns
    -------
    List of (slug, window_start_utc, window_end_utc) tuples.
    """
    results = []
    current_date = start_date
    while current_date <= end_date:
        for hour in range(24):
            # Window start in ET
            et_start = datetime(
                current_date.year, current_date.month, current_date.day,
                hour, 0, 0, tzinfo=ET,
            )
            et_end = et_start + timedelta(hours=1)

            # Build slug: {asset}-up-or-down-{month}-{day}-{year}-{hour}{ampm}-et
            month_name = _MONTH_NAMES[current_date.month]
            day = current_date.day  # no zero-padding
            year = current_date.year
            hour_suffix = _hour_to_slug_suffix(hour)

            slug = f"{asset}-up-or-down-{month_name}-{day}-{year}-{hour_suffix}-et"

            # Convert to UTC for API calls / parquet timestamps
            window_start_utc = et_start.astimezone(timezone.utc)
            window_end_utc = et_end.astimezone(timezone.utc)

            results.append((slug, window_start_utc, window_end_utc))
        current_date += timedelta(days=1)
    return results


# ── Market metadata fetch ────────────────────────────────────────────────────

def fetch_event(slug: str, window_start_utc: datetime, window_end_utc: datetime):
    """Look up a Polymarket event by slug. Returns metadata dict or None."""
    url = f"{GAMMA}/events?slug={urllib.parse.quote(slug)}"
    arr = http_get(url)
    if not arr:
        return None
    ev = arr[0]
    if not ev.get("markets"):
        return None
    mk = ev["markets"][0]
    cid = mk.get("conditionId")
    if not cid:
        return None

    # Check if market is resolved
    closed = mk.get("closed", False)
    if not closed:
        return None  # Still open — skip

    outcomes = mk.get("outcomes", [])
    tokens = mk.get("clobTokenIds", [])
    if isinstance(outcomes, str):
        outcomes = json.loads(outcomes)
    if isinstance(tokens, str):
        tokens = json.loads(tokens)
    if "Up" not in outcomes or "Down" not in outcomes:
        return None
    up_idx = outcomes.index("Up")
    dn_idx = outcomes.index("Down")

    return {
        "slug": slug,
        "cid": cid,
        "token_up": str(tokens[up_idx]),
        "token_dn": str(tokens[dn_idx]),
        "window_start_ms": int(window_start_utc.timestamp() * 1000),
        "window_end_ms": int(window_end_utc.timestamp() * 1000),
    }


# ── Trade tape ───────────────────────────────────────────────────────────────

def fetch_trades(cid: str, max_pages: int = 20) -> list[dict]:
    """Paginate /trades?market={cid}. Higher max_pages for 1h (more trades).

    The data-api enforces a hard offset cap (around 3500). Once we hit a
    400, we stop paginating and return whatever we've collected.
    """
    out = []
    for page in range(max_pages):
        url = f"{DATA}/trades?market={cid}&limit=500&offset={page * 500}&takerOnly=true"
        try:
            rows = http_get(url)
        except HttpStatusError as exc:
            if exc.code == 400:
                break
            raise
        if not rows:
            break
        out.extend(rows)
        if len(rows) < 500:
            break
    return out


# ── Binance klines (paginated for 3600s windows) ────────────────────────────

def fetch_binance_klines(
    start_ms: int,
    end_ms: int,
    symbol: str = "BTCUSDT",
) -> list[list]:
    """Fetch 1-second klines from Binance, paginating in 1000-bar chunks.

    Binance kline API returns max 1000 bars per request. For a 3600-second
    window we need 4 requests.
    """
    all_klines = []
    cursor_ms = start_ms
    while cursor_ms < end_ms:
        url = (
            f"{BINANCE}/api/v3/klines?symbol={symbol}&interval=1s"
            f"&startTime={cursor_ms}&endTime={end_ms}&limit=1000"
        )
        batch = http_get(url)
        if not batch:
            break
        all_klines.extend(batch)
        # Advance cursor past the last kline's open time
        last_open_ms = int(batch[-1][0])
        cursor_ms = last_open_ms + 1000  # next second
        if len(batch) < 1000:
            break
        # Brief pause to respect Binance rate limits
        time.sleep(0.1)
    return all_klines


# ── Parquet construction ─────────────────────────────────────────────────────

def build_parquet(
    meta: dict,
    trades: list[dict] | None = None,
    klines: list[list] | None = None,
    binance_symbol: str = "BTCUSDT",
) -> pd.DataFrame | None:
    """Build a per-second snapshot DataFrame matching recording.py schema.

    Reuses the same bid/ask approximation from trade tape as
    analysis/polymarket_rest_backfill.py.
    """
    if trades is None:
        trades = fetch_trades(meta["cid"])
    if klines is None:
        klines = fetch_binance_klines(
            meta["window_start_ms"], meta["window_end_ms"],
            symbol=binance_symbol,
        )
    if not klines:
        return None

    # Sort trades chronologically
    trades.sort(key=lambda t: t["timestamp"])

    # Rolling state: latest known bid/ask from trade tape
    rolling = {
        "last_buy_up": None, "last_sell_up": None,
        "last_buy_dn": None, "last_sell_dn": None,
        "last_trade_up": None,
        "last_trade_dn": None,
    }

    rows = []
    trade_idx = 0
    n_trades = len(trades)
    window_start_price = float(klines[0][1])  # binance open at first second

    for k in klines:
        ts_ms = int(k[0])
        ts_s = ts_ms // 1000
        binance_close = float(k[4])

        # Apply all trades up to and including this second
        while trade_idx < n_trades and trades[trade_idx]["timestamp"] <= ts_s:
            t = trades[trade_idx]
            outcome = t["outcome"]
            side = t["side"]
            price = float(t["price"])
            size = float(t["size"])
            if outcome == "Up":
                if side == "BUY":
                    rolling["last_buy_up"] = price
                else:
                    rolling["last_sell_up"] = price
                rolling["last_trade_up"] = (price, size, side)
            elif outcome == "Down":
                if side == "BUY":
                    rolling["last_buy_dn"] = price
                else:
                    rolling["last_sell_dn"] = price
                rolling["last_trade_dn"] = (price, size, side)
            trade_idx += 1

        # Approximate book state at this second
        ask_up = rolling["last_buy_up"]
        bid_up = rolling["last_sell_up"]
        ask_dn = rolling["last_buy_dn"]
        bid_dn = rolling["last_sell_dn"]

        # Cross-side fallback: bid_up ~ 1 - ask_dn, etc.
        if bid_up is None and ask_dn is not None:
            bid_up = max(0.001, 1.0 - ask_dn)
        if ask_up is None and bid_dn is not None:
            ask_up = min(0.999, 1.0 - bid_dn)
        if bid_dn is None and ask_up is not None:
            bid_dn = max(0.001, 1.0 - ask_up)
        if ask_dn is None and bid_up is not None:
            ask_dn = min(0.999, 1.0 - bid_up)

        # Skip rows where we have no book state yet (warmup period)
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

        lt_up = rolling["last_trade_up"]
        lt_dn = rolling["last_trade_dn"]

        time_remaining_s = max(0.0, (meta["window_end_ms"] - ts_ms) / 1000.0)
        rows.append({
            "ts_ms": ts_ms,
            "market_slug": meta["slug"],
            "condition_id": meta["cid"],
            "token_id_up": meta["token_up"],
            "token_id_down": meta["token_dn"],
            "window_start_ms": meta["window_start_ms"],
            "window_end_ms": meta["window_end_ms"],
            "time_remaining_s": time_remaining_s,
            "chainlink_price": binance_close,
            "window_start_price": window_start_price,
            "binance_mid": binance_close,
            # UP side
            "best_bid_up": bid_up,
            "best_ask_up": ask_up,
            "size_bid_up": np.nan,
            "size_ask_up": np.nan,
            "mid_up": mid_up,
            "spread_up": spread_up,
            **{f"bid_px_up_{i}": np.nan for i in range(1, 6)},
            **{f"bid_sz_up_{i}": np.nan for i in range(1, 6)},
            **{f"ask_px_up_{i}": np.nan for i in range(1, 6)},
            **{f"ask_sz_up_{i}": np.nan for i in range(1, 6)},
            "bid_depth5_up": np.nan,
            "ask_depth5_up": np.nan,
            "imbalance5_up": np.nan,
            # DOWN side
            "best_bid_down": bid_dn,
            "best_ask_down": ask_dn,
            "size_bid_down": np.nan,
            "size_ask_down": np.nan,
            "mid_down": mid_dn,
            "spread_down": spread_dn,
            **{f"bid_px_down_{i}": np.nan for i in range(1, 6)},
            **{f"bid_sz_down_{i}": np.nan for i in range(1, 6)},
            **{f"ask_px_down_{i}": np.nan for i in range(1, 6)},
            **{f"ask_sz_down_{i}": np.nan for i in range(1, 6)},
            "bid_depth5_down": np.nan,
            "ask_depth5_down": np.nan,
            "imbalance5_down": np.nan,
            # Trade tape
            "last_trade_px_up":   lt_up[0] if lt_up else np.nan,
            "last_trade_sz_up":   lt_up[1] if lt_up else np.nan,
            "last_trade_side_up": lt_up[2] if lt_up else None,
            "last_trade_px_down":   lt_dn[0] if lt_dn else np.nan,
            "last_trade_sz_down":   lt_dn[1] if lt_dn else np.nan,
            "last_trade_side_down": lt_dn[2] if lt_dn else None,
        })

    if not rows:
        return None
    return pd.DataFrame(rows)


# ── Single-slug backfill ─────────────────────────────────────────────────────

def backfill_one(
    slug: str,
    window_start_utc: datetime,
    window_end_utc: datetime,
    out_dir: Path,
    binance_symbol: str = "BTCUSDT",
    overwrite: bool = False,
) -> str:
    """Returns one of: 'written', 'skipped', 'not_resolved', 'no_event', 'no_data', 'error'."""
    out_path = out_dir / f"{slug}.parquet"
    if out_path.exists() and not overwrite:
        return "skipped"

    # Skip windows that haven't ended yet
    now_utc = datetime.now(timezone.utc)
    if window_end_utc > now_utc:
        return "not_resolved"

    try:
        meta = fetch_event(slug, window_start_utc, window_end_utc)
    except Exception as exc:
        print(f"  {slug}: gamma error {exc}")
        return "error"
    if meta is None:
        return "no_event"

    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_trades = ex.submit(fetch_trades, meta["cid"])
            f_klines = ex.submit(
                fetch_binance_klines,
                meta["window_start_ms"],
                meta["window_end_ms"],
                binance_symbol,
            )
            trades = f_trades.result()
            klines = f_klines.result()
        df = build_parquet(meta, trades=trades, klines=klines, binance_symbol=binance_symbol)
    except Exception as exc:
        print(f"  {slug}: build error {exc}")
        return "error"

    if df is None or len(df) < 120:
        # 1h window should have at least a couple minutes of data
        return "no_data"

    df.to_parquet(out_path, index=False, compression="snappy")
    return "written"


# ── CLI ──────────────────────────────────────────────────────────────────────

_ASSET_BINANCE = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "solana": "SOLUSDT",
    "xrp": "XRPUSDT",
}

_ASSET_SUBDIR = {
    "bitcoin": "btc_1h",
    "ethereum": "eth_1h",
    "solana": "sol_1h",
    "xrp": "xrp_1h",
}


def main():
    p = argparse.ArgumentParser(
        description="Polymarket 1h Up/Down → parquet backfill",
    )
    p.add_argument(
        "--asset",
        choices=list(_ASSET_BINANCE.keys()),
        default="bitcoin",
        help="Asset to backfill (default: bitcoin)",
    )
    p.add_argument("--days", type=int, help="Number of days back from today to backfill")
    p.add_argument("--start", help="Start date YYYY-MM-DD (ET)")
    p.add_argument("--end", help="End date YYYY-MM-DD (ET)")
    p.add_argument("--out-dir", help="Output directory (default: data/{asset}_1h)")
    p.add_argument("--overwrite", action="store_true", help="Re-fetch existing parquets")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print slugs without fetching",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent slug workers (default: 4, conservative for rate limits)",
    )
    args = p.parse_args()

    # Determine date range
    if args.days:
        today_et = datetime.now(ET).date()
        end_date = today_et
        start_date = today_et - timedelta(days=args.days)
    elif args.start and args.end:
        start_date = date.fromisoformat(args.start)
        end_date = date.fromisoformat(args.end)
    else:
        p.error("Either --days or both --start and --end are required")

    # Output directory
    subdir = _ASSET_SUBDIR[args.asset]
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "data" / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    binance_symbol = _ASSET_BINANCE[args.asset]

    # Generate all slugs
    slugs = generate_hourly_slugs(args.asset, start_date, end_date)
    print(f"Generated {len(slugs)} hourly slugs for {args.asset} "
          f"from {start_date} to {end_date}")

    if args.dry_run:
        for slug, ws, we in slugs:
            status = "FUTURE" if we > datetime.now(timezone.utc) else "past"
            print(f"  {slug}  |  {ws.isoformat()} → {we.isoformat()}  [{status}]")
        print(f"\nTotal: {len(slugs)} slugs ({args.asset})")
        return

    print(f"Backfilling → {out_dir} (workers={args.workers})")
    counts = {
        "written": 0, "skipped": 0, "not_resolved": 0,
        "no_event": 0, "no_data": 0, "error": 0,
    }
    t0 = time.time()
    done_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                backfill_one, slug, ws, we, out_dir,
                binance_symbol, args.overwrite,
            ): slug
            for slug, ws, we in slugs
        }
        for fut in as_completed(futures):
            slug_name = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                print(f"  {slug_name}: future error {exc}")
                result = "error"
            counts[result] += 1
            done_count += 1
            if result == "written":
                print(f"  OK {slug_name}")
            if done_count % 25 == 0 or done_count == len(slugs):
                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (len(slugs) - done_count) / rate if rate > 0 else 0
                print(f"  [{done_count:5d}/{len(slugs)}] {counts}  "
                      f"rate={rate:.1f}/s eta={eta:.0f}s")

    print(f"\nDONE in {time.time() - t0:.0f}s: {counts}")


if __name__ == "__main__":
    main()
