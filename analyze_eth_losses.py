#!/usr/bin/env python3
"""Diagnose why ETH is underperforming.

Breaks down ETH W/L by: side, hour-of-day UTC, book_age bucket, z-score bucket,
and entry-vs-final price reversal. Finds whichever dimension has the most
extreme WR gap — that's the cause.

Reads live_trades_*.jsonl (same format as analyze_wl.py). Matches each
RESOLVE row to its originating FILL row by (slug, ordinal) since resolves
pop open_positions FIFO.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


Z_RE = re.compile(r"z=(\d+(?:\.\d+)?)")
BOOK_AGE_RE = re.compile(r"book_age=(\d+)ms")
DELTA_RE = re.compile(r"delta=([+-]?\$\d+(?:\.\d+)?)")
ASK_RE = re.compile(r"ask=(0?\.\d+)")


TS_FRAC_RE = re.compile(r"(\.\d+)")


def parse_ts(s: str) -> datetime:
    """Rust chrono serializes with nanosecond precision (9 fractional digits);
    Python's fromisoformat accepts at most microseconds (6). Truncate if needed."""
    s = s.replace("Z", "+00:00")
    m = TS_FRAC_RE.search(s)
    if m and len(m.group(1)) > 7:  # '.' + >6 digits
        s = s[:m.start()] + m.group(1)[:7] + s[m.end():]
    return datetime.fromisoformat(s)


def extract_reason_fields(reason: str) -> dict:
    out = {}
    if m := Z_RE.search(reason):
        out["z"] = float(m.group(1))
    if m := BOOK_AGE_RE.search(reason):
        out["book_age_ms"] = int(m.group(1))
    if m := DELTA_RE.search(reason):
        raw = m.group(1).replace("$", "").replace("+", "")
        try:
            out["delta"] = float(raw)
        except ValueError:
            pass
    return out


def find_default_logs() -> list[Path]:
    seen: dict[Path, float] = {}
    for d in [Path.cwd(), Path.cwd() / "vps_logs", Path.home(),
              Path.home() / "polybot" / "rust", Path.home() / "polybot"]:
        if not d.is_dir():
            continue
        for m in d.glob("live_trades*.jsonl"):
            try: r = m.resolve()
            except OSError: continue
            if r not in seen:
                seen[r] = m.stat().st_mtime
    return [p for p, _ in sorted(seen.items(), key=lambda kv: kv[1])]


def bucket_wr(rows: list[dict], key_fn) -> None:
    """Group rows by key_fn(row) and print W/L/WR/PnL per bucket."""
    buckets: dict = defaultdict(lambda: {"w": 0, "l": 0, "pnl": 0.0, "cost": 0.0})
    for r in rows:
        k = key_fn(r)
        if k is None:
            continue
        pnl = float(r.get("trade_pnl", 0.0))
        cost = float(r.get("cost", 0.0))
        if pnl > 0:
            buckets[k]["w"] += 1
        else:
            buckets[k]["l"] += 1
        buckets[k]["pnl"] += pnl
        buckets[k]["cost"] += cost

    try:
        ordered = sorted(buckets.keys())
    except TypeError:
        ordered = list(buckets.keys())

    for k in ordered:
        s = buckets[k]
        n = s["w"] + s["l"]
        if n == 0:
            continue
        wr = s["w"] / n * 100
        roi = s["pnl"] / s["cost"] * 100 if s["cost"] > 0 else 0.0
        avg = s["pnl"] / n
        print(f"  {str(k):<14} n={n:<4}  W/L={s['w']:>3}/{s['l']:<3}  "
              f"WR={wr:>5.1f}%  PnL={s['pnl']:>+7.2f}  ROI={roi:>+6.2f}%  avg={avg:>+6.3f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", nargs="+", type=Path)
    p.add_argument("--asset", choices=["btc", "eth"], default="eth",
                   help="which asset to analyze (default: eth)")
    p.add_argument("--market", choices=["15m", "5m", "both"], default="both",
                   help="15m, 5m, or both (default)")
    args = p.parse_args()
    asset_tag = args.asset.lower()

    paths = args.log or find_default_logs()
    if not paths:
        print("no live_trades*.jsonl found", file=sys.stderr)
        return 1

    # Load all events, build slug→[fills] and slug→[resolves] maps (ts-sorted)
    fills_by_slug: dict[str, list[dict]] = defaultdict(list)
    resolves_by_slug: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        with open(path) as f:
            for line in f:
                try: r = json.loads(line.strip())
                except: continue
                if r.get("event") == "RESOLVE":
                    resolves_by_slug[r.get("slug", "")].append(r)
                elif "reason" in r and "order_id" in r:
                    fills_by_slug[r.get("slug", "")].append(r)

    for d in (fills_by_slug, resolves_by_slug):
        for slug in d:
            d[slug].sort(key=lambda r: r.get("ts", ""))

    # Pair each resolve with its matching fill (filtered by asset)
    rows = []
    for slug, resolves in resolves_by_slug.items():
        if asset_tag not in slug:
            continue
        if args.market == "15m" and "updown-15m" not in slug:
            continue
        if args.market == "5m" and "updown-5m" not in slug:
            continue
        fills = fills_by_slug.get(slug, [])
        for i, resolve in enumerate(resolves):
            fill = fills[i] if i < len(fills) else {}
            reason_fields = extract_reason_fields(fill.get("reason", ""))
            merged = {
                **resolve,
                **reason_fields,
                "ts_dt": parse_ts(resolve.get("ts", "1970-01-01T00:00:00Z")),
                "fill_ts_dt": parse_ts(fill["ts"]) if fill.get("ts") else None,
                "window_is_15m": "updown-15m" in slug,
            }
            rows.append(merged)

    if not rows:
        print(f"no {asset_tag.upper()} rows found", file=sys.stderr)
        return 1

    total_n = len(rows)
    total_pnl = sum(r.get("trade_pnl", 0.0) for r in rows)
    total_cost = sum(r.get("cost", 0.0) for r in rows)
    wins = sum(1 for r in rows if r.get("trade_pnl", 0.0) > 0)
    losses = total_n - wins

    print(f"{asset_tag.upper()} {args.market} — n={total_n}  W/L={wins}/{losses}  "
          f"WR={wins/total_n*100:.1f}%  PnL=${total_pnl:+.2f}  "
          f"ROI={total_pnl/total_cost*100 if total_cost else 0:+.2f}%")
    print()

    # ── 1. Side breakdown ─────────────────────────────────────
    print("[1] By side (BuyUp vs BuyDown):")
    bucket_wr(rows, lambda r: r.get("side", "?"))
    print()

    # ── 2. Hour-of-day UTC ────────────────────────────────────
    print("[2] By hour-of-day UTC (fire time):")
    bucket_wr(rows, lambda r: f"{r['fill_ts_dt'].hour:02d}z" if r.get("fill_ts_dt") else None)
    print()

    # ── 3. Z-score bucket ─────────────────────────────────────
    def z_bucket(r):
        z = r.get("z")
        if z is None:
            return None
        if z < 2: return "z<2 (weak)"
        if z < 3: return "z2-3"
        if z < 5: return "z3-5"
        if z < 8: return "z5-8"
        return "z8+ (strong)"
    print("[3] By Binance z-score at fire time:")
    bucket_wr(rows, z_bucket)
    print()

    # ── 4. Book-age bucket ────────────────────────────────────
    def age_bucket(r):
        a = r.get("book_age_ms")
        if a is None:
            return None
        if a < 1000: return "<1000ms"
        if a < 2000: return "1000-2000"
        if a < 3000: return "2000-3000"
        if a < 4000: return "3000-4000"
        return "4000+"
    print("[4] By Polymarket book staleness at fire time:")
    bucket_wr(rows, age_bucket)
    print()

    # ── 5. Entry vs final Chainlink price ─────────────────────
    # For BuyUp wins: final > entry. For BuyDown wins: final < entry.
    # If the PRICE MOVED IN OUR DIRECTION but we lost, something broke
    # (oracle lag, resolution snafu). If price reverted, it's a signal-
    # quality issue.
    print("[5] Entry → final price movement (Chainlink):")
    moved_with_us = {"w": 0, "l": 0, "pnl": 0.0}
    moved_against = {"w": 0, "l": 0, "pnl": 0.0}
    flat = {"w": 0, "l": 0, "pnl": 0.0}
    for r in rows:
        entry = r.get("entry_price")
        final = r.get("final_price")
        side = r.get("side", "")
        pnl = r.get("trade_pnl", 0.0)
        if entry is None or final is None:
            continue
        move = final - entry
        if abs(move) < 0.01:
            bucket = flat
        elif (side == "BuyUp" and move > 0) or (side == "BuyDown" and move < 0):
            bucket = moved_with_us
        else:
            bucket = moved_against
        if pnl > 0: bucket["w"] += 1
        else: bucket["l"] += 1
        bucket["pnl"] += pnl

    for name, b in [("moved WITH our bet", moved_with_us), ("moved AGAINST", moved_against), ("flat (<$0.01)", flat)]:
        n = b["w"] + b["l"]
        if n == 0: continue
        wr = b["w"]/n*100
        avg = b["pnl"]/n
        print(f"  {name:<22} n={n:<4}  W/L={b['w']:>3}/{b['l']:<3}  "
              f"WR={wr:>5.1f}%  PnL={b['pnl']:>+7.2f}  avg={avg:>+6.3f}")
    print()

    # ── 6. Price-move magnitude histogram (losing trades only).
    # Buckets are scaled per-asset since BTC moves in $10-100s and ETH in
    # $0.10-10s.
    if asset_tag == "btc":
        move_edges = [5, 15, 30, 60, 120, 250]
        move_labels = ["< $5", "$5-15", "$15-30", "$30-60", "$60-120", "$120-250", "$250+"]
    else:
        move_edges = [1, 3, 5, 10, 20, 40]
        move_labels = ["< $1", "$1-3", "$3-5", "$5-10", "$10-20", "$20-40", "$40+"]

    print(f"[6] Price move magnitude (|final - entry|) on LOSING {asset_tag.upper()} trades:")
    move_buckets: dict = defaultdict(lambda: {"n": 0, "pnl": 0.0})
    for r in rows:
        if r.get("trade_pnl", 0.0) > 0:
            continue
        entry = r.get("entry_price")
        final = r.get("final_price")
        if entry is None or final is None:
            continue
        move = abs(final - entry)
        idx = next((i for i, e in enumerate(move_edges) if move < e), len(move_edges))
        k = move_labels[idx]
        move_buckets[k]["n"] += 1
        move_buckets[k]["pnl"] += r.get("trade_pnl", 0.0)
    for k in move_labels:
        b = move_buckets.get(k)
        if not b or b["n"] == 0: continue
        print(f"  {k:<10}  n={b['n']:<4}  total_loss=${b['pnl']:+.2f}  avg=${b['pnl']/b['n']:+.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
