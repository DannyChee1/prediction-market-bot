#!/usr/bin/env python3
"""Per-slot W/L analysis — does fire #2 in a window outperform fire #1?

Each window (unique slug) can host up to `max_positions_per_window` fires.
We tag them slot 1, 2, ... by entry timestamp, then break out WR/PnL/ROI
per (market, slot). Answers: "is the second fire in a window worth it, or
am I diluting my edge on correlated signals?"

Input: the same RESOLVE rows from live_trades_*.jsonl that analyze_wl.py
reads. RESOLVEs preserve the order positions entered (resolve_position
pops open_positions FIFO), so the Nth RESOLVE for a slug = Nth fire.

Usage:
    ./analyze_slots.py                           # auto-find logs in cwd / ~/ / ~/polybot/
    ./analyze_slots.py --log PATH [PATH ...]     # explicit
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def classify_slug(slug: str) -> str:
    if "updown-15m" in slug and "btc" in slug: return "BTC 15m"
    if "updown-5m"  in slug and "btc" in slug: return "BTC 5m"
    if "updown-15m" in slug and "eth" in slug: return "ETH 15m"
    if "updown-5m"  in slug and "eth" in slug: return "ETH 5m"
    return "other"


def find_default_logs() -> list[Path]:
    """Match analyze_wl.py's auto-discovery."""
    seen: dict[Path, float] = {}
    for d in [Path.cwd(), Path.cwd() / "vps_logs", Path.home(),
              Path.home() / "polybot" / "rust", Path.home() / "polybot"]:
        if not d.is_dir():
            continue
        for m in d.glob("live_trades*.jsonl"):
            try:
                r = m.resolve()
            except OSError:
                continue
            if r not in seen:
                seen[r] = m.stat().st_mtime
    return [p for p, _ in sorted(seen.items(), key=lambda kv: kv[1])]


def parse(paths: list[Path]):
    by_slug: dict[str, list[dict]] = defaultdict(list)
    total, resolves = 0, 0
    for path in paths:
        with open(path) as f:
            for line in f:
                total += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("event") != "RESOLVE":
                    continue
                resolves += 1
                slug = r.get("slug", "")
                if slug:
                    by_slug[slug].append(r)
    # Sort each slug's events by timestamp so slot ordering is chronological.
    for slug in by_slug:
        by_slug[slug].sort(key=lambda r: r.get("ts", ""))
    return by_slug, total, resolves


def aggregate(by_slug):
    # bucket[(market, slot)] → stats
    buckets: dict[tuple[str, int], dict] = defaultdict(
        lambda: {"w": 0, "l": 0, "pnl": 0.0, "cost": 0.0, "n": 0}
    )
    # Track how many slugs had at least N fires (for "% of windows that had slot N")
    slot_coverage: dict[tuple[str, int], int] = defaultdict(int)
    windows_per_market: dict[str, int] = defaultdict(int)

    for slug, events in by_slug.items():
        market = classify_slug(slug)
        windows_per_market[market] += 1
        max_slot = len(events)
        for idx, r in enumerate(events, start=1):
            slot = idx
            key = (market, slot)
            pnl = float(r.get("trade_pnl", 0.0))
            cost = float(r.get("cost", 0.0))
            if pnl > 0:
                buckets[key]["w"] += 1
            else:
                buckets[key]["l"] += 1
            buckets[key]["pnl"] += pnl
            buckets[key]["cost"] += cost
            buckets[key]["n"] += 1
        for slot in range(1, max_slot + 1):
            slot_coverage[(market, slot)] += 1

    return buckets, slot_coverage, windows_per_market


def fmt_row(market, slot, s, coverage, total_windows):
    n = s["w"] + s["l"]
    wr = (s["w"] / n * 100.0) if n else 0.0
    roi = (s["pnl"] / s["cost"] * 100.0) if s["cost"] > 0 else 0.0
    avg = (s["pnl"] / n) if n else 0.0
    pct = (coverage / total_windows * 100.0) if total_windows else 0.0
    return (
        f"{market:<10} slot{slot:>2}  "
        f"{s['w']:>4}/{s['l']:<4}  {wr:>5.1f}%  "
        f"{s['pnl']:>+8.2f}  {roi:>+7.2f}%  {avg:>+6.3f}  "
        f"({coverage} windows, {pct:.0f}%)"
    )


def compare_edge(buckets, market):
    """Print the slot-2 vs slot-1 delta for a market. Negative means fire #2
    is worse than fire #1 — evidence for a ramping threshold."""
    s1 = buckets.get((market, 1))
    s2 = buckets.get((market, 2))
    if not s1 or not s2 or s2["n"] == 0:
        return None
    n1, n2 = s1["w"] + s1["l"], s2["w"] + s2["l"]
    wr1 = s1["w"] / n1 * 100 if n1 else 0
    wr2 = s2["w"] / n2 * 100 if n2 else 0
    roi1 = s1["pnl"] / s1["cost"] * 100 if s1["cost"] else 0
    roi2 = s2["pnl"] / s2["cost"] * 100 if s2["cost"] else 0
    avg1 = s1["pnl"] / n1 if n1 else 0
    avg2 = s2["pnl"] / n2 if n2 else 0
    return {
        "wr_delta_pp": wr2 - wr1,
        "roi_delta_pp": roi2 - roi1,
        "avg_delta": avg2 - avg1,
        "n1": n1, "n2": n2,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", nargs="+", type=Path)
    args = p.parse_args()

    if args.log:
        paths = args.log
    else:
        paths = find_default_logs()
        if not paths:
            print("no live_trades*.jsonl found; pass --log PATH", file=sys.stderr)
            return 1
    for path in paths:
        if not path.is_file():
            print(f"not a file: {path}", file=sys.stderr)
            return 1

    by_slug, total_rows, resolve_rows = parse(paths)
    buckets, coverage, windows_per_market = aggregate(by_slug)

    print("log: " + ", ".join(str(p) for p in paths))
    print(f"rows: {total_rows} total, {resolve_rows} RESOLVE, {len(by_slug)} unique windows resolved")
    print()

    print(f"{'market':<10} {'slot':<6}  {'W/L':>9}  {'WR':>6}  {'PnL':>8}  {'ROI':>8}  {'avg':>6}  coverage")
    print("-" * 80)

    order = ["BTC 15m", "BTC 5m", "ETH 15m", "ETH 5m", "other"]
    for market in order:
        slots = sorted({k[1] for k in buckets if k[0] == market})
        if not slots:
            continue
        total_windows = windows_per_market[market]
        for slot in slots:
            s = buckets[(market, slot)]
            if s["n"] == 0:
                continue
            cov = coverage.get((market, slot), 0)
            print(fmt_row(market, slot, s, cov, total_windows))
        # Slot 2 vs slot 1 delta for this market
        cmp = compare_edge(buckets, market)
        if cmp:
            sign_wr  = "+" if cmp["wr_delta_pp"]  >= 0 else ""
            sign_roi = "+" if cmp["roi_delta_pp"] >= 0 else ""
            sign_avg = "+" if cmp["avg_delta"]    >= 0 else ""
            print(
                f"           slot2−slot1: "
                f"WR {sign_wr}{cmp['wr_delta_pp']:.1f}pp  "
                f"ROI {sign_roi}{cmp['roi_delta_pp']:.2f}pp  "
                f"avg {sign_avg}${cmp['avg_delta']:.3f}  "
                f"(n={cmp['n1']} vs {cmp['n2']})"
            )
        print()

    # Combined/total across all markets
    agg_by_slot: dict[int, dict] = defaultdict(
        lambda: {"w": 0, "l": 0, "pnl": 0.0, "cost": 0.0, "n": 0}
    )
    for (_market, slot), s in buckets.items():
        for k in agg_by_slot[slot]:
            agg_by_slot[slot][k] += s[k]

    print("--- TOTAL (all markets combined) ---")
    for slot in sorted(agg_by_slot):
        s = agg_by_slot[slot]
        if s["n"] == 0:
            continue
        n = s["w"] + s["l"]
        wr = s["w"] / n * 100 if n else 0
        roi = s["pnl"] / s["cost"] * 100 if s["cost"] > 0 else 0
        avg = s["pnl"] / n if n else 0
        print(
            f"slot{slot:>2}  {s['w']:>4}/{s['l']:<4}  {wr:>5.1f}%  "
            f"{s['pnl']:>+8.2f}  {roi:>+7.2f}%  {avg:>+6.3f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
