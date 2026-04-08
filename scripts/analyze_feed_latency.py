#!/usr/bin/env python3
"""
F1 Phase 1 — Analyze feed_latency.jsonl from measure_feed_latency.py.

Computes per-feed:
  - Sample count
  - p50 / p95 / p99 / p99.9 of (age_ms, gap_to_prev_ms)
  - Min / max
  - Cross-correlation between feeds (when does one lead the other)

Run after a measurement campaign:
    uv run python scripts/analyze_feed_latency.py feed_latency.jsonl

Optional:
    --plot       generate matplotlib histograms (requires matplotlib)
    --window-min N   only analyze the last N minutes of data
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def percentile(sorted_list, p):
    if not sorted_list:
        return None
    if p <= 0:
        return sorted_list[0]
    if p >= 100:
        return sorted_list[-1]
    k = (len(sorted_list) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_list[int(k)]
    d0 = sorted_list[int(f)] * (c - k)
    d1 = sorted_list[int(c)] * (k - f)
    return d0 + d1


def fmt_ms(v):
    if v is None:
        return "  --  "
    if v >= 1000:
        return f"{v/1000:6.2f}s"
    return f"{v:6.0f}ms"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=str,
                        help="Path to feed_latency.jsonl from measure_feed_latency.py")
    parser.add_argument("--window-min", type=int, default=0,
                        help="Only analyze the last N minutes (0=all)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate matplotlib histogram plots")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"  ERROR: {in_path} not found")
        return 1

    rows_by_feed = defaultdict(list)
    with open(in_path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows_by_feed[d["feed"]].append(d)

    if not rows_by_feed:
        print(f"  No data in {in_path}")
        return 1

    print()
    print(f"  Source: {in_path}")
    total = sum(len(v) for v in rows_by_feed.values())
    print(f"  Records: {total}")

    if args.window_min > 0:
        latest_ms = max(
            r["ts_local_ms"] for rows in rows_by_feed.values() for r in rows
        )
        cutoff = latest_ms - args.window_min * 60 * 1000
        for feed in list(rows_by_feed.keys()):
            rows_by_feed[feed] = [r for r in rows_by_feed[feed]
                                  if r["ts_local_ms"] >= cutoff]
        kept = sum(len(v) for v in rows_by_feed.values())
        print(f"  Window: last {args.window_min} min ({kept} records kept)")

    print()
    header = (f"  {'feed':<20} {'n':>7}  "
              f"{'age p50':>9} {'age p95':>9} {'age p99':>9} {'age max':>9}  "
              f"{'gap p50':>9} {'gap p95':>9} {'gap p99':>9} {'gap max':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for feed in sorted(rows_by_feed.keys()):
        rows = rows_by_feed[feed]
        ages = sorted(r["age_ms"] for r in rows if r.get("age_ms") is not None)
        gaps = sorted(r["gap_to_prev_ms"] for r in rows
                      if r.get("gap_to_prev_ms") is not None)

        age_p50 = percentile(ages, 50)
        age_p95 = percentile(ages, 95)
        age_p99 = percentile(ages, 99)
        age_max = ages[-1] if ages else None

        gap_p50 = percentile(gaps, 50)
        gap_p95 = percentile(gaps, 95)
        gap_p99 = percentile(gaps, 99)
        gap_max = gaps[-1] if gaps else None

        print(f"  {feed:<20} {len(rows):>7}  "
              f"{fmt_ms(age_p50)} {fmt_ms(age_p95)} {fmt_ms(age_p99)} {fmt_ms(age_max)}  "
              f"{fmt_ms(gap_p50)} {fmt_ms(gap_p95)} {fmt_ms(gap_p99)} {fmt_ms(gap_max)}")

    print()
    print("  Notes:")
    print("    age_ms = local_recv - server_event_ms (only for feeds that")
    print("             expose a server timestamp; binance_bookticker is None)")
    print("    gap_to_prev_ms = ms since previous message on the same feed")
    print()

    # ── Rebroadcast tax estimate ──
    # If we have both rtds_chainlink (which has age_ms = rebroadcast tax)
    # and binance_trade (which has age_ms = network round-trip), the
    # difference is the EXTRA latency Polymarket adds on top of normal
    # network latency. This is what direct Chainlink would save.
    rtds_ages = [r["age_ms"] for r in rows_by_feed.get("rtds_chainlink", [])
                 if r.get("age_ms") is not None]
    binance_ages = [r["age_ms"] for r in rows_by_feed.get("binance_trade", [])
                    if r.get("age_ms") is not None]
    if rtds_ages and binance_ages:
        rtds_p50 = percentile(sorted(rtds_ages), 50)
        bnc_p50 = percentile(sorted(binance_ages), 50)
        tax = rtds_p50 - bnc_p50
        print(f"  Estimated rebroadcast tax (rtds_p50 - binance_p50):")
        print(f"    rtds_chainlink p50:  {fmt_ms(rtds_p50)}")
        print(f"    binance_trade p50:   {fmt_ms(bnc_p50)}")
        print(f"    EXTRA tax from RTDS: {fmt_ms(tax)}")
        print()
        if tax > 200:
            print(f"  → F2 (direct Chainlink) is HIGH ROI ({fmt_ms(tax)} savings)")
        elif tax > 50:
            print(f"  → F2 (direct Chainlink) is MEDIUM ROI ({fmt_ms(tax)} savings)")
        else:
            print(f"  → F2 (direct Chainlink) is LOW ROI (only {fmt_ms(tax)} savings)")

    # ── Plot ──
    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("  --plot requires matplotlib. pip install matplotlib")
            return 0
        fig, axes = plt.subplots(len(rows_by_feed), 2,
                                  figsize=(12, 3 * len(rows_by_feed)))
        if len(rows_by_feed) == 1:
            axes = [axes]
        for ax_row, (feed, rows) in zip(axes, sorted(rows_by_feed.items())):
            ages = [r["age_ms"] for r in rows if r.get("age_ms") is not None]
            gaps = [r["gap_to_prev_ms"] for r in rows
                    if r.get("gap_to_prev_ms") is not None]
            if ages:
                ax_row[0].hist(ages, bins=50, alpha=0.7)
                ax_row[0].set_title(f"{feed} — age_ms")
                ax_row[0].set_xlabel("ms")
                ax_row[0].set_ylabel("count")
            else:
                ax_row[0].text(0.5, 0.5, "no age_ms", ha="center", va="center")
                ax_row[0].set_title(f"{feed} — age_ms (none)")
            if gaps:
                ax_row[1].hist(gaps, bins=50, alpha=0.7, color="orange")
                ax_row[1].set_title(f"{feed} — gap_to_prev_ms")
                ax_row[1].set_xlabel("ms")
            else:
                ax_row[1].text(0.5, 0.5, "no gaps", ha="center", va="center")
        plt.tight_layout()
        out_png = in_path.with_suffix(".png")
        plt.savefig(out_png, dpi=100)
        print(f"  Plot saved to {out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
