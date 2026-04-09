#!/usr/bin/env python3
"""Quick PnL/PF report for live BTC trades.

Usage:
    uv run python scripts/show_pf.py                  # all-time
    uv run python scripts/show_pf.py --since 2h       # last 2 hours
    uv run python scripts/show_pf.py --since 30m      # last 30 minutes
    uv run python scripts/show_pf.py --since 2026-04-09T01:54  # since timestamp

PF (Profit Factor) = sum(wins) / |sum(losses)|
  PF > 1.0 = profitable
  PF = 1.0 = break-even
  PF < 1.0 = losing money
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "live_trades_btc.jsonl"


def parse_since(s: str | None) -> datetime | None:
    if not s:
        return None
    # Relative duration like "30m", "2h", "1d"
    m = re.fullmatch(r"(\d+)\s*([smhd])", s.strip().lower())
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        delta = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}[unit]
        return datetime.now(timezone.utc) - timedelta(**{delta: n})
    # Absolute ISO timestamp
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        print(f"ERROR: bad --since value '{s}' (use '30m', '2h', '1d', or ISO timestamp)")
        sys.exit(1)
    # If naive (no tz), assume UTC so comparisons against tz-aware log timestamps work
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def bucket_for(slug: str) -> str:
    if "15m" in slug:
        return "15m"
    if "5m" in slug:
        return "5m"
    return "?"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", help="filter trades since this point (e.g. '2h', '30m', or ISO timestamp)")
    ap.add_argument("--log", default=str(LOG), help=f"path to trade log (default: {LOG})")
    args = ap.parse_args()

    cutoff = parse_since(args.since)

    # buckets: market -> {n, w, l, wp, lp, sides}
    buckets: dict[str, dict] = defaultdict(lambda: {
        "n": 0, "w": 0, "l": 0, "wp": 0.0, "lp": 0.0,
        "sides": defaultdict(lambda: {"n": 0, "w": 0, "pnl": 0.0}),
    })

    with open(args.log) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") != "resolution":
                continue
            ts_str = r.get("ts", "")
            if cutoff and ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    if ts < cutoff:
                        continue
                except ValueError:
                    pass
            slug = r.get("market_slug", "")
            k = bucket_for(slug)
            b = buckets[k]
            pnl = r.get("pnl", 0.0)
            side = r.get("side", "?")
            b["n"] += 1
            if pnl > 0.001:
                b["w"] += 1
                b["wp"] += pnl
            elif pnl < -0.001:
                b["l"] += 1
                b["lp"] += abs(pnl)
            b["sides"][side]["n"] += 1
            b["sides"][side]["pnl"] += pnl
            if r.get("side") == r.get("outcome"):
                b["sides"][side]["w"] += 1

    if not buckets:
        when = f"since {args.since}" if args.since else "in log"
        print(f"No resolved trades {when}. Bot may not have any fills yet.")
        return

    # Header
    if cutoff:
        print(f"Window: since {cutoff.isoformat()}\n")
    else:
        print("Window: all-time\n")

    print(f"{'mkt':5s} {'n':>4s} {'win%':>6s} {'PnL':>10s} {'avg_w':>8s} {'avg_l':>8s} {'PF':>6s}")
    print("-" * 60)
    total_n = total_pnl = total_wp = total_lp = 0
    for k in sorted(buckets):
        d = buckets[k]
        wr = d["w"] / d["n"] * 100 if d["n"] else 0
        pnl = d["wp"] - d["lp"]
        avg_w = d["wp"] / d["w"] if d["w"] else 0
        avg_l = d["lp"] / d["l"] if d["l"] else 0
        pf = d["wp"] / d["lp"] if d["lp"] > 0 else float("inf")
        pf_s = f"{pf:>5.2f}" if pf != float("inf") else "  inf"
        print(f"{k:5s} {d['n']:>4d} {wr:>5.1f}% ${pnl:>+8.2f} ${avg_w:>+6.2f} ${avg_l:>+6.2f} {pf_s}")
        total_n += d["n"]
        total_pnl += pnl
        total_wp += d["wp"]
        total_lp += d["lp"]

    if len(buckets) > 1:
        total_pf = total_wp / total_lp if total_lp > 0 else float("inf")
        total_pf_s = f"{total_pf:>5.2f}" if total_pf != float("inf") else "  inf"
        print("-" * 60)
        print(f"{'TOT':5s} {total_n:>4d} {' ':>6s} ${total_pnl:>+8.2f} {' ':>8s} {' ':>8s} {total_pf_s}")

    # Side breakdown
    print()
    print("By side:")
    for k in sorted(buckets):
        b = buckets[k]
        for side, s in sorted(b["sides"].items()):
            wr = s["w"] / s["n"] * 100 if s["n"] else 0
            print(f"  {k:4s} {side:5s}  n={s['n']:3d}  win={wr:5.1f}%  pnl=${s['pnl']:+.2f}")

    # Equity curve waypoints
    print()
    print("Recent equity curve (last 10 resolutions):")
    recent = []
    with open(args.log) as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("type") == "resolution" and "bankroll_after" in r:
                    if cutoff:
                        ts = datetime.fromisoformat(r["ts"].replace("Z", "+00:00"))
                        if ts < cutoff:
                            continue
                    recent.append(r)
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
    for r in recent[-10:]:
        m = bucket_for(r.get("market_slug", ""))
        side = r.get("side", "?")
        win = "W" if r.get("side") == r.get("outcome") else "L"
        print(f"  {r['ts'][:19]}  {m:4s} {side:5s} {win}  pnl=${r.get('pnl',0):+6.2f}  bk=${r.get('bankroll_after',0):7.2f}")


if __name__ == "__main__":
    main()
