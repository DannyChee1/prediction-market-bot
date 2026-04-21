#!/usr/bin/env python3
"""Per-market W/L/PnL/ROI breakdown from the bot's live_trades_*.jsonl.

Usage:
    ./analyze_wl.py                         # auto-discover log file
    ./analyze_wl.py --log /path/to/log      # explicit path
    ./analyze_wl.py --log a.jsonl b.jsonl   # multiple files (merged)

Output columns:
    W / L     wins / losses by market
    WR        win rate
    PnL       realized P&L (payout - cost, fees NOT included)
    ROI       PnL / total cost spent
    avg       average trade PnL
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


def classify_slug(slug: str) -> str:
    if "updown-15m" in slug and "btc" in slug:
        return "BTC 15m"
    if "updown-5m" in slug and "btc" in slug:
        return "BTC 5m"
    if "updown-15m" in slug and "eth" in slug:
        return "ETH 15m"
    if "updown-5m" in slug and "eth" in slug:
        return "ETH 5m"
    return "other"


def find_default_logs() -> list[Path]:
    """Scan reasonable directories for ALL live_trades*.jsonl files and return
    every match (sorted by mtime). Returning multiple files lets us merge
    history across log-name changes (e.g. `live_trades_btc_arb.jsonl` written
    by the old BTC-only build + `live_trades_arb.jsonl` written by the
    multi-asset build after a version upgrade)."""
    search_dirs = [
        Path.cwd(),
        Path.home(),
        Path.home() / "polybot" / "rust",
        Path.home() / "polybot",
    ]
    seen: dict[Path, float] = {}
    for d in search_dirs:
        if not d.is_dir():
            continue
        for match in d.glob("live_trades*.jsonl"):
            try:
                resolved = match.resolve()
            except OSError:
                continue
            if resolved not in seen:
                seen[resolved] = match.stat().st_mtime
    # Oldest first so time-ordered tallies are stable
    return [p for p, _ in sorted(seen.items(), key=lambda kv: kv[1])]


def parse(paths: list[Path]):
    stats = defaultdict(lambda: {
        "w": 0, "l": 0, "pnl": 0.0, "cost": 0.0, "n": 0,
    })
    total_rows = 0
    resolve_rows = 0

    for path in paths:
        with open(path) as f:
            for line in f:
                total_rows += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("event") != "RESOLVE":
                    continue
                resolve_rows += 1

                slug = r.get("slug", "")
                key = classify_slug(slug)

                pnl = float(r.get("trade_pnl", 0.0))
                cost = float(r.get("cost", 0.0))
                # Ground truth for win = comparing side to winner. `trade_pnl > 0`
                # is equivalent (shares * (1 - avg) > 0 iff won) and simpler here.
                if pnl > 0.0:
                    stats[key]["w"] += 1
                else:
                    stats[key]["l"] += 1
                stats[key]["pnl"] += pnl
                stats[key]["cost"] += cost
                stats[key]["n"] += 1

    return stats, total_rows, resolve_rows


def fmt_row(name: str, s: dict) -> str:
    n = s["w"] + s["l"]
    wr = (s["w"] / n * 100.0) if n else 0.0
    roi = (s["pnl"] / s["cost"] * 100.0) if s["cost"] > 0 else 0.0
    avg = (s["pnl"] / n) if n else 0.0
    return (
        f"{name:<10} {s['w']:>4} {s['l']:>4} {wr:>5.1f}% "
        f"{s['pnl']:>+8.2f} {roi:>+7.2f}% {avg:>+6.3f}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", nargs="+", type=Path, help="Path(s) to live_trades_*.jsonl")
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

    stats, total_rows, resolve_rows = parse(paths)

    paths_str = ", ".join(str(p) for p in paths)
    print(f"log: {paths_str}")
    print(f"rows: {total_rows} total, {resolve_rows} RESOLVE")
    print()
    print(f"{'market':<10} {'W':>4} {'L':>4} {'WR':>6} {'PnL':>8} {'ROI':>8} {'avg':>6}")
    print("-" * 52)

    order = ["BTC 15m", "BTC 5m", "ETH 15m", "ETH 5m", "other"]
    for k in order:
        if k in stats:
            print(fmt_row(k, stats[k]))

    # Totals
    total = {"w": 0, "l": 0, "pnl": 0.0, "cost": 0.0, "n": 0}
    for s in stats.values():
        for k in total:
            total[k] += s[k]
    print("-" * 52)
    print(fmt_row("TOTAL", total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
