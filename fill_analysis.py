#!/usr/bin/env python3
"""
Fill Quality & PnL Analysis

Reads live_trades_*.jsonl and dashboard_paper_trades.jsonl to report:
  - Win rate overall, per market, and per timeframe
  - PnL distribution and Sharpe ratio
  - Flat rate (windows where no trade was taken)
  - Brier score and calibration (requires p_at_entry field)
  - PnL by edge bucket: 0-5%, 5-10%, 10-15%, 15%+
  - Stopping criterion: if Sharpe < 0.5, flag to stop

Usage:
    python fill_analysis.py
    python fill_analysis.py --min-windows 20
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PAPER_TRADES_LOG = Path("dashboard_paper_trades.jsonl")
LIVE_TRADES_GLOB = "live_trades_*.jsonl"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def load_live_trades() -> list[dict]:
    records = []
    for path in sorted(Path(".").glob(LIVE_TRADES_GLOB)):
        records.extend(load_jsonl(path))
    return records


# ── Analysis helpers ──────────────────────────────────────────────────────────

def _sharpe(pnls: list[float]) -> float:
    if len(pnls) < 2:
        return float("nan")
    mean = sum(pnls) / len(pnls)
    std  = math.sqrt(sum((x - mean) ** 2 for x in pnls) / (len(pnls) - 1))
    return (mean / std) * math.sqrt(len(pnls)) if std > 0 else float("nan")


def _pct(num: int, denom: int) -> str:
    return f"{num/denom*100:.1f}%" if denom > 0 else "n/a"


def _brier_score(trades: list[dict]) -> float | None:
    """Mean squared error between p_at_entry and actual outcome (1=won, 0=lost)."""
    valid = [
        (t["p_at_entry"], 1.0 if t.get("won") else 0.0)
        for t in trades
        if t.get("p_at_entry") is not None and t.get("won") is not None
    ]
    if not valid:
        return None
    return sum((p - o) ** 2 for p, o in valid) / len(valid)


def _edge_buckets(trades: list[dict]) -> None:
    """Print PnL breakdown by edge-at-entry bucket."""
    buckets: dict[str, list[dict]] = {
        "0–5%":   [],
        "5–10%":  [],
        "10–15%": [],
        "15–20%": [],
        "≥20%":   [],
    }
    no_edge = []
    for t in trades:
        e = t.get("edge_at_entry")
        if e is None:
            no_edge.append(t)
            continue
        if e < 0.05:
            buckets["0–5%"].append(t)
        elif e < 0.10:
            buckets["5–10%"].append(t)
        elif e < 0.15:
            buckets["10–15%"].append(t)
        elif e < 0.20:
            buckets["15–20%"].append(t)
        else:
            buckets["≥20%"].append(t)

    print(f"\n  PnL by edge-at-entry bucket:")
    print(f"    {'bucket':10s}  {'n':>5}  {'win%':>6}  {'net_pnl':>9}  {'avg_pnl':>8}  {'sharpe':>7}")
    print(f"    {'-'*55}")
    for label, group in buckets.items():
        if not group:
            continue
        wins    = sum(1 for t in group if t.get("won"))
        pnls    = [t.get("pnl_usd", 0.0) for t in group]
        net_pnl = sum(pnls)
        avg_pnl = net_pnl / len(pnls)
        s       = _sharpe(pnls)
        print(f"    {label:10s}  {len(group):>5}  {wins/len(group)*100:>5.1f}%"
              f"  ${net_pnl:>8.2f}  ${avg_pnl:>7.2f}  {s:>7.2f}")
    if no_edge:
        print(f"    {'(no edge)':10s}  {len(no_edge):>5}  (older records without edge_at_entry)")


# ── Paper trades analysis (dashboard_paper_trades.jsonl) ─────────────────────

def analyze_paper_trades(min_windows: int):
    records = load_jsonl(PAPER_TRADES_LOG)
    if not records:
        print(f"[paper trades] No data in {PAPER_TRADES_LOG}.")
        print("  Run dashboard.py — data is written on each window resolution.")
        return

    traded   = [r for r in records if r.get("traded")]
    untraded = [r for r in records if not r.get("traded")]

    print(f"\n{'='*65}")
    print("PAPER TRADE LOG  (dashboard_paper_trades.jsonl)")
    print(f"{'='*65}")
    print(f"  Total resolved windows: {len(records)}")
    print(f"  Traded:    {len(traded)}  ({_pct(len(traded), len(records))})")
    print(f"  No trade:  {len(untraded)}  ({_pct(len(untraded), len(records))})")

    if not traded:
        print("\n  No trades to analyze yet.")
        _stopping_criterion([], min_windows, label="paper")
        return

    won   = [r for r in traded if r.get("won")]
    pnls  = [r["pnl_usd"] for r in traded if "pnl_usd" in r]
    total_pnl = sum(pnls)
    win_rate  = len(won) / len(traded)
    sharpe    = _sharpe(pnls)

    print(f"\n  Win rate:  {win_rate:.3f}  ({len(won)}/{len(traded)})")
    print(f"  Net PnL:   ${total_pnl:.2f}")
    print(f"  Sharpe:    {sharpe:.2f}  (per-trade, not annualized)")

    # Brier score (requires newer records with p_at_entry)
    brier = _brier_score(traded)
    if brier is not None:
        # Naive baseline: always predict 0.5 → Brier = 0.25
        naive_brier = 0.25
        skill = 1.0 - brier / naive_brier
        print(f"  Brier:     {brier:.4f}  (naive=0.25, skill_score={skill:+.3f})")
        if skill < 0:
            print("             WARNING: model is WORSE than random (p estimates are off)")
        elif skill < 0.05:
            print("             Model barely beats random — calibration is poor")
    else:
        print("  Brier:     n/a  (need newer records with p_at_entry field)")

    # Per-timeframe breakdown
    timeframes: dict[str, list[dict]] = {}
    for r in traded:
        tf = r.get("timeframe", "unknown")
        timeframes.setdefault(tf, []).append(r)

    if len(timeframes) > 1:
        print(f"\n  Per-timeframe:")
        for tf, trs in sorted(timeframes.items()):
            w = sum(1 for t in trs if t.get("won"))
            p = sum(t.get("pnl_usd", 0.0) for t in trs)
            s = _sharpe([t.get("pnl_usd", 0.0) for t in trs])
            print(f"    {tf:12s}  n={len(trs):4d}  win={w/len(trs):.3f}  pnl=${p:+.2f}  sharpe={s:.2f}")

    # Edge-bucket analysis
    _edge_buckets(traded)

    # No-trade reason breakdown (from signal field if available)
    _stopping_criterion(pnls, min_windows, label="paper")


# ── Live trades analysis (live_trades_*.jsonl) ────────────────────────────────

def analyze_live_trades(min_windows: int):
    records = load_live_trades()
    if not records:
        print(f"\n[live trades] No {LIVE_TRADES_GLOB} files found.")
        print("  Run live_trader.py to generate live trade data.")
        return

    resolutions  = [r for r in records if r.get("type") == "resolution"]
    flat_summs   = [r for r in records if r.get("type") == "flat_summary"]

    print(f"\n{'='*65}")
    print(f"LIVE TRADE LOG  ({LIVE_TRADES_GLOB})")
    print(f"{'='*65}")
    print(f"  Total records:    {len(records)}")
    print(f"  Resolutions:      {len(resolutions)}")
    print(f"  Flat summaries:   {len(flat_summs)}")

    if resolutions:
        won  = [r for r in resolutions if r.get("outcome") == r.get("side")]
        pnls = [float(r.get("pnl", 0.0)) for r in resolutions]
        total_pnl = sum(pnls)
        win_rate  = len(won) / len(resolutions)
        sharpe    = _sharpe(pnls)

        print(f"\n  Win rate:  {win_rate:.3f}  ({len(won)}/{len(resolutions)})")
        print(f"  Net PnL:   ${total_pnl:.2f}")
        print(f"  Avg PnL:   ${total_pnl/len(resolutions):.2f} per trade")
        print(f"  Sharpe:    {sharpe:.2f}")

        # Per-market
        markets: dict[str, list] = {}
        for r in resolutions:
            slug = r.get("market_slug", "unknown")
            markets.setdefault(slug, []).append(r)
        if len(markets) > 1:
            print(f"\n  Per-market breakdown:")
            for slug, trs in sorted(markets.items()):
                w = sum(1 for t in trs if t.get("outcome") == t.get("side"))
                p = sum(float(t.get("pnl", 0)) for t in trs)
                print(f"    {slug:40s}  n={len(trs):4d}  win={w/len(trs):.3f}  pnl=${p:.2f}")

        _stopping_criterion(pnls, min_windows, label="live")

    # Flat reason analysis
    if flat_summs:
        print(f"\n  Flat reason breakdown (why signals were flat):")
        combined: dict[str, int] = {}
        for s in flat_summs:
            for reason, count in (s.get("reasons") or {}).items():
                combined[reason] = combined.get(reason, 0) + count
        total_flats = sum(combined.values())
        for reason, count in sorted(combined.items(), key=lambda x: -x[1]):
            print(f"    {reason:40s}  {count:6d}  ({_pct(count, total_flats)})")


# ── Stopping criterion ────────────────────────────────────────────────────────

def _stopping_criterion(pnls: list[float], min_windows: int, label: str):
    print(f"\n  {'—'*55}")
    print(f"  STOPPING CRITERION ({label})")
    n = len(pnls)

    if n < min_windows:
        print(f"  Status: COLLECTING DATA — only {n}/{min_windows} windows recorded.")
        print(f"          Run longer before evaluating edge.")
        return

    sharpe = _sharpe(pnls)
    if math.isnan(sharpe):
        print("  Status: INSUFFICIENT DATA for Sharpe calculation.")
        return

    mean_pnl = sum(pnls) / n
    total    = sum(pnls)

    print(f"  Windows:  {n}")
    print(f"  Sharpe:   {sharpe:.2f}  (per-trade)")
    print(f"  Avg PnL:  ${mean_pnl:.2f}/window")
    print(f"  Total:    ${total:.2f}")

    if sharpe >= 1.5:
        print("  STATUS: STRONG EDGE (Sharpe ≥ 1.5) — consider increasing size.")
    elif sharpe >= 1.0:
        print("  STATUS: EDGE CONFIRMED (Sharpe ≥ 1.0) — continue running.")
    elif sharpe >= 0.5:
        print("  STATUS: MARGINAL (0.5 ≤ Sharpe < 1.0) — collect more data.")
    else:
        print("  STATUS: STOP — Sharpe < 0.5. Edge not confirmed after sufficient data.")
        print("          The diffusion signal alone is likely not beating vig.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-windows", type=int, default=30,
                   help="Minimum windows before evaluating stopping criterion (default 30)")
    args = p.parse_args()

    analyze_paper_trades(args.min_windows)
    analyze_live_trades(args.min_windows)


if __name__ == "__main__":
    main()
