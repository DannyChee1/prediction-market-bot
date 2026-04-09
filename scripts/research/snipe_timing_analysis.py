#!/usr/bin/env python3
"""
Snipe Timing Analysis: analyze Phase 1 observer data for timing gaps,
liquidity profiles, and fill probability.

Reads JSONL files from data/snipe_research/observer_*.jsonl and produces
summary statistics about the resolution timing pipeline and order book
liquidity at 99c.

Usage:
    python scripts/research/snipe_timing_analysis.py              # analyze all data
    python scripts/research/snipe_timing_analysis.py --latest 10  # last 10 files
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

DATA_DIR = _PROJECT_ROOT / "data" / "snipe_research"


def load_observer_data(limit: int = 0) -> list[dict]:
    """Load all observer JSONL files. Returns list of window dicts.

    Each window dict contains:
      slug, pre_end_rows, post_end_rows, gamma_transitions
    """
    files = sorted(DATA_DIR.glob("observer_*.jsonl"))
    if not files:
        print(f"No observer data found in {DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    if limit > 0:
        files = files[-limit:]

    windows = []
    for f in files:
        slug = f.stem.replace("observer_", "")
        pre_rows = []
        post_rows = []

        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("phase") == "pre_end":
                    pre_rows.append(row)
                elif row.get("phase") == "post_end":
                    post_rows.append(row)

        if not pre_rows and not post_rows:
            continue

        # Extract gamma transition timeline from post_end rows
        gamma_transitions = _extract_gamma_transitions(post_rows)

        windows.append({
            "slug": slug,
            "file": str(f),
            "pre_end_rows": pre_rows,
            "post_end_rows": post_rows,
            "gamma": gamma_transitions,
        })

    return windows


def _extract_gamma_transitions(post_rows: list[dict]) -> dict:
    """Extract timing of gamma state transitions from post_end rows."""
    result = {
        "first_closed_s": None,
        "first_uma_resolved_s": None,
        "first_settled_s": None,
    }

    for row in post_rows:
        t = row.get("time_to_end_s", 0)

        if row.get("gamma_closed") and result["first_closed_s"] is None:
            result["first_closed_s"] = t

        if row.get("gamma_uma_status") == "resolved" and result["first_uma_resolved_s"] is None:
            result["first_uma_resolved_s"] = t

        prices = row.get("gamma_outcome_prices")
        if prices and result["first_settled_s"] is None:
            try:
                pf = sorted([float(p) for p in prices])
                if pf == [0.0, 1.0]:
                    result["first_settled_s"] = t
            except (ValueError, TypeError):
                pass

    return result


def analyze_timing(windows: list[dict]) -> None:
    """Analyze and print resolution timing statistics."""
    closed_times = []
    uma_times = []
    settled_times = []

    for w in windows:
        g = w["gamma"]
        if g["first_closed_s"] is not None:
            closed_times.append(g["first_closed_s"])
        if g["first_uma_resolved_s"] is not None:
            uma_times.append(g["first_uma_resolved_s"])
        if g["first_settled_s"] is not None:
            settled_times.append(g["first_settled_s"])

    print("\n" + "=" * 60)
    print("Resolution Timing Analysis")
    print("=" * 60)
    print(f"Windows with data: {len(windows)}")

    for label, times in [
        ("window_end -> closed=True", closed_times),
        ("window_end -> uma=resolved", uma_times),
        ("window_end -> prices=[0,1]", settled_times),
    ]:
        if not times:
            print(f"\n  {label}: no data")
            continue
        arr = np.array(times)
        print(f"\n  {label} (n={len(arr)}):")
        print(f"    min={arr.min():.1f}s  p25={np.percentile(arr, 25):.1f}s  "
              f"median={np.median(arr):.1f}s  p75={np.percentile(arr, 75):.1f}s  "
              f"max={arr.max():.1f}s")

    # Gap analysis: closed -> uma_resolved
    if closed_times and uma_times:
        gaps = []
        for w in windows:
            g = w["gamma"]
            if g["first_closed_s"] is not None and g["first_uma_resolved_s"] is not None:
                gap = g["first_uma_resolved_s"] - g["first_closed_s"]
                gaps.append(gap)
        if gaps:
            arr = np.array(gaps)
            print(f"\n  Gap: closed -> uma_resolved (n={len(arr)}):")
            print(f"    min={arr.min():.1f}s  median={np.median(arr):.1f}s  "
                  f"max={arr.max():.1f}s")


def analyze_liquidity(windows: list[dict]) -> None:
    """Analyze order book liquidity at 99c around resolution."""
    print("\n" + "=" * 60)
    print("Liquidity Analysis (winning side at 99c)")
    print("=" * 60)

    # Collect ask sizes at 99c for the predicted winner across time buckets
    time_buckets = [
        ("Last 60s (pre-end)", -60, 0),
        ("Last 30s (pre-end)", -30, 0),
        ("Last 10s (pre-end)", -10, 0),
        ("Last 5s (pre-end)", -5, 0),
        ("Post-end 0-30s", 0, 30),
        ("Post-end 30-60s", 30, 60),
        ("Post-end 60-120s", 60, 120),
    ]

    for bucket_name, t_lo, t_hi in time_buckets:
        sizes = []
        n_windows = 0
        n_with_liq = 0

        for w in windows:
            rows = w["pre_end_rows"] if t_hi <= 0 else w["post_end_rows"]
            bucket_rows = [r for r in rows if t_lo <= r.get("time_to_end_s", 999) <= t_hi]
            if not bucket_rows:
                continue

            n_windows += 1
            max_ask_99 = max(r.get("winner_ask_99", 0) for r in bucket_rows)
            if max_ask_99 > 0:
                n_with_liq += 1
                sizes.append(max_ask_99)

        if n_windows == 0:
            print(f"\n  {bucket_name}: no data")
            continue

        pct = n_with_liq / n_windows * 100
        med = np.median(sizes) if sizes else 0
        print(f"\n  {bucket_name} (n={n_windows}):")
        print(f"    Windows with ask at 99c: {n_with_liq}/{n_windows} ({pct:.1f}%)")
        if sizes:
            print(f"    Median ask size: {med:.0f} shares")
            print(f"    Mean ask size: {np.mean(sizes):.0f} shares")


def analyze_competition(windows: list[dict]) -> None:
    """Analyze how fast 99c liquidity disappears (competition signal)."""
    print("\n" + "=" * 60)
    print("Competition Analysis")
    print("=" * 60)

    # For each window, find when the 99c ask first appears and when it disappears
    for w in windows[:5]:  # Show first 5 as examples
        all_rows = w["pre_end_rows"] + w["post_end_rows"]
        times_with_liq = [r["time_to_end_s"] for r in all_rows
                          if r.get("winner_ask_99", 0) > 0]
        if times_with_liq:
            print(f"\n  {w['slug']}:")
            print(f"    99c liquidity present at times: "
                  f"{min(times_with_liq):.1f}s to {max(times_with_liq):.1f}s "
                  f"({len(times_with_liq)} snapshots)")


def analyze_book_evolution(windows: list[dict]) -> None:
    """Show how the book evolves from pre-end through post-end."""
    print("\n" + "=" * 60)
    print("Book Evolution (sample windows)")
    print("=" * 60)

    for w in windows[:3]:
        print(f"\n  {w['slug']}:")
        all_rows = w["pre_end_rows"] + w["post_end_rows"]
        # Show snapshots at key moments
        key_times = [-30, -10, -5, -1, 1, 5, 10, 30, 60]
        for target_t in key_times:
            closest = min(all_rows, key=lambda r: abs(r.get("time_to_end_s", 999) - target_t),
                          default=None)
            if closest and abs(closest["time_to_end_s"] - target_t) < 3:
                t = closest["time_to_end_s"]
                w99 = closest.get("winner_ask_99", 0)
                w98 = closest.get("winner_ask_98", 0)
                winner = closest.get("predicted_winner", "?")
                up_ba = closest.get("up_best_ask")
                down_ba = closest.get("down_best_ask")
                gamma_c = closest.get("gamma_closed")
                gamma_u = closest.get("gamma_uma_status")
                print(f"    t={t:+7.1f}s: winner={winner} "
                      f"ask99={w99:>5.0f} ask98={w98:>5.0f} "
                      f"UP_ask={up_ba}  DOWN_ask={down_ba}  "
                      f"closed={gamma_c}  uma={gamma_u}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze snipe observer data for timing and liquidity",
    )
    parser.add_argument("--latest", type=int, default=0,
                        help="Only analyze the N most recent files (0=all)")
    args = parser.parse_args()

    windows = load_observer_data(limit=args.latest)
    print(f"Loaded {len(windows)} windows from {DATA_DIR}")

    analyze_timing(windows)
    analyze_liquidity(windows)
    analyze_competition(windows)
    analyze_book_evolution(windows)

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
