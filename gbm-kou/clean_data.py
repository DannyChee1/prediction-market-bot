#!/usr/bin/env python3
"""
Audit and optionally delete incomplete parquet recording windows.

A window is considered incomplete if:
  1. Recording started more than MAX_START_GAP_S seconds late
     (time_remaining_s at first row < window_duration - MAX_START_GAP_S)
  2. Recording ended before the window closed
     (time_remaining_s at last row > MIN_FINAL_REMAINING_S)

Usage:
    python3 clean_data.py           # dry run: report only
    python3 clean_data.py --delete  # delete incomplete files
"""

import argparse
import pandas as pd
from pathlib import Path

MIN_FINAL_REMAINING_S = 5.0
MAX_START_GAP_S = 30.0

# Expected window duration per subdirectory name
WINDOW_DURATION = {
    "btc_15m": 900, "eth_15m": 900, "sol_15m": 900, "xrp_15m": 900,
    "btc_5m":  300, "eth_5m":  300, "sol_5m":  300, "xrp_5m":  300,
}


def check_file(f: Path, window_dur_s: float) -> tuple[bool, str]:
    """Return (is_complete, reason). reason is empty string if complete."""
    try:
        df = pd.read_parquet(f)
    except Exception as e:
        return False, f"unreadable: {e}"

    if df.empty:
        return False, "empty file"

    first_remaining = float(df["time_remaining_s"].iloc[0])
    last_remaining  = float(df["time_remaining_s"].iloc[-1])

    # Check started too late
    min_start = window_dur_s - MAX_START_GAP_S
    if first_remaining < min_start:
        gap = window_dur_s - first_remaining
        return False, f"started {gap:.0f}s late (first remaining={first_remaining:.1f}s)"

    # Check ended too early
    if last_remaining > MIN_FINAL_REMAINING_S:
        return False, f"ended early (last remaining={last_remaining:.1f}s)"

    return True, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true",
                        help="Delete incomplete files (default: dry run)")
    parser.add_argument("--data-dir", default="data",
                        help="Root data directory (default: data)")
    args = parser.parse_args()

    base = Path(args.data_dir)
    if not base.exists():
        print(f"Data directory not found: {base}")
        return

    total = complete = incomplete = unreadable = 0
    incomplete_files: list[Path] = []

    for subdir, window_dur in sorted(WINDOW_DURATION.items()):
        d = base / subdir
        if not d.exists():
            continue

        files = sorted(d.glob("*.parquet"))
        sub_total = sub_ok = sub_bad = 0

        for f in files:
            total += 1
            sub_total += 1
            ok, reason = check_file(f, window_dur)
            if ok:
                complete += 1
                sub_ok += 1
            else:
                incomplete += 1
                sub_bad += 1
                incomplete_files.append(f)
                if "unreadable" in reason:
                    unreadable += 1

        print(f"{subdir:12s}  {sub_total:4d} total  {sub_ok:4d} complete  {sub_bad:4d} incomplete")

    print()
    print(f"TOTAL: {total} files — {complete} complete, {incomplete} incomplete"
          + (f" ({unreadable} unreadable)" if unreadable else ""))

    if not incomplete_files:
        print("Nothing to remove.")
        return

    if not args.delete:
        print(f"\nDry run — pass --delete to remove {incomplete} incomplete files.")
        print("\nFirst 20 incomplete files:")
        for f in incomplete_files[:20]:
            _, reason = check_file(f, WINDOW_DURATION.get(f.parent.name, 900))
            print(f"  {f.parent.name}/{f.name}  [{reason}]")
    else:
        print(f"\nDeleting {incomplete} incomplete files...")
        for f in incomplete_files:
            f.unlink()
            print(f"  deleted {f.parent.name}/{f.name}")
        print("Done.")


if __name__ == "__main__":
    main()
