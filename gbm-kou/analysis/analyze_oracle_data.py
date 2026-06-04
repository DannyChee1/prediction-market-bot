#!/usr/bin/env python3
"""
Oracle Data Analysis — Chainlink update frequency, sigma, z-scores,
staleness, and book depth for BTC vs ETH 15m markets.
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Helpers ──────────────────────────────────────────────────────────────────

def get_price_col(df):
    if "chainlink_price" in df.columns:
        return "chainlink_price"
    if "chainlink_btc" in df.columns:
        return "chainlink_btc"
    if "chainlink_eth" in df.columns:
        return "chainlink_eth"
    raise KeyError("No chainlink price column found")


def load_windows(data_dir, n=10):
    """Load n complete windows from data_dir, returning list of DataFrames."""
    files = sorted(data_dir.glob("*.parquet"))
    windows = []
    for f in files:
        df = pd.read_parquet(f)
        if df.empty:
            continue
        # Normalize column name
        if "chainlink_btc" in df.columns and "chainlink_price" not in df.columns:
            df.rename(columns={"chainlink_btc": "chainlink_price"}, inplace=True)
        if "chainlink_eth" in df.columns and "chainlink_price" not in df.columns:
            df.rename(columns={"chainlink_eth": "chainlink_price"}, inplace=True)
        # Completeness check
        if "time_remaining_s" in df.columns:
            if df["time_remaining_s"].iloc[-1] > 10:
                continue  # incomplete window
        # Need enough data (at least 200 rows for 90s lookback + samples)
        if len(df) < 200:
            continue
        windows.append(df)
        if len(windows) >= n:
            break
    return windows


def dedup_prices(prices, timestamps):
    """Deduplicate consecutive identical prices. Returns list of (idx, price, ts_ms)."""
    changes = []
    for i, p in enumerate(prices):
        ts = timestamps[i]
        if p > 0 and (not changes or p != changes[-1][1]):
            changes.append((i, p, ts))
    return changes


def compute_sigma_deduped(prices, timestamps):
    """Per-second sigma using deduped log-return / sqrt(dt) method from backtest.py."""
    changes = dedup_prices(prices, timestamps)
    if len(changes) < 3:
        return 0.0
    log_rets = []
    for j in range(1, len(changes)):
        dt = (changes[j][2] - changes[j - 1][2]) / 1000.0  # ms -> seconds
        if dt > 0:
            lr = math.log(changes[j][1] / changes[j - 1][1])
            log_rets.append(lr / math.sqrt(dt))
    if len(log_rets) < 2:
        return 0.0
    return float(np.std(log_rets, ddof=1))


def norm_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


# ── ANALYSIS 1: Chainlink update frequency ──────────────────────────────────

def analyze_update_frequency(windows, label, lookback_s=90):
    """Count unique Chainlink price changes per lookback window."""
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 1: Chainlink Update Frequency — {label}")
    print(f"  Lookback: {lookback_s}s")
    print(f"{'='*70}")

    all_counts = []
    for w_idx, df in enumerate(windows):
        prices = df["chainlink_price"].tolist()
        ts_list = df["ts_ms"].tolist()

        # Sample every 30 rows after warmup
        for idx in range(lookback_s, len(df), 30):
            lo = max(0, idx - lookback_s)
            price_slice = prices[lo:idx + 1]
            ts_slice = ts_list[lo:idx + 1]

            # Check for gaps > 5s
            has_gap = False
            for k in range(1, len(ts_slice)):
                if ts_slice[k] - ts_slice[k - 1] > 5000:
                    has_gap = True
                    break
            if has_gap:
                continue

            changes = dedup_prices(price_slice, ts_slice)
            all_counts.append(len(changes))

    if not all_counts:
        print("  No valid samples found.")
        return all_counts

    arr = np.array(all_counts)
    print(f"  Samples:  {len(arr)}")
    print(f"  Min:      {arr.min()}")
    print(f"  Median:   {np.median(arr):.1f}")
    print(f"  Mean:     {arr.mean():.1f}")
    print(f"  Max:      {arr.max()}")
    print(f"  Std:      {arr.std():.1f}")

    # Distribution histogram
    bins = [0, 5, 10, 15, 20, 30, 50, 100, 200]
    print(f"\n  Distribution:")
    for i in range(len(bins) - 1):
        count = np.sum((arr >= bins[i]) & (arr < bins[i + 1]))
        pct = count / len(arr) * 100
        print(f"    [{bins[i]:>3}, {bins[i+1]:>3}): {count:>5} ({pct:>5.1f}%)")
    count = np.sum(arr >= bins[-1])
    pct = count / len(arr) * 100
    print(f"    [{bins[-1]:>3},  + ): {count:>5} ({pct:>5.1f}%)")

    return all_counts


# ── ANALYSIS 2: Raw sigma estimates ─────────────────────────────────────────

def analyze_sigma(windows, label, lookback_s=90, btc_floor=2e-5, eth_floor=1.5e-5):
    """Compute per-second sigma distribution."""
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 2: Raw Sigma Estimates — {label}")
    print(f"  Lookback: {lookback_s}s, BTC floor: {btc_floor:.1e}, ETH floor: {eth_floor:.1e}")
    print(f"{'='*70}")

    all_sigmas = []
    for w_idx, df in enumerate(windows):
        prices = df["chainlink_price"].tolist()
        ts_list = df["ts_ms"].tolist()

        for idx in range(lookback_s, len(df), 30):
            lo = max(0, idx - lookback_s)
            price_slice = prices[lo:idx + 1]
            ts_slice = ts_list[lo:idx + 1]

            has_gap = False
            for k in range(1, len(ts_slice)):
                if ts_slice[k] - ts_slice[k - 1] > 5000:
                    has_gap = True
                    break
            if has_gap:
                continue

            sigma = compute_sigma_deduped(price_slice, ts_slice)
            if sigma > 0:
                all_sigmas.append(sigma)

    if not all_sigmas:
        print("  No valid sigma samples.")
        return all_sigmas

    arr = np.array(all_sigmas)
    print(f"  Samples:       {len(arr)}")
    print(f"  Min:           {arr.min():.2e}")
    print(f"  P5:            {np.percentile(arr, 5):.2e}")
    print(f"  P10:           {np.percentile(arr, 10):.2e}")
    print(f"  P25:           {np.percentile(arr, 25):.2e}")
    print(f"  Median:        {np.median(arr):.2e}")
    print(f"  Mean:          {arr.mean():.2e}")
    print(f"  P75:           {np.percentile(arr, 75):.2e}")
    print(f"  P90:           {np.percentile(arr, 90):.2e}")
    print(f"  P95:           {np.percentile(arr, 95):.2e}")
    print(f"  Max:           {arr.max():.2e}")

    below_btc = np.sum(arr < btc_floor)
    below_eth = np.sum(arr < eth_floor)
    print(f"\n  Below BTC floor ({btc_floor:.1e}): {below_btc}/{len(arr)} ({below_btc/len(arr)*100:.1f}%)")
    print(f"  Below ETH floor ({eth_floor:.1e}): {below_eth}/{len(arr)} ({below_eth/len(arr)*100:.1f}%)")

    # Zero sigma (too few changes)
    zero_count = sum(1 for s in all_sigmas if s == 0)
    print(f"  Zero sigma:    {zero_count} (excluded from stats above)")

    return all_sigmas


# ── ANALYSIS 3: Z-score distribution ────────────────────────────────────────

def analyze_z_scores(windows, label, lookback_s=90, max_z=1.5, sigma_floor=0.0):
    """Compute z = delta / (sigma * sqrt(tau)) distribution."""
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 3: Z-Score Distribution — {label}")
    print(f"  max_z: {max_z}, sigma_floor: {sigma_floor:.1e}")
    print(f"{'='*70}")

    all_z_raw = []
    all_z_capped = []
    for w_idx, df in enumerate(windows):
        prices = df["chainlink_price"].tolist()
        ts_list = df["ts_ms"].tolist()
        start_px = df["window_start_price"].dropna().iloc[0]
        if pd.isna(start_px) or start_px == 0:
            continue

        for idx in range(lookback_s, len(df), 30):
            lo = max(0, idx - lookback_s)
            price_slice = prices[lo:idx + 1]
            ts_slice = ts_list[lo:idx + 1]

            has_gap = False
            for k in range(1, len(ts_slice)):
                if ts_slice[k] - ts_slice[k - 1] > 5000:
                    has_gap = True
                    break
            if has_gap:
                continue

            row = df.iloc[idx]
            tau = row["time_remaining_s"]
            if tau <= 0:
                continue

            sigma = compute_sigma_deduped(price_slice, ts_slice)
            sigma = max(sigma, sigma_floor)
            if sigma <= 0:
                continue

            delta = (row["chainlink_price"] - start_px) / start_px
            z_raw = delta / (sigma * math.sqrt(tau))
            z_capped = max(-max_z, min(max_z, z_raw))

            all_z_raw.append(z_raw)
            all_z_capped.append(z_capped)

    if not all_z_raw:
        print("  No valid z samples.")
        return

    raw = np.array(all_z_raw)
    capped = np.array(all_z_capped)

    print(f"  Samples:       {len(raw)}")
    print(f"\n  Raw z distribution:")
    print(f"    Min:         {raw.min():.3f}")
    print(f"    P5:          {np.percentile(raw, 5):.3f}")
    print(f"    P25:         {np.percentile(raw, 25):.3f}")
    print(f"    Median:      {np.median(raw):.3f}")
    print(f"    Mean:        {raw.mean():.3f}")
    print(f"    P75:         {np.percentile(raw, 75):.3f}")
    print(f"    P95:         {np.percentile(raw, 95):.3f}")
    print(f"    Max:         {raw.max():.3f}")
    print(f"    Std:         {raw.std():.3f}")

    hit_cap_pos = np.sum(raw > max_z)
    hit_cap_neg = np.sum(raw < -max_z)
    hit_cap = hit_cap_pos + hit_cap_neg
    print(f"\n  Z-cap hits (|z| > {max_z}):")
    print(f"    Positive cap: {hit_cap_pos}/{len(raw)} ({hit_cap_pos/len(raw)*100:.1f}%)")
    print(f"    Negative cap: {hit_cap_neg}/{len(raw)} ({hit_cap_neg/len(raw)*100:.1f}%)")
    print(f"    Total capped: {hit_cap}/{len(raw)} ({hit_cap/len(raw)*100:.1f}%)")

    # Histogram of z_raw
    z_bins = [-10, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 10]
    print(f"\n  Raw z histogram:")
    for i in range(len(z_bins) - 1):
        count = np.sum((raw >= z_bins[i]) & (raw < z_bins[i + 1]))
        pct = count / len(raw) * 100
        bar = "#" * int(pct)
        print(f"    [{z_bins[i]:>5.1f}, {z_bins[i+1]:>5.1f}): {count:>5} ({pct:>5.1f}%) {bar}")

    # p_model distribution from capped z
    p_models = np.array([norm_cdf(z) for z in capped])
    print(f"\n  p_model distribution (from capped z):")
    print(f"    Min:         {p_models.min():.4f}")
    print(f"    P5:          {np.percentile(p_models, 5):.4f}")
    print(f"    P25:         {np.percentile(p_models, 25):.4f}")
    print(f"    Median:      {np.median(p_models):.4f}")
    print(f"    P75:         {np.percentile(p_models, 75):.4f}")
    print(f"    P95:         {np.percentile(p_models, 95):.4f}")
    print(f"    Max:         {p_models.max():.4f}")


# ── ANALYSIS 4: Oracle staleness ────────────────────────────────────────────

def analyze_staleness(windows, label, lookbacks=(20, 90)):
    """Compare number of unique price changes in different lookback windows."""
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 4: Oracle Staleness — {label}")
    print(f"  Comparing lookbacks: {lookbacks}")
    print(f"{'='*70}")

    for lb in lookbacks:
        counts = []
        for df in windows:
            prices = df["chainlink_price"].tolist()
            ts_list = df["ts_ms"].tolist()

            for idx in range(max(lookbacks), len(df), 30):
                lo = max(0, idx - lb)
                price_slice = prices[lo:idx + 1]
                ts_slice = ts_list[lo:idx + 1]

                has_gap = False
                for k in range(1, len(ts_slice)):
                    if ts_slice[k] - ts_slice[k - 1] > 5000:
                        has_gap = True
                        break
                if has_gap:
                    continue

                changes = dedup_prices(price_slice, ts_slice)
                counts.append(len(changes))

        if not counts:
            print(f"\n  Lookback {lb}s: no valid samples")
            continue

        arr = np.array(counts)
        print(f"\n  Lookback {lb}s ({len(arr)} samples):")
        print(f"    Min:         {arr.min()}")
        print(f"    P10:         {np.percentile(arr, 10):.0f}")
        print(f"    P25:         {np.percentile(arr, 25):.0f}")
        print(f"    Median:      {np.median(arr):.0f}")
        print(f"    Mean:        {arr.mean():.1f}")
        print(f"    P75:         {np.percentile(arr, 75):.0f}")
        print(f"    P90:         {np.percentile(arr, 90):.0f}")
        print(f"    Max:         {arr.max()}")

        # Critically low counts (< 3 means sigma = 0)
        below_3 = np.sum(arr < 3)
        below_5 = np.sum(arr < 5)
        print(f"    < 3 changes: {below_3}/{len(arr)} ({below_3/len(arr)*100:.1f}%) -- sigma=0!")
        print(f"    < 5 changes: {below_5}/{len(arr)} ({below_5/len(arr)*100:.1f}%)")

    # Inter-update time analysis
    print(f"\n  Inter-update time analysis (time between consecutive unique prices):")
    all_dts = []
    for df in windows:
        prices = df["chainlink_price"].tolist()
        ts_list = df["ts_ms"].tolist()
        changes = dedup_prices(prices, ts_list)
        for j in range(1, len(changes)):
            dt_s = (changes[j][2] - changes[j - 1][2]) / 1000.0
            all_dts.append(dt_s)

    if all_dts:
        dt_arr = np.array(all_dts)
        print(f"    Samples:     {len(dt_arr)}")
        print(f"    Min:         {dt_arr.min():.2f}s")
        print(f"    P25:         {np.percentile(dt_arr, 25):.2f}s")
        print(f"    Median:      {np.median(dt_arr):.2f}s")
        print(f"    Mean:        {dt_arr.mean():.2f}s")
        print(f"    P75:         {np.percentile(dt_arr, 75):.2f}s")
        print(f"    P90:         {np.percentile(dt_arr, 90):.2f}s")
        print(f"    P95:         {np.percentile(dt_arr, 95):.2f}s")
        print(f"    Max:         {dt_arr.max():.2f}s")
        stale = np.sum(dt_arr > 5)
        print(f"    > 5s stale:  {stale}/{len(dt_arr)} ({stale/len(dt_arr)*100:.1f}%)")


# ── ANALYSIS 5: Spread and book depth ───────────────────────────────────────

def analyze_spread_depth(windows, label):
    """Analyze bid-ask spreads and top-of-book depth."""
    print(f"\n{'='*70}")
    print(f"  ANALYSIS 5: Spread and Book Depth — {label}")
    print(f"{'='*70}")

    spreads_up = []
    spreads_down = []
    depth_bid_up = []
    depth_ask_up = []
    depth_bid_down = []
    depth_ask_down = []
    tob_bid_sz_up = []
    tob_ask_sz_up = []
    tob_bid_sz_down = []
    tob_ask_sz_down = []

    for df in windows:
        for _, row in df.iterrows():
            bb_up = row.get("best_bid_up")
            ba_up = row.get("best_ask_up")
            bb_down = row.get("best_bid_down")
            ba_down = row.get("best_ask_down")

            if pd.notna(bb_up) and pd.notna(ba_up) and ba_up > bb_up:
                spreads_up.append(ba_up - bb_up)
            if pd.notna(bb_down) and pd.notna(ba_down) and ba_down > bb_down:
                spreads_down.append(ba_down - bb_down)

            # Top-of-book sizes
            sz_bid_up = row.get("size_bid_up")
            sz_ask_up = row.get("size_ask_up")
            sz_bid_down = row.get("size_bid_down")
            sz_ask_down = row.get("size_ask_down")

            if pd.notna(sz_bid_up) and sz_bid_up > 0:
                tob_bid_sz_up.append(sz_bid_up)
            if pd.notna(sz_ask_up) and sz_ask_up > 0:
                tob_ask_sz_up.append(sz_ask_up)
            if pd.notna(sz_bid_down) and sz_bid_down > 0:
                tob_bid_sz_down.append(sz_bid_down)
            if pd.notna(sz_ask_down) and sz_ask_down > 0:
                tob_ask_sz_down.append(sz_ask_down)

            # Full depth (top 5)
            bd5_up = row.get("bid_depth5_up")
            ad5_up = row.get("ask_depth5_up")
            bd5_down = row.get("bid_depth5_down")
            ad5_down = row.get("ask_depth5_down")

            if pd.notna(bd5_up):
                depth_bid_up.append(bd5_up)
            if pd.notna(ad5_up):
                depth_ask_up.append(ad5_up)
            if pd.notna(bd5_down):
                depth_bid_down.append(bd5_down)
            if pd.notna(ad5_down):
                depth_ask_down.append(ad5_down)

    def print_stats(name, vals):
        if not vals:
            print(f"  {name}: no data")
            return
        arr = np.array(vals)
        print(f"  {name} ({len(arr)} samples):")
        print(f"    Min:     {arr.min():.4f}")
        print(f"    P10:     {np.percentile(arr, 10):.4f}")
        print(f"    P25:     {np.percentile(arr, 25):.4f}")
        print(f"    Median:  {np.median(arr):.4f}")
        print(f"    Mean:    {arr.mean():.4f}")
        print(f"    P75:     {np.percentile(arr, 75):.4f}")
        print(f"    P90:     {np.percentile(arr, 90):.4f}")
        print(f"    Max:     {arr.max():.4f}")

    print(f"\n  --- Spreads ---")
    print_stats("Up spread", spreads_up)
    print()
    print_stats("Down spread", spreads_down)

    # Spread histogram
    if spreads_up:
        arr = np.array(spreads_up)
        print(f"\n  Up spread histogram:")
        bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 1.0]
        for i in range(len(bins) - 1):
            count = np.sum((arr >= bins[i]) & (arr < bins[i + 1]))
            pct = count / len(arr) * 100
            print(f"    [{bins[i]:.2f}, {bins[i+1]:.2f}): {count:>6} ({pct:>5.1f}%)")

    print(f"\n  --- Top-of-Book Size (shares at best bid/ask) ---")
    print_stats("Bid size Up", tob_bid_sz_up)
    print()
    print_stats("Ask size Up", tob_ask_sz_up)
    print()
    print_stats("Bid size Down", tob_bid_sz_down)
    print()
    print_stats("Ask size Down", tob_ask_sz_down)

    print(f"\n  --- Top-5 Depth (total shares across 5 levels) ---")
    print_stats("Bid depth5 Up", depth_bid_up)
    print()
    print_stats("Ask depth5 Up", depth_ask_up)
    print()
    print_stats("Bid depth5 Down", depth_bid_down)
    print()
    print_stats("Ask depth5 Down", depth_ask_down)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    N_WINDOWS = 10

    print(f"Loading {N_WINDOWS} BTC 15m windows...")
    btc_windows = load_windows(DATA_DIR / "btc_15m", n=N_WINDOWS)
    print(f"  Loaded {len(btc_windows)} BTC windows")

    print(f"Loading {N_WINDOWS} ETH 15m windows...")
    eth_windows = load_windows(DATA_DIR / "eth_15m", n=N_WINDOWS)
    print(f"  Loaded {len(eth_windows)} ETH windows")

    # ── Analysis 1: Update frequency ──
    btc_counts = analyze_update_frequency(btc_windows, "BTC 15m", lookback_s=90)
    eth_counts = analyze_update_frequency(eth_windows, "ETH 15m", lookback_s=90)

    # ── Analysis 2: Raw sigma ──
    btc_sigmas = analyze_sigma(btc_windows, "BTC 15m", lookback_s=90)
    eth_sigmas = analyze_sigma(eth_windows, "ETH 15m", lookback_s=90)

    # ── Analysis 3: Z-scores (without floor, to see raw behavior) ──
    print("\n  --- Z-scores WITHOUT sigma floor (raw behavior) ---")
    analyze_z_scores(btc_windows, "BTC 15m (no floor)", lookback_s=90, sigma_floor=0.0)
    analyze_z_scores(eth_windows, "ETH 15m (no floor)", lookback_s=90, sigma_floor=0.0)

    # Z-scores with floor (as used in production)
    print("\n  --- Z-scores WITH sigma floor (production) ---")
    analyze_z_scores(btc_windows, "BTC 15m (floor=2e-5)", lookback_s=90, sigma_floor=2e-5)
    analyze_z_scores(eth_windows, "ETH 15m (floor=1.5e-5)", lookback_s=90, sigma_floor=1.5e-5)

    # ── Analysis 4: Oracle staleness ──
    analyze_staleness(btc_windows, "BTC 15m", lookbacks=(20, 90))
    analyze_staleness(eth_windows, "ETH 15m", lookbacks=(20, 90))

    # ── Analysis 5: Spread and book depth ──
    analyze_spread_depth(btc_windows, "BTC 15m")
    analyze_spread_depth(eth_windows, "ETH 15m")


if __name__ == "__main__":
    main()
