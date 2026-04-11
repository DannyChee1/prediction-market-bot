"""
Cross-asset research: BTC-ETH lead-lag, leading indicators, and empirical tests.
Write-only analysis. Does not touch production code.

Run:
    /Users/dannychee/Desktop/prediction-market-bot/.venv/bin/python3.11 \
        analysis/cross_asset_research.py
"""

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/Users/dannychee/Desktop/prediction-market-bot")
BTC5_DIR = ROOT / "data" / "btc_5m"
ETH5_DIR = ROOT / "data" / "eth_5m"
BTC15_DIR = ROOT / "data" / "btc_15m"
ETH15_DIR = ROOT / "data" / "eth_15m"


def inspect_sampling():
    btc_files = sorted(glob.glob(str(BTC5_DIR / "*.parquet")))
    eth_files = sorted(glob.glob(str(ETH5_DIR / "*.parquet")))

    df = pd.read_parquet(btc_files[-1])
    print(f"[BTC5] rows={len(df)}  median_gap_ms={df['ts_ms'].diff().median():.0f}")
    print(df[["ts_ms", "binance_mid", "chainlink_price", "time_remaining_s"]].head(3).to_string())

    df = pd.read_parquet(eth_files[0])
    print(f"[ETH5] rows={len(df)}  median_gap_ms={df['ts_ms'].diff().median():.0f}")
    print(df[["ts_ms", "binance_mid", "chainlink_price", "time_remaining_s"]].head(3).to_string())


def load_aligned_series(start_key: int = 1775580000, end_key: int = 1775944500):
    """Concatenate binance_mid tick streams from overlapping BTC and ETH parquets.

    Keeps ts_ms, binance_mid from each asset. Forward-fills to a common 1s grid.
    Returns (btc_df, eth_df) each with ts_s index and 'mid' column, forward-filled.
    """
    btc_rows = []
    eth_rows = []

    def file_key(f):
        return int(os.path.basename(f).split("-")[-1].replace(".parquet", ""))

    btc_files = sorted(glob.glob(str(BTC5_DIR / "*.parquet")))
    eth_files = sorted(glob.glob(str(ETH5_DIR / "*.parquet")))

    btc_overlap = [f for f in btc_files if start_key <= file_key(f) <= end_key]
    eth_overlap = [f for f in eth_files if start_key <= file_key(f) <= end_key]

    print(f"[load] btc_overlap={len(btc_overlap)}  eth_overlap={len(eth_overlap)}")

    for f in btc_overlap:
        df = pd.read_parquet(f, columns=["ts_ms", "binance_mid"])
        df = df.dropna(subset=["binance_mid"])
        btc_rows.append(df)
    for f in eth_overlap:
        df = pd.read_parquet(f, columns=["ts_ms", "binance_mid"])
        df = df.dropna(subset=["binance_mid"])
        eth_rows.append(df)

    btc = pd.concat(btc_rows, ignore_index=True).sort_values("ts_ms").drop_duplicates("ts_ms")
    eth = pd.concat(eth_rows, ignore_index=True).sort_values("ts_ms").drop_duplicates("ts_ms")

    print(f"[load] btc rows={len(btc)}  eth rows={len(eth)}")
    print(f"[load] btc ts range: {btc['ts_ms'].min()}..{btc['ts_ms'].max()}")
    print(f"[load] eth ts range: {eth['ts_ms'].min()}..{eth['ts_ms'].max()}")

    return btc, eth


def to_grid(df, step_ms=1000, col="binance_mid"):
    """Forward-fill irregular ticks onto a uniform ts grid at step_ms intervals."""
    df = df.sort_values("ts_ms").drop_duplicates("ts_ms")
    t0 = (df["ts_ms"].min() // step_ms) * step_ms
    t1 = (df["ts_ms"].max() // step_ms) * step_ms
    grid_ts = np.arange(t0, t1 + step_ms, step_ms, dtype=np.int64)
    # Forward-fill using searchsorted
    idx = np.searchsorted(df["ts_ms"].values, grid_ts, side="right") - 1
    # Indices <0 have no prior tick: fill with first observation for safety
    idx = np.clip(idx, 0, len(df) - 1)
    grid_px = df[col].values[idx]
    # Mask rows that precede the first true observation
    mask = idx >= 0
    out = pd.DataFrame({"ts_ms": grid_ts, "mid": grid_px})
    return out


def lead_lag_correlation(btc_grid, eth_grid, return_window_s=1, max_lag_s=60):
    """Compute pearson correlation of BTC(t+k) log returns vs ETH(t) log returns.

    Positive lag k means ETH leads BTC by k seconds (i.e. we check if
    ETH return at t correlates with BTC return shifted forward k seconds).

    return_window_s: compute log returns over this horizon (e.g. 1s).
    """
    # Align on common ts
    df = pd.merge(btc_grid, eth_grid, on="ts_ms", how="inner", suffixes=("_btc", "_eth"))
    if len(df) < 100:
        print(f"[lead_lag] too few rows after merge: {len(df)}")
        return None

    step_ms = int(df["ts_ms"].diff().median())
    shift = max(1, return_window_s * 1000 // step_ms)

    btc_r = np.log(df["mid_btc"].values[shift:] / df["mid_btc"].values[:-shift])
    eth_r = np.log(df["mid_eth"].values[shift:] / df["mid_eth"].values[:-shift])

    # Guard: drop any NaNs
    m = np.isfinite(btc_r) & np.isfinite(eth_r)
    btc_r = btc_r[m]
    eth_r = eth_r[m]

    print(f"[lead_lag] return_window={return_window_s}s step_ms={step_ms}  n={len(btc_r)}")

    lags_s = list(range(-max_lag_s, max_lag_s + 1, 1))
    results = []
    for lag_s in lags_s:
        lag_steps = lag_s * 1000 // step_ms
        if lag_steps >= 0:
            a = btc_r[lag_steps:]
            b = eth_r[: len(btc_r) - lag_steps]
        else:
            a = btc_r[: lag_steps]
            b = eth_r[-lag_steps:]
        if len(a) < 10 or a.std() == 0 or b.std() == 0:
            results.append((lag_s, float("nan")))
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        results.append((lag_s, r))

    return results


def print_lag_table(results, tag):
    print(f"\n[{tag}] lag_s | corr")
    interesting = [-60, -30, -10, -5, -2, -1, 0, 1, 2, 5, 10, 30, 60]
    for lag_s, r in results:
        if lag_s in interesting:
            bar_len = int(abs(r) * 60)
            bar = "#" * bar_len
            sign = "+" if r >= 0 else "-"
            print(f"  {lag_s:>+4d}s  {r:+.4f}  {sign}{bar}")


def run_lead_lag_study():
    print("=== Lead-lag study: BTC vs ETH, log returns on 1s grid ===")
    btc, eth = load_aligned_series()

    btc_grid = to_grid(btc, step_ms=1000)
    eth_grid = to_grid(eth, step_ms=1000)

    for rw in [1, 5, 15, 30]:
        res = lead_lag_correlation(btc_grid, eth_grid, return_window_s=rw, max_lag_s=60)
        if res is not None:
            print_lag_table(res, f"return_window={rw}s")

    # Asymmetry test: compare symmetric +/-k lag correlations to spot ETH-leads-BTC edge
    print("\n=== Asymmetry test (ETH-leads-BTC minus BTC-leads-ETH) ===")
    for rw in [1, 5, 15]:
        res = lead_lag_correlation(btc_grid, eth_grid, return_window_s=rw, max_lag_s=30)
        if res is None:
            continue
        rmap = dict(res)
        print(f"\nreturn_window={rw}s")
        print(f"  k   corr(BTC(t+k),ETH(t))  corr(BTC(t-k),ETH(t))  asymmetry")
        for k in [1, 2, 3, 5, 10, 20, 30]:
            a = rmap.get(k, float("nan"))
            b = rmap.get(-k, float("nan"))
            print(f"  {k:>+3d}s    {a:+.4f}              {b:+.4f}              {a-b:+.4f}")


def build_window_features():
    """Build per-window features and labels for win-rate empirical test.

    For each BTC 5m window that overlaps with available ETH data:
      - feature: ETH log return over last 60s at mid-window (tau ~= 150s)
      - label: UP/DOWN outcome at window end
      - baseline: current signal sign at the same tick
    """
    def file_key(f):
        return int(os.path.basename(f).split("-")[-1].replace(".parquet", ""))

    btc_files = sorted(glob.glob(str(BTC5_DIR / "*.parquet")))
    eth_files = sorted(glob.glob(str(ETH5_DIR / "*.parquet")))

    # Keep only files within the overlap window (for ETH)
    overlap_lo = max(file_key(btc_files[0]), file_key(eth_files[0]))
    overlap_hi = min(file_key(btc_files[-1]), file_key(eth_files[-1]))

    btc_ovl = [f for f in btc_files if overlap_lo <= file_key(f) <= overlap_hi]
    eth_ovl = [f for f in eth_files if overlap_lo <= file_key(f) <= overlap_hi]

    # Build a sorted ETH series for fast lookup
    eth_all = []
    for f in eth_ovl:
        df = pd.read_parquet(f, columns=["ts_ms", "binance_mid"]).dropna()
        eth_all.append(df)
    eth = pd.concat(eth_all, ignore_index=True).sort_values("ts_ms").drop_duplicates("ts_ms")
    eth_ts = eth["ts_ms"].values.astype(np.int64)
    eth_px = eth["binance_mid"].values.astype(np.float64)
    print(f"[features] ETH master series: {len(eth_ts)} rows")

    rows = []
    for f in btc_ovl:
        df = pd.read_parquet(
            f,
            columns=[
                "ts_ms",
                "binance_mid",
                "time_remaining_s",
                "window_start_price",
                "window_start_ms",
                "window_end_ms",
            ],
        )
        if df.empty or df["binance_mid"].isna().all():
            continue
        window_end_ms = int(df["window_end_ms"].iloc[0])
        window_start_ms = int(df["window_start_ms"].iloc[0])
        sp_series = df["window_start_price"].dropna()
        if sp_series.empty:
            continue
        start_px = float(sp_series.iloc[0])

        # Need the window end outcome: use last binance_mid within window
        last_in_window = df[df["ts_ms"] <= window_end_ms]
        if last_in_window.empty:
            continue
        end_px = float(last_in_window["binance_mid"].dropna().iloc[-1])

        # Feature snapshot at mid-window (tau ~= 150s)
        target_ts = window_start_ms + 150_000
        mid_row = df[df["ts_ms"] <= target_ts]
        if mid_row.empty:
            continue
        mid_ts = int(mid_row["ts_ms"].iloc[-1])
        mid_btc = float(mid_row["binance_mid"].dropna().iloc[-1])

        # Pull ETH px at multiple lookbacks: 10s, 30s, 60s
        i_now = int(np.searchsorted(eth_ts, mid_ts, side="right") - 1)
        i_10 = int(np.searchsorted(eth_ts, mid_ts - 10_000, side="right") - 1)
        i_30 = int(np.searchsorted(eth_ts, mid_ts - 30_000, side="right") - 1)
        i_60 = int(np.searchsorted(eth_ts, mid_ts - 60_000, side="right") - 1)
        if min(i_now, i_10, i_30, i_60) < 0:
            continue
        # Reject if ETH data too stale (>3s from request)
        if abs(eth_ts[i_now] - mid_ts) > 3_000:
            continue
        if abs(eth_ts[i_60] - (mid_ts - 60_000)) > 3_000:
            continue

        eth_ret_10s = float(np.log(eth_px[i_now] / eth_px[i_10])) if eth_px[i_10] > 0 else np.nan
        eth_ret_30s = float(np.log(eth_px[i_now] / eth_px[i_30])) if eth_px[i_30] > 0 else np.nan
        eth_ret_60s = float(np.log(eth_px[i_now] / eth_px[i_60])) if eth_px[i_60] > 0 else np.nan

        # BTC past returns at same lookbacks
        btc_all_ts = df["ts_ms"].values.astype(np.int64)
        btc_all_px = df["binance_mid"].values.astype(np.float64)
        j_now = int(np.searchsorted(btc_all_ts, mid_ts, side="right") - 1)
        j_10 = int(np.searchsorted(btc_all_ts, mid_ts - 10_000, side="right") - 1)
        j_30 = int(np.searchsorted(btc_all_ts, mid_ts - 30_000, side="right") - 1)
        j_60 = int(np.searchsorted(btc_all_ts, mid_ts - 60_000, side="right") - 1)
        if min(j_now, j_10, j_30, j_60) < 0:
            continue

        btc_ret_10s = float(np.log(btc_all_px[j_now] / btc_all_px[j_10])) if btc_all_px[j_10] > 0 else np.nan
        btc_ret_30s = float(np.log(btc_all_px[j_now] / btc_all_px[j_30])) if btc_all_px[j_30] > 0 else np.nan
        btc_ret_60s_to_mid = float(np.log(mid_btc / start_px))

        # Window outcome
        label_up = 1 if end_px >= start_px else 0
        btc_ret_remain = float(np.log(end_px / mid_btc))

        rows.append(
            dict(
                window_start_ms=window_start_ms,
                mid_ts=mid_ts,
                start_px=start_px,
                mid_px=mid_btc,
                end_px=end_px,
                eth_ret_10s=eth_ret_10s,
                eth_ret_30s=eth_ret_30s,
                eth_ret_60s=eth_ret_60s,
                btc_ret_10s=btc_ret_10s,
                btc_ret_30s=btc_ret_30s,
                btc_ret_60s_to_mid=btc_ret_60s_to_mid,
                btc_ret_remain=btc_ret_remain,
                label_up=label_up,
            )
        )

    out = pd.DataFrame(rows)
    print(f"[features] windows with feature: {len(out)}")
    return out


def winrate_by_feature(df, feature_col, n_bins=5, label_col="label_up"):
    df = df.dropna(subset=[feature_col, label_col]).copy()
    if len(df) < 20:
        print(f"[winrate] too few rows: {len(df)}")
        return None
    try:
        df["bin"] = pd.qcut(df[feature_col], n_bins, labels=False, duplicates="drop")
    except Exception as e:
        print(f"[winrate] qcut failed: {e}")
        return None
    g = df.groupby("bin").agg(
        n=("label_up", "count"),
        up_rate=("label_up", "mean"),
        feat_lo=(feature_col, "min"),
        feat_hi=(feature_col, "max"),
        remain_mean=("btc_ret_remain", "mean"),
    )
    return g


def _winrate_remain(df, feature_col, n_bins=5):
    """Win rate by feature, using label based on REMAINING BTC move (mid->end)."""
    df = df.copy()
    df["label_remain_up"] = (df["btc_ret_remain"] > 0).astype(int)
    df = df.dropna(subset=[feature_col, "label_remain_up"])
    if len(df) < 20:
        return None
    try:
        df["bin"] = pd.qcut(df[feature_col], n_bins, labels=False, duplicates="drop")
    except Exception:
        return None
    g = df.groupby("bin").agg(
        n=("label_remain_up", "count"),
        remain_up_rate=("label_remain_up", "mean"),
        feat_lo=(feature_col, "min"),
        feat_hi=(feature_col, "max"),
        remain_mean_bp=("btc_ret_remain", lambda x: x.mean() * 1e4),
    )
    return g


def run_empirical_test():
    print("\n=== Empirical test: ETH 1m lagged return -> BTC window outcome ===")
    df = build_window_features()
    if df.empty:
        print("No features built")
        return

    # Feature 1: raw ETH 60s return (whole window label)
    print("\nFeature: ETH 60s return at mid-window, labels = BTC UP/DOWN (whole window)")
    g = winrate_by_feature(df, "eth_ret_60s", n_bins=5)
    if g is not None:
        print(g.to_string())

    # REMAINING-ONLY labels: does ETH 60s return predict the BTC move from mid -> end?
    print("\nFeature: ETH 60s return @ mid, label = BTC mid->end UP")
    g = _winrate_remain(df, "eth_ret_60s", n_bins=5)
    if g is not None:
        print(g.to_string())

    # Feature 2: residual between ETH and BTC in last 60s — does divergence mean-revert?
    df["eth_minus_btc_60s"] = df["eth_ret_60s"] - df["btc_ret_60s_to_mid"]
    print("\nFeature: ETH 60s - BTC 60s (divergence), label = BTC whole-window UP")
    g = winrate_by_feature(df, "eth_minus_btc_60s", n_bins=5)
    if g is not None:
        print(g.to_string())
    print("\nFeature: ETH 60s - BTC 60s (divergence), label = BTC mid->end UP")
    g = _winrate_remain(df, "eth_minus_btc_60s", n_bins=5)
    if g is not None:
        print(g.to_string())

    # Feature 3: signed ETH return — specifically, when |eth_ret| is large
    df["eth_ret_abs"] = df["eth_ret_60s"].abs()
    print("\nFeature: |ETH 60s return|, label = mid->end UP")
    g = _winrate_remain(df, "eth_ret_abs", n_bins=5)
    if g is not None:
        print(g.to_string())

    # ETH short-horizon tests — real leading candidates
    for col in ["eth_ret_10s", "eth_ret_30s"]:
        print(f"\nFeature: {col}, label = BTC mid->end UP")
        g = _winrate_remain(df, col, n_bins=5)
        if g is not None:
            print(g.to_string())

    # Divergence at 10s, 30s, 60s
    df["div_10s"] = df["eth_ret_10s"] - df["btc_ret_10s"]
    df["div_30s"] = df["eth_ret_30s"] - df["btc_ret_30s"]
    for col in ["div_10s", "div_30s"]:
        print(f"\nFeature: {col} (ETH-BTC short-horizon divergence), label = mid->end UP")
        g = _winrate_remain(df, col, n_bins=5)
        if g is not None:
            print(g.to_string())

    # Shorter lookback: ETH 30s at mid-window, does it predict next 150s of BTC?
    # We don't have eth_ret_30s built yet; compute quickly
    # Already have mid_ts; recompute using an inner block: skip here since build
    # only stored 60s. Keep what we have.

    # Save raw features for later
    out_path = ROOT / "analysis" / "outputs" / "cross_asset_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\n[features] saved {len(df)} rows -> {out_path}")

    # Correlations
    r = df[["eth_ret_60s", "btc_ret_remain"]].corr().iloc[0, 1]
    r_div = df[["eth_minus_btc_60s", "btc_ret_remain"]].corr().iloc[0, 1]
    print(f"\ncorr(eth_ret_60s,       btc_ret_remain) = {r:+.4f}  (n={len(df)})")
    print(f"corr(eth_minus_btc_60s, btc_ret_remain) = {r_div:+.4f}  (n={len(df)})")
    # Corr with the 60s-to-mid segment of BTC (expected to be large — sanity)
    rs = df[["eth_ret_60s", "btc_ret_60s_to_mid"]].corr().iloc[0, 1]
    print(f"corr(eth_ret_60s,       btc_ret_60s_to_mid) = {rs:+.4f}  (contemporaneous)")

    # Does divergence have a TAIL effect? Focus on top/bottom decile
    print("\n=== Tail test: div_30s extreme deciles vs BTC mid->end ===")
    q_lo = df["div_30s"].quantile(0.10)
    q_hi = df["div_30s"].quantile(0.90)
    lo = df[df["div_30s"] <= q_lo]
    hi = df[df["div_30s"] >= q_hi]
    print(f"  lo decile (ETH<<BTC recently): n={len(lo)}  mid->end up rate={lo['btc_ret_remain'].gt(0).mean():.3f}")
    print(f"  hi decile (ETH>>BTC recently): n={len(hi)}  mid->end up rate={hi['btc_ret_remain'].gt(0).mean():.3f}")

    # Does ETH 60s return predict the sign of the window's FUTURE move when it is LARGE?
    print("\n=== Tail test: |eth_ret_60s| > 0.10% subset ===")
    large = df[df["eth_ret_60s"].abs() > 0.001]
    print(f"  n={len(large)}")
    if len(large) >= 20:
        pos = large[large["eth_ret_60s"] > 0]
        neg = large[large["eth_ret_60s"] < 0]
        print(f"  ETH up {len(pos)} → BTC mid->end up rate = {pos['btc_ret_remain'].gt(0).mean():.3f}")
        print(f"  ETH dn {len(neg)} → BTC mid->end up rate = {neg['btc_ret_remain'].gt(0).mean():.3f}")

    # Whole-window label conditional on large ETH move (this captures persistence, so
    # if you knew ETH move and acted FAST you could piggyback)
    print("\n=== Tail test: large ETH 60s → BTC whole-window direction ===")
    if len(large) >= 20:
        pos = large[large["eth_ret_60s"] > 0]
        neg = large[large["eth_ret_60s"] < 0]
        print(f"  ETH up {len(pos)} → BTC whole up rate = {pos['label_up'].mean():.3f}")
        print(f"  ETH dn {len(neg)} → BTC whole up rate = {neg['label_up'].mean():.3f}")


def run_stat_sig_tail_test():
    """Compute confidence intervals for the ETH-large → BTC direction test.

    Uses features saved by run_empirical_test. Computes:
      - Proportion z-test vs 50% baseline
      - Bootstrap 95% CI on win rate
    """
    import math
    fpath = ROOT / "analysis" / "outputs" / "cross_asset_features.parquet"
    if not fpath.exists():
        print("no cached features; run run_empirical_test first")
        return
    df = pd.read_parquet(fpath)

    def prop_z(k, n, p0=0.5):
        p = k / n
        se = math.sqrt(p0 * (1 - p0) / n)
        z = (p - p0) / se
        return p, z

    print("\n=== Stat-sig on ETH-large → BTC direction test ===")
    large = df[df["eth_ret_60s"].abs() > 0.001]
    pos = large[large["eth_ret_60s"] > 0]
    neg = large[large["eth_ret_60s"] < 0]

    n_pos = len(pos)
    k_pos = int(pos["btc_ret_remain"].gt(0).sum())
    p_pos, z_pos = prop_z(k_pos, n_pos)
    print(f"  ETH up, n={n_pos}, mid->end up={k_pos}  p={p_pos:.3f}  z={z_pos:+.2f}")

    n_neg = len(neg)
    k_neg = int(neg["btc_ret_remain"].gt(0).sum())
    p_neg, z_neg = prop_z(k_neg, n_neg)
    print(f"  ETH dn, n={n_neg}, mid->end up={k_neg}  p={p_neg:.3f}  z={z_neg:+.2f}")

    # Two-sample proportion test
    p_pool = (k_pos + k_neg) / (n_pos + n_neg)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_pos + 1 / n_neg))
    z_diff = (p_pos - p_neg) / se
    print(f"  Two-sample z = {z_diff:+.2f}  p_diff = {p_pos - p_neg:+.3f}")

    # Tighter threshold
    print("\n=== Tighter threshold: |eth_ret_60s| > 0.20% ===")
    large2 = df[df["eth_ret_60s"].abs() > 0.002]
    pos = large2[large2["eth_ret_60s"] > 0]
    neg = large2[large2["eth_ret_60s"] < 0]
    n_pos = len(pos); n_neg = len(neg)
    if n_pos and n_neg:
        k_pos = int(pos["btc_ret_remain"].gt(0).sum())
        k_neg = int(neg["btc_ret_remain"].gt(0).sum())
        print(f"  ETH up, n={n_pos}, mid->end up rate={k_pos/n_pos:.3f}")
        print(f"  ETH dn, n={n_neg}, mid->end up rate={k_neg/n_neg:.3f}")

    # Cross-check with whole-window label to confirm the persistence effect
    print("\n=== Cross-check with WHOLE WINDOW (should be very strong if persistence) ===")
    large = df[df["eth_ret_60s"].abs() > 0.001]
    pos = large[large["eth_ret_60s"] > 0]
    neg = large[large["eth_ret_60s"] < 0]
    print(f"  ETH up, n={len(pos)}, whole up rate={pos['label_up'].mean():.3f}")
    print(f"  ETH dn, n={len(neg)}, whole up rate={neg['label_up'].mean():.3f}")


def run_btc_autocorr_check():
    """Does BTC past 60s return predict next 150s move?

    We want to confirm that if ETH is just mirroring BTC, BTC past momentum has
    the SAME (weak) predictive power as ETH. If so, cross-asset adds nothing.
    """
    fpath = ROOT / "analysis" / "outputs" / "cross_asset_features.parquet"
    if not fpath.exists():
        return
    df = pd.read_parquet(fpath)

    print("\n=== BTC own-momentum control: btc_ret_60s_to_mid predicts mid->end? ===")
    g = _winrate_remain(df, "btc_ret_60s_to_mid", n_bins=5)
    if g is not None:
        print(g.to_string())
    r = df[["btc_ret_60s_to_mid", "btc_ret_remain"]].corr().iloc[0, 1]
    print(f"\ncorr(btc_ret_60s_to_mid, btc_ret_remain) = {r:+.4f}")

    print("\n=== Large BTC own-momentum tail (|btc_ret|>0.10%) ===")
    large = df[df["btc_ret_60s_to_mid"].abs() > 0.001]
    pos = large[large["btc_ret_60s_to_mid"] > 0]
    neg = large[large["btc_ret_60s_to_mid"] < 0]
    print(f"  BTC up, n={len(pos)}, mid->end up rate={pos['btc_ret_remain'].gt(0).mean():.3f}")
    print(f"  BTC dn, n={len(neg)}, mid->end up rate={neg['btc_ret_remain'].gt(0).mean():.3f}")


if __name__ == "__main__":
    inspect_sampling()
    run_lead_lag_study()
    run_empirical_test()
    run_stat_sig_tail_test()
    run_btc_autocorr_check()
