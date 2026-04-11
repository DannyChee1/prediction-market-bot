#!/usr/bin/env python3
"""Sigma lookback empirical research.

Question: what σ lookback best predicts FUTURE realized volatility for
BTC 5m / 15m Polymarket windows?

Method:
  1. Build a continuous BTC price timeline from all parquets (binance_mid only).
  2. For each window, at multiple decision times T (early/mid/late), compute σ
     at many lookbacks (30s through 1 month) using ONLY data up to T.
  3. Compute the ACTUAL realized volatility from T to T+horizon (the ground
     truth that the lookback was trying to forecast).
  4. Score each lookback by forecast error vs the actual.

Completeness rules (strict — no fabrication):
  - binance_mid must be non-null and >0 for every sample used.
  - Lookback samples must be contiguous (gap ≤ 5s between consecutive ticks).
  - Forecast horizon samples must also be contiguous.
  - Skip the entire decision point if either condition fails.
  - For cross-window lookbacks (>5min), the lookback may span the gap
    between windows ONLY IF the gap between adjacent parquets is ≤ 60s.
    Larger gaps → skip.
  - Aggregation only includes valid (non-skipped) decision points.

Loss functions:
  - MAE   : |σ_pred − σ_actual|
  - log MAE: |log(σ_pred) − log(σ_actual)|  (scale-invariant; better for σ)
  - QLIKE  : log(σ_pred²) + σ_actual²/σ_pred²  (standard vol-forecast loss)

Outputs: console table + tasks/findings/sigma_lookback_research_2026-04-11.md

Usage:
    python analysis/sigma_lookback_research.py --market btc_5m
    python analysis/sigma_lookback_research.py --market btc_5m --quick
    python analysis/sigma_lookback_research.py --market btc --max-windows 2000
"""
from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

ROOT = Path("/Users/dannychee/Desktop/prediction-market-bot")
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402

from market_config import get_config  # noqa: E402

# Lookbacks to test (seconds). Mixture of within-window and cross-window.
# Within-window: 30, 60, 90, 120, 180, 240, 300s
# Cross-window: 600s (10m), 1800s (30m), 3600s (1h), 21600s (6h),
#               86400s (1d), 604800s (1w), 2592000s (30d)
LOOKBACKS = [
    30, 60, 90, 120, 180, 240, 300,
    600, 1800, 3600, 21600, 86400, 604800, 2592000,
]

# Forecast horizons to test (seconds): how far ahead to verify.
# 60s = "what will σ be in the next minute"
# 120s = "next 2 minutes"
# 300s = "next 5 minutes" (full window)
HORIZONS = [60, 120, 300]

# Max gap between consecutive samples to consider them "contiguous"
MAX_GAP_S = 5

# Max gap between adjacent parquets to allow chaining for long lookbacks
MAX_PARQUET_GAP_S = 60

# Decision-time tau positions to sample within each window.
# Skip the very start (need history) and the very end (need forecast horizon).
DECISION_TAUS = [240, 180, 120, 60]  # taus REMAINING when decision is made


def realized_sigma_per_s(prices: list[float], ts_ms: list[int]) -> float | None:
    """Realized σ from log returns, normalized by sqrt(dt).

    Returns the standard deviation of (log(p_i / p_{i-1}) / sqrt(dt_s))
    across consecutive valid samples. Same convention as `_compute_vol_deduped`'s
    fallback path so the result is directly comparable to the bot's σ values.

    Returns None if fewer than 3 valid log-return observations.
    """
    if len(prices) < 4:
        return None
    log_rets = []
    for i in range(1, len(prices)):
        if prices[i] <= 0 or prices[i - 1] <= 0:
            continue
        if prices[i] == prices[i - 1]:
            continue  # skip duplicates
        dt_s = (ts_ms[i] - ts_ms[i - 1]) / 1000.0
        if dt_s <= 0:
            continue
        lr = math.log(prices[i] / prices[i - 1])
        log_rets.append(lr / math.sqrt(dt_s))
    if len(log_rets) < 3:
        return None
    return float(np.std(log_rets, ddof=1))


def load_continuous_timeline(market: str) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Load all binance_mid prices for a market, sorted by timestamp.

    Returns:
        ts_ms: int64 array of timestamps in ms
        prices: float64 array of binance_mid prices
        window_bounds: list of (start_idx, end_idx) for each window
                       so we can map a sample back to its source window
    """
    cfg = get_config(market)
    data_dir = ROOT / "data" / cfg.data_subdir
    files = sorted(data_dir.glob("*.parquet"))
    print(f"  loading {len(files)} parquets from {data_dir}", file=sys.stderr)

    chunks = []
    window_starts = []
    n_skipped_no_binance = 0
    n_skipped_too_short = 0
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["ts_ms", "binance_mid",
                                              "window_start_ms", "window_end_ms"])
        except Exception:
            continue
        if "binance_mid" not in df.columns:
            n_skipped_no_binance += 1
            continue
        df = df.dropna(subset=["binance_mid", "ts_ms"])
        df = df[df["binance_mid"] > 0]
        if len(df) < 5:
            n_skipped_too_short += 1
            continue
        df = df.sort_values("ts_ms").reset_index(drop=True)
        chunks.append(df)
        window_starts.append((int(df["window_start_ms"].iloc[0]),
                              int(df["window_end_ms"].iloc[0])))

    if not chunks:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64), []

    print(f"  kept {len(chunks)} parquets "
          f"(skipped {n_skipped_no_binance} no-binance, "
          f"{n_skipped_too_short} too-short)", file=sys.stderr)

    # Concatenate everything into a single timeline
    all_df = pd.concat(chunks, ignore_index=True)
    all_df = all_df.sort_values("ts_ms").reset_index(drop=True)
    # Drop exact-duplicate timestamps (can happen at window boundaries)
    all_df = all_df.drop_duplicates(subset="ts_ms", keep="first").reset_index(drop=True)

    ts_ms = all_df["ts_ms"].values.astype(np.int64)
    prices = all_df["binance_mid"].values.astype(np.float64)

    # Build window bounds in the concatenated array
    window_bounds = []
    for ws, we in window_starts:
        # Find indices where ts_ms falls within [ws, we]
        lo = int(np.searchsorted(ts_ms, ws, side="left"))
        hi = int(np.searchsorted(ts_ms, we, side="right"))
        if hi - lo >= 5:
            window_bounds.append((lo, hi))

    return ts_ms, prices, window_bounds


def get_contiguous_slice(
    ts_ms: np.ndarray,
    prices: np.ndarray,
    end_idx: int,
    lookback_ms: int,
    max_gap_s: int = MAX_GAP_S,
    max_parquet_gap_s: int = MAX_PARQUET_GAP_S,
) -> tuple[list[float], list[int]] | None:
    """Walk backward from end_idx for `lookback_ms` worth of samples.

    Returns None if the slice contains a gap larger than max_parquet_gap_s,
    or if there are fewer than 4 samples (insufficient for σ).
    """
    if end_idx <= 0:
        return None
    target_start_ms = ts_ms[end_idx] - lookback_ms

    # Binary search for the start
    start_idx = int(np.searchsorted(ts_ms, target_start_ms, side="left"))
    if start_idx >= end_idx - 3:
        return None

    sl_ts = ts_ms[start_idx:end_idx + 1]
    sl_px = prices[start_idx:end_idx + 1]

    # Verify contiguity
    gaps_ms = np.diff(sl_ts)
    if len(gaps_ms) == 0:
        return None
    max_gap_ms = int(gaps_ms.max())
    if max_gap_ms > max_parquet_gap_s * 1000:
        return None  # contains a hole bigger than allowed

    return sl_px.tolist(), sl_ts.tolist()


def get_forward_slice(
    ts_ms: np.ndarray,
    prices: np.ndarray,
    start_idx: int,
    horizon_ms: int,
    max_gap_s: int = MAX_GAP_S,
    window_end_idx: int | None = None,
) -> tuple[list[float], list[int]] | None:
    """Walk forward from start_idx for `horizon_ms` worth of samples.

    If window_end_idx is given, the slice is clipped to stay inside the window
    (the forecast must be verified within the SAME window — different windows
    have different drift regimes).

    Returns None if the slice has gaps > max_gap_s or fewer than 4 samples.
    """
    if start_idx >= len(ts_ms) - 3:
        return None
    target_end_ms = ts_ms[start_idx] + horizon_ms

    end_idx = int(np.searchsorted(ts_ms, target_end_ms, side="right"))
    if window_end_idx is not None:
        end_idx = min(end_idx, window_end_idx)
    if end_idx - start_idx < 4:
        return None

    sl_ts = ts_ms[start_idx:end_idx]
    sl_px = prices[start_idx:end_idx]

    # Verify contiguity (forward must be tight — no big gaps)
    gaps_ms = np.diff(sl_ts)
    if len(gaps_ms) == 0:
        return None
    max_gap_ms = int(gaps_ms.max())
    if max_gap_ms > max_gap_s * 1000:
        return None

    return sl_px.tolist(), sl_ts.tolist()


def run_experiment(market: str, max_windows: int | None, quick: bool) -> dict:
    ts_ms, prices, window_bounds = load_continuous_timeline(market)
    if len(ts_ms) == 0:
        print(f"  No data for {market}", file=sys.stderr)
        return {}
    print(f"  total samples: {len(ts_ms):,}  "
          f"windows: {len(window_bounds):,}", file=sys.stderr)

    if max_windows is not None and len(window_bounds) > max_windows:
        # Sample evenly across the timeline
        step = len(window_bounds) // max_windows
        window_bounds = window_bounds[::step][:max_windows]
        print(f"  subsampled to {len(window_bounds)} windows", file=sys.stderr)

    cfg = get_config(market)
    win_dur_s = int(cfg.window_duration_s)

    # For each (lookback, horizon), accumulate forecast errors
    errors: dict[tuple[int, int], dict] = {}
    for lb in LOOKBACKS:
        for h in HORIZONS:
            errors[(lb, h)] = {
                "n": 0,
                "mae": [],
                "log_mae": [],
                "qlike": [],
                "lookback_sigma": [],
                "actual_sigma": [],
            }

    n_processed = 0
    n_skip_lookback = 0
    n_skip_forward = 0
    n_skip_no_actual = 0

    decision_taus = DECISION_TAUS if not quick else [180]

    for w_lo, w_hi in window_bounds:
        for tau_remaining in decision_taus:
            # Decision time: tau_remaining seconds before window end
            target_decision_ms = ts_ms[w_hi - 1] - tau_remaining * 1000
            # Find the closest sample at or before this time
            decision_idx = int(np.searchsorted(ts_ms, target_decision_ms, side="right")) - 1
            if decision_idx < w_lo or decision_idx >= w_hi:
                continue

            for lb in LOOKBACKS:
                lb_slice = get_contiguous_slice(ts_ms, prices, decision_idx, lb * 1000)
                if lb_slice is None:
                    n_skip_lookback += 1
                    continue
                lb_px, lb_ts = lb_slice
                sigma_pred = realized_sigma_per_s(lb_px, lb_ts)
                if sigma_pred is None or sigma_pred <= 0:
                    n_skip_lookback += 1
                    continue

                for h in HORIZONS:
                    fwd_slice = get_forward_slice(
                        ts_ms, prices, decision_idx, h * 1000,
                        window_end_idx=w_hi,
                    )
                    if fwd_slice is None:
                        n_skip_forward += 1
                        continue
                    fwd_px, fwd_ts = fwd_slice
                    sigma_actual = realized_sigma_per_s(fwd_px, fwd_ts)
                    if sigma_actual is None or sigma_actual <= 0:
                        n_skip_no_actual += 1
                        continue

                    rec = errors[(lb, h)]
                    rec["n"] += 1
                    rec["mae"].append(abs(sigma_pred - sigma_actual))
                    rec["log_mae"].append(abs(math.log(sigma_pred) - math.log(sigma_actual)))
                    # QLIKE: log(p²) + a²/p²  -- robust vol forecast loss
                    p2 = sigma_pred * sigma_pred
                    a2 = sigma_actual * sigma_actual
                    rec["qlike"].append(math.log(p2) + a2 / p2)
                    rec["lookback_sigma"].append(sigma_pred)
                    rec["actual_sigma"].append(sigma_actual)
            n_processed += 1

    print(f"  processed {n_processed:,} decision points  "
          f"(skipped: {n_skip_lookback:,} lookback / "
          f"{n_skip_forward:,} forward / "
          f"{n_skip_no_actual:,} no-actual)", file=sys.stderr)

    return errors


def summarise(errors: dict, market: str) -> str:
    """Build a per-horizon table showing best lookback by each loss function."""
    out_lines = []
    out_lines.append(f"\n## {market}\n")

    for h in HORIZONS:
        out_lines.append(f"\n### Forecast horizon: {h}s\n")
        out_lines.append(f"| lookback | n | MAE | log_MAE | QLIKE | "
                         f"med_pred σ | med_actual σ |")
        out_lines.append(f"|---:|---:|---:|---:|---:|---:|---:|")

        rows = []
        for lb in LOOKBACKS:
            rec = errors.get((lb, h))
            if not rec or rec["n"] == 0:
                continue
            mae = float(np.median(rec["mae"]))
            log_mae = float(np.median(rec["log_mae"]))
            qlike = float(np.median(rec["qlike"]))
            med_pred = float(np.median(rec["lookback_sigma"]))
            med_actual = float(np.median(rec["actual_sigma"]))
            rows.append((lb, rec["n"], mae, log_mae, qlike, med_pred, med_actual))

        # Find best by each metric
        if not rows:
            continue
        best_mae = min(rows, key=lambda r: r[2])[0]
        best_log = min(rows, key=lambda r: r[3])[0]
        best_qlike = min(rows, key=lambda r: r[4])[0]

        for lb, n, mae, log_mae, qlike, mp, ma in rows:
            mark = ""
            if lb == best_mae:
                mark += " ⭐MAE"
            if lb == best_log:
                mark += " ⭐logMAE"
            if lb == best_qlike:
                mark += " ⭐QLIKE"
            lb_str = (f"{lb}s" if lb < 600
                      else f"{lb//60}m" if lb < 3600
                      else f"{lb//3600}h" if lb < 86400
                      else f"{lb//86400}d")
            out_lines.append(f"| {lb_str}{mark} | {n} | {mae:.2e} | "
                             f"{log_mae:.3f} | {qlike:.2f} | "
                             f"{mp:.2e} | {ma:.2e} |")

        # Print verbal summary
        out_lines.append(f"\n  **Best by MAE:** {best_mae}s   "
                         f"**Best by log-MAE:** {best_log}s   "
                         f"**Best by QLIKE:** {best_qlike}s\n")

    return "\n".join(out_lines)


def main():
    ap = argparse.ArgumentParser(
        description="Sigma lookback empirical research"
    )
    ap.add_argument("--market", default="btc_5m",
                    help="Market key (default btc_5m). 'btc' = btc 15m.")
    ap.add_argument("--max-windows", type=int, default=None,
                    help="Subsample to this many windows (default: all)")
    ap.add_argument("--quick", action="store_true",
                    help="Quick mode: only one decision tau per window")
    args = ap.parse_args()

    markets = ["btc_5m", "btc"] if args.market == "all" else [args.market]

    full_report = ["# Sigma Lookback Research — 2026-04-11\n"]
    full_report.append(
        "Empirical study of which σ lookback best forecasts future "
        "realized volatility on BTC Polymarket windows.\n\n"
        "**Method:** for each window, at multiple decision times, compute σ "
        "from many lookbacks using ONLY data up to that point, then compare "
        "to the realized σ over the next forecast horizon. Strict completeness "
        "rules: gaps > 5s in forecast slice → skip; gaps > 60s in lookback "
        "slice → skip; binance_mid only (no chainlink fallback).\n\n"
        "**Loss functions:** MAE (raw), log-MAE (scale-invariant), QLIKE "
        "(standard vol-forecast loss from Patton 2011).\n\n"
        "Lower = better for all three.\n"
    )

    for m in markets:
        print(f"\n=== {m} ===", file=sys.stderr)
        errors = run_experiment(m, args.max_windows, args.quick)
        if errors:
            full_report.append(summarise(errors, m))

    out_path = ROOT / "tasks" / "findings" / "sigma_lookback_research_2026-04-11.md"
    out_path.write_text("\n".join(full_report))
    print(f"\n  Wrote report to {out_path}", file=sys.stderr)
    print("\n".join(full_report))


if __name__ == "__main__":
    main()
