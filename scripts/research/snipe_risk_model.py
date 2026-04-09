#!/usr/bin/env python3
"""
Snipe Risk Model: backtest flip risk on historical BTC 5m parquet data.

For each historical window, computes the probability that BTC crosses the
threshold (window_start_price) in the remaining time using a diffusion model.
Answers the question: "If I buy the winning side at 99c with N seconds left,
how often does the price actually flip?"

Usage:
    python scripts/research/snipe_risk_model.py                    # full backtest
    python scripts/research/snipe_risk_model.py --limit 100        # first 100 windows
    python scripts/research/snipe_risk_model.py --threshold 0.01   # single threshold
    python scripts/research/snipe_risk_model.py --workers 8        # parallel workers
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.special import erfc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = _PROJECT_ROOT / "data" / "btc_5m"
OUT_DIR = _PROJECT_ROOT / "data" / "snipe_research"

# Analysis window: only look at last 60 seconds of each market
ANALYSIS_WINDOW_S = 60

# Sigma estimation: rolling window of chainlink prices
SIGMA_ROLLING_S = 90

# Minimum rows in a parquet to be considered valid
MIN_ROWS = 60

# Fallback sigma when estimation fails (per-second BTC vol)
FALLBACK_SIGMA = 3e-5

# Thresholds to sweep: flip_risk must be below these
DEFAULT_THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.05]

# Entry price for snipe (buy winning side at this price)
ENTRY_PRICE = 0.99

# Alternative entry price to check
ALT_ENTRY_PRICE = 0.98

# Trade size for PnL simulation
TRADE_SIZE_USD = 30.0

# Polymarket taker fee: 2% * p * (1 - p)
TAKER_FEE_RATE = 0.02


# ---------------------------------------------------------------------------
# Fee calculation (matches backtest_core.poly_fee)
# ---------------------------------------------------------------------------

def poly_fee(p: float) -> float:
    """Polymarket taker fee for binary market."""
    return TAKER_FEE_RATE * p * (1.0 - p)


# ---------------------------------------------------------------------------
# Flip risk model
# ---------------------------------------------------------------------------

def flip_risk(delta_usd: float, sigma_per_s: float, tau_s: float,
              btc_price: float) -> float:
    """Probability that BTC crosses the threshold in remaining time.

    Uses complementary error function for the Brownian motion crossing
    probability. This is the standard result for P(B_t crosses level d)
    where B_t is Brownian motion with volatility sigma.

    Parameters
    ----------
    delta_usd : current price - window_start_price (positive = UP winning)
    sigma_per_s : per-second volatility of BTC price
    tau_s : seconds remaining until window end
    btc_price : current BTC price (for normalizing delta to log-return space)
    """
    if tau_s <= 0 or sigma_per_s <= 0:
        return 0.0

    # Normalize delta to log-return space
    delta_norm = abs(delta_usd) / btc_price
    sigma_remaining = sigma_per_s * np.sqrt(tau_s)

    if sigma_remaining <= 0:
        return 0.0

    # P(crossing zero) = erfc(|d| / (sigma * sqrt(2*tau)))
    z = delta_norm / (sigma_remaining * np.sqrt(2))
    return float(erfc(z))


# ---------------------------------------------------------------------------
# Sigma estimation from price series
# ---------------------------------------------------------------------------

def estimate_sigma_rolling(prices: np.ndarray, window: int = SIGMA_ROLLING_S
                           ) -> np.ndarray:
    """Compute rolling realized volatility from 1-second log prices.

    Returns an array of per-second sigma estimates, one per price observation.
    Uses a trailing window of `window` seconds. Positions before the window
    is full use the available history (minimum 10 observations).
    """
    n = len(prices)
    sigmas = np.full(n, FALLBACK_SIGMA)

    if n < 10:
        return sigmas

    log_prices = np.log(prices)
    log_returns = np.diff(log_prices)

    for i in range(n):
        # The log return at position i corresponds to the move from i-1 to i.
        # Use returns from max(0, i-window) to i.
        start = max(0, i - window)
        end = i
        if end <= start:
            continue

        chunk = log_returns[start:end]
        # Remove zeros (stale prices where chainlink didn't update)
        nonzero = chunk[chunk != 0.0]

        if len(nonzero) < 5:
            continue

        sigmas[i] = float(np.std(nonzero))

    return sigmas


# ---------------------------------------------------------------------------
# Per-window result
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    """Results from analyzing a single 5-minute window."""
    filename: str
    valid: bool = False
    outcome: Optional[str] = None  # "UP" or "DOWN"
    n_rows: int = 0

    # Per-threshold results: threshold -> {safe_entry: bool, first_safe_s: float,
    #   predicted_correct: bool, flip_risk_at_entry: float}
    threshold_results: dict = field(default_factory=dict)

    # Liquidity info (last 60s of window)
    has_ask_99_60s: bool = False
    has_ask_99_30s: bool = False
    has_ask_99_10s: bool = False
    has_ask_98_60s: bool = False
    median_ask_size_99: float = 0.0

    # Flip case details (when predicted winner at safe entry actually lost)
    flip_details: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Analyze a single parquet file
# ---------------------------------------------------------------------------

def _find_ask_at_level(row: pd.Series, side: str, target_px: float,
                       tol: float = 0.005) -> tuple[bool, float]:
    """Check if there's an ask within tolerance of target price for the given side.

    Returns (available, size). Searches through ask levels 1-5.
    """
    for lvl in range(1, 6):
        px_col = f"ask_px_{side}_{lvl}"
        sz_col = f"ask_sz_{side}_{lvl}"
        if px_col not in row.index or sz_col not in row.index:
            continue
        px = row[px_col]
        sz = row[sz_col]
        if pd.isna(px) or pd.isna(sz):
            continue
        if abs(px - target_px) <= tol:
            return True, float(sz)
    return False, 0.0


def analyze_window(filepath: str, thresholds: list[float]) -> WindowResult:
    """Analyze a single parquet file for snipe opportunities."""
    path = Path(filepath)
    result = WindowResult(filename=path.name)

    try:
        df = pd.read_parquet(path)
    except Exception:
        return result

    result.n_rows = len(df)
    if len(df) < MIN_ROWS:
        return result

    # Sort by time_remaining descending (300 -> 0) so iloc[-1] is the end
    df = df.sort_values("time_remaining_s", ascending=False).reset_index(drop=True)

    # --- Determine actual outcome ---
    # Use the last valid chainlink_price (closest to time_remaining_s == 0)
    tail = df.dropna(subset=["chainlink_price", "window_start_price"])
    if tail.empty:
        return result

    # Row with minimum time_remaining_s that has valid data
    final_row = tail.loc[tail["time_remaining_s"].idxmin()]
    final_price = float(final_row["chainlink_price"])
    start_price = float(final_row["window_start_price"])

    # Exact tie: skip (ambiguous outcome)
    if final_price == start_price:
        return result

    outcome = "UP" if final_price > start_price else "DOWN"
    result.outcome = outcome
    result.valid = True

    # --- Restrict to analysis window (last 60s) ---
    mask_60s = df["time_remaining_s"] <= ANALYSIS_WINDOW_S
    analysis = df[mask_60s].copy()
    if analysis.empty:
        result.valid = False
        return result

    # --- Compute sigma from rolling window over full parquet ---
    # (We need prices BEFORE the analysis window for the rolling estimator)
    all_prices = df["chainlink_price"].values.astype(float)
    # Replace NaN with forward fill
    nan_mask = np.isnan(all_prices)
    if nan_mask.all():
        result.valid = False
        return result
    if nan_mask.any():
        # Forward fill: replace NaN with last valid value
        for i in range(len(all_prices)):
            if nan_mask[i] and i > 0:
                all_prices[i] = all_prices[i - 1]
        # Backward fill any remaining leading NaNs
        first_valid = np.argmin(nan_mask)
        all_prices[:first_valid] = all_prices[first_valid]

    sigmas_all = estimate_sigma_rolling(all_prices, window=SIGMA_ROLLING_S)

    # Map sigmas back to the analysis window rows
    analysis_indices = analysis.index.values
    sigmas = sigmas_all[analysis_indices]
    taus = analysis["time_remaining_s"].values.astype(float)
    prices = all_prices[analysis_indices]
    start_prices = analysis["window_start_price"].values.astype(float)
    deltas = prices - start_prices

    # --- Vectorized flip risk computation ---
    delta_norms = np.abs(deltas) / prices
    sigma_remaining = sigmas * np.sqrt(np.maximum(taus, 0))

    # Avoid division by zero
    safe_denom = np.where(sigma_remaining > 0, sigma_remaining, 1.0)
    z = delta_norms / (safe_denom * np.sqrt(2))
    risks = np.where((taus > 0) & (sigma_remaining > 0), erfc(z), 0.0)

    # Predicted winner at each second
    predicted_up = deltas > 0
    predicted_down = deltas < 0
    predicted_winner = np.where(predicted_up, "UP", np.where(predicted_down, "DOWN", "NONE"))

    # Whether the prediction was correct (matches actual outcome)
    prediction_correct = predicted_winner == outcome

    # --- Liquidity check ---
    winning_side = "up" if outcome == "UP" else "down"

    # Check asks at 0.99 and 0.98 for the winning side
    ask_99_flags = np.zeros(len(analysis), dtype=bool)
    ask_98_flags = np.zeros(len(analysis), dtype=bool)
    ask_sizes_99 = np.zeros(len(analysis), dtype=float)

    for i, (_, row) in enumerate(analysis.iterrows()):
        avail_99, sz_99 = _find_ask_at_level(row, winning_side, 0.99, tol=0.005)
        ask_99_flags[i] = avail_99
        ask_sizes_99[i] = sz_99
        avail_98, _ = _find_ask_at_level(row, winning_side, 0.98, tol=0.005)
        ask_98_flags[i] = avail_98

    result.has_ask_99_60s = bool(ask_99_flags.any())
    result.has_ask_99_30s = bool(ask_99_flags[taus <= 30].any()) if (taus <= 30).any() else False
    result.has_ask_99_10s = bool(ask_99_flags[taus <= 10].any()) if (taus <= 10).any() else False
    result.has_ask_98_60s = bool(ask_98_flags.any())

    valid_sizes = ask_sizes_99[ask_sizes_99 > 0]
    result.median_ask_size_99 = float(np.median(valid_sizes)) if len(valid_sizes) > 0 else 0.0

    # --- Per-threshold analysis ---
    for threshold in thresholds:
        safe_mask = risks < threshold
        # Also require that there's a definite leader (delta != 0)
        safe_mask = safe_mask & (deltas != 0)

        if not safe_mask.any():
            result.threshold_results[threshold] = {
                "safe_entry": False,
                "first_safe_s": None,
                "predicted_correct": None,
                "flip_risk_at_entry": None,
            }
            continue

        # First safe entry: the one with the HIGHEST time_remaining_s
        safe_indices = np.where(safe_mask)[0]
        first_safe_idx = safe_indices[0]  # highest tau since sorted desc

        first_safe_tau = float(taus[first_safe_idx])
        first_safe_correct = bool(prediction_correct[first_safe_idx])
        first_safe_risk = float(risks[first_safe_idx])

        result.threshold_results[threshold] = {
            "safe_entry": True,
            "first_safe_s": first_safe_tau,
            "predicted_correct": first_safe_correct,
            "flip_risk_at_entry": first_safe_risk,
        }

        # Record flip details when the safe entry was actually WRONG
        if not first_safe_correct:
            result.flip_details.append({
                "threshold": threshold,
                "tau_s": first_safe_tau,
                "delta_usd": float(deltas[first_safe_idx]),
                "sigma": float(sigmas[first_safe_idx]),
                "flip_risk": first_safe_risk,
                "outcome": outcome,
                "predicted": str(predicted_winner[first_safe_idx]),
                "final_price": final_price,
                "start_price": start_price,
            })

    return result


# ---------------------------------------------------------------------------
# Batch processing with multiprocessing
# ---------------------------------------------------------------------------

def process_batch(filepaths: list[str], thresholds: list[float],
                  workers: int = 4, show_progress: bool = True
                  ) -> list[WindowResult]:
    """Process multiple parquet files in parallel."""
    results: list[WindowResult] = []
    total = len(filepaths)
    done = 0
    t0 = _time.monotonic()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(analyze_window, fp, thresholds): fp
            for fp in filepaths
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                fp = futures[future]
                print(f"  ERROR processing {fp}: {exc}", file=sys.stderr)
                results.append(WindowResult(filename=Path(fp).name))

            done += 1
            if show_progress and done % 500 == 0:
                elapsed = _time.monotonic() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  [{done:,}/{total:,}] "
                      f"{rate:.0f} windows/s, ETA {eta:.0f}s")

    return results


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(results: list[WindowResult], thresholds: list[float]
                    ) -> dict:
    """Aggregate per-window results into summary statistics."""
    valid = [r for r in results if r.valid]
    n_total = len(results)
    n_valid = len(valid)

    # Per-threshold aggregation
    threshold_summaries = {}
    for thr in thresholds:
        safe_windows = [r for r in valid if r.threshold_results.get(thr, {}).get("safe_entry")]
        n_safe = len(safe_windows)

        if n_safe == 0:
            threshold_summaries[thr] = {
                "n_safe": 0,
                "pct_safe": 0.0,
                "median_first_safe_s": None,
                "win_rate": None,
                "n_wins": 0,
                "n_losses": 0,
                "simulated_pnl": 0.0,
            }
            continue

        first_safe_times = [r.threshold_results[thr]["first_safe_s"] for r in safe_windows]
        wins = [r for r in safe_windows if r.threshold_results[thr]["predicted_correct"]]
        losses = [r for r in safe_windows if not r.threshold_results[thr]["predicted_correct"]]

        n_wins = len(wins)
        n_losses = len(losses)
        win_rate = n_wins / n_safe if n_safe > 0 else 0.0

        # PnL simulation: buy at ENTRY_PRICE, pay taker fee
        fee = poly_fee(ENTRY_PRICE)
        cost_per_share = ENTRY_PRICE + fee
        shares_per_trade = TRADE_SIZE_USD / cost_per_share
        # Win: receive $1.00 per share -> profit = (1 - cost_per_share) * shares
        # Loss: receive $0 -> loss = cost_per_share * shares
        profit_per_win = (1.0 - cost_per_share) * shares_per_trade
        loss_per_loss = cost_per_share * shares_per_trade
        simulated_pnl = n_wins * profit_per_win - n_losses * loss_per_loss

        threshold_summaries[thr] = {
            "n_safe": n_safe,
            "pct_safe": n_safe / n_valid * 100 if n_valid > 0 else 0.0,
            "median_first_safe_s": float(np.median(first_safe_times)),
            "win_rate": win_rate * 100,
            "n_wins": n_wins,
            "n_losses": n_losses,
            "simulated_pnl": round(simulated_pnl, 2),
        }

    # Liquidity summary
    has_99_60 = sum(1 for r in valid if r.has_ask_99_60s)
    has_99_30 = sum(1 for r in valid if r.has_ask_99_30s)
    has_99_10 = sum(1 for r in valid if r.has_ask_99_10s)
    has_98_60 = sum(1 for r in valid if r.has_ask_98_60s)
    sizes_99 = [r.median_ask_size_99 for r in valid if r.median_ask_size_99 > 0]

    liquidity = {
        "ask_99_60s_pct": has_99_60 / n_valid * 100 if n_valid > 0 else 0.0,
        "ask_99_30s_pct": has_99_30 / n_valid * 100 if n_valid > 0 else 0.0,
        "ask_99_10s_pct": has_99_10 / n_valid * 100 if n_valid > 0 else 0.0,
        "ask_98_60s_pct": has_98_60 / n_valid * 100 if n_valid > 0 else 0.0,
        "median_ask_size_99": float(np.median(sizes_99)) if sizes_99 else 0.0,
    }

    # Flip case details
    all_flips = []
    for r in valid:
        all_flips.extend(r.flip_details)

    return {
        "n_total": n_total,
        "n_valid": n_valid,
        "thresholds": threshold_summaries,
        "liquidity": liquidity,
        "flips": all_flips,
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_summary(summary: dict, thresholds: list[float]) -> None:
    """Print human-readable summary table."""
    n_total = summary["n_total"]
    n_valid = summary["n_valid"]
    liq = summary["liquidity"]
    flips = summary["flips"]

    fee = poly_fee(ENTRY_PRICE)
    cost = ENTRY_PRICE + fee

    print()
    print("Resolution Snipe Risk Model Backtest")
    print("=" * 70)
    print(f"Windows analyzed:    {n_total:,}")
    print(f"Windows with valid data: {n_valid:,}")
    print(f"Entry price:         {ENTRY_PRICE}")
    print(f"Taker fee at {ENTRY_PRICE}: {fee:.6f}")
    print(f"Effective cost:      {cost:.6f}")
    print(f"Trade size:          ${TRADE_SIZE_USD:.0f}")
    print()

    # Threshold table
    header = (f"{'Threshold':>10} | {'Safe entries':>22} | "
              f"{'Earliest safe (med)':>20} | {'Win rate':>10} | "
              f"{'PnL ($' + f'{TRADE_SIZE_USD:.0f}/trade)':>18}")
    print(header)
    print("-" * len(header))

    for thr in thresholds:
        ts = summary["thresholds"].get(thr, {})
        n_safe = ts.get("n_safe", 0)
        pct_safe = ts.get("pct_safe", 0.0)
        med_s = ts.get("median_first_safe_s")
        win_rate = ts.get("win_rate")
        pnl = ts.get("simulated_pnl", 0.0)

        safe_str = f"{n_safe:,} ({pct_safe:.1f}%)"
        med_str = f"{med_s:.1f}s remaining" if med_s is not None else "N/A"
        wr_str = f"{win_rate:.2f}%" if win_rate is not None else "N/A"
        pnl_str = f"${pnl:,.2f}"

        print(f"{thr*100:>9.1f}% | {safe_str:>22} | "
              f"{med_str:>20} | {wr_str:>10} | {pnl_str:>18}")

    print()
    print("Liquidity at 99c (predicted winner side):")
    print(f"  Windows with any ask at 0.99 in last 60s: {liq['ask_99_60s_pct']:.1f}%")
    print(f"  Windows with ask at 0.99 in last 30s:     {liq['ask_99_30s_pct']:.1f}%")
    print(f"  Windows with ask at 0.99 in last 10s:     {liq['ask_99_10s_pct']:.1f}%")
    print(f"  Windows with ask at 0.98 in last 60s:     {liq['ask_98_60s_pct']:.1f}%")
    print(f"  Median ask size at 0.99 when available:    {liq['median_ask_size_99']:.0f} shares")
    print()

    # Flip analysis
    for thr in thresholds:
        thr_flips = [f for f in flips if f["threshold"] == thr]
        n_flip = len(thr_flips)
        ts = summary["thresholds"].get(thr, {})
        n_safe = ts.get("n_safe", 0)

        print(f"Flips at {thr*100:.1f}% threshold: "
              f"{n_flip} windows lost out of {n_safe} safe entries")

        if thr_flips:
            # Show worst cases (sorted by flip_risk descending - most surprising)
            worst = sorted(thr_flips, key=lambda x: x["flip_risk"])[:5]
            for w in worst:
                print(f"    tau={w['tau_s']:.0f}s  delta=${w['delta_usd']:.2f}  "
                      f"sigma={w['sigma']:.2e}  risk={w['flip_risk']:.4f}  "
                      f"predicted={w['predicted']}  actual={w['outcome']}")

    print()


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def save_results(summary: dict, results: list[WindowResult],
                 out_dir: Path) -> Path:
    """Save detailed results to JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "risk_model_results.json"

    # Convert WindowResult dataclasses to dicts for serialization
    per_window = []
    for r in results:
        if not r.valid:
            continue
        d = asdict(r)
        # Convert numpy types to Python native for JSON
        per_window.append(d)

    payload = {
        "summary": summary,
        "per_window": per_window,
    }

    # Custom serializer for numpy types
    def _default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=_default)

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest flip risk model on BTC 5m parquet data")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to first N window files (0 = all)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Run with a single threshold instead of sweep")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4)")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving JSON results")
    args = parser.parse_args()

    # Resolve thresholds
    if args.threshold is not None:
        thresholds = [args.threshold]
    else:
        thresholds = DEFAULT_THRESHOLDS

    # Discover parquet files
    if not DATA_DIR.exists():
        print(f"ERROR: Data directory not found: {DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    parquet_files = sorted(str(p) for p in DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        print(f"ERROR: No parquet files in {DATA_DIR}", file=sys.stderr)
        sys.exit(1)

    if args.limit > 0:
        parquet_files = parquet_files[:args.limit]

    print(f"Snipe Risk Model Backtest")
    print(f"  Data dir:    {DATA_DIR}")
    print(f"  Windows:     {len(parquet_files):,}")
    print(f"  Thresholds:  {[f'{t*100:.1f}%' for t in thresholds]}")
    print(f"  Workers:     {args.workers}")
    print(f"  Entry price: {ENTRY_PRICE}")
    print()

    t0 = _time.monotonic()
    results = process_batch(parquet_files, thresholds,
                            workers=args.workers, show_progress=True)
    elapsed = _time.monotonic() - t0

    print(f"\nProcessed {len(results):,} windows in {elapsed:.1f}s "
          f"({len(results)/elapsed:.0f} windows/s)")

    # Compute and print summary
    summary = compute_summary(results, thresholds)
    print_summary(summary, thresholds)

    # Save results
    if not args.no_save:
        out_path = save_results(summary, results, OUT_DIR)
        print(f"Detailed results saved to: {out_path}")


if __name__ == "__main__":
    main()
