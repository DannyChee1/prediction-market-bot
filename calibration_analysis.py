#!/usr/bin/env python3
"""
Calibration analysis: Is our pure GBM p_up accurate?

Evaluates the INDEPENDENT probability estimate (no Bayesian calibration)
across all complete parquet windows. Produces reliability diagrams and
summary statistics per market.
"""

import math
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from backtest import (
    _compute_vol_deduped, norm_cdf, fast_t_cdf,
    MIN_FINAL_REMAINING_S, MAX_START_GAP_S,
)
from market_config import MARKET_CONFIGS

DATA_DIR = Path(__file__).parent / "data"

# ── CDF dispatch (mirrors _model_cdf but standalone) ──────────────────────

def model_cdf(z: float, cfg, sigma_per_s: float = 1e-5, tau: float = 300.0) -> float:
    """Pure GBM probability — no calibration table involved."""
    if cfg.tail_mode == "normal":
        return norm_cdf(z)
    if cfg.tail_mode == "kou":
        eta1, eta2 = cfg.kou_eta1, cfg.kou_eta2
        p_up, q_dn = cfg.kou_p_up, 1.0 - cfg.kou_p_up
        zeta = (p_up * eta1 / max(eta1 - 1, 0.01)
                + q_dn * eta2 / (eta2 + 1) - 1.0)
        drift_z = -cfg.kou_lambda * zeta * math.sqrt(tau) / max(sigma_per_s, 1e-8)
        return norm_cdf(z + drift_z)
    if cfg.tail_mode == "student_t":
        return fast_t_cdf(z, cfg.tail_nu_default)
    return norm_cdf(z)


# ── Load complete windows ──────────────────────────────────────────────────

def load_complete_windows(data_dir: Path):
    """Yield (df, start_px, outcome) for each complete window."""
    files = sorted(data_dir.glob("*.parquet"))
    for f in files:
        df = pd.read_parquet(f)
        if df.empty:
            continue
        if "chainlink_btc" in df.columns and "chainlink_price" not in df.columns:
            df.rename(columns={"chainlink_btc": "chainlink_price"}, inplace=True)
        if "chainlink_price" not in df.columns:
            continue

        # Completeness check
        if "window_end_ms" in df.columns:
            if df["ts_ms"].iloc[-1] < df["window_end_ms"].iloc[0]:
                continue
        else:
            if df["time_remaining_s"].iloc[-1] > MIN_FINAL_REMAINING_S:
                continue

        # Start gap check
        if ("window_start_ms" in df.columns and "window_end_ms" in df.columns
                and "time_remaining_s" in df.columns):
            window_dur_s = (df["window_end_ms"].iloc[0] - df["window_start_ms"].iloc[0]) / 1000
            if df["time_remaining_s"].iloc[0] < window_dur_s - MAX_START_GAP_S:
                continue

        sp = df["window_start_price"].dropna()
        if sp.empty:
            continue
        start_px = float(sp.iloc[0])
        final_px = float(df["chainlink_price"].iloc[-1])
        if pd.isna(start_px) or pd.isna(final_px) or start_px == 0:
            continue
        outcome = 1 if final_px >= start_px else 0
        yield df, start_px, outcome


# ── Collect (p_gbm, outcome) observations ──────────────────────────────────

def collect_observations(market_key: str, cfg, vol_lookback_s: int = 90,
                         max_z: float = 1.0, sample_every: int = 30):
    """Return list of (p_gbm, outcome, tau) tuples — pure GBM, no calibration."""
    data_dir = DATA_DIR / cfg.data_subdir
    if not data_dir.exists():
        return []

    observations = []
    n_windows = 0

    for df, start_px, outcome in load_complete_windows(data_dir):
        n_windows += 1
        prices = df["chainlink_price"].tolist()
        ts_list = df["ts_ms"].tolist()

        for idx in range(vol_lookback_s, len(df), sample_every):
            row = df.iloc[idx]
            tau = row["time_remaining_s"]
            if tau <= 0:
                continue

            lo = max(0, idx - vol_lookback_s)
            price_slice = prices[lo:idx + 1]
            ts_slice = ts_list[lo:idx + 1]

            # Skip gaps
            has_gap = False
            for k in range(1, len(ts_slice)):
                if ts_slice[k] - ts_slice[k - 1] > 5000:
                    has_gap = True
                    break
            if has_gap:
                continue

            sigma = _compute_vol_deduped(price_slice, ts_slice)
            if sigma <= 0:
                continue

            # Apply market config sigma bounds
            sigma = max(sigma, cfg.min_sigma)
            if cfg.max_sigma:
                sigma = min(sigma, cfg.max_sigma)

            delta = (row["chainlink_price"] - start_px) / start_px
            z_raw = delta / (sigma * math.sqrt(tau))
            z_capped = max(-max_z, min(max_z, z_raw))

            p_gbm = model_cdf(z_capped, cfg, sigma_per_s=sigma, tau=tau)
            observations.append((p_gbm, outcome, tau))

    print(f"  {market_key}: {n_windows} windows, {len(observations)} observations")
    return observations


# ── Calibration metrics ────────────────────────────────────────────────────

def compute_calibration(obs, n_bins=10):
    """Compute reliability diagram data + summary stats."""
    if not obs:
        return None

    preds = np.array([o[0] for o in obs])
    actuals = np.array([o[1] for o in obs])

    # Brier score
    brier = float(np.mean((preds - actuals) ** 2))

    # Log loss (clipped to avoid log(0))
    eps = 1e-8
    p_clip = np.clip(preds, eps, 1 - eps)
    logloss = -float(np.mean(actuals * np.log(p_clip) + (1 - actuals) * np.log(1 - p_clip)))

    # Binned calibration
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_freqs = []
    bin_counts = []
    bin_avg_pred = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (preds >= lo) & (preds <= hi)
        else:
            mask = (preds >= lo) & (preds < hi)
        n = mask.sum()
        bin_counts.append(int(n))
        if n > 0:
            bin_centers.append((lo + hi) / 2)
            bin_freqs.append(float(actuals[mask].mean()))
            bin_avg_pred.append(float(preds[mask].mean()))
        else:
            bin_centers.append((lo + hi) / 2)
            bin_freqs.append(np.nan)
            bin_avg_pred.append(np.nan)

    # ECE (Expected Calibration Error)
    total = len(preds)
    ece = 0.0
    for i in range(n_bins):
        if bin_counts[i] > 0 and not np.isnan(bin_freqs[i]):
            ece += (bin_counts[i] / total) * abs(bin_freqs[i] - bin_avg_pred[i])

    # Overall UP rate
    up_rate = float(actuals.mean())

    return {
        "brier": brier,
        "logloss": logloss,
        "ece": ece,
        "up_rate": up_rate,
        "n_obs": len(preds),
        "bin_centers": bin_centers,
        "bin_freqs": bin_freqs,
        "bin_counts": bin_counts,
        "bin_avg_pred": bin_avg_pred,
        "preds": preds,
        "actuals": actuals,
    }


def compute_calibration_by_tau(obs, tau_edges, n_bins=10):
    """Split observations by time-remaining buckets and compute calibration per bucket."""
    results = {}
    for i in range(len(tau_edges) - 1):
        lo, hi = tau_edges[i], tau_edges[i + 1]
        label = f"{int(lo)}-{int(hi)}s"
        bucket_obs = [(p, o, t) for p, o, t in obs if lo <= t < hi]
        if bucket_obs:
            results[label] = compute_calibration(bucket_obs)
    return results


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_all_markets(all_results: dict, output_path: str):
    """Create a multi-panel reliability diagram for all markets."""
    markets = [k for k in all_results if all_results[k] is not None]
    if not markets:
        print("No results to plot.")
        return

    n = len(markets)
    cols = min(n, 4)
    rows = math.ceil(n / cols)

    fig = plt.figure(figsize=(5 * cols, 6 * rows))
    gs = GridSpec(rows, cols, figure=fig, hspace=0.45, wspace=0.35)

    for idx, mkt in enumerate(markets):
        r = all_results[mkt]
        row, col = divmod(idx, cols)
        ax = fig.add_subplot(gs[row, col])

        # Reliability line
        valid = [(c, f, n, ap) for c, f, n, ap in
                 zip(r["bin_centers"], r["bin_freqs"], r["bin_counts"], r["bin_avg_pred"])
                 if not np.isnan(f) and n >= 5]
        if valid:
            avg_preds = [v[3] for v in valid]
            freqs = [v[1] for v in valid]
            counts = [v[2] for v in valid]
            ax.plot(avg_preds, freqs, "o-", color="steelblue", linewidth=2,
                    markersize=6, label="Model")

            # Size annotation
            for ap, fr, ct in zip(avg_preds, freqs, counts):
                ax.annotate(f"n={ct}", (ap, fr), textcoords="offset points",
                            xytext=(0, 8), fontsize=6, ha="center", color="gray")

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect")

        # Climatology line (base rate)
        ax.axhline(r["up_rate"], color="red", linestyle=":", alpha=0.5,
                    label=f"Base rate={r['up_rate']:.2f}")

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Predicted P(UP)")
        ax.set_ylabel("Observed frequency of UP")
        ax.set_title(f"{mkt}\nBrier={r['brier']:.4f}  ECE={r['ece']:.4f}  "
                      f"LogLoss={r['logloss']:.4f}\n"
                      f"N={r['n_obs']:,}  UP%={r['up_rate']:.1%}")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.suptitle("P(UP) Calibration: Pure GBM Model (No Bayesian Calibration)\n"
                 "X = avg predicted P(UP) per bin | Y = actual fraction that went UP",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_path}")
    plt.close()


def plot_tau_breakdown(all_results: dict, all_tau_results: dict, output_path: str):
    """Create tau-stratified reliability plots for the main markets."""
    main_markets = [k for k in ["BTC 15m", "BTC 5m", "ETH 15m", "ETH 5m"]
                    if k in all_tau_results and all_tau_results[k]]
    if not main_markets:
        return

    fig, axes = plt.subplots(1, len(main_markets), figsize=(6 * len(main_markets), 5))
    if len(main_markets) == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 4))

    for ax, mkt in zip(axes, main_markets):
        tau_res = all_tau_results[mkt]
        for i, (label, r) in enumerate(tau_res.items()):
            valid = [(ap, f) for ap, f, n in
                     zip(r["bin_avg_pred"], r["bin_freqs"], r["bin_counts"])
                     if not np.isnan(f) and n >= 5]
            if valid:
                ax.plot([v[0] for v in valid], [v[1] for v in valid],
                        "o-", color=colors[i % len(colors)], linewidth=1.5,
                        markersize=5, label=f"{label} (n={r['n_obs']:,})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Predicted P(UP)")
        ax.set_ylabel("Observed frequency of UP")
        ax.set_title(f"{mkt} — by time remaining")
        ax.legend(fontsize=7)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Calibration by Time Remaining (Pure GBM)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_prediction_distribution(all_results: dict, output_path: str):
    """Histogram of predicted probabilities, colored by actual outcome."""
    markets = [k for k in all_results if all_results[k] is not None]
    if not markets:
        return

    n = len(markets)
    cols = min(n, 4)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, mkt in enumerate(markets):
        r = all_results[mkt]
        row, col = divmod(idx, cols)
        ax = axes[row][col]

        up_mask = r["actuals"] == 1
        ax.hist(r["preds"][up_mask], bins=30, alpha=0.6, color="green",
                label=f"Actual UP ({up_mask.sum():,})", density=True)
        ax.hist(r["preds"][~up_mask], bins=30, alpha=0.6, color="red",
                label=f"Actual DOWN ({(~up_mask).sum():,})", density=True)
        ax.axvline(0.5, color="black", linestyle="--", alpha=0.5)
        ax.set_xlabel("Predicted P(UP)")
        ax.set_ylabel("Density")
        ax.set_title(mkt)
        ax.legend(fontsize=7)

    # Hide unused axes
    for idx in range(len(markets), rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Distribution of P(UP) predictions by actual outcome",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CALIBRATION ANALYSIS: Pure GBM P(UP) — No Bayesian Fusion")
    print("=" * 60)

    all_results = {}
    all_tau_results = {}

    for key, cfg in MARKET_CONFIGS.items():
        display = cfg.display_name
        print(f"\nProcessing {display} ({cfg.tail_mode})...")
        obs = collect_observations(key, cfg)
        if not obs:
            print(f"  No data for {display}")
            all_results[display] = None
            continue

        r = compute_calibration(obs, n_bins=10)
        all_results[display] = r

        # Tau breakdown
        if cfg.window_duration_s == 900:
            tau_edges = [0, 150, 300, 600, 900]
        else:
            tau_edges = [0, 60, 120, 180, 300]
        all_tau_results[display] = compute_calibration_by_tau(obs, tau_edges)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Market':<12} {'Tail':>10} {'N obs':>10} {'UP%':>8} "
          f"{'Brier':>8} {'ECE':>8} {'LogLoss':>8}")
    print("-" * 80)
    for mkt, r in all_results.items():
        if r is None:
            continue
        cfg_key = [k for k, c in MARKET_CONFIGS.items() if c.display_name == mkt][0]
        tail = MARKET_CONFIGS[cfg_key].tail_mode
        print(f"{mkt:<12} {tail:>10} {r['n_obs']:>10,} {r['up_rate']:>7.1%} "
              f"{r['brier']:>8.4f} {r['ece']:>8.4f} {r['logloss']:>8.4f}")

    # Reference: perfect uninformed model (always predict base rate)
    print("-" * 80)
    print("Reference: Brier of always-predict-0.5 = 0.2500")
    print("           Brier < 0.25 → model adds value vs coin flip")
    print("           ECE close to 0 → well-calibrated")

    # Print per-bin details for main markets
    for mkt in ["BTC 15m", "BTC 5m", "ETH 15m", "ETH 5m"]:
        if mkt not in all_results or all_results[mkt] is None:
            continue
        r = all_results[mkt]
        print(f"\n{'─' * 50}")
        print(f"  {mkt} — Bin Details")
        print(f"  {'Pred Range':<14} {'Avg Pred':>10} {'Actual UP%':>12} {'Count':>8}")
        for i in range(len(r["bin_centers"])):
            lo = max(0, r["bin_centers"][i] - 0.05)
            hi = min(1, r["bin_centers"][i] + 0.05)
            freq_str = f"{r['bin_freqs'][i]:.3f}" if not np.isnan(r['bin_freqs'][i]) else "N/A"
            avg_str = f"{r['bin_avg_pred'][i]:.3f}" if not np.isnan(r['bin_avg_pred'][i]) else "N/A"
            print(f"  [{lo:.2f}, {hi:.2f})  {avg_str:>10} {freq_str:>12} {r['bin_counts'][i]:>8,}")

    # Tau breakdown summaries
    for mkt in ["BTC 15m", "BTC 5m", "ETH 15m", "ETH 5m"]:
        if mkt not in all_tau_results or not all_tau_results[mkt]:
            continue
        print(f"\n{'─' * 50}")
        print(f"  {mkt} — By Time Remaining")
        print(f"  {'Tau Bucket':<14} {'N':>8} {'Brier':>8} {'ECE':>8} {'UP%':>8}")
        for label, r in all_tau_results[mkt].items():
            print(f"  {label:<14} {r['n_obs']:>8,} {r['brier']:>8.4f} "
                  f"{r['ece']:>8.4f} {r['up_rate']:>7.1%}")

    # Generate plots
    out_dir = Path(__file__).parent
    plot_all_markets(all_results, str(out_dir / "calibration_reliability.png"))
    plot_tau_breakdown(all_results, all_tau_results, str(out_dir / "calibration_by_tau.png"))
    plot_prediction_distribution(all_results, str(out_dir / "calibration_distributions.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
