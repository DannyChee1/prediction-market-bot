#!/usr/bin/env python3
"""
Optimize BTC 5m Kou jump-diffusion parameters by grid search.

Finds the (lambda, p_up, eta1, eta2) that minimize ECE on all complete
BTC 5m parquet windows, using the pure model CDF (no Bayesian calibration).
"""

import math
import itertools
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backtest import _compute_vol_deduped, norm_cdf, MIN_FINAL_REMAINING_S, MAX_START_GAP_S

DATA_DIR = Path(__file__).parent / "data" / "btc_5m"

# ── Preload all windows into memory ────────────────────────────────────────

@dataclass
class WindowData:
    """Pre-extracted data for one window."""
    observations: list  # [(delta, sigma, tau), ...]
    outcome: int        # 1=UP, 0=DOWN


def preload_windows(vol_lookback_s: int = 90, sample_every: int = 30):
    """Load all complete BTC 5m windows and extract (delta, sigma, tau) tuples."""
    files = sorted(DATA_DIR.glob("*.parquet"))
    windows = []

    for f in files:
        df = pd.read_parquet(f)
        if df.empty:
            continue
        if "chainlink_btc" in df.columns and "chainlink_price" not in df.columns:
            df.rename(columns={"chainlink_btc": "chainlink_price"}, inplace=True)
        if "chainlink_price" not in df.columns:
            continue

        # Completeness
        if "window_end_ms" in df.columns:
            if df["ts_ms"].iloc[-1] < df["window_end_ms"].iloc[0]:
                continue
        else:
            if df["time_remaining_s"].iloc[-1] > MIN_FINAL_REMAINING_S:
                continue

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

        prices = df["chainlink_price"].tolist()
        ts_list = df["ts_ms"].tolist()
        obs = []

        for idx in range(vol_lookback_s, len(df), sample_every):
            row = df.iloc[idx]
            tau = row["time_remaining_s"]
            if tau <= 0:
                continue

            lo = max(0, idx - vol_lookback_s)
            price_slice = prices[lo:idx + 1]
            ts_slice = ts_list[lo:idx + 1]

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

            # Apply BTC 5m sigma bounds
            sigma = max(sigma, 7e-05)

            delta = (row["chainlink_price"] - start_px) / start_px
            obs.append((delta, sigma, tau))

        if obs:
            windows.append(WindowData(observations=obs, outcome=outcome))

    return windows


def evaluate_params(windows, lam, p_up, eta1, eta2, max_z=1.0, n_bins=10):
    """Evaluate Kou params on preloaded windows. Returns (brier, ece, logloss)."""
    q_dn = 1.0 - p_up
    zeta = (p_up * eta1 / max(eta1 - 1, 0.01)
            + q_dn * eta2 / (eta2 + 1) - 1.0)

    preds = []
    actuals = []

    for w in windows:
        for delta, sigma, tau in w.observations:
            z_raw = delta / (sigma * math.sqrt(tau))
            z_capped = max(-max_z, min(max_z, z_raw))

            drift_z = -lam * zeta * math.sqrt(tau) / max(sigma, 1e-8)
            p_gbm = norm_cdf(z_capped + drift_z)

            preds.append(p_gbm)
            actuals.append(w.outcome)

    preds = np.array(preds)
    actuals = np.array(actuals)

    # Brier
    brier = float(np.mean((preds - actuals) ** 2))

    # Log loss
    eps = 1e-8
    p_clip = np.clip(preds, eps, 1 - eps)
    logloss = -float(np.mean(actuals * np.log(p_clip) + (1 - actuals) * np.log(1 - p_clip)))

    # ECE
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(preds)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (preds >= lo) & (preds < hi) if i < n_bins - 1 else (preds >= lo) & (preds <= hi)
        n = mask.sum()
        if n > 0:
            avg_pred = float(preds[mask].mean())
            avg_actual = float(actuals[mask].mean())
            ece += (n / total) * abs(avg_actual - avg_pred)

    return brier, ece, logloss


def get_bin_details(windows, lam, p_up, eta1, eta2, max_z=1.0, n_bins=10):
    """Get per-bin calibration details for a param set."""
    q_dn = 1.0 - p_up
    zeta = (p_up * eta1 / max(eta1 - 1, 0.01)
            + q_dn * eta2 / (eta2 + 1) - 1.0)

    preds = []
    actuals = []

    for w in windows:
        for delta, sigma, tau in w.observations:
            z_raw = delta / (sigma * math.sqrt(tau))
            z_capped = max(-max_z, min(max_z, z_raw))
            drift_z = -lam * zeta * math.sqrt(tau) / max(sigma, 1e-8)
            p_gbm = norm_cdf(z_capped + drift_z)
            preds.append(p_gbm)
            actuals.append(w.outcome)

    preds = np.array(preds)
    actuals = np.array(actuals)
    bin_edges = np.linspace(0, 1, n_bins + 1)

    rows = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (preds >= lo) & (preds < hi) if i < n_bins - 1 else (preds >= lo) & (preds <= hi)
        n = int(mask.sum())
        avg_pred = float(preds[mask].mean()) if n > 0 else float("nan")
        avg_actual = float(actuals[mask].mean()) if n > 0 else float("nan")
        rows.append((lo, hi, avg_pred, avg_actual, n))
    return rows, preds, actuals


def main():
    print("Loading BTC 5m windows...")
    windows = preload_windows()
    print(f"Loaded {len(windows)} complete windows, "
          f"{sum(len(w.observations) for w in windows)} observations")
    print(f"UP rate: {sum(w.outcome for w in windows) / len(windows):.1%}\n")

    # Current params for reference
    print("=" * 70)
    print("CURRENT PARAMS: lambda=0.100, p_up=0.526, eta1=1254.1, eta2=1200.5")
    b, e, l = evaluate_params(windows, 0.100, 0.526, 1254.1, 1200.5)
    print(f"  Brier={b:.4f}  ECE={e:.4f}  LogLoss={l:.4f}")
    print()

    # Also test lambda=0 (pure normal CDF, no jump correction) as baseline
    print("BASELINE: lambda=0 (pure Normal CDF, no jumps)")
    b0, e0, l0 = evaluate_params(windows, 0.0, 0.5, 1000, 1000)
    print(f"  Brier={b0:.4f}  ECE={e0:.4f}  LogLoss={l0:.4f}")
    print()

    # ── Phase 1: Coarse grid on lambda (most impactful param) ──────────
    print("=" * 70)
    print("PHASE 1: Coarse lambda sweep (p_up=0.526, eta1=1254.1, eta2=1200.5)")
    print("-" * 70)

    lambdas = [0.0, 0.001, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020,
               0.030, 0.050, 0.070, 0.100, 0.150]
    results = []
    for lam in lambdas:
        b, e, l = evaluate_params(windows, lam, 0.526, 1254.1, 1200.5)
        results.append((lam, b, e, l))
        print(f"  lambda={lam:.3f}  Brier={b:.4f}  ECE={e:.4f}  LogLoss={l:.4f}")

    best_lam = min(results, key=lambda r: r[2])
    print(f"\n  Best by ECE: lambda={best_lam[0]:.3f} (ECE={best_lam[2]:.4f})")

    # ── Phase 2: Fine grid around best lambda ──────────────────────────
    print("\n" + "=" * 70)
    center = best_lam[0]
    lo_lam = max(0, center - 0.005)
    hi_lam = center + 0.005
    fine_lambdas = np.linspace(lo_lam, hi_lam, 11)
    print(f"PHASE 2: Fine lambda sweep [{lo_lam:.4f} - {hi_lam:.4f}]")
    print("-" * 70)

    fine_results = []
    for lam in fine_lambdas:
        b, e, l = evaluate_params(windows, float(lam), 0.526, 1254.1, 1200.5)
        fine_results.append((float(lam), b, e, l))
        print(f"  lambda={lam:.4f}  Brier={b:.4f}  ECE={e:.4f}  LogLoss={l:.4f}")

    best_fine_lam = min(fine_results, key=lambda r: r[2])
    print(f"\n  Best by ECE: lambda={best_fine_lam[0]:.4f} (ECE={best_fine_lam[2]:.4f})")

    # ── Phase 3: Joint sweep of lambda + p_up ──────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 3: Joint lambda x p_up sweep")
    print("-" * 70)

    best_l = best_fine_lam[0]
    lam_range = [max(0, best_l - 0.003), best_l, min(0.15, best_l + 0.003)]
    p_up_range = [0.48, 0.49, 0.50, 0.51, 0.52, 0.526, 0.54, 0.55, 0.56]

    joint_results = []
    for lam in lam_range:
        for pu in p_up_range:
            b, e, l = evaluate_params(windows, lam, pu, 1254.1, 1200.5)
            joint_results.append((lam, pu, b, e, l))

    joint_results.sort(key=lambda r: r[3])
    print(f"  {'lambda':>8} {'p_up':>6} {'Brier':>8} {'ECE':>8} {'LogLoss':>8}")
    for lam, pu, b, e, l in joint_results[:15]:
        print(f"  {lam:>8.4f} {pu:>6.3f} {b:>8.4f} {e:>8.4f} {l:>8.4f}")

    best_joint = joint_results[0]
    print(f"\n  Best: lambda={best_joint[0]:.4f}, p_up={best_joint[1]:.3f} "
          f"(ECE={best_joint[3]:.4f})")

    # ── Phase 4: Sweep eta1/eta2 with best lambda/p_up ─────────────────
    print("\n" + "=" * 70)
    print("PHASE 4: eta1/eta2 sweep with best lambda/p_up")
    print("-" * 70)

    best_lam_final = best_joint[0]
    best_pup_final = best_joint[1]

    eta_range = [500, 800, 1000, 1200, 1254.1, 1500, 2000, 3000, 5000]
    eta_results = []
    for e1 in eta_range:
        for e2 in eta_range:
            b, e, l = evaluate_params(windows, best_lam_final, best_pup_final, e1, e2)
            eta_results.append((e1, e2, b, e, l))

    eta_results.sort(key=lambda r: r[3])
    print(f"  {'eta1':>8} {'eta2':>8} {'Brier':>8} {'ECE':>8} {'LogLoss':>8}")
    for e1, e2, b, e, l in eta_results[:15]:
        print(f"  {e1:>8.1f} {e2:>8.1f} {b:>8.4f} {e:>8.4f} {l:>8.4f}")

    best_eta = eta_results[0]

    # ── Final summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    final_lam = best_lam_final
    final_pup = best_pup_final
    final_e1 = best_eta[0]
    final_e2 = best_eta[1]

    print(f"\n  CURRENT:   lambda=0.100, p_up=0.526, eta1=1254.1, eta2=1200.5")
    b_old, e_old, l_old = evaluate_params(windows, 0.100, 0.526, 1254.1, 1200.5)
    print(f"             Brier={b_old:.4f}  ECE={e_old:.4f}  LogLoss={l_old:.4f}")

    print(f"\n  NO JUMPS:  lambda=0 (pure Normal CDF)")
    print(f"             Brier={b0:.4f}  ECE={e0:.4f}  LogLoss={l0:.4f}")

    print(f"\n  OPTIMIZED: lambda={final_lam:.4f}, p_up={final_pup:.3f}, "
          f"eta1={final_e1:.1f}, eta2={final_e2:.1f}")
    b_new, e_new, l_new = evaluate_params(windows, final_lam, final_pup, final_e1, final_e2)
    print(f"             Brier={b_new:.4f}  ECE={e_new:.4f}  LogLoss={l_new:.4f}")

    print(f"\n  Improvement: ECE {e_old:.4f} -> {e_new:.4f} "
          f"({(e_old - e_new) / e_old * 100:.1f}% reduction)")

    # ── Bin details comparison ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("BIN DETAILS: Old vs Optimized")
    print("=" * 70)

    rows_old, _, _ = get_bin_details(windows, 0.100, 0.526, 1254.1, 1200.5)
    rows_new, preds_new, actuals_new = get_bin_details(
        windows, final_lam, final_pup, final_e1, final_e2)

    print(f"\n  {'Bin':<14} {'Old Pred':>10} {'Old Act%':>10} {'New Pred':>10} "
          f"{'New Act%':>10} {'Count':>8}")
    for (lo, hi, op, oa, _), (_, _, np_, na, n) in zip(rows_old, rows_new):
        op_s = f"{op:.3f}" if not math.isnan(op) else "N/A"
        oa_s = f"{oa:.3f}" if not math.isnan(oa) else "N/A"
        np_s = f"{np_:.3f}" if not math.isnan(np_) else "N/A"
        na_s = f"{na:.3f}" if not math.isnan(na) else "N/A"
        print(f"  [{lo:.2f},{hi:.2f})  {op_s:>10} {oa_s:>10} {np_s:>10} {na_s:>10} {n:>8}")

    # ── Plot: lambda vs ECE ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Lambda sweep
    lams_all = [r[0] for r in results]
    eces_all = [r[2] for r in results]
    axes[0].plot(lams_all, eces_all, "o-", color="steelblue", linewidth=2)
    axes[0].axvline(final_lam, color="green", linestyle="--", alpha=0.7,
                     label=f"Optimal={final_lam:.4f}")
    axes[0].axvline(0.100, color="red", linestyle="--", alpha=0.7,
                     label="Current=0.100")
    axes[0].set_xlabel("kou_lambda")
    axes[0].set_ylabel("ECE")
    axes[0].set_title("Lambda vs ECE (BTC 5m)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Reliability: old vs new
    for ax_idx, (label, lam, pu, e1, e2, color) in enumerate([
        ("Current", 0.100, 0.526, 1254.1, 1200.5, "red"),
        ("Optimized", final_lam, final_pup, final_e1, final_e2, "green"),
    ]):
        rows, preds, actuals = get_bin_details(windows, lam, pu, e1, e2)
        ax = axes[ax_idx + 1]
        valid = [(ap, aa, n) for (lo, hi, ap, aa, n) in rows
                 if not math.isnan(ap) and not math.isnan(aa) and n >= 5]
        if valid:
            ax.plot([v[0] for v in valid], [v[1] for v in valid],
                    "o-", color=color, linewidth=2, markersize=6)
            for ap, aa, n in valid:
                ax.annotate(f"n={n}", (ap, aa), textcoords="offset points",
                            xytext=(0, 8), fontsize=6, ha="center", color="gray")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Predicted P(UP)")
        ax.set_ylabel("Actual UP%")
        brier_val = evaluate_params(windows, lam, pu, e1, e2)[0]
        ece_val = evaluate_params(windows, lam, pu, e1, e2)[1]
        ax.set_title(f"{label}\nBrier={brier_val:.4f} ECE={ece_val:.4f}")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.suptitle("BTC 5m Kou Parameter Optimization", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("kou_5m_optimization.png", dpi=150, bbox_inches="tight")
    print("\nSaved: kou_5m_optimization.png")


if __name__ == "__main__":
    main()
