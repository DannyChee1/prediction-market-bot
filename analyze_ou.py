#!/usr/bin/env python3
"""
OU vs GBM test: Is BTC price mean-reverting within 15m/5m windows?

This is the most fundamental question about the model. If prices within
windows are OU (mean-reverting to start price), the GBM z-score signal
points in the WRONG direction for large z values — you should bet against
large z-scores, not with them.

Tests run:
  1. ADF stationarity test on fractional deviation (X_t = price/start - 1)
     Stationary = mean-reverting = OU. Non-stationary = random walk = GBM.
  2. OLS regression: ΔX = α + β*X_{t-1}  →  β < 0 means OU
     Half-life = ln(2) / |β| (in seconds)
  3. Empirical: bin by early z-score magnitude, track mean z at each tau
     If z decays toward 0 over time → OU. If it stays → GBM.
  4. Directional accuracy: betting WITH vs AGAINST large z-scores
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.stattools import adfuller

DATA_DIR = Path("data")
VOL_LOOKBACK_S = 90
MAX_Z = 1.5


def compute_sigma(prices, timestamps):
    changes = []
    for i, p in enumerate(prices):
        if p > 0 and (not changes or p != changes[-1][1]):
            changes.append((timestamps[i], p))
    if len(changes) < 3:
        return 0.0
    log_rets = []
    for j in range(1, len(changes)):
        dt = (changes[j][0] - changes[j-1][0]) / 1000.0
        if dt > 0:
            lr = math.log(changes[j][1] / changes[j-1][1])
            log_rets.append(lr / math.sqrt(dt))
    if len(log_rets) < 2:
        return 0.0
    return float(np.std(log_rets, ddof=1))


def norm_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def load_windows(subdir: Path) -> list[pd.DataFrame]:
    windows = []
    for f in sorted(subdir.glob("*.parquet")):
        df = pd.read_parquet(f)
        if df.empty:
            continue
        if "chainlink_price" not in df.columns and "chainlink_btc" not in df.columns:
            continue
        if "chainlink_btc" in df.columns and "chainlink_price" not in df.columns:
            df = df.rename(columns={"chainlink_btc": "chainlink_price"})
        windows.append(df)
    return windows


def ou_params_from_window(df: pd.DataFrame) -> dict | None:
    """Fit OU parameters from a single window using OLS on ΔX = α + β*X_{t-1}."""
    start_px = df["window_start_price"].dropna()
    if start_px.empty:
        return None
    start_px = float(start_px.iloc[0])
    if start_px == 0:
        return None

    prices = df["chainlink_price"].values
    # Fractional deviation from start price
    X = (prices - start_px) / start_px

    # ΔX regression
    dX = np.diff(X)
    X_lag = X[:-1]

    if len(dX) < 20:
        return None

    # OLS: ΔX = α + β * X_lag
    A = np.column_stack([np.ones(len(X_lag)), X_lag])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, dX, rcond=None)
    except Exception:
        return None

    alpha, beta = coeffs[0], coeffs[1]

    # OU mean reversion: β should be negative
    # Mean = -α/β, speed = -β (per second at 1Hz data)
    if beta >= 0:
        return {"beta": beta, "mean_reverting": False, "half_life_s": None}

    theta = -beta  # mean reversion speed per second
    half_life_s = math.log(2) / theta

    # ADF test on X series
    adf_result = adfuller(X, maxlag=10, autolag="AIC")
    adf_pvalue = adf_result[1]

    return {
        "beta": beta,
        "theta": theta,
        "half_life_s": half_life_s,
        "adf_pvalue": adf_pvalue,
        "mean_reverting": adf_pvalue < 0.05,
        "ou_mean": -alpha / beta if beta != 0 else 0.0,
    }


def z_trajectory_analysis(windows: list[pd.DataFrame]) -> pd.DataFrame:
    """
    For each window, compute z-score at multiple tau points.
    Returns DataFrame of (window_id, tau, z, outcome_up) for trajectory analysis.
    """
    TAU_POINTS = [800, 700, 600, 500, 400, 300, 200, 100, 60, 30]
    rows = []

    for win_id, df in enumerate(windows):
        start_px = df["window_start_price"].dropna()
        if start_px.empty:
            continue
        start_px = float(start_px.iloc[0])
        if start_px == 0:
            continue

        final_px = float(df["chainlink_price"].iloc[-1])
        outcome_up = 1 if final_px >= start_px else 0

        prices = df["chainlink_price"].tolist()
        ts_list = df["ts_ms"].tolist()
        tau_all = df["time_remaining_s"].tolist()

        for target_tau in TAU_POINTS:
            best_idx = min(range(len(tau_all)),
                           key=lambda i: abs(tau_all[i] - target_tau))
            if abs(tau_all[best_idx] - target_tau) > 30:
                continue

            lo = max(0, best_idx - VOL_LOOKBACK_S)
            sigma = compute_sigma(prices[lo:best_idx+1], ts_list[lo:best_idx+1])
            if sigma <= 0:
                continue

            current_px = prices[best_idx]
            delta = (current_px - start_px) / start_px
            actual_tau = tau_all[best_idx]
            z_raw = delta / (sigma * math.sqrt(actual_tau))
            z = max(-MAX_Z, min(MAX_Z, z_raw))

            rows.append({
                "win_id": win_id,
                "tau": target_tau,
                "z": z,
                "z_abs": abs(z),
                "outcome_up": outcome_up,
            })

    return pd.DataFrame(rows)


def main():
    DIRS_15M = ["btc_15m", "eth_15m", "sol_15m", "xrp_15m"]
    DIRS_5M  = ["btc_5m",  "eth_5m",  "sol_5m",  "xrp_5m"]

    print("=" * 70)
    print("1. ADF + OU PARAMETER ESTIMATION")
    print("   Testing if price deviations from window start are mean-reverting")
    print("=" * 70)

    all_ou = []
    for dirname in DIRS_15M + DIRS_5M:
        d = DATA_DIR / dirname
        if not d.exists():
            continue
        windows = load_windows(d)
        results = [ou_params_from_window(w) for w in windows]
        results = [r for r in results if r is not None]

        n_mr = sum(1 for r in results if r["mean_reverting"])
        n_total = len(results)
        half_lives = [r["half_life_s"] for r in results
                      if r["mean_reverting"] and r["half_life_s"] is not None]
        betas = [r["beta"] for r in results]

        print(f"\n  {dirname}  ({n_total} windows)")
        print(f"    Mean-reverting (ADF p<0.05): {n_mr}/{n_total} = {n_mr/n_total:.1%}")
        print(f"    Avg beta (ΔX ~ β*X_lag):     {np.mean(betas):+.5f}  "
              f"({'<0 = OU' if np.mean(betas) < 0 else '>0 = explosive'})")
        if half_lives:
            print(f"    Median half-life:            {np.median(half_lives):.0f}s  "
                  f"(range {np.percentile(half_lives,25):.0f}–{np.percentile(half_lives,75):.0f}s)")
        all_ou.extend(results)

    all_betas = [r["beta"] for r in all_ou]
    print(f"\n  OVERALL: avg beta = {np.mean(all_betas):+.5f}")
    if np.mean(all_betas) < 0:
        print("  → Prices are mean-reverting within windows (OU dominates GBM)")
        print("  → Large z-scores tend to REVERT — model should discount them")
    else:
        print("  → Prices are not significantly mean-reverting (GBM holds)")
        print("  → z-score direction is a valid signal")

    print("\n" + "=" * 70)
    print("2. Z-SCORE TRAJECTORY: Does a large early z decay toward 0?")
    print("=" * 70)

    # Use BTC 15m for trajectory analysis (largest dataset)
    d = DATA_DIR / "btc_15m"
    if d.exists():
        windows = load_windows(d)
        traj = z_trajectory_analysis(windows)

        print(f"\n  BTC 15m — {len(traj['win_id'].unique())} windows")
        print(f"  {'tau':>6}  {'n':>5}  {'mean |z|':>9}  {'std |z|':>9}  "
              f"{'z>0 wins UP':>12}  {'z<0 wins DOWN':>14}")

        for tau in [800, 600, 400, 200, 60]:
            g = traj[traj["tau"] == tau]
            if len(g) < 10:
                continue
            up_correct = g[(g["z"] > 0.1)]["outcome_up"].mean() if len(g[g["z"] > 0.1]) > 0 else float("nan")
            dn_correct = (1 - g[(g["z"] < -0.1)]["outcome_up"].mean()) if len(g[g["z"] < -0.1]) > 0 else float("nan")
            print(f"  {tau:>6}s  {len(g):>5}  {g['z_abs'].mean():>9.4f}  "
                  f"{g['z_abs'].std():>9.4f}  {up_correct:>12.3f}  {dn_correct:>14.3f}")

    print("\n" + "=" * 70)
    print("3. DOES z DECAY? — Tracking z-score across tau for the same window")
    print("=" * 70)

    d = DATA_DIR / "btc_15m"
    if d.exists():
        windows = load_windows(d)
        traj = z_trajectory_analysis(windows)

        # For windows with large z at tau=700, what is mean |z| at tau=100?
        early = traj[traj["tau"] == 700][["win_id", "z", "z_abs"]].rename(
            columns={"z": "z_early", "z_abs": "z_abs_early"})
        late = traj[traj["tau"] == 100][["win_id", "z", "z_abs"]].rename(
            columns={"z": "z_late", "z_abs": "z_abs_late"})
        merged = early.merge(late, on="win_id")

        if len(merged) > 20:
            # Bin by early |z|
            merged["early_bin"] = pd.cut(merged["z_abs_early"],
                                         bins=[0, 0.3, 0.6, 0.9, 1.2, 1.5],
                                         labels=["0-0.3", "0.3-0.6", "0.6-0.9", "0.9-1.2", "1.2-1.5"])
            print(f"\n  Early z (tau=700s) → Late z (tau=100s) trajectory:")
            print(f"  {'early |z| bin':15}  {'n':>5}  {'mean early |z|':>14}  "
                  f"{'mean late |z|':>13}  {'decay ratio':>12}  {'sign preserved':>14}")
            for bin_label, grp in merged.groupby("early_bin", observed=True):
                if len(grp) < 5:
                    continue
                decay = grp["z_abs_late"].mean() / grp["z_abs_early"].mean()
                sign_ok = (np.sign(grp["z_early"]) == np.sign(grp["z_late"])).mean()
                print(f"  {str(bin_label):15}  {len(grp):>5}  "
                      f"{grp['z_abs_early'].mean():>14.4f}  "
                      f"{grp['z_abs_late'].mean():>13.4f}  "
                      f"{decay:>12.3f}  {sign_ok:>14.3f}")
            print()
            print("  decay ratio < 1.0 = z reverts (OU)")
            print("  decay ratio ≈ 1.0 = z persists (GBM)")
            print("  sign preserved = direction of z stays consistent")

    print("\n" + "=" * 70)
    print("4. BETTING WITH vs AGAINST LARGE Z-SCORES")
    print("   (does signal direction improve with early entry restriction?)")
    print("=" * 70)

    for dirname in ["btc_15m", "btc_5m"]:
        d = DATA_DIR / dirname
        if not d.exists():
            continue
        windows = load_windows(d)
        traj = z_trajectory_analysis(windows)

        print(f"\n  {dirname}:")
        for min_z in [0.3, 0.6, 1.0]:
            for tau in [600, 300, 100]:
                g = traj[traj["tau"] == tau]
                g_up = g[g["z"] >= min_z]
                g_dn = g[g["z"] <= -min_z]
                if len(g_up) + len(g_dn) < 20:
                    continue
                # WITH signal
                acc_with = (
                    (g_up["outcome_up"].sum() + (1 - g_dn["outcome_up"]).sum()) /
                    (len(g_up) + len(g_dn))
                ) if len(g_up) + len(g_dn) > 0 else float("nan")
                # AGAINST signal
                acc_against = 1.0 - acc_with
                print(f"    |z|>={min_z:.1f} tau={tau:3d}s  "
                      f"n={len(g_up)+len(g_dn):4d}  "
                      f"WITH={acc_with:.3f}  AGAINST={acc_against:.3f}  "
                      f"{'→ BET WITH' if acc_with > 0.53 else '→ BET AGAINST' if acc_against > 0.53 else '→ COIN FLIP'}")


if __name__ == "__main__":
    main()
