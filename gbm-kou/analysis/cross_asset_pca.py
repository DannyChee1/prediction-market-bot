#!/usr/bin/env python3
"""
Cross-asset PCA: BTC / ETH / SOL / XRP z-score decomposition.

At each tau checkpoint within a window, we have a z-score for each asset.
PC1 captures the market-wide crypto beta factor.
PC1-residuals are asset-specific divergence signals.

Tests:
  1. Variance explained by each PC
  2. PC1 loadings (which assets drive the market factor?)
  3. Directional accuracy: raw z vs PC1 vs asset-specific residual
  4. Correlation matrix of z-scores across assets

Usage:
    python3 cross_asset_pca.py
    python3 cross_asset_pca.py --timeframe 5m
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_DIR      = Path("data")
VOL_LOOKBACK  = 90   # seconds
MAX_Z         = 1.5
TAU_CHECKPOINTS = [750, 600, 450, 300, 150, 60]

ASSETS_15M = ["btc_15m", "eth_15m", "sol_15m", "xrp_15m"]
ASSETS_5M  = ["btc_5m",  "eth_5m",  "sol_5m",  "xrp_5m"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_sigma(prices: list[float], timestamps: list[int]) -> float:
    changes: list[tuple[int, float]] = []
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


def load_windows(subdir: Path) -> dict[int, pd.DataFrame]:
    """Load parquet files keyed by window_start_ms."""
    result: dict[int, pd.DataFrame] = {}
    for f in sorted(subdir.glob("*.parquet")):
        df = pd.read_parquet(f)
        if df.empty:
            continue
        if "chainlink_btc" in df.columns and "chainlink_price" not in df.columns:
            df.rename(columns={"chainlink_btc": "chainlink_price"}, inplace=True)
        if "chainlink_price" not in df.columns:
            continue
        if "window_start_ms" not in df.columns:
            continue
        wstart = int(df["window_start_ms"].iloc[0])
        result[wstart] = df
    return result


def z_at_tau(
    df: pd.DataFrame,
    target_tau: float,
    start_px: float,
) -> float | None:
    """Compute z-score at a specific tau checkpoint."""
    tau_all  = df["time_remaining_s"].tolist()
    prices   = df["chainlink_price"].tolist()
    ts_list  = df["ts_ms"].tolist()

    best_idx = min(range(len(tau_all)), key=lambda i: abs(tau_all[i] - target_tau))
    if abs(tau_all[best_idx] - target_tau) > 30:
        return None

    lo    = max(0, best_idx - VOL_LOOKBACK)
    sigma = compute_sigma(prices[lo:best_idx+1], ts_list[lo:best_idx+1])
    if sigma <= 0:
        return None

    current_px = prices[best_idx]
    actual_tau = tau_all[best_idx]
    delta  = (current_px - start_px) / start_px
    z_raw  = delta / (sigma * math.sqrt(actual_tau))
    return float(max(-MAX_Z, min(MAX_Z, z_raw)))


# ── Main ──────────────────────────────────────────────────────────────────────

def build_aligned_matrix(
    asset_dirs: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build (N, K) z-score matrix where rows = (window, tau) and
    cols = assets.  Returns (X, outcomes, col_names).

    Only includes rows where all K assets have valid z-scores at that tau.
    """
    # Load all windows per asset
    asset_windows: dict[str, dict[int, pd.DataFrame]] = {}
    for name in asset_dirs:
        d = DATA_DIR / name
        if d.exists():
            asset_windows[name] = load_windows(d)
        else:
            asset_windows[name] = {}

    # Find window_start_ms values present in ALL assets
    common_starts = set(asset_windows[asset_dirs[0]].keys())
    for name in asset_dirs[1:]:
        common_starts &= set(asset_windows[name].keys())

    rows_X:   list[list[float]] = []
    rows_out: list[int]         = []

    for wstart in sorted(common_starts):
        # Compute outcome for first asset (BTC) — use its own price
        df0 = asset_windows[asset_dirs[0]][wstart]
        sp0 = df0["window_start_price"].dropna()
        if sp0.empty:
            continue
        sp0 = float(sp0.iloc[0])
        if sp0 == 0:
            continue
        fp0 = float(df0["chainlink_price"].iloc[-1])
        outcome = 1 if fp0 >= sp0 else 0

        for target_tau in TAU_CHECKPOINTS:
            zs: list[float] = []
            for name in asset_dirs:
                df = asset_windows[name][wstart]
                sp = df["window_start_price"].dropna()
                if sp.empty:
                    break
                sp = float(sp.iloc[0])
                if sp == 0:
                    break
                z = z_at_tau(df, target_tau, sp)
                if z is None:
                    break
                zs.append(z)
            else:
                if len(zs) == len(asset_dirs):
                    rows_X.append(zs)
                    rows_out.append(outcome)

    X   = np.array(rows_X,  dtype=np.float64)
    out = np.array(rows_out, dtype=np.int32)
    return X, out, asset_dirs


def directional_accuracy(z: np.ndarray, outcome: np.ndarray, min_z: float) -> tuple[float, int]:
    mask = np.abs(z) >= min_z
    z_m  = z[mask]
    o_m  = outcome[mask]
    if len(z_m) == 0:
        return float("nan"), 0
    correct = np.sum((z_m > 0) == (o_m == 1))
    return float(correct) / len(z_m), len(z_m)


def find_best_asset_group(asset_dirs: list[str]) -> tuple[list[str], int]:
    """Find the largest group of assets with at least 50 common windows."""
    from itertools import combinations

    # Load window start sets (fast: read only first row per file)
    window_sets: dict[str, set[int]] = {}
    for name in asset_dirs:
        d = DATA_DIR / name
        if not d.exists():
            continue
        s: set[int] = set()
        for f in sorted(d.glob("*.parquet")):
            try:
                df = pd.read_parquet(f, columns=["window_start_ms"])
                s.add(int(df["window_start_ms"].iloc[0]))
            except Exception:
                pass
        if s:
            window_sets[name] = s

    available = list(window_sets.keys())
    if len(available) < 2:
        return available, 0

    best_group: list[str] = []
    best_n = 0
    # Try largest groups first
    for size in range(len(available), 1, -1):
        for combo in combinations(available, size):
            common = window_sets[combo[0]].copy()
            for a in combo[1:]:
                common &= window_sets[a]
            if len(common) > best_n:
                best_n = len(common)
                best_group = list(combo)
        if best_n >= 50:
            break

    return best_group, best_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", choices=["15m", "5m"], default="15m")
    parser.add_argument("--assets", nargs="+", default=None,
                        help="Specific assets to include (overrides auto-detection)")
    args = parser.parse_args()

    asset_dirs = ASSETS_15M if args.timeframe == "15m" else ASSETS_5M

    if args.assets:
        present = [a for a in args.assets if (DATA_DIR / a).exists()]
    else:
        best_group, n_common = find_best_asset_group(asset_dirs)
        if n_common < 20:
            print(f"No asset group with ≥20 common windows found.")
            print(f"Best: {best_group} with {n_common} windows")
            print("Tip: Record multiple assets simultaneously for cross-asset PCA.")
            return
        present = best_group
        print(f"Auto-selected {len(present)} assets with {n_common} common windows: {present}")

    if len(present) < 2:
        print(f"Need at least 2 asset dirs; found: {present}")
        return

    print(f"\nLoading {args.timeframe} data for: {present}")
    X, outcomes, labels = build_aligned_matrix(present)
    if X.ndim < 2 or X.shape[0] < 20:
        print(f"Insufficient aligned data: shape={X.shape}")
        return
    N, K = X.shape
    print(f"Aligned matrix: {N} observations × {K} assets")
    print(f"Positive rate (UP): {outcomes.mean():.1%}\n")

    if N < 20:
        print("Not enough aligned windows — need more data.")
        return

    # ── 1. Correlation matrix ─────────────────────────────────────────────────
    print("=" * 60)
    print("1. Z-SCORE CORRELATION MATRIX")
    print("=" * 60)
    corr = np.corrcoef(X.T)
    header = "".join(f"{a[:5]:>9}" for a in present)
    print(f"{'':>9}{header}")
    for i, a in enumerate(present):
        row = "".join(f"{corr[i,j]:>9.3f}" for j in range(K))
        print(f"{a[:8]:>9}{row}")

    # ── 2. PCA ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. PCA — VARIANCE EXPLAINED")
    print("=" * 60)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    pca    = PCA(n_components=K)
    pca.fit(X_sc)

    print(f"\n  {'PC':>4}  {'var_expl':>9}  {'cumulative':>11}  loadings")
    cumvar = 0.0
    for i, (var, comp) in enumerate(zip(pca.explained_variance_ratio_, pca.components_)):
        cumvar += var
        loading_str = "  ".join(f"{present[j][:5]}={comp[j]:+.3f}" for j in range(K))
        print(f"  PC{i+1:>2}  {var:>9.3f}  {cumvar:>11.3f}  {loading_str}")

    # ── 3. Directional accuracy: raw z vs PC1 vs PC1-residual ────────────────
    print("\n" + "=" * 60)
    print("3. DIRECTIONAL ACCURACY")
    print("=" * 60)

    # Project onto PCs
    X_pc  = pca.transform(X_sc)  # (N, K)
    pc1   = X_pc[:, 0]           # market-wide factor
    pc1_sign_flip = -1.0 if pca.components_[0, 0] < 0 else 1.0
    pc1  *= pc1_sign_flip         # ensure PC1 positive = market up

    # Per-asset residuals after removing PC1
    pc1_contrib = np.outer(pc1, pca.components_[0]) * pc1_sign_flip
    residuals   = X_sc - pc1_contrib  # shape (N, K)

    print(f"\n  min_z threshold: signals where |z|>=min_z are taken")
    print(f"  {'signal':>22}  {'min_z':>6}  {'n':>6}  {'accuracy':>9}")

    for min_z in [0.3, 0.6, 1.0]:
        # Raw BTC z-score
        acc, n = directional_accuracy(X[:, 0], outcomes, min_z)
        print(f"  {'BTC raw z':>22}  {min_z:>6.1f}  {n:>6}  {acc:>9.3f}")

        # PC1
        acc_pc1, n_pc1 = directional_accuracy(pc1, outcomes, min_z)
        print(f"  {'PC1 (market factor)':>22}  {min_z:>6.1f}  {n_pc1:>6}  {acc_pc1:>9.3f}")

        # BTC residual (idiosyncratic)
        acc_r, n_r = directional_accuracy(residuals[:, 0], outcomes, min_z)
        print(f"  {'BTC residual (idio)':>22}  {min_z:>6.1f}  {n_r:>6}  {acc_r:>9.3f}")
        print()

    # ── 4. PC1 as leading indicator: does PC1 of other assets predict BTC? ───
    print("=" * 60)
    print("4. CROSS-ASSET LEADING: does PC1 predict BTC outcome?")
    print("   (positive = PC1 aligns with BTC; useful as corroborating signal)")
    print("=" * 60)

    btc_z = X[:, 0]  # raw BTC z-score
    for min_z in [0.3, 0.6]:
        # Filter: PC1 and BTC agree on direction
        agree_mask = (np.sign(pc1) == np.sign(btc_z)) & (np.abs(pc1) >= min_z)
        disagree_mask = (np.sign(pc1) != np.sign(btc_z)) & (np.abs(pc1) >= min_z)
        if agree_mask.sum() > 10:
            acc_agree = ((btc_z[agree_mask] > 0) == (outcomes[agree_mask] == 1)).mean()
            print(f"  |PC1|>={min_z:.1f} + BTC agree  : "
                  f"n={agree_mask.sum():4d}  BTC accuracy={acc_agree:.3f}")
        if disagree_mask.sum() > 10:
            acc_dis = ((btc_z[disagree_mask] > 0) == (outcomes[disagree_mask] == 1)).mean()
            print(f"  |PC1|>={min_z:.1f} + BTC DISAGREE: "
                  f"n={disagree_mask.sum():4d}  BTC accuracy={acc_dis:.3f}")

    print("\nConclusion:")
    pc1_btc_corr = float(np.corrcoef(pc1, btc_z)[0, 1])
    print(f"  corr(PC1, BTC_z) = {pc1_btc_corr:.4f}")
    if abs(pc1_btc_corr) > 0.7:
        print("  → PC1 closely tracks BTC z-score. Little additive value beyond BTC alone.")
        print("    Residuals may carry idiosyncratic signal worth exploring.")
    else:
        print("  → PC1 diverges from BTC z-score. Cross-asset factor provides "
              "independent information.")


if __name__ == "__main__":
    main()
