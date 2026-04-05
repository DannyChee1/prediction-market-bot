#!/usr/bin/env python3
"""
2-state HMM regime detection on prediction-market windows.

Each window is characterised by observable features (no future leakage).
The HMM captures latent market regime transitions across consecutive windows.

State interpretation emerges from data:
  - High-vol, large early-z state  → trending (z signal reliable)
  - Low-vol, small early-z state   → choppy/mean-reverting (z less reliable)

Input features (all observable before outcome):
  - log_sigma    : realised vol from first half of window
  - vol_regime   : recent/baseline vol ratio (regime of vol)
  - z_early_abs  : |z| at an early tau (≥600s remaining) — no future info
  - hour_sin/cos : time-of-day cyclical encoding

Analysis features (computed after for characterisation only):
  - signal_accuracy : did z direction match outcome?

Usage:
    python3 hmm_regime.py
    python3 hmm_regime.py --asset btc_15m --n-states 2
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

DATA_DIR      = Path("data")
VOL_LOOKBACK  = 90
# "Early" tau: use first available checkpoint ≥600s (window has ≥600s left)
EARLY_TAU     = 700   # target tau for early z-score
MAX_Z         = 1.5


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


def window_features(df: pd.DataFrame) -> dict | None:
    """
    Extract feature vector for one window.

    All input features are observable before the window ends.
    Outcome is computed for analysis only — never used as an HMM input.
    """
    if "chainlink_btc" in df.columns and "chainlink_price" not in df.columns:
        df = df.rename(columns={"chainlink_btc": "chainlink_price"})
    if "chainlink_price" not in df.columns:
        return None

    start_px_series = df["window_start_price"].dropna()
    if start_px_series.empty:
        return None
    start_px = float(start_px_series.iloc[0])
    if start_px == 0:
        return None

    prices  = df["chainlink_price"].tolist()
    ts_list = df["ts_ms"].tolist()
    tau_all = df["time_remaining_s"].tolist()

    # ── Outcome (analysis only, not used as HMM input) ────────────────────────
    final_px   = prices[-1]
    outcome_up = 1 if final_px >= start_px else 0

    # ── Feature 1: sigma from first half of window ────────────────────────────
    half = len(prices) // 2
    sigma_first = compute_sigma(prices[:half], ts_list[:half])
    if sigma_first <= 0:
        return None

    # ── Feature 2: vol regime ratio ───────────────────────────────────────────
    sigma_full = compute_sigma(prices, ts_list)
    vol_regime = sigma_full / sigma_first if (sigma_first > 0 and sigma_full > 0) else 1.0

    # ── Feature 3: early z-score at tau ≈ EARLY_TAU (no future info) ─────────
    best_idx = min(range(len(tau_all)), key=lambda i: abs(tau_all[i] - EARLY_TAU))
    if abs(tau_all[best_idx] - EARLY_TAU) > 60:
        return None   # window too short to have early signal
    lo     = max(0, best_idx - VOL_LOOKBACK)
    sigma_e = compute_sigma(prices[lo:best_idx+1], ts_list[lo:best_idx+1])
    if sigma_e <= 0:
        return None
    actual_tau_e = tau_all[best_idx]
    if actual_tau_e <= 0:
        return None
    delta_e  = (prices[best_idx] - start_px) / start_px
    z_early  = delta_e / (sigma_e * math.sqrt(actual_tau_e))
    z_early  = max(-MAX_Z, min(MAX_Z, z_early))

    # ── Feature 4: hour of day (cyclic) ──────────────────────────────────────
    ts_first = ts_list[0]
    hour = pd.Timestamp(ts_first, unit="ms", tz="UTC").hour
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)

    # ── Analysis-only: final signal accuracy ─────────────────────────────────
    # Compute z at tau≈60s (late signal) for directional accuracy analysis
    best_late = min(range(len(tau_all)), key=lambda i: abs(tau_all[i] - 60))
    lo_l = max(0, best_late - VOL_LOOKBACK)
    sigma_l = compute_sigma(prices[lo_l:best_late+1], ts_list[lo_l:best_late+1])
    if sigma_l > 0 and tau_all[best_late] > 0:
        delta_l  = (prices[best_late] - start_px) / start_px
        z_late   = delta_l / (sigma_l * math.sqrt(tau_all[best_late]))
        z_late   = max(-MAX_Z, min(MAX_Z, z_late))
        sig_correct = int((z_late > 0) == (outcome_up == 1))
    else:
        sig_correct = -1  # unknown

    return {
        # HMM input features
        "log_sigma":    math.log(sigma_first + 1e-10),
        "vol_regime":   vol_regime,
        "z_early_abs":  abs(z_early),
        "hour_sin":     hour_sin,
        "hour_cos":     hour_cos,
        # Analysis-only
        "z_early":      z_early,
        "outcome_up":   outcome_up,
        "sig_correct":  sig_correct,   # 1=correct, 0=wrong, -1=unknown
        "window_ts":    ts_first,
    }


def load_all_windows(subdir: Path) -> list[dict]:
    rows = []
    for f in sorted(subdir.glob("*.parquet")):
        df = pd.read_parquet(f)
        if df.empty:
            continue
        feat = window_features(df)
        if feat:
            rows.append(feat)
    rows.sort(key=lambda r: r["window_ts"])
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset",    default="btc_15m")
    parser.add_argument("--n-states", type=int, default=2)
    args = parser.parse_args()

    d = DATA_DIR / args.asset
    if not d.exists():
        print(f"Directory not found: {d}")
        return

    print(f"Loading {args.asset} windows...")
    windows = load_all_windows(d)
    print(f"  {len(windows)} windows with valid early z-scores")
    if len(windows) < 30:
        print("  Not enough windows for HMM (need ≥30).")
        return

    # Input features — no outcome, no look-ahead
    INPUT_COLS = ["log_sigma", "vol_regime", "z_early_abs", "hour_sin", "hour_cos"]
    X_raw = np.array([[w[c] for c in INPUT_COLS] for w in windows], dtype=np.float64)
    outcomes    = np.array([w["outcome_up"]  for w in windows], dtype=np.int32)
    sig_correct = np.array([w["sig_correct"] for w in windows], dtype=np.int32)

    # Walk-forward: fit scaler only on train portion to avoid data leakage
    train_n = int(len(X_raw) * 0.70)
    scaler  = StandardScaler()
    scaler.fit(X_raw[:train_n])
    X       = scaler.transform(X_raw)
    X_train = X[:train_n]

    print(f"\nFitting {args.n_states}-state Gaussian HMM on first {train_n} windows "
          f"(train split)...")
    hmm = GaussianHMM(
        n_components=args.n_states,
        covariance_type="full",
        n_iter=200,
        random_state=42,
        verbose=False,
    )
    hmm.fit(X_train)
    print(f"  Converged: {hmm.monitor_.converged}")

    states = hmm.predict(X)

    # ── State characterisation ────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("STATE CHARACTERISATION  (input feature means + analysis metrics)")
    print("=" * 68)
    print(f"  {'St':>3}  {'n':>5}  {'%win':>5}  {'log_sig':>8}  {'vol_reg':>8}  "
          f"{'|z_early|':>9}  {'dir_acc':>8}  label")

    state_labels: dict[int, str] = {}
    state_dir_acc: dict[int, float] = {}

    for s in range(args.n_states):
        mask = states == s
        n    = int(mask.sum())
        if n == 0:
            state_labels[s] = f"state_{s}"
            state_dir_acc[s] = 0.5
            continue

        log_sig_m  = X_raw[mask, 0].mean()
        vol_reg_m  = X_raw[mask, 1].mean()
        z_abs_m    = X_raw[mask, 2].mean()

        # Directional accuracy: how often did z_early direction match outcome?
        valid = sig_correct[mask] >= 0
        dir_acc = float(sig_correct[mask][valid].mean()) if valid.sum() > 0 else float("nan")

        # Label based on vol and z characteristics
        is_high_vol = log_sig_m > float(X_raw[:, 0].mean())
        is_high_z   = z_abs_m   > float(X_raw[:, 2].mean())
        if is_high_vol and is_high_z:
            label = "volatile/trending"
        elif is_high_z:
            label = "trending"
        elif is_high_vol:
            label = "volatile/choppy"
        else:
            label = "quiet/choppy"

        state_labels[s]   = label
        state_dir_acc[s]  = dir_acc

        print(f"  {s:>3}  {n:>5}  {n/len(windows):>5.1%}  {log_sig_m:>8.4f}  "
              f"{vol_reg_m:>8.4f}  {z_abs_m:>9.4f}  {dir_acc:>8.3f}  {label}")

    # ── Transition matrix ─────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("TRANSITION MATRIX  (row=from, col=to)")
    print("=" * 68)
    trans = hmm.transmat_
    header = "".join(f"  {'s'+str(j):>6}" for j in range(args.n_states))
    print(f"  {'':>4}{header}")
    for i in range(args.n_states):
        row = "".join(f"  {trans[i,j]:>6.3f}" for j in range(args.n_states))
        lbl = state_labels.get(i, "?")
        print(f"  s{i}  {row}   ({lbl})")
    persist = [trans[i,i] for i in range(args.n_states)]
    print(f"\n  Persistence (self-transition): "
          + "  ".join(f"s{i}={p:.3f}" for i, p in enumerate(persist)))
    for i, p in enumerate(persist):
        if p > 0 and p < 1:
            hl = math.log(0.5) / math.log(p)
            print(f"  → s{i} ({state_labels[i]}) half-life ≈ {hl:.0f} consecutive windows")

    # ── Holdout evaluation ────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"HOLDOUT EVALUATION  (last {100*(1-0.70):.0f}% of windows = {len(windows)-train_n} windows)")
    print("=" * 68)

    val_states  = states[train_n:]
    val_sig_cor = sig_correct[train_n:]

    # Filter to windows with valid signal
    valid_mask   = val_sig_cor >= 0
    val_s_valid  = val_states[valid_mask]
    val_sc_valid = val_sig_cor[valid_mask]

    if valid_mask.sum() == 0:
        print("  No valid signal windows in holdout.")
        return

    base_acc = float(val_sc_valid.mean())
    print(f"\n  Base rate (all valid holdout windows): "
          f"n={valid_mask.sum()}  dir_acc={base_acc:.3f}\n")
    print(f"  {'filter':>24}  {'n':>5}  {'dir_acc':>8}  {'lift':>7}  {'coverage':>9}")
    print(f"  {'all':>24}  {valid_mask.sum():>5}  {base_acc:>8.3f}  {0:>+7.3f}  {1.0:>9.1%}")

    for s in range(args.n_states):
        mask = val_s_valid == s
        n    = int(mask.sum())
        if n < 5:
            continue
        acc  = float(val_sc_valid[mask].mean())
        lift = acc - base_acc
        cov  = n / valid_mask.sum()
        lbl  = state_labels.get(s, f"state_{s}")
        print(f"  {f's{s} ({lbl})':>24}  {n:>5}  {acc:>8.3f}  {lift:>+7.3f}  {cov:>9.1%}")

    # Best state for trading
    best_s = max(range(args.n_states),
                 key=lambda s: state_dir_acc.get(s, 0))
    print(f"\n  Best state to trade: s{best_s} ({state_labels[best_s]}) "
          f"— dir_acc={state_dir_acc[best_s]:.3f}")

    # ── Feature means per state (unstandardised) ──────────────────────────────
    print("\n" + "=" * 68)
    print("FEATURE MEANS PER STATE  (unstandardised)")
    print("=" * 68)
    print(f"  {'feature':>20}  "
          + "  ".join(f"{'s'+str(s)+' ('+state_labels[s]+')':>20}" for s in range(args.n_states)))
    for j, col in enumerate(INPUT_COLS):
        vals = "  ".join(
            f"{X_raw[states == s, j].mean():>20.5f}" for s in range(args.n_states)
        )
        print(f"  {col:>20}  {vals}")

    print("\n  Summary:")
    print("  - dir_acc = fraction of windows where z_early direction matched outcome")
    print("  - 0.5 = random; >0.5 = z signal is correct; <0.5 = z signal misleads")
    print("  - Use regime state as a filter: trade only in high dir_acc states")


if __name__ == "__main__":
    main()
