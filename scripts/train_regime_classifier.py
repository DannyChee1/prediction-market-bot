#!/usr/bin/env python3
"""
Train a 2-state HMM regime classifier and save it for live use by
DiffusionSignal (Quant Guild #51 — Hidden Markov Models).

The HMM is fitted on the SAME features that analysis/hmm_regime.py
computes (so its offline characterisation transfers directly):
    [log_sigma, vol_regime, z_early_abs, hour_sin, hour_cos]

State characterisation:
  After fitting, each latent state's `signal_accuracy` is computed
  on the held-out (last 30%) of windows. The state with the LOWER
  accuracy gets a smaller `kelly_mult` so we trade smaller in the
  regime where the diffusion signal is less reliable. The exact
  multipliers are configurable via --high-mult / --low-mult.

Output:    regime_classifier_<market>.pkl (in the repo root by default)

Usage:
    python scripts/train_regime_classifier.py --market btc_15m
    python scripts/train_regime_classifier.py --market btc_5m \
        --high-mult 1.0 --low-mult 0.4
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis.hmm_regime import load_all_windows  # noqa: E402
from regime_classifier import RegimeClassifier  # noqa: E402

DATA_DIR = REPO_ROOT / "data"

INPUT_COLS = ["log_sigma", "vol_regime", "z_early_abs", "hour_sin", "hour_cos"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True,
                    choices=["btc_15m", "btc_5m", "eth_15m", "eth_5m",
                             "sol_15m", "sol_5m", "xrp_15m", "xrp_5m"])
    ap.add_argument("--n-states", type=int, default=2)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--high-mult", type=float, default=1.0,
                    help="kelly_fraction multiplier in the high-accuracy state")
    ap.add_argument("--low-mult", type=float, default=0.5,
                    help="kelly_fraction multiplier in the low-accuracy state")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output pkl path (default: regime_classifier_<market>.pkl)")
    args = ap.parse_args()

    if args.out is None:
        # Default path includes the full market key so 15m and 5m models
        # don't clobber each other.
        args.out = REPO_ROOT / f"regime_classifier_{args.market}.pkl"

    d = DATA_DIR / args.market
    if not d.exists():
        print(f"ERROR: data dir not found: {d}", file=sys.stderr)
        sys.exit(1)

    # 5m markets only have 300s per window, so the default EARLY_TAU=700
    # is too large. Pick an early-tau that's roughly half the window
    # duration.
    if args.market.endswith("_5m"):
        early_tau = 200.0
    else:
        early_tau = 700.0
    print(f"Loading {args.market} windows from {d} (early_tau={early_tau:.0f}s)...")
    windows = load_all_windows(d, early_tau=early_tau)
    print(f"  {len(windows)} windows with valid early z-scores")
    if len(windows) < 100:
        print("  Not enough windows; need at least 100", file=sys.stderr)
        sys.exit(1)

    X_raw = np.array([[w[c] for c in INPUT_COLS] for w in windows],
                     dtype=np.float64)
    sig_correct = np.array([w["sig_correct"] for w in windows], dtype=np.int32)

    train_n = int(len(X_raw) * args.train_frac)
    scaler = StandardScaler()
    scaler.fit(X_raw[:train_n])
    X = scaler.transform(X_raw)

    print(f"\nFitting {args.n_states}-state Gaussian HMM on first "
          f"{train_n} windows ...")
    hmm = GaussianHMM(
        n_components=args.n_states,
        covariance_type="full",
        n_iter=200,
        random_state=42,
        verbose=False,
    )
    hmm.fit(X[:train_n])
    if not hmm.monitor_.converged:
        print("  WARNING: HMM did not converge")
    states = hmm.predict(X)

    # Characterise each state on HELD-OUT windows
    val_states = states[train_n:]
    val_sig = sig_correct[train_n:]

    state_acc: dict[int, float] = {}
    print("\n" + "=" * 60)
    print(f"  HOLDOUT STATE CHARACTERISATION ({len(val_states)} windows)")
    print("=" * 60)
    print(f"  {'state':>6}  {'n':>6}  {'sig_acc':>8}")
    for s in range(args.n_states):
        mask = val_states == s
        n = int(mask.sum())
        valid = val_sig[mask] >= 0
        if valid.sum() > 0:
            acc = float(val_sig[mask][valid].mean())
        else:
            acc = float("nan")
        state_acc[s] = acc
        print(f"  {s:>6}  {n:>6}  {acc:>8.4f}")

    # Pick high/low states by accuracy. Ties → state 0 = high.
    valid_states = {s: a for s, a in state_acc.items() if not math.isnan(a)}
    if len(valid_states) < args.n_states:
        print("  WARNING: some states have no holdout data; using uniform mults")
        kelly_mult = {s: 1.0 for s in range(args.n_states)}
        labels = {s: f"state_{s}" for s in range(args.n_states)}
    else:
        sorted_by_acc = sorted(valid_states.items(),
                               key=lambda kv: kv[1], reverse=True)
        kelly_mult = {}
        labels = {}
        for rank, (s, acc) in enumerate(sorted_by_acc):
            if rank == 0:
                kelly_mult[s] = args.high_mult
                labels[s] = f"high_acc_{acc:.3f}"
            else:
                # Linearly interpolate between high and low for >2 states
                t = rank / max(len(sorted_by_acc) - 1, 1)
                m = args.high_mult * (1 - t) + args.low_mult * t
                kelly_mult[s] = m
                labels[s] = f"low_acc_{acc:.3f}"

    print("\n  Assigned kelly multipliers:")
    for s in range(args.n_states):
        print(f"    state {s} → {labels.get(s, '?')}, "
              f"kelly_mult={kelly_mult.get(s, 1.0):.3f}")

    # Save
    rc = RegimeClassifier(
        hmm=hmm,
        scaler=scaler,
        state_labels=labels,
        state_kelly_mult=kelly_mult,
    )
    rc.save(args.out)
    print(f"\n  → wrote {args.out}")
    print(f"  Load with: RegimeClassifier.load({str(args.out)!r})")


if __name__ == "__main__":
    main()
