"""
Live HMM regime classifier (Quant Guild #51 — Hidden Markov Models).

This module provides a thin wrapper around a trained Gaussian HMM
(from analysis/hmm_regime.py) so it can be loaded and used at runtime
by `DiffusionSignal`. The wrapper:

  * loads (HMM, scaler, state_labels, state_kelly_mult) from a pickle
  * exposes `classify_window(features) -> (state_idx, label, kelly_mult)`
  * is None-safe — DiffusionSignal treats `regime_classifier=None` as
    a no-op so the live signal works whether or not a model is loaded

Pattern intentionally mirrors filtration_model.py (FiltrationModel) so
the load/save semantics and the training-to-inference flow are the same.

Feature vector (must match analysis/hmm_regime.py:window_features):
  [log_sigma, vol_regime, z_early_abs, hour_sin, hour_cos]

Usage:
    from regime_classifier import RegimeClassifier
    rc = RegimeClassifier.load("regime_classifier_btc.pkl")
    state_idx, label, kelly_mult = rc.classify_window(feature_vector)

To train (offline):
    python analysis/hmm_regime.py --asset btc_15m --save regime_classifier_btc.pkl
"""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class RegimeClassifier:
    """Wraps an hmmlearn GaussianHMM + a StandardScaler + state metadata."""

    hmm: object              # hmmlearn.hmm.GaussianHMM (kept opaque)
    scaler: object           # sklearn StandardScaler (kept opaque)
    state_labels: dict       # {state_idx: human label}
    state_kelly_mult: dict   # {state_idx: kelly multiplier in [0, 1]}
    feature_names: tuple = ("log_sigma", "vol_regime", "z_early_abs",
                            "hour_sin", "hour_cos")

    def classify_window(self, features: Sequence[float]) -> tuple[int, str, float]:
        """Return (state_idx, label, kelly_mult) for one window's features.

        Features must be in the same order as `feature_names`. The
        scaler is applied identically to training. We return the most
        likely state via `predict_proba` (argmax of posterior) so the
        live decision is consistent with how the offline analysis
        characterised states.
        """
        if len(features) != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features in order "
                f"{self.feature_names}, got {len(features)}"
            )
        import numpy as np  # local import to avoid hard dep at import time
        x = np.array(features, dtype=np.float64).reshape(1, -1)
        x_s = self.scaler.transform(x)
        # `predict_proba` returns shape (n_samples, n_states)
        proba = self.hmm.predict_proba(x_s)[0]
        state_idx = int(np.argmax(proba))
        label = self.state_labels.get(state_idx, f"state_{state_idx}")
        kelly_mult = float(self.state_kelly_mult.get(state_idx, 1.0))
        return state_idx, label, kelly_mult

    @classmethod
    def load(cls, path: str | Path) -> "RegimeClassifier":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"RegimeClassifier model not found at {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(
            hmm=data["hmm"],
            scaler=data["scaler"],
            state_labels=data["state_labels"],
            state_kelly_mult=data.get("state_kelly_mult",
                                      {k: 1.0 for k in data["state_labels"]}),
        )

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "hmm": self.hmm,
                "scaler": self.scaler,
                "state_labels": self.state_labels,
                "state_kelly_mult": self.state_kelly_mult,
            }, f)


# ── Standalone feature extractors (live-friendly) ───────────────────────────
#
# These mirror analysis/hmm_regime.py:window_features but are designed to
# operate on incremental tick data rather than a full DataFrame. The live
# signal calls them once per window (after enough ticks have arrived) and
# caches the result in ctx.


def compute_window_regime_features(
    prices: Sequence[float],
    timestamps: Sequence[int],
    early_tau_target: float,
    vol_lookback_s: int = 90,
    max_z: float = 1.5,
    sigma_floor: float = 1e-12,
) -> list[float] | None:
    """Compute the 5-feature HMM input vector from in-window price history.

    Returns None if there isn't enough data yet (e.g. fewer than ~half
    of the window's ticks have come in).

    `early_tau_target` is the seconds-remaining at which `z_early_abs`
    should be computed (700s for 15m markets, ~200s for 5m markets).
    The caller is responsible for picking the right target for the
    market timeframe.
    """
    n = len(prices)
    if n < max(vol_lookback_s, 30) or n != len(timestamps):
        return None

    # Feature 1: log_sigma over the first half of currently-available history
    half = n // 2
    sigma_first = _compute_sigma_normalised(prices[:half], timestamps[:half])
    if sigma_first <= 0:
        return None

    # Feature 2: vol regime ratio = full / first-half
    sigma_full = _compute_sigma_normalised(prices, timestamps)
    if sigma_full <= 0:
        return None
    vol_regime = sigma_full / max(sigma_first, sigma_floor)

    # Feature 3: |z_early| at the early-tau index
    # Find the index closest to (start + (window_dur - early_tau_target))
    # Caller passes prices/timestamps that *start* at the window beginning,
    # so the elapsed time at index i is (timestamps[i] - timestamps[0]) / 1000.
    # We want the index where time_elapsed ≈ window_dur - early_tau_target.
    # But we don't know window_dur here, so we approximate using the latest
    # available index and let the caller pass what makes sense.
    # In practice the live signal calls this when tau is just past the
    # target, so using the latest index is a fine approximation.
    best_idx = n - 1
    lo = max(0, best_idx - vol_lookback_s)
    sigma_e = _compute_sigma_normalised(prices[lo:best_idx + 1],
                                        timestamps[lo:best_idx + 1])
    if sigma_e <= 0:
        return None
    if early_tau_target <= 0:
        return None
    delta_e = (prices[best_idx] - prices[0]) / max(prices[0], 1e-12)
    z_early = delta_e / (sigma_e * math.sqrt(early_tau_target))
    z_early = max(-max_z, min(max_z, z_early))

    # Features 4 & 5: hour-of-day cyclic encoding (UTC, from window start)
    import datetime as _dt
    hour = _dt.datetime.fromtimestamp(
        timestamps[0] / 1000, tz=_dt.timezone.utc
    ).hour
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)

    return [
        math.log(sigma_first + 1e-10),
        vol_regime,
        abs(z_early),
        hour_sin,
        hour_cos,
    ]


def _compute_sigma_normalised(prices: Sequence[float],
                              timestamps: Sequence[int]) -> float:
    """Sample stdev of dt-normalised log returns. Mirrors compute_sigma()
    in analysis/hmm_regime.py exactly so feature parity is preserved.
    """
    if len(prices) < 3:
        return 0.0
    rets: list[float] = []
    for j in range(1, len(prices)):
        if prices[j] <= 0 or prices[j - 1] <= 0 or prices[j] == prices[j - 1]:
            continue
        dt = (timestamps[j] - timestamps[j - 1]) / 1000.0
        if dt <= 0:
            continue
        r = math.log(prices[j] / prices[j - 1]) / math.sqrt(dt)
        rets.append(r)
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(max(var, 0.0))
