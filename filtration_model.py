"""
XGBoost filtration model: predicts whether the current z-score signal
direction is likely to be correct given market microstructure conditions.

Used in two places:
  - train_filtration.py: offline training
  - backtest.py / DiffusionSignal: online inference gate

Feature set is defined ONCE here (extract_features) to guarantee
train/inference parity. Never compute features differently in training
vs live — that is a look-ahead bug waiting to happen.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

MODEL_PATH = Path(__file__).parent / "filtration_model.pkl"


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(
    z: float,
    sigma: float,
    tau: float,
    spread_up: float,
    spread_down: float,
    imbalance5_up: float,
    imbalance5_down: float,
    buy_pressure: float,      # fraction of recent trades that were BUY (0-1)
    vol_regime_ratio: float,  # recent sigma / baseline sigma (>1 = vol spike)
    mid_up_momentum: float,   # change in mid_up over last 60s (signed)
    hour_of_day: int,         # 0-23
    is_weekend: int,          # 0 or 1
    asset_id: int,            # 0=btc_15m, 1=eth_15m, 2=sol_15m, 3=xrp_15m,
                              # 4=btc_5m, 5=eth_5m, 6=sol_5m, 7=xrp_5m
) -> list[float]:
    """
    Extract the feature vector used for both training and inference.
    Order and content must never diverge between training and live.

    All inputs must be computed from data available AT the decision
    timestamp — no future information.
    """
    return [
        z,                          # z-score (signed)
        abs(z),                     # |z| — magnitude of signal
        z * z,                      # z² — nonlinear signal strength
        sigma,                      # realized vol
        math.log(sigma + 1e-10),    # log-sigma (more normal distribution)
        tau,                        # time remaining (seconds)
        math.sqrt(tau),             # sqrt(tau) — GBM time scaling
        math.log(tau + 1.0),        # log(tau)
        spread_up,
        spread_down,
        (spread_up + spread_down) / 2.0,   # avg spread
        imbalance5_up,              # positive = more bid depth (buy pressure)
        imbalance5_down,
        imbalance5_up - imbalance5_down,   # cross-market imbalance
        buy_pressure,               # fraction of recent trades that were BUY
        buy_pressure - 0.5,         # centered buy pressure
        vol_regime_ratio,           # >1 = vol spike, <1 = quiet
        math.log(vol_regime_ratio + 1e-3),
        mid_up_momentum,            # positive = mid_up rising
        abs(mid_up_momentum),
        float(hour_of_day),
        math.sin(2 * math.pi * hour_of_day / 24),   # cyclic hour encoding
        math.cos(2 * math.pi * hour_of_day / 24),
        float(is_weekend),
        float(asset_id),
        # interaction terms
        z * spread_up,              # high z + wide spread = uncertain
        z * vol_regime_ratio,       # high z during vol spike
        abs(z) * tau,               # signal strength × time pressure
        imbalance5_up * z,          # book agrees with signal?
    ]


FEATURE_NAMES = [
    "z", "z_abs", "z_sq", "sigma", "log_sigma",
    "tau", "sqrt_tau", "log_tau",
    "spread_up", "spread_down", "avg_spread",
    "imbalance5_up", "imbalance5_down", "cross_imbalance",
    "buy_pressure", "buy_pressure_centered",
    "vol_regime_ratio", "log_vol_regime",
    "mid_up_momentum", "mid_up_momentum_abs",
    "hour_of_day", "hour_sin", "hour_cos", "is_weekend",
    "asset_id",
    "z_x_spread", "z_x_vol_regime", "z_abs_x_tau", "imbalance_x_z",
]

ASSET_IDS = {
    "btc_15m": 0, "eth_15m": 1, "sol_15m": 2, "xrp_15m": 3,
    "btc_5m":  4, "eth_5m":  5, "sol_5m":  6, "xrp_5m":  7,
}


# ── Model wrapper ─────────────────────────────────────────────────────────────

class CalibratedWrapper:
    """Wraps XGBoost + logistic calibration so the pair is picklable."""

    def __init__(self, base, cal):
        self._base = base
        self._cal = cal
        self.feature_importances_ = base.feature_importances_

    def predict_proba(self, X):
        raw = self._base.predict_proba(X)[:, 1]
        return self._cal.predict_proba(raw.reshape(-1, 1))


class FiltrationModel:
    """Thin wrapper around a trained XGBoost classifier."""

    def __init__(self, model, threshold: float = 0.55):
        self._model = model
        self.threshold = threshold

    @classmethod
    def load(cls, path: Path = MODEL_PATH, threshold: float = 0.55) -> "FiltrationModel":
        if not path.exists():
            raise FileNotFoundError(
                f"Filtration model not found at {path}. "
                "Run train_filtration.py first."
            )
        with open(path, "rb") as f:
            model = pickle.load(f)
        return cls(model, threshold)

    def predict_proba(self, features: list[float]) -> float:
        """Return probability that the current signal direction is correct."""
        X = np.array(features, dtype=np.float32).reshape(1, -1)
        return float(self._model.predict_proba(X)[0, 1])

    def should_trade(self, features: list[float]) -> bool:
        return self.predict_proba(features) >= self.threshold

    @staticmethod
    def save(model, path: Path = MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model saved to {path}")
