"""
backtest_core — math helpers, vol estimators, dataclasses, calibration table,
simple signals.

Extracted from backtest.py during the P10.2 module split. NO internal cycles:
this module imports nothing from `backtest` or `signal_diffusion`. Both of
those modules import from here.
"""

from __future__ import annotations

import argparse
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from market_config import MARKET_CONFIGS, DEFAULT_MARKET, get_config

DATA_DIR = Path(__file__).parent / "data"

# Windows whose final row has time_remaining_s above this are incomplete
MIN_FINAL_REMAINING_S = 5.0

# Skip windows where recording started more than this many seconds late
MAX_START_GAP_S = 30.0

# Adaptive sigma floor: short-window σ (vol_lookback_s) cannot drop below
# MIN_SIGMA_RATIO × long-window σ (vol_regime_lookback_s). Prevents the
# "sigma collapse" pattern where 90s realized variance briefly bottoms out
# during a quiet sub-period of an otherwise volatile session, which would
# otherwise blow up z = delta / (sigma · √τ) and spike p_model into the
# max_z cap for one tick. Empirically, the worst spikes saw 8-20× σ drops
# in a single 60s diagnostic interval; 0.5 keeps the short window
# responsive but anchors it to the longer baseline so single-tick spikes
# can't fire trades. Disabled when the long-window baseline isn't yet
# computable (history < vol_regime_lookback_s).
MIN_SIGMA_RATIO = 0.5


# ── Fee & math helpers ──────────────────────────────────────────────────────

def poly_fee(p: float, maker: bool = False) -> float:
    """Polymarket fee for crypto markets.

    Taker: feeRate * p * (1-p)
    Maker: 0% (no fee for limit orders that provide liquidity)

    feeRate by category (as of March 2026):
      Crypto:     0.072  (max 1.80% at p=0.50)
      Economics:  0.060  (max 1.50%)
      Sports:     0.030  (max 0.75%)
      Politics:   0.040  (max 1.00%)
      Geopolitics: 0     (free)

    The bot trades crypto markets → feeRate = 0.072.

    Previously used 0.02 (3.6x too low), which inflated every taker-mode
    backtest PnL. Updated 2026-04-09 per Polymarket docs.
    """
    if maker:
        return 0.0
    return 0.072 * p * (1.0 - p)


def norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erfc (no scipy needed)."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


# ── Time-of-day / day-of-week volatility priors ──────────────────────────────
# Default multipliers and global mean σ — these are the static fallback
# values used when no per-market `hourly_priors.json` exists. Originally
# computed from 1844 BTC windows + 1000-hour Binance API klines.
#
# Notebook #93 (non-stationarity) warns that these constants drift with
# market regime. The data-driven path:
#
#     scripts/regen_hourly_priors.py --market btc_15m
#
# rebuilds them from the latest parquets and writes
# data/<market>/hourly_priors.json. `_time_prior_sigma` automatically
# loads the JSON if it exists for the market the timestamp belongs to.
# Recommended cadence: weekly via cron (see scripts/regen_hourly_priors.py).
_HOURLY_VOL_MULT: dict[int, float] = {
    0: 0.99, 1: 1.42, 2: 1.01, 3: 0.78, 4: 0.83, 5: 0.82,
    6: 0.80, 7: 0.79, 8: 0.80, 9: 0.79, 10: 0.67, 11: 0.71,
    12: 0.86, 13: 1.17, 14: 1.56, 15: 1.69, 16: 1.35, 17: 1.27,
    18: 1.09, 19: 1.01, 20: 0.89, 21: 0.94, 22: 0.96, 23: 0.85,
}
_DOW_VOL_MULT: dict[int, float] = {
    0: 1.36, 1: 1.19, 2: 1.41, 3: 1.21, 4: 0.90, 5: 0.46, 6: 0.59,
}
# Global mean sigma. Empirical 90s-rolling Yang-Zhang median across btc_5m
# and btc_15m parquets (measured 2026-04-11 on ~300 windows/market at tau 50,
# 150, 250): p50 ≈ 3.5e-5, p25 ≈ 2.1e-5, p75 ≈ 5.3e-5, mean ≈ 4.2e-5.
# The old 8.9e-5 value (from the 2025 code) was derived from daily σ ÷ √86400
# — i.e. a long-horizon estimator — which overstates the 90s realized vol
# the signal actually uses by ~2.5×. When the time-prior is invoked (first
# ~15s of a window before Yang-Zhang has enough bars), returning 8.9e-5 ×
# hour/dow multipliers produced priors in the 3-18e-5 range, with most hours
# well above the empirical median. Replaced with the empirical mean (4.2e-5)
# so the prior roughly matches what the signal will converge to once it has
# real history. This is Test #1 from the 2026-04-11 calibration audit.
_GLOBAL_MEAN_SIGMA: float = 4.2e-05

# Per-market overrides loaded lazily from data/<subdir>/hourly_priors.json.
# Keyed by data_subdir (e.g. "btc_15m"). Populated on first call to
# `_load_priors_for_subdir` and cached for the process lifetime. The keys
# are stringified dict keys because that's what JSON gives us back.
_PRIORS_CACHE: dict[str, dict] = {}


def _load_priors_for_subdir(subdir: str) -> dict | None:
    """Load hourly_priors.json for a market subdir, with caching."""
    if subdir in _PRIORS_CACHE:
        return _PRIORS_CACHE[subdir] or None
    path = DATA_DIR / subdir / "hourly_priors.json"
    if not path.exists():
        _PRIORS_CACHE[subdir] = {}
        return None
    try:
        import json as _json
        with open(path) as f:
            data = _json.load(f)
        # JSON dict keys are strings — convert to ints
        data["hourly_mult"] = {
            int(k): float(v) for k, v in data.get("hourly_mult", {}).items()
        }
        data["dow_mult"] = {
            int(k): float(v) for k, v in data.get("dow_mult", {}).items()
        }
        _PRIORS_CACHE[subdir] = data
        return data
    except Exception:
        _PRIORS_CACHE[subdir] = {}
        return None


def _time_prior_sigma(ts_ms: int, data_subdir: str | None = None) -> float:
    """Return a time-of-day / day-of-week sigma prior for the given timestamp.

    If `data_subdir` is supplied AND a hourly_priors.json file exists for
    it, the data-driven multipliers and global mean from that JSON are
    used. Otherwise we fall back to the hardcoded constants above.
    """
    import datetime as _dt
    utc = _dt.datetime.fromtimestamp(ts_ms / 1000, tz=_dt.timezone.utc)

    if data_subdir:
        priors = _load_priors_for_subdir(data_subdir)
        if priors:
            h_mult = priors.get("hourly_mult", {}).get(utc.hour, 1.0)
            d_mult = priors.get("dow_mult", {}).get(utc.weekday(), 1.0)
            return float(priors.get("global_mean_sigma", _GLOBAL_MEAN_SIGMA)
                         * h_mult * d_mult)

    h_mult = _HOURLY_VOL_MULT.get(utc.hour, 1.0)
    d_mult = _DOW_VOL_MULT.get(utc.weekday(), 1.0)
    return _GLOBAL_MEAN_SIGMA * h_mult * d_mult


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction evaluation for regularized incomplete beta."""
    _MAX_ITER = 64
    _EPS = 1e-12
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, _MAX_ITER + 1):
        m2 = 2 * m
        # even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        # odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_prefix = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    prefix = math.exp(log_prefix)
    # Use symmetry for better convergence
    if x < (a + 1.0) / (a + b + 2.0):
        return prefix * _betacf(a, b, x) / a
    else:
        return 1.0 - prefix * _betacf(b, a, 1.0 - x) / b


def fast_t_cdf(x: float, nu: float) -> float:
    """Student-t CDF via regularized incomplete beta function (exact)."""
    if nu > 200:
        return norm_cdf(x)
    u = nu / (nu + x * x)
    ib = _betainc(nu / 2.0, 0.5, u)
    if x >= 0:
        return 1.0 - 0.5 * ib
    else:
        return 0.5 * ib


# ── Kou jump-diffusion CDF ────────────────────────────────────────────────

def _poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability mass function P(N=k) for rate lam."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def kou_cdf(x: float, sigma: float, lam: float, p_up: float,
            eta1: float, eta2: float, tau: float,
            mu_override: float | None = None) -> float:
    """CDF of log-return under Kou double-exponential jump-diffusion.

    P(X_tau < x) where X_tau = (drift)*tau + sigma*W_tau + sum_jumps.

    For our digital option: P(UP) = 1 - kou_cdf(-delta, ...) where
    delta = log(S_t / S_0).

    Uses CLT approximation when expected jumps (lam*tau) > 5, exact
    Poisson-weighted Gaussian convolution otherwise.

    `sigma` must be the CONTINUOUS-component σ (not total realized σ).
    The function adds jump variance on top of it via `lam_tau * ej2`,
    so passing a total-σ (which already absorbs jumps) will double-count
    variance. Use `bipower_variation_per_s` to get the continuous
    component.

    `mu_override`: when None, the risk-neutral (Q-measure) martingale
    drift `-σ²/2 - λ·ζ` is used — that's the option-pricing convention.
    For physical-measure binary prediction (P(S_T > S_0) on short-
    horizon crypto), pass `mu_override=0.0` so the drift term
    disappears entirely; short-horizon crypto has ~zero expected return
    per second and we just want the spread of the distribution around
    the current price.
    """
    q_down = 1.0 - p_up
    # Jump moments
    ej  = p_up / eta1 - q_down / eta2            # E[Y_i]
    ej2 = 2.0 * p_up / (eta1**2) + 2.0 * q_down / (eta2**2)  # E[Y_i^2]

    if mu_override is not None:
        mu = float(mu_override)
    else:
        # Drift correction (risk-neutral): mu = -sigma^2/2 - lam*(E[e^Y]-1)
        # For small jumps: E[e^Y]-1 ≈ ej + ej2/2
        zeta = p_up * eta1 / (eta1 - 1.0) + q_down * eta2 / (eta2 + 1.0) - 1.0 \
            if eta1 > 1.0 else ej + ej2 / 2.0
        mu = -0.5 * sigma**2 - lam * zeta

    lam_tau = lam * tau

    if lam_tau > 5.0 or lam <= 0:
        # CLT: total process is approximately Gaussian
        total_mean = mu * tau + lam_tau * ej
        total_var  = sigma**2 * tau + lam_tau * ej2
        if total_var <= 0:
            return norm_cdf(x)
        return norm_cdf((x - total_mean) / math.sqrt(total_var))

    # Exact: Poisson-weighted sum with n jumps ≤ N_max
    n_max = min(20, int(lam_tau + 4 * math.sqrt(max(lam_tau, 1))) + 1)
    cdf = 0.0
    for n in range(n_max + 1):
        pw = _poisson_pmf(n, lam_tau)
        if pw < 1e-12:
            continue
        # With n jumps: X ~ N(mu*tau + n*ej, sigma^2*tau + n*ej2)
        m_n = mu * tau + n * ej
        v_n = sigma**2 * tau + n * ej2
        if v_n <= 0:
            cdf += pw * (1.0 if x >= m_n else 0.0)
        else:
            cdf += pw * norm_cdf((x - m_n) / math.sqrt(v_n))
    return cdf


# ── Calibration table ──────────────────────────────────────────────────────

def _build_ohlc_bars(
    prices: list[float],
    timestamps: list[int] | None = None,
    bar_s: float = 5.0,
) -> list[tuple[float, float, float, float]]:
    """Build OHLC micro-bars from tick data, skipping duplicate prices.

    Returns list of (open, high, low, close) tuples.
    Timestamps are in ms; if None, assumes 1 tick = 1 second.
    """
    # Collect actual price changes with timestamps
    changes: list[tuple[float, int]] = []  # (price, ts_ms)
    for i, p in enumerate(prices):
        ts = timestamps[i] if timestamps is not None else i * 1000
        if p > 0 and (not changes or p != changes[-1][0]):
            changes.append((p, ts))
    if len(changes) < 2:
        return []

    bar_ms = bar_s * 1000.0
    bars: list[tuple[float, float, float, float]] = []
    bar_start = changes[0][1]
    o = h = l = c = changes[0][0]

    for px, ts in changes[1:]:
        if ts - bar_start >= bar_ms:
            bars.append((o, h, l, c))
            o = h = l = c = px
            bar_start = ts
        else:
            h = max(h, px)
            l = min(l, px)
            c = px

    # Final partial bar only if it has meaningful content
    if c != o or h != l:
        bars.append((o, h, l, c))

    return bars


def _yang_zhang_vol(bars: list[tuple[float, float, float, float]], bar_s: float = 5.0) -> float:
    """Yang-Zhang volatility estimator from OHLC bars.

    Returns per-second volatility (sigma_per_s).
    Requires at least 3 bars (2 consecutive pairs for overnight returns).
    """
    n = len(bars)
    if n < 3:
        return 0.0

    # Components across consecutive bars
    log_oc = []  # open-to-prev-close ("overnight")
    log_co = []  # close-to-open (intrabar)
    rs_vals = []  # Rogers-Satchell per bar

    for i in range(1, n):
        o, h, l, c = bars[i]
        prev_c = bars[i - 1][3]
        if prev_c <= 0 or o <= 0 or h <= 0 or l <= 0 or c <= 0:
            continue
        if h < l:
            continue

        log_oc.append(math.log(o / prev_c))
        log_co.append(math.log(c / o))

        log_ho = math.log(h / o)
        log_lo = math.log(l / o)
        log_hc = math.log(h / c)
        log_lc = math.log(l / c)
        rs_vals.append(log_ho * log_hc + log_lo * log_lc)

    m = len(log_oc)
    if m < 2:
        return 0.0

    # Variances (sample variance with ddof=1 for overnight and close-open)
    mean_oc = sum(log_oc) / m
    var_oc = sum((x - mean_oc) ** 2 for x in log_oc) / (m - 1)

    mean_co = sum(log_co) / m
    var_co = sum((x - mean_co) ** 2 for x in log_co) / (m - 1)

    # Rogers-Satchell: mean (not variance)
    var_rs = sum(rs_vals) / m

    # Yang-Zhang weighting
    k = 0.34 / (1.34 + (m + 1) / (m - 1))
    var_yz = var_oc + k * var_co + (1 - k) * var_rs

    if var_yz <= 0:
        return 0.0

    # Convert from per-bar to per-second
    return math.sqrt(var_yz / bar_s)


def _compute_vol_deduped(
    prices: list[float],
    timestamps: list[int] | None = None,
) -> float:
    """Yang-Zhang realized vol from price series via 5s OHLC micro-bars.

    Standalone version of DiffusionSignal._compute_vol for use outside
    the signal class (e.g. build_calibration_table).

    Falls back to simple stdev of time-normalized log returns when fewer
    than 3 OHLC bars are available.
    """
    bars = _build_ohlc_bars(prices, timestamps, bar_s=5.0)
    if len(bars) >= 3:
        result = _yang_zhang_vol(bars, bar_s=5.0)
        if result > 0:
            return result

    # Fallback: simple stdev (original method)
    changes: list[tuple[int, float, int]] = []
    for i, p in enumerate(prices):
        ts = timestamps[i] if timestamps is not None else i * 1000
        if p > 0 and (not changes or p != changes[-1][1]):
            changes.append((i, p, ts))
    if len(changes) < 3:
        return 0.0
    log_rets = []
    for j in range(1, len(changes)):
        if timestamps is not None:
            dt = (changes[j][2] - changes[j - 1][2]) / 1000.0
        else:
            dt = changes[j][0] - changes[j - 1][0]
        if dt > 0:
            lr = math.log(changes[j][1] / changes[j - 1][1])
            log_rets.append(lr / math.sqrt(dt))
    if len(log_rets) < 2:
        return 0.0
    return float(np.std(log_rets, ddof=1))


class CalibrationTable:
    """Walk-forward calibrated p(UP) lookup by (z_bin, tau_bin)."""

    Z_BIN_WIDTH = 0.5       # bins: -2.0, -1.5, ..., +2.0
    TAU_EDGES = [0, 120, 300, 600, 900]  # 4 buckets
    MIN_OBS = 20             # minimum observations per cell

    def __init__(self, table: dict, counts: dict):
        self.table = table    # {(z_bin, tau_idx): p_up}
        self.counts = counts  # {(z_bin, tau_idx): n}

    def lookup(self, z_capped: float, tau: float) -> float:
        """Return calibrated p(UP). For fusion mode, use lookup_with_count."""
        p, _ = self.lookup_with_count(z_capped, tau)
        return p

    def lookup_with_count(self, z_capped: float, tau: float) -> tuple[float, int]:
        """Return (p_calibrated, n_observations) for fusion with GBM prior."""
        z_bin = round(z_capped / self.Z_BIN_WIDTH) * self.Z_BIN_WIDTH
        tau_idx = self._tau_idx(tau)

        # Try exact (z_bin, tau_idx)
        key = (z_bin, tau_idx)
        if key in self.table and self.counts.get(key, 0) >= self.MIN_OBS:
            return self.table[key], self.counts[key]

        # Fall back to z-bin only (average across tau buckets)
        z_vals = [(self.table[k], self.counts[k])
                  for k in self.table if k[0] == z_bin and self.counts.get(k, 0) >= 5]
        if z_vals:
            total_n = sum(n for _, n in z_vals)
            return sum(p * n for p, n in z_vals) / total_n, total_n

        # No calibration data for this cell
        return norm_cdf(z_capped), 0

    def _tau_idx(self, tau: float) -> int:
        for i in range(len(self.TAU_EDGES) - 1):
            if self.TAU_EDGES[i] <= tau < self.TAU_EDGES[i + 1]:
                return i
        return len(self.TAU_EDGES) - 2  # clamp to last bucket


_CROSS_ASSET_TAU_CHECKPOINTS = [750, 600, 450, 300, 150, 60]


def _cross_asset_compute_sigma(prices: list[float], timestamps: list[int]) -> float:
    """Realized vol from log returns normalized by sqrt(dt)."""
    changes: list[tuple[int, float]] = []
    for i, p in enumerate(prices):
        if p > 0 and (not changes or p != changes[-1][1]):
            changes.append((timestamps[i], p))
    if len(changes) < 3:
        return 0.0
    log_rets = []
    for j in range(1, len(changes)):
        dt = (changes[j][0] - changes[j - 1][0]) / 1000.0
        if dt > 0:
            lr = math.log(changes[j][1] / changes[j - 1][1])
            log_rets.append(lr / math.sqrt(dt))
    if len(log_rets) < 2:
        return 0.0
    return float(np.std(log_rets, ddof=1))


def build_cross_asset_lookup(
    secondary_dir: Path,
    vol_lookback: int = 90,
    max_z: float = 1.5,
) -> dict[int, dict[int, float]]:
    """
    Precompute z-scores for a secondary asset at fixed tau checkpoints.

    Returns: {window_start_ms: {tau_checkpoint: z}}

    Usage: pass as cross_asset_z_lookup= to DiffusionSignal.
    The backtester injects _window_start_ms into ctx per window.
    """
    result: dict[int, dict[int, float]] = {}
    for f in sorted(secondary_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        if df.empty:
            continue
        if "chainlink_btc" in df.columns and "chainlink_price" not in df.columns:
            df.rename(columns={"chainlink_btc": "chainlink_price"}, inplace=True)
        if "chainlink_price" not in df.columns or "window_start_ms" not in df.columns:
            continue
        sp_series = df["window_start_price"].dropna()
        if sp_series.empty:
            continue
        start_px = float(sp_series.iloc[0])
        if start_px == 0:
            continue
        wstart = int(df["window_start_ms"].iloc[0])
        prices  = df["chainlink_price"].tolist()
        ts_list = df["ts_ms"].tolist()
        tau_all = df["time_remaining_s"].tolist()

        tau_map: dict[int, float] = {}
        for target_tau in _CROSS_ASSET_TAU_CHECKPOINTS:
            best_idx = min(range(len(tau_all)), key=lambda i: abs(tau_all[i] - target_tau))
            if abs(tau_all[best_idx] - target_tau) > 30:
                continue
            lo = max(0, best_idx - vol_lookback)
            sigma = _cross_asset_compute_sigma(prices[lo:best_idx + 1], ts_list[lo:best_idx + 1])
            if sigma <= 0:
                continue
            actual_tau = tau_all[best_idx]
            if actual_tau <= 0:
                continue
            delta = (prices[best_idx] - start_px) / start_px
            z_raw = delta / (sigma * math.sqrt(actual_tau))
            tau_map[target_tau] = float(max(-max_z, min(max_z, z_raw)))

        if tau_map:
            result[wstart] = tau_map
    return result


def _lookup_cross_asset_z(
    lookup: dict[int, dict[int, float]],
    window_start_ms: int | None,
    tau: float,
) -> float | None:
    """Return secondary asset z at the nearest tau checkpoint for the given window."""
    if window_start_ms is None or window_start_ms not in lookup:
        return None
    tau_map = lookup[window_start_ms]
    if not tau_map:
        return None
    closest = min(tau_map.keys(), key=lambda t: abs(t - tau))
    # Only use if within 120s of a checkpoint (avoid stale lookups near expiry)
    if abs(closest - tau) > 120:
        return None
    return tau_map[closest]


def build_calibration_table(
    data_dir: Path,
    max_z: float = 1.0,
    vol_lookback_s: int = 90,
    return_sigmas: bool = False,
    train_frac: float = 1.0,
) -> CalibrationTable | tuple[CalibrationTable, list[float]]:
    """Build a walk-forward calibration table from complete parquet windows.

    train_frac controls how much of the data (by time order) is used to
    build the table.  train_frac=1.0 uses everything (backward-compatible).
    train_frac=0.7 uses the first 70% of windows by timestamp — the correct
    walk-forward split so the table never contains future windows.

    When train_frac < 1.0 the function also attaches `val_cutoff_ts` to the
    returned CalibrationTable so callers can skip train-period windows during
    evaluation.

    If return_sigmas=True, also returns per-window sigma values (computed
    during the same file scan) for sigma_calibration rebuild.
    """
    # Load and sort all complete windows
    files = sorted(data_dir.glob("*.parquet"))
    windows: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_parquet(f)
        if df.empty:
            continue
        if "chainlink_btc" in df.columns and "chainlink_price" not in df.columns:
            df.rename(columns={"chainlink_btc": "chainlink_price"}, inplace=True)
        # Check completeness: need window_end_ms
        if "window_end_ms" in df.columns:
            end_ms = df["window_end_ms"].iloc[0]
            last_ts = df["ts_ms"].iloc[-1]
            if last_ts < end_ms:
                continue  # incomplete
        else:
            # Fall back to time_remaining_s check
            final_remaining = df["time_remaining_s"].iloc[-1]
            if final_remaining > MIN_FINAL_REMAINING_S:
                continue

        # Skip windows where recording started too late (not a full window)
        if ("window_start_ms" in df.columns and "window_end_ms" in df.columns
                and "time_remaining_s" in df.columns):
            window_dur_s = (df["window_end_ms"].iloc[0] - df["window_start_ms"].iloc[0]) / 1000
            first_remaining = df["time_remaining_s"].iloc[0]
            if first_remaining < window_dur_s - MAX_START_GAP_S:
                continue

        windows.append(df)

    windows.sort(key=lambda d: d["ts_ms"].iloc[0])

    # Walk-forward split: only use first train_frac windows to build the table
    val_cutoff_ts: int | None = None
    if train_frac < 1.0:
        split_idx = max(1, int(len(windows) * train_frac))
        val_cutoff_ts = int(windows[split_idx]["ts_ms"].iloc[0])
        train_windows = windows[:split_idx]
    else:
        train_windows = windows

    all_obs: list[tuple[float, float, int]] = []  # (z_capped, tau, outcome_up)
    window_sigmas: list[float] = []  # one per window for sigma_calibration

    for df in train_windows:
        # Determine outcome
        start_prices = df["window_start_price"].dropna()
        if start_prices.empty:
            continue
        start_px = float(start_prices.iloc[0])
        final_px = float(df["chainlink_price"].iloc[-1])
        if pd.isna(start_px) or pd.isna(final_px) or start_px == 0:
            continue
        outcome = 1 if final_px >= start_px else 0

        # Extract signals every 30 rows after warmup
        prices = df["chainlink_price"].tolist()
        ts_list = df["ts_ms"].tolist()

        # Per-window sigma for sigma_calibration (full window vol)
        ws = _compute_vol_deduped(prices, ts_list)
        if ws > 0:
            window_sigmas.append(ws)

        for idx in range(vol_lookback_s, len(df), 30):
            row = df.iloc[idx]
            tau = row["time_remaining_s"]
            if tau <= 0:
                continue

            lo = max(0, idx - vol_lookback_s)
            price_slice = prices[lo:idx + 1]
            ts_slice = ts_list[lo:idx + 1]

            # Skip if lookback window has a gap > 5 seconds between rows
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

            delta = (row["chainlink_price"] - start_px) / start_px
            z_raw = delta / (sigma * math.sqrt(tau))
            z_capped = max(-max_z, min(max_z, z_raw))

            all_obs.append((z_capped, tau, outcome))

    # Build table from training observations only
    table = _build_table_from_obs(all_obs)
    if val_cutoff_ts is not None:
        table.val_cutoff_ts = val_cutoff_ts  # attach for caller use
    if return_sigmas:
        return table, window_sigmas
    return table


def _build_table_from_obs(
    obs: list[tuple[float, float, int]],
) -> CalibrationTable:
    """Build a CalibrationTable from a list of (z_capped, tau, outcome) tuples."""
    from collections import defaultdict

    cell_outcomes: dict[tuple[float, int], list[int]] = defaultdict(list)
    z_bin_width = CalibrationTable.Z_BIN_WIDTH
    tau_edges = CalibrationTable.TAU_EDGES

    def tau_idx(tau: float) -> int:
        for i in range(len(tau_edges) - 1):
            if tau_edges[i] <= tau < tau_edges[i + 1]:
                return i
        return len(tau_edges) - 2

    for z_capped, tau, outcome in obs:
        z_bin = round(z_capped / z_bin_width) * z_bin_width
        ti = tau_idx(tau)
        cell_outcomes[(z_bin, ti)].append(outcome)

    table: dict[tuple[float, int], float] = {}
    counts: dict[tuple[float, int], int] = {}
    for key, outcomes in cell_outcomes.items():
        table[key] = sum(outcomes) / len(outcomes)
        counts[key] = len(outcomes)

    return CalibrationTable(table, counts)


# ── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Snapshot:
    ts_ms: int
    market_slug: str
    time_remaining_s: float
    chainlink_price: float
    window_start_price: float

    best_bid_up: Optional[float]
    best_ask_up: Optional[float]
    best_bid_down: Optional[float]
    best_ask_down: Optional[float]

    size_bid_up: Optional[float]
    size_ask_up: Optional[float]
    size_bid_down: Optional[float]
    size_ask_down: Optional[float]

    ask_levels_up: tuple[tuple[float, float], ...]
    ask_levels_down: tuple[tuple[float, float], ...]
    bid_levels_up: tuple[tuple[float, float], ...]
    bid_levels_down: tuple[tuple[float, float], ...]

    @staticmethod
    def from_row(row: pd.Series) -> Optional[Snapshot]:
        # Support both old "chainlink_btc" and new "chainlink_price" columns
        price_val = row.get("chainlink_price")
        if price_val is None or pd.isna(price_val):
            price_val = row.get("chainlink_btc")
        if (
            price_val is None
            or pd.isna(price_val)
            or pd.isna(row.get("window_start_price"))
            or pd.isna(row.get("time_remaining_s"))
        ):
            return None

        def _f(val):
            return None if pd.isna(val) else float(val)

        def _ask_levels(side: str) -> tuple[tuple[float, float], ...]:
            levels = []
            for i in range(1, 6):
                px, sz = row.get(f"ask_px_{side}_{i}"), row.get(f"ask_sz_{side}_{i}")
                if px is not None and sz is not None and not pd.isna(px) and not pd.isna(sz):
                    levels.append((float(px), float(sz)))
            levels.sort(key=lambda x: x[0])
            return tuple(levels)

        def _bid_levels(side: str) -> tuple[tuple[float, float], ...]:
            levels = []
            for i in range(1, 6):
                px, sz = row.get(f"bid_px_{side}_{i}"), row.get(f"bid_sz_{side}_{i}")
                if px is not None and sz is not None and not pd.isna(px) and not pd.isna(sz):
                    levels.append((float(px), float(sz)))
            levels.sort(key=lambda x: -x[0])
            return tuple(levels)

        return Snapshot(
            ts_ms=int(row["ts_ms"]),
            market_slug=str(row["market_slug"]),
            time_remaining_s=float(row["time_remaining_s"]),
            chainlink_price=float(price_val),
            window_start_price=float(row["window_start_price"]),
            best_bid_up=_f(row.get("best_bid_up")),
            best_ask_up=_f(row.get("best_ask_up")),
            best_bid_down=_f(row.get("best_bid_down")),
            best_ask_down=_f(row.get("best_ask_down")),
            size_bid_up=_f(row.get("size_bid_up")),
            size_ask_up=_f(row.get("size_ask_up")),
            size_bid_down=_f(row.get("size_bid_down")),
            size_ask_down=_f(row.get("size_ask_down")),
            ask_levels_up=_ask_levels("up"),
            ask_levels_down=_ask_levels("down"),
            bid_levels_up=_bid_levels("up"),
            bid_levels_down=_bid_levels("down"),
        )


@dataclass(frozen=True, slots=True)
class Decision:
    action: str       # "BUY_UP" | "BUY_DOWN" | "FLAT"
    edge: float
    size_usd: float
    reason: str


@dataclass(frozen=True, slots=True)
class Fill:
    market_slug: str
    side: str              # "UP" or "DOWN"
    entry_ts_ms: int
    time_remaining_s: float
    entry_price: float     # effective price per share (includes fee + slippage)
    fee_per_share: float
    shares: float
    cost_usd: float        # total = shares * entry_price
    signal_name: str
    decision_reason: str
    # Expected price range at fill time (1-sigma band)
    btc_at_fill: float = 0.0
    start_price: float = 0.0
    expected_low: float = 0.0      # start_price - sigma * sqrt(tau) * btc
    expected_high: float = 0.0     # start_price + sigma * sqrt(tau) * btc


@dataclass(frozen=True, slots=True)
class TradeResult:
    fill: Fill
    outcome_up: int        # 1 if Up wins, 0 if Down wins
    final_btc: float       # actual BTC price at window end
    payout: float
    pnl: float
    pnl_pct: float


# ── Signal interface ────────────────────────────────────────────────────────

class Signal(ABC):
    name: str

    @abstractmethod
    def decide(self, snapshot: Snapshot, ctx: dict) -> Decision: ...


class AlwaysUp(Signal):
    name = "always_up"

    def __init__(self, bankroll: float):
        self.bankroll = bankroll

    def decide(self, snapshot: Snapshot, ctx: dict) -> Decision:
        ask = snapshot.best_ask_up
        if ask is None or ask <= 0 or ask >= 1.0:
            return Decision("FLAT", 0.0, 0.0, "no valid ask")
        fee = poly_fee(ask)
        if ask + fee >= 1.0:
            return Decision("FLAT", 0.0, 0.0, "eff price >= 1")
        return Decision("BUY_UP", 0.0, self.bankroll * 0.05, "always buy Up")


class AlwaysDown(Signal):
    name = "always_down"

    def __init__(self, bankroll: float):
        self.bankroll = bankroll

    def decide(self, snapshot: Snapshot, ctx: dict) -> Decision:
        ask = snapshot.best_ask_down
        if ask is None or ask <= 0 or ask >= 1.0:
            return Decision("FLAT", 0.0, 0.0, "no valid ask")
        fee = poly_fee(ask)
        if ask + fee >= 1.0:
            return Decision("FLAT", 0.0, 0.0, "eff price >= 1")
        return Decision("BUY_DOWN", 0.0, self.bankroll * 0.05, "always buy Down")


class RandomCoinFlip(Signal):
    name = "random"

    def __init__(self, bankroll: float, seed: int = 42):
        self.bankroll = bankroll
        self.rng = random.Random(seed)

    def decide(self, snapshot: Snapshot, ctx: dict) -> Decision:
        side = "BUY_UP" if self.rng.random() < 0.5 else "BUY_DOWN"
        ask = snapshot.best_ask_up if side == "BUY_UP" else snapshot.best_ask_down
        if ask is None or ask <= 0 or ask >= 1.0:
            return Decision("FLAT", 0.0, 0.0, "no valid ask")
        fee = poly_fee(ask)
        if ask + fee >= 1.0:
            return Decision("FLAT", 0.0, 0.0, "eff price >= 1")
        return Decision(side, 0.0, self.bankroll * 0.05, f"coin flip -> {side}")


def compute_vamp(
    bid_levels: tuple[tuple[float, float], ...],
    ask_levels: tuple[tuple[float, float], ...],
) -> float | None:
    """Volume Adjusted Mid Price from bid+ask depth levels.

    VAMP = Σ(price × size) / Σ(size)  across both sides of the book.
    Returns None if no depth available.
    """
    total_value = 0.0
    total_size = 0.0
    for px, sz in bid_levels:
        if sz > 0:
            total_value += px * sz
            total_size += sz
    for px, sz in ask_levels:
        if sz > 0:
            total_value += px * sz
            total_size += sz
    if total_size == 0:
        return None
    return total_value / total_size
