#!/usr/bin/env python3
"""
Polymarket BTC Up/Down Backtest Engine

Replays recorded 1-second snapshots from parquet files and evaluates
trading signals on binary BTC 15-minute Up/Down markets.

Usage:
    python backtest.py
    python backtest.py --signal all
    python backtest.py --bankroll 5000 --signal diffusion --latency 250 --slippage 0.002
    python backtest.py --sensitivity
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


# ── Fee & math helpers ──────────────────────────────────────────────────────

def poly_fee(p: float, maker: bool = False) -> float:
    """Polymarket fee for 15-min crypto markets.

    Taker: 2% * p * (1-p)  — standard Polymarket binary market fee.
    Maker: 0%              — no fee for limit orders that provide liquidity.

    NOTE: verify taker fee rate against current Polymarket documentation
    before relying on backtest P&L figures for taker-mode strategies.
    """
    if maker:
        return 0.0
    return 0.02 * p * (1.0 - p)


def norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erfc (no scipy needed)."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


# ── Time-of-day / day-of-week volatility priors ──────────────────────────────
# Multipliers relative to global mean sigma, computed from 1844 BTC windows.
# Used as Bayesian prior when raw vol is zero (quiet periods / weekends).
# Blended from our 1844-window parquet data + 1000-hour Binance API klines.
_HOURLY_VOL_MULT: dict[int, float] = {
    0: 0.99, 1: 1.42, 2: 1.01, 3: 0.78, 4: 0.83, 5: 0.82,
    6: 0.80, 7: 0.79, 8: 0.80, 9: 0.79, 10: 0.67, 11: 0.71,
    12: 0.86, 13: 1.17, 14: 1.56, 15: 1.69, 16: 1.35, 17: 1.27,
    18: 1.09, 19: 1.01, 20: 0.89, 21: 0.94, 22: 0.96, 23: 0.85,
}
_DOW_VOL_MULT: dict[int, float] = {
    0: 1.36, 1: 1.19, 2: 1.41, 3: 1.21, 4: 0.90, 5: 0.46, 6: 0.59,
}
# Global mean sigma (per 1-second non-zero return) across all BTC data.
_GLOBAL_MEAN_SIGMA: float = 8.9e-05


def _time_prior_sigma(ts_ms: int) -> float:
    """Return a time-of-day / day-of-week sigma prior for the given timestamp."""
    import datetime as _dt
    utc = _dt.datetime.fromtimestamp(ts_ms / 1000, tz=_dt.timezone.utc)
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
            eta1: float, eta2: float, tau: float) -> float:
    """CDF of log-return under Kou double-exponential jump-diffusion.

    P(X_tau < x) where X_tau = (drift)*tau + sigma*W_tau + sum_jumps.

    For our digital option: P(UP) = 1 - kou_cdf(-delta, ...) where
    delta = log(S_t / S_0).

    Uses CLT approximation when expected jumps (lam*tau) > 5, exact
    Poisson-weighted Gaussian convolution otherwise.
    """
    q_down = 1.0 - p_up
    # Jump moments
    ej  = p_up / eta1 - q_down / eta2            # E[Y_i]
    ej2 = 2.0 * p_up / (eta1**2) + 2.0 * q_down / (eta2**2)  # E[Y_i^2]

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


class DiffusionSignal(Signal):
    """
    Models BTC as GBM. p_model = Phi(delta / (sigma * sqrt(tau))).
    Buys whichever side has edge > threshold, sized by half-Kelly.
    """
    name = "diffusion"

    def __init__(
        self,
        bankroll: float,
        vol_lookback_s: int = 90,
        min_sigma: float = 1e-6,
        edge_threshold: float = 0.04,
        early_edge_mult: float = 1.2,
        window_duration: float = 900.0,
        max_bet_fraction: float = 0.05,
        min_order_shares: float = 5.0,
        kelly_fraction: float = 0.25,
        slippage: float = 0.0,
        max_z: float = 1.0,
        momentum_lookback_s: int = 30,
        max_spread: float = 0.05,
        spread_edge_penalty: float = 1.0,
        vol_regime_lookback_s: int = 300,
        vol_regime_mult: float = 3.0,
        max_entry_time_s: float | None = None,
        reversion_discount: float = 0.0,
        momentum_majority: float = 1.0,
        maker_mode: bool = False,
        edge_threshold_step: float = 0.0,
        calibration_table: CalibrationTable | None = None,
        maker_warmup_s: float = 100.0,
        min_entry_price: float = 0.10,
        cal_prior_strength: float = 500.0,
        cal_max_weight: float = 0.40,
        inventory_skew: float = 0.02,
        maker_withdraw_s: float = 60.0,
        sigma_ema_alpha: float = 0.30,
        max_sigma: float | None = None,
        vamp_mode: str = "none",
        vamp_filter_threshold: float = 0.07,
        max_entry_price: float = 1.0,
        # NO/DOWN bias: reduce edge threshold for DOWN by this fraction
        # of base threshold (optimism tax — YES/UP tends to be overpriced)
        down_edge_bonus: float = 0.0,
        # Microstructure toxicity filter
        toxicity_threshold: float = 0.75,
        toxicity_edge_mult: float = 1.5,
        # Volatility kill switch (absolute EMA ceiling)
        vol_kill_sigma: float | None = None,
        # Regime-scaled z: scale z by sigma_calibration / sigma_ema
        regime_z_scale: bool = False,
        sigma_calibration: float | None = None,
        # VPIN flow toxicity filter
        vpin_threshold: float = 0.95,
        vpin_edge_mult: float = 1.5,
        vpin_window: int = 20,
        vpin_bar_s: float = 60.0,
        # Oracle lag detection (Binance vs Chainlink discrepancy)
        oracle_lag_threshold: float = 0.002,
        oracle_lag_mult: float = 2.0,
        # Order book imbalance alpha: shift p_model by obi_weight * OBI
        obi_weight: float = 0.03,
        # Chainlink blend: seconds before expiry to start blending
        # effective price from Binance toward Chainlink for z-score.
        # At chainlink_blend_s remaining → 100% Binance;
        # at 0 remaining → 100% Chainlink (matches resolution source).
        chainlink_blend_s: float = 120.0,
        # Fat-tail CDF: "normal", "student_t", "kou", "market_adaptive"
        tail_mode: str = "normal",
        tail_nu_default: float = 20.0,
        # Kou jump-diffusion parameters (only used when tail_mode="kou")
        kou_lambda: float = 0.007,   # jump intensity per observation
        kou_p_up: float = 0.51,      # probability of upward jump
        kou_eta1: float = 1100.0,    # rate of upward jumps (1/mean_size)
        kou_eta2: float = 1100.0,    # rate of downward jumps
        # Market-adaptive parameters (only used when tail_mode="market_adaptive")
        market_blend_alpha: float = 0.30,  # weight on GBM vs market
        # Direct p_model blend with contract mid (anti-fade-the-market)
        # Applied AFTER _p_model() for any tail_mode. 0=off, 0.3=BTC 5m default.
        market_blend: float = 0.0,
        # Avellaneda-Stoikov unified quoting mode
        as_mode: bool = False,
        gamma_inv: float = 0.15,       # risk aversion for inventory penalty
        gamma_spread: float = 0.75,    # risk aversion for base spread
        min_edge: float = 0.05,        # floor on required edge
        tox_spread: float = 0.05,      # additive spread from toxicity
        vpin_spread: float = 0.05,     # additive spread from VPIN
        lag_spread: float = 0.08,      # additive spread from oracle lag
        edge_step: float = 0.01,       # additive spread per fill
        contract_vol_lookback_s: int = 60,  # lookback for contract mid vol
        # Kalman filter for sigma estimation (replaces EMA when True)
        use_kalman_sigma: bool = True,
        kalman_q: float = 0.10,   # process noise ratio (vol-of-vol / vol)
        kalman_r: float = 0.075,  # observation noise ratio (≈ 1/sqrt(2*90))
        # XGBoost filtration model (optional confidence gate)
        filtration_model=None,         # FiltrationModel instance or None
        filtration_threshold: float = 0.55,
        filtration_asset_id: int = 0,  # see filtration_model.ASSET_IDS
        filtration_baseline_vol_s: int = 300,
        # Oracle hard cancel: return FLAT immediately when Binance-Chainlink
        # gap exceeds this fraction.  Set to 0.0 to disable (default).
        # 0.004 = cancel when gap > 0.4% (Chainlink triggers at 0.5%).
        oracle_cancel_threshold: float = 0.0,
        # Cross-asset disagreement gate: skip trade when a correlated asset
        # (e.g. ETH when trading BTC) has a z-score pointing the other way.
        # Requires cross_asset_z_lookup precomputed by build_cross_asset_lookup().
        cross_asset_z_lookup: dict | None = None,
        cross_asset_min_z: float = 0.3,   # both |z| must exceed this to veto
        # Minimum |z| to enter: filter out low-conviction setups
        min_entry_z: float = 0.0,
    ):
        self.bankroll = bankroll
        self.vol_lookback_s = vol_lookback_s
        self.min_sigma = min_sigma
        self.edge_threshold = edge_threshold
        self.early_edge_mult = early_edge_mult
        self.window_duration = window_duration
        self.max_bet_fraction = max_bet_fraction
        self.min_order_shares = min_order_shares
        self.kelly_fraction = kelly_fraction
        self.slippage = slippage
        self.max_z = max_z
        self.momentum_lookback_s = momentum_lookback_s
        self.max_spread = max_spread
        self.spread_edge_penalty = spread_edge_penalty
        self.vol_regime_lookback_s = vol_regime_lookback_s
        self.vol_regime_mult = vol_regime_mult
        self.max_entry_time_s = max_entry_time_s
        self.reversion_discount = reversion_discount
        self.momentum_majority = momentum_majority
        self.maker_mode = maker_mode
        self.edge_threshold_step = edge_threshold_step
        self.calibration_table = calibration_table
        self.maker_warmup_s = maker_warmup_s
        self.min_entry_price = min_entry_price
        self.cal_prior_strength = cal_prior_strength
        self.cal_max_weight = cal_max_weight
        self.inventory_skew = inventory_skew
        self.maker_withdraw_s = maker_withdraw_s
        self.sigma_ema_alpha = sigma_ema_alpha
        self.max_sigma = max_sigma
        self.vamp_mode = vamp_mode                    # "none", "cost", "filter"
        self.vamp_filter_threshold = vamp_filter_threshold
        self.max_entry_price = max_entry_price
        self.down_edge_bonus = down_edge_bonus
        self.toxicity_threshold = toxicity_threshold
        self.toxicity_edge_mult = toxicity_edge_mult
        self.vol_kill_sigma = vol_kill_sigma
        self.regime_z_scale = regime_z_scale
        self.sigma_calibration = sigma_calibration
        self.vpin_threshold = vpin_threshold
        self.vpin_edge_mult = vpin_edge_mult
        self.vpin_window = vpin_window
        self.vpin_bar_s = vpin_bar_s
        self.oracle_lag_threshold = oracle_lag_threshold
        self.oracle_lag_mult = oracle_lag_mult
        self.obi_weight = obi_weight
        self.chainlink_blend_s = chainlink_blend_s
        self.tail_mode = tail_mode
        self.tail_nu_default = tail_nu_default
        self.kou_lambda = kou_lambda
        self.kou_p_up = kou_p_up
        self.kou_eta1 = kou_eta1
        self.kou_eta2 = kou_eta2
        self.market_blend_alpha = market_blend_alpha
        self.market_blend = market_blend
        self.as_mode = as_mode
        self.gamma_inv = gamma_inv
        self.gamma_spread = gamma_spread
        self.min_edge = min_edge
        self.tox_spread = tox_spread
        self.vpin_spread = vpin_spread
        self.lag_spread = lag_spread
        self.edge_step = edge_step
        self.contract_vol_lookback_s = contract_vol_lookback_s
        self.use_kalman_sigma = use_kalman_sigma
        self.kalman_q = kalman_q
        self.kalman_r = kalman_r
        self.filtration_model = filtration_model
        self.filtration_threshold = filtration_threshold
        self.filtration_asset_id = filtration_asset_id
        self.filtration_baseline_vol_s = filtration_baseline_vol_s
        self.oracle_cancel_threshold = oracle_cancel_threshold
        self.cross_asset_z_lookup = cross_asset_z_lookup
        self.cross_asset_min_z = cross_asset_min_z
        self.min_entry_z = min_entry_z

    def _compute_vol(
        self,
        prices: list[float],
        timestamps: list[int] | None = None,
    ) -> float:
        """Yang-Zhang realized vol from 5s OHLC micro-bars.

        Constructs OHLC bars from tick data (skipping duplicate prices),
        then applies the Yang-Zhang estimator which is up to 14x more
        statistically efficient than simple stdev of returns.

        Falls back to simple stdev when fewer than 3 OHLC bars are
        available (e.g. during early warmup).
        """
        return _compute_vol_deduped(prices, timestamps)

    def _record_book_state(self, snapshot: "Snapshot", ctx: dict) -> None:
        """Persist lightweight book state for diagnostics/filtration features."""
        ctx["_last_ts_ms"] = snapshot.ts_ms
        if snapshot.best_bid_up is None or snapshot.best_ask_up is None:
            return
        mid_up = (snapshot.best_bid_up + snapshot.best_ask_up) / 2.0
        mid_hist = ctx.setdefault("_mid_up_history", [])
        mid_hist.append(float(mid_up))
        if len(mid_hist) > 600:
            del mid_hist[:-600]

    def _maybe_update_tail_nu(self, hist: list[float], ctx: dict) -> None:
        """Estimate Student-t degrees of freedom from recent realized kurtosis."""
        if self.tail_mode != "student_t" or len(hist) < 30:
            return

        k_n = min(90, len(hist))
        k_prices = hist[-k_n:]
        k_rets = [
            math.log(k_prices[i] / k_prices[i - 1])
            for i in range(1, len(k_prices))
            if k_prices[i - 1] > 0 and k_prices[i] > 0
        ]
        if len(k_rets) < 20:
            return

        mean = sum(k_rets) / len(k_rets)
        var = sum((r - mean) ** 2 for r in k_rets) / len(k_rets)
        if var <= 0:
            return

        m4 = sum((r - mean) ** 4 for r in k_rets) / len(k_rets)
        excess_kurt = (m4 / (var * var)) - 3.0
        if excess_kurt > 0.2:
            ctx["_tail_nu"] = max(
                self.tail_nu_default,
                min(30.0, 4.0 + 6.0 / excess_kurt),
            )
        else:
            ctx["_tail_nu"] = 30.0  # near-Gaussian

    def _apply_regime_z_scale(
        self,
        z_raw: float,
        sigma_per_s: float,
        ctx: dict,
    ) -> tuple[float, float]:
        """Adjust z by current-vs-calibration vol regime.

        When live vol is above the calibration regime, shrink z.
        When live vol is below the calibration regime, amplify z.
        """
        factor = 1.0
        if (
            self.regime_z_scale
            and self.sigma_calibration
            and self.sigma_calibration > 0
            and sigma_per_s > 0
        ):
            factor = self.sigma_calibration / sigma_per_s
            factor = max(0.5, min(2.0, factor))
            z_raw *= factor
        ctx["_regime_z_factor"] = factor
        return z_raw, factor

    def _smoothed_sigma(self, raw_sigma: float, ctx: dict) -> float:
        """Kalman filter for sigma estimation (falls back to EMA if disabled).

        Kalman filter advantages over EMA:
          - Adaptive gain: reacts fast to vol spikes, smooth during calm periods
          - Principled noise model: Q (process noise) and R (observation noise)
          - Tracks uncertainty explicitly via posterior variance P

        State:   x  = true (latent) volatility
        Process: x_t = x_{t-1} + w,  w ~ N(0, Q)
        Obs:     y_t = x_t + v,       v ~ N(0, R)

        Q = (x * kalman_q)^2  — vol-of-vol proportional to current vol
        R = (x * kalman_r)^2  — measurement noise (Yang-Zhang ~90 samples
                                 has std ≈ sigma/sqrt(2N), so r ≈ 1/sqrt(180))
        """
        if raw_sigma == 0.0:
            # No measurable vol in lookback. Use best available prior:
            # 1. Kalman state from recent ticks (most accurate)
            # 2. Time-of-day / day-of-week historical prior (covers cold start)
            x = ctx.get("_kalman_x")
            if x is not None and x > 0:
                return max(min(x, self.max_sigma) if self.max_sigma else x,
                           self.min_sigma)
            # Cold start: use time-based prior so we don't go FLAT
            # on the very first window (especially weekends).
            ts_hist = ctx.get("ts_history", [])
            ts_ms = ctx.get("_last_ts_ms", 0) or (ts_hist[-1] if ts_hist else 0)
            if ts_ms > 0:
                prior = _time_prior_sigma(ts_ms)
                return max(min(prior, self.max_sigma) if self.max_sigma else prior,
                           self.min_sigma)
            return 0.0

        if not self.use_kalman_sigma:
            # Legacy EMA path
            ema = ctx.get("_sigma_ema")
            ema = raw_sigma if ema is None else (
                self.sigma_ema_alpha * raw_sigma + (1 - self.sigma_ema_alpha) * ema
            )
            ctx["_sigma_ema"] = ema
            if self.max_sigma is not None:
                ema = min(ema, self.max_sigma)
            return max(ema, self.min_sigma)

        # Kalman filter path
        x = ctx.get("_kalman_x")
        P = ctx.get("_kalman_P")

        if x is None:
            # Initialise: start at first observation, high uncertainty
            x = raw_sigma
            P = (raw_sigma * self.kalman_r) ** 2

        # Process and observation noise (scale with current estimate)
        Q = (x * self.kalman_q) ** 2
        R = (x * self.kalman_r) ** 2
        if R < 1e-20:
            R = 1e-20

        # Predict
        P_pred = P + Q

        # Update (Kalman gain)
        K = P_pred / (P_pred + R)
        x_new = x + K * (raw_sigma - x)
        P_new = (1.0 - K) * P_pred

        ctx["_kalman_x"] = x_new
        ctx["_kalman_P"] = P_new
        ctx["_kalman_gain"] = K  # expose for diagnostics

        if self.max_sigma is not None:
            x_new = min(x_new, self.max_sigma)
        return max(x_new, self.min_sigma)

    def _check_filtration(
        self,
        z: float,
        sigma: float,
        tau: float,
        snapshot: "Snapshot",
        ctx: dict,
    ) -> bool:
        """Return True if filtration model approves this trade, False to filter out.

        Returns True unconditionally when no filtration model is loaded,
        or when |z| is too small to have meaningful signal direction.
        """
        if self.filtration_model is None:
            return True
        if abs(z) < 0.10:
            return True  # no directional signal to filter

        from filtration_model import extract_features

        # Baseline vol for regime ratio (use longer lookback from history)
        hist = ctx.get("price_history", [])
        ts_hist = ctx.get("ts_history", [])
        lo_base = max(0, len(hist) - self.filtration_baseline_vol_s)
        sigma_base = _compute_vol_deduped(hist[lo_base:], ts_hist[lo_base:])
        vol_regime_ratio = sigma / sigma_base if sigma_base > 0 else 1.0

        # Buy pressure from recent trade side history
        trade_sides = ctx.get("_trade_side_history", [])
        if trade_sides:
            recent = trade_sides[-60:]
            buy_pressure = sum(1 for s in recent if s == "BUY") / len(recent)
        else:
            buy_pressure = 0.5

        # Mid momentum
        mid_history = ctx.get("_mid_up_history", [])
        mid_momentum = (float(mid_history[-1]) - float(mid_history[-61]))  \
            if len(mid_history) >= 62 else 0.0

        ts_ms = ctx.get("_last_ts_ms", 0)
        dt = pd.Timestamp(ts_ms, unit="ms", tz="UTC") if ts_ms else None
        hour_of_day = dt.hour if dt else 12
        is_weekend  = int(dt.weekday() >= 5) if dt else 0

        spread_up   = (snapshot.best_ask_up or 0.5) - (snapshot.best_bid_up or 0.5)
        spread_down = (snapshot.best_ask_down or 0.5) - (snapshot.best_bid_down or 0.5)
        imb_up   = sum(sz for _, sz in snapshot.bid_levels_up) - \
                   sum(sz for _, sz in snapshot.ask_levels_up)
        total_up = sum(sz for _, sz in snapshot.bid_levels_up) + \
                   sum(sz for _, sz in snapshot.ask_levels_up)
        imb_up = imb_up / total_up if total_up > 0 else 0.0

        imb_dn   = sum(sz for _, sz in snapshot.bid_levels_down) - \
                   sum(sz for _, sz in snapshot.ask_levels_down)
        total_dn = sum(sz for _, sz in snapshot.bid_levels_down) + \
                   sum(sz for _, sz in snapshot.ask_levels_down)
        imb_dn = imb_dn / total_dn if total_dn > 0 else 0.0

        features = extract_features(
            z=z,
            sigma=sigma,
            tau=tau,
            spread_up=max(0.0, spread_up),
            spread_down=max(0.0, spread_down),
            imbalance5_up=imb_up,
            imbalance5_down=imb_dn,
            buy_pressure=buy_pressure,
            vol_regime_ratio=vol_regime_ratio,
            mid_up_momentum=mid_momentum,
            hour_of_day=hour_of_day,
            is_weekend=is_weekend,
            asset_id=self.filtration_asset_id,
        )

        confidence = self.filtration_model.predict_proba(features)
        ctx["_filtration_confidence"] = confidence
        return confidence >= self.filtration_threshold

    def _smoothed_sigma_p(self, raw: float, ctx: dict) -> float:
        """EMA smoothing for contract mid vol (separate state from BTC sigma)."""
        if raw == 0.0:
            return 0.0
        ema = ctx.get("_sigma_p_ema")
        if ema is None:
            ema = raw
        else:
            ema = self.sigma_ema_alpha * raw + (1 - self.sigma_ema_alpha) * ema
        ctx["_sigma_p_ema"] = ema
        return ema

    def _contract_sigma_p(self, ctx: dict) -> float:
        """Realized vol of contract mid price (sigma_p per second).

        Uses the same Yang-Zhang estimator as BTC vol but on the UP
        contract mid = (best_bid + best_ask) / 2 over a rolling window.
        """
        mids = ctx.get("_contract_mids", [])
        ts = ctx.get("_contract_mid_ts", [])
        if len(mids) < 10:
            return 0.0  # insufficient data — min_edge floor covers warmup
        cutoff = ts[-1] - self.contract_vol_lookback_s * 1000
        start = next((i for i, t in enumerate(ts) if t >= cutoff), 0)
        raw = _compute_vol_deduped(mids[start:], ts[start:])
        return self._smoothed_sigma_p(raw, ctx)

    @staticmethod
    def _compute_toxicity(snapshot: "Snapshot", max_spread: float) -> float:
        """Composite toxicity score in [0, 1] from microstructure signals.

        Components (each normalized to [0, 1]):
          1. Spread width:   avg(spread_up, spread_down) / max_spread
          2. Book imbalance: abs(total_bid_depth - total_ask_depth) / total_depth
          3. Parity gap:     abs(up_mid + down_mid - 1.0)

        Final score = weighted average (40% spread, 30% imbalance, 30% gap).
        """
        bid_up = snapshot.best_bid_up or 0.0
        ask_up = snapshot.best_ask_up or 1.0
        bid_down = snapshot.best_bid_down or 0.0
        ask_down = snapshot.best_ask_down or 1.0

        # 1. Normalized spread
        spread_up = ask_up - bid_up
        spread_down = ask_down - bid_down
        avg_spread = (spread_up + spread_down) / 2.0
        spread_score = min(1.0, avg_spread / max_spread) if max_spread > 0 else 0.0

        # 2. Book imbalance across all levels
        bid_depth_up = sum(sz for _, sz in snapshot.bid_levels_up)
        ask_depth_up = sum(sz for _, sz in snapshot.ask_levels_up)
        bid_depth_down = sum(sz for _, sz in snapshot.bid_levels_down)
        ask_depth_down = sum(sz for _, sz in snapshot.ask_levels_down)
        total_bid = bid_depth_up + bid_depth_down
        total_ask = ask_depth_up + ask_depth_down
        total_depth = total_bid + total_ask
        imbalance_score = abs(total_bid - total_ask) / total_depth if total_depth > 0 else 0.0

        # 3. Mid-parity deviation
        # Book mid = average of up-mid and (1 - down-mid) — they should agree
        mid_up = (bid_up + ask_up) / 2.0
        mid_down = (bid_down + ask_down) / 2.0
        # Implied probability from up and down mids should sum to ~1.0
        # Deviation from this parity signals mispricing / toxic flow
        parity_gap = abs((mid_up + mid_down) - 1.0)
        gap_score = min(1.0, parity_gap / 0.10)  # normalize: 0.10 gap = score 1.0

        return 0.40 * spread_score + 0.30 * imbalance_score + 0.30 * gap_score

    @staticmethod
    def _compute_vpin(bars, window: int) -> float:
        """Volume-weighted VPIN: sum(|sell - buy|) / sum(total) over last W bars.

        Returns 0.0 when insufficient bars (graceful warmup).
        """
        if len(bars) < window:
            return 0.0
        recent = list(bars)[-window:]
        imbalance_sum = 0.0
        volume_sum = 0.0
        for buy_vol, sell_vol in recent:
            imbalance_sum += abs(sell_vol - buy_vol)
            volume_sum += buy_vol + sell_vol
        if volume_sum <= 0:
            return 0.0
        return imbalance_sum / volume_sum

    @staticmethod
    def _compute_oracle_lag(chainlink_price: float, binance_mid) -> float:
        """Price discrepancy between Binance mid and Chainlink oracle.

        Returns abs(binance_mid - chainlink) / chainlink.
        Returns 0.0 if binance_mid is None (graceful degradation).
        """
        if binance_mid is None or chainlink_price <= 0:
            return 0.0
        return abs(binance_mid - chainlink_price) / chainlink_price

    def _model_cdf(self, z: float, ctx: dict) -> float:
        """CDF dispatch: normal, student_t, kou, or market_adaptive."""
        if self.tail_mode == "normal":
            return norm_cdf(z)
        if self.tail_mode == "kou":
            # Our sigma_per_s already includes jump variance (it's total sigma).
            # Kou's drift correction is the only new information.
            # zeta = p*eta1/(eta1-1) + q*eta2/(eta2+1) - 1
            eta1, eta2 = self.kou_eta1, self.kou_eta2
            p_up, q_dn = self.kou_p_up, 1.0 - self.kou_p_up
            zeta = (p_up * eta1 / max(eta1 - 1, 0.01)
                    + q_dn * eta2 / (eta2 + 1) - 1.0)
            sigma = ctx.get("_sigma_per_s", 1e-5)
            tau = ctx.get("_tau", 300.0)
            drift_z = -self.kou_lambda * zeta * math.sqrt(tau) / max(sigma, 1e-8)
            return norm_cdf(z + drift_z)
        if self.tail_mode == "market_adaptive":
            p_gbm = norm_cdf(z)
            market_mid = ctx.get("_market_mid", 0.5)
            choppiness = ctx.get("_choppiness", 1.0)
            elapsed_frac = ctx.get("_elapsed_frac", 0.5)

            # 1. Blend GBM with market price
            alpha = self.market_blend_alpha
            p_blend = alpha * p_gbm + (1.0 - alpha) * market_mid

            # 2. Choppiness discount: pull toward 0.5 when choppy
            # choppiness = actual_crossovers / expected_crossovers
            # >1 means choppier than GBM predicts → less confident
            chop_factor = min(1.0, 1.0 / max(choppiness, 0.3))
            p_chop = chop_factor * p_blend + (1.0 - chop_factor) * 0.5

            # 3. Time-confidence: S-curve, low at 0-25%, rises through 50%+
            # Logistic: f(t) = 1 / (1 + exp(-k*(t - t0)))
            t_conf = 1.0 / (1.0 + math.exp(-8.0 * (elapsed_frac - 0.35)))
            p_final = t_conf * p_chop + (1.0 - t_conf) * 0.5
            return p_final
        # student_t fallback
        nu = ctx.get("_tail_nu", self.tail_nu_default)
        return fast_t_cdf(z, nu)

    def _p_model(self, z_capped: float, tau: float, ctx: dict | None = None) -> float:
        """Model probability of UP via Bayesian fusion of GBM + calibration.

        p = w * p_calibrated + (1 - w) * p_gbm
        w = n / (n + n0)

        With few observations, leans on GBM prior.
        With many observations, leans on calibration.
        """
        p_gbm = self._model_cdf(z_capped, ctx) if ctx is not None else norm_cdf(z_capped)
        if self.calibration_table is not None:
            p_cal, n = self.calibration_table.lookup_with_count(z_capped, tau)
            if n > 0:
                w = min(n / (n + self.cal_prior_strength), self.cal_max_weight)
                return w * p_cal + (1 - w) * p_gbm
        return p_gbm

    def decide(self, snapshot: Snapshot, ctx: dict) -> Decision:
        effective_price = ctx.get("_binance_mid") or snapshot.chainlink_price

        # Build price + timestamp history
        hist = ctx.setdefault("price_history", [])
        ts_hist = ctx.setdefault("ts_history", [])
        if ctx.pop("_live_history_appended", False):
            pass  # signal_ticker already appended this tick
        else:
            hist.append(effective_price)
            ts_hist.append(snapshot.ts_ms)

        if len(hist) < 2:
            return Decision("FLAT", 0.0, 0.0,
                            f"need history ({len(hist)}s collected)")

        ask_up = snapshot.best_ask_up
        ask_down = snapshot.best_ask_down
        bid_up = snapshot.best_bid_up
        bid_down = snapshot.best_bid_down
        if ask_up is None or ask_down is None or bid_up is None or bid_down is None:
            return Decision("FLAT", 0.0, 0.0, "missing book")
        if ask_up <= 0 or ask_up >= 1 or ask_down <= 0 or ask_down >= 1:
            return Decision("FLAT", 0.0, 0.0, "invalid asks")
        self._record_book_state(snapshot, ctx)

        # Spread gate: wide spreads signal uncertain/illiquid pricing
        spread_up = ask_up - bid_up
        spread_down = ask_down - bid_down
        if spread_up > self.max_spread or spread_down > self.max_spread:
            return Decision("FLAT", 0.0, 0.0,
                f"spread too wide (up={spread_up:.3f} down={spread_down:.3f} "
                f"max={self.max_spread})")

        # Microstructure toxicity
        toxicity = self._compute_toxicity(snapshot, self.max_spread)
        ctx["_toxicity"] = toxicity

        tau = snapshot.time_remaining_s
        if tau <= 0:
            return Decision("FLAT", 0.0, 0.0, "window expired")

        # Late-entry gate: only trade when time remaining <= max_entry_time_s
        if self.max_entry_time_s is not None and tau > self.max_entry_time_s:
            return Decision("FLAT", 0.0, 0.0,
                f"too early ({tau:.0f}s left > {self.max_entry_time_s:.0f}s gate)")

        # Dynamic edge threshold: higher early, decays with sqrt(tau)
        dyn_threshold = self.edge_threshold * (
            1.0 + self.early_edge_mult * math.sqrt(tau / self.window_duration)
        )

        # Toxicity penalty: widen threshold in adverse microstructure
        if toxicity > self.toxicity_threshold:
            excess = (toxicity - self.toxicity_threshold) / (1.0 - self.toxicity_threshold)
            dyn_threshold *= 1.0 + self.toxicity_edge_mult * excess

        # VPIN flow toxicity penalty
        trade_bars = ctx.get("_trade_bars", [])
        vpin = self._compute_vpin(trade_bars, self.vpin_window)
        ctx["_vpin"] = vpin
        if vpin > self.vpin_threshold:
            vpin_excess = (vpin - self.vpin_threshold) / (1.0 - self.vpin_threshold)
            dyn_threshold *= 1.0 + self.vpin_edge_mult * vpin_excess

        # Oracle lag penalty: widen threshold when Binance mid diverges from Chainlink
        binance_mid = ctx.get("_binance_mid")
        oracle_lag = self._compute_oracle_lag(snapshot.chainlink_price, binance_mid)
        ctx["_oracle_lag"] = oracle_lag
        # Hard cancel: skip this tick entirely when gap exceeds cancel threshold
        if self.oracle_cancel_threshold > 0 and oracle_lag > self.oracle_cancel_threshold:
            return Decision("FLAT", 0.0, 0.0,
                            f"oracle hard cancel (lag={oracle_lag:.4f} > {self.oracle_cancel_threshold:.4f})")
        if oracle_lag > self.oracle_lag_threshold:
            lag_excess = min((oracle_lag - self.oracle_lag_threshold) / self.oracle_lag_threshold, 1.0)
            dyn_threshold *= 1.0 + self.oracle_lag_mult * lag_excess

        # Realized vol (short window for model)
        raw_sigma = self._compute_vol(hist[-self.vol_lookback_s:], ts_hist[-self.vol_lookback_s:])

        # Vol regime filter: compare recent vol to longer baseline
        if len(hist) >= self.vol_regime_lookback_s:
            sigma_baseline = self._compute_vol(hist[-self.vol_regime_lookback_s:], ts_hist[-self.vol_regime_lookback_s:])
            if sigma_baseline > 0 and raw_sigma > self.vol_regime_mult * sigma_baseline:
                return Decision("FLAT", 0.0, 0.0,
                    f"vol spike ({raw_sigma:.2e} > "
                    f"{self.vol_regime_mult}x baseline {sigma_baseline:.2e})")

        # EMA smoothing + asset cap
        sigma_per_s = self._smoothed_sigma(raw_sigma, ctx)
        if sigma_per_s == 0.0:
            return Decision("FLAT", 0.0, 0.0, "zero vol")

        # Vol kill switch
        if self.vol_kill_sigma is not None and sigma_per_s > self.vol_kill_sigma:
            return Decision("FLAT", 0.0, 0.0,
                f"vol kill switch (sigma={sigma_per_s:.2e} "
                f"> {self.vol_kill_sigma:.2e})")

        self._maybe_update_tail_nu(hist, ctx)

        # Near-expiry Chainlink blend (same as decide_both_sides)
        price = effective_price
        if (self.chainlink_blend_s > 0
                and tau < self.chainlink_blend_s
                and snapshot.chainlink_price
                and snapshot.chainlink_price > 0):
            blend_w = 1.0 - (tau / self.chainlink_blend_s)
            price = (1.0 - blend_w) * effective_price + blend_w * snapshot.chainlink_price

        # Model probability (z capped to prevent overconfidence)
        delta = (price - snapshot.window_start_price) / snapshot.window_start_price
        z_raw = delta / (sigma_per_s * math.sqrt(tau))

        # Regime-scaled z (same as decide_both_sides)
        z_raw, _ = self._apply_regime_z_scale(z_raw, sigma_per_s, ctx)

        z = max(-self.max_z, min(self.max_z, z_raw))
        ctx["_z_raw"] = z_raw
        ctx["_z"] = z
        # Populate context for kou / market_adaptive CDF
        ctx["_sigma_per_s"] = sigma_per_s
        ctx["_tau"] = tau
        dur_s = ctx.get("_window_duration_s", tau + 1.0)
        if "_window_duration_s" not in ctx:
            ctx["_window_duration_s"] = snapshot.time_remaining_s  # first call ≈ full duration
        ctx["_elapsed_frac"] = 1.0 - tau / dur_s if dur_s > 0 else 0.5
        ctx["_market_mid"] = getattr(snapshot, "mid_up", 0.5) or 0.5
        # choppiness: track zero-crossings of delta
        _prev_sign = ctx.get("_prev_delta_sign", 0)
        _cur_sign = 1 if delta > 0 else (-1 if delta < 0 else 0)
        if _prev_sign != 0 and _cur_sign != 0 and _cur_sign != _prev_sign:
            ctx["_crossovers"] = ctx.get("_crossovers", 0) + 1
        ctx["_prev_delta_sign"] = _cur_sign
        # expected crossovers under GBM ≈ sqrt(2 * elapsed / pi) * sigma_factor
        elapsed_s = dur_s - tau
        expected_xo = max(1.0, math.sqrt(2.0 * max(elapsed_s, 1.0) / math.pi))
        ctx["_choppiness"] = ctx.get("_crossovers", 0) / expected_xo

        # Minimum z gate: filter low-conviction setups
        if self.min_entry_z > 0 and abs(z) < self.min_entry_z:
            return Decision("FLAT", 0.0, 0.0,
                            f"min_z gate (|z|={abs(z):.3f} < {self.min_entry_z:.3f})")

        # Cross-asset disagreement veto
        if self.cross_asset_z_lookup is not None:
            wstart = ctx.get("_window_start_ms")
            z2 = _lookup_cross_asset_z(self.cross_asset_z_lookup, wstart, tau)
            if (z2 is not None
                    and abs(z) >= self.cross_asset_min_z
                    and abs(z2) >= self.cross_asset_min_z
                    and z * z2 < 0):
                return Decision("FLAT", 0.0, 0.0,
                                f"cross-asset veto (z={z:.3f}, z2={z2:.3f})")

        p_model = self._p_model(z, tau, ctx)
        ctx["_p_model_raw"] = p_model
        ctx["_p_display"] = norm_cdf(z)
        ctx["_sigma_per_s"] = sigma_per_s

        # Filtration gate: XGBoost confidence check
        if not self._check_filtration(z, sigma_per_s, tau, snapshot, ctx):
            conf = ctx.get("_filtration_confidence", 0.0)
            return Decision("FLAT", 0.0, 0.0,
                            f"filtration ({conf:.3f} < {self.filtration_threshold})")

        # Mean-reversion discount: pull p_model toward 0.5
        if self.reversion_discount > 0:
            p_model = p_model * (1 - self.reversion_discount) + 0.5 * self.reversion_discount

        # Market blend: pull p_model toward contract mid (anti-fade-the-market)
        if self.market_blend > 0:
            mid_up = (bid_up + ask_up) / 2.0
            p_model = (1.0 - self.market_blend) * p_model + self.market_blend * mid_up
            p_model = max(0.01, min(0.99, p_model))

        ctx["_p_model_trade"] = p_model

        # Effective costs (taker mode)
        p_up_cost = ask_up + poly_fee(ask_up) + self.slippage
        p_down_cost = ask_down + poly_fee(ask_down) + self.slippage

        # Edges (penalized by spread — wider spread = less trustworthy pricing)
        edge_up = p_model - p_up_cost - self.spread_edge_penalty * spread_up
        edge_down = (1.0 - p_model) - p_down_cost - self.spread_edge_penalty * spread_down
        ctx["_edge_up"] = edge_up
        ctx["_edge_down"] = edge_down
        ctx["_dyn_threshold_up"] = dyn_threshold
        ctx["_dyn_threshold_down"] = dyn_threshold

        if edge_up >= edge_down and edge_up > dyn_threshold:
            side, edge, eff_price, p_side = "BUY_UP", edge_up, p_up_cost, p_model
        elif edge_down > edge_up and edge_down > dyn_threshold:
            side, edge, eff_price, p_side = "BUY_DOWN", edge_down, p_down_cost, 1.0 - p_model
        else:
            return Decision("FLAT", 0.0, 0.0,
                            f"no edge (up={edge_up:.4f} down={edge_down:.4f} "
                            f"thresh={dyn_threshold:.4f})")

        if eff_price > self.max_entry_price:
            return Decision("FLAT", 0.0, 0.0,
                f"entry {eff_price:.3f} > max {self.max_entry_price:.2f}")

        # Momentum confirmation: majority of recent prices must be on
        # the same side of start price — prevents whipsawing when price
        # oscillates near the start price.
        start_px = snapshot.window_start_price
        mom_n = min(self.momentum_lookback_s, len(hist))
        if mom_n >= 2:
            mom_prices = hist[-mom_n:]
            if side == "BUY_UP":
                frac_ok = sum(1 for p in mom_prices if p >= start_px) / len(mom_prices)
                if frac_ok < self.momentum_majority:
                    return Decision("FLAT", 0.0, 0.0,
                        f"momentum fail: only {frac_ok:.0%} above start in last {mom_n}s")
            else:
                frac_ok = sum(1 for p in mom_prices if p <= start_px) / len(mom_prices)
                if frac_ok < self.momentum_majority:
                    return Decision("FLAT", 0.0, 0.0,
                        f"momentum fail: only {frac_ok:.0%} below start in last {mom_n}s")

        # Order book imbalance: require buy pressure on the chosen side
        # imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
        if side == "BUY_UP":
            bid_d = sum(sz for _, sz in snapshot.bid_levels_up)
            ask_d = sum(sz for _, sz in snapshot.ask_levels_up)
        else:
            bid_d = sum(sz for _, sz in snapshot.bid_levels_down)
            ask_d = sum(sz for _, sz in snapshot.ask_levels_down)
        total_d = bid_d + ask_d
        imbalance = (bid_d - ask_d) / total_d if total_d > 0 else 0.0
        if imbalance < 0:
            return Decision("FLAT", 0.0, 0.0,
                f"imbalance disagrees ({imbalance:+.3f} for {side})")

        # Delta velocity: require price moving in the direction of the bet.
        # OLS slope over last 30s of BTC prices.
        vel_n = min(30, len(hist))
        if vel_n >= 5:
            recent = hist[-vel_n:]
            x_mean = (vel_n - 1) / 2.0
            y_mean = sum(recent) / vel_n
            num = sum((i - x_mean) * (p - y_mean) for i, p in enumerate(recent))
            den = sum((i - x_mean) ** 2 for i in range(vel_n))
            slope = num / den if den > 0 else 0.0
            if side == "BUY_UP" and slope < 0:
                return Decision("FLAT", 0.0, 0.0,
                    f"delta velocity disagrees (slope={slope:+.2f} $/s for BUY_UP)")
            elif side == "BUY_DOWN" and slope > 0:
                return Decision("FLAT", 0.0, 0.0,
                    f"delta velocity disagrees (slope={slope:+.2f} $/s for BUY_DOWN)")

        if eff_price >= 1.0:
            return Decision("FLAT", 0.0, 0.0, "eff price >= 1")

        # Half-Kelly
        kelly_f = max(0.0, (p_side - eff_price) / (1.0 - eff_price))
        frac = min(self.kelly_fraction * kelly_f, self.max_bet_fraction)
        if frac <= 0:
            return Decision("FLAT", 0.0, 0.0, "kelly <= 0")

        size_usd = self.bankroll * frac

        # Floor to exchange minimum (5 shares), skip if bankroll can't cover
        min_usd = self.min_order_shares * eff_price
        if size_usd < min_usd:
            if self.bankroll >= min_usd:
                size_usd = min_usd
            else:
                return Decision("FLAT", 0.0, 0.0,
                    f"bankroll ${self.bankroll:.2f} < min order ${min_usd:.2f} (5 shares)")

        # Store expected price range in ctx for the engine to attach to Fill
        price = snapshot.chainlink_price
        move_1sig = sigma_per_s * math.sqrt(tau) * price
        ctx["_expected_range"] = {
            "btc_at_fill": price,
            "start_price": snapshot.window_start_price,
            "expected_low": snapshot.window_start_price - move_1sig,
            "expected_high": snapshot.window_start_price + move_1sig,
        }

        sprd = spread_up if side == "BUY_UP" else spread_down
        reason = (
            f"p={p_model:.4f} sig={sigma_per_s:.2e} z={z:.2f}"
            f"{'(cap)' if abs(z_raw) > self.max_z else ''} "
            f"tau={tau:.0f}s edge={edge:.4f} thresh={dyn_threshold:.4f} "
            f"spread={sprd:.3f} kelly={kelly_f:.4f} ${size_usd:.0f}"
        )
        return Decision(side, edge, size_usd, reason)

    def _size_decision(
        self, side: str, edge: float, eff_price: float,
        p_side: float, snapshot: Snapshot, sigma_per_s: float,
        tau: float, z: float, z_raw: float, p_model: float,
        dyn_threshold: float, spread: float,
    ) -> Decision:
        """Shared sizing logic for a single side. Returns a Decision."""
        if eff_price >= 1.0:
            return Decision("FLAT", 0.0, 0.0, "eff price >= 1")
        if eff_price > self.max_entry_price:
            return Decision("FLAT", 0.0, 0.0,
                f"entry {eff_price:.3f} > max {self.max_entry_price:.2f}")
        if eff_price < self.min_entry_price:
            return Decision("FLAT", 0.0, 0.0,
                f"entry {eff_price:.2f} < min {self.min_entry_price:.2f}")

        kelly_f = max(0.0, (p_side - eff_price) / (1.0 - eff_price))
        frac = min(self.kelly_fraction * kelly_f, self.max_bet_fraction)
        if frac <= 0:
            return Decision("FLAT", 0.0, 0.0, "kelly <= 0")

        size_usd = self.bankroll * frac
        min_usd = self.min_order_shares * eff_price
        if size_usd < min_usd:
            if self.bankroll >= min_usd:
                size_usd = min_usd
            else:
                return Decision("FLAT", 0.0, 0.0,
                    f"bankroll ${self.bankroll:.2f} < min order ${min_usd:.2f}")

        # Store expected range
        price = snapshot.chainlink_price
        move_1sig = sigma_per_s * math.sqrt(tau) * price

        reason = (
            f"p={p_model:.4f} sig={sigma_per_s:.2e} z={z:.2f}"
            f"{'(cap)' if abs(z_raw) > self.max_z else ''} "
            f"tau={tau:.0f}s edge={edge:.4f} thresh={dyn_threshold:.4f} "
            f"spread={spread:.3f} kelly={kelly_f:.4f} ${size_usd:.0f}"
        )
        return Decision(side, edge, size_usd, reason)

    def decide_both_sides(
        self, snapshot: Snapshot, ctx: dict,
    ) -> tuple[Decision, Decision]:
        """Evaluate Up and Down independently for maker mode.

        Returns (up_decision, down_decision) — both can be non-FLAT.
        Skips momentum, imbalance, and velocity filters.
        Uses mid-price with 0% fee for cost calculation.
        """
        flat = Decision("FLAT", 0.0, 0.0, "")

        effective_price = ctx.get("_binance_mid") or snapshot.chainlink_price

        # Build price + timestamp history (always, even during warmup, so vol is ready)
        # In live mode, signal_ticker already appended and sets a flag.
        hist = ctx.setdefault("price_history", [])
        ts_hist = ctx.setdefault("ts_history", [])
        if ctx.pop("_live_history_appended", False):
            pass  # signal_ticker already appended this tick
        else:
            hist.append(effective_price)
            ts_hist.append(snapshot.ts_ms)

        # Early p_model for display: compute whenever possible so the
        # dashboard shows live probabilities even during warmup / gates.
        # Uses raw (unsmoothed) sigma and pure GBM norm_cdf (not
        # calibration table) so the display updates continuously —
        # calibration bins are wide and would cause the value to appear
        # stuck.  The full computation later overwrites _p_model_raw
        # with the refined calibrated value for trading decisions.
        # Require enough history for a reliable vol estimate — too few
        # samples give a tiny sigma that sends z to the ±max_z cap,
        # making every market show the same p_up/p_down.
        _min_hist_display = max(10, self.vol_lookback_s // 3)
        if len(hist) >= _min_hist_display and snapshot.time_remaining_s > 0 and snapshot.window_start_price:
            _raw = self._compute_vol(
                hist[-self.vol_lookback_s:], ts_hist[-self.vol_lookback_s:]
            )
            if _raw > 0:
                _tau = snapshot.time_remaining_s
                _delta = (effective_price - snapshot.window_start_price) / snapshot.window_start_price
                _z = _delta / (_raw * math.sqrt(_tau))
                _z = max(-self.max_z, min(self.max_z, _z))
                ctx["_p_display"] = norm_cdf(_z)
                ctx["_p_model_raw"] = self._p_model(_z, _tau, ctx)
                ctx["_sigma_per_s"] = _raw

        # NOTE: maker warmup and end-of-window withdrawal are handled by
        # _evaluate_maker() in tracker.py.  We must NOT early-return here
        # because that blocks the full model computation (vol, z, p_model,
        # edges) which the dashboard and fill logs need every tick.

        if len(hist) < 2:
            reason = f"need history ({len(hist)}s collected)"
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))

        ask_up = snapshot.best_ask_up
        ask_down = snapshot.best_ask_down
        bid_up = snapshot.best_bid_up
        bid_down = snapshot.best_bid_down
        if ask_up is None or ask_down is None or bid_up is None or bid_down is None:
            reason = "missing book"
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))
        if ask_up <= 0 or ask_up >= 1 or ask_down <= 0 or ask_down >= 1:
            reason = "invalid asks"
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))
        self._record_book_state(snapshot, ctx)

        # Spread gate
        spread_up = ask_up - bid_up
        spread_down = ask_down - bid_down
        if spread_up > self.max_spread or spread_down > self.max_spread:
            reason = (f"spread too wide (up={spread_up:.3f} down={spread_down:.3f} "
                      f"max={self.max_spread})")
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))

        # Track contract mid prices for A-S realized vol (sigma_p)
        if self.as_mode:
            mid_up = (bid_up + ask_up) / 2.0
            contract_mids = ctx.setdefault("_contract_mids", [])
            contract_mid_ts = ctx.setdefault("_contract_mid_ts", [])
            contract_mids.append(mid_up)
            contract_mid_ts.append(snapshot.ts_ms)

        # VAMP computation
        vamp_up = compute_vamp(snapshot.bid_levels_up, snapshot.ask_levels_up)
        vamp_down = compute_vamp(snapshot.bid_levels_down, snapshot.ask_levels_down)

        # VAMP filter: skip when book is too gappy (large gap between VAMP and best bid)
        if self.vamp_mode == "filter":
            if (vamp_up is not None and
                    abs(vamp_up - bid_up) > self.vamp_filter_threshold):
                reason = (f"vamp filter (up gap={abs(vamp_up - bid_up):.3f} "
                          f"> {self.vamp_filter_threshold})")
                return (Decision("FLAT", 0.0, 0.0, reason),
                        Decision("FLAT", 0.0, 0.0, reason))
            if (vamp_down is not None and
                    abs(vamp_down - bid_down) > self.vamp_filter_threshold):
                reason = (f"vamp filter (down gap={abs(vamp_down - bid_down):.3f} "
                          f"> {self.vamp_filter_threshold})")
                return (Decision("FLAT", 0.0, 0.0, reason),
                        Decision("FLAT", 0.0, 0.0, reason))

        # Microstructure toxicity filter: composite score from spread,
        # book imbalance, and mid-parity gap.  Rather than a hard cutoff,
        # toxicity widens the edge threshold proportionally so we still
        # trade in mildly toxic regimes but at a higher bar.
        toxicity = self._compute_toxicity(snapshot, self.max_spread)
        ctx["_toxicity"] = toxicity

        tau = snapshot.time_remaining_s
        if tau <= 0:
            reason = "window expired"
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))

        # Vol
        raw_sigma = self._compute_vol(hist[-self.vol_lookback_s:], ts_hist[-self.vol_lookback_s:])

        # Vol regime filter
        if len(hist) >= self.vol_regime_lookback_s:
            sigma_baseline = self._compute_vol(hist[-self.vol_regime_lookback_s:], ts_hist[-self.vol_regime_lookback_s:])
            if sigma_baseline > 0 and raw_sigma > self.vol_regime_mult * sigma_baseline:
                reason = (f"vol spike ({raw_sigma:.2e} > "
                          f"{self.vol_regime_mult}x baseline {sigma_baseline:.2e})")
                return (Decision("FLAT", 0.0, 0.0, reason),
                        Decision("FLAT", 0.0, 0.0, reason))

        # EMA smoothing + asset cap
        sigma_per_s = self._smoothed_sigma(raw_sigma, ctx)
        if sigma_per_s == 0.0:
            reason = "zero vol"
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))

        # Volatility kill switch: hard pause when EMA sigma exceeds
        # an absolute ceiling.  Distinct from the relative regime filter
        # above — this catches sustained high-vol episodes where the
        # baseline itself has drifted up.
        if self.vol_kill_sigma is not None and sigma_per_s > self.vol_kill_sigma:
            reason = (f"vol kill switch (sigma={sigma_per_s:.2e} "
                      f"> {self.vol_kill_sigma:.2e})")
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))

        self._maybe_update_tail_nu(hist, ctx)

        # Microstructure metrics: compute excess values for threshold/spread
        trade_bars = ctx.get("_trade_bars", [])
        vpin = self._compute_vpin(trade_bars, self.vpin_window)
        ctx["_vpin"] = vpin
        tox_excess = (max(0.0, (toxicity - self.toxicity_threshold)
                         / (1.0 - self.toxicity_threshold))
                      if toxicity > self.toxicity_threshold else 0.0)
        vpin_excess = (max(0.0, (vpin - self.vpin_threshold)
                          / (1.0 - self.vpin_threshold))
                       if vpin > self.vpin_threshold else 0.0)

        binance_mid = ctx.get("_binance_mid")
        oracle_lag = self._compute_oracle_lag(snapshot.chainlink_price, binance_mid)
        ctx["_oracle_lag"] = oracle_lag
        # Hard cancel: skip both sides when gap exceeds cancel threshold
        if self.oracle_cancel_threshold > 0 and oracle_lag > self.oracle_cancel_threshold:
            reason = f"oracle hard cancel (lag={oracle_lag:.4f} > {self.oracle_cancel_threshold:.4f})"
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))
        lag_excess = (min((oracle_lag - self.oracle_lag_threshold)
                         / self.oracle_lag_threshold, 1.0)
                      if oracle_lag > self.oracle_lag_threshold else 0.0)

        # Legacy multiplicative threshold (used when as_mode=False)
        if not self.as_mode:
            window_trades = ctx.get("window_trade_count", 0)
            base_threshold = self.edge_threshold + self.edge_threshold_step * window_trades
            dyn_threshold = base_threshold * (
                1.0 + self.early_edge_mult * math.sqrt(tau / self.window_duration)
            )
            if tox_excess > 0:
                dyn_threshold *= 1.0 + self.toxicity_edge_mult * tox_excess
            if vpin_excess > 0:
                dyn_threshold *= 1.0 + self.vpin_edge_mult * vpin_excess
            if lag_excess > 0:
                dyn_threshold *= 1.0 + self.oracle_lag_mult * lag_excess
        else:
            dyn_threshold = 0.0  # placeholder, A-S computes opt_spread later

        # Near-expiry Chainlink blend: as tau → 0, Chainlink determines
        # resolution so we blend the pricing price from Binance toward
        # Chainlink to avoid betting on a Binance-Chainlink divergence.
        # Vol estimation keeps using Binance (more accurate tick data).
        price = effective_price
        chainlink_blend_w = 0.0
        if (self.chainlink_blend_s > 0
                and tau < self.chainlink_blend_s
                and snapshot.chainlink_price
                and snapshot.chainlink_price > 0):
            chainlink_blend_w = 1.0 - (tau / self.chainlink_blend_s)
            price = ((1.0 - chainlink_blend_w) * effective_price
                     + chainlink_blend_w * snapshot.chainlink_price)
        ctx["_chainlink_blend_w"] = chainlink_blend_w

        # z normalization: fractional delta / (sigma * sqrt(tau))
        delta = (price - snapshot.window_start_price) / snapshot.window_start_price
        z_raw = delta / (sigma_per_s * math.sqrt(tau))

        # Regime-scaled z: adjust z for current vol regime relative to
        # the calibration-period vol.  High-vol regimes shrink z (wider
        # distribution → less confident directional signal), low-vol
        # regimes amplify it.  Clamped to [0.5, 2.0] to avoid extreme
        # suppression or overconfidence from a single bar.
        z_raw, _ = self._apply_regime_z_scale(z_raw, sigma_per_s, ctx)

        z = max(-self.max_z, min(self.max_z, z_raw))
        ctx["_z_raw"] = z_raw
        ctx["_z"] = z
        # Populate context for kou / market_adaptive CDF
        ctx["_sigma_per_s"] = sigma_per_s
        ctx["_tau"] = tau
        dur_s = ctx.get("_window_duration_s", tau + 1.0)
        if "_window_duration_s" not in ctx:
            ctx["_window_duration_s"] = snapshot.time_remaining_s
        ctx["_elapsed_frac"] = 1.0 - tau / dur_s if dur_s > 0 else 0.5
        ctx["_market_mid"] = getattr(snapshot, "mid_up", 0.5) or 0.5
        _prev_sign = ctx.get("_prev_delta_sign", 0)
        _cur_sign = 1 if delta > 0 else (-1 if delta < 0 else 0)
        if _prev_sign != 0 and _cur_sign != 0 and _cur_sign != _prev_sign:
            ctx["_crossovers"] = ctx.get("_crossovers", 0) + 1
        ctx["_prev_delta_sign"] = _cur_sign
        elapsed_s = dur_s - tau
        expected_xo = max(1.0, math.sqrt(2.0 * max(elapsed_s, 1.0) / math.pi))
        ctx["_choppiness"] = ctx.get("_crossovers", 0) / expected_xo

        # Minimum z gate: filter low-conviction setups
        if self.min_entry_z > 0 and abs(z) < self.min_entry_z:
            reason = f"min_z gate (|z|={abs(z):.3f} < {self.min_entry_z:.3f})"
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))

        # Cross-asset disagreement veto
        if self.cross_asset_z_lookup is not None:
            wstart = ctx.get("_window_start_ms")
            z2 = _lookup_cross_asset_z(self.cross_asset_z_lookup, wstart, tau)
            if (z2 is not None
                    and abs(z) >= self.cross_asset_min_z
                    and abs(z2) >= self.cross_asset_min_z
                    and z * z2 < 0):
                reason = f"cross-asset veto (z={z:.3f}, z2={z2:.3f})"
                return (Decision("FLAT", 0.0, 0.0, reason),
                        Decision("FLAT", 0.0, 0.0, reason))

        p_model = self._p_model(z, tau, ctx)

        # Filtration gate: XGBoost confidence check
        if not self._check_filtration(z, sigma_per_s, tau, snapshot, ctx):
            conf = ctx.get("_filtration_confidence", 0.0)
            reason = f"filtration ({conf:.3f} < {self.filtration_threshold})"
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))

        # Expose model state so OrderMixin can snapshot it at fill time
        ctx["_p_model_raw"] = p_model
        ctx["_p_display"] = norm_cdf(z)  # smooth GBM for dashboard (no binning)
        ctx["_sigma_per_s"] = sigma_per_s
        ctx["_dyn_threshold_up"] = dyn_threshold

        if self.reversion_discount > 0:
            p_model = p_model * (1 - self.reversion_discount) + 0.5 * self.reversion_discount

        # Market blend: pull p_model toward contract mid (anti-fade-the-market).
        # When the model says BUY at $0.10 but the market trades at $0.10, the
        # market is usually right.  Backtested on 1159 BTC 5m windows: blend=0.3
        # turns -10% recent ROI into +10%.
        if self.market_blend > 0:
            mid_up = (bid_up + ask_up) / 2.0
            p_model = (1.0 - self.market_blend) * p_model + self.market_blend * mid_up
            p_model = max(0.01, min(0.99, p_model))

        # Order book imbalance → continuous p_model shift
        # OBI > 0 means more bid depth (buying pressure) → nudge p_up higher
        bid_depth_up = sum(sz for _, sz in snapshot.bid_levels_up)
        ask_depth_up = sum(sz for _, sz in snapshot.ask_levels_up)
        bid_depth_down = sum(sz for _, sz in snapshot.bid_levels_down)
        ask_depth_down = sum(sz for _, sz in snapshot.ask_levels_down)
        total_depth_up = bid_depth_up + ask_depth_up
        total_depth_down = bid_depth_down + ask_depth_down
        obi_up = (bid_depth_up - ask_depth_up) / total_depth_up if total_depth_up > 0 else 0.0
        obi_down = (bid_depth_down - ask_depth_down) / total_depth_down if total_depth_down > 0 else 0.0
        ctx["_obi_up"] = obi_up
        ctx["_obi_down"] = obi_down

        if self.obi_weight > 0:
            # Positive OBI on UP side → buy pressure → p_up higher
            # Positive OBI on DOWN side → buy pressure → p_down higher (p_up lower)
            p_model += self.obi_weight * (obi_up - obi_down)
            p_model = max(0.01, min(0.99, p_model))
        ctx["_p_model_trade"] = p_model

        # Cost basis: VAMP (volume-weighted mid) or best bid
        if self.vamp_mode == "cost":
            cost_up = vamp_up if vamp_up is not None else bid_up
            cost_down = vamp_down if vamp_down is not None else bid_down
        else:
            cost_up = bid_up
            cost_down = bid_down

        # Expose cost basis so OrderMixin can include it in model snapshot
        ctx["_cost_up"] = cost_up
        ctx["_cost_down"] = cost_down

        # Store expected range in ctx
        move_1sig = sigma_per_s * math.sqrt(tau) * price
        ctx["_expected_range"] = {
            "btc_at_fill": price,
            "start_price": snapshot.window_start_price,
            "expected_low": snapshot.window_start_price - move_1sig,
            "expected_high": snapshot.window_start_price + move_1sig,
        }

        # DOWN bonus: check book balance for down_edge_bonus
        down_bonus_active = False
        down_share = 0.5
        if self.down_edge_bonus > 0:
            total_d = total_depth_up + total_depth_down
            down_d = total_depth_down
            down_share = down_d / total_d if total_d > 0 else 0.5
            if 0.3 <= down_share <= 0.7:
                down_bonus_active = True
        ctx["_down_bonus_active"] = down_bonus_active
        ctx["_down_share"] = down_share

        # ── A-S unified quoting ───────────────────────────────────────
        if self.as_mode:
            # Realized contract vol (model-free, from Yang-Zhang on contract mids)
            sigma_p = self._contract_sigma_p(ctx)
            sigma_p_sq = sigma_p ** 2
            # A-S uses total remaining variance = sigma² * tau (NOT tau/T)
            total_var = sigma_p_sq * tau
            ctx["_sigma_p"] = sigma_p

            # Reservation price: shift belief by inventory risk
            # Higher total_var → larger penalty (more uncertain contract)
            n_up = ctx.get("inventory_up", 0)
            n_down = ctx.get("inventory_down", 0)
            q_up = n_up - n_down  # net long UP exposure
            inv_pen = self.gamma_inv * total_var
            r_up = p_model - inv_pen * q_up
            r_down = (1.0 - p_model) + inv_pen * q_up
            ctx["_inventory_skew"] = inv_pen

            # Edge from reservation value
            edge_up = r_up - cost_up - self.spread_edge_penalty * spread_up
            edge_down = r_down - cost_down - self.spread_edge_penalty * spread_down
            ctx["_edge_up"] = edge_up
            ctx["_edge_down"] = edge_down

            # Optimal spread: additive adverse selection components
            window_trades = ctx.get("window_trade_count", 0)
            base_spread = max(self.gamma_spread * total_var / 2.0,
                              self.min_edge)
            opt_spread = (base_spread
                          + self.tox_spread * tox_excess
                          + self.vpin_spread * vpin_excess
                          + self.lag_spread * lag_excess
                          + self.edge_step * window_trades)
            opt_spread_down = (opt_spread * (1.0 - self.down_edge_bonus)
                               if down_bonus_active else opt_spread)

            dyn_threshold = opt_spread
            dyn_threshold_down = opt_spread_down
            ctx["_dyn_threshold_up"] = opt_spread
            ctx["_dyn_threshold_down"] = opt_spread_down

            # Evaluate each side
            up_dec = flat
            down_dec = flat

            if edge_up > opt_spread:
                up_dec = self._size_decision(
                    "BUY_UP", edge_up, cost_up, r_up,
                    snapshot, sigma_per_s, tau, z, z_raw, p_model,
                    opt_spread, spread_up,
                )
            if edge_down > opt_spread_down:
                down_dec = self._size_decision(
                    "BUY_DOWN", edge_down, cost_down, r_down,
                    snapshot, sigma_per_s, tau, z, z_raw, p_model,
                    opt_spread_down, spread_down,
                )

        # ── Legacy multiplicative quoting ─────────────────────────────
        else:
            edge_up = p_model - cost_up - self.spread_edge_penalty * spread_up
            edge_down = (1.0 - p_model) - cost_down - self.spread_edge_penalty * spread_down
            ctx["_edge_up"] = edge_up
            ctx["_edge_down"] = edge_down

            # Inventory skew (legacy)
            if self.inventory_skew > 0:
                n_up = ctx.get("inventory_up", 0)
                n_down = ctx.get("inventory_down", 0)
                skew = self.inventory_skew * (tau / self.window_duration)
                ctx["_inventory_skew"] = skew
                edge_up -= skew * n_up
                edge_up += skew * n_down
                edge_down -= skew * n_down
                edge_down += skew * n_up

            dyn_threshold_down = dyn_threshold
            if down_bonus_active:
                dyn_threshold_down = dyn_threshold * (1.0 - self.down_edge_bonus)
            ctx["_dyn_threshold_down"] = dyn_threshold_down

            up_dec = flat
            down_dec = flat

            if edge_up > dyn_threshold:
                up_dec = self._size_decision(
                    "BUY_UP", edge_up, cost_up, p_model,
                    snapshot, sigma_per_s, tau, z, z_raw, p_model,
                    dyn_threshold, spread_up,
                )
            if edge_down > dyn_threshold_down:
                down_dec = self._size_decision(
                    "BUY_DOWN", edge_down, cost_down, 1.0 - p_model,
                    snapshot, sigma_per_s, tau, z, z_raw, p_model,
                    dyn_threshold_down, spread_down,
                )

        # If neither has edge, report combined reason
        if up_dec.action == "FLAT" and down_dec.action == "FLAT":
            if edge_up <= dyn_threshold and edge_down <= dyn_threshold_down:
                reason = (f"no edge (up={edge_up:.4f} down={edge_down:.4f} "
                          f"thresh={dyn_threshold:.4f}"
                          f"{f' down_thresh={dyn_threshold_down:.4f}' if dyn_threshold_down != dyn_threshold else ''})")
                up_dec = Decision("FLAT", 0.0, 0.0, reason)
                down_dec = Decision("FLAT", 0.0, 0.0, reason)

        # Min entry price gate: override any non-FLAT decision if bid is too cheap
        if self.min_entry_price > 0:
            if up_dec.action != "FLAT" and cost_up < self.min_entry_price:
                up_dec = Decision("FLAT", 0.0, 0.0,
                    f"entry {cost_up:.3f} < min {self.min_entry_price:.2f}")
            if down_dec.action != "FLAT" and cost_down < self.min_entry_price:
                down_dec = Decision("FLAT", 0.0, 0.0,
                    f"entry {cost_down:.3f} < min {self.min_entry_price:.2f}")

        return (up_dec, down_dec)


# ── Book walking ────────────────────────────────────────────────────────────

def walk_book(
    ask_levels: tuple[tuple[float, float], ...],
    desired_shares: float,
    slippage: float,
) -> tuple[float, float, float]:
    """
    Walk the ask side of the L2 book.
    Returns (filled_shares, total_cost, avg_price_per_share).
    total_cost includes fees and slippage.
    """
    if not ask_levels or desired_shares <= 0:
        return (0.0, 0.0, 0.0)

    filled = 0.0
    total_cost = 0.0
    raw_cost = 0.0

    for price, size in ask_levels:
        if filled >= desired_shares:
            break
        take = min(size, desired_shares - filled)
        fee = poly_fee(price)
        total_cost += take * (price + fee + slippage)
        raw_cost += take * price
        filled += take

    if filled <= 0:
        return (0.0, 0.0, 0.0)
    return (filled, total_cost, total_cost / filled)


# ── Backtest engine ─────────────────────────────────────────────────────────

class BacktestEngine:
    def __init__(
        self,
        signal: Signal,
        data_dir: Path = DATA_DIR,
        latency_ms: int = 0,
        slippage: float = 0.0,
        initial_bankroll: float = 10_000.0,
        max_trades_per_window: int = 1,
    ):
        self.signal = signal
        self.data_dir = data_dir
        self.latency_ms = latency_ms
        self.slippage = slippage
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.max_trades_per_window = max_trades_per_window

    def load_data(self) -> pd.DataFrame:
        if not self.data_dir.exists() or not any(self.data_dir.glob("*.parquet")):
            return pd.DataFrame()
        # Read each file individually and normalize column names before
        # concatenating.  Older files use "chainlink_btc", newer ones use
        # "chainlink_price".  Pyarrow drops mismatched columns when reading
        # a directory of mixed-schema parquets, so we must handle it here.
        frames = []
        for f in sorted(self.data_dir.glob("*.parquet")):
            part = pd.read_parquet(f)
            if "chainlink_btc" in part.columns and "chainlink_price" not in part.columns:
                part.rename(columns={"chainlink_btc": "chainlink_price"}, inplace=True)
            frames.append(part)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        if df.empty:
            return df
        df.sort_values("ts_ms", inplace=True, ignore_index=True)
        df.drop_duplicates(subset=["market_slug", "ts_ms"], keep="last", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def _resolve_window(self, window_df: pd.DataFrame) -> Optional[tuple[int, float]]:
        """Returns (outcome_up, final_btc) or None if incomplete."""
        final_remaining = window_df["time_remaining_s"].iloc[-1]
        if final_remaining > MIN_FINAL_REMAINING_S:
            return None

        # Skip windows where recording started too late (not a full window)
        if ("window_start_ms" in window_df.columns
                and "window_end_ms" in window_df.columns):
            window_dur_s = (
                window_df["window_end_ms"].iloc[0]
                - window_df["window_start_ms"].iloc[0]
            ) / 1000
            first_remaining = window_df["time_remaining_s"].iloc[0]
            if first_remaining < window_dur_s - MAX_START_GAP_S:
                return None

        start_price = window_df["window_start_price"].dropna()
        if start_price.empty:
            return None
        start_price = start_price.iloc[0]

        price_col = "chainlink_price" if "chainlink_price" in window_df.columns else "chainlink_btc"
        final_btc = float(window_df[price_col].iloc[-1])
        if pd.isna(final_btc) or pd.isna(start_price):
            return None

        outcome = 1 if final_btc >= start_price else 0
        return outcome, final_btc

    def _execute_fill(self, snap: Snapshot, decision: Decision,
                      ctx: dict) -> Optional[Fill]:
        if decision.action == "BUY_UP":
            side, ask_levels, best_ask = "UP", snap.ask_levels_up, snap.best_ask_up
            best_bid = snap.best_bid_up
        elif decision.action == "BUY_DOWN":
            side, ask_levels, best_ask = "DOWN", snap.ask_levels_down, snap.best_ask_down
            best_bid = snap.best_bid_down
        else:
            return None

        if not ask_levels or best_ask is None or best_ask <= 0:
            return None

        # Maker mode: fill at bid with 0% fee
        maker_mode = getattr(self.signal, "maker_mode", False)
        if maker_mode and best_bid is not None and best_bid > 0:
            entry_price = best_bid  # maker fills at bid: 0% fee
            if entry_price <= 0 or entry_price >= 1.0:
                return None
            desired_shares = decision.size_usd / entry_price
            filled = round(desired_shares, 1)
            if filled < 5.0:
                return None  # below exchange minimum
            total_cost = filled * entry_price
            if total_cost > self.bankroll:
                return None  # can't afford
            fee_avg = 0.0
        else:
            eff_est = best_ask + poly_fee(best_ask) + self.slippage
            if eff_est <= 0 or eff_est >= 1.0:
                return None

            desired_shares = decision.size_usd / eff_est
            filled, total_cost, avg_price = walk_book(ask_levels, desired_shares, self.slippage)

            if filled <= 0 or total_cost <= 0:
                return None

            entry_price = avg_price
            # Compute average raw price for fee reporting
            raw_total = 0.0
            temp = 0.0
            for px, sz in ask_levels:
                take = min(sz, desired_shares - temp)
                if take <= 0:
                    break
                raw_total += take * px
                temp += take
            raw_avg = raw_total / temp if temp > 0 else 0
            fee_avg = entry_price - raw_avg - self.slippage

        # Expected price range from signal (if available)
        rng = ctx.get("_expected_range", {})

        return Fill(
            market_slug=snap.market_slug,
            side=side,
            entry_ts_ms=snap.ts_ms,
            time_remaining_s=snap.time_remaining_s,
            entry_price=entry_price,
            fee_per_share=fee_avg,
            shares=filled,
            cost_usd=total_cost,
            signal_name=self.signal.name,
            decision_reason=decision.reason,
            btc_at_fill=rng.get("btc_at_fill", snap.chainlink_price),
            start_price=rng.get("start_price", snap.window_start_price),
            expected_low=rng.get("expected_low", 0.0),
            expected_high=rng.get("expected_high", 0.0),
        )

    def _run_window(
        self, window_df: pd.DataFrame, outcome_up: int,
        final_btc: float,
    ) -> list[TradeResult]:
        ctx: dict = {"inventory_up": 0, "inventory_down": 0}
        if "window_start_ms" in window_df.columns:
            ctx["_window_start_ms"] = int(window_df["window_start_ms"].iloc[0])
        pending: Optional[tuple[int, Decision]] = None
        results: list[TradeResult] = []
        last_fill_ts: int = 0
        filled_sides: set = set()  # anti-hedge: track filled sides per window
        cooldown_ms = 30_000  # minimum 30s between bets
        maker_mode = getattr(self.signal, "maker_mode", False)
        if maker_mode:
            cooldown_ms = 5_000

        has_binance = "binance_mid" in window_df.columns

        for _, row in window_df.iterrows():
            snap = Snapshot.from_row(row)
            if snap is None:
                continue

            # Inject Binance mid into ctx for sigma estimation (when available)
            if has_binance and pd.notna(row.get("binance_mid")) and row["binance_mid"] > 0:
                ctx["_binance_mid"] = float(row["binance_mid"])

            # Execute pending order after latency
            if pending is not None:
                exec_ts, decision = pending
                if snap.ts_ms >= exec_ts:
                    fill = self._execute_fill(snap, decision, ctx)
                    if fill is not None:
                        results.append(self._resolve_fill(fill, outcome_up, final_btc))
                        self.bankroll += results[-1].pnl
                        if hasattr(self.signal, "bankroll"):
                            self.signal.bankroll = self.bankroll
                        last_fill_ts = snap.ts_ms
                        ctx["window_trade_count"] = ctx.get("window_trade_count", 0) + 1
                        if "UP" in decision.action:
                            ctx["inventory_up"] = ctx.get("inventory_up", 0) + fill.shares
                        elif "DOWN" in decision.action:
                            ctx["inventory_down"] = ctx.get("inventory_down", 0) + fill.shares
                    pending = None

            # Cooldown between bets
            if snap.ts_ms - last_fill_ts < cooldown_ms and last_fill_ts > 0:
                continue

            # Max trades per window
            if self.max_trades_per_window is not None and len(results) >= self.max_trades_per_window:
                break

            # Maker mode: evaluate both sides independently
            if maker_mode and hasattr(self.signal, "decide_both_sides"):
                # Enforce maker warmup (matches live bot's _evaluate_maker)
                elapsed = self.signal.window_duration - snap.time_remaining_s
                if elapsed < self.signal.maker_warmup_s:
                    continue
                # Enforce maker withdraw
                if snap.time_remaining_s < getattr(self.signal, "maker_withdraw_s", 60.0):
                    continue

                up_dec, down_dec = self.signal.decide_both_sides(snap, ctx)
                for decision in [up_dec, down_dec]:
                    if decision.action != "FLAT" and decision.size_usd > 0:
                        if self.max_trades_per_window is not None and len(results) >= self.max_trades_per_window:
                            break
                        # Anti-hedge: don't bet both sides in the same window
                        side_label = "UP" if "UP" in decision.action else "DOWN"
                        opposite = "DOWN" if side_label == "UP" else "UP"
                        if opposite in filled_sides:
                            continue
                        fill = self._execute_fill(snap, decision, ctx)
                        if fill is not None:
                            filled_sides.add(side_label)
                            results.append(self._resolve_fill(fill, outcome_up, final_btc))
                            self.bankroll += results[-1].pnl
                            if hasattr(self.signal, "bankroll"):
                                self.signal.bankroll = self.bankroll
                            last_fill_ts = snap.ts_ms
                            ctx["window_trade_count"] = ctx.get("window_trade_count", 0) + 1
                            if "UP" in decision.action:
                                ctx["inventory_up"] = ctx.get("inventory_up", 0) + fill.shares
                            elif "DOWN" in decision.action:
                                ctx["inventory_down"] = ctx.get("inventory_down", 0) + fill.shares
                continue

            # FOK mode: run single-side signal
            decision = self.signal.decide(snap, ctx)
            if decision.action != "FLAT" and decision.size_usd > 0:
                if self.latency_ms <= 0:
                    fill = self._execute_fill(snap, decision, ctx)
                    if fill is not None:
                        results.append(self._resolve_fill(fill, outcome_up, final_btc))
                        self.bankroll += results[-1].pnl
                        if hasattr(self.signal, "bankroll"):
                            self.signal.bankroll = self.bankroll
                        last_fill_ts = snap.ts_ms
                        ctx["window_trade_count"] = ctx.get("window_trade_count", 0) + 1
                        if "UP" in decision.action:
                            ctx["inventory_up"] = ctx.get("inventory_up", 0) + fill.shares
                        elif "DOWN" in decision.action:
                            ctx["inventory_down"] = ctx.get("inventory_down", 0) + fill.shares
                else:
                    pending = (snap.ts_ms + self.latency_ms, decision)

        return results

    @staticmethod
    def _resolve_fill(fill: Fill, outcome_up: int, final_btc: float) -> TradeResult:
        won = (fill.side == "UP" and outcome_up == 1) or \
              (fill.side == "DOWN" and outcome_up == 0)
        payout = fill.shares if won else 0.0
        pnl = payout - fill.cost_usd
        pnl_pct = pnl / fill.cost_usd if fill.cost_usd > 0 else 0.0
        return TradeResult(fill=fill, outcome_up=outcome_up, final_btc=final_btc,
                           payout=payout, pnl=pnl, pnl_pct=pnl_pct)

    def _run_slug_list(
        self,
        df: pd.DataFrame,
        slugs: list[str],
        verbose: bool = True,
    ) -> tuple[list[TradeResult], list[float]]:
        """Run the engine over an ordered list of window slugs.

        One trade per window (max_trades_per_window enforced).
        Bankroll compounds between windows but NOT within a window.
        Returns (results, bankroll_history).
        """
        results: list[TradeResult] = []
        bankroll_hist = [self.bankroll]

        for slug in slugs:
            window_df = df[df["market_slug"] == slug]
            resolved = self._resolve_window(window_df)
            if resolved is None:
                if verbose:
                    print(f"  SKIP {slug} (incomplete)")
                continue

            outcome, final_btc = resolved
            if hasattr(self.signal, "bankroll"):
                self.signal.bankroll = self.bankroll

            pre_bankroll = self.bankroll
            window_results = self._run_window(window_df, outcome, final_btc)

            if window_results:
                results.extend(window_results)
                if verbose:
                    for r in window_results:
                        rem_m = int(r.fill.time_remaining_s) // 60
                        rem_s = int(r.fill.time_remaining_s) % 60
                        print(
                            f"  {slug}: {r.fill.side} "
                            f"@ {r.fill.entry_price:.4f} "
                            f"x {r.fill.shares:.1f}sh "
                            f"[{rem_m}:{rem_s:02d} left] "
                            f"-> {'UP' if r.outcome_up else 'DOWN'} "
                            f"pnl=${r.pnl:+.2f} ({r.pnl_pct:+.1%})"
                        )
                    window_pnl = self.bankroll - pre_bankroll
                    print(f"    window net: ${window_pnl:+.2f}  "
                          f"bank=${self.bankroll:.2f}")
            elif verbose:
                print(f"  {slug}: FLAT")

            bankroll_hist.append(self.bankroll)

        return results, bankroll_hist

    def run(self) -> tuple[list[TradeResult], dict, pd.DataFrame]:
        df = self.load_data()
        if df.empty:
            print("  No data found.")
            metrics = self._compute_metrics([], [self.bankroll])
            return [], metrics, pd.DataFrame()
        slugs = list(df["market_slug"].unique())
        self.bankroll = self.initial_bankroll
        results, bankroll_hist = self._run_slug_list(df, slugs, verbose=True)
        trades_df = self._build_trades_df(results)
        metrics = self._compute_metrics(results, bankroll_hist)
        return results, metrics, trades_df

    def run_walk_forward(
        self,
        train_frac: float = 0.7,
        verbose_test: bool = True,
    ) -> tuple[dict, dict, pd.DataFrame]:
        """Walk-forward backtest: train on first train_frac windows, test on rest.

        Calibration table is built from TRAIN windows only, then frozen.
        The DiffusionSignal's ctx resets between windows (no bleedover).
        Returns (train_metrics, test_metrics, test_trades_df).
        """
        df = self.load_data()
        if df.empty:
            print("  No data found.")
            empty = self._compute_metrics([], [self.initial_bankroll])
            return empty, empty, pd.DataFrame()

        # Order windows chronologically
        slugs_ordered = (
            df.groupby("market_slug")["ts_ms"]
            .min()
            .sort_values()
            .index.tolist()
        )

        split = max(1, int(len(slugs_ordered) * train_frac))
        train_slugs = slugs_ordered[:split]
        test_slugs = slugs_ordered[split:]

        cutoff_ts = (
            df[df["market_slug"] == test_slugs[0]]["ts_ms"].min()
            if test_slugs else None
        )

        import datetime
        if cutoff_ts:
            cutoff_dt = datetime.datetime.fromtimestamp(
                cutoff_ts / 1000
            ).strftime("%Y-%m-%d %H:%M")
        else:
            cutoff_dt = "N/A"

        print(f"  Walk-forward split: {len(train_slugs)} train / "
              f"{len(test_slugs)} test windows")
        print(f"  Test period starts: {cutoff_dt}")

        # ── TRAIN PASS: collect observations, build calibration ────────���──
        print(f"\n{'='*62}")
        print(f"  TRAIN PASS ({len(train_slugs)} windows) — building calibration")
        print(f"{'='*62}")
        self.bankroll = self.initial_bankroll
        if hasattr(self.signal, "calibration_table"):
            self.signal.calibration_table = None  # no cal during train pass
        train_results, train_bk_hist = self._run_slug_list(
            df, train_slugs, verbose=False
        )
        train_metrics = self._compute_metrics(train_results, train_bk_hist)
        print(f"  Train: {train_metrics['n_trades']} trades  "
              f"win={train_metrics['win_rate']:.1%}  "
              f"pnl=${train_metrics['total_pnl']:+,.0f}")

        # Build calibration from train outcomes
        obs: list[tuple[float, float, int]] = []
        for r in train_results:
            z_str = r.fill.decision_reason
            # extract z from "p=X sig=X z=Z tau=Ts ..."
            try:
                z_part = [p for p in z_str.split() if p.startswith("z=")][0]
                z_val = float(z_part.split("=")[1].rstrip("(cap)"))
            except (IndexError, ValueError):
                continue
            obs.append((
                max(-1.0, min(1.0, z_val)),
                r.fill.time_remaining_s,
                r.outcome_up,
            ))
        if obs and hasattr(self.signal, "calibration_table"):
            cal = _build_table_from_obs(obs)
            self.signal.calibration_table = cal
            print(f"  Calibration table built: {len(cal.table)} cells, "
                  f"{sum(cal.counts.values())} observations")

        # ── TEST PASS: evaluate on held-out windows ───────────────────────
        print(f"\n{'='*62}")
        print(f"  TEST PASS ({len(test_slugs)} windows) — out-of-sample")
        print(f"{'='*62}")
        self.bankroll = self.initial_bankroll
        if hasattr(self.signal, "bankroll"):
            self.signal.bankroll = self.initial_bankroll
        test_results, test_bk_hist = self._run_slug_list(
            df, test_slugs, verbose=verbose_test
        )
        test_metrics = self._compute_metrics(test_results, test_bk_hist)
        test_metrics["n_windows"] = len(test_slugs)
        train_metrics["n_windows"] = len(train_slugs)
        test_trades_df = self._build_trades_df(test_results)
        return train_metrics, test_metrics, test_trades_df

    def _build_trades_df(self, results: list[TradeResult]) -> pd.DataFrame:
        if not results:
            return pd.DataFrame()
        rows = []
        for r in results:
            in_range = (r.fill.expected_low <= r.final_btc <= r.fill.expected_high
                        if r.fill.expected_low > 0 else None)
            rows.append({
                "market_slug": r.fill.market_slug,
                "signal": r.fill.signal_name,
                "side": r.fill.side,
                "entry_ts_ms": r.fill.entry_ts_ms,
                "time_left_s": round(r.fill.time_remaining_s, 1),
                "entry_price": round(r.fill.entry_price, 6),
                "fee_per_share": round(r.fill.fee_per_share, 6),
                "shares": round(r.fill.shares, 2),
                "cost_usd": round(r.fill.cost_usd, 2),
                "outcome": "UP" if r.outcome_up else "DOWN",
                "payout": round(r.payout, 2),
                "pnl": round(r.pnl, 2),
                "pnl_pct": round(r.pnl_pct, 4),
                "btc_at_fill": round(r.fill.btc_at_fill, 2),
                "start_price": round(r.fill.start_price, 2),
                "expected_low": round(r.fill.expected_low, 2),
                "expected_high": round(r.fill.expected_high, 2),
                "final_btc": round(r.final_btc, 2),
                "in_range": in_range,
                "reason": r.fill.decision_reason,
            })
        return pd.DataFrame(rows)

    def _compute_metrics(
        self, results: list[TradeResult], bankroll_hist: list[float]
    ) -> dict:
        base = {
            "signal": self.signal.name,
            "latency_ms": self.latency_ms,
            "slippage": self.slippage,
            "initial_bankroll": self.initial_bankroll,
            "final_bankroll": self.bankroll,
        }
        if not results:
            base.update(n_trades=0, total_pnl=0.0, total_fees=0.0,
                        win_rate=0.0, avg_pnl=0.0, avg_win=0.0,
                        avg_loss=0.0, max_drawdown=0.0, max_dd_pct=0.0,
                        sharpe=0.0, sharpe_deflated=0.0,
                        n_trials=getattr(self, "n_trials", 1))
            return base

        pnls = [r.pnl for r in results]
        total_fees = sum(r.fill.fee_per_share * r.fill.shares for r in results)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # Max drawdown
        peak = bankroll_hist[0]
        max_dd = 0.0
        max_dd_pct = 0.0
        for b in bankroll_hist:
            if b > peak:
                peak = b
            dd = peak - b
            dd_pct = dd / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct

        # Sharpe (96 windows/day for 15-min markets)
        mean_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0
        sharpe = (mean_pnl / std_pnl) * math.sqrt(96) if std_pnl > 0 else 0.0

        # Deflated Sharpe (Bailey & Lopez de Prado, 2014)
        # Penalizes for multiple testing: SR* = SR - sqrt(2 * log(N) / T)
        # where N = number of parameter configs tried, T = number of trades.
        # N_trials is set by the caller via the engine; defaults to 1 (no penalty).
        n_trials = getattr(self, "n_trials", 1)
        if n_trials > 1 and len(pnls) > 1:
            haircut = math.sqrt(2.0 * math.log(n_trials) / len(pnls)) * math.sqrt(96)
            sharpe_deflated = sharpe - haircut
        else:
            sharpe_deflated = sharpe

        base.update(
            n_trades=len(results),
            total_pnl=round(sum(pnls), 2),
            total_fees=round(total_fees, 2),
            win_rate=round(len(wins) / len(results), 4),
            avg_pnl=round(mean_pnl, 2),
            avg_win=round(float(np.mean(wins)), 2) if wins else 0.0,
            avg_loss=round(float(np.mean(losses)), 2) if losses else 0.0,
            max_drawdown=round(max_dd, 2),
            max_dd_pct=round(max_dd_pct, 4),
            sharpe=round(sharpe, 2),
            sharpe_deflated=round(sharpe_deflated, 2),
            n_trials=n_trials,
        )
        return base


# ── Sensitivity grid ────────────────────────────────────────────────────────

def run_sensitivity(
    initial_bankroll: float = 10_000.0,
    latency_grid: list[int] | None = None,
    slippage_grid: list[float] | None = None,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    if latency_grid is None:
        latency_grid = [0, 250, 500, 1000]
    if slippage_grid is None:
        slippage_grid = [0.0, 0.001, 0.002]

    rows = []
    total = len(latency_grid) * len(slippage_grid)
    i = 0

    for lat in latency_grid:
        for slip in slippage_grid:
            i += 1
            print(f"\n--- Sensitivity {i}/{total}: latency={lat}ms slippage={slip} ---")
            signal = DiffusionSignal(bankroll=initial_bankroll, slippage=slip)
            engine = BacktestEngine(
                signal=signal,
                data_dir=data_dir,
                latency_ms=lat,
                slippage=slip,
                initial_bankroll=initial_bankroll,
            )
            engine.n_trials = total
            _, metrics, _ = engine.run()
            rows.append(metrics)

    return pd.DataFrame(rows)


# ── CLI ─────────────────────────────────────────────────────────────────────

def print_summary(metrics: dict, trades_df: pd.DataFrame):
    if not trades_df.empty:
        print("\n  Per-trade detail:")
        print("-" * 72)
        for _, t in trades_df.iterrows():
            rem = t["time_left_s"]
            rem_m, rem_s = int(rem) // 60, int(rem) % 60
            print(f"  {t['market_slug']}")
            print(f"    Side:        {t['side']}  |  Time left: {rem_m}:{rem_s:02d}")
            print(f"    Entry price: {t['entry_price']:.6f}  (fee/sh: {t['fee_per_share']:.6f})")
            print(f"    Shares:      {t['shares']:.2f}  |  Cost: ${t['cost_usd']:.2f}")
            # Expected price range
            if t.get("expected_low", 0) > 0:
                in_range = t.get("in_range")
                tag = "YES" if in_range else "NO"
                print(f"    BTC at fill:  ${t['btc_at_fill']:,.2f}  |  Start: ${t['start_price']:,.2f}")
                print(f"    Expected:     ${t['expected_low']:,.2f} - ${t['expected_high']:,.2f} (1-sigma)")
                print(f"    Final BTC:    ${t['final_btc']:,.2f}  |  In range: {tag}")
            print(f"    Outcome:     {t['outcome']}  |  Payout: ${t['payout']:.2f}")
            print(f"    PnL:         ${t['pnl']:+.2f} ({t['pnl_pct']:+.2%})")
            print(f"    Reason:      {t['reason']}")
            print("-" * 72)

    print(f"\n{'='*62}")
    print(f"  BACKTEST SUMMARY: {metrics['signal']}")
    print(f"{'='*62}")
    print(f"  Bankroll:       ${metrics['initial_bankroll']:,.0f} -> ${metrics['final_bankroll']:,.2f}")
    print(f"  Latency:        {metrics['latency_ms']}ms  |  Slippage: {metrics['slippage']}")
    print(f"  Trades:         {metrics['n_trades']}")
    print(f"  Win rate:       {metrics['win_rate']:.1%}")
    print(f"  Total PnL:      ${metrics['total_pnl']:+.2f}")
    print(f"  Total fees:     ${metrics['total_fees']:.2f}")
    print(f"  Avg PnL/trade:  ${metrics['avg_pnl']:+.2f}")
    print(f"  Avg win:        ${metrics['avg_win']:+.2f}")
    print(f"  Avg loss:       ${metrics['avg_loss']:+.2f}")
    print(f"  Max drawdown:   ${metrics['max_drawdown']:.2f} ({metrics['max_dd_pct']:.1%})")
    print(f"  Sharpe (ann.):  {metrics['sharpe']:.2f}")
    if metrics.get("n_trials", 1) > 1:
        print(f"  Sharpe (defl.): {metrics['sharpe_deflated']:.2f}  "
              f"(N={metrics['n_trials']} trials)")
    print(f"{'='*62}")


def print_walk_forward_summary(
    signal_name: str,
    train_m: dict,
    test_m: dict,
    test_trades_df: pd.DataFrame,
):
    """Print side-by-side train vs test comparison."""
    print_summary(test_m, test_trades_df)

    print(f"\n{'='*62}")
    print(f"  WALK-FORWARD COMPARISON: {signal_name}")
    print(f"{'='*62}")
    print(f"  {'Metric':<22}  {'TRAIN (in-sample)':>18}  {'TEST (out-of-sample)':>20}")
    print(f"  {'-'*62}")

    def fmt_row(label, train_val, test_val):
        print(f"  {label:<22}  {train_val:>18}  {test_val:>20}")

    fmt_row("Windows", str(train_m.get("n_windows", train_m.get("n_trades", 0))),
            str(test_m.get("n_windows", test_m.get("n_trades", 0))))
    fmt_row("Trades fired",
            f"{train_m.get('n_trades', 0)}",
            f"{test_m.get('n_trades', 0)}")
    fmt_row("Win rate",
            f"{train_m.get('win_rate', 0):.1%}",
            f"{test_m.get('win_rate', 0):.1%}")
    fmt_row("Total PnL",
            f"${train_m.get('total_pnl', 0):+,.0f}",
            f"${test_m.get('total_pnl', 0):+,.0f}")
    fmt_row("Bankroll end",
            f"${train_m.get('final_bankroll', 0):,.0f}",
            f"${test_m.get('final_bankroll', 0):,.0f}")
    fmt_row("Sharpe (ann.)",
            f"{train_m.get('sharpe', 0):.2f}",
            f"{test_m.get('sharpe', 0):.2f}")
    fmt_row("Max drawdown",
            f"{train_m.get('max_dd_pct', 0):.1%}",
            f"{test_m.get('max_dd_pct', 0):.1%}")
    print(f"{'='*62}")

    # Signal quality verdict
    test_wr = test_m.get("win_rate", 0)
    test_pnl = test_m.get("total_pnl", 0)
    test_n = test_m.get("n_trades", 0)
    if test_n < 10:
        verdict = "⚠ TOO FEW TEST TRADES — results not meaningful"
    elif test_wr >= 0.55 and test_pnl > 0:
        verdict = "✓ EDGE DETECTED — win rate and PnL positive out-of-sample"
    elif test_wr >= 0.50 and test_pnl > 0:
        verdict = "~ MARGINAL EDGE — positive but needs more data"
    elif test_wr >= 0.55:
        verdict = "~ MARGINAL EDGE — win rate positive but PnL negative (high loss size)"
    else:
        verdict = "✗ NO EDGE — win rate below 50% out-of-sample"
    print(f"\n  Verdict: {verdict}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Up/Down Backtest — walk-forward, one entry per window"
    )
    parser.add_argument("--market", default=DEFAULT_MARKET,
                        choices=list(MARKET_CONFIGS),
                        help="Market to backtest (default: btc)")
    parser.add_argument("--bankroll", type=float, default=10_000.0)
    parser.add_argument("--signal", default="diffusion",
                        choices=["diffusion", "always_up", "always_down", "random", "all"])
    parser.add_argument("--latency", type=int, default=0, help="ms")
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maker", action="store_true",
                        help="Use maker (limit order) mode: 0%% fee, dual-side evaluation")
    parser.add_argument("--train-frac", type=float, default=0.7,
                        help="Fraction of windows used for training/calibration (default 0.7). "
                             "Backtest results reflect only the held-out test period. "
                             "Use 1.0 to skip split and run all windows (in-sample).")
    parser.add_argument("--min-entry-price", type=float, default=0.10,
                        help="Minimum bid/entry price to accept (default 0.10)")
    parser.add_argument("--cal-prior-strength", type=float, default=100.0,
                        help="Bayesian prior strength n0 for GBM/calibration fusion (default 100)")
    parser.add_argument("--maker-withdraw", type=float, default=60.0,
                        help="Stop new orders when tau < N seconds (default 60)")
    parser.add_argument("--oracle-cancel-threshold", type=float, default=0.0,
                        help="Hard-cancel when Binance-Chainlink gap exceeds this fraction.")
    parser.add_argument("--cross-asset-dir", type=str, default=None,
                        help="Secondary asset data subdir for cross-asset disagreement veto "
                             "(e.g. 'eth_15m').")
    parser.add_argument("--cross-asset-min-z", type=float, default=0.3,
                        help="Minimum |z| threshold for cross-asset veto (default 0.3)")
    parser.add_argument("--min-z", type=float, default=0.0,
                        help="Minimum |z-score| to enter a trade (default 0.0 = disabled, "
                             "recommended 0.7 based on walk-forward analysis)")
    args = parser.parse_args()

    config = get_config(args.market)
    data_dir = DATA_DIR / config.data_subdir

    # Per-market signal overrides
    eth_overrides = {}
    if args.market == "eth":
        eth_overrides = dict(
            edge_threshold=0.15,
            reversion_discount=0.10,
            momentum_lookback_s=15,
            momentum_majority=0.7,
            spread_edge_penalty=0.2,
        )

    # Maker mode overrides
    maker_overrides = {}
    if args.maker:
        maker_overrides = dict(
            maker_mode=True,
            max_bet_fraction=0.02,
            edge_threshold=0.08,
            momentum_majority=0.0,
            spread_edge_penalty=0.0,
            window_duration=config.window_duration_s,
        )

    # Cross-asset disagreement lookup
    cross_asset_lookup = None
    if args.cross_asset_dir:
        secondary_dir = DATA_DIR / args.cross_asset_dir
        if secondary_dir.exists():
            print(f"  Building cross-asset z lookup from {secondary_dir} ...")
            cross_asset_lookup = build_cross_asset_lookup(secondary_dir)
            print(f"  Cross-asset lookup: {len(cross_asset_lookup)} windows indexed")
        else:
            print(f"  WARNING: cross-asset dir not found: {secondary_dir}")

    # VAMP mode
    vamp_kw = {}
    base_market = args.market.replace("_5m", "")
    if base_market == "btc":
        vamp_kw = dict(vamp_mode="cost")
    elif base_market == "eth":
        vamp_kw = dict(vamp_mode="filter", vamp_filter_threshold=0.07)

    # 5m timing overrides
    is_5m = "_5m" in args.market
    maker_warmup = 30.0 if is_5m else 100.0
    maker_withdraw = 30.0 if is_5m else args.maker_withdraw
    if is_5m:
        print(f"  5m overrides: warmup={maker_warmup:.0f}s, withdraw={maker_withdraw:.0f}s")

    signal_map = {
        "diffusion": lambda: DiffusionSignal(
            bankroll=args.bankroll, slippage=args.slippage,
            calibration_table=None,  # walk-forward will inject after train pass
            min_entry_price=args.min_entry_price,
            cal_prior_strength=args.cal_prior_strength,
            maker_warmup_s=maker_warmup,
            maker_withdraw_s=maker_withdraw,
            max_sigma=config.max_sigma,
            min_sigma=config.min_sigma,
            oracle_cancel_threshold=args.oracle_cancel_threshold,
            cross_asset_z_lookup=cross_asset_lookup,
            cross_asset_min_z=args.cross_asset_min_z,
            min_entry_z=args.min_z,
            tail_mode=config.tail_mode,
            tail_nu_default=config.tail_nu_default,
            kou_lambda=config.kou_lambda,
            kou_p_up=config.kou_p_up,
            kou_eta1=config.kou_eta1,
            kou_eta2=config.kou_eta2,
            **{**eth_overrides, **maker_overrides, **vamp_kw}),
        "always_up": lambda: AlwaysUp(bankroll=args.bankroll),
        "always_down": lambda: AlwaysDown(bankroll=args.bankroll),
        "random": lambda: RandomCoinFlip(bankroll=args.bankroll, seed=args.seed),
    }

    names = ["always_up", "always_down", "random", "diffusion"] \
        if args.signal == "all" else [args.signal]

    mode_str = "MAKER" if args.maker else "FOK"
    train_frac = args.train_frac
    use_split = train_frac < 1.0

    for name in names:
        signal = signal_map[name]()
        engine = BacktestEngine(
            signal=signal,
            data_dir=data_dir,
            latency_ms=args.latency,
            slippage=args.slippage,
            initial_bankroll=args.bankroll,
        )
        print(f"\n{'='*62}")
        print(f"  Running: {signal.name} ({config.display_name}) [{mode_str}]")
        if use_split:
            print(f"  Mode: walk-forward (train={train_frac:.0%} / test={1-train_frac:.0%}), "
                  f"1 trade per window")
        else:
            print(f"  Mode: full in-sample, 1 trade per window")
        print(f"{'='*62}")

        if use_split:
            train_m, test_m, test_trades_df = engine.run_walk_forward(
                train_frac=train_frac,
                verbose_test=True,
            )
            print_walk_forward_summary(signal.name, train_m, test_m, test_trades_df)
        else:
            _, metrics, trades_df = engine.run()
            print_summary(metrics, trades_df)


if __name__ == "__main__":
    main()
