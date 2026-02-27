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

def poly_fee(p: float) -> float:
    """Polymarket taker fee for 15-min crypto markets."""
    return 0.25 * (p * (1.0 - p)) ** 2


def norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erfc (no scipy needed)."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


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


def build_calibration_table(
    data_dir: Path,
    max_z: float = 1.0,
    vol_lookback_s: int = 90,
) -> CalibrationTable:
    """Build a walk-forward calibration table from all complete parquet windows.

    For each window, extracts signals every 30 rows (matching
    analyze_calibration.py), records (z_capped, tau, outcome_up) tuples,
    and builds the lookup from ALL past observations.
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

    # Walk-forward: accumulate observations, build table from all past data
    all_obs: list[tuple[float, float, int]] = []  # (z_capped, tau, outcome_up)

    for df in windows:
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

    # Build final table from all observations
    return _build_table_from_obs(all_obs)


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
        max_bet_fraction: float = 0.0125,
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
        reversion_discount: float = 0.07,
        momentum_majority: float = 1.0,
        maker_mode: bool = False,
        edge_threshold_step: float = 0.0,
        calibration_table: CalibrationTable | None = None,
        maker_warmup_s: float = 100.0,
        min_entry_price: float = 0.10,
        cal_prior_strength: float = 500.0,
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
        # Regime-scaled z: scale z by sigma_ema / sigma_calibration
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

    def _smoothed_sigma(self, raw_sigma: float, ctx: dict) -> float:
        """Apply EMA smoothing and asset-specific cap to raw sigma."""
        if raw_sigma == 0.0:
            return 0.0
        ema = ctx.get("_sigma_ema")
        if ema is None:
            ema = raw_sigma
        else:
            ema = self.sigma_ema_alpha * raw_sigma + (1 - self.sigma_ema_alpha) * ema
        ctx["_sigma_ema"] = ema
        if self.max_sigma is not None:
            ema = min(ema, self.max_sigma)
        ema = max(ema, self.min_sigma)
        return ema

    @staticmethod
    def _compute_toxicity(snapshot: "Snapshot", max_spread: float) -> float:
        """Composite toxicity score in [0, 1] from microstructure signals.

        Components (each normalized to [0, 1]):
          1. Spread width:   avg(spread_up, spread_down) / max_spread
          2. Book imbalance: abs(total_bid_depth - total_ask_depth) / total_depth
          3. Mid-oracle gap: abs(book_mid - chainlink) / chainlink

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

        # 3. Mid–oracle deviation
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
        """VPIN from trade bars: mean |sell - buy| / total over last W bars.

        Returns 0.0 when insufficient bars (graceful warmup).
        """
        if len(bars) < window:
            return 0.0
        recent = list(bars)[-window:]
        total_sum = 0.0
        imbalance_sum = 0.0
        for buy_vol, sell_vol in recent:
            total = buy_vol + sell_vol
            if total > 0:
                imbalance_sum += abs(sell_vol - buy_vol) / total
            # Empty bars (no trades) contribute 0
        return imbalance_sum / window

    @staticmethod
    def _compute_oracle_lag(chainlink_price: float, binance_mid) -> float:
        """Price discrepancy between Binance mid and Chainlink oracle.

        Returns abs(binance_mid - chainlink) / chainlink.
        Returns 0.0 if binance_mid is None (graceful degradation).
        """
        if binance_mid is None or chainlink_price <= 0:
            return 0.0
        return abs(binance_mid - chainlink_price) / chainlink_price

    def _p_model(self, z_capped: float, tau: float) -> float:
        """Model probability of UP via Bayesian fusion of GBM + calibration.

        p = w * p_calibrated + (1 - w) * p_gbm
        w = n / (n + n0)

        With few observations, leans on GBM prior.
        With many observations, leans on calibration.
        """
        p_gbm = norm_cdf(z_capped)
        if self.calibration_table is not None:
            p_cal, n = self.calibration_table.lookup_with_count(z_capped, tau)
            if n > 0:
                w = n / (n + self.cal_prior_strength)
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
        if oracle_lag > self.oracle_lag_threshold:
            lag_excess = min((oracle_lag - self.oracle_lag_threshold) / self.oracle_lag_threshold, 1.0)
            dyn_threshold *= 1.0 + self.oracle_lag_mult * lag_excess

        # Realized vol (short window for model)
        raw_sigma = self._compute_vol(hist[-self.vol_lookback_s:], ts_hist[-self.vol_lookback_s:])
        if raw_sigma == 0.0:
            return Decision("FLAT", 0.0, 0.0, "zero vol")

        # Vol regime filter: compare recent vol to longer baseline
        if len(hist) >= self.vol_regime_lookback_s:
            sigma_baseline = self._compute_vol(hist[-self.vol_regime_lookback_s:], ts_hist[-self.vol_regime_lookback_s:])
            if sigma_baseline > 0 and raw_sigma > self.vol_regime_mult * sigma_baseline:
                return Decision("FLAT", 0.0, 0.0,
                    f"vol spike ({raw_sigma:.2e} > "
                    f"{self.vol_regime_mult}x baseline {sigma_baseline:.2e})")

        # EMA smoothing + asset cap
        sigma_per_s = self._smoothed_sigma(raw_sigma, ctx)

        # Vol kill switch
        if self.vol_kill_sigma is not None and sigma_per_s > self.vol_kill_sigma:
            return Decision("FLAT", 0.0, 0.0,
                f"vol kill switch (sigma={sigma_per_s:.2e} "
                f"> {self.vol_kill_sigma:.2e})")

        # Model probability (z capped to prevent overconfidence)
        delta = (effective_price - snapshot.window_start_price) / snapshot.window_start_price
        z_raw = delta / (sigma_per_s * math.sqrt(tau))

        # Regime-scaled z (same as decide_both_sides)
        regime_z_factor = 1.0
        if self.regime_z_scale and self.sigma_calibration and self.sigma_calibration > 0:
            scale = sigma_per_s / self.sigma_calibration
            scale = max(0.5, min(2.0, scale))
            z_raw *= scale
            regime_z_factor = scale
        ctx["_regime_z_factor"] = regime_z_factor

        z = max(-self.max_z, min(self.max_z, z_raw))
        p_model = self._p_model(z, tau)

        # Mean-reversion discount: pull p_model toward 0.5
        if self.reversion_discount > 0:
            p_model = p_model * (1 - self.reversion_discount) + 0.5 * self.reversion_discount

        # Effective costs
        p_up_cost = ask_up + poly_fee(ask_up) + self.slippage
        p_down_cost = ask_down + poly_fee(ask_down) + self.slippage

        # Edges (penalized by spread — wider spread = less trustworthy pricing)
        edge_up = p_model - p_up_cost - self.spread_edge_penalty * spread_up
        edge_down = (1.0 - p_model) - p_down_cost - self.spread_edge_penalty * spread_down

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
                ctx["_p_model_raw"] = self._p_model(_z, _tau)
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

        # Spread gate
        spread_up = ask_up - bid_up
        spread_down = ask_down - bid_down
        if spread_up > self.max_spread or spread_down > self.max_spread:
            reason = (f"spread too wide (up={spread_up:.3f} down={spread_down:.3f} "
                      f"max={self.max_spread})")
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))

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
        if raw_sigma == 0.0:
            reason = "zero vol"
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))

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

        # Volatility kill switch: hard pause when EMA sigma exceeds
        # an absolute ceiling.  Distinct from the relative regime filter
        # above — this catches sustained high-vol episodes where the
        # baseline itself has drifted up.
        if self.vol_kill_sigma is not None and sigma_per_s > self.vol_kill_sigma:
            reason = (f"vol kill switch (sigma={sigma_per_s:.2e} "
                      f"> {self.vol_kill_sigma:.2e})")
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))

        # Dynamic threshold with optional step for window_trade_count
        window_trades = ctx.get("window_trade_count", 0)
        base_threshold = self.edge_threshold + self.edge_threshold_step * window_trades
        dyn_threshold = base_threshold * (
            1.0 + self.early_edge_mult * math.sqrt(tau / self.window_duration)
        )

        # Toxicity penalty: widen edge threshold proportionally when
        # microstructure is adverse.  The multiplier ramps linearly from
        # 1.0 at toxicity <= threshold to (1 + toxicity_edge_mult) at
        # toxicity = 1.0, so we still trade in mild conditions but
        # demand much more edge in toxic ones.
        if toxicity > self.toxicity_threshold:
            excess = (toxicity - self.toxicity_threshold) / (1.0 - self.toxicity_threshold)
            dyn_threshold *= 1.0 + self.toxicity_edge_mult * excess

        # VPIN flow toxicity penalty (before DOWN bonus derivation
        # so DOWN bonus sees the VPIN-widened threshold)
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
        if oracle_lag > self.oracle_lag_threshold:
            lag_excess = min((oracle_lag - self.oracle_lag_threshold) / self.oracle_lag_threshold, 1.0)
            dyn_threshold *= 1.0 + self.oracle_lag_mult * lag_excess

        # z normalization: fractional delta / (sigma * sqrt(tau))
        price = effective_price
        delta = (price - snapshot.window_start_price) / snapshot.window_start_price
        z_raw = delta / (sigma_per_s * math.sqrt(tau))

        # Regime-scaled z: adjust z for current vol regime relative to
        # the calibration-period vol.  High-vol regimes shrink z (wider
        # distribution → less confident directional signal), low-vol
        # regimes amplify it.  Clamped to [0.5, 2.0] to avoid extreme
        # suppression or overconfidence from a single bar.
        regime_z_factor = 1.0
        if self.regime_z_scale and self.sigma_calibration and self.sigma_calibration > 0:
            scale = sigma_per_s / self.sigma_calibration
            scale = max(0.5, min(2.0, scale))
            z_raw *= scale
            regime_z_factor = scale
        ctx["_regime_z_factor"] = regime_z_factor

        z = max(-self.max_z, min(self.max_z, z_raw))
        p_model = self._p_model(z, tau)

        # Expose model state so OrderMixin can snapshot it at fill time
        ctx["_p_model_raw"] = p_model
        ctx["_p_display"] = norm_cdf(z)  # smooth GBM for dashboard (no binning)
        ctx["_sigma_per_s"] = sigma_per_s
        ctx["_dyn_threshold_up"] = dyn_threshold

        if self.reversion_discount > 0:
            p_model = p_model * (1 - self.reversion_discount) + 0.5 * self.reversion_discount

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

        # Edges — bid pricing already accounts for spread naturally
        edge_up = p_model - cost_up - self.spread_edge_penalty * spread_up
        edge_down = (1.0 - p_model) - cost_down - self.spread_edge_penalty * spread_down
        ctx["_edge_up"] = edge_up
        ctx["_edge_down"] = edge_down

        # Avellaneda-Stoikov dynamic inventory skew: penalize adding to an
        # existing side, scaled by tau/window_duration.  Full penalty at window
        # start, near-zero at expiry (safe to load up when outcome is clearer).
        if self.inventory_skew > 0:
            n_up = ctx.get("inventory_up", 0)
            n_down = ctx.get("inventory_down", 0)
            skew = self.inventory_skew * (tau / self.window_duration)
            ctx["_inventory_skew"] = skew
            edge_up -= skew * n_up
            edge_up += skew * n_down
            edge_down -= skew * n_down
            edge_down += skew * n_up

        # Store expected range in ctx
        move_1sig = sigma_per_s * math.sqrt(tau) * price
        ctx["_expected_range"] = {
            "btc_at_fill": price,
            "start_price": snapshot.window_start_price,
            "expected_low": snapshot.window_start_price - move_1sig,
            "expected_high": snapshot.window_start_price + move_1sig,
        }

        # NO/DOWN bias (optimism tax): YES/UP tends to be overpriced
        # on prediction markets, so we lower the threshold for DOWN.
        # Only applies when the book isn't already heavily one-sided
        # (imbalance < 0.5) to avoid over-concentrating risk when
        # the market clearly disagrees with our model.
        dyn_threshold_down = dyn_threshold
        down_bonus_active = False
        down_share = 0.5
        if self.down_edge_bonus > 0:
            # Reuse depths computed for OBI above
            total_d = total_depth_up + total_depth_down
            down_d = total_depth_down
            down_share = down_d / total_d if total_d > 0 else 0.5
            # Only apply bonus when book is reasonably balanced (30-70% range)
            if 0.3 <= down_share <= 0.7:
                dyn_threshold_down = dyn_threshold * (1.0 - self.down_edge_bonus)
                down_bonus_active = True
        ctx["_down_bonus_active"] = down_bonus_active
        ctx["_down_share"] = down_share
        ctx["_dyn_threshold_down"] = dyn_threshold_down

        # Evaluate each side independently
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
        max_trades_per_window: int | None = None,
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
        pending: Optional[tuple[int, Decision]] = None
        results: list[TradeResult] = []
        last_fill_ts: int = 0
        filled_sides: set = set()  # anti-hedge: track filled sides per window
        cooldown_ms = 30_000  # minimum 30s between bets
        maker_mode = getattr(self.signal, "maker_mode", False)
        if maker_mode:
            cooldown_ms = 5_000

        for _, row in window_df.iterrows():
            snap = Snapshot.from_row(row)
            if snap is None:
                continue

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
                            ctx["inventory_up"] = ctx.get("inventory_up", 0) + 1
                        elif "DOWN" in decision.action:
                            ctx["inventory_down"] = ctx.get("inventory_down", 0) + 1
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
                                ctx["inventory_up"] = ctx.get("inventory_up", 0) + 1
                            elif "DOWN" in decision.action:
                                ctx["inventory_down"] = ctx.get("inventory_down", 0) + 1
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
                            ctx["inventory_up"] = ctx.get("inventory_up", 0) + 1
                        elif "DOWN" in decision.action:
                            ctx["inventory_down"] = ctx.get("inventory_down", 0) + 1
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

    def run(self) -> tuple[list[TradeResult], dict, pd.DataFrame]:
        df = self.load_data()
        if df.empty:
            print("  No data found.")
            metrics = self._compute_metrics([], [self.bankroll])
            return [], metrics, pd.DataFrame()
        slugs = df["market_slug"].unique()
        results: list[TradeResult] = []
        self.bankroll = self.initial_bankroll
        bankroll_hist = [self.bankroll]

        for slug in slugs:
            window_df = df[df["market_slug"] == slug]
            resolved = self._resolve_window(window_df)

            if resolved is None:
                print(f"  SKIP {slug} (incomplete)")
                continue

            outcome, final_btc = resolved

            if hasattr(self.signal, "bankroll"):
                self.signal.bankroll = self.bankroll

            # bankroll is updated inside _run_window per fill
            pre_bankroll = self.bankroll
            window_results = self._run_window(window_df, outcome, final_btc)

            if window_results:
                results.extend(window_results)
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
                print(f"    window net: ${window_pnl:+.2f} "
                      f"({len(window_results)} trade{'s' if len(window_results) != 1 else ''}) "
                      f"bank=${self.bankroll:.2f}")
            else:
                print(f"  {slug}: FLAT (no trades)")

            bankroll_hist.append(self.bankroll)

        trades_df = self._build_trades_df(results)
        metrics = self._compute_metrics(results, bankroll_hist)
        return results, metrics, trades_df

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


def main():
    parser = argparse.ArgumentParser(description="Polymarket Up/Down Backtest")
    parser.add_argument("--market", default=DEFAULT_MARKET,
                        choices=list(MARKET_CONFIGS),
                        help="Market to backtest (default: btc)")
    parser.add_argument("--bankroll", type=float, default=10_000.0)
    parser.add_argument("--signal", default="diffusion",
                        choices=["diffusion", "always_up", "always_down", "random", "all"])
    parser.add_argument("--latency", type=int, default=0, help="ms")
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maker", action="store_true",
                        help="Use maker (limit order) mode: 0%% fee, dual-side evaluation")
    parser.add_argument("--max-trades-per-window", type=int, default=None,
                        help="Override max trades per window")
    parser.add_argument("--calibrated", action="store_true",
                        help="Use empirically calibrated probabilities instead of Phi(z)")
    parser.add_argument("--min-entry-price", type=float, default=0.10,
                        help="Minimum bid/entry price to accept (default 0.10)")
    parser.add_argument("--cal-prior-strength", type=float, default=100.0,
                        help="Bayesian prior strength n0 for GBM/calibration fusion (default 100)")
    parser.add_argument("--inventory-skew", type=float, default=0.02,
                        help="Edge penalty per same-side position (default 0.02)")
    parser.add_argument("--maker-withdraw", type=float, default=60.0,
                        help="Stop new orders when tau < N seconds (default 60)")
    args = parser.parse_args()

    config = get_config(args.market)
    data_dir = DATA_DIR / config.data_subdir

    if args.sensitivity:
        print(f"Running sensitivity analysis ({config.display_name})...\n")
        sens_df = run_sensitivity(initial_bankroll=args.bankroll, data_dir=data_dir)
        print(f"\n{'='*62}")
        print("  SENSITIVITY GRID (DiffusionSignal)")
        print(f"{'='*62}")
        cols = ["latency_ms", "slippage", "n_trades", "total_pnl",
                "win_rate", "sharpe", "sharpe_deflated", "final_bankroll"]
        print(sens_df[cols].to_string(index=False))
        return

    # Per-market signal overrides (ETH needs tighter filters due to mean reversion)
    eth_overrides = {}
    eth_engine_kw = {}
    if args.market == "eth":
        eth_overrides = dict(
            edge_threshold=0.15,        # higher bar (BTC default 0.10)
            reversion_discount=0.10,    # ETH mean-reverts, discount p toward 0.5
            momentum_lookback_s=15,     # shorter lookback (ETH oscillates more)
            momentum_majority=0.7,      # 70% majority instead of 100% (BTC default)
            spread_edge_penalty=0.2,    # reduced from 1.0 (avoids double-counting)
        )
        eth_engine_kw = dict(max_trades_per_window=1)

    # Maker mode overrides
    maker_overrides = {}
    if args.maker:
        maker_overrides = dict(
            maker_mode=True,
            max_bet_fraction=0.02,
            edge_threshold=0.08,
            momentum_majority=0.0,
            spread_edge_penalty=0.0,  # bid pricing handles spread naturally
            window_duration=config.window_duration_s,
        )
        if "max_trades_per_window" not in eth_engine_kw:
            eth_engine_kw["max_trades_per_window"] = 1

    if args.max_trades_per_window is not None:
        eth_engine_kw["max_trades_per_window"] = args.max_trades_per_window

    # Calibration: build empirical lookup table and adjust thresholds
    cal_table = None
    calibrated_overrides = {}
    if args.calibrated:
        print(f"  Building calibration table from {data_dir} ...")
        cal_table = build_calibration_table(data_dir)
        n_cells = len(cal_table.table)
        n_obs = sum(cal_table.counts.values())
        print(f"  Calibration table: {n_cells} cells, {n_obs} observations")
        # Calibrated edges are smaller/honest — still require meaningful edge
        if args.maker:
            calibrated_overrides = dict(edge_threshold=0.04, early_edge_mult=0.4)
        else:
            calibrated_overrides = dict(edge_threshold=0.04, early_edge_mult=0.4)

    # VAMP mode: BTC uses cost-based, ETH uses filter-based
    vamp_kw = {}
    base_market = args.market.replace("_5m", "")
    if base_market == "btc":
        vamp_kw = dict(vamp_mode="cost")
    elif base_market == "eth":
        vamp_kw = dict(vamp_mode="filter", vamp_filter_threshold=0.07)

    # 5m market overrides: scale timing for 300s windows
    is_5m = "_5m" in args.market
    maker_warmup = 100.0
    maker_withdraw = args.maker_withdraw
    five_m_kw = {}
    if is_5m:
        if base_market == "btc":
            maker_warmup = 30.0
            maker_withdraw = 30.0
        elif base_market == "eth":
            maker_warmup = 30.0
            maker_withdraw = 20.0
            five_m_kw["edge_threshold"] = 0.04
            five_m_kw["early_edge_mult"] = 0.4
            five_m_kw["reversion_discount"] = 0.10
            eth_engine_kw["max_trades_per_window"] = 2  # fewer trades = higher Sharpe
        print(f"  5m overrides: warmup={maker_warmup:.0f}s, withdraw={maker_withdraw:.0f}s")

    signal_map = {
        "diffusion": lambda: DiffusionSignal(
            bankroll=args.bankroll, slippage=args.slippage,
            calibration_table=cal_table,
            min_entry_price=args.min_entry_price,
            cal_prior_strength=args.cal_prior_strength,
            inventory_skew=args.inventory_skew,
            maker_warmup_s=maker_warmup,
            maker_withdraw_s=maker_withdraw,
            max_sigma=config.max_sigma,
            min_sigma=config.min_sigma,
            **{**eth_overrides, **maker_overrides, **calibrated_overrides, **vamp_kw, **five_m_kw}),
        "always_up": lambda: AlwaysUp(bankroll=args.bankroll),
        "always_down": lambda: AlwaysDown(bankroll=args.bankroll),
        "random": lambda: RandomCoinFlip(bankroll=args.bankroll, seed=args.seed),
    }

    names = ["always_up", "always_down", "random", "diffusion"] \
        if args.signal == "all" else [args.signal]

    mode_str = "MAKER" if args.maker else "FOK"
    if args.calibrated:
        mode_str += "+CAL"
    for name in names:
        signal = signal_map[name]()
        engine = BacktestEngine(
            signal=signal,
            data_dir=data_dir,
            latency_ms=args.latency,
            slippage=args.slippage,
            initial_bankroll=args.bankroll,
            **eth_engine_kw,
        )
        print(f"\n{'='*62}")
        print(f"  Running: {signal.name} ({config.display_name}) [{mode_str}]")
        print(f"{'='*62}")
        _, metrics, trades_df = engine.run()
        print_summary(metrics, trades_df)


if __name__ == "__main__":
    main()
