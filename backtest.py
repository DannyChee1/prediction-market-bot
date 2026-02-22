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


# ── Fee & math helpers ──────────────────────────────────────────────────────

def poly_fee(p: float) -> float:
    """Polymarket taker fee for 15-min crypto markets."""
    return 0.25 * (p * (1.0 - p)) ** 2


def norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erfc (no scipy needed)."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


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


class DiffusionSignal(Signal):
    """
    Models BTC as GBM. p_model = Phi(delta / (sigma * sqrt(tau))).
    Buys whichever side has edge > threshold, sized by half-Kelly.
    """
    name = "diffusion"

    def __init__(
        self,
        bankroll: float,
        vol_lookback_s: int = 20,
        min_sigma: float = 1e-6,
        edge_threshold: float = 0.10,
        early_edge_mult: float = 4.0,
        window_duration: float = 900.0,
        max_bet_fraction: float = 0.0125,
        min_order_shares: float = 5.0,
        kelly_fraction: float = 0.25,
        slippage: float = 0.0,
        max_z: float = 2.0,
        momentum_lookback_s: int = 30,
        max_spread: float = 0.05,
        spread_edge_penalty: float = 1.0,
        vol_regime_lookback_s: int = 120,
        vol_regime_mult: float = 3.0,
        max_entry_time_s: float | None = None,
        reversion_discount: float = 0.0,
        momentum_majority: float = 1.0,
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

    def _compute_vol(self, prices: list[float]) -> float:
        """Realized vol from price series, ignoring stale (duplicate) ticks.

        Chainlink oracle updates irregularly. Between updates the same
        price is recorded every tick, producing zero-returns that
        artificially deflate sigma.  We skip consecutive duplicates and
        normalize each return by sqrt(dt) to get per-second volatility.
        """
        # Collect (index, price) for each actual price change
        changes: list[tuple[int, float]] = []
        for i, p in enumerate(prices):
            if p > 0 and (not changes or p != changes[-1][1]):
                changes.append((i, p))

        if len(changes) < 3:
            return 0.0

        # Time-normalized log returns
        log_rets = []
        for j in range(1, len(changes)):
            dt = changes[j][0] - changes[j-1][0]   # ticks ≈ seconds
            if dt > 0:
                lr = math.log(changes[j][1] / changes[j-1][1])
                log_rets.append(lr / math.sqrt(dt))

        if len(log_rets) < 2:
            return 0.0
        return max(float(np.std(log_rets, ddof=1)), self.min_sigma)

    def decide(self, snapshot: Snapshot, ctx: dict) -> Decision:
        # Build price history
        hist = ctx.setdefault("price_history", [])
        hist.append(snapshot.chainlink_price)

        if len(hist) < max(2, self.vol_lookback_s):
            return Decision("FLAT", 0.0, 0.0,
                            f"need {self.vol_lookback_s}s history ({len(hist)})")

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

        tau = snapshot.time_remaining_s
        if tau <= 0:
            return Decision("FLAT", 0.0, 0.0, "window expired")

        # Late-entry gate: only trade when time remaining <= max_entry_time_s
        if self.max_entry_time_s is not None and tau > self.max_entry_time_s:
            return Decision("FLAT", 0.0, 0.0,
                f"too early ({tau:.0f}s left > {self.max_entry_time_s:.0f}s gate)")

        # Dynamic edge threshold: higher early, decays with sqrt(tau)
        # At tau=900s: base * (1 + mult), at tau=0: base
        dyn_threshold = self.edge_threshold * (
            1.0 + self.early_edge_mult * math.sqrt(tau / self.window_duration)
        )

        # Realized vol (short window for model)
        sigma_per_s = self._compute_vol(hist[-self.vol_lookback_s:])
        if sigma_per_s == 0.0:
            return Decision("FLAT", 0.0, 0.0, "zero vol")

        # Vol regime filter: compare recent vol to longer baseline
        # If short-term vol > mult * long-term vol, market is stressed
        if len(hist) >= self.vol_regime_lookback_s:
            sigma_baseline = self._compute_vol(hist[-self.vol_regime_lookback_s:])
            if sigma_baseline > 0 and sigma_per_s > self.vol_regime_mult * sigma_baseline:
                return Decision("FLAT", 0.0, 0.0,
                    f"vol spike ({sigma_per_s:.2e} > "
                    f"{self.vol_regime_mult}x baseline {sigma_baseline:.2e})")

        # Model probability (z capped to prevent overconfidence)
        delta = snapshot.chainlink_price - snapshot.window_start_price
        z_raw = delta / (sigma_per_s * math.sqrt(tau))
        z = max(-self.max_z, min(self.max_z, z_raw))
        p_model = norm_cdf(z)

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
        elif decision.action == "BUY_DOWN":
            side, ask_levels, best_ask = "DOWN", snap.ask_levels_down, snap.best_ask_down
        else:
            return None

        if not ask_levels or best_ask is None or best_ask <= 0:
            return None

        eff_est = best_ask + poly_fee(best_ask) + self.slippage
        if eff_est <= 0 or eff_est >= 1.0:
            return None

        desired_shares = decision.size_usd / eff_est
        filled, total_cost, avg_price = walk_book(ask_levels, desired_shares, self.slippage)

        if filled <= 0 or total_cost <= 0:
            return None

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
        fee_avg = avg_price - raw_avg - self.slippage

        # Expected price range from signal (if available)
        rng = ctx.get("_expected_range", {})

        return Fill(
            market_slug=snap.market_slug,
            side=side,
            entry_ts_ms=snap.ts_ms,
            time_remaining_s=snap.time_remaining_s,
            entry_price=avg_price,
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
        ctx: dict = {}
        pending: Optional[tuple[int, Decision]] = None
        results: list[TradeResult] = []
        last_fill_ts: int = 0
        cooldown_ms = 30_000  # minimum 30s between bets

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
                    pending = None

            # Cooldown between bets
            if snap.ts_ms - last_fill_ts < cooldown_ms and last_fill_ts > 0:
                continue

            # Max trades per window
            if self.max_trades_per_window is not None and len(results) >= self.max_trades_per_window:
                break

            # Run signal
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
                        sharpe=0.0)
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
                "win_rate", "sharpe", "final_bankroll"]
        print(sens_df[cols].to_string(index=False))
        return

    # Per-market signal overrides (ETH needs tighter filters due to mean reversion)
    eth_overrides = {}
    eth_engine_kw = {}
    if args.market == "eth":
        eth_overrides = dict(
            edge_threshold=0.15,        # higher bar (BTC default 0.10)
            reversion_discount=0.15,    # ETH mean-reverts ~33%, discount p toward 0.5
            momentum_lookback_s=15,     # shorter lookback (ETH oscillates more)
            momentum_majority=0.7,      # 70% majority instead of 100% (BTC default)
            spread_edge_penalty=0.2,    # reduced from 1.0 (avoids double-counting)
        )
        eth_engine_kw = dict(max_trades_per_window=1)

    signal_map = {
        "diffusion": lambda: DiffusionSignal(
            bankroll=args.bankroll, slippage=args.slippage, **eth_overrides),
        "always_up": lambda: AlwaysUp(bankroll=args.bankroll),
        "always_down": lambda: AlwaysDown(bankroll=args.bankroll),
        "random": lambda: RandomCoinFlip(bankroll=args.bankroll, seed=args.seed),
    }

    names = ["always_up", "always_down", "random", "diffusion"] \
        if args.signal == "all" else [args.signal]

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
        print(f"  Running: {signal.name} ({config.display_name})")
        print(f"{'='*62}")
        _, metrics, trades_df = engine.run()
        print_summary(metrics, trades_df)


if __name__ == "__main__":
    main()
