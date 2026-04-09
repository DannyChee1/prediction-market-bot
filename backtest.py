#!/usr/bin/env python3
"""
Polymarket BTC Up/Down Backtest Engine

Replays recorded 1-second snapshots from parquet files and evaluates
trading signals on binary BTC 15-minute Up/Down markets.

This module is the entry point and harness. The model itself
(DiffusionSignal) lives in signal_diffusion.py and the math/vol
helpers + dataclasses live in backtest_core.py — both are re-exported
below so existing `from backtest import X` imports keep working.

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

# ── Re-exports from backtest_core (backward compat) ─────────────────
# Existing callers do `from backtest import Snapshot, poly_fee, ...`.
# They keep working because we re-export every public symbol here.
from backtest_core import (  # noqa: F401
    DATA_DIR,
    MIN_FINAL_REMAINING_S,
    MAX_START_GAP_S,
    MIN_SIGMA_RATIO,
    poly_fee,
    norm_cdf,
    fast_t_cdf,
    kou_cdf,
    _betacf,
    _betainc,
    _poisson_pmf,
    _build_ohlc_bars,
    _yang_zhang_vol,
    _compute_vol_deduped,
    _time_prior_sigma,
    _load_priors_for_subdir,
    _HOURLY_VOL_MULT,
    _DOW_VOL_MULT,
    _GLOBAL_MEAN_SIGMA,
    _PRIORS_CACHE,
    _cross_asset_compute_sigma,
    build_cross_asset_lookup,
    _lookup_cross_asset_z,
    _CROSS_ASSET_TAU_CHECKPOINTS,
    CalibrationTable,
    build_calibration_table,
    _build_table_from_obs,
    Snapshot,
    Decision,
    Fill,
    TradeResult,
    Signal,
    AlwaysUp,
    AlwaysDown,
    RandomCoinFlip,
    compute_vamp,
)

# DiffusionSignal lives in signal_diffusion (extracted to keep this file
# under ~2k lines). Re-exported for backward compat.
from signal_diffusion import DiffusionSignal  # noqa: F401


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
        same_direction_stacking_only: bool = True,
        window_duration_s: float | None = None,
    ):
        self.signal = signal
        self.data_dir = data_dir
        self.latency_ms = latency_ms
        self.slippage = slippage
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.max_trades_per_window = max_trades_per_window
        # When max_trades_per_window > 1, restrict subsequent in-window
        # trades to the same direction as the first (averaging in, not
        # hedging). Mirrors the live tracker.py guard so backtest A/B
        # results actually reflect what live will do.
        self.same_direction_stacking_only = same_direction_stacking_only
        # Annualization factor for Sharpe — derived from window length so
        # 5m markets get sqrt(288) and 15m markets get sqrt(96).
        # Note: this is windows-per-DAY (matches the legacy convention,
        # which was sqrt(96) for 15m). Real per-year annualization would
        # multiply by 365 — but doing that here would silently inflate
        # every reported Sharpe ~19x and break comparison with all
        # historical reports / memory notes. Stay on per-day for parity.
        if window_duration_s is not None and window_duration_s > 0:
            self.windows_per_day = 86400.0 / window_duration_s
        else:
            self.windows_per_day = 96.0  # 15m legacy default

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
                    pending_side = "UP" if "UP" in decision.action else "DOWN"
                    pending_opp = "DOWN" if pending_side == "UP" else "UP"
                    # Same-direction guard for pending fills (parity with live)
                    if (self.same_direction_stacking_only
                            and pending_opp in filled_sides):
                        pending = None
                    else:
                        fill = self._execute_fill(snap, decision, ctx)
                        if fill is not None:
                            filled_sides.add(pending_side)
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
                # Same-direction stacking guard (parity with live tracker.py).
                # When max_trades_per_window > 1 and a prior trade in this
                # window already filled on one side, subsequent trades MUST
                # match that side. Without this guard the FOK backtest would
                # accept opposite-direction hedges that live blocks.
                fok_side = "UP" if "UP" in decision.action else "DOWN"
                fok_opp = "DOWN" if fok_side == "UP" else "UP"
                if (self.same_direction_stacking_only
                        and fok_opp in filled_sides):
                    continue
                if self.latency_ms <= 0:
                    fill = self._execute_fill(snap, decision, ctx)
                    if fill is not None:
                        filled_sides.add(fok_side)
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
            # Note: `str.rstrip("(cap)")` strips any trailing character
            # in "()cap", which silently corrupts `"1.0a"` → `"1."`.
            # Use removesuffix for a literal-suffix strip.
            try:
                z_part = [p for p in z_str.split() if p.startswith("z=")][0]
                z_raw = z_part.split("=", 1)[1].removesuffix("(cap)")
                z_val = float(z_raw)
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

        # Sharpe — derive sqrt(windows/day) from the engine's window length
        # so 5m markets get sqrt(288) and 15m markets get sqrt(96). The
        # legacy hardcoded sqrt(96) inflated all 5m Sharpe numbers by
        # ~1.73× and made cross-market comparisons meaningless.
        ann_factor = math.sqrt(self.windows_per_day)
        mean_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0
        sharpe = (mean_pnl / std_pnl) * ann_factor if std_pnl > 0 else 0.0

        # Deflated Sharpe (Bailey & Lopez de Prado, 2014)
        # Penalizes for multiple testing: SR* = SR - sqrt(2 * log(N) / T)
        # where N = number of parameter configs tried, T = number of trades.
        # N_trials is set by the caller via the engine; defaults to 1 (no penalty).
        n_trials = getattr(self, "n_trials", 1)
        if n_trials > 1 and len(pnls) > 1:
            haircut = math.sqrt(2.0 * math.log(n_trials) / len(pnls)) * ann_factor
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
                # Sensitivity is a 15m default sweep — no per-config plumbing
                # but at least set the window length so Sharpe is correct.
                window_duration_s=900.0,
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


def build_diffusion_signal(
    market: str,
    *,
    bankroll: float = 10_000.0,
    slippage: float = 0.0,
    min_entry_price: float = 0.10,
    cal_prior_strength: float = 100.0,
    maker_withdraw: float = 60.0,
    oracle_cancel_threshold: float = 0.0,
    oracle_lead_bias: float = 0.0,
    cross_asset_lookup: dict | None = None,
    cross_asset_min_z: float = 0.3,
    min_z: float | None = None,
    edge_persistence_s: float = 0.0,
    maker: bool = False,
    use_regime_classifier: bool = True,
    use_filtration: bool = True,
    filtration_threshold: float = 0.55,
    filtration_mode: str = "size_mult",
    filtration_size_mult_floor: float = 0.45,
    filtration_ev_full: float = 0.50,
    filtration_model_path: str | None = None,
    market_blend_override: float | None = None,
    tail_mode_override: str | None = None,
):
    """Construct the production DiffusionSignal for a given market.

    This is the SINGLE source of truth for how the diffusion signal is
    parameterized. Both `main()` (the CLI) and `scripts/dump_trades.py`
    (and any future runner) MUST go through this function so behavior
    cannot drift between paths.

    Behavior is identical to the inline construction that used to live
    in `main()` — every per-market override and maker/eth/vamp branch
    has been preserved verbatim.
    """
    config = get_config(market)

    # Per-market signal overrides
    eth_overrides = {}
    if market == "eth":
        # M1 fix: don't override edge_threshold here. config.edge_threshold
        # is already plumbed via the explicit `edge_threshold=` kwarg below,
        # so eth_overrides clobbering it would silently win over any future
        # market_config.py change. The other knobs stay because they're
        # eth-specific tuning that doesn't belong in the per-market config
        # (yet).
        eth_overrides = dict(
            reversion_discount=0.10,
            momentum_lookback_s=15,
            momentum_majority=0.7,
            spread_edge_penalty=0.2,
        )

    # Maker mode overrides.
    # 2026-04-09: removed `edge_threshold=0.08` — it collided with the
    # explicit `edge_threshold=config.edge_threshold` kwarg in the
    # DiffusionSignal constructor (TypeError: multiple values). The
    # per-market config's edge_threshold is the tuned value and should
    # flow through unchanged. See btc_5m config comment at
    # market_config.py:229-235 which explicitly warns not to raise it.
    maker_overrides = {}
    if maker:
        maker_overrides = dict(
            maker_mode=True,
            max_bet_fraction=0.02,
            momentum_majority=0.0,
            spread_edge_penalty=0.0,
            # window_duration is now always passed from config (below),
            # so no need to include it in maker_overrides.
        )

    # VAMP mode (per base asset)
    vamp_kw = {}
    base_market = market.replace("_5m", "")
    if base_market == "btc":
        vamp_kw = dict(vamp_mode="cost")
    elif base_market == "eth":
        vamp_kw = dict(vamp_mode="filter", vamp_filter_threshold=0.07)

    # 5m timing overrides
    is_5m = "_5m" in market
    maker_warmup_s = 30.0 if is_5m else 100.0
    maker_withdraw_s = 30.0 if is_5m else maker_withdraw

    # Optional regime classifier (Quant Guild #51). Loads from
    # `regime_classifier_<data_subdir>.pkl` if it exists (e.g.
    # regime_classifier_btc_15m.pkl, regime_classifier_btc_5m.pkl).
    # The data_subdir is the canonical identifier shared with the
    # trainer. None-safe — no behavior change if no model is trained.
    regime_classifier = None
    regime_early_tau_s = None
    if use_regime_classifier:
        try:
            from regime_classifier import RegimeClassifier
            from pathlib import Path as _Path
            pkl_path = (_Path(__file__).parent
                        / f"regime_classifier_{config.data_subdir}.pkl")
            if pkl_path.exists():
                regime_classifier = RegimeClassifier.load(pkl_path)
                # Use the same early-tau the trainer used
                regime_early_tau_s = 200.0 if is_5m else 700.0
        except (ImportError, FileNotFoundError):
            regime_classifier = None

    # Optional filtration model (XGBoost confidence gate). Loads from
    # `filtration_model.pkl` if it exists. Trained across all markets
    # in one model with asset_id as a feature; we pass the per-market
    # asset_id here so the model knows which market it's gating.
    # None-safe — absence of the pkl is a hard no-op (returns True).
    # The _check_filtration gate has an early-exit at abs(z) < 0.10
    # which means it ONLY acts on directional setups, never on
    # indecision ticks.
    filtration_model = None
    filtration_asset_id = 0
    if use_filtration:
        try:
            from filtration_model import FiltrationModel, ASSET_IDS
            from pathlib import Path as _Path
            if filtration_model_path:
                fpkl = _Path(filtration_model_path)
            else:
                fpkl = _Path(__file__).parent / "filtration_model.pkl"
            if fpkl.exists():
                filtration_model = FiltrationModel.load(
                    fpkl, threshold=filtration_threshold
                )
                filtration_asset_id = ASSET_IDS.get(config.data_subdir, 0)
        except (ImportError, FileNotFoundError):
            filtration_model = None

    # Override resolution: CLI value if explicitly provided, else config.
    effective_blend = (market_blend_override
                       if market_blend_override is not None
                       else config.market_blend)
    effective_tail_mode = (tail_mode_override
                           if tail_mode_override is not None
                           else config.tail_mode)
    # min_entry_z: CLI None → config default. Backtest CLI used to
    # default to 0.0 which silently overrode the config's per-market
    # min_entry_z (e.g. btc 15m config says 0.15, but backtest ran at
    # 0.0 — every parameter tuned on that backtest is wrong).
    effective_min_z = min_z if min_z is not None else config.min_entry_z

    return DiffusionSignal(
        bankroll=bankroll,
        slippage=slippage,
        calibration_table=None,  # walk-forward will inject after train pass
        min_entry_price=min_entry_price,
        cal_prior_strength=cal_prior_strength,
        maker_warmup_s=maker_warmup_s,
        maker_withdraw_s=maker_withdraw_s,
        max_sigma=config.max_sigma,
        min_sigma=config.min_sigma,
        edge_threshold=config.edge_threshold,
        max_model_market_disagreement=config.max_model_market_disagreement,
        oracle_cancel_threshold=oracle_cancel_threshold,
        oracle_lead_bias=oracle_lead_bias,
        cross_asset_z_lookup=cross_asset_lookup,
        cross_asset_min_z=cross_asset_min_z,
        min_entry_z=effective_min_z,
        tail_mode=effective_tail_mode,
        tail_nu_default=config.tail_nu_default,
        kou_lambda=config.kou_lambda,
        kou_p_up=config.kou_p_up,
        kou_eta1=config.kou_eta1,
        kou_eta2=config.kou_eta2,
        market_blend=effective_blend,
        max_book_age_ms=config.max_book_age_ms,
        max_chainlink_age_ms=config.max_chainlink_age_ms,
        max_binance_age_ms=config.max_binance_age_ms,
        max_trade_tape_age_ms=config.max_trade_tape_age_ms,
        sigma_estimator=config.sigma_estimator,
        regime_classifier=regime_classifier,
        regime_early_tau_s=regime_early_tau_s,
        filtration_model=filtration_model,
        filtration_threshold=filtration_threshold,
        filtration_asset_id=filtration_asset_id,
        filtration_mode=filtration_mode,
        filtration_size_mult_floor=filtration_size_mult_floor,
        filtration_ev_full=filtration_ev_full,
        hawkes_params=config.hawkes_params,
        data_subdir=config.data_subdir,
        # Always pass window_duration from config. Previously this was
        # only in maker_overrides, so FOK backtests on 5m markets used
        # the class default (900s), which broke dyn_threshold by ~42%
        # and miscomputed inventory_skew and maker_warmup throughout.
        window_duration=config.window_duration_s,
        edge_persistence_s=edge_persistence_s,
        **{**eth_overrides, **maker_overrides, **vamp_kw},
    )


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
    parser.add_argument("--oracle-lead-bias", type=float, default=0.0,
                        help="F4: bias on p_model from Binance→Chainlink lead-lag. "
                             "When binance_mid > chainlink, the next chainlink update will "
                             "likely move UP (rebroadcast tax = ~1.2s). At gap = "
                             "oracle_lag_threshold, bias = oracle_lead_bias (e.g. 0.05 = +5pp). "
                             "Default 0.0 = disabled. Recommended starting value: 0.05.")
    parser.add_argument("--cross-asset-dir", type=str, default=None,
                        help="Secondary asset data subdir for cross-asset disagreement veto "
                             "(e.g. 'eth_15m').")
    parser.add_argument("--cross-asset-min-z", type=float, default=0.3,
                        help="Minimum |z| threshold for cross-asset veto (default 0.3)")
    parser.add_argument("--min-z", type=float, default=None,
                        help="Minimum |z-score| to enter a trade (default: from market_config; "
                             "pass 0.0 to disable, 0.7 for strict filtering)")
    parser.add_argument("--edge-persistence-s", type=float, default=0.0,
                        help="Edge must persist for N seconds before firing (default: 0 = disabled in backtest)")
    parser.add_argument("--market-blend", type=float, default=None,
                        help="Override market_blend from config (0.0 = pure model, "
                             "0.5 = 50/50, 1.0 = pure market consensus). "
                             "If omitted, uses config value (btc_5m=0.3, btc=0.5).")
    parser.add_argument("--tail-mode", default=None,
                        choices=[None, "normal", "kou", "kou_full", "student_t", "market_adaptive"],
                        help="Override tail_mode from config. Use 'kou_full' for the "
                             "proper Kou jump-diffusion path (bipower variation for "
                             "continuous σ + physical-measure drift). Default uses config.")
    parser.add_argument("--no-filtration", action="store_true",
                        help="Disable the XGBoost filtration confidence gate (it "
                             "auto-loads from filtration_model.pkl by default).")
    parser.add_argument("--filtration-threshold", type=float, default=0.55,
                        help="Confidence threshold for the filtration gate (default 0.55). "
                             "Lower = more permissive, higher = more strict.")
    parser.add_argument("--filtration-mode", choices=["gate", "size_mult"],
                        default="size_mult",
                        help="How to use filtration confidence. 'size_mult' (default) "
                             "shrinks Kelly size proportional to confidence (no binary cut). "
                             "'gate' skips trades below threshold (legacy behavior).")
    parser.add_argument("--filtration-size-mult-floor", type=float, default=0.45,
                        help="Confidence floor for size_mult mode (default 0.45). "
                             "Below this, the trade is fully suppressed (mult=0). "
                             "From floor to 1.0, mult scales linearly to 1.0.")
    parser.add_argument("--filtration-ev-full", type=float, default=0.50,
                        help="EV ceiling for regression-mode filtration (default 0.50). "
                             "Predicted PnL/$ at-or-above this saturates the multiplier. "
                             "Below 0, the trade is fully suppressed.")
    parser.add_argument("--filtration-model-path", type=str, default=None,
                        help="Path to filtration model pkl. Defaults to "
                             "filtration_model.pkl. Use filtration_model_pnl.pkl for "
                             "the regression model trained with --target regression.")
    args = parser.parse_args()

    config = get_config(args.market)
    data_dir = DATA_DIR / config.data_subdir

    # Cross-asset disagreement lookup (CLI-only feature)
    cross_asset_lookup = None
    if args.cross_asset_dir:
        secondary_dir = DATA_DIR / args.cross_asset_dir
        if secondary_dir.exists():
            print(f"  Building cross-asset z lookup from {secondary_dir} ...")
            cross_asset_lookup = build_cross_asset_lookup(secondary_dir)
            print(f"  Cross-asset lookup: {len(cross_asset_lookup)} windows indexed")
        else:
            print(f"  WARNING: cross-asset dir not found: {secondary_dir}")

    # 5m timing overrides — printed here for log clarity; the actual
    # values are baked into build_diffusion_signal().
    is_5m = "_5m" in args.market
    if is_5m:
        print(f"  5m overrides: warmup=30s, withdraw=30s")

    signal_map = {
        "diffusion": lambda: build_diffusion_signal(
            args.market,
            bankroll=args.bankroll,
            slippage=args.slippage,
            min_entry_price=args.min_entry_price,
            cal_prior_strength=args.cal_prior_strength,
            maker_withdraw=args.maker_withdraw,
            oracle_cancel_threshold=args.oracle_cancel_threshold,
            oracle_lead_bias=args.oracle_lead_bias,
            cross_asset_lookup=cross_asset_lookup,
            cross_asset_min_z=args.cross_asset_min_z,
            min_z=args.min_z,
            maker=args.maker,
            use_filtration=not args.no_filtration,
            filtration_threshold=args.filtration_threshold,
            filtration_mode=args.filtration_mode,
            filtration_size_mult_floor=args.filtration_size_mult_floor,
            filtration_ev_full=args.filtration_ev_full,
            filtration_model_path=args.filtration_model_path,
            market_blend_override=args.market_blend,
            tail_mode_override=args.tail_mode,
            edge_persistence_s=args.edge_persistence_s,
        ),
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
            # Plumb live-parity knobs from market_config so backtest A/B
            # actually reflects what the live trader will do.
            max_trades_per_window=config.max_trades_per_window,
            same_direction_stacking_only=config.same_direction_stacking_only,
            window_duration_s=config.window_duration_s,
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
