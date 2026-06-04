#!/usr/bin/env python3
"""
Tick-Level Out-of-Sample Backtest Simulator

Replays every 1-second snapshot in randomly selected windows, simulating
realistic maker order lifecycle with adverse-selection-aware fills.

Key differences from backtest.py:
  - Evaluates signal EVERY tick (not every 90s)
  - Maker orders only fill when best_ask <= limit_price (cross-based)
  - Supports dual-side posting (both UP and DOWN simultaneously)
  - Random OOS window selection with separate calibration set
  - Tracks adverse selection metrics post-fill

Usage:
    python tick_backtest.py --market btc_15m --windows 50 --seed 42
    python tick_backtest.py --market btc_15m --windows 50 --dual-side --compare
    python tick_backtest.py --market btc_5m  --windows 100 --calibrated --max-z 1.5
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

from backtest import (
    DATA_DIR,
    CalibrationTable,
    Decision,
    DiffusionSignal,
    Snapshot,
    build_calibration_table,
    norm_cdf,
    poly_fee,
)
from market_config import get_config

# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class RestingOrder:
    side: str               # "UP" or "DOWN"
    limit_price: float      # price posted at (best_bid at placement)
    shares: float
    cost_est: float         # limit_price * shares (reserved from bankroll)
    placed_tick: int        # row index when placed
    placed_ts_ms: int
    model_p: float          # p_model at placement time
    edge_at_entry: float    # edge when order was placed


@dataclass
class SimFill:
    side: str
    entry_price: float
    shares: float
    cost_usd: float
    fill_tick: int
    fill_ts_ms: int
    model_p_at_fill: float
    # Adverse selection: chainlink price delta 30s after fill
    chainlink_at_fill: float = 0.0
    chainlink_30s_later: float = 0.0
    won: bool = False
    payout: float = 0.0
    pnl: float = 0.0


@dataclass
class WindowResult:
    slug: str
    outcome_up: bool
    fills: list[SimFill]
    orders_placed: int
    orders_filled: int
    orders_cancelled: int
    pnl: float
    window_duration_s: float = 0.0


# ── Tick-level simulation ────────────────────────────────────────────────────


def run_window(
    df: pd.DataFrame,
    signal: DiffusionSignal,
    bankroll: float,
    *,
    dual_side: bool = False,
    max_order_age_s: float = 45.0,
    edge_cancel_threshold: float = 0.04,
    requote_cooldown_s: float = 3.0,
    maker_warmup_s: float = 60.0,
    maker_withdraw_s: float = 30.0,
    window_duration_s: float = 900.0,
    max_fills: int = 6,
    min_requote_ticks: int = 2,
) -> tuple[WindowResult, float]:
    """Simulate one window tick-by-tick with realistic maker order lifecycle.

    Returns (WindowResult, updated_bankroll).
    """
    if df.empty:
        return WindowResult("", False, [], 0, 0, 0, 0.0), bankroll

    slug = str(df.iloc[0].get("market_slug", ""))

    # Resolve outcome from final row
    final_row = df.iloc[-1]
    first_row = df.iloc[0]
    chainlink_col = "chainlink_price" if "chainlink_price" in df.columns else "chainlink_btc"

    start_price_val = first_row.get("window_start_price")
    final_price_val = final_row.get(chainlink_col)
    if start_price_val is None or pd.isna(start_price_val):
        return WindowResult(slug, False, [], 0, 0, 0, 0.0), bankroll
    if final_price_val is None or pd.isna(final_price_val):
        return WindowResult(slug, False, [], 0, 0, 0, 0.0), bankroll

    start_price = float(start_price_val)
    final_price = float(final_price_val)
    outcome_up = final_price >= start_price

    # Check window completeness: must end near expiry AND start near window open
    final_remaining = float(final_row.get("time_remaining_s", 999))
    if final_remaining > 5.0:
        return WindowResult(slug, outcome_up, [], 0, 0, 0, 0.0), bankroll
    first_remaining = float(first_row.get("time_remaining_s", 0))
    start_gap = window_duration_s - first_remaining
    if start_gap > 30.0:
        return WindowResult(slug, outcome_up, [], 0, 0, 0, 0.0), bankroll

    # State
    resting: dict[str, RestingOrder | None] = {"UP": None, "DOWN": None}
    fills: list[SimFill] = []
    ctx: dict = {}
    stats = {"placed": 0, "filled": 0, "cancelled": 0}

    # Precompute chainlink prices array for adverse selection lookback
    chainlink_prices = df[chainlink_col].values
    ts_ms_values = df["ts_ms"].values

    n_rows = len(df)

    for tick in range(n_rows):
        row = df.iloc[tick]
        snap = Snapshot.from_row(row)
        if snap is None:
            continue

        tau = snap.time_remaining_s
        elapsed = window_duration_s - tau

        # ── 1. Check fill conditions on resting orders ───────────────
        for side in ["UP", "DOWN"]:
            order = resting[side]
            if order is None:
                continue

            # Fill when best_ask drops to our limit price (someone crosses spread)
            best_ask = snap.best_ask_up if side == "UP" else snap.best_ask_down
            if best_ask is not None and best_ask <= order.limit_price:
                # Compute current p_model for logging
                p_model_now = ctx.get("_p_model_raw", 0.5)

                fill = SimFill(
                    side=side,
                    entry_price=order.limit_price,
                    shares=order.shares,
                    cost_usd=order.cost_est,
                    fill_tick=tick,
                    fill_ts_ms=snap.ts_ms,
                    model_p_at_fill=p_model_now,
                    chainlink_at_fill=snap.chainlink_price,
                )
                fills.append(fill)
                resting[side] = None
                stats["filled"] += 1

        # ── 2. Run signal ────────────────────────────────────────────
        up_dec, down_dec = signal.decide_both_sides(snap, ctx)

        # ── 3. Gate checks ───────────────────────────────────────────
        in_warmup = elapsed < maker_warmup_s
        in_withdraw = tau < maker_withdraw_s

        # Cancel all orders during withdraw
        if in_withdraw:
            for side in ["UP", "DOWN"]:
                if resting[side] is not None:
                    bankroll += resting[side].cost_est
                    resting[side] = None
                    stats["cancelled"] += 1
            continue

        if in_warmup:
            # Still accumulate price history via signal, but don't trade
            continue

        # ── 4. Manage resting orders per side ────────────────────────
        for side, dec in [("UP", up_dec), ("DOWN", down_dec)]:
            order = resting[side]
            best_bid = snap.best_bid_up if side == "UP" else snap.best_bid_down

            if order is not None:
                age_s = (snap.ts_ms - order.placed_ts_ms) / 1000.0
                current_edge = dec.edge if dec.action != "FLAT" else 0.0

                should_cancel = False
                cancel_reason = ""

                if current_edge < edge_cancel_threshold:
                    should_cancel = True
                    cancel_reason = "edge_gone"
                elif age_s > max_order_age_s:
                    should_cancel = True
                    cancel_reason = "age"

                if should_cancel:
                    bankroll += order.cost_est
                    resting[side] = None
                    stats["cancelled"] += 1

                    # If cancelled for age but still has edge, fall through to place new
                    if cancel_reason == "age" and dec.action != "FLAT" and dec.size_usd > 0:
                        pass  # fall through to placement below
                    else:
                        continue

                elif (best_bid is not None
                      and best_bid >= order.limit_price + min_requote_ticks * 0.01
                      and current_edge > order.edge_at_entry
                      and (snap.ts_ms - order.placed_ts_ms) / 1000.0 >= requote_cooldown_s):
                    # Requote at better bid
                    bankroll += order.cost_est
                    resting[side] = None
                    stats["cancelled"] += 1
                    # Fall through to place new order at updated bid
                else:
                    continue  # order is fine, keep resting

            # ── Place new order ──────────────────────────────────────
            if dec.action == "FLAT" or dec.size_usd <= 0:
                continue

            # Max fills per window
            if len(fills) >= max_fills:
                continue

            # Anti-hedge / exposure gating
            opposite = "DOWN" if side == "UP" else "UP"
            has_opposite = any(f.side == opposite for f in fills)
            if has_opposite:
                if not dual_side:
                    continue
                # Smart dual-side: only allow if it reduces net imbalance
                up_sh = sum(f.shares for f in fills if f.side == "UP")
                dn_sh = sum(f.shares for f in fills if f.side == "DOWN")
                net_exp = up_sh - dn_sh
                if side == "UP" and net_exp > 0:
                    continue
                if side == "DOWN" and net_exp < 0:
                    continue

            if best_bid is None or best_bid <= 0 or best_bid >= 1.0:
                continue

            shares = dec.size_usd / best_bid
            cost = shares * best_bid

            if cost > bankroll:
                if bankroll <= 0:
                    continue
                shares = bankroll / best_bid
                cost = bankroll
            if shares < 5.0:
                continue

            bankroll -= cost
            resting[side] = RestingOrder(
                side=side,
                limit_price=best_bid,
                shares=shares,
                cost_est=cost,
                placed_tick=tick,
                placed_ts_ms=snap.ts_ms,
                model_p=ctx.get("_p_model_raw", 0.5),
                edge_at_entry=dec.edge,
            )
            stats["placed"] += 1

    # ── 5. Cancel remaining resting orders ────────────────────────────
    for side in ["UP", "DOWN"]:
        if resting[side] is not None:
            bankroll += resting[side].cost_est
            stats["cancelled"] += 1
            resting[side] = None

    # ── 6. Resolve fills & compute PnL ────────────────────────────────
    total_pnl = 0.0
    for fill in fills:
        won = (fill.side == "UP" and outcome_up) or (fill.side == "DOWN" and not outcome_up)
        payout = fill.shares if won else 0.0
        pnl = payout - fill.cost_usd
        fill.won = won
        fill.payout = payout
        fill.pnl = pnl
        total_pnl += pnl
        bankroll += payout

    # ── 7. Backfill adverse selection (30s after fill) ────────────────
    for fill in fills:
        future_tick = fill.fill_tick + 30
        if future_tick < len(chainlink_prices):
            fill.chainlink_30s_later = float(chainlink_prices[future_tick])
        else:
            fill.chainlink_30s_later = float(chainlink_prices[-1])

    # Update signal bankroll for Kelly sizing in next window
    signal.bankroll = bankroll

    result = WindowResult(
        slug=slug,
        outcome_up=outcome_up,
        fills=fills,
        orders_placed=stats["placed"],
        orders_filled=stats["filled"],
        orders_cancelled=stats["cancelled"],
        pnl=total_pnl,
        window_duration_s=window_duration_s,
    )
    return result, bankroll


def run_window_instant(
    df: pd.DataFrame,
    signal: DiffusionSignal,
    bankroll: float,
    *,
    dual_side: bool = False,
    maker_warmup_s: float = 60.0,
    maker_withdraw_s: float = 30.0,
    window_duration_s: float = 900.0,
    cooldown_s: float = 5.0,
    max_fills: int = 6,
) -> tuple[WindowResult, float]:
    """Instant-fill comparison mode: fill at best_bid immediately on signal.

    Used for --compare to quantify the adverse selection gap.
    """
    if df.empty:
        return WindowResult("", False, [], 0, 0, 0, 0.0), bankroll

    slug = str(df.iloc[0].get("market_slug", ""))
    chainlink_col = "chainlink_price" if "chainlink_price" in df.columns else "chainlink_btc"

    start_price_val = df.iloc[0].get("window_start_price")
    final_price_val = df.iloc[-1].get(chainlink_col)
    if start_price_val is None or pd.isna(start_price_val):
        return WindowResult(slug, False, [], 0, 0, 0, 0.0), bankroll
    if final_price_val is None or pd.isna(final_price_val):
        return WindowResult(slug, False, [], 0, 0, 0, 0.0), bankroll

    start_price = float(start_price_val)
    final_price = float(final_price_val)
    outcome_up = final_price >= start_price
    final_remaining = float(df.iloc[-1].get("time_remaining_s", 999))
    if final_remaining > 5.0:
        return WindowResult(slug, outcome_up, [], 0, 0, 0, 0.0), bankroll
    first_remaining = float(df.iloc[0].get("time_remaining_s", 0))
    start_gap = window_duration_s - first_remaining
    if start_gap > 30.0:
        return WindowResult(slug, outcome_up, [], 0, 0, 0, 0.0), bankroll

    fills: list[SimFill] = []
    ctx: dict = {}
    stats = {"placed": 0, "filled": 0, "cancelled": 0}
    last_fill_ts = 0
    chainlink_prices = df[chainlink_col].values

    n_rows = len(df)
    for tick in range(n_rows):
        row = df.iloc[tick]
        snap = Snapshot.from_row(row)
        if snap is None:
            continue

        tau = snap.time_remaining_s
        elapsed = window_duration_s - tau

        if elapsed < maker_warmup_s or tau < maker_withdraw_s:
            # Still run signal for price history
            signal.decide_both_sides(snap, ctx)
            continue

        up_dec, down_dec = signal.decide_both_sides(snap, ctx)

        # Cooldown
        if snap.ts_ms - last_fill_ts < cooldown_s * 1000:
            continue

        for side, dec in [("UP", up_dec), ("DOWN", down_dec)]:
            if dec.action == "FLAT" or dec.size_usd <= 0:
                continue

            if len(fills) >= max_fills:
                continue

            has_opposite = any(f.side != side for f in fills)
            if has_opposite:
                if not dual_side:
                    continue
                up_sh = sum(f.shares for f in fills if f.side == "UP")
                dn_sh = sum(f.shares for f in fills if f.side == "DOWN")
                net_exp = up_sh - dn_sh
                if side == "UP" and net_exp > 0:
                    continue
                if side == "DOWN" and net_exp < 0:
                    continue

            best_bid = snap.best_bid_up if side == "UP" else snap.best_bid_down
            if best_bid is None or best_bid <= 0 or best_bid >= 1.0:
                continue

            shares = dec.size_usd / best_bid
            cost = shares * best_bid
            if cost > bankroll:
                if bankroll <= 0:
                    continue
                shares = bankroll / best_bid
                cost = bankroll
            if shares < 5.0:
                continue

            bankroll -= cost
            stats["placed"] += 1
            stats["filled"] += 1

            p_model_now = ctx.get("_p_model_raw", 0.5)
            fill = SimFill(
                side=side,
                entry_price=best_bid,
                shares=shares,
                cost_usd=cost,
                fill_tick=tick,
                fill_ts_ms=snap.ts_ms,
                model_p_at_fill=p_model_now,
                chainlink_at_fill=snap.chainlink_price,
            )
            fills.append(fill)
            last_fill_ts = snap.ts_ms

    # Resolve
    total_pnl = 0.0
    for fill in fills:
        won = (fill.side == "UP" and outcome_up) or (fill.side == "DOWN" and not outcome_up)
        payout = fill.shares if won else 0.0
        pnl = payout - fill.cost_usd
        fill.won = won
        fill.payout = payout
        fill.pnl = pnl
        total_pnl += pnl
        bankroll += payout

        future_tick = fill.fill_tick + 30
        if future_tick < len(chainlink_prices):
            fill.chainlink_30s_later = float(chainlink_prices[future_tick])
        else:
            fill.chainlink_30s_later = float(chainlink_prices[-1])

    signal.bankroll = bankroll

    return WindowResult(
        slug=slug, outcome_up=outcome_up, fills=fills,
        orders_placed=stats["placed"], orders_filled=stats["filled"],
        orders_cancelled=stats["cancelled"], pnl=total_pnl,
        window_duration_s=window_duration_s,
    ), bankroll


# ── Aggregation & output ─────────────────────────────────────────────────────


def compute_adverse_selection(fills: list[SimFill], start_price: float) -> dict:
    """Compute adverse selection metrics from fills."""
    up_moves = []
    down_moves = []

    for f in fills:
        if f.chainlink_at_fill == 0 or f.chainlink_30s_later == 0:
            continue
        # Fractional price change 30s after fill
        delta = (f.chainlink_30s_later - f.chainlink_at_fill) / f.chainlink_at_fill
        if f.side == "UP":
            # For UP buyer: price dropping is adverse
            up_moves.append(delta)
        else:
            # For DOWN buyer: price rising is adverse
            down_moves.append(-delta)  # flip so negative = adverse for both

    return {
        "up_avg": np.mean(up_moves) if up_moves else 0.0,
        "down_avg": np.mean(down_moves) if down_moves else 0.0,
        "up_count": len(up_moves),
        "down_count": len(down_moves),
        "overall_avg": np.mean(up_moves + down_moves) if (up_moves or down_moves) else 0.0,
    }


def print_results(
    results: list[WindowResult],
    *,
    market: str,
    n_windows: int,
    seed: int,
    dual_side: bool,
    calibrated: bool,
    bankroll_start: float,
    bankroll_end: float,
    elapsed_s: float,
    compare_results: list[WindowResult] | None = None,
):
    """Print formatted results summary."""
    all_fills = [f for r in results for f in r.fills]
    total_placed = sum(r.orders_placed for r in results)
    total_filled = sum(r.orders_filled for r in results)
    total_cancelled = sum(r.orders_cancelled for r in results)
    total_pnl = sum(r.pnl for r in results)
    windows_traded = sum(1 for r in results if r.fills)

    fill_rate = total_filled / total_placed if total_placed > 0 else 0.0

    # Win stats
    wins = [f for f in all_fills if f.won]
    losses = [f for f in all_fills if not f.won]
    win_rate = len(wins) / len(all_fills) if all_fills else 0.0

    # Size-weighted win rate
    total_cost = sum(f.cost_usd for f in all_fills)
    win_cost = sum(f.cost_usd for f in wins)
    win_rate_weighted = win_cost / total_cost if total_cost > 0 else 0.0

    # Avg PnL
    avg_pnl = total_pnl / len(all_fills) if all_fills else 0.0
    avg_win_pnl = np.mean([f.pnl for f in wins]) if wins else 0.0
    avg_loss_pnl = np.mean([f.pnl for f in losses]) if losses else 0.0

    # Per-side
    up_fills = [f for f in all_fills if f.side == "UP"]
    down_fills = [f for f in all_fills if f.side == "DOWN"]
    up_wins = sum(1 for f in up_fills if f.won)
    down_wins = sum(1 for f in down_fills if f.won)
    up_pnl = sum(f.pnl for f in up_fills)
    down_pnl = sum(f.pnl for f in down_fills)

    # Avg fill price
    avg_fill_price = np.mean([f.entry_price for f in all_fills]) if all_fills else 0.0

    # Avg resting time (for realistic fills, approximate from tick indices)
    resting_times = []
    for r in results:
        for f in r.fills:
            # Each tick is ~1 second
            resting_times.append(f.fill_tick)  # rough proxy

    # Sharpe ratio (per-window PnL series)
    window_pnls = [r.pnl for r in results if r.fills]
    if len(window_pnls) >= 2:
        pnl_std = np.std(window_pnls, ddof=1)
        sharpe = (np.mean(window_pnls) / pnl_std * math.sqrt(96)) if pnl_std > 0 else 0.0
    else:
        sharpe = 0.0

    # Max drawdown
    cumulative = np.cumsum([r.pnl for r in results])
    peak = np.maximum.accumulate(cumulative + bankroll_start)
    drawdown = peak - (cumulative + bankroll_start)
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
    max_dd_pct = max_dd / bankroll_start if bankroll_start > 0 else 0.0

    # Adverse selection
    adv = compute_adverse_selection(all_fills, 0.0)

    # Avg entry price for winners vs losers
    avg_win_entry = np.mean([f.entry_price for f in wins]) if wins else 0.0
    avg_loss_entry = np.mean([f.entry_price for f in losses]) if losses else 0.0

    ds_str = "ON" if dual_side else "OFF"
    cal_str = "ON" if calibrated else "OFF"

    print(f"\n{'=' * 64}")
    print(f"  TICK-LEVEL OOS BACKTEST  |  {market}  |  {n_windows} windows")
    print(f"  Seed: {seed}  |  Dual-side: {ds_str}  |  Calibrated: {cal_str}")
    print(f"  Elapsed: {elapsed_s:.1f}s")
    print(f"{'=' * 64}")

    print(f"\n  FILL STATISTICS")
    print(f"  {'─' * 56}")
    print(f"  Orders placed:            {total_placed}")
    print(f"  Orders filled:            {total_filled}  ({fill_rate:.1%} fill rate)")
    print(f"  Orders cancelled:         {total_cancelled}")
    print(f"  Avg fill price:           {avg_fill_price:.4f}")
    print(f"  Avg entry (winners):      {avg_win_entry:.4f}")
    print(f"  Avg entry (losers):       {avg_loss_entry:.4f}")

    print(f"\n  ADVERSE SELECTION")
    print(f"  {'─' * 56}")
    print(f"  Avg 30s price move (UP fills):   {adv['up_avg']:+.6f}  (n={adv['up_count']})")
    print(f"  Avg 30s price move (DOWN fills): {adv['down_avg']:+.6f}  (n={adv['down_count']})")
    print(f"  Overall avg (negative=adverse):  {adv['overall_avg']:+.6f}")

    print(f"\n  PnL SUMMARY")
    print(f"  {'─' * 56}")
    print(f"  Windows traded:           {windows_traded} / {n_windows}")
    print(f"  Total fills:              {len(all_fills)}")
    print(f"  Wins:                     {len(wins)} ({win_rate:.1%})")
    print(f"  Win rate (size-weighted): {win_rate_weighted:.1%}")
    print(f"  Total PnL:                ${total_pnl:+,.2f}")
    print(f"  Avg PnL per fill:         ${avg_pnl:+,.2f}")
    print(f"  Avg winning PnL:          ${avg_win_pnl:+,.2f}")
    print(f"  Avg losing PnL:           ${avg_loss_pnl:+,.2f}")
    if avg_loss_pnl != 0:
        print(f"  Win/Loss ratio:           {abs(avg_win_pnl / avg_loss_pnl):.2f}x")
    print(f"  Sharpe (annualized):      {sharpe:.2f}")
    print(f"  Max drawdown:             ${max_dd:,.2f} ({max_dd_pct:.1%})")
    print(f"  Final bankroll:           ${bankroll_end:,.2f}")

    print(f"\n  PER-SIDE BREAKDOWN")
    print(f"  {'─' * 56}")
    up_wr = up_wins / len(up_fills) if up_fills else 0.0
    down_wr = down_wins / len(down_fills) if down_fills else 0.0
    print(f"  UP:    {len(up_fills):>4} fills, {up_wins:>3} wins ({up_wr:.1%}), PnL=${up_pnl:+,.2f}")
    print(f"  DOWN:  {len(down_fills):>4} fills, {down_wins:>3} wins ({down_wr:.1%}), PnL=${down_pnl:+,.2f}")

    # Comparison mode
    if compare_results is not None:
        comp_fills = [f for r in compare_results for f in r.fills]
        comp_pnl = sum(r.pnl for r in compare_results)
        comp_wins = sum(1 for f in comp_fills if f.won)
        comp_wr = comp_wins / len(comp_fills) if comp_fills else 0.0
        gap = comp_pnl - total_pnl
        gap_pct = gap / comp_pnl if comp_pnl != 0 else 0.0

        print(f"\n  COMPARISON: INSTANT FILLS vs REALISTIC FILLS")
        print(f"  {'─' * 56}")
        print(f"  Instant-fill PnL:         ${comp_pnl:+,.2f}  ({len(comp_fills)} fills, {comp_wr:.1%} WR)")
        print(f"  Realistic-fill PnL:       ${total_pnl:+,.2f}  ({len(all_fills)} fills, {win_rate:.1%} WR)")
        print(f"  Gap (adverse selection):   ${gap:+,.2f} ({gap_pct:.1%} of theoretical)")

    print(f"\n{'=' * 64}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Tick-level OOS backtest with realistic maker fills",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--market", "-m", default="btc_15m",
                        help="Market to test (btc_15m, btc_5m, eth_15m, etc.)")
    parser.add_argument("--windows", "-w", type=int, default=50,
                        help="Number of random windows to simulate")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--dual-side", action="store_true",
                        help="Enable dual-side posting (no anti-hedge)")
    parser.add_argument("--calibrated", action="store_true",
                        help="Build and use calibration table from non-selected windows")
    parser.add_argument("--cal-max-weight", type=float, default=0.70,
                        help="Max calibration weight in Bayesian mix (0=pure GBM, 1=pure cal, default: 0.70)")
    parser.add_argument("--cal-prior-strength", type=float, default=50.0,
                        help="Prior strength n0 for calibration ramp-up (default: 50)")
    parser.add_argument("--compare", action="store_true",
                        help="Also run instant-fill mode for comparison")

    # Signal overrides
    parser.add_argument("--max-z", type=float, default=None,
                        help="Override max_z")
    parser.add_argument("--reversion-discount", type=float, default=None,
                        help="Override reversion_discount")
    parser.add_argument("--edge-threshold", type=float, default=0.08,
                        help="Edge threshold for entry")
    parser.add_argument("--early-edge-mult", type=float, default=1.2,
                        help="Early edge multiplier")
    parser.add_argument("--max-bet-fraction", type=float, default=0.05,
                        help="Max fraction of bankroll per trade")
    parser.add_argument("--kelly-fraction", type=float, default=0.25,
                        help="Kelly fraction for sizing (0.25 = quarter-Kelly)")

    # Order management
    parser.add_argument("--max-order-age", type=float, default=45.0,
                        help="Cancel orders older than N seconds")
    parser.add_argument("--edge-cancel", type=float, default=0.04,
                        help="Cancel orders when edge drops below this")
    parser.add_argument("--requote-cooldown", type=float, default=3.0,
                        help="Minimum seconds between requotes")
    parser.add_argument("--maker-warmup", type=float, default=None,
                        help="Seconds before first order (default: 60 for 15m, 30 for 5m)")
    parser.add_argument("--maker-withdraw", type=float, default=None,
                        help="Stop orders when tau < N (default: 60 for 15m, 20 for 5m)")

    parser.add_argument("--max-fills", type=int, default=6,
                        help="Max fills (positions) per window (default: 6)")
    parser.add_argument("--min-requote-ticks", type=int, default=2,
                        help="Min tick improvement before requoting (default: 2 = $0.02)")
    parser.add_argument("--min-sigma", type=float, default=None,
                        help="Override min_sigma floor (default: from market_config)")

    # Fat-tail CDF
    parser.add_argument("--tail-mode", choices=["student_t", "normal"],
                        default="student_t",
                        help="CDF for z→probability (default: student_t)")
    parser.add_argument("--tail-nu", type=float, default=3.0,
                        help="Student-t nu floor / default (default: 3.0 = data-driven)")

    # Avellaneda-Stoikov unified quoting
    parser.add_argument("--as-mode", action="store_true", default=False,
                        help="Enable A-S reservation price quoting (default: off)")
    parser.add_argument("--gamma-inv", type=float, default=0.15,
                        help="A-S gamma for inventory penalty (default: 0.15)")
    parser.add_argument("--gamma-spread", type=float, default=0.75,
                        help="A-S gamma for base spread (default: 0.75)")
    parser.add_argument("--min-edge", type=float, default=0.05,
                        help="Floor on required edge in A-S mode (default: 0.05)")
    parser.add_argument("--tox-spread", type=float, default=0.05,
                        help="Additive spread from toxicity (default: 0.05)")
    parser.add_argument("--vpin-spread", type=float, default=0.05,
                        help="Additive spread from VPIN (default: 0.05)")
    parser.add_argument("--lag-spread", type=float, default=0.08,
                        help="Additive spread from oracle lag (default: 0.08)")
    parser.add_argument("--edge-step", type=float, default=0.01,
                        help="Additive spread per fill (default: 0.01)")
    parser.add_argument("--contract-vol-lookback", type=int, default=60,
                        help="Lookback (s) for contract mid vol (default: 60)")

    # Bankroll
    parser.add_argument("--bankroll", type=float, default=1000.0,
                        help="Initial bankroll")

    args = parser.parse_args()

    # ── Load market config ──────────────────────────────────────────
    # Accept both "btc_15m" and "btc" style keys
    market_key = args.market
    _ALIASES = {"btc_15m": "btc", "eth_15m": "eth", "sol_15m": "sol", "xrp_15m": "xrp"}
    market_key = _ALIASES.get(market_key, market_key)

    config = get_config(market_key)
    data_dir = DATA_DIR / config.data_subdir
    window_duration_s = config.window_duration_s
    is_5m = market_key.endswith("_5m")

    if not data_dir.exists():
        print(f"ERROR: data directory not found: {data_dir}")
        sys.exit(1)

    # ── Discover parquet files ──────────────────────────────────────
    all_files = sorted(data_dir.glob("*.parquet"))
    if not all_files:
        print(f"ERROR: no parquet files in {data_dir}")
        sys.exit(1)

    print(f"Found {len(all_files)} windows in {data_dir}")

    # ── Split: calibration (first 60%) vs sample pool (last 40%) ───
    split_idx = int(len(all_files) * 0.60)
    cal_files = all_files[:split_idx]
    pool_files = all_files[split_idx:]

    if len(pool_files) < args.windows:
        print(f"WARNING: only {len(pool_files)} files in OOS pool, requested {args.windows}")
        args.windows = len(pool_files)

    # Random selection from pool
    rng = random.Random(args.seed)
    selected_files = rng.sample(pool_files, args.windows)
    selected_files.sort()  # process in chronological order

    print(f"Calibration set: {len(cal_files)} windows")
    print(f"OOS pool: {len(pool_files)} windows")
    print(f"Selected: {args.windows} windows (seed={args.seed})")

    # ── Build calibration table ─────────────────────────────────────
    cal_table = None
    if args.calibrated:
        print("Building calibration table from calibration set...", end=" ", flush=True)
        try:
            cal_table = build_calibration_table(data_dir, vol_lookback_s=90)
            n_cells = len(cal_table.table)
            n_obs = sum(cal_table.counts.values())
            print(f"{n_cells} cells, {n_obs} observations")
        except Exception as exc:
            print(f"FAILED: {exc}")
            cal_table = None

    # ── Configure signal ────────────────────────────────────────────
    # Defaults based on market
    default_max_z = 0.5 if "btc" in args.market else (0.7 if "eth" in args.market else 1.0)
    default_reversion = 0.30 if "btc" in args.market else (0.20 if "eth" in args.market else 0.0)

    max_z = args.max_z if args.max_z is not None else default_max_z
    reversion_discount = args.reversion_discount if args.reversion_discount is not None else default_reversion

    # Warmup / withdraw defaults
    if args.maker_warmup is not None:
        maker_warmup = args.maker_warmup
    else:
        maker_warmup = 30.0 if is_5m else 60.0

    if args.maker_withdraw is not None:
        maker_withdraw = args.maker_withdraw
    else:
        maker_withdraw = 20.0 if is_5m else 60.0

    signal_kw = dict(
        bankroll=args.bankroll,
        max_z=max_z,
        reversion_discount=reversion_discount,
        edge_threshold=args.edge_threshold,
        early_edge_mult=args.early_edge_mult,
        max_bet_fraction=args.max_bet_fraction,
        kelly_fraction=args.kelly_fraction,
        window_duration=window_duration_s,
        maker_mode=True,
        maker_warmup_s=maker_warmup,
        maker_withdraw_s=maker_withdraw,
        momentum_majority=0.0,       # disabled for maker mode
        spread_edge_penalty=0.0,     # handled by fill model instead
        slippage=0.0,                # no slippage for maker
        min_sigma=args.min_sigma if args.min_sigma is not None else config.min_sigma,
        max_sigma=config.max_sigma,
        vol_lookback_s=30 if is_5m else 90,
    )
    if cal_table is not None:
        signal_kw["calibration_table"] = cal_table
        signal_kw["cal_prior_strength"] = args.cal_prior_strength
        signal_kw["cal_max_weight"] = args.cal_max_weight

    signal_kw["tail_mode"] = args.tail_mode
    signal_kw["tail_nu_default"] = args.tail_nu

    # A-S quoting params
    signal_kw["as_mode"] = args.as_mode
    signal_kw["gamma_inv"] = args.gamma_inv
    signal_kw["gamma_spread"] = args.gamma_spread
    signal_kw["min_edge"] = args.min_edge
    signal_kw["tox_spread"] = args.tox_spread
    signal_kw["vpin_spread"] = args.vpin_spread
    signal_kw["lag_spread"] = args.lag_spread
    signal_kw["edge_step"] = args.edge_step
    signal_kw["contract_vol_lookback_s"] = args.contract_vol_lookback

    signal = DiffusionSignal(**signal_kw)

    print(f"\nSignal config:")
    print(f"  max_z={max_z}, reversion={reversion_discount}, edge={args.edge_threshold}")
    print(f"  warmup={maker_warmup}s, withdraw={maker_withdraw}s")
    print(f"  max_order_age={args.max_order_age}s, edge_cancel={args.edge_cancel}")
    print(f"  dual_side={args.dual_side}, tail_mode={args.tail_mode}, tail_nu={args.tail_nu}")
    if args.as_mode:
        print(f"  A-S mode: gamma_inv={args.gamma_inv}, gamma_spread={args.gamma_spread}, min_edge={args.min_edge}")
        print(f"  A-S spread: tox={args.tox_spread}, vpin={args.vpin_spread}, lag={args.lag_spread}, step={args.edge_step}")

    # ── Run simulation ──────────────────────────────────────────────
    print(f"\nRunning tick-level simulation on {args.windows} windows...")
    t_start = _time.time()

    bankroll = args.bankroll
    results: list[WindowResult] = []

    for i, fpath in enumerate(selected_files):
        df = pd.read_parquet(fpath)
        if df.empty:
            continue

        # Normalize column names
        if "chainlink_btc" in df.columns and "chainlink_price" not in df.columns:
            df = df.rename(columns={"chainlink_btc": "chainlink_price"})

        # Sort by timestamp
        df = df.sort_values("ts_ms").reset_index(drop=True)

        # Reset signal context for new window
        ctx_backup = {}
        signal.bankroll = bankroll

        result, bankroll = run_window(
            df, signal, bankroll,
            dual_side=args.dual_side,
            max_order_age_s=args.max_order_age,
            edge_cancel_threshold=args.edge_cancel,
            requote_cooldown_s=args.requote_cooldown,
            maker_warmup_s=maker_warmup,
            maker_withdraw_s=maker_withdraw,
            window_duration_s=window_duration_s,
            max_fills=args.max_fills,
            min_requote_ticks=args.min_requote_ticks,
        )
        results.append(result)

        # Progress
        n_fills = len(result.fills)
        if (i + 1) % 10 == 0 or i == len(selected_files) - 1:
            total_fills = sum(len(r.fills) for r in results)
            running_pnl = sum(r.pnl for r in results)
            print(f"  [{i+1}/{args.windows}] fills={total_fills}, PnL=${running_pnl:+,.2f}, bankroll=${bankroll:,.2f}")

    elapsed = _time.time() - t_start

    # ── Comparison mode ─────────────────────────────────────────────
    compare_results = None
    if args.compare:
        print(f"\nRunning instant-fill comparison...")
        signal_cmp = DiffusionSignal(**signal_kw)
        bankroll_cmp = args.bankroll
        compare_results = []

        for fpath in selected_files:
            df = pd.read_parquet(fpath)
            if df.empty:
                continue
            if "chainlink_btc" in df.columns and "chainlink_price" not in df.columns:
                df = df.rename(columns={"chainlink_btc": "chainlink_price"})
            df = df.sort_values("ts_ms").reset_index(drop=True)

            signal_cmp.bankroll = bankroll_cmp
            result, bankroll_cmp = run_window_instant(
                df, signal_cmp, bankroll_cmp,
                dual_side=args.dual_side,
                maker_warmup_s=maker_warmup,
                maker_withdraw_s=maker_withdraw,
                window_duration_s=window_duration_s,
            )
            compare_results.append(result)

    # ── Print results ───────────────────────────────────────────────
    print_results(
        results,
        market=args.market,
        n_windows=args.windows,
        seed=args.seed,
        dual_side=args.dual_side,
        calibrated=args.calibrated,
        bankroll_start=args.bankroll,
        bankroll_end=bankroll,
        elapsed_s=elapsed,
        compare_results=compare_results,
    )


if __name__ == "__main__":
    main()
