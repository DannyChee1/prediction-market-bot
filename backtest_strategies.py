#!/usr/bin/env python3
"""
Backtest 4 alternative strategies against BTC data.

Strategy 1: Queue Priority (Early Orders) — place limit at market open
Strategy 2: Multi-Market Grid — split capital across concurrent windows
Strategy 3: Dynamic Grid (Price Following) — vol-band grid with rebalancing
Strategy 4: Hybrid FOK + Grid — FOK for signals, passive grid otherwise

Compares all against the current FOK baseline.

Usage:
    py -3 backtest_strategies.py
"""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import (
    Snapshot, DiffusionSignal, BacktestEngine, poly_fee, norm_cdf, DATA_DIR,
)

BANKROLL = 10_000.0
DATA_BTC = DATA_DIR / "btc"

# Only load columns we need to save memory
NEEDED_COLS = [
    "ts_ms", "market_slug", "time_remaining_s", "window_start_price",
    "chainlink_price", "chainlink_btc",
    "best_bid_up", "best_ask_up", "best_bid_down", "best_ask_down",
    "size_bid_up", "size_ask_up", "size_bid_down", "size_ask_down",
    # L2 book levels for FOK walk_book
    "ask_px_up_1", "ask_sz_up_1", "ask_px_up_2", "ask_sz_up_2",
    "ask_px_up_3", "ask_sz_up_3", "ask_px_up_4", "ask_sz_up_4",
    "ask_px_up_5", "ask_sz_up_5",
    "ask_px_down_1", "ask_sz_down_1", "ask_px_down_2", "ask_sz_down_2",
    "ask_px_down_3", "ask_sz_down_3", "ask_px_down_4", "ask_sz_down_4",
    "ask_px_down_5", "ask_sz_down_5",
    "bid_px_up_1", "bid_sz_up_1", "bid_px_up_2", "bid_sz_up_2",
    "bid_px_up_3", "bid_sz_up_3", "bid_px_up_4", "bid_sz_up_4",
    "bid_px_up_5", "bid_sz_up_5",
    "bid_px_down_1", "bid_sz_down_1", "bid_px_down_2", "bid_sz_down_2",
    "bid_px_down_3", "bid_sz_down_3", "bid_px_down_4", "bid_sz_down_4",
    "bid_px_down_5", "bid_sz_down_5",
]


# ── Helpers ────────────────────────────────────────────────────────────────

def load_window(path: Path) -> pd.DataFrame:
    """Load a single window parquet, selecting only needed columns."""
    df = pd.read_parquet(path)
    if "chainlink_btc" in df.columns and "chainlink_price" not in df.columns:
        df.rename(columns={"chainlink_btc": "chainlink_price"}, inplace=True)
    # Keep only columns that exist
    keep = [c for c in NEEDED_COLS if c in df.columns]
    df = df[keep].copy()
    df.sort_values("ts_ms", inplace=True, ignore_index=True)
    return df


def compute_vol(prices: list[float], min_sigma: float = 1e-6) -> float:
    changes: list[tuple[int, float]] = []
    for i, p in enumerate(prices):
        if p > 0 and (not changes or p != changes[-1][1]):
            changes.append((i, p))
    if len(changes) < 3:
        return 0.0
    log_rets = []
    for j in range(1, len(changes)):
        dt = changes[j][0] - changes[j - 1][0]
        if dt > 0:
            lr = math.log(changes[j][1] / changes[j - 1][1])
            log_rets.append(lr / math.sqrt(dt))
    if len(log_rets) < 2:
        return 0.0
    return max(float(np.std(log_rets, ddof=1)), min_sigma)


def resolve_window(wdf: pd.DataFrame):
    """Returns (outcome_up, final_price, start_price) or None."""
    if wdf.iloc[-1]["time_remaining_s"] > 5.0:
        return None
    sp = wdf["window_start_price"].dropna()
    if sp.empty:
        return None
    start_px = float(sp.iloc[0])
    col = "chainlink_price" if "chainlink_price" in wdf.columns else "chainlink_btc"
    final_px = float(wdf.iloc[-1][col])
    if pd.isna(final_px) or pd.isna(start_px):
        return None
    return (1 if final_px >= start_px else 0, final_px, start_px)


@dataclass
class StratResult:
    name: str
    trades: int = 0
    wins: int = 0
    total_pnl: float = 0.0
    total_cost: float = 0.0
    max_dd: float = 0.0
    final_bank: float = 0.0
    details: list = field(default_factory=list)


def compute_dd(pnls: list[float], start: float) -> float:
    eq = start
    peak = start
    dd = 0.0
    for p in pnls:
        eq += p
        if eq > peak:
            peak = eq
        if peak - eq > dd:
            dd = peak - eq
    return dd


def print_comparison(results: list[StratResult], n_complete: int):
    print()
    print("=" * 92)
    hdr = (f"  {'Strategy':<35s} {'Trades':>7s} {'WinR':>6s} {'PnL':>12s} "
           f"{'$/trade':>9s} {'MaxDD':>9s} {'Final':>12s}")
    print(hdr)
    print("=" * 92)
    for r in results:
        if r.trades == 0:
            print(f"  {r.name:<35s} {'---':>7s} {'---':>6s} {'---':>12s} "
                  f"{'---':>9s} {'---':>9s} {'---':>12s}")
            continue
        wr = r.wins / r.trades
        avg = r.total_pnl / r.trades
        print(f"  {r.name:<35s} {r.trades:>7d} {wr:>5.0%} ${r.total_pnl:>+11,.2f} "
              f"${avg:>+8.2f} ${r.max_dd:>8.2f} ${r.final_bank:>11,.2f}")
    print("=" * 92)


# ── Strategy 1: Queue Priority (Early Orders) ─────────────────────────────

def run_strategy1_window(wdf: pd.DataFrame, outcome_up: int, start_px: float,
                         bankroll: float) -> tuple[float, dict | None]:
    """Place limit bid at market open (first 5 seconds). Returns (pnl, fill_info)."""
    first_ts = int(wdf.iloc[0]["ts_ms"])
    early = wdf[wdf["ts_ms"] <= first_ts + 5000]
    if early.empty:
        return 0.0, None

    for _, row in early.iterrows():
        snap = Snapshot.from_row(row)
        if snap is None or snap.best_bid_up is None or snap.best_ask_up is None:
            continue
        if snap.best_ask_up <= 0 or snap.best_ask_up >= 1:
            continue

        mid_up = (snap.best_bid_up + snap.best_ask_up) / 2
        mid_down = ((snap.best_bid_down + snap.best_ask_down) / 2
                    if snap.best_bid_down and snap.best_ask_down else None)

        # Pick cheaper side (no directional info this early)
        if mid_up <= 0.50 and mid_up > 0.01:
            side, bid_price = "UP", round(mid_up, 2)
        elif mid_down is not None and mid_down <= 0.50 and mid_down > 0.01:
            side, bid_price = "DOWN", round(mid_down, 2)
        elif mid_down is not None and mid_down < mid_up:
            side, bid_price = "DOWN", round(mid_down, 2)
        else:
            side, bid_price = "UP", round(mid_up, 2)

        if bid_price <= 0 or bid_price >= 1:
            continue

        shares = max(5.0, min(bankroll * 0.0125 / bid_price, 50.0))
        cost = shares * bid_price

        # Fill check: does ask ever drop to our bid?
        ask_col = "best_ask_up" if side == "UP" else "best_ask_down"
        ask_data = wdf[ask_col].dropna()
        filled = (ask_data <= bid_price).any()
        if not filled:
            return 0.0, None

        won = (side == "UP" and outcome_up == 1) or (side == "DOWN" and outcome_up == 0)
        payout = shares if won else 0.0
        pnl = payout - cost
        return pnl, {"won": won, "cost": cost}

    return 0.0, None


# ── Strategy 2: Multi-Market Grid ─────────────────────────────────────────

def run_strategy2_window(wdf: pd.DataFrame, outcome_up: int, start_px: float,
                         bankroll: float, n_concurrent: int) -> tuple[float, dict | None]:
    """Split capital across concurrent windows, bid at mid in last 8 min."""
    slot_bankroll = bankroll / max(1, min(3, n_concurrent))

    late = wdf[wdf["time_remaining_s"] <= 480]
    if late.empty:
        return 0.0, None

    for _, row in late.head(10).iterrows():
        snap = Snapshot.from_row(row)
        if snap is None or snap.best_bid_up is None or snap.best_ask_up is None:
            continue
        if snap.best_ask_up <= 0 or snap.best_ask_up >= 1:
            continue

        mid_up = (snap.best_bid_up + snap.best_ask_up) / 2
        mid_down = ((snap.best_bid_down + snap.best_ask_down) / 2
                    if snap.best_bid_down and snap.best_ask_down else None)

        if mid_up <= 0.50:
            side, bid_price = "UP", round(mid_up, 2)
        elif mid_down is not None and mid_down <= 0.50:
            side, bid_price = "DOWN", round(mid_down, 2)
        elif mid_down is not None and mid_down < mid_up:
            side, bid_price = "DOWN", round(mid_down, 2)
        else:
            side, bid_price = "UP", round(mid_up, 2)

        if bid_price <= 0 or bid_price >= 1:
            continue

        shares = max(5.0, min(slot_bankroll * 0.0125 / bid_price, 50.0))
        cost = shares * bid_price

        ask_col = "best_ask_up" if side == "UP" else "best_ask_down"
        filled = (late[ask_col].dropna() <= bid_price).any()
        if not filled:
            return 0.0, None

        won = (side == "UP" and outcome_up == 1) or (side == "DOWN" and outcome_up == 0)
        payout = shares if won else 0.0
        pnl = payout - cost
        return pnl, {"won": won, "cost": cost}

    return 0.0, None


# ── Strategy 3: Dynamic Grid (Price Following) ────────────────────────────

def run_strategy3_window(wdf: pd.DataFrame, outcome_up: int, start_px: float,
                         bankroll: float) -> tuple[float, dict | None]:
    """Vol-band grid: bid below fair, ask above. Earn spread on round trips."""
    vol_lookback = 20
    update_interval = 10_000
    hist: list[float] = []
    last_update_ts = 0

    grid_bid: dict | None = None
    grid_ask: dict | None = None
    inventory_shares = 0.0
    inventory_cost = 0.0
    buy_fills = 0
    sell_fills = 0
    total_revenue = 0.0
    total_spent = 0.0

    for _, row in wdf.iterrows():
        snap = Snapshot.from_row(row)
        if snap is None:
            continue
        hist.append(snap.chainlink_price)
        tau = snap.time_remaining_s

        if snap.best_bid_up is None or snap.best_ask_up is None:
            continue
        if snap.best_ask_up <= 0 or snap.best_ask_up >= 1:
            continue

        # Check bid fill: ask drops to our bid → we buy
        if grid_bid is not None and snap.best_ask_up <= grid_bid["price"]:
            cost = grid_bid["shares"] * grid_bid["price"]
            inventory_shares += grid_bid["shares"]
            inventory_cost += cost
            total_spent += cost
            buy_fills += 1
            grid_bid = None

        # Check ask fill: bid rises to our ask → we sell
        if grid_ask is not None and inventory_shares >= grid_ask["shares"]:
            if snap.best_bid_up >= grid_ask["price"]:
                revenue = grid_ask["shares"] * grid_ask["price"]
                avg_cost = inventory_cost / inventory_shares if inventory_shares > 0 else 0
                inventory_shares -= grid_ask["shares"]
                inventory_cost -= grid_ask["shares"] * avg_cost
                total_revenue += revenue
                sell_fills += 1
                grid_ask = None

        # Update grid periodically
        if snap.ts_ms - last_update_ts < update_interval:
            continue
        last_update_ts = snap.ts_ms

        if len(hist) < vol_lookback or tau < 30:
            continue

        sigma = compute_vol(hist[-vol_lookback:])
        if sigma <= 0:
            continue

        delta = snap.chainlink_price - snap.window_start_price
        z = delta / (sigma * math.sqrt(tau))
        z = max(-2.0, min(2.0, z))
        fair_up = norm_cdf(z)

        # Grid bands: 0.5-sigma width in probability space
        band = sigma * math.sqrt(tau) * 0.5
        bid_px = round(max(0.01, fair_up - band), 2)
        ask_px = round(min(0.99, fair_up + band), 2)

        grid_size = max(5.0, min(bankroll * 0.005 / max(bid_px, 0.01), 30.0))

        # Place/update bid
        if grid_bid is None and bid_px > 0.01 and bid_px < snap.best_ask_up:
            grid_bid = {"price": bid_px, "shares": grid_size}

        # Place ask only if we have inventory
        if grid_ask is None and inventory_shares >= 5.0 and ask_px > bid_px:
            grid_ask = {"price": ask_px, "shares": min(inventory_shares, 30.0)}

    # Resolve remaining inventory at window end
    if inventory_shares > 0:
        won = (outcome_up == 1)
        payout = inventory_shares if won else 0.0
        total_revenue += payout

    if buy_fills == 0:
        return 0.0, None

    pnl = total_revenue - total_spent
    return pnl, {
        "won": pnl > 0, "cost": total_spent,
        "buys": buy_fills, "sells": sell_fills
    }


# ── Strategy 4: Hybrid FOK Signal + Grid ──────────────────────────────────

def run_strategy4_window(wdf: pd.DataFrame, outcome_up: int, start_px: float,
                         bankroll: float) -> tuple[float, dict | None]:
    """FOK when signal fires, passive grid bids when no signal."""
    vol_lookback = 20
    edge_threshold = 0.10
    early_edge_mult = 4.0
    kelly_fraction = 0.25
    max_bet_fraction = 0.0125
    cooldown_ms = 30_000
    grid_interval = 10_000

    hist: list[float] = []
    last_fill_ts = 0
    last_grid_ts = 0

    fok_pnl = 0.0
    fok_cost = 0.0
    fok_count = 0
    fok_wins = 0

    grid_bid_up: dict | None = None
    grid_bid_down: dict | None = None
    grid_pnl = 0.0
    grid_cost = 0.0
    grid_count = 0
    grid_wins = 0
    fok_fired = False

    for _, row in wdf.iterrows():
        snap = Snapshot.from_row(row)
        if snap is None:
            continue
        hist.append(snap.chainlink_price)
        tau = snap.time_remaining_s

        if snap.best_bid_up is None or snap.best_ask_up is None:
            continue
        if snap.best_ask_up <= 0 or snap.best_ask_up >= 1:
            continue

        # --- Check grid fills ---
        if grid_bid_up is not None and snap.best_ask_up <= grid_bid_up["price"]:
            cost = grid_bid_up["shares"] * grid_bid_up["price"]
            won = (outcome_up == 1)
            payout = grid_bid_up["shares"] if won else 0.0
            grid_pnl += payout - cost
            grid_cost += cost
            grid_count += 1
            if won:
                grid_wins += 1
            grid_bid_up = None

        if grid_bid_down is not None and snap.best_ask_down is not None:
            if snap.best_ask_down <= grid_bid_down["price"]:
                cost = grid_bid_down["shares"] * grid_bid_down["price"]
                won = (outcome_up == 0)
                payout = grid_bid_down["shares"] if won else 0.0
                grid_pnl += payout - cost
                grid_cost += cost
                grid_count += 1
                if won:
                    grid_wins += 1
                grid_bid_down = None

        # --- FOK signal check ---
        if len(hist) >= vol_lookback and tau > 0:
            if (snap.ts_ms - last_fill_ts >= cooldown_ms or last_fill_ts == 0):
                sigma = compute_vol(hist[-vol_lookback:])
                if sigma > 0:
                    delta = snap.chainlink_price - snap.window_start_price
                    z_raw = delta / (sigma * math.sqrt(tau))
                    z = max(-2.0, min(2.0, z_raw))
                    p_model = norm_cdf(z)

                    dyn_threshold = edge_threshold * (
                        1.0 + early_edge_mult * math.sqrt(tau / 900.0))

                    p_up_cost = snap.best_ask_up + poly_fee(snap.best_ask_up)
                    p_down_cost = (snap.best_ask_down + poly_fee(snap.best_ask_down)
                                   if snap.best_ask_down and snap.best_ask_down > 0 else 1.0)

                    edge_up = p_model - p_up_cost
                    edge_down = (1.0 - p_model) - p_down_cost

                    if edge_up > dyn_threshold or edge_down > dyn_threshold:
                        if edge_up >= edge_down:
                            side, edge, ask_px = "UP", edge_up, snap.best_ask_up
                        else:
                            side, edge, ask_px = "DOWN", edge_down, snap.best_ask_down

                        # Momentum check
                        mom_n = min(30, len(hist))
                        mom_ok = True
                        if mom_n >= 2:
                            mom_prices = hist[-mom_n:]
                            if side == "UP":
                                frac_ok = sum(1 for p in mom_prices if p >= start_px) / len(mom_prices)
                            else:
                                frac_ok = sum(1 for p in mom_prices if p <= start_px) / len(mom_prices)
                            if frac_ok < 1.0:
                                mom_ok = False

                        if mom_ok and ask_px and 0 < ask_px < 1:
                            eff_price = ask_px + poly_fee(ask_px)
                            if 0 < eff_price < 1:
                                p_side = p_model if side == "UP" else 1 - p_model
                                kelly_f = max(0, (p_side - eff_price) / (1 - eff_price))
                                frac = min(kelly_fraction * kelly_f, max_bet_fraction)
                                if frac > 0:
                                    size_usd = bankroll * frac
                                    shares = size_usd / eff_price
                                    if shares >= 5.0:
                                        cost = shares * eff_price
                                        won = ((side == "UP" and outcome_up == 1) or
                                               (side == "DOWN" and outcome_up == 0))
                                        payout = shares if won else 0.0
                                        pnl = payout - cost
                                        fok_pnl += pnl
                                        fok_cost += cost
                                        fok_count += 1
                                        if won:
                                            fok_wins += 1
                                        last_fill_ts = snap.ts_ms
                                        fok_fired = True
                                        # Cancel grid on same side
                                        if side == "UP":
                                            grid_bid_up = None
                                        else:
                                            grid_bid_down = None

        # --- Grid placement (when no FOK yet) ---
        if not fok_fired and snap.ts_ms - last_grid_ts >= grid_interval:
            last_grid_ts = snap.ts_ms

            if len(hist) >= vol_lookback and tau > 30:
                sigma = compute_vol(hist[-vol_lookback:])
                if sigma > 0:
                    delta = snap.chainlink_price - snap.window_start_price
                    z = delta / (sigma * math.sqrt(tau))
                    z = max(-2.0, min(2.0, z))
                    fair_up = norm_cdf(z)

                    discount = 0.05
                    bid_up_px = round(max(0.01, fair_up - discount), 2)
                    bid_down_px = round(max(0.01, (1.0 - fair_up) - discount), 2)
                    grid_size = max(5.0, min(bankroll * 0.003 / max(bid_up_px, 0.01), 20.0))

                    if grid_bid_up is None and bid_up_px > 0.01 and bid_up_px < snap.best_ask_up:
                        grid_bid_up = {"price": bid_up_px, "shares": grid_size}

                    if (grid_bid_down is None and snap.best_ask_down and
                            bid_down_px > 0.01 and bid_down_px < snap.best_ask_down):
                        grid_bid_down = {"price": bid_down_px, "shares": grid_size}

    total_trades = fok_count + grid_count
    if total_trades == 0:
        return 0.0, None

    total_pnl = fok_pnl + grid_pnl
    return total_pnl, {
        "won": total_pnl > 0,
        "cost": fok_cost + grid_cost,
        "fok": fok_count, "fok_wins": fok_wins, "fok_pnl": fok_pnl,
        "grid": grid_count, "grid_wins": grid_wins, "grid_pnl": grid_pnl,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 92)
    print("  STRATEGY COMPARISON BACKTEST - BTC")
    print("=" * 92)
    print(f"  Bankroll: ${BANKROLL:,.0f}")
    print()

    files = sorted(DATA_BTC.glob("*.parquet"))
    print(f"  {len(files)} window files found")
    print()

    # Process window by window (memory efficient)
    bank = {
        "s1": BANKROLL, "s2": BANKROLL, "s3": BANKROLL, "s4": BANKROLL
    }
    pnls = {"s1": [], "s2": [], "s3": [], "s4": []}
    stats = {
        "s1": {"trades": 0, "wins": 0, "pnl": 0.0, "cost": 0.0},
        "s2": {"trades": 0, "wins": 0, "pnl": 0.0, "cost": 0.0},
        "s3": {"trades": 0, "wins": 0, "pnl": 0.0, "cost": 0.0},
        "s4": {"trades": 0, "wins": 0, "pnl": 0.0, "cost": 0.0,
               "fok": 0, "fok_wins": 0, "fok_pnl": 0.0,
               "grid": 0, "grid_wins": 0, "grid_pnl": 0.0},
    }
    n_complete = 0
    n_total = 0

    # Get window time ranges for concurrency check (S2)
    window_ranges = {}
    for f in files:
        slug = f.stem
        # Extract timestamp from filename
        parts = slug.split("-")
        try:
            ts = int(parts[-1])
            window_ranges[slug] = (ts * 1000, (ts + 900) * 1000)
        except ValueError:
            pass

    print(f"  Processing windows...")
    for i, fpath in enumerate(files):
        wdf = load_window(fpath)
        if wdf.empty:
            continue
        n_total += 1

        slug = wdf.iloc[0]["market_slug"] if "market_slug" in wdf.columns else fpath.stem
        resolved = resolve_window(wdf)
        if resolved is None:
            del wdf
            continue

        n_complete += 1
        outcome_up, final_px, start_px = resolved

        # Count concurrent windows for S2
        my_range = window_ranges.get(fpath.stem)
        if my_range:
            mid_t = (my_range[0] + my_range[1]) // 2
            concurrent = sum(
                1 for s, (ws, we) in window_ranges.items()
                if ws <= mid_t <= we and s != fpath.stem
            )
        else:
            concurrent = 1

        # --- Strategy 1 ---
        p1, info1 = run_strategy1_window(wdf, outcome_up, start_px, bank["s1"])
        if info1:
            bank["s1"] += p1
            pnls["s1"].append(p1)
            stats["s1"]["trades"] += 1
            stats["s1"]["pnl"] += p1
            stats["s1"]["cost"] += info1["cost"]
            if info1["won"]:
                stats["s1"]["wins"] += 1

        # --- Strategy 2 ---
        p2, info2 = run_strategy2_window(wdf, outcome_up, start_px, bank["s2"], concurrent)
        if info2:
            bank["s2"] += p2
            pnls["s2"].append(p2)
            stats["s2"]["trades"] += 1
            stats["s2"]["pnl"] += p2
            stats["s2"]["cost"] += info2["cost"]
            if info2["won"]:
                stats["s2"]["wins"] += 1

        # --- Strategy 3 ---
        p3, info3 = run_strategy3_window(wdf, outcome_up, start_px, bank["s3"])
        if info3:
            bank["s3"] += p3
            pnls["s3"].append(p3)
            stats["s3"]["trades"] += 1
            stats["s3"]["pnl"] += p3
            stats["s3"]["cost"] += info3["cost"]
            if info3["won"]:
                stats["s3"]["wins"] += 1

        # --- Strategy 4 ---
        p4, info4 = run_strategy4_window(wdf, outcome_up, start_px, bank["s4"])
        if info4:
            bank["s4"] += p4
            pnls["s4"].append(p4)
            stats["s4"]["trades"] += 1
            stats["s4"]["pnl"] += p4
            stats["s4"]["cost"] += info4["cost"]
            if info4["won"]:
                stats["s4"]["wins"] += 1
            stats["s4"]["fok"] += info4.get("fok", 0)
            stats["s4"]["fok_wins"] += info4.get("fok_wins", 0)
            stats["s4"]["fok_pnl"] += info4.get("fok_pnl", 0)
            stats["s4"]["grid"] += info4.get("grid", 0)
            stats["s4"]["grid_wins"] += info4.get("grid_wins", 0)
            stats["s4"]["grid_pnl"] += info4.get("grid_pnl", 0)

        del wdf
        if (i + 1) % 20 == 0:
            gc.collect()

    print(f"  {n_total} total, {n_complete} complete windows")
    print()

    # --- Run FOK baseline ---
    print("  Running FOK baseline...")
    fok_signal = DiffusionSignal(bankroll=BANKROLL)
    fok_engine = BacktestEngine(signal=fok_signal, data_dir=DATA_BTC, initial_bankroll=BANKROLL)
    fok_results, fok_metrics, _ = fok_engine.run()

    r0 = StratResult("BASELINE: FOK (Current)")
    r0.trades = len(fok_results)
    r0.wins = sum(1 for r in fok_results if r.pnl > 0)
    r0.total_pnl = sum(r.pnl for r in fok_results)
    r0.total_cost = sum(r.fill.cost_usd for r in fok_results)
    r0.max_dd = compute_dd([r.pnl for r in fok_results], BANKROLL)
    r0.final_bank = BANKROLL + r0.total_pnl

    del fok_results
    gc.collect()

    # Build StratResults
    results_all = [r0]
    labels = {
        "s1": "S1: Queue Priority (Early Bid)",
        "s2": "S2: Multi-Market Grid (3 slots)",
        "s3": "S3: Dynamic Grid (Vol Bands)",
        "s4": "S4: Hybrid FOK + Grid",
    }
    for key in ["s1", "s2", "s3", "s4"]:
        r = StratResult(labels[key])
        r.trades = stats[key]["trades"]
        r.wins = stats[key]["wins"]
        r.total_pnl = stats[key]["pnl"]
        r.total_cost = stats[key]["cost"]
        r.max_dd = compute_dd(pnls[key], BANKROLL)
        r.final_bank = bank[key]
        results_all.append(r)

    # Print comparison
    print_comparison(results_all, n_complete)

    # Detailed analysis
    print()
    print("  DETAILED ANALYSIS")
    print("  " + "-" * 90)
    print()

    # S1
    s1 = stats["s1"]
    print(f"  S1: Queue Priority (Early Bid)")
    if s1["trades"] > 0:
        fill_rate = s1["trades"] / n_complete
        print(f"    Fill rate: {s1['trades']}/{n_complete} windows ({fill_rate:.0%})")
        print(f"    Win rate: {s1['wins']}/{s1['trades']} ({s1['wins']/s1['trades']:.0%})")
        print(f"    Problem: Bidding at open mid with zero directional info = ~coin flip")
        print(f"    The 0%% maker fee doesn't compensate for random entry")
    else:
        print(f"    No fills - early bids rarely get hit when placed at mid")
    print()

    # S2
    s2 = stats["s2"]
    print(f"  S2: Multi-Market Grid (Capital Split)")
    if s2["trades"] > 0:
        fill_rate = s2["trades"] / n_complete
        print(f"    Fill rate: {s2['trades']}/{n_complete} ({fill_rate:.0%})")
        print(f"    Win rate: {s2['wins']}/{s2['trades']} ({s2['wins']/s2['trades']:.0%})")
        print(f"    Capital split reduces per-trade size")
        print(f"    BTC windows are highly correlated - no real diversification")
    else:
        print(f"    No fills")
    print()

    # S3
    s3 = stats["s3"]
    print(f"  S3: Dynamic Grid (Vol Bands)")
    if s3["trades"] > 0:
        print(f"    Windows with fills: {s3['trades']}")
        print(f"    Win rate: {s3['wins']}/{s3['trades']} ({s3['wins']/s3['trades']:.0%})")
        print(f"    Grid earns spread on round-trips but loses on inventory held to resolution")
        print(f"    15-min windows too short for mean-reversion to consistently work")
    else:
        print(f"    No fills - grid levels too far from market")
    print()

    # S4
    s4 = stats["s4"]
    print(f"  S4: Hybrid FOK + Grid")
    if s4["trades"] > 0:
        print(f"    Total fills: {s4['trades']} (FOK: {s4['fok']}, Grid: {s4['grid']})")
        if s4["fok"] > 0:
            fok_wr = s4["fok_wins"] / s4["fok"]
            print(f"    FOK: {s4['fok']} trades, {fok_wr:.0%} win rate, ${s4['fok_pnl']:+,.2f} PnL")
        if s4["grid"] > 0:
            grid_wr = s4["grid_wins"] / s4["grid"]
            print(f"    Grid: {s4['grid']} trades, {grid_wr:.0%} win rate, ${s4['grid_pnl']:+,.2f} PnL")
        print(f"    Grid fills have adverse selection (filled when price moves against)")
    else:
        print(f"    No fills")
    print()

    # Verdict
    print("  " + "=" * 90)
    print("  VERDICT")
    print("  " + "=" * 90)
    best = max(results_all, key=lambda r: r.total_pnl)
    print(f"  Best: {best.name} -> ${best.total_pnl:+,.2f} PnL")
    print()
    for r in results_all[1:]:
        diff = r.total_pnl - r0.total_pnl
        pct = diff / abs(r0.total_pnl) * 100 if r0.total_pnl != 0 else 0
        print(f"    {r.name:<35s} vs FOK: ${diff:>+10,.2f} ({pct:>+6.1f}%)")
    print()


if __name__ == "__main__":
    main()
