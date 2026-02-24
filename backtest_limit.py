#!/usr/bin/env python3
"""
Limit Order (Maker) Strategy Backtest

Simulates placing GTC limit orders instead of FOK market orders.
Compares maker strategy (0% fee, better prices, adverse selection risk)
vs current FOK taker strategy (~1.5% fee, guaranteed fills).

Usage:
    py -3 backtest_limit.py --market btc
    py -3 backtest_limit.py --market btc --aggression 0.7
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import (
    Snapshot, Fill, TradeResult, DiffusionSignal, BacktestEngine,
    poly_fee, norm_cdf, DATA_DIR,
)
from market_config import MARKET_CONFIGS, DEFAULT_MARKET, get_config


# ── Limit order dataclasses ─────────────────────────────────────────────────

@dataclass
class LimitOrder:
    """A simulated limit order resting on the book."""
    side: str         # "BUY_UP" or "BUY_DOWN"
    price: float      # our bid price (maker, no fee)
    shares: float
    placed_ts_ms: int
    placed_tau: float
    edge: float
    reason: str


@dataclass
class LimitFillResult:
    """Result of a filled limit order after window resolution."""
    market_slug: str
    side: str         # "UP" or "DOWN"
    entry_price: float
    shares: float
    cost_usd: float
    fill_tau: float
    outcome_up: int
    won: bool
    payout: float
    pnl: float
    pnl_pct: float
    orders_placed: int
    orders_cancelled: int
    reason: str


# ── Limit order strategy ────────────────────────────────────────────────────

class LimitMakerBacktest:
    """Simulates a limit-order maker strategy on recorded data.

    Instead of buying at the ask (taker), we post bids inside the spread
    and get filled when takers sell into us.  Benefits:
      - 0% maker fee (vs ~1.5% taker fee)
      - Better entry prices (bid < ask)
    Risks:
      - Not always filled (opportunity cost)
      - Adverse selection (filled when price moves against us)
    """

    def __init__(
        self,
        bankroll: float = 10_000.0,
        # Model params (same as DiffusionSignal)
        vol_lookback_s: int = 20,
        min_sigma: float = 1e-6,
        edge_threshold: float = 0.10,
        early_edge_mult: float = 4.0,
        window_duration: float = 900.0,
        max_z: float = 2.0,
        # Sizing
        kelly_fraction: float = 0.25,
        max_bet_fraction: float = 0.0125,
        min_order_shares: float = 5.0,
        # Limit-order specific
        update_interval_s: float = 5.0,    # seconds between order updates
        aggression: float = 0.3,           # 0=conservative, 1=aggressive
        max_orders_per_window: int = 100,
        # Filters (same as DiffusionSignal)
        momentum_lookback_s: int = 30,
        momentum_majority: float = 1.0,
        reversion_discount: float = 0.0,
    ):
        self.bankroll = bankroll
        self.initial_bankroll = bankroll
        self.vol_lookback_s = vol_lookback_s
        self.min_sigma = min_sigma
        self.edge_threshold = edge_threshold
        self.early_edge_mult = early_edge_mult
        self.window_duration = window_duration
        self.max_z = max_z
        self.kelly_fraction = kelly_fraction
        self.max_bet_fraction = max_bet_fraction
        self.min_order_shares = min_order_shares
        self.update_interval_s = update_interval_s
        self.aggression = aggression
        self.max_orders_per_window = max_orders_per_window
        self.momentum_lookback_s = momentum_lookback_s
        self.momentum_majority = momentum_majority
        self.reversion_discount = reversion_discount

    def _compute_vol(self, prices: list[float]) -> float:
        """Realized vol from price series, ignoring stale ticks."""
        if len(prices) < 2:
            return 0.0
        unique = [prices[0]]
        for p in prices[1:]:
            if p != unique[-1]:
                unique.append(p)
        if len(unique) < 2:
            return 0.0
        log_rets = [math.log(unique[i] / unique[i - 1])
                    for i in range(1, len(unique))]
        if len(log_rets) < 2:
            return 0.0
        return max(float(np.std(log_rets, ddof=1)), self.min_sigma)

    def _compute_model(self, snap: Snapshot, hist: list[float]):
        """Compute GBM model probability. Returns (p_model, sigma, z) or None."""
        tau = snap.time_remaining_s
        if tau <= 0 or len(hist) < self.vol_lookback_s:
            return None

        sigma = self._compute_vol(hist[-self.vol_lookback_s:])
        if sigma < self.min_sigma:
            return None

        delta = snap.chainlink_price - snap.window_start_price
        z_raw = delta / (sigma * math.sqrt(tau))
        z = max(-self.max_z, min(self.max_z, z_raw))
        p_model = norm_cdf(z)

        if self.reversion_discount > 0:
            p_model = p_model * (1 - self.reversion_discount) + 0.5 * self.reversion_discount

        return p_model, sigma, z

    def _pick_side(self, p_model: float, snap: Snapshot, hist: list[float]):
        """Determine favored side with momentum filter. Returns side or None."""
        # Need clear directional signal
        if 0.45 < p_model < 0.55:
            return None

        side = "BUY_UP" if p_model > 0.5 else "BUY_DOWN"

        # Momentum filter
        start_px = snap.window_start_price
        mom_n = min(self.momentum_lookback_s, len(hist))
        if mom_n >= 2:
            mom_prices = hist[-mom_n:]
            if side == "BUY_UP":
                frac_ok = sum(1 for p in mom_prices if p >= start_px) / len(mom_prices)
            else:
                frac_ok = sum(1 for p in mom_prices if p <= start_px) / len(mom_prices)
            if frac_ok < self.momentum_majority:
                return None

        return side

    def _compute_bid(self, p_model: float, side: str, snap: Snapshot, tau: float):
        """Compute optimal maker bid price. Returns (bid_price, edge) or (None, 0)."""
        dyn_threshold = self.edge_threshold * (
            1.0 + self.early_edge_mult * math.sqrt(tau / self.window_duration)
        )

        if side == "BUY_UP":
            fair_value = p_model
            best_ask = snap.best_ask_up
            best_bid = snap.best_bid_up
        else:
            fair_value = 1.0 - p_model
            best_ask = snap.best_ask_down
            best_bid = snap.best_bid_down

        if not best_ask or not best_bid or best_ask <= 0 or best_ask >= 1:
            return None, 0

        spread = best_ask - best_bid
        if spread > 0.05:  # skip if spread is too wide
            return None, 0

        # Conservative bid: fair value minus full threshold (max edge)
        conservative_bid = fair_value - dyn_threshold

        # Aggressive bid: just below the ask (capture spread)
        aggressive_bid = best_ask - 0.01

        # Interpolate based on aggression
        bid_price = conservative_bid + self.aggression * (aggressive_bid - conservative_bid)
        bid_price = round(bid_price, 2)

        # Clamp
        if bid_price <= 0.01:
            bid_price = 0.01
        if bid_price >= best_ask:
            bid_price = round(best_ask - 0.01, 2)
        if bid_price <= 0:
            return None, 0

        # Edge at this bid (maker = 0% fee)
        edge = fair_value - bid_price
        if edge < dyn_threshold * 0.3:  # at least 30% of threshold
            return None, 0

        return bid_price, edge

    def _compute_size(self, edge: float, bid_price: float):
        """Half-kelly sizing. Returns shares."""
        if bid_price <= 0 or bid_price >= 1:
            return 0
        payout_mult = 1.0 / bid_price - 1.0
        if payout_mult <= 0:
            return 0
        kelly = edge / payout_mult
        if kelly <= 0:
            return 0
        frac = min(self.kelly_fraction * kelly, self.max_bet_fraction)
        size_usd = self.bankroll * frac
        min_usd = self.min_order_shares * bid_price
        if size_usd < min_usd:
            if self.bankroll >= min_usd:
                size_usd = min_usd
            else:
                return 0
        shares = size_usd / bid_price
        return max(self.min_order_shares, round(shares, 1))

    def run_window(self, window_df: pd.DataFrame, market_slug: str,
                   outcome_up: int, final_price: float):
        """Simulate limit order strategy for one window."""
        hist: list[float] = []
        pending: LimitOrder | None = None
        fill_result: LimitFillResult | None = None
        last_update_ts = 0
        order_count = 0
        cancel_count = 0

        for _, row in window_df.iterrows():
            snap = Snapshot.from_row(row)
            if snap is None:
                continue

            # Build price history
            hist.append(snap.chainlink_price)
            tau = snap.time_remaining_s

            # Already filled — hold to resolution
            if fill_result is not None:
                continue

            # Check if pending order is filled
            if pending is not None:
                filled = False
                if pending.side == "BUY_UP" and snap.best_ask_up is not None:
                    # Filled if ask drops to or below our bid
                    if snap.best_ask_up <= pending.price:
                        filled = True
                elif pending.side == "BUY_DOWN" and snap.best_ask_down is not None:
                    if snap.best_ask_down <= pending.price:
                        filled = True

                if filled:
                    side_label = "UP" if pending.side == "BUY_UP" else "DOWN"
                    cost = pending.shares * pending.price
                    won = ((side_label == "UP" and outcome_up == 1) or
                           (side_label == "DOWN" and outcome_up == 0))
                    payout = pending.shares if won else 0.0
                    pnl = payout - cost

                    fill_result = LimitFillResult(
                        market_slug=market_slug,
                        side=side_label,
                        entry_price=pending.price,
                        shares=pending.shares,
                        cost_usd=cost,
                        fill_tau=tau,
                        outcome_up=outcome_up,
                        won=won,
                        payout=payout,
                        pnl=pnl,
                        pnl_pct=pnl / cost if cost > 0 else 0,
                        orders_placed=order_count,
                        orders_cancelled=cancel_count,
                        reason=pending.reason,
                    )
                    self.bankroll += pnl
                    continue

            # Should we update/place an order?
            time_since_update = (snap.ts_ms - last_update_ts) / 1000.0
            if time_since_update < self.update_interval_s:
                continue

            if order_count >= self.max_orders_per_window:
                continue

            # Compute model
            model = self._compute_model(snap, hist)
            if model is None:
                continue
            p_model, sigma, z = model

            # Pick side
            side = self._pick_side(p_model, snap, hist)
            if side is None:
                if pending is not None:
                    pending = None
                    cancel_count += 1
                continue

            # Compute bid
            bid_price, edge = self._compute_bid(p_model, side, snap, tau)
            if bid_price is None:
                if pending is not None:
                    pending = None
                    cancel_count += 1
                continue

            # Compute size
            shares = self._compute_size(edge, bid_price)
            if shares < self.min_order_shares:
                if pending is not None:
                    pending = None
                    cancel_count += 1
                continue

            # Check if order changed enough to update
            if pending is not None:
                if (pending.side == side and
                        abs(pending.price - bid_price) < 0.01):
                    # No significant change, keep current order
                    last_update_ts = snap.ts_ms
                    continue
                else:
                    cancel_count += 1  # cancel old order

            # Place/update order
            reason = (f"p={p_model:.4f} sig={sigma:.2e} z={z:.2f} "
                      f"tau={tau:.0f}s bid={bid_price:.2f} edge={edge:.4f}")
            pending = LimitOrder(
                side=side,
                price=bid_price,
                shares=shares,
                placed_ts_ms=snap.ts_ms,
                placed_tau=tau,
                edge=edge,
                reason=reason,
            )
            order_count += 1
            last_update_ts = snap.ts_ms

        return fill_result, order_count, cancel_count


# ── Main backtest runner ────────────────────────────────────────────────────

def load_data(data_dir: Path) -> pd.DataFrame:
    """Load and normalize parquet data."""
    if not data_dir.exists() or not any(data_dir.glob("*.parquet")):
        return pd.DataFrame()
    frames = []
    for f in sorted(data_dir.glob("*.parquet")):
        part = pd.read_parquet(f)
        if "chainlink_btc" in part.columns and "chainlink_price" not in part.columns:
            part.rename(columns={"chainlink_btc": "chainlink_price"}, inplace=True)
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    return df


def run_comparison(market: str, aggression: float, bankroll: float):
    """Run both limit and FOK strategies and compare."""
    config = get_config(market)
    data_dir = DATA_DIR / config.data_subdir
    df = load_data(data_dir)
    if df.empty:
        print(f"No data for {market}")
        return

    # Group by window
    windows = sorted(df["market_slug"].unique())
    print(f"  {config.display_name} — {len(windows)} windows loaded")
    print(f"  Bankroll: ${bankroll:,.0f}  |  Aggression: {aggression}")
    print()

    # --- Run FOK (taker) strategy ---
    eth_overrides = {}
    eth_engine_kw = {}
    if market == "eth":
        eth_overrides = dict(
            edge_threshold=0.15,
            reversion_discount=0.15,
            momentum_lookback_s=15,
            momentum_majority=0.7,
            spread_edge_penalty=0.2,
        )
        eth_engine_kw = dict(max_trades_per_window=1)

    fok_signal = DiffusionSignal(bankroll=bankroll, **eth_overrides)
    fok_engine = BacktestEngine(
        signal=fok_signal,
        data_dir=data_dir,
        initial_bankroll=bankroll,
        **eth_engine_kw,
    )
    fok_results, fok_metrics, _ = fok_engine.run()

    # --- Run Limit (maker) strategy ---
    limit_strat = LimitMakerBacktest(
        bankroll=bankroll,
        aggression=aggression,
        edge_threshold=eth_overrides.get("edge_threshold", 0.10),
        reversion_discount=eth_overrides.get("reversion_discount", 0.0),
        momentum_lookback_s=eth_overrides.get("momentum_lookback_s", 30),
        momentum_majority=eth_overrides.get("momentum_majority", 1.0),
    )

    limit_fills: list[LimitFillResult] = []
    limit_total_orders = 0
    limit_total_cancels = 0
    windows_no_fill = 0

    for slug in windows:
        wdf = df[df["market_slug"] == slug].sort_values("ts_ms")
        if wdf.empty:
            continue

        # Check completeness
        final_remaining = wdf.iloc[-1]["time_remaining_s"]
        if final_remaining > 5.0:
            continue

        # Determine outcome
        last = wdf.iloc[-1]
        chainlink_col = "chainlink_price"
        if chainlink_col not in last:
            continue
        final_px = float(last[chainlink_col])
        start_px = float(last["window_start_price"])
        outcome_up = 1 if final_px >= start_px else 0

        fill, n_orders, n_cancels = limit_strat.run_window(
            wdf, slug, outcome_up, final_px)
        limit_total_orders += n_orders
        limit_total_cancels += n_cancels

        if fill is not None:
            limit_fills.append(fill)
        else:
            windows_no_fill += 1

    # --- Print comparison ---
    complete_windows = len(windows) - sum(
        1 for s in windows
        if df[df["market_slug"] == s].iloc[-1]["time_remaining_s"] > 5.0
    )

    print("=" * 76)
    print(f"  {'METRIC':<35s} {'FOK (Taker)':>18s} {'Limit (Maker)':>18s}")
    print("=" * 76)

    fok_n = len(fok_results)
    lim_n = len(limit_fills)

    fok_wins = sum(1 for r in fok_results if r.pnl > 0)
    lim_wins = sum(1 for r in limit_fills if r.won)

    fok_pnl = sum(r.pnl for r in fok_results)
    lim_pnl = sum(r.pnl for r in limit_fills)

    fok_wr = fok_wins / fok_n if fok_n > 0 else 0
    lim_wr = lim_wins / lim_n if lim_n > 0 else 0

    # Entry prices
    fok_avg_entry = (np.mean([r.fill.entry_price for r in fok_results])
                     if fok_results else 0)
    lim_avg_entry = (np.mean([r.entry_price for r in limit_fills])
                     if limit_fills else 0)

    # Fees saved
    fok_fees = sum(poly_fee(r.fill.entry_price) * r.fill.shares
                   for r in fok_results)

    # Drawdown
    def max_dd(pnls, start):
        eq = start
        peak = start
        dd = 0
        for p in pnls:
            eq += p
            if eq > peak:
                peak = eq
            if peak - eq > dd:
                dd = peak - eq
        return dd

    fok_dd = max_dd([r.pnl for r in fok_results], bankroll)
    lim_dd = max_dd([r.pnl for r in limit_fills], bankroll)

    def fmt(val, fmt_str):
        return format(val, fmt_str) if val else "---"

    fok_fr = f"{fok_n}/{complete_windows} ({fok_n/complete_windows:.0%})" if complete_windows else "---"
    lim_fr = f"{lim_n}/{complete_windows} ({lim_n/complete_windows:.0%})" if complete_windows else "---"
    fok_avg_pnl = f"${fok_pnl/fok_n:>+.2f}" if fok_n else "---"
    lim_avg_pnl = f"${lim_pnl/lim_n:>+.2f}" if lim_n else "---"

    print(f"  {'Complete windows':<35s} {complete_windows:>18d} {complete_windows:>18d}")
    print(f"  {'Trades':<35s} {fok_n:>18d} {lim_n:>18d}")
    print(f"  {'Fill rate':<35s} {fok_fr:>18s} {lim_fr:>18s}")
    print(f"  {'Win rate':<35s} {fok_wr:>17.0%} {lim_wr:>17.0%}")
    print(f"  {'Total PnL':<35s} ${fok_pnl:>+16,.2f} ${lim_pnl:>+16,.2f}")
    print(f"  {'Avg PnL / trade':<35s} {fok_avg_pnl:>18s} {lim_avg_pnl:>18s}")
    print(f"  {'Avg entry price':<35s} {fok_avg_entry:>18.4f} {lim_avg_entry:>18.4f}")
    print(f"  {'Taker fees paid':<35s} ${fok_fees:>16.2f} {'$0.00':>18s}")
    print(f"  {'Max drawdown':<35s} ${fok_dd:>16.2f} ${lim_dd:>16.2f}")
    print(f"  {'Orders placed':<35s} {'N/A':>18s} {limit_total_orders:>18d}")
    print(f"  {'Orders cancelled':<35s} {'N/A':>18s} {limit_total_cancels:>18d}")

    print("=" * 76)
    print()

    # Per-trade detail for limit
    if limit_fills:
        print(f"  Limit order fills ({lim_n} trades):")
        print(f"  {'Window':<35s} {'Side':>5s} {'Entry':>7s} {'Shares':>7s} "
              f"{'Cost':>8s} {'PnL':>9s} {'Time':>6s} {'Orders':>7s}")
        print(f"  {'-'*35} {'-'*5} {'-'*7} {'-'*7} {'-'*8} {'-'*9} {'-'*6} {'-'*7}")
        for f in limit_fills:
            tag = "W" if f.won else "L"
            m_left = int(f.fill_tau) // 60
            s_left = int(f.fill_tau) % 60
            print(f"  {f.market_slug:<35s} {f.side:>5s} "
                  f"{f.entry_price:>7.4f} {f.shares:>7.1f} "
                  f"${f.cost_usd:>7.2f} ${f.pnl:>+8.2f}{tag} "
                  f"{m_left}:{s_left:02d} {f.orders_placed:>7d}")
    else:
        print("  No limit order fills.")

    print()

    # Fee advantage analysis
    if fok_results and limit_fills:
        print("  Fee advantage analysis:")
        print(f"    FOK taker fees paid:     ${fok_fees:>10.2f} "
              f"(avg ${fok_fees/fok_n:.2f}/trade)")
        print(f"    Limit maker fees paid:   ${'0.00':>10s}")
        # Average entry price comparison for windows where both traded
        fok_slugs = {r.fill.market_slug: r for r in fok_results}
        lim_slugs = {r.market_slug: r for r in limit_fills}
        common = set(fok_slugs) & set(lim_slugs)
        if common:
            fok_entries = [fok_slugs[s].fill.entry_price for s in common]
            lim_entries = [lim_slugs[s].entry_price for s in common]
            avg_improvement = np.mean(
                [f - l for f, l in zip(fok_entries, lim_entries)])
            print(f"    Avg entry improvement:   {avg_improvement:>+10.4f} "
                  f"({len(common)} common windows)")
            print(f"    (Limit gets better prices by ~{avg_improvement:.1%} per share)")

    return fok_results, limit_fills


def main():
    parser = argparse.ArgumentParser(
        description="Limit order (maker) vs FOK (taker) backtest comparison"
    )
    parser.add_argument("--market", default=DEFAULT_MARKET,
                        choices=list(MARKET_CONFIGS))
    parser.add_argument("--bankroll", type=float, default=10_000.0)
    parser.add_argument("--aggression", type=float, default=0.3,
                        help="Bid aggression 0-1 (0=conservative, 1=aggressive)")
    args = parser.parse_args()

    print()
    print("=" * 76)
    print(f"  LIMIT ORDER vs FOK BACKTEST — {args.market.upper()}")
    print("=" * 76)
    print()

    run_comparison(args.market, args.aggression, args.bankroll)

    # Also run at different aggression levels
    print()
    print("=" * 76)
    print("  AGGRESSION SWEEP")
    print("=" * 76)
    print()
    print(f"  {'Aggression':>12s} {'Fills':>7s} {'WinR':>7s} {'PnL':>12s} "
          f"{'$/trade':>10s} {'AvgEntry':>10s} {'Orders':>8s}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*12} {'-'*10} {'-'*10} {'-'*8}")

    config = get_config(args.market)
    data_dir = DATA_DIR / config.data_subdir
    df = load_data(data_dir)

    eth_overrides = {}
    if args.market == "eth":
        eth_overrides = dict(
            edge_threshold=0.15,
            reversion_discount=0.15,
            momentum_lookback_s=15,
            momentum_majority=0.7,
        )

    for agg in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        strat = LimitMakerBacktest(
            bankroll=args.bankroll,
            aggression=agg,
            edge_threshold=eth_overrides.get("edge_threshold", 0.10),
            reversion_discount=eth_overrides.get("reversion_discount", 0.0),
            momentum_lookback_s=eth_overrides.get("momentum_lookback_s", 30),
            momentum_majority=eth_overrides.get("momentum_majority", 1.0),
        )

        fills = []
        total_orders = 0
        windows = sorted(df["market_slug"].unique())

        for slug in windows:
            wdf = df[df["market_slug"] == slug].sort_values("ts_ms")
            if wdf.empty:
                continue
            final_remaining = wdf.iloc[-1]["time_remaining_s"]
            if final_remaining > 5.0:
                continue

            last = wdf.iloc[-1]
            final_px = float(last["chainlink_price"])
            start_px = float(last["window_start_price"])
            outcome_up = 1 if final_px >= start_px else 0

            fill, n_orders, _ = strat.run_window(wdf, slug, outcome_up, final_px)
            total_orders += n_orders
            if fill is not None:
                fills.append(fill)

        n = len(fills)
        if n > 0:
            wins = sum(1 for f in fills if f.won)
            pnl = sum(f.pnl for f in fills)
            avg_entry = np.mean([f.entry_price for f in fills])
            flag = " <--" if pnl > 0 else ""
            print(f"  {agg:>12.1f} {n:>7d} {wins/n:>6.0%} ${pnl:>+11.2f} "
                  f"${pnl/n:>+9.2f} {avg_entry:>10.4f} {total_orders:>8d}{flag}")
        else:
            print(f"  {agg:>12.1f} {'---':>7s} {'---':>7s} {'---':>12s} "
                  f"{'---':>10s} {'---':>10s} {total_orders:>8d}")


if __name__ == "__main__":
    main()
