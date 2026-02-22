#!/usr/bin/env python3
"""
Parameter sweep: find optimal early_edge_mult and max_trades_per_window
for a $40 bankroll on BTC 15-min Up/Down markets.
"""

import sys
from pathlib import Path
from itertools import product

from backtest import DiffusionSignal, BacktestEngine

DATA_DIR = Path("data/btc")
BANKROLL = 40.0

# Parameter grid
EARLY_EDGE_MULTS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
MAX_TRADES_PER_WINDOW = [1, 2, 3]

def run_sweep(early_mult, max_trades):
    signal = DiffusionSignal(
        bankroll=BANKROLL,
        early_edge_mult=early_mult,
        min_order_usd=1.0,
    )
    engine = BacktestEngine(
        signal=signal,
        data_dir=DATA_DIR,
        initial_bankroll=BANKROLL,
    )
    # Patch cooldown to control max trades per window
    # BacktestEngine uses 30s cooldown internally — override via engine attribute
    # Actually, the engine hardcodes cooldown_ms=30_000 in _run_window.
    # To limit trades per window, we need to modify the engine or signal.
    # For now, just run and count — the cooldown already limits to ~30 trades max.
    results, metrics, _ = engine.run()

    if not results:
        return None

    # Compute per-window trade counts
    window_trades = {}
    for r in results:
        slug = r.fill.market_slug
        window_trades[slug] = window_trades.get(slug, 0) + 1

    # Filter results to respect max_trades_per_window
    kept = []
    window_count = {}
    # Re-run with trade limit by filtering results in order
    # This is approximate — proper way would be to modify the engine
    # But since cooldown already limits, most windows have 1-3 trades anyway
    for r in results:
        slug = r.fill.market_slug
        window_count[slug] = window_count.get(slug, 0) + 1
        if window_count[slug] <= max_trades:
            kept.append(r)

    if not kept:
        return None

    wins = [r for r in kept if r.pnl > 0]
    total_pnl = sum(r.pnl for r in kept)
    n = len(kept)
    win_rate = len(wins) / n if n > 0 else 0

    # Compute max drawdown
    equity = BANKROLL
    peak = BANKROLL
    max_dd = 0
    for r in kept:
        equity += r.pnl
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    final = equity
    max_dd_pct = max_dd / peak if peak > 0 else 0

    # Avg trades per window
    n_windows = len(set(r.fill.market_slug for r in kept))
    avg_trades = n / n_windows if n_windows > 0 else 0

    # Trades per window distribution
    wt_counts = {}
    for r in kept:
        slug = r.fill.market_slug
        wt_counts[slug] = wt_counts.get(slug, 0) + 1
    windows_with_trades = len(wt_counts)
    total_windows = metrics.get("_total_windows", n_windows)

    return {
        "early_mult": early_mult,
        "max_trades": max_trades,
        "n_trades": n,
        "n_windows_traded": windows_with_trades,
        "avg_trades_per_window": round(avg_trades, 1),
        "win_rate": round(win_rate, 3),
        "total_pnl": round(total_pnl, 2),
        "final_bankroll": round(final, 2),
        "max_dd": round(max_dd, 2),
        "max_dd_pct": round(max_dd_pct, 3),
        "pnl_per_trade": round(total_pnl / n, 2) if n > 0 else 0,
    }


def main():
    print(f"Sweeping parameters on {len(list(DATA_DIR.glob('*.parquet')))} BTC windows")
    print(f"Bankroll: ${BANKROLL:.0f}")
    print()

    results = []
    for early_mult, max_trades in product(EARLY_EDGE_MULTS, MAX_TRADES_PER_WINDOW):
        r = run_sweep(early_mult, max_trades)
        if r:
            results.append(r)
            flag = ""
            if r["total_pnl"] > 0:
                flag = " <--"
            print(
                f"  mult={early_mult:.1f}  max_t={max_trades}  "
                f"trades={r['n_trades']:3d}  "
                f"win={r['win_rate']:.0%}  "
                f"PnL=${r['total_pnl']:+7.2f}  "
                f"DD=${r['max_dd']:5.2f} ({r['max_dd_pct']:.0%})  "
                f"$/trade={r['pnl_per_trade']:+.2f}"
                f"{flag}"
            )
        else:
            print(f"  mult={early_mult:.1f}  max_t={max_trades}  -- no trades --")

    print()
    print("=" * 70)

    # Sort by PnL
    profitable = [r for r in results if r["total_pnl"] > 0]
    if profitable:
        # Rank by PnL / max_dd ratio (risk-adjusted)
        for r in profitable:
            r["risk_adj"] = r["total_pnl"] / r["max_dd"] if r["max_dd"] > 0 else 999
        profitable.sort(key=lambda x: -x["risk_adj"])

        print("Top 5 by risk-adjusted PnL (PnL / MaxDD):")
        for i, r in enumerate(profitable[:5], 1):
            print(
                f"  #{i}  mult={r['early_mult']:.1f}  max_t={r['max_trades']}  "
                f"trades={r['n_trades']}  win={r['win_rate']:.0%}  "
                f"PnL=${r['total_pnl']:+.2f}  DD=${r['max_dd']:.2f}  "
                f"ratio={r['risk_adj']:.2f}"
            )
    else:
        print("No profitable configurations found.")

    # Also show by raw PnL
    if results:
        results.sort(key=lambda x: -x["total_pnl"])
        print()
        print("Top 5 by raw PnL:")
        for i, r in enumerate(results[:5], 1):
            print(
                f"  #{i}  mult={r['early_mult']:.1f}  max_t={r['max_trades']}  "
                f"trades={r['n_trades']}  win={r['win_rate']:.0%}  "
                f"PnL=${r['total_pnl']:+.2f}  DD=${r['max_dd']:.2f}  "
                f"final=${r['final_bankroll']:.2f}"
            )


if __name__ == "__main__":
    main()
