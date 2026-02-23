#!/usr/bin/env python3
"""Quick parameter sweep for BTC: early_edge_mult x trade-count/sizing."""

from backtest import DiffusionSignal, BacktestEngine
from pathlib import Path

DATA_DIR = Path("data/btc")
BANKROLL = 10_000.0

configs = [
    # (label, early_edge_mult, max_trades_per_window, kelly_fraction, max_bet_fraction)
    # --- 1 trade per window, 1x sizing (current defaults) ---
    ("mult=4 1trade 1x",   4.0, 1, 0.25,  0.0125),
    ("mult=2 1trade 1x",   2.0, 1, 0.25,  0.0125),
    ("mult=1 1trade 1x",   1.0, 1, 0.25,  0.0125),
    # --- 1-2 trades per window, 0.5x sizing ---
    ("mult=4 2trade 0.5x", 4.0, 2, 0.125, 0.00625),
    ("mult=2 2trade 0.5x", 2.0, 2, 0.125, 0.00625),
    ("mult=1 2trade 0.5x", 1.0, 2, 0.125, 0.00625),
    # --- unlimited trades, 0.25x sizing ---
    ("mult=4 multi 0.25x", 4.0, None, 0.0625, 0.003125),
    ("mult=2 multi 0.25x", 2.0, None, 0.0625, 0.003125),
    ("mult=1 multi 0.25x", 1.0, None, 0.0625, 0.003125),
]

print(f"BTC parameter sweep on {len(list(DATA_DIR.glob('*.parquet')))} windows")
print(f"Bankroll: ${BANKROLL:,.0f}\n")

header = (f"{'Config':<22s} {'Trades':>6s} {'WinR':>6s} {'PnL':>10s} "
          f"{'$/trade':>8s} {'MaxDD':>8s} {'DD%':>5s} {'Sharpe':>7s} {'Final':>12s}")
print(header)
print("-" * len(header))

results = []
for label, mult, max_t, kelly_f, max_bet_f in configs:
    signal = DiffusionSignal(
        bankroll=BANKROLL,
        early_edge_mult=mult,
        kelly_fraction=kelly_f,
        max_bet_fraction=max_bet_f,
        min_order_shares=5.0,
    )
    engine = BacktestEngine(
        signal=signal,
        data_dir=DATA_DIR,
        initial_bankroll=BANKROLL,
        max_trades_per_window=max_t,
    )
    trade_results, metrics, _ = engine.run()

    n = len(trade_results)
    if n == 0:
        print(f"{label:<22s} {'--':>6s} {'--':>6s} {'--':>10s} {'--':>8s} {'--':>8s} {'--':>5s} {'--':>7s} {'--':>12s}")
        continue

    wins = sum(1 for r in trade_results if r.pnl > 0)
    total_pnl = sum(r.pnl for r in trade_results)
    wr = wins / n

    # Max drawdown
    equity = BANKROLL
    peak = BANKROLL
    max_dd = 0
    for r in trade_results:
        equity += r.pnl
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    final = equity
    dd_pct = max_dd / peak if peak > 0 else 0

    # Sharpe (annualized from per-trade returns)
    import numpy as np
    returns = [r.pnl / r.fill.cost_usd for r in trade_results if r.fill.cost_usd > 0]
    sharpe = (np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(n)) if len(returns) > 1 and np.std(returns, ddof=1) > 0 else 0

    flag = " <--" if total_pnl > 0 else ""
    print(f"{label:<22s} {n:>6d} {wr:>5.0%} {total_pnl:>+10.2f} "
          f"{total_pnl/n:>+8.2f} {max_dd:>8.2f} {dd_pct:>4.1%} {sharpe:>7.2f} {final:>12,.2f}{flag}")

    results.append((label, n, wr, total_pnl, max_dd, dd_pct, sharpe, final))

print()
# Rank by risk-adjusted return (PnL / MaxDD)
profitable = [(l, n, wr, pnl, dd, ddp, sh, f) for l, n, wr, pnl, dd, ddp, sh, f in results if pnl > 0]
if profitable:
    profitable.sort(key=lambda x: -x[3] / x[4] if x[4] > 0 else -999)
    print("Top by risk-adjusted PnL (PnL/MaxDD):")
    for i, (l, n, wr, pnl, dd, ddp, sh, f) in enumerate(profitable[:5], 1):
        ratio = pnl / dd if dd > 0 else 999
        print(f"  #{i} {l:<22s} PnL=${pnl:+.2f}  DD=${dd:.2f}  ratio={ratio:.2f}  Sharpe={sh:.2f}")
