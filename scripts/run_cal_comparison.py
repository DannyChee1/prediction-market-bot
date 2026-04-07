"""
Calibration Table A/B Comparison
================================
Runs 6 backtest variants per market to determine whether the calibration
table helps, hurts, or is redundant with reversion_discount.

Variants:
  1. no_cal              — No calibration, reversion_discount=0.07 (current default)
  2. no_cal_no_rev       — No calibration, reversion_discount=0.00 (pure GBM)
  3. cal_500             — Calibration prior=500 + reversion_discount=0.07 (current live)
  4. cal_500_no_rev      — Calibration prior=500 + reversion_discount=0.00 (no double-count)
  5. cal_2000            — Weaker calibration prior=2000 + reversion_discount=0.07
  6. cal_2000_no_rev     — Weaker calibration prior=2000 + reversion_discount=0.00

Usage:
  python run_cal_comparison.py                  # BTC 15m (default)
  python run_cal_comparison.py --market btc_5m
  python run_cal_comparison.py --market eth
  python run_cal_comparison.py --all            # all markets
"""

import argparse
import sys
from pathlib import Path

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from backtest import (
    BacktestEngine,
    DiffusionSignal,
    build_calibration_table,
    print_summary,
    DATA_DIR,
    MARKET_CONFIGS,
    get_config,
)


VARIANTS = [
    # (label, use_cal, cal_prior_strength, reversion_discount)
    ("no_cal",           False, None,  0.07),
    ("no_cal_no_rev",    False, None,  0.00),
    ("cal_500",          True,  500,   0.07),
    ("cal_500_no_rev",   True,  500,   0.00),
    ("cal_2000",         True,  2000,  0.07),
    ("cal_2000_no_rev",  True,  2000,  0.00),
]


def run_comparison(market: str, maker: bool = True, bankroll: float = 10_000.0):
    config = get_config(market)
    data_dir = DATA_DIR / config.data_subdir

    if not data_dir.exists() or not list(data_dir.glob("*.parquet")):
        print(f"  [SKIP] No data in {data_dir}")
        return None

    # Build calibration table once (shared across cal variants)
    print(f"  Building calibration table from {data_dir} ...")
    cal_table = build_calibration_table(data_dir)
    n_cells = len(cal_table.table)
    n_obs = sum(cal_table.counts.values())
    print(f"  Calibration table: {n_cells} cells, {n_obs} observations\n")

    # Per-market overrides (match main backtest.py logic)
    base_market = market.replace("_5m", "")
    is_5m = "_5m" in market

    eth_overrides = {}
    eth_engine_kw = {}
    if base_market == "eth":
        eth_overrides = dict(
            edge_threshold=0.15,
            momentum_lookback_s=15,
            momentum_majority=0.7,
            spread_edge_penalty=0.2,
        )
        eth_engine_kw = dict(max_trades_per_window=1)

    maker_overrides = {}
    if maker:
        maker_overrides = dict(
            maker_mode=True,
            max_bet_fraction=0.02,
            edge_threshold=0.08,
            momentum_majority=0.0,
            spread_edge_penalty=0.0,
            window_duration=config.window_duration_s,
        )
        if "max_trades_per_window" not in eth_engine_kw:
            eth_engine_kw["max_trades_per_window"] = 1

    vamp_kw = {}
    if base_market == "btc":
        vamp_kw = dict(vamp_mode="cost")
    elif base_market == "eth":
        vamp_kw = dict(vamp_mode="filter", vamp_filter_threshold=0.07)

    maker_warmup = 100.0
    maker_withdraw = 60.0
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
            # reversion_discount handled per-variant via effective_rev
            eth_engine_kw["max_trades_per_window"] = 2

    # Run each variant
    results = []
    for label, use_cal, cal_prior, rev_discount in VARIANTS:
        # 5m ETH has its own reversion_discount override
        effective_rev = rev_discount
        if is_5m and base_market == "eth" and rev_discount == 0.07:
            effective_rev = 0.10

        cal_overrides = {}
        if use_cal and maker:
            cal_overrides = dict(edge_threshold=0.04, early_edge_mult=0.4)

        signal = DiffusionSignal(
            bankroll=bankroll,
            slippage=0.0,
            calibration_table=cal_table if use_cal else None,
            cal_prior_strength=cal_prior if use_cal else 500.0,
            min_entry_price=0.10,
            inventory_skew=0.02,
            maker_warmup_s=maker_warmup,
            maker_withdraw_s=maker_withdraw,
            max_sigma=config.max_sigma,
            min_sigma=config.min_sigma,
            reversion_discount=effective_rev,
            **{**eth_overrides, **maker_overrides, **cal_overrides, **vamp_kw, **five_m_kw},
        )

        engine = BacktestEngine(
            signal=signal,
            data_dir=data_dir,
            latency_ms=0,
            slippage=0.0,
            initial_bankroll=bankroll,
            **eth_engine_kw,
        )

        print(f"  Running: {label:<20s} (cal={use_cal}, prior={cal_prior}, rev={effective_rev:.2f})")
        _, metrics, trades_df = engine.run()

        results.append((label, metrics, trades_df))

    # Summary table
    mode_str = "MAKER" if maker else "FOK"
    print(f"\n{'='*90}")
    print(f"  CALIBRATION COMPARISON — {config.display_name} [{mode_str}]")
    print(f"{'='*90}")
    print(f"  {'Variant':<20s} {'Trades':>7s} {'WinRate':>8s} {'PnL':>10s} {'Sharpe':>8s} {'DeflSR':>8s} {'MaxDD':>8s} {'Final$':>10s}")
    print(f"  {'-'*20} {'-'*7} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")

    for label, m, _ in results:
        print(f"  {label:<20s} {m.get('n_trades',0):>7d} {m.get('win_rate',0):>7.1%} "
              f"{m.get('total_pnl',0):>+10.2f} {m.get('sharpe',0):>8.3f} "
              f"{m.get('sharpe_deflated',0):>8.3f} {m.get('max_drawdown',0):>8.2f} "
              f"{m.get('final_bankroll',0):>10.2f}")

    print()

    # Detailed output for best variant
    best = max(results, key=lambda r: r[1].get("total_pnl", 0))
    print(f"  Best by PnL: {best[0]}")
    best_sr = max(results, key=lambda r: r[1].get("sharpe_deflated", 0))
    print(f"  Best by Deflated Sharpe: {best_sr[0]}")
    print()

    return results


def main():
    parser = argparse.ArgumentParser(description="Calibration Table A/B Comparison")
    parser.add_argument("--market", default="btc",
                        choices=list(MARKET_CONFIGS),
                        help="Market to test (default: btc)")
    parser.add_argument("--all", action="store_true",
                        help="Run all markets")
    parser.add_argument("--fok", action="store_true",
                        help="Use FOK mode instead of maker mode")
    parser.add_argument("--bankroll", type=float, default=10_000.0)
    args = parser.parse_args()

    markets = list(MARKET_CONFIGS) if args.all else [args.market]
    maker = not args.fok

    for market in markets:
        print(f"\n{'#'*90}")
        print(f"  MARKET: {market}")
        print(f"{'#'*90}\n")
        run_comparison(market, maker=maker, bankroll=args.bankroll)


if __name__ == "__main__":
    main()
