"""
Timezone / Time-of-Day Analysis
================================
Analyzes how trading performance varies by UTC hour and trading session.
Groups backtest results into sessions and per-hour buckets to identify
when the signal has edge and when it doesn't.

Sessions (UTC):
  Asia:    00:00 - 08:00 UTC
  Europe:  08:00 - 14:00 UTC
  US:      14:00 - 21:00 UTC
  Off:     21:00 - 00:00 UTC

Also analyzes per-hour: win rate, PnL, avg edge, volatility, spread.

Usage:
  python run_tz_analysis.py                  # BTC 15m (default)
  python run_tz_analysis.py --market btc_5m
  python run_tz_analysis.py --all            # BTC + ETH (15m + 5m)
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from backtest import (
    BacktestEngine,
    DiffusionSignal,
    DATA_DIR,
    MARKET_CONFIGS,
    get_config,
)

# Trading sessions (UTC hour ranges, inclusive start, exclusive end)
SESSIONS = {
    "Asia  (00-08)": range(0, 8),
    "Europe(08-14)": range(8, 14),
    "US    (14-21)": range(14, 21),
    "Off   (21-00)": range(21, 24),
}


def get_session(hour: int) -> str:
    for name, hours in SESSIONS.items():
        if hour in hours:
            return name
    return "Unknown"


def analyze_raw_data(data_dir: Path, subdir: str):
    """Analyze raw parquet data for spread, volatility, and book quality by hour."""
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        return None

    hour_stats = defaultdict(lambda: {
        "spreads_up": [], "spreads_down": [],
        "vols": [], "imbalances": [], "n_rows": 0,
        "mid_up": [], "mid_down": [],
    })

    for f in files:
        df = pd.read_parquet(f)
        if df.empty:
            continue
        if "chainlink_btc" in df.columns and "chainlink_price" not in df.columns:
            df.rename(columns={"chainlink_btc": "chainlink_price"}, inplace=True)

        # Check completeness
        if "time_remaining_s" in df.columns and df["time_remaining_s"].iloc[-1] > 5:
            continue

        for _, row in df.iterrows():
            ts = row["ts_ms"] / 1000
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            h = dt.hour
            stats = hour_stats[h]
            stats["n_rows"] += 1

            if all(c in df.columns for c in ["best_ask_up", "best_bid_up",
                                               "best_ask_down", "best_bid_down"]):
                spread_up = row.get("best_ask_up", 0) - row.get("best_bid_up", 0)
                spread_down = row.get("best_ask_down", 0) - row.get("best_bid_down", 0)
                if 0 < spread_up < 1:
                    stats["spreads_up"].append(spread_up)
                if 0 < spread_down < 1:
                    stats["spreads_down"].append(spread_down)

            if "mid_up" in df.columns:
                mid = row.get("mid_up")
                if pd.notna(mid) and 0 < mid < 1:
                    stats["mid_up"].append(mid)
            if "mid_down" in df.columns:
                mid = row.get("mid_down")
                if pd.notna(mid) and 0 < mid < 1:
                    stats["mid_down"].append(mid)

    return hour_stats


def run_tz_analysis(market: str, maker: bool = True, bankroll: float = 10_000.0):
    config = get_config(market)
    data_dir = DATA_DIR / config.data_subdir

    if not data_dir.exists() or not list(data_dir.glob("*.parquet")):
        print(f"  [SKIP] No data in {data_dir}")
        return

    base_market = market.replace("_5m", "")
    is_5m = "_5m" in market

    # Use pure GBM (no_cal_no_rev) — the best config from calibration analysis
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

    signal = DiffusionSignal(
        bankroll=bankroll,
        slippage=0.0,
        calibration_table=None,
        min_entry_price=0.10,
        inventory_skew=0.02,
        maker_warmup_s=maker_warmup,
        maker_withdraw_s=maker_withdraw,
        max_sigma=config.max_sigma,
        min_sigma=config.min_sigma,
        reversion_discount=0.0,
        **{**eth_overrides, **maker_overrides, **vamp_kw, **five_m_kw},
    )

    engine = BacktestEngine(
        signal=signal,
        data_dir=data_dir,
        latency_ms=0,
        slippage=0.0,
        initial_bankroll=bankroll,
        **eth_engine_kw,
    )

    print(f"  Running backtest (no_cal_no_rev, maker)...")
    results, metrics, trades_df = engine.run()

    if trades_df.empty:
        print(f"  No trades — skipping\n")
        return

    # Add hour column
    trades_df["utc_hour"] = trades_df["entry_ts_ms"].apply(
        lambda ts: datetime.fromtimestamp(ts / 1000, tz=timezone.utc).hour
    )
    trades_df["session"] = trades_df["utc_hour"].apply(get_session)
    trades_df["won"] = trades_df["pnl"] > 0

    # ── Per-Hour Analysis ──
    print(f"\n{'='*95}")
    print(f"  TIME-OF-DAY ANALYSIS — {config.display_name} [MAKER, no_cal_no_rev]")
    print(f"{'='*95}")

    print(f"\n  {'Hour':>6s} {'Trades':>7s} {'WinRate':>8s} {'PnL':>10s} {'AvgPnL':>9s} "
          f"{'AvgPrice':>9s} {'AvgTLeft':>9s} {'UP%':>6s} {'DOWN%':>6s}")
    print(f"  {'-'*6} {'-'*7} {'-'*8} {'-'*10} {'-'*9} {'-'*9} {'-'*9} {'-'*6} {'-'*6}")

    hour_results = {}
    for h in range(24):
        hdf = trades_df[trades_df["utc_hour"] == h]
        if hdf.empty:
            print(f"  {h:02d}:00  {0:>7d} {'—':>8s} {'—':>10s} {'—':>9s} {'—':>9s} {'—':>9s} {'—':>6s} {'—':>6s}")
            hour_results[h] = {"trades": 0, "pnl": 0, "win_rate": 0}
            continue

        n = len(hdf)
        wins = hdf["won"].sum()
        wr = wins / n
        total_pnl = hdf["pnl"].sum()
        avg_pnl = total_pnl / n
        avg_price = hdf["entry_price"].mean()
        avg_tleft = hdf["time_left_s"].mean()
        up_pct = (hdf["side"] == "UP").mean()
        down_pct = (hdf["side"] == "DOWN").mean()

        hour_results[h] = {"trades": n, "pnl": total_pnl, "win_rate": wr}

        pnl_color = "+" if total_pnl >= 0 else ""
        print(f"  {h:02d}:00  {n:>7d} {wr:>7.1%} {pnl_color}{total_pnl:>9.2f} "
              f"{avg_pnl:>+9.2f} {avg_price:>9.4f} {avg_tleft:>9.1f} "
              f"{up_pct:>5.0%} {down_pct:>5.0%}")

    # ── Per-Session Analysis ──
    print(f"\n  {'Session':<16s} {'Trades':>7s} {'WinRate':>8s} {'PnL':>10s} {'AvgPnL':>9s} {'PnL/Trade':>10s}")
    print(f"  {'-'*16} {'-'*7} {'-'*8} {'-'*10} {'-'*9} {'-'*10}")

    for session_name, hour_range in SESSIONS.items():
        sdf = trades_df[trades_df["utc_hour"].isin(hour_range)]
        if sdf.empty:
            print(f"  {session_name:<16s} {0:>7d} {'—':>8s} {'—':>10s} {'—':>9s} {'—':>10s}")
            continue

        n = len(sdf)
        wins = sdf["won"].sum()
        wr = wins / n
        total_pnl = sdf["pnl"].sum()
        avg_pnl = total_pnl / n

        print(f"  {session_name:<16s} {n:>7d} {wr:>7.1%} {total_pnl:>+10.2f} "
              f"{avg_pnl:>+9.2f} {avg_pnl:>+10.2f}")

    # ── Outcome bias by hour (does UP/DOWN win more at certain hours?) ──
    print(f"\n  Outcome bias by hour (fraction of windows where UP won):")
    # Need to go back to raw data for this
    files = sorted(data_dir.glob("*.parquet"))
    hour_outcomes = defaultdict(list)
    for f in files:
        df = pd.read_parquet(f)
        if df.empty:
            continue
        if "chainlink_btc" in df.columns and "chainlink_price" not in df.columns:
            df.rename(columns={"chainlink_btc": "chainlink_price"}, inplace=True)
        if "time_remaining_s" not in df.columns or df["time_remaining_s"].iloc[-1] > 5:
            continue

        start_px = df["window_start_price"].dropna().iloc[0] if not df["window_start_price"].dropna().empty else None
        if start_px is None or start_px == 0:
            continue
        final_px = df["chainlink_price"].iloc[-1]
        outcome_up = 1 if final_px >= start_px else 0

        ts = df["ts_ms"].iloc[0] / 1000
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour_outcomes[dt.hour].append(outcome_up)

    print(f"  {'Hour':>6s} {'Windows':>8s} {'UP_win%':>8s} {'DOWN_win%':>9s}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*9}")
    for h in range(24):
        outcomes = hour_outcomes.get(h, [])
        if not outcomes:
            continue
        n = len(outcomes)
        up_rate = sum(outcomes) / n
        print(f"  {h:02d}:00  {n:>8d} {up_rate:>7.1%} {1-up_rate:>8.1%}")

    # ── Spread analysis by hour (from raw data, sampled) ──
    print(f"\n  Avg spread by hour (sampled from raw data):")
    hour_spreads = defaultdict(list)
    for f in list(files)[:200]:  # sample first 200 files for speed
        df = pd.read_parquet(f)
        if df.empty:
            continue
        if "chainlink_btc" in df.columns and "chainlink_price" not in df.columns:
            df.rename(columns={"chainlink_btc": "chainlink_price"}, inplace=True)
        if "time_remaining_s" not in df.columns or df["time_remaining_s"].iloc[-1] > 5:
            continue
        needed = ["best_ask_up", "best_bid_up", "best_ask_down", "best_bid_down"]
        if not all(c in df.columns for c in needed):
            continue
        # Sample every 30th row
        for idx in range(0, len(df), 30):
            row = df.iloc[idx]
            ts = row["ts_ms"] / 1000
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            spread_up = row["best_ask_up"] - row["best_bid_up"]
            spread_down = row["best_ask_down"] - row["best_bid_down"]
            if 0 < spread_up < 0.5 and 0 < spread_down < 0.5:
                hour_spreads[dt.hour].append((spread_up + spread_down) / 2)

    print(f"  {'Hour':>6s} {'AvgSpread':>10s} {'Samples':>8s}")
    print(f"  {'-'*6} {'-'*10} {'-'*8}")
    for h in range(24):
        spreads = hour_spreads.get(h, [])
        if not spreads:
            continue
        print(f"  {h:02d}:00  {np.mean(spreads):>10.4f} {len(spreads):>8d}")

    # ── Best/worst hours summary ──
    active_hours = {h: v for h, v in hour_results.items() if v["trades"] > 0}
    if active_hours:
        best_hour = max(active_hours, key=lambda h: active_hours[h]["pnl"])
        worst_hour = min(active_hours, key=lambda h: active_hours[h]["pnl"])
        best_wr = max(active_hours, key=lambda h: active_hours[h]["win_rate"])
        worst_wr = min(active_hours, key=lambda h: active_hours[h]["win_rate"])

        print(f"\n  Summary:")
        print(f"    Best PnL hour:     {best_hour:02d}:00 UTC  (${active_hours[best_hour]['pnl']:+.2f})")
        print(f"    Worst PnL hour:    {worst_hour:02d}:00 UTC  (${active_hours[worst_hour]['pnl']:+.2f})")
        print(f"    Best WR hour:      {best_wr:02d}:00 UTC  ({active_hours[best_wr]['win_rate']:.1%})")
        print(f"    Worst WR hour:     {worst_wr:02d}:00 UTC  ({active_hours[worst_wr]['win_rate']:.1%})")

    print()


def main():
    parser = argparse.ArgumentParser(description="Timezone / Time-of-Day Analysis")
    parser.add_argument("--market", default="btc",
                        choices=list(MARKET_CONFIGS),
                        help="Market to analyze (default: btc)")
    parser.add_argument("--all", action="store_true",
                        help="Run BTC + ETH (15m + 5m)")
    parser.add_argument("--bankroll", type=float, default=10_000.0)
    args = parser.parse_args()

    markets = ["btc", "btc_5m", "eth", "eth_5m"] if args.all else [args.market]

    for market in markets:
        print(f"\n{'#'*95}")
        print(f"  MARKET: {market}")
        print(f"{'#'*95}\n")
        run_tz_analysis(market, maker=True, bankroll=args.bankroll)


if __name__ == "__main__":
    main()
