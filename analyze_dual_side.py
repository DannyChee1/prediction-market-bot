#!/usr/bin/env python3
"""Analyze dual-side fill economics: are we capturing or paying the spread?"""
from __future__ import annotations
import argparse, math, random, sys, numpy as np, pandas as pd
from pathlib import Path

from backtest import DATA_DIR, DiffusionSignal, Snapshot, build_calibration_table
from market_config import get_config
from tick_backtest import run_window, WindowResult

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", default="btc_15m")
    parser.add_argument("--windows", "-w", type=int, default=195)
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--min-sigma", type=float, default=None)
    # Signal params
    parser.add_argument("--max-z", type=float, default=3.0)
    parser.add_argument("--reversion-discount", type=float, default=0.0)
    parser.add_argument("--edge-threshold", type=float, default=0.12)
    parser.add_argument("--gamma-spread", type=float, default=1.5)
    parser.add_argument("--maker-warmup", type=float, default=None)
    parser.add_argument("--maker-withdraw", type=float, default=None)
    parser.add_argument("--max-order-age", type=float, default=30.0)
    parser.add_argument("--edge-cancel", type=float, default=0.06)
    parser.add_argument("--max-fills", type=int, default=2)
    args = parser.parse_args()

    market_key = args.market
    _ALIASES = {"btc_15m": "btc", "eth_15m": "eth", "sol_15m": "sol"}
    market_key = _ALIASES.get(market_key, market_key)
    config = get_config(market_key)
    data_dir = DATA_DIR / config.data_subdir
    is_5m = market_key.endswith("_5m")
    window_duration_s = config.window_duration_s

    all_files = sorted(data_dir.glob("*.parquet"))
    split_idx = int(len(all_files) * 0.60)
    pool_files = all_files[split_idx:]
    rng = random.Random(args.seed)
    n = min(args.windows, len(pool_files))
    selected = sorted(rng.sample(pool_files, n))

    warmup = args.maker_warmup if args.maker_warmup else (30.0 if is_5m else 100.0)
    withdraw = args.maker_withdraw if args.maker_withdraw else (20.0 if is_5m else 60.0)

    signal = DiffusionSignal(
        bankroll=1000.0, max_z=args.max_z, reversion_discount=args.reversion_discount,
        edge_threshold=args.edge_threshold, max_bet_fraction=0.05, kelly_fraction=0.25,
        window_duration=window_duration_s, maker_mode=True, maker_warmup_s=warmup,
        maker_withdraw_s=withdraw, momentum_majority=0.0, spread_edge_penalty=0.0,
        slippage=0.0, min_sigma=args.min_sigma if args.min_sigma else config.min_sigma,
        max_sigma=config.max_sigma, vol_lookback_s=30 if is_5m else 90,
        as_mode=True, gamma_spread=args.gamma_spread, gamma_inv=0.15, min_edge=0.05,
        tox_spread=0.05, vpin_spread=0.05, lag_spread=0.08, edge_step=0.01,
        tail_mode="student_t", tail_nu_default=3.0,
    )

    bankroll = 1000.0
    both_side_windows = []
    single_side_windows = []

    for fpath in selected:
        df = pd.read_parquet(fpath)
        if df.empty:
            continue
        if "chainlink_btc" in df.columns and "chainlink_price" not in df.columns:
            df = df.rename(columns={"chainlink_btc": "chainlink_price"})
        df = df.sort_values("ts_ms").reset_index(drop=True)
        signal.bankroll = bankroll

        result, bankroll = run_window(
            df, signal, bankroll, dual_side=True,
            max_order_age_s=args.max_order_age, edge_cancel_threshold=args.edge_cancel,
            maker_warmup_s=warmup, maker_withdraw_s=withdraw,
            window_duration_s=window_duration_s, max_fills=args.max_fills,
        )
        if not result.fills:
            continue

        up_fills = [f for f in result.fills if f.side == "UP"]
        dn_fills = [f for f in result.fills if f.side == "DOWN"]

        if up_fills and dn_fills:
            up_avg = np.mean([f.entry_price for f in up_fills])
            dn_avg = np.mean([f.entry_price for f in dn_fills])
            combined = up_avg + dn_avg
            up_cost = sum(f.cost_usd for f in up_fills)
            dn_cost = sum(f.cost_usd for f in dn_fills)
            both_side_windows.append({
                "slug": result.slug, "outcome_up": result.outcome_up,
                "up_entry": up_avg, "dn_entry": dn_avg, "combined": combined,
                "spread_captured": combined < 1.0,
                "up_cost": up_cost, "dn_cost": dn_cost,
                "pnl": result.pnl,
                "up_pnl": sum(f.pnl for f in up_fills),
                "dn_pnl": sum(f.pnl for f in dn_fills),
            })
        else:
            side = "UP" if up_fills else "DOWN"
            fills = up_fills or dn_fills
            avg_entry = np.mean([f.entry_price for f in fills])
            single_side_windows.append({
                "slug": result.slug, "side": side, "entry": avg_entry,
                "pnl": result.pnl, "won": fills[0].won,
            })

    print(f"\n{'='*70}")
    print(f"  DUAL-SIDE SPREAD ANALYSIS  |  {args.market}")
    print(f"{'='*70}")

    print(f"\n  Single-side windows: {len(single_side_windows)}")
    print(f"  Both-side windows:   {len(both_side_windows)}")

    if both_side_windows:
        print(f"\n  BOTH-SIDE FILL DETAILS:")
        print(f"  {'─'*64}")
        capturing = [w for w in both_side_windows if w["spread_captured"]]
        paying = [w for w in both_side_windows if not w["spread_captured"]]
        print(f"  Spread captured (UP+DN < 1.0): {len(capturing)}")
        print(f"  Spread paid     (UP+DN >= 1.0): {len(paying)}")

        if capturing:
            avg_combined = np.mean([w["combined"] for w in capturing])
            avg_pnl = np.mean([w["pnl"] for w in capturing])
            print(f"    Captured avg combined entry: {avg_combined:.4f} (save {1-avg_combined:.4f}/share)")
            print(f"    Captured avg PnL: ${avg_pnl:+,.2f}")
        if paying:
            avg_combined = np.mean([w["combined"] for w in paying])
            avg_pnl = np.mean([w["pnl"] for w in paying])
            print(f"    Paying avg combined entry:   {avg_combined:.4f} (overpay {avg_combined-1:.4f}/share)")
            print(f"    Paying avg PnL: ${avg_pnl:+,.2f}")

        print(f"\n  PER-WINDOW BREAKDOWN:")
        print(f"  {'─'*64}")
        print(f"  {'Window':<35} {'UP@':>6} {'DN@':>6} {'Sum':>6} {'Spread':>8} {'PnL':>8}")
        for w in both_side_windows:
            sp = "CAPTURE" if w["spread_captured"] else "PAY"
            print(f"  {w['slug']:<35} {w['up_entry']:>6.3f} {w['dn_entry']:>6.3f} "
                  f"{w['combined']:>6.3f} {sp:>8} ${w['pnl']:>+7.2f}")

        total_both_pnl = sum(w["pnl"] for w in both_side_windows)
        total_single_pnl = sum(w["pnl"] for w in single_side_windows)
        print(f"\n  SUMMARY:")
        print(f"  {'─'*64}")
        print(f"  Both-side total PnL:   ${total_both_pnl:+,.2f}")
        print(f"  Single-side total PnL: ${total_single_pnl:+,.2f}")
        print(f"  Overall PnL:           ${total_both_pnl + total_single_pnl:+,.2f}")

    if single_side_windows:
        s_pnl = sum(w["pnl"] for w in single_side_windows)
        s_wins = sum(1 for w in single_side_windows if w["won"])
        s_wr = s_wins / len(single_side_windows)
        avg_e = np.mean([w["entry"] for w in single_side_windows])
        print(f"\n  Single-side: {len(single_side_windows)} windows, {s_wins} wins ({s_wr:.1%}), "
              f"avg entry {avg_e:.3f}, PnL ${s_pnl:+,.2f}")

    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
