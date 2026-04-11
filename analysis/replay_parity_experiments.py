#!/usr/bin/env python3
"""Parity experiments: replay live windows with/without calibration table.

Two experiments in one script:
  A. Parity restoration: run the SAME signal config as live (calibration table
     attached) over the same 455 live-traded windows and compare p_model to
     live's logged p_model.
  B. Calibration isolation: same sample, run TWICE (with cal, without cal),
     compute claimed vs realized edge and point-biserial correlations.

Inputs:
  /tmp/_analysis_rows.json  — prior audit's extracted live trade rows

Outputs:
  /tmp/_parity_with_cal.json
  /tmp/_parity_without_cal.json

Usage:
  python analysis/replay_parity_experiments.py --with-calibration
  python analysis/replay_parity_experiments.py --without-calibration
  python analysis/replay_parity_experiments.py --both
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path('/Users/dannychee/Desktop/prediction-market-bot')
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from backtest import DATA_DIR, build_diffusion_signal  # noqa: E402
from backtest_core import Snapshot, build_calibration_table  # noqa: E402
from market_config import get_config  # noqa: E402


def parquet_path(slug: str) -> Path | None:
    if slug.startswith('btc-updown-5m'):
        return ROOT / 'data' / 'btc_5m' / f'{slug}.parquet'
    if slug.startswith('btc-updown-15m'):
        return ROOT / 'data' / 'btc_15m' / f'{slug}.parquet'
    if slug.startswith('bitcoin-up-or-down'):
        p = ROOT / 'data' / 'btc_1h' / f'{slug}.parquet'
        if p.exists():
            return p
        p2 = ROOT / 'data' / 'btc_1h_real' / f'{slug}.parquet'
        if p2.exists():
            return p2
        return None
    return None


def market_key_for(slug: str) -> str | None:
    if slug.startswith('btc-updown-5m'):
        return 'btc_5m'
    if slug.startswith('btc-updown-15m'):
        return 'btc'
    if slug.startswith('bitcoin-up-or-down'):
        return 'btc_1h'
    return None


# One calibration table per market key; built once and reused across windows.
_CAL_CACHE: dict[str, object] = {}


def get_cal_table(market_key: str):
    """Build (and cache) the calibration table for a market, same as live."""
    if market_key in _CAL_CACHE:
        return _CAL_CACHE[market_key]
    config = get_config(market_key)
    cal_dir = DATA_DIR / config.data_subdir
    print(f'  [{market_key}] building calibration table from {cal_dir}...',
          file=sys.stderr)
    cal = build_calibration_table(cal_dir, vol_lookback_s=90)
    n_cells = len(cal.table)
    n_obs = sum(cal.counts.values())
    print(f'  [{market_key}] calibration: {n_cells} cells, {n_obs} obs',
          file=sys.stderr)
    _CAL_CACHE[market_key] = cal
    return cal


def replay_window_to_tau(slug: str, target_tau: float, market_key: str,
                         use_calibration: bool):
    fp = parquet_path(slug)
    if fp is None or not fp.exists():
        return None
    try:
        df = pd.read_parquet(fp)
    except Exception:
        return None
    if len(df) == 0:
        return None

    sig = build_diffusion_signal(market_key, bankroll=100.0, maker=False)
    if use_calibration:
        sig.calibration_table = get_cal_table(market_key)
        sig.cal_prior_strength = 50.0
        sig.cal_max_weight = 0.70
    else:
        sig.calibration_table = None

    ctx: dict = {"inventory_up": 0, "inventory_down": 0}
    if "window_start_ms" in df.columns:
        ctx["_window_start_ms"] = int(df["window_start_ms"].iloc[0])

    has_binance = "binance_mid" in df.columns

    best_at_tau = None
    min_dtau = 1e9

    for _, row in df.iterrows():
        snap = Snapshot.from_row(row)
        if snap is None:
            continue
        if has_binance and pd.notna(row.get("binance_mid")) and row["binance_mid"] > 0:
            ctx["_binance_mid"] = float(row["binance_mid"])
        try:
            up_dec, down_dec = sig.decide_both_sides(snap, ctx)
        except Exception as exc:
            return {"error": str(exc)}

        dtau = abs(float(snap.time_remaining_s) - target_tau)
        if dtau < min_dtau:
            min_dtau = dtau
            # Capture BOTH:
            #   _p_model_trade = post-cal + post-market_blend (what live logs)
            #   _p_model_raw   = post-cal, pre-market_blend (intermediate)
            p_trade = ctx.get("_p_model_trade")
            p_raw = ctx.get("_p_model_raw")
            sigma = ctx.get("_sigma_per_s") or ctx.get("sigma_per_s")
            best_at_tau = {
                "tau": float(snap.time_remaining_s),
                "up_action": up_dec.action,
                "up_edge": getattr(up_dec, "edge", None),
                "up_size_usd": up_dec.size_usd,
                "up_reason": getattr(up_dec, "reason", None),
                "down_action": down_dec.action,
                "down_edge": getattr(down_dec, "edge", None),
                "down_size_usd": down_dec.size_usd,
                "down_reason": getattr(down_dec, "reason", None),
                "p_model_trade": p_trade,
                "p_model_raw": p_raw,
                "sigma_per_s": sigma,
                "best_bid_up": snap.best_bid_up,
                "best_ask_up": snap.best_ask_up,
                "best_bid_down": snap.best_bid_down,
                "best_ask_down": snap.best_ask_down,
                "chainlink_price": snap.chainlink_price,
                "binance_mid": ctx.get("_binance_mid"),
                "dtau_s": dtau,
            }
    return best_at_tau


def run(use_calibration: bool, out_path: str):
    rows = json.load(open('/tmp/_analysis_rows.json'))
    replays = []
    label = 'WITH-CAL' if use_calibration else 'NO-CAL'
    for i, r in enumerate(rows):
        slug = r.get('market_slug')
        if not slug:
            continue
        mk = market_key_for(slug)
        if mk is None:
            continue
        tgt_tau = float(r.get('tau') or 0)
        try:
            res = replay_window_to_tau(slug, tgt_tau, mk, use_calibration)
        except Exception as exc:
            res = {"error": str(exc)}
        if res is not None:
            res['live_tau'] = tgt_tau
            res['live_side'] = r['side']
            res['live_cost_basis'] = r['cost_basis']
            res['live_p_model'] = r.get('p_model')
            res['live_p_side'] = r['p_side']
            res['live_edge_claimed'] = r['edge_claimed']
            res['live_outcome'] = r['outcome']
            res['live_win'] = r['win']
            res['live_realized_edge'] = r['realized_edge']
            res['market_slug'] = slug
            res['market_key'] = mk
            replays.append(res)
        if (i + 1) % 50 == 0:
            print(f'  [{label}] replayed {i + 1}/{len(rows)}', file=sys.stderr)
    with open(out_path, 'w') as f:
        json.dump(replays, f, default=str)
    print(f'[{label}] done: {len(replays)} replays -> {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--with-calibration', action='store_true')
    ap.add_argument('--without-calibration', action='store_true')
    ap.add_argument('--both', action='store_true')
    args = ap.parse_args()

    if not (args.with_calibration or args.without_calibration or args.both):
        ap.error('pass --with-calibration, --without-calibration, or --both')

    if args.with_calibration or args.both:
        run(True, '/tmp/_parity_with_cal.json')
    if args.without_calibration or args.both:
        run(False, '/tmp/_parity_without_cal.json')


if __name__ == '__main__':
    main()
