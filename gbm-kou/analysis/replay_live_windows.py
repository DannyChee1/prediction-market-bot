#!/usr/bin/env python3
"""Replay live-traded windows through the backtest signal.

For each live trade, re-run the signal on the same parquet window up to the
live fill's tau, capture the signal's decision at that moment, and compare
to the live claim. Tests live-vs-backtest parity on the exact same windows
the live bot traded.

Outputs:
  /tmp/_replays.json
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

ROOT = Path('/Users/dannychee/Desktop/prediction-market-bot')
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from backtest_core import Snapshot  # noqa: E402
from backtest import build_diffusion_signal  # noqa: E402
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


def replay_window_to_tau(slug: str, target_tau: float, market_key: str):
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
    ctx: dict = {"inventory_up": 0, "inventory_down": 0}
    if "window_start_ms" in df.columns:
        ctx["_window_start_ms"] = int(df["window_start_ms"].iloc[0])

    has_binance = "binance_mid" in df.columns

    best_at_tau = None
    min_dtau = 1e9

    # Also capture: did the backtest EVER fire a trade in this window at all,
    # and if so what was the first fire?
    first_fire = None
    fire_reasons_up = []
    fire_reasons_down = []

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

        if first_fire is None:
            if up_dec.action != 'FLAT' and up_dec.size_usd > 0:
                first_fire = {
                    'tau': float(snap.time_remaining_s),
                    'side': 'UP',
                    'action': up_dec.action,
                    'edge': up_dec.edge,
                    'size_usd': up_dec.size_usd,
                    'cost_basis': snap.best_ask_up,
                }
            elif down_dec.action != 'FLAT' and down_dec.size_usd > 0:
                first_fire = {
                    'tau': float(snap.time_remaining_s),
                    'side': 'DOWN',
                    'action': down_dec.action,
                    'edge': down_dec.edge,
                    'size_usd': down_dec.size_usd,
                    'cost_basis': snap.best_ask_down,
                }

        dtau = abs(float(snap.time_remaining_s) - target_tau)
        if dtau < min_dtau:
            min_dtau = dtau
            p_up_raw = ctx.get("_p_model_raw") or ctx.get("p_model")
            p_up_calibrated = ctx.get("p_model")
            sigma = ctx.get("_sigma") or ctx.get("sigma_per_s")
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
                "p_model_raw": p_up_raw,
                "p_model_calibrated": p_up_calibrated,
                "sigma_per_s": sigma,
                "best_bid_up": snap.best_bid_up,
                "best_ask_up": snap.best_ask_up,
                "best_bid_down": snap.best_bid_down,
                "best_ask_down": snap.best_ask_down,
                "chainlink_price": snap.chainlink_price,
                "binance_mid": ctx.get("_binance_mid"),
                "dtau_s": dtau,
            }
        # Continue past target_tau so we can still find first_fire later in window
    if best_at_tau is not None:
        best_at_tau['first_fire'] = first_fire
    return best_at_tau


def main():
    rows = json.load(open('/tmp/_analysis_rows.json'))
    replays = []
    for i, r in enumerate(rows):
        slug = r.get('market_slug')
        if not slug:
            continue
        mk = market_key_for(slug)
        if mk is None:
            continue
        tgt_tau = float(r.get('tau') or 0)
        try:
            res = replay_window_to_tau(slug, tgt_tau, mk)
        except Exception as exc:
            res = {"error": str(exc)}
        if res is not None:
            res['live_tau'] = tgt_tau
            res['live_side'] = r['side']
            res['live_cost_basis'] = r['cost_basis']
            res['live_p_side'] = r['p_side']
            res['live_edge_claimed'] = r['edge_claimed']
            res['live_outcome'] = r['outcome']
            res['live_win'] = r['win']
            res['market_slug'] = slug
            res['market_key'] = mk
            res['live_realized_edge'] = r['realized_edge']
            replays.append(res)
        if (i + 1) % 25 == 0:
            print(f'replayed {i + 1}/{len(rows)}', file=sys.stderr)
    out = '/tmp/_replays.json'
    with open(out, 'w') as f:
        json.dump(replays, f, default=str)
    print(f'done: {len(replays)} replays -> {out}')


if __name__ == '__main__':
    main()
