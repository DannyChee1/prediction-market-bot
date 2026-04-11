#!/usr/bin/env python3
"""Diff live vs replay p_model for extreme cases.

For every live trade with |p_model - 0.5| > 0.40, replay the matching
parquet window cold-start through decide_both_sides() and record side-by-side
the full set of intermediate variables:

  sigma_per_s (post-smoothing), sigma_raw (pre-smoothing),
  delta, z_raw, z_capped, p_gbm, p_model_raw, p_model_trade,
  price_history_len, effective_price, window_start_price,
  chainlink_price, binance_mid.

Sources:
  /Users/dannychee/Desktop/prediction-market-bot/live_trades_btc*.jsonl
    - limit_order (provides market_slug, chainlink_price, cost_basis,
      p_model, p_side, sigma_per_s, tau)
    - limit_fill (provides tau, p_model, sigma_per_s at fill time)
    - diagnostic (provides delta, delta_chainlink, window_start_price,
      chainlink_price, sigma_per_s, hist_len — the last one preceding
      each fill, for the same slug, is our live-side sample)

Outputs:
  /tmp/_extreme_divergence_rows.json
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

ROOT = Path('/Users/dannychee/Desktop/prediction-market-bot')
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from backtest import build_diffusion_signal  # noqa: E402
from backtest_core import Snapshot  # noqa: E402


def parquet_path(slug: str) -> Path | None:
    if slug.startswith('btc-updown-5m'):
        return ROOT / 'data' / 'btc_5m' / f'{slug}.parquet'
    if slug.startswith('btc-updown-15m'):
        return ROOT / 'data' / 'btc_15m' / f'{slug}.parquet'
    return None


def market_key_for(slug: str) -> str | None:
    if slug.startswith('btc-updown-5m'):
        return 'btc_5m'
    if slug.startswith('btc-updown-15m'):
        return 'btc'
    return None


def collect_live_extreme_fills():
    """Walk every live_trades_btc*.jsonl (excluding 1h), collect fills with
    |p_model - 0.5| > 0.40. For each, resolve market_slug via the most
    recent preceding limit_order (same order_id), and attach the most
    recent diagnostic for the same slug.
    """
    files = sorted(glob.glob(str(ROOT / 'live_trades_btc*.jsonl')))
    out = []
    for f in files:
        if 'btc_1h' in f:
            continue
        orders = {}   # order_id -> limit_order dict
        diag_by_slug = {}  # slug -> list of diagnostic dicts (chronological)
        with open(f) as fp:
            for line in fp:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                t = r.get('type')
                if t == 'limit_order':
                    oid = r.get('order_id')
                    if oid:
                        orders[oid] = r
                elif t == 'diagnostic':
                    slug = r.get('market_slug')
                    if slug:
                        diag_by_slug.setdefault(slug, []).append(r)
                elif t in ('limit_fill', 'ws_fill'):
                    p = r.get('p_model')
                    if p is None or abs(p - 0.5) <= 0.40:
                        continue
                    oid = r.get('order_id')
                    order = orders.get(oid) if oid else None
                    slug = order.get('market_slug') if order else None
                    diag = None
                    if slug and slug in diag_by_slug:
                        # Find the latest diagnostic at or before fill ts
                        fill_ts = r.get('ts', '')
                        candidates = [
                            d for d in diag_by_slug[slug] if d.get('ts', '') <= fill_ts
                        ]
                        if candidates:
                            diag = candidates[-1]
                    out.append({
                        'file': Path(f).name,
                        'fill': r,
                        'order': order,
                        'slug': slug,
                        'diag': diag,
                    })
    return out


def replay_window_capture(slug: str, target_tau: float, market_key: str) -> dict | None:
    """Replay the window parquet cold-start and return intermediate vars
    captured at the row closest to target_tau."""
    fp = parquet_path(slug)
    if fp is None or not fp.exists():
        return {'error': 'parquet missing', 'slug': slug}
    try:
        df = pd.read_parquet(fp)
    except Exception as exc:
        return {'error': f'parquet read error: {exc}', 'slug': slug}
    if len(df) == 0:
        return {'error': 'empty parquet', 'slug': slug}

    sig = build_diffusion_signal(market_key, bankroll=100.0, maker=False)
    # Replay behavior: no calibration table (we know from prior audit it
    # moves p_model <2pp, not the bug).
    sig.calibration_table = None

    ctx: dict = {'inventory_up': 0, 'inventory_down': 0}
    if 'window_start_ms' in df.columns:
        ctx['_window_start_ms'] = int(df['window_start_ms'].iloc[0])

    has_binance = 'binance_mid' in df.columns

    best = None
    min_dtau = 1e9

    for _, row in df.iterrows():
        snap = Snapshot.from_row(row)
        if snap is None:
            continue
        if has_binance and pd.notna(row.get('binance_mid')) and row['binance_mid'] > 0:
            ctx['_binance_mid'] = float(row['binance_mid'])
        try:
            sig.decide_both_sides(snap, ctx)
        except Exception as exc:
            return {'error': f'decide exc: {exc}', 'slug': slug}

        dtau = abs(float(snap.time_remaining_s) - target_tau)
        if dtau < min_dtau:
            min_dtau = dtau
            hist = ctx.get('price_history', [])
            ts_hist = ctx.get('ts_history', [])
            sigma_per_s = ctx.get('_sigma_per_s')
            z_cap = ctx.get('_z')
            z_raw = ctx.get('_z_raw')
            # Derive raw sigma from the pre-smoothed computation used
            # in the display path (line 1632). When the fast path ran,
            # it stored _raw into _sigma_per_s BEFORE the main path
            # overwrote with smoothed sigma. After main path runs, we
            # only see the smoothed value here.
            effective_price = ctx.get('_binance_mid') or snap.chainlink_price
            ws_price = snap.window_start_price or 0.0
            delta = 0.0
            if ws_price > 0 and effective_price:
                delta = (effective_price - ws_price) / ws_price

            best = {
                'tau_matched': float(snap.time_remaining_s),
                'dtau': dtau,
                'sigma_per_s': sigma_per_s,
                'z_raw': z_raw,
                'z_capped': z_cap,
                'p_gbm': None if z_cap is None else _norm_cdf(z_cap),
                'p_model_raw': ctx.get('_p_model_raw'),
                'p_model_trade': ctx.get('_p_model_trade'),
                'delta_frac': delta,
                'delta_dollar': (effective_price - ws_price) if ws_price > 0 else 0.0,
                'effective_price': effective_price,
                'window_start_price': ws_price,
                'chainlink_price': snap.chainlink_price,
                'binance_mid': ctx.get('_binance_mid'),
                'price_history_len': len(hist),
                'ts_history_span_s': (ts_hist[-1] - ts_hist[0]) / 1000.0 if len(ts_hist) >= 2 else 0.0,
                'hist_len_vol_lookback': min(len(hist), sig.vol_lookback_s),
                'dedup_ratio': _dedup_ratio(hist[-sig.vol_lookback_s:]),
                'tau_target': target_tau,
            }
    return best


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _dedup_ratio(prices: list) -> float:
    """Fraction of prices that are UNIQUE vs total length.
    Higher = more distinct price levels, lower = many duplicates."""
    if len(prices) < 2:
        return 0.0
    unique = len({float(p) for p in prices})
    return unique / len(prices)


def main():
    print('Collecting live extreme fills...', file=sys.stderr)
    live_fills = collect_live_extreme_fills()
    print(f'  {len(live_fills)} extreme fills', file=sys.stderr)

    results = []
    for i, lf in enumerate(live_fills):
        fill = lf['fill']
        order = lf['order']
        slug = lf['slug']
        diag = lf['diag']
        tau = fill.get('tau')

        row = {
            'file': lf['file'],
            'fill_ts': fill.get('ts'),
            'slug': slug,
            'side': fill.get('side'),
            'tau': tau,
            'live_p_model': fill.get('p_model'),
            'live_p_side': fill.get('p_side'),
            'live_sigma_per_s': fill.get('sigma_per_s'),
            'live_cost_basis': fill.get('cost_basis'),
        }
        if order is not None:
            row['order_chainlink_price'] = order.get('chainlink_price')
            row['order_sigma_per_s'] = order.get('sigma_per_s')
        if diag is not None:
            row['diag_delta'] = diag.get('delta')
            row['diag_delta_chainlink'] = diag.get('delta_chainlink')
            row['diag_window_start_price'] = diag.get('window_start_price')
            row['diag_chainlink_price'] = diag.get('chainlink_price')
            row['diag_sigma_per_s'] = diag.get('sigma_per_s')
            row['diag_hist_len'] = diag.get('hist_len')
            row['diag_p_model'] = diag.get('p_model')
            row['diag_reason'] = diag.get('reason')
            row['diag_tau'] = diag.get('tau')

        if slug is None or tau is None:
            row['replay_error'] = 'missing slug or tau'
            results.append(row)
            continue
        mk = market_key_for(slug)
        if mk is None:
            row['replay_error'] = 'unknown market'
            results.append(row)
            continue

        replay = replay_window_capture(slug, float(tau), mk)
        if replay is None:
            row['replay_error'] = 'no replay'
        elif 'error' in replay:
            row['replay_error'] = replay['error']
        else:
            for k, v in replay.items():
                row[f'replay_{k}'] = v

            # Derived diffs
            live_sig = fill.get('sigma_per_s')
            replay_sig = replay.get('sigma_per_s')
            if live_sig is not None and replay_sig is not None and replay_sig > 0:
                row['sigma_ratio_live_over_replay'] = float(live_sig) / float(replay_sig)

            lpm = fill.get('p_model')
            rpm = replay.get('p_model_trade') or replay.get('p_model_raw')
            if lpm is not None and rpm is not None:
                row['p_model_abs_diff'] = abs(float(lpm) - float(rpm))

        results.append(row)
        if (i + 1) % 10 == 0:
            print(f'  replayed {i + 1}/{len(live_fills)}', file=sys.stderr)

    out_path = '/tmp/_extreme_divergence_rows.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, default=str, indent=2)
    print(f'wrote {len(results)} rows to {out_path}')


if __name__ == '__main__':
    main()
