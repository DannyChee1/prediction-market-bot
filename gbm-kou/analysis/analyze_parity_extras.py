#!/usr/bin/env python3
"""Additional diagnostics for the parity experiment.

- Per-bucket (p_side) calibration of both replay variants.
- Ratio of (B1 abs_mean diff) vs (B2 abs_mean diff) = effect size of calibration.
- Confirm whether calibration table is near no-op (cal_max_weight * (n/(n+n0))).
- Inspect cells of the calibration table.
- Check that replay's per-trade edge differs from live by a huge amount.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path('/Users/dannychee/Desktop/prediction-market-bot')
sys.path.insert(0, str(ROOT))

from backtest_core import build_calibration_table  # noqa: E402
from backtest import DATA_DIR  # noqa: E402


def load(p: str) -> list[dict]:
    return json.load(open(p))


def replay_p_side(rep: dict) -> float | None:
    p = rep.get('p_model_trade')
    if p is None:
        return None
    side = rep.get('live_side')
    if side == 'UP':
        return float(p)
    if side == 'DOWN':
        return 1.0 - float(p)
    return None


def per_bucket(data: list[dict], label: str):
    print(f'\n--- {label} ---')
    print('p_side bucket            n    live_WR  replay_p_side_mean  gap')
    buckets = [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6),
               (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    for lo, hi in buckets:
        cases = []
        for r in data:
            ps = replay_p_side(r)
            if ps is None or r.get('live_win') is None:
                continue
            if lo <= ps < hi:
                cases.append(r)
        if not cases:
            continue
        n = len(cases)
        wr = sum(r['live_win'] for r in cases) / n
        mean_ps = sum(replay_p_side(r) for r in cases) / n
        print(f'  [{lo:.1f},{hi:.1f})  n={n:4d}  WR={wr:5.3f}  '
              f'mean_p_side={mean_ps:5.3f}  gap={mean_ps - wr:+.3f}')


def live_bucket(data: list[dict], label: str):
    print(f'\n--- {label} (using LIVE p_side) ---')
    print('p_side bucket            n    live_WR  mean_live_p_side  gap')
    buckets = [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6),
               (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    for lo, hi in buckets:
        cases = []
        for r in data:
            ps = r.get('live_p_side')
            if ps is None or r.get('live_win') is None:
                continue
            if lo <= ps < hi:
                cases.append(r)
        if not cases:
            continue
        n = len(cases)
        wr = sum(r['live_win'] for r in cases) / n
        mean_ps = sum(r['live_p_side'] for r in cases) / n
        print(f'  [{lo:.1f},{hi:.1f})  n={n:4d}  WR={wr:5.3f}  '
              f'mean_p_side={mean_ps:5.3f}  gap={mean_ps - wr:+.3f}')


def inspect_cal_table():
    for mk, subdir in (('btc_5m', 'btc_5m'), ('btc', 'btc_15m')):
        print(f'\n--- CAL TABLE for {mk} ({subdir}) ---')
        cal = build_calibration_table(DATA_DIR / subdir, vol_lookback_s=90)
        print(f'n cells: {len(cal.table)}')
        print(f'total obs: {sum(cal.counts.values())}')
        print(f'table entries (showing all, sorted by count):')
        items = [(k, cal.table[k], cal.counts[k]) for k in cal.table]
        items.sort(key=lambda x: -x[2])
        for k, p, n in items[:20]:
            # cal_max_weight=0.70, cal_prior_strength=50.0
            w = min(n / (n + 50.0), 0.70)
            print(f'  z_bin={k[0]:+.2f} tau_bin={k[1]:6.0f}  n={n:6d}  p_cal={p:.4f}  weight={w:.3f}')


def per_trade_compare(with_cal: list[dict], without_cal: list[dict]):
    """Compare replay-per-trade to live-per-trade claimed edge and p_side."""
    print('\n--- PER-TRADE CLAIMED EDGE: live vs replay ---')
    for label, data in [('WITH CAL', with_cal), ('NO CAL', without_cal)]:
        diffs_edge = []
        diffs_pside = []
        for r in data:
            live_ec = r.get('live_edge_claimed')
            live_ps = r.get('live_p_side')
            repl_ps = replay_p_side(r)
            cb = r.get('live_cost_basis')
            if (live_ec is None or live_ps is None or repl_ps is None
                    or cb is None):
                continue
            repl_ec = repl_ps - float(cb)
            diffs_edge.append(live_ec - repl_ec)
            diffs_pside.append(live_ps - repl_ps)
        if not diffs_edge:
            continue
        abs_edge = [abs(d) for d in diffs_edge]
        abs_ps = [abs(d) for d in diffs_pside]
        print(f'  [{label}] n={len(diffs_edge)}')
        print(f'    abs_mean(live_edge - replay_edge): {sum(abs_edge)/len(abs_edge):.4f}')
        print(f'    abs_mean(live_p_side - replay_p_side): {sum(abs_ps)/len(abs_ps):.4f}')
        ge10_edge = sum(1 for d in abs_edge if d > 0.10)
        ge10_ps = sum(1 for d in abs_ps if d > 0.10)
        print(f'    |live_edge - replay_edge| > 0.10: {ge10_edge}/{len(diffs_edge)} '
              f'({100*ge10_edge/len(diffs_edge):.1f}%)')
        print(f'    |live_p_side - replay_p_side| > 0.10: {ge10_ps}/{len(diffs_edge)} '
              f'({100*ge10_ps/len(diffs_edge):.1f}%)')


def extreme_live_cases(with_cal: list[dict]):
    """On trades where live p_model was extreme (< 0.10 or > 0.90),
    what did replay produce?"""
    print('\n--- EXTREME LIVE P_MODEL CASES (|live_p - 0.5| > 0.40) ---')
    for label, data in [('WITH CAL', with_cal)]:
        n_extreme = 0
        n_replay_extreme = 0
        worst = []
        for r in data:
            lp = r.get('live_p_model')
            rp = r.get('p_model_trade')
            if lp is None or rp is None:
                continue
            if abs(lp - 0.5) > 0.40:
                n_extreme += 1
                if abs(rp - 0.5) > 0.40:
                    n_replay_extreme += 1
                worst.append((abs(lp - rp), lp, rp, r.get('live_side'),
                              r.get('market_slug')))
        print(f'  [{label}] n live extreme: {n_extreme}')
        print(f'  [{label}] n where replay is ALSO extreme: {n_replay_extreme}')
        worst.sort(reverse=True)
        print(f'  top 5 live-vs-replay divergence on extreme cases:')
        for d, lp, rp, side, slug in worst[:5]:
            print(f'    slug={slug} side={side}  live={lp:.4f}  replay={rp:.4f}  |diff|={d:.4f}')


def main():
    with_cal = load('/tmp/_parity_with_cal.json')
    without_cal = load('/tmp/_parity_without_cal.json')

    inspect_cal_table()
    per_trade_compare(with_cal, without_cal)
    extreme_live_cases(with_cal)

    print('\n========================================================================')
    print('CALIBRATION BUCKET (replay p_side) vs live outcome')
    print('========================================================================')
    per_bucket(with_cal, 'B1: WITH calibration')
    per_bucket(without_cal, 'B2: WITHOUT calibration')
    live_bucket(with_cal, 'LIVE reference')


if __name__ == '__main__':
    main()
