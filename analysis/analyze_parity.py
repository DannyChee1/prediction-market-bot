#!/usr/bin/env python3
"""Analyze parity experiment outputs and print all tables for the findings.

Reads:
  /tmp/_parity_with_cal.json
  /tmp/_parity_without_cal.json

Prints:
  - Experiment A: distribution of |live_p_model - replay_p_model|
  - Experiment B: claimed vs realized edge for each variant
  - Correlations
  - p_model distribution summaries
"""
from __future__ import annotations

import json
import math
import statistics as stats
from pathlib import Path


def load(p: str) -> list[dict]:
    return json.load(open(p))


def quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    values = sorted(values)
    def q(p: float) -> float:
        if not values:
            return float('nan')
        k = p * (len(values) - 1)
        lo = int(math.floor(k))
        hi = int(math.ceil(k))
        if lo == hi:
            return values[lo]
        return values[lo] + (values[hi] - values[lo]) * (k - lo)
    return {
        'min': values[0],
        'p05': q(0.05),
        'p25': q(0.25),
        'p50': q(0.50),
        'p75': q(0.75),
        'p95': q(0.95),
        'max': values[-1],
        'mean': sum(values) / len(values),
        'stdev': stats.pstdev(values) if len(values) > 1 else 0.0,
    }


def replay_p_side(rep: dict) -> float | None:
    """Convert replay p_model (always P(UP)) to p_side (P(live's chosen side))."""
    p = rep.get('p_model_trade')
    if p is None:
        return None
    side = rep.get('live_side')
    if side == 'UP':
        return float(p)
    if side == 'DOWN':
        return 1.0 - float(p)
    return None


def pt_biserial(x: list[float], y: list[int]) -> tuple[float, float, int]:
    """Pearson correlation on continuous x, binary y (0/1). Returns (r, t, n)."""
    n = len(x)
    if n < 3:
        return (float('nan'), float('nan'), n)
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den_x = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    den_y = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if den_x == 0 or den_y == 0:
        return (float('nan'), float('nan'), n)
    r = num / (den_x * den_y)
    if 1 - r * r <= 0:
        return (r, float('inf'), n)
    t = r * math.sqrt((n - 2) / (1 - r * r))
    return (r, t, n)


def pct(a: int, b: int) -> str:
    if b == 0:
        return '0.0%'
    return f'{100 * a / b:.1f}%'


def summarize_experiment_a(with_cal: list[dict], without_cal: list[dict]):
    """Compare replay p_model to live p_model with and without calibration."""
    print('=' * 72)
    print('EXPERIMENT A — PARITY RESTORATION')
    print('=' * 72)

    for label, data in [('WITH CALIBRATION', with_cal),
                        ('WITHOUT CALIBRATION (prior replay method)', without_cal)]:
        print(f'\n--- {label} ---')
        # Subset where both live and replay have a p_model at the target tau
        diffs = []
        live_ps = []
        replay_ps = []
        n_total = len(data)
        n_no_replay = 0
        n_no_live = 0
        for r in data:
            live_p = r.get('live_p_model')
            repl_p = r.get('p_model_trade')
            if repl_p is None:
                n_no_replay += 1
                continue
            if live_p is None:
                n_no_live += 1
                continue
            diffs.append(float(live_p) - float(repl_p))
            live_ps.append(float(live_p))
            replay_ps.append(float(repl_p))

        print(f'n total: {n_total}, n replay missing: {n_no_replay}, '
              f'n live missing: {n_no_live}, n matched: {len(diffs)}')
        if not diffs:
            continue

        abs_diffs = [abs(d) for d in diffs]
        gt_05 = sum(1 for d in abs_diffs if d > 0.05)
        gt_10 = sum(1 for d in abs_diffs if d > 0.10)
        gt_20 = sum(1 for d in abs_diffs if d > 0.20)

        mean_d = sum(diffs) / len(diffs)
        sd_d = stats.pstdev(diffs) if len(diffs) > 1 else 0.0
        mean_abs = sum(abs_diffs) / len(abs_diffs)

        print(f'mean diff (live - replay): {mean_d:+.4f}')
        print(f'stdev diff: {sd_d:.4f}')
        print(f'abs_mean diff: {mean_abs:.4f}')
        print(f'|diff| > 0.05: {gt_05}/{len(diffs)} ({pct(gt_05, len(diffs))})')
        print(f'|diff| > 0.10: {gt_10}/{len(diffs)} ({pct(gt_10, len(diffs))})')
        print(f'|diff| > 0.20: {gt_20}/{len(diffs)} ({pct(gt_20, len(diffs))})')

        live_q = quantiles(live_ps)
        repl_q = quantiles(replay_ps)
        print()
        print(f'{"stat":<8} {"live":>10} {"replay":>10}')
        for k in ('min', 'p05', 'p25', 'p50', 'p75', 'p95', 'max', 'mean', 'stdev'):
            print(f'{k:<8} {live_q[k]:>10.4f} {repl_q[k]:>10.4f}')


def side_agreement(data: list[dict]) -> dict:
    """On how many trades does replay produce edge > threshold on the SAME side as live."""
    n_live_side_edge = 0
    n_other_side_edge = 0
    n_no_edge = 0
    n_total = 0
    for r in data:
        if r.get('p_model_trade') is None:
            continue
        n_total += 1
        up_edge = r.get('up_edge') or 0
        down_edge = r.get('down_edge') or 0
        side = r.get('live_side')
        if side == 'UP':
            own = up_edge
            other = down_edge
        else:
            own = down_edge
            other = up_edge
        if own > 0:
            n_live_side_edge += 1
        elif other > 0:
            n_other_side_edge += 1
        else:
            n_no_edge += 1
    return dict(
        n_total=n_total,
        n_same_side_edge=n_live_side_edge,
        n_other_side_edge=n_other_side_edge,
        n_no_edge=n_no_edge,
    )


def summarize_experiment_b(with_cal: list[dict], without_cal: list[dict]):
    print()
    print('=' * 72)
    print('EXPERIMENT B — CALIBRATION TABLE ISOLATION')
    print('=' * 72)

    for label, data in [('B1: WITH calibration table (live parity)', with_cal),
                        ('B2: WITHOUT calibration table (pure Gaussian)', without_cal)]:
        print(f'\n--- {label} ---')
        # Only look at rows where replay computed a p_model
        cases = []
        for r in data:
            p_trade = r.get('p_model_trade')
            if p_trade is None:
                continue
            p_side = replay_p_side(r)
            if p_side is None:
                continue
            cb = r.get('live_cost_basis')
            if cb is None:
                continue
            realized = r.get('live_realized_edge')
            win = r.get('live_win')
            if realized is None or win is None:
                continue
            edge_claimed_replay = p_side - float(cb)
            cases.append({
                'p_side': p_side,
                'p_up': float(p_trade),
                'cost_basis': float(cb),
                'edge_claimed_replay': edge_claimed_replay,
                'edge_claimed_live': r.get('live_edge_claimed'),
                'realized_edge': float(realized),
                'win': int(win),
            })

        n = len(cases)
        if n == 0:
            print('no cases')
            continue
        print(f'n cases: {n}')

        # Claimed vs realized
        mean_claimed_replay = sum(c['edge_claimed_replay'] for c in cases) / n
        mean_claimed_live = sum(c['edge_claimed_live'] for c in cases) / n
        mean_realized = sum(c['realized_edge'] for c in cases) / n

        print(f'  mean live claimed edge:   {mean_claimed_live:+.4f}')
        print(f'  mean replay claimed edge: {mean_claimed_replay:+.4f}')
        print(f'  mean realized edge:       {mean_realized:+.4f}')
        print(f'  gap (replay - realized):  {mean_claimed_replay - mean_realized:+.4f}')
        print(f'  gap (live  - realized):   {mean_claimed_live - mean_realized:+.4f}')

        # Correlations
        xs_edge = [c['edge_claimed_replay'] for c in cases]
        xs_pside = [c['p_side'] for c in cases]
        ys = [c['win'] for c in cases]

        r_edge, t_edge, _ = pt_biserial(xs_edge, ys)
        r_pside, t_pside, _ = pt_biserial(xs_pside, ys)
        print(f'  corr(replay_edge_claimed, win): r={r_edge:+.4f}  t={t_edge:+.2f}')
        print(f'  corr(replay_p_side,        win): r={r_pside:+.4f}  t={t_pside:+.2f}')

        # p_model (P(UP)) distribution
        p_ups = [c['p_up'] for c in cases]
        q = quantiles(p_ups)
        print('  replay p_model (P(UP)) distribution:')
        for k in ('min', 'p05', 'p25', 'p50', 'p75', 'p95', 'max', 'mean', 'stdev'):
            print(f'    {k:<6} {q[k]:.4f}')

        # Side agreement
        agree = side_agreement(data)
        print(f'  edge>0 on same side as live: {agree["n_same_side_edge"]}/{agree["n_total"]} '
              f'({pct(agree["n_same_side_edge"], agree["n_total"])})')
        print(f'  edge>0 on opposite side:     {agree["n_other_side_edge"]}/{agree["n_total"]} '
              f'({pct(agree["n_other_side_edge"], agree["n_total"])})')
        print(f'  no edge on either side:      {agree["n_no_edge"]}/{agree["n_total"]} '
              f'({pct(agree["n_no_edge"], agree["n_total"])})')


def summarize_live(with_cal: list[dict]):
    """Print live's own claimed/realized for reference."""
    print()
    print('=' * 72)
    print('LIVE BASELINE (reference)')
    print('=' * 72)
    ns = 0
    mean_claimed = 0.0
    mean_realized = 0.0
    wins = 0
    for r in with_cal:
        ec = r.get('live_edge_claimed')
        re_ = r.get('live_realized_edge')
        w = r.get('live_win')
        if ec is None or re_ is None or w is None:
            continue
        ns += 1
        mean_claimed += ec
        mean_realized += re_
        wins += w
    mean_claimed /= ns
    mean_realized /= ns
    print(f'n live trades:          {ns}')
    print(f'win rate:               {wins}/{ns} = {100*wins/ns:.1f}%')
    print(f'mean claimed edge:      {mean_claimed:+.4f}')
    print(f'mean realized edge:     {mean_realized:+.4f}')
    print(f'gap (claimed - real):   {mean_claimed - mean_realized:+.4f}')

    # live p_model distribution
    live_ps = [r['live_p_model'] for r in with_cal if r.get('live_p_model') is not None]
    q = quantiles(live_ps)
    print('live p_model distribution:')
    for k in ('min', 'p05', 'p25', 'p50', 'p75', 'p95', 'max', 'mean', 'stdev'):
        print(f'  {k:<6} {q[k]:.4f}')


def main():
    with_cal = load('/tmp/_parity_with_cal.json')
    without_cal = load('/tmp/_parity_without_cal.json')
    summarize_live(with_cal)
    summarize_experiment_a(with_cal, without_cal)
    summarize_experiment_b(with_cal, without_cal)


if __name__ == '__main__':
    main()
