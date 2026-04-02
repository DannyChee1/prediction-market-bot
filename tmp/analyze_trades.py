import json

for name in ['btc_15m', 'btc_5m', 'eth_15m', 'eth_5m']:
    print(f'\n=== {name} ===')

    fill_map = {}
    order_slug_map = {}
    all_res = []

    with open(f'live_trades_{name}.jsonl') as f:
        for line in f:
            d = json.loads(line)
            t = d.get('type')
            if t == 'limit_order' and d.get('success'):
                oid = d.get('order_id', '')
                if oid:
                    order_slug_map[oid] = d.get('market_slug', '')
            elif t in ('limit_fill', 'partial_fill'):
                oid = d.get('order_id', '')
                slug = order_slug_map.get(oid, '')
                if slug and slug not in fill_map:
                    d['_slug'] = slug
                    fill_map[slug] = d
            elif t == 'resolution':
                slug = d['market_slug']
                f0 = fill_map.get(slug)
                if f0:
                    d['_p_side'] = f0.get('p_side', 0)
                    d['_edge'] = f0.get('edge', 0)
                    d['_sigma'] = f0.get('sigma_per_s', 0)
                    d['_tau'] = f0.get('tau', 0)
                    d['_cost_basis'] = f0.get('cost_basis', f0.get('price', 0))
                else:
                    d['_p_side'] = d['_edge'] = d['_sigma'] = d['_tau'] = d['_cost_basis'] = 0
                all_res.append(d)

    for d in all_res:
        won = d['pnl'] > 0
        tag = 'W' if won else 'L'
        print(f"  {d['side']:4s} p={d['_p_side']:.3f} cost={d['_cost_basis']:.2f} "
              f"edge={d['_edge']:.3f} sig={d['_sigma']:.1e} tau={d['_tau']:>4.0f}s "
              f"sh={d['shares']:>5.1f} pnl=${d['pnl']:>+6.2f} {tag}  {d['outcome']}")

    n = len(all_res)
    wins = sum(1 for d in all_res if d['pnl'] > 0)
    total_pnl = sum(d['pnl'] for d in all_res)
    print(f"  --- {n} trades, {wins}W/{n-wins}L ({wins/n*100:.0f}%), PnL=${total_pnl:+.2f}")

# Combined analysis
print("\n\n=== COMBINED ANALYSIS ===")
all_trades = []
for name in ['btc_15m', 'btc_5m', 'eth_15m', 'eth_5m']:
    fill_map = {}
    order_slug_map = {}
    with open(f'live_trades_{name}.jsonl') as f:
        for line in f:
            d = json.loads(line)
            t = d.get('type')
            if t == 'limit_order' and d.get('success'):
                oid = d.get('order_id', '')
                if oid:
                    order_slug_map[oid] = d.get('market_slug', '')
            elif t in ('limit_fill', 'partial_fill'):
                oid = d.get('order_id', '')
                slug = order_slug_map.get(oid, '')
                if slug and slug not in fill_map:
                    d['_slug'] = slug
                    fill_map[slug] = d
            elif t == 'resolution':
                slug = d['market_slug']
                f0 = fill_map.get(slug)
                if f0:
                    all_trades.append({
                        'market': name,
                        'side': d['side'],
                        'outcome': d['outcome'],
                        'p_side': f0.get('p_side', 0),
                        'cost_basis': f0.get('cost_basis', f0.get('price', 0)),
                        'cost_usd': d['cost_usd'],
                        'pnl': d['pnl'],
                        'edge': f0.get('edge', 0),
                        'won': d['pnl'] > 0,
                        'sigma': f0.get('sigma_per_s', 0),
                        'tau': f0.get('tau', 0),
                        'shares': d['shares'],
                    })

# Last 40 trades
print(f"\nLast 40 trades:")
last40 = all_trades[-40:]
for t in last40:
    tag = 'W' if t['won'] else 'L'
    print(f"  {t['market']:10s} {t['side']:4s} p={t['p_side']:.3f} cost={t['cost_basis']:.2f} "
          f"edge={t['edge']:.3f} sig={t['sigma']:.1e} tau={t['tau']:>4.0f}s "
          f"pnl=${t['pnl']:>+6.2f} {tag}")

wins40 = sum(1 for t in last40 if t['won'])
pnl40 = sum(t['pnl'] for t in last40)
print(f"\nLast 40: {wins40}W/{40-wins40}L ({wins40/40*100:.0f}%), PnL=${pnl40:+.2f}")

# Break down by model confidence
print("\n--- By model confidence (last 40) ---")
high = [t for t in last40 if t['p_side'] >= 0.70]
mid = [t for t in last40 if 0.50 <= t['p_side'] < 0.70]
low = [t for t in last40 if t['p_side'] < 0.50]

for label, group in [("p>=0.70 (strong signal)", high), ("p 0.50-0.70 (weak signal)", mid), ("p<0.50 (contrarian)", low)]:
    if not group:
        print(f"  {label}: no trades")
        continue
    w = sum(1 for t in group if t['won'])
    p = sum(t['pnl'] for t in group)
    avg_edge = sum(t['edge'] for t in group) / len(group)
    avg_cost = sum(t['cost_basis'] for t in group) / len(group)
    avg_p = sum(t['p_side'] for t in group) / len(group)
    print(f"  {label}: {len(group)} trades, {w}W ({w/len(group)*100:.0f}%), "
          f"PnL=${p:+.2f}, avg_p={avg_p:.3f}, avg_cost={avg_cost:.2f}, avg_edge={avg_edge:.3f}")

# Check: how often does model predict correctly (p_side > 0.5 = bet on favored side)
print("\n--- Model calibration check (ALL trades) ---")
for label, group in [("ALL", all_trades), ("Last 40", last40)]:
    if not group:
        continue
    # When p_side > 0.50, the model favors our side - what's actual win rate?
    favored = [t for t in group if t['p_side'] >= 0.50]
    unfavored = [t for t in group if t['p_side'] < 0.50]

    if favored:
        fw = sum(1 for t in favored if t['won'])
        fp = sum(t['pnl'] for t in favored)
        avg_p = sum(t['p_side'] for t in favored) / len(favored)
        print(f"  {label} - Favored (p>=0.50): {len(favored)} trades, {fw}W ({fw/len(favored)*100:.0f}%), "
              f"avg_p_side={avg_p:.3f}, PnL=${fp:+.2f}")
    if unfavored:
        uw = sum(1 for t in unfavored if t['won'])
        up = sum(t['pnl'] for t in unfavored)
        avg_p = sum(t['p_side'] for t in unfavored) / len(unfavored)
        # These are cheap bets - need to win less often
        avg_cost = sum(t['cost_basis'] for t in unfavored) / len(unfavored)
        print(f"  {label} - Unfavored (p<0.50): {len(unfavored)} trades, {uw}W ({uw/len(unfavored)*100:.0f}%), "
              f"avg_p_side={avg_p:.3f}, avg_cost={avg_cost:.2f}, PnL=${up:+.2f}")

# Sigma analysis - are sigma-capped trades worse?
print("\n--- Sigma cap analysis (ALL trades) ---")
capped = [t for t in all_trades if t['sigma'] >= 7.9e-05]  # at or near 8e-05 cap
uncapped = [t for t in all_trades if 0 < t['sigma'] < 7.9e-05]
for label, group in [("Sigma at cap (>=8e-5)", capped), ("Sigma below cap", uncapped)]:
    if not group:
        continue
    w = sum(1 for t in group if t['won'])
    p = sum(t['pnl'] for t in group)
    avg_p = sum(t['p_side'] for t in group) / len(group)
    print(f"  {label}: {len(group)} trades, {w}W ({w/len(group)*100:.0f}%), "
          f"PnL=${p:+.2f}, avg_p_side={avg_p:.3f}")
