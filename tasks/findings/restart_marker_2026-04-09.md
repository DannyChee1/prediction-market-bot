# Live trading restart marker — 2026-04-09

## Restart timestamp

**~2026-04-09T01:54:25Z** (UTC)

This is the moment we restarted the bot after applying four targeted fixes.
Use this as the cutoff for "post-fix" PnL/PF analysis:

```bash
uv run python scripts/show_pf.py --since 2026-04-09T01:54
```

## State at restart

From `live_state_btc.json` (saved 2026-04-09T00:01):
- Bankroll: **$80.84**
- Initial bankroll (this run): $104.54
- Lifetime PnL: -$27.89
- Lifetime trades: 29
- Wins: 13, Losses: 16
- Peak bankroll: $111.23
- Max drawdown: $30.39 (27.3%)

(Restarting with `--resume` continues this state.)

## What changed in this restart

| File | Change | Purpose |
|---|---|---|
| `market_config.py:108` | btc 15m `max_trades_per_window: 2 → 1` | Stop in-window stacking that was doubling 15m losses (last session: two simultaneous 15m UP losses in same window) |
| `market_config.py:82` | btc 15m `min_sigma: 1e-5 → 2e-5` | Raise sigma floor from ~p5 to ~p25 of empirical distribution. Defends against sigma collapse causing p_model to pin at the z-cap (0.8413). |
| `tracker.py:637` | Diagnostic emit cadence `60s → 5s` | Dashboard now reflects model state in near-real-time. Was previously 30+ seconds stale, making it impossible to see what model was reacting to between snapshots. |
| `signal_diffusion.py` | New `edge_persistence_s` parameter (5s for 5m, 10s for 15m) | Edge must persist for N seconds before firing. Defends against fast-spike chasing where the model briefly crosses the edge threshold and then mean-reverts. Motivated by 00:35:50 fill investigation showing the bot fired ~10s after a sharp price move that the dashboard couldn't even see yet. |
| `live_trader.py:1131` | Pass `edge_persistence_s` per market | Activates the gate per timeframe. |

## Restart command

```bash
uv run python live_trader.py --market btc_5m --resume
```

**Note**: deliberately running `btc_5m` only (no 15m) for first session post-fix.
The 15m market lost -$27.18 (-50% ROI) in the previous session. Even with the
fixes targeted at 15m's failure mode, isolating to 5m gives a cleaner test of
whether the underlying strategy + my changes can produce a positive session.

## What to watch for

**Phase 1 success criteria** (next 6-12 hours of trading):
1. **5m PF > 1.0** sustained over a few hours of trading (~30+ trades)
2. **Bankroll trending up**, not just sideways
3. **Edge persistence gate firing** in FLAT bucket reasons (proves it's active)
4. **0.8413 cap-hit frequency dropping** on diagnostic snapshots (proves min_sigma fix is working — but note 5m's floor was NOT raised, so the cap may still hit on 5m even after restart)

**Phase 1 failure → escalate**: If 5m PF stays <1.0 after ~6 hours of trading,
move to Phase 2:
- Bump btc_5m `min_sigma: 1e-5 → 2e-5` (mirror what we did for 15m)
- Bump `edge_threshold: 0.06 → 0.08` (require more conviction)

## Known unaddressed issues at this restart

1. **Live vs backtest divergence not fixed.** Backtest still uses idealized maker fills with zero latency. Live still has adverse selection. The edge persistence gate partially addresses this but doesn't close the gap.
2. **btc_5m min_sigma still at 1e-5.** Will need bumping if cap continues to hit constantly on 5m.
3. **`max_z=1.0` cap unchanged.** Will only revisit if floor fixes don't reduce cap-hits enough.
4. **Adverse selection in maker mode is structural.** No real fix until we either (a) quote less aggressively or (b) bump edge_threshold significantly.

## Next checkpoint

**~2026-04-09T07:54Z** (6 hours post-restart) — pull `show_pf.py --since 2026-04-09T01:54` and decide whether Phase 2 is needed.
