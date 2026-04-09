# Live trading restart marker — 2026-04-09

## Restart timestamp

**Clean restart: 2026-04-09T02:58Z** (UTC) — fresh state, fresh log, fresh $100 bankroll

(Earlier restart attempts at 01:54 / 02:33 were not clean. Old state and log
were preserved in `live_state_btc.20260409T025838Z.bak.json` and
`live_trades_btc.20260409T025838Z.bak.jsonl` for historical reference.)

Use this as the cutoff for "post-fix" PnL/PF analysis:

```bash
uv run python scripts/show_pf.py --since 2026-04-09T02:58
```

## State at clean restart (2026-04-09T02:58)

- Old state file backed up to `live_state_btc.20260409T025838Z.bak.json`
- Old trade log backed up to `live_trades_btc.20260409T025838Z.bak.jsonl`
- Fresh start: $100 bankroll, 0 trades, 0 PnL, 0 drawdown

For historical context, the backed-up state had:
- Final bankroll: $79.81
- Lifetime PnL: -$28.92 (over 34 lifetime trades)
- Peak bankroll: $111.23
- Max drawdown: $33.34 (30%)

## What changed in this restart

| File | Change | Purpose |
|---|---|---|
| `market_config.py:108` | btc 15m `max_trades_per_window: 2 → 1` | Stop in-window stacking that was doubling 15m losses (last session: two simultaneous 15m UP losses in same window) |
| `market_config.py:82` | btc 15m `min_sigma: 1e-5 → 2e-5` | Raise sigma floor from ~p5 to ~p25 of empirical distribution. Defends against sigma collapse causing p_model to pin at the z-cap (0.8413). |
| `tracker.py:637` | Diagnostic emit cadence `60s → 5s` | Dashboard now reflects model state in near-real-time. Was previously 30+ seconds stale, making it impossible to see what model was reacting to between snapshots. |
| `signal_diffusion.py` | New `edge_persistence_s` parameter (5s for 5m, 10s for 15m) | Edge must persist for N seconds before firing. Defends against fast-spike chasing where the model briefly crosses the edge threshold and then mean-reverts. Motivated by 00:35:50 fill investigation showing the bot fired ~10s after a sharp price move that the dashboard couldn't even see yet. |
| `live_trader.py:1131` | Pass `edge_persistence_s` per market | Activates the gate per timeframe. |

## Restart command (clean)

```bash
caffeinate -i uv run python live_trader.py --market btc --bankroll 100
```

**Notes**:
- NO `--resume` flag → fresh state, fresh PnL tracking, $100 starts honored.
- `--market btc` runs BOTH 5m and 15m → tests all 4 fixes simultaneously.
- `caffeinate -i` keeps the Mac awake while the bot runs.
- Both new gates are active: edge persistence (5s for 5m, 10s for 15m) +
  diagnostic cadence 5s (so dashboard stays in near-real-time).

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
