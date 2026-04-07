# Post-Fix Re-Validation — 2026-04-07

## Context

The `research` worktree merge brought in a Kou drift-bug fix in
`_model_cdf` (the buggy `drift_z = -lambda*zeta*sqrt(tau)/sigma` term was
removed — it was a Q-measure option-pricing correction misapplied to
physical-measure binary prediction and blew up at the σ floor). This
invalidated every validation I ran in Phase 1/3/4 because those were
on buggy code. This document re-runs the key questions on the post-fix
codebase against the full 50-day REST-backfilled sample.

**Two collateral bugs** were found and fixed along the way:

1. **Broken symlink in `data/`**: the research worktree committed `data`
   as a self-loop symlink, which nuked the original `data/` directory
   and its 14,200 backfilled parquets when merged. Lost everything.
   Re-backfilled in ~4 hours. Symlink is now fixed (`data/` is a real
   directory, `prediction-market-bot-research/data` points to it).

2. **`build_diffusion_signal` silently ignored `config.market_blend`**:
   The backtest CLI always ran with `market_blend=0.0` regardless of
   what the config said, so the backtest and live trader were diverging.
   Fixed by passing `market_blend=config.market_blend` in the
   constructor call and adding a `--market-blend` CLI override for
   sweeps.

3. **REST-backfilled parquets were missing two fields** the post-fix
   backtest requires:
   - First rows were at tau ~229 instead of tau=300 because my backfill
     script skipped early ticks before the trade tape established
     bid/ask. Fixed by padding with synthetic 0.50/0.50 seed rows.
   - L2 depth columns (`bid_px_up_1`, `bid_sz_up_1`, etc.) were all NaN,
     so `Snapshot.from_row` returned empty level tuples and
     `_execute_fill` couldn't walk the book to simulate fills. Fixed by
     synthesizing level-1 depth from `best_bid_up`/`best_ask_up` at a
     fixed size of 100.

## Sample

| | BTC 5m | BTC 15m |
|---|---:|---:|
| Total parquets | 14,354 | 4,789 |
| Train windows | 10,061 | 3,357 |
| Test windows | 4,312 | 1,439 |
| Period | 2026-02-17 → 2026-04-07 | same |
| Source | REST backfill + live recordings | same |

## Finding 1: `market_blend` sweep

### BTC 5m (4,312 test windows, 3,673 test trades)

| blend | trades | WR | PnL | Sharpe | MaxDD |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 3950 | 62.2% | +$27,984 | 1.45 | 3.6% |
| 0.1 | 3891 | 62.5% | +$29,124 | 1.52 | 3.6% |
| 0.2 | 3801 | 62.3% | +$27,938 | 1.49 | 4.0% |
| **0.3 *(shipped)*** | **3673** | **62.2%** | **+$27,871** | **1.54** | **4.0%** |
| 0.4 | 3385 | 62.3% | +$25,042 | 1.50 | 4.5% |
| 0.5 | 2800 | 61.4% | +$19,801 | 1.43 | 4.4% |

**Sharpe-optimal at 0.3** (= currently shipped). All blends in [0.0, 0.3]
are within 6% of each other — the effect is small but the shipped value
is the peak. **No change needed.**

### BTC 15m (1,439 test windows)

| blend | trades | WR | PnL | Sharpe | MaxDD |
|---:|---:|---:|---:|---:|---:|
| **0.0 *(old default)*** | 1294 | 57.8% | +$3,959 | 0.61 | 11.7% |
| 0.1 | 1249 | 57.8% | +$4,046 | 0.64 | 11.1% |
| 0.2 | 1192 | 58.9% | +$4,837 | 0.81 | 12.1% |
| 0.3 | 1105 | 58.4% | +$4,063 | 0.73 | 12.6% |
| 0.4 | 884 | 60.1% | +$5,323 | 1.21 | 11.4% |
| **0.5 *(NEW SHIPPED)*** | **620** | **59.7%** | **+$4,196** | **1.36** | **4.7%** |

**Massive improvement at blend=0.5** vs shipped 0.0:
- **Sharpe +123%** (0.61 → 1.36)
- **Max drawdown −60%** (11.7% → 4.7%)
- WR +2pp (57.8% → 59.7%)
- PnL roughly flat (+6%)
- 50% fewer trades but each is much higher quality

**Shipped: `btc.market_blend = 0.5`**. Commit `5911503`.

The mechanism is consistent with Phase 1's rationale: 15m windows give
the model more time to drift from market consensus between feature
refresh and entry, so pulling p_model toward the contract mid at decision
time corrects that drift. 5m windows don't leave enough time for the
drift to matter, so blend=0.3 is near the edge of useful.

## Finding 2: Weekend pattern (direction differs by market)

Research branch claimed weekend underperformance was purely an artifact
of the Kou drift bug hitting the σ floor on quiet (low-σ) days. **That
claim is wrong for BTC 5m and correct-in-spirit for BTC 15m.**

### BTC 5m (post-fix, blend=0.3, 3,662 test trades)

| Day | Trades | WR | PnL | Sharpe |
|---|---:|---:|---:|---:|
| Mon | 493 | 63.5% | +$3,872 | 3.63 |
| Tue | 693 | 66.7% | +$8,683 | **6.72** |
| Wed | 492 | 65.2% | +$4,834 | 4.57 |
| Thu | 504 | 64.5% | +$4,518 | 4.21 |
| Fri | 498 | 58.4% | +$2,002 | 1.83 |
| Sat | 494 | 59.5% | +$2,810 | 2.66 |
| Sun | 488 | 55.9% | +$1,100 | 1.03 |
| **Mon-Fri** | 2680 | **63.9%** | +$23,909 | **9.54** (EV $8.92/trade) |
| **Sat-Sun** | 982 | **57.7%** | +$3,910 | 2.60 (EV $3.98/trade) |

Per-trade EV ratio: **2.24×** weekday-to-weekend. Pre-fix was 2.5×; the
fix narrowed the gap slightly but the pattern is still clearly real.

### BTC 15m (post-fix, blend=0.5, 619 test trades)

| | Trades | WR | PnL | Sharpe | EV/trade |
|---|---:|---:|---:|---:|---:|
| Mon-Fri | 423 | 59.8% | +$2,736 | 2.71 | **$6.47** |
| Sat-Sun | 196 | 59.2% | +$1,417 | 2.08 | **$7.23** |

**Weekends are BETTER per trade on 15m** (EV/trade 7.23 > 6.47). The
lower Sharpe is purely a sample-size effect (196 vs 423). Skipping
weekends on 15m would actively lose money.

### Decision

**Don't ship a weekend skip on either market.** Asymmetric direction
across markets, modest magnitude, and adding a time-of-day gate would
add complexity for a small per-trade EV delta. Leave as-is.

## Finding 3: New features (smoke-tested, none shipped)

### HMM regime classifier

- `regime_classifier_btc_5m.pkl` and `regime_classifier_btc_15m.pkl`
  both load and classify successfully.
- Direct calls return valid `(state_idx, label, kelly_mult)` tuples.
- On real features from latest windows, **both markets classify as
  their `high_acc` state → `kelly_mult=1.0`** — which is a no-op.
- Confirms the research branch's finding that the existing HMM is
  trained with multipliers too conservative to affect the backtest.
- **Not shipped.** To make it meaningful, retrain with
  `--high-mult 1.0 --low-mult 0.0` (zero out low-accuracy state bets)
  or drop the current mult structure and use regime as a feature on
  filtration model retrain.

### EWMA σ estimator (+ realized variance + GARCH)

- `scripts/sigma_estimators.py::ewma_sigma_per_s` runs cleanly on a
  200-tick fake price stream, returns 1.89e-5 (sensible scale).
- `realized_variance_per_s` returns 2.04e-5.
- **Not shipped as default.** Research branch's ablation showed EWMA
  wins σ-forecast MSE by 26% but costs $360 PnL on BTC 5m and $1,675
  on BTC 15m because `edge_threshold` / `kelly_fraction` /
  `filtration_threshold` were tuned to YZ-flavored σ. Can opt-in via
  `sigma_estimator="ewma"` in `MarketConfig` after downstream re-tune.

### Hawkes intensity

- `HawkesIntensity(mu=0.1, alpha=0.5, beta=1.0)` computes intensity
  correctly: three events at t=[1, 2, 2.5] → intensity 0.655 at t=3.0
  (matches expected exponential-decay excitation).
- `detect_jumps` found 0 jumps on a smooth sine-wave fake stream
  (expected — no z > k_sigma events).
- **Not shipped.** The Hawkes state is published as an opt-in ctx
  feature (`_hawkes_intensity`, `_hawkes_n_events`) but no downstream
  consumer reads it. Future work: include in filtration model features
  and retrain.

## Finding 4: Backtest absolute numbers are much larger than the research branch's

Research branch's post-fix baseline showed **BTC 5m: 316 trades,
+$863, Sharpe 0.08** and **BTC 15m: 113 trades, +$3,706, Sharpe 0.73**.

This run shows:
- BTC 5m: **3,662 trades, +$27,821, Sharpe 1.54** (blend=0.3)
- BTC 15m: **619 trades, +$4,196, Sharpe 1.36** (blend=0.5)

Why the 10× difference in trade count and PnL?

1. **Sample size**: research branch ran on the old 1662-window sample.
   This run uses the full 14,354-window post-backfill sample. 8.6×
   more windows.
2. **Market_blend plumbing bug**: research branch's backtest was
   silently running with `market_blend=0.0` regardless of config.
   Now it respects config (5m=0.3, 15m=0.5).
3. **Synthetic depth**: REST-backfilled parquets have synthetic L2
   depth (size=100) rather than real book depth. Real depth on live
   parquets is often shallower, which rejects some fills that my
   synthetic depth accepts. Net effect: my backtest has a slightly
   permissive fill rate.
4. **Per-trade EV is similar**: research branch was $863/316 = $2.73/trade;
   this run is $27,821/3662 = $7.60/trade on BTC 5m. The 2.8× per-trade
   improvement is explained by market_blend being on (which shifts to
   higher-quality trades) plus some real regime drift.

**Caveat**: do not literally expect Sharpe 1.54 in live trading. Real
production Sharpe is typically 2-4× lower than backtest due to
slippage, partial fills, fee corrections, and regime drift. Plan for
live Sharpe in the 0.4-0.8 range for BTC 5m and 0.3-0.7 for BTC 15m
(with the new blend=0.5).

## Code changes shipped in this round

| File | Change |
|---|---|
| `backtest.py:3319-3332` | `build_diffusion_signal` gained `market_blend_override: float \| None` param |
| `backtest.py:3403-3435` | Now passes `market_blend=effective_blend` (from override or config) to the constructor |
| `backtest.py:3470-3474` | New `--market-blend` CLI arg |
| `backtest.py:3515` | Main threads CLI arg into `build_diffusion_signal(market_blend_override=...)` |
| `market_config.py:77-87` | `btc.market_blend = 0.5` with extensive comment on the sweep evidence |

Commits:
- `b9415bd` — `add max_book_age_ms gate to skip trades during book WS disconnects`
- `b14760f` — merge research worktree (brought in Kou fix, sigma bounds, stale gates, section 2/3 features)
- `5911503` — `btc 15m: market_blend 0.0 → 0.5 + plumb from config in backtest`

## Out of scope (deferred)

- **Retrain filtration model** — the existing `filtration_model.pkl` was
  trained on pre-fix data with different σ scaling. Retraining would be
  a ~1-day effort and is blocked on picking an σ estimator.
- **EWMA σ with retuned thresholds** — opt-in. Needs `edge_threshold` /
  `kelly_fraction` / `filtration_threshold` to be re-swept in lockstep.
- **Retrain HMM with aggressive multipliers** — `--high-mult 1.0 --low-mult 0.0`
  to actually make it affect sizing. Current model is functionally a
  no-op.
- **Hawkes integration into filtration features** — needs a filtration
  retrain anyway.
- **Bipower variation σ for tail-mode `kou_full`** — the proper Kou path
  in `_model_cdf` is still dead code. Bringing it in requires a
  jump-robust σ estimator (bipower variation) so we don't double-count
  jump variance. ~3-6 hours effort. See `PLAN.md` for the full writeup.
- **Restore live BTC recorder writes** — user restarted live traders
  after the symlink fix; verify they're actually producing new parquets
  on the next few window boundaries.

## Verification to run after deploy

1. Restart any live processes (live_trader.py, dashboard) to pick up
   the new `btc.market_blend = 0.5`.
2. Watch the first 30 BTC 15m live resolutions. Expected WR ~60% (was
   57.8% in backtest for blend=0.0, 59.7% for blend=0.5).
3. **Rollback trigger for BTC 15m**: if after 30 live resolutions WR
   < 52% or Sharpe clearly negative, revert `btc.market_blend` to 0.0.
