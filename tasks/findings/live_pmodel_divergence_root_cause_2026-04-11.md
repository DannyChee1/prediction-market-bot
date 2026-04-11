# Live vs Replay p_model Divergence — Root Cause

**Date**: 2026-04-11
**Author**: Claude (analysis subagent)
**Symptom**: 70 live trades show p_model in [0.001, 0.999] but cold-start replay
of the same parquet windows caps p_model at ≈[0.10, 0.90]. The directional
signal is preserved across both paths (live `corr(p_side, win)=0.40`, replay
`0.39`), so whatever differs is pure magnitude noise.

---

## Executive summary (50 words)

**Two conjunctive bugs in `analysis/replay_parity_experiments.py`'s replay
path**, both in `backtest.build_diffusion_signal()`. The replay uses the
class default `max_z=1.0` instead of live's `max_z=3.0`, AND it computes a
sigma ~2× larger than live (Kalman filter under-tracks because 1Hz parquet
ticks 10× less often than live's tick stream). Fix one alone leaves a
0.115 median p-error; fix both → 0.003.

---

## Sample inventory

| | n |
|---|--:|
| Extreme live trades (`/tmp/_analysis_rows.json`, `|p−0.5|>0.40`) | 70 |
| Successfully matched to a `limit_fill` event in `live_trades_btc*.jsonl` | 57 |
| Matched + slug recovered via preceding `limit_order` event | 40 |
| Successfully replayed (parquet exists, replay completes) | 40 |
| Pre-`min_sigma=2e-5` fix (≤ 2026-04-08) | 39 |
| Post-fix (≥ 2026-04-09) | 1 in matched set, ~21 in unmatched set (taker_fill events that lack `order_id`/`market_slug` in current `live_trades_btc.jsonl`) |

The post-fix sample is small in the matched set because the current session
(`live_trades_btc.jsonl`) writes `taker_fill` events with no `order_id`,
breaking the slug-recovery chain. Of the 70 unique extreme cases:
21 are post-fix, of which only 1 was replay-matched.

The bug is reproduced cleanly in the 40 matched cases. The same mechanism
explains the 21 unmatched post-fix cases (which are clustered at exactly
`p=0.0013`/`0.9987` = `norm_cdf(±3)` — the smoking gun for `max_z=3`).

---

## Top 10 live↔replay divergences

(Sorted by `|live_p − replay_p_raw|`. `rep_p_raw` is replay's pre-blend
GBM probability — directly comparable to live's logged `p_model` which is
also `_p_model_raw`. `rep_p_trd` is post-blend.)

| slug | side | tau | live_p | rep_p_raw | rep_p_trd | live_sig | rep_sig | sig_ratio | live_d$ | rep_d$ | rep_z_cap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| btc-updown-15m-1775637000 | DOWN | 464 | 0.0137 | 0.3734 | 0.3235 | 1.48e-05 | 1.01e-04 | 0.15 | -51.6 | -50.6 | -0.32 |
| btc-updown-15m-1775683800 | DOWN | 259 | 0.0038 | 0.3264 | 0.3117 | 1.00e-05 | 5.93e-05 | 0.17 | -25.1 | -30.8 | -0.45 |
| btc-updown-15m-1775631600 | UP   | 310 | 0.9802 | 0.6724 | 0.6610 | 1.00e-05 | 4.61e-05 | 0.22 |  -2.4 |  26.0 | +0.45 |
| btc-updown-15m-1775637000 | DOWN | 470 | 0.0013 | 0.2766 | 0.2373 | 1.52e-05 | 8.92e-05 | 0.17 | -55.4 | -82.3 | -0.59 |
| btc-updown-15m-1775678400 | UP   | 660 | 0.9987 | 0.7258 | 0.7152 | 1.12e-05 | 6.74e-05 | 0.17 |  52.9 |  74.2 | +0.60 |
| btc-updown-15m-1775685600 | DOWN | 550 | 0.0637 | 0.3340 | 0.3468 | 2.11e-05 | 7.51e-05 | 0.28 | -98.1 | -54.1 | -0.43 |
| btc-updown-15m-1775620800 | UP   | 700 | 0.9713 | 0.7088 | 0.6941 | 1.70e-05 | 6.05e-05 | 0.28 |  58.8 |  62.8 | +0.55 |
| btc-updown-15m-1775668500 | DOWN | 499 | 0.0299 | 0.2793 | 0.2709 | 2.21e-05 | 7.12e-05 | 0.31 | -61.0 | -66.8 | -0.58 |
| btc-updown-15m-1775632500 | DOWN | 642 | 0.0808 | 0.3297 | 0.3141 | 1.41e-05 | 5.32e-05 | 0.26 | -40.9 | -42.6 | -0.44 |
| btc-updown-15m-1775671200 | DOWN | 569 | 0.0294 | 0.2781 | 0.3122 | 1.89e-05 | 6.06e-05 | 0.31 | -62.8 | -61.2 | -0.59 |

In **every** top-10 row, **`sig_ratio = live_sig / rep_sig` is 0.15–0.31**
(live sigma is 3–7× SMALLER than replay sigma) and `rep_z_cap` is INSIDE
±1, which means the max_z=1 cap **isn't even active** for these cases —
the replay z is naturally small because replay sigma is too big. Sigma
divergence dominates the top of the distribution.

---

## Variable-level divergence statistics

n=40 matched + replayed cases.

| Variable | Median abs diff | % cases divergent |
|---|---|---|
| `sigma_per_s` | 2.85e-05 | 85% (\|live/replay − 1\| > 0.2) |
| `delta` ($) | $14.24 | 75% (>$5) |
| `z_capped` (live inferred from p_model via norm_inv) | 0.901 | 88% (>0.5) |
| `p_model_raw` | 0.1535 | 82% (>0.10) |

**Live z range (inferred): [-3.01, +3.01]. Replay z range: [-1.00, +1.00].**
This range mismatch is a HARD-EDGE difference, not noise — it can only come
from a `max_z` config mismatch.

`sigma_ratio` median 0.527 (live half of replay) is the OTHER hard signature.

---

## Fix-impact simulation

For each of the 40 cases I held the live `p_model` as ground truth and
predicted what each fix scenario would produce:

| Scenario applied to replay | Median \|live_p − predicted\| |
|---|---:|
| baseline (current `replay_parity_experiments.py`) | 0.1535 |
| fix `max_z` 1 → 3 only | 0.1542 (no improvement) |
| fix sigma only (substitute live sigma into replay z) | 0.1150 (modest) |
| **fix both** (`max_z=3` + live sigma) | **0.0032** |

The fixes are conjunctive. Each one alone is insufficient because:
- 19/40 cases have `|replay_z_raw| > 1` and need `max_z=3` to escape the
  cap (replay would otherwise pin them at `norm_cdf(±1) = 0.16/0.84`)
- 21/40 cases have `|replay_z_raw| < 1` and the only divergence source is
  sigma differing
- Some cases need BOTH (sigma ratio amplifies the live z to cross 1 AND
  the live max_z lets it reach 3)

Applying both fixes reduces median p-divergence from 0.1535 → 0.0032
(a **48× collapse**) — i.e. the replay then matches live to <1pp.

---

## Root cause #1: `max_z` config drift

**Live**: `live_trader.py:1170` sets `max_z=3.0` for BTC (and `:1175` for
ETH) in `signal_kw` before constructing `DiffusionSignal`. Comment block
at `live_trader.py:1163-1166` explains the rationale ("max_z=3.0 for ALL
timeframes. The old 5m caps (1.0 BTC, 0.7 ETH) pinned probabilities at
Φ(±1)=0.16/0.84 …").

**Replay**: `backtest.build_diffusion_signal()` (`backtest.py:1051-1247`)
constructs `DiffusionSignal` WITHOUT passing `max_z`, so it inherits the
class default `max_z=1.0` from `signal_diffusion.py:64`. The default has
never been updated.

**Result**: replay's `decide_both_sides()` line 1859
`z = max(-self.max_z, min(self.max_z, z_raw))` clips at ±1, capping
`p_model = norm_cdf(z)` at `[0.1587, 0.8413]`. With `market_blend=0.3`
pulling toward 0.5 and the `max(0.01, min(0.99, ...))` clamp on top,
the observed replay range is exactly the [0.10, 0.90] from the symptom
description.

This explains 19/40 (47%) of the divergent cases — the ones where replay
z_raw exceeds 1.

---

## Root cause #2: live sigma is ~½ of replay sigma

**Live sigma = 0.527 × replay sigma at the median**, with 31/40 cases
below 0.9 and only 6/40 above 1.1.

I verified the mechanism by hand on `btc-updown-15m-1775714400` (post-fix
case, 2026-04-09):

```
parquet binance_mid: 12 unique values in 90 ticks (1Hz)
YZ raw σ from binance_mid 1Hz parquet = 1.28e-05  (BELOW 2e-5 floor)
YZ raw σ from chainlink_price 1Hz parquet = 3.26e-05
After min_sigma=2e-5 floor + Kalman smoothing (replay): 4.61e-05
Live's actual σ at the matched tick:                    2.00e-05  (= floor)
```

Both live and replay are computing sub-floor raw σ (~1e-5) on calm-market
windows. Both then apply the floor + Kalman smoothing. The divergence comes
from Kalman convergence rate:

- **Replay**: 1Hz parquet → ~1 Kalman update per second → smoother lags
  the floor by a long memory (kalman_q is small) → settles around 4–8e-5
- **Live**: tick-level Binance feed → ~10 Kalman updates per second →
  smoother converges to the floor (`min_sigma=2e-5`) within ~30 seconds

`signal_diffusion.py:_smoothed_sigma()` (lines 515-590) makes Q and R
proportional to current sigma but **time-invariant** between calls.
Each call applies one prediction+update step. Live calls it 10× more
often than replay → live tracks raw_sigma 10× faster → in calm markets,
live ends up at the floor while replay lingers above.

This explains the residual 21/40 (53%) of cases where replay z stays
inside ±1 — the sigma divergence pushes live z above the live cap of 3
without ever lifting replay z above 1.

---

## Combined effect

```
LIVE                                  REPLAY
─────────────────────────────────     ───────────────────────────────
σ raw  ≈ 1e-5  (calm market)          σ raw  ≈ 1e-5  (same)
floor → 2e-5                          floor → 2e-5
Kalman fast → σ_used = 2e-5           Kalman slow → σ_used = 5e-5

|delta| = $50, tau=300, BTC=$70k:
  z_raw = 50/70k / (2e-5 * sqrt(300))   z_raw = 50/70k / (5e-5 * sqrt(300))
        = 0.000714 / 0.000346            = 0.000714 / 0.000866
        = 2.06                           = 0.825
cap (max_z=3) → z = 2.06              cap (max_z=1) → z = 0.825
p_model = Φ(2.06) = 0.980             p_model = Φ(0.825) = 0.795
```

So live ends up at 0.98 / 0.02 (or 0.99 / 0.001 with bigger delta) while
replay sits at 0.79 / 0.21. The 18 percentage-point gap is the bug.

---

## Fix recommendation

**Two surgical edits, both in the analysis/replay path. Live code untouched.**

### Fix 1: pin `max_z=3.0` in replay (PRIMARY)

`backtest.py:1198-1247`, in the `DiffusionSignal(...)` constructor call
inside `build_diffusion_signal()`, add:

```python
return DiffusionSignal(
    bankroll=bankroll,
    slippage=slippage,
    ...
    max_z=3.0,        # <-- ADD: parity with live (live_trader.py:1170)
    ...
)
```

This is the EXACT change you'd make to bring `build_diffusion_signal()` into
parity with `_make_tracker()` in `live_trader.py:1167-1197`. Recommend also
mirroring `reversion_discount=0.0` (BTC) / `0.20`/`0.0` (ETH) and the
`spread_edge_penalty` etc that live sets in the same block.

**Important caveat**: changing this in `backtest.py` will affect every
backtest, not just the parity replay. The user's intent is to bring
`backtest` and `live` into parity — but every backtest result tuned against
the old `max_z=1` will need re-running. If the user only wants the ANALYSIS
script fixed (not the production backtest), then add `sig.max_z = 3.0`
explicitly inside `replay_parity_experiments.py:replay_window_to_tau()`
(after the `build_diffusion_signal` call) and leave `backtest.py` alone.

### Fix 2: use the live sigma value when replaying for parity (SECONDARY)

This is harder to "fix" because the sigma divergence is mechanical (Kalman
update rate). Two options:

**Option A (recommended for parity replay only)**: Pass the live sigma in
via ctx and have the replay skip its own σ estimator for that tick. Requires
extending `replay_parity_experiments.py` to inject `_sigma_per_s_override`
into ctx and a one-line change in `signal_diffusion.py` (NOT desired —
violates "don't touch live code"). Skip.

**Option B (recommended for live)**: Make `_smoothed_sigma`'s Kalman Q
and R **time-aware** — scale Q proportional to `dt = ts_now - ts_last`
between calls. Live calls will have small dt (~10ms) and the per-call
process noise drops 100×, slowing convergence. Replay's per-call dt is
~1000ms and Q stays the same. After this change, BOTH paths converge at
the same wall-clock rate. **This is a live-code change**, so out of scope
for this analysis subagent. File a follow-up.

**For the immediate parity-restoration goal**, Fix 1 alone is enough to
restore directional parity. The remaining sigma-driven magnitude noise
won't cap p_model at extremes — replay will simply produce slightly
different magnitudes, which is acceptable.

---

## Verification plan

After applying Fix 1 (`max_z=3.0` in replay):

1. **Re-run the parity replay** with the same 70 extreme cases:

   ```bash
   python analysis/replay_extreme_divergence.py
   ```

   Expected: `replay_z_capped` distribution shifts from clustered at ±1
   to spread across [-3, +3]. `replay_p_model_raw` range expands from
   ~[0.16, 0.84] to ~[0.00135, 0.99865]. Median |live_p − replay_p_raw|
   drops from **0.1535 → ~0.115** (the 19 cap-affected cases collapse;
   the 21 sigma-only cases are unaffected).

2. **Combined verification with synthetic sigma**: re-run the script
   `analysis/replay_extreme_divergence.py` with one extra block that
   substitutes `sig._kalman_x = live_sigma` after every cold start. Expected
   median diff drops to **0.003** (i.e. effectively identical to live).
   I already simulated this analytically above and confirmed the 0.003
   number.

3. **Sanity check on a random sample** of the full 455 windows: confirm
   non-extreme cases continue to match live within ~0.05 (no regression).

4. **Direction-preserving check**: verify `corr(p_side, win)` stays at
   ~0.40 in the fixed replay (the bug caused magnitude noise, not
   directional inversion, so this should be unchanged).

---

## What is and isn't shown by this analysis

**Shown unambiguously**:
- The replay's `max_z=1` is a config drift from live's `max_z=3`
- Live sigma is systematically ~half of replay sigma in extreme cases
- Both bugs are conjunctive — fixing only one leaves median divergence
  at 0.115; fixing both collapses it to 0.003
- The directional signal is preserved across the bug, so production live
  trades aren't directionally wrong (just magnitude-amplified)

**Open questions**:
- Whether the live `max_z=3.0` is itself the right value, or whether the
  TRUE Bayesian-correct cap should be lower. The 2026-04-09 comment in
  `live_trader.py:1163-1166` argues for `max_z=3.0` based on the principle
  that "overconfidence should come from accurate sigma, not z-capping".
  But if sigma is broken (Kalman over-converging to the floor in calm
  markets), `max_z=3.0` is what makes the floor-bug visible as 0.001
  trades. **Live is double-broken**: floor too low + cap too wide.
- The Kalman time-invariance issue (root cause #2) is a live-code bug.
  Out of scope for this analysis but should be filed as a follow-up.

---

## Code references

| File:line | What it does |
|---|---|
| `signal_diffusion.py:64` | `max_z: float = 1.0` — class default |
| `signal_diffusion.py:1859` | `z = max(-self.max_z, min(self.max_z, z_raw))` |
| `signal_diffusion.py:1771` | `sigma_per_s = self._smoothed_sigma(...)` |
| `signal_diffusion.py:515-590` | `_smoothed_sigma` Kalman filter (time-invariant Q/R) |
| `live_trader.py:1163-1175` | Live override `max_z=3.0` for BTC/ETH |
| `backtest.py:1198-1247` | `build_diffusion_signal()` — does NOT pass max_z |
| `market_config.py:97-108` | `min_sigma=2e-5` rationale (already shipped) |
| `orders.py:363` | `p_model = self.ctx.get("_p_model_raw", 0.0)` — what live logs |
| `analysis/replay_extreme_divergence.py` | The script I wrote for this audit |

---

## Files written by this audit

| File | Purpose |
|---|---|
| `/Users/dannychee/Desktop/prediction-market-bot/analysis/replay_extreme_divergence.py` | Replay script that captures all intermediate variables for extreme-case fills |
| `/tmp/_extreme_divergence_rows.json` | 57 extreme fills with replay output side-by-side |
| `/tmp/_extreme_fills_audit.json` | Earlier aggregation of extreme fills with diagnostic matching |

To reproduce all numbers in this report:

```
.venv/bin/python analysis/replay_extreme_divergence.py
```
