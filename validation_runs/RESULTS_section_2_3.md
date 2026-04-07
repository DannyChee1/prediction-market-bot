# Section 2 + 3 Implementation & Validation Results

Date: 2026-04-07

This is the follow-up to `validation_runs/RESULTS.md`, which covered the
correctness bug fixes (Bugs A/B/C). This report covers the modeling
improvements (Section 2) and the validation tests (Section 3) from the
original analysis.

## Headline result

**Every Section-2 modeling improvement is now built and tested, but only
two are wired in by default.** The rest are intentionally opt-in because
ablation showed that flipping them on naively *hurts* backtest PnL —
not because the improvement is wrong, but because downstream
hyperparameters (`edge_threshold`, `kelly_fraction`, `filtration_threshold`)
were tuned to the legacy implementations and need to be re-tuned in
lockstep before opt-in pays off.

**Every Section-3 validation test is built, exercised on real data, and
its findings documented below.** The tests surface several non-obvious
results — most importantly that BTC 5m's apparent positive PnL is fragile
and BTC 15m's edge is real but small.

The post-fix **default** baseline (which the user already saw in
`RESULTS.md`) is preserved exactly:

| | BTC 5m | BTC 15m |
|---|---:|---:|
| Trades (test) | 316 | 113 |
| Win rate | 58.5% | 63.7% |
| Total PnL | +$863 | +$3,706 |
| Sharpe | 0.08 | 0.73 |
| Max DD | 42.1% | 29.0% |

(BTC 15m PnL went from +$3,663 → +$3,706 because the data-driven hourly
priors (Section 2 / item 27) take effect on cold-start σ. +1.2% on the
same sample.)

---

## Section 2 — Modeling improvements

| # | Item | Built | Wired in by default | Notes |
|---|---|---:|---:|---|
| 2a | GARCH(1,1) σ + EWMA σ + RV σ + A/B framework | ✅ | ❌ (opt-in) | EWMA wins σ-forecast A/B by 26-39% but hurts PnL when swapped in unilaterally — see ablation |
| 2b | Hawkes-modulated jump intensity | ✅ | ❌ (opt-in) | Published as `_hawkes_intensity` ctx feature; downstream consumers must opt in via `hawkes_params` tuple |
| 2c | HMM regime classifier wired into live signal | ✅ | ✅ (loads pkl if present) | Trains a 2-state HMM per market; multiplies `kelly_fraction` by per-state mult. Empirically a no-op on the current backtest because the chosen states don't differ enough in accuracy to matter — see below |
| 2d | Kalman OBI smoothing | ✅ | ❌ (opt-in via `use_kalman_obi`) | Smoothing hurts BTC 5m by ~$3.7k in ablation because it lets in trades the raw gate filtered |
| 2e | Filtration warmup feature parity (mid_momentum) | ✅ | ❌ (opt-in via `mid_momentum_parity`) | Logically correct but breaks the existing `filtration_model.pkl` calibration — needs filtration retrain to deploy |
| — | Data-driven hourly vol priors | ✅ | ✅ | Picks up `data/<market>/hourly_priors.json` automatically; falls back to hardcoded constants |

### 2a — Sigma estimators

**Built:** `scripts/sigma_estimators.py` provides three jump-robust σ
estimators (`realized_variance_per_s`, `ewma_sigma_per_s`,
`garch11_sigma_per_s`) plus `fit_garch11()` (coarse-grid QML).
`scripts/validate_sigma_estimators.py` runs an A/B test on real parquet
windows: walk train→test, fit GARCH on train, compute 1-step forecast
MSE on test for all four estimators.

**A/B result** (`validation_runs/sigma_estimators/`):

| Market | YZ rmse | EWMA rmse | RV rmse | GARCH rmse | Winner | Improvement vs YZ |
|---|---:|---:|---:|---:|---|---:|
| BTC 15m | 3.43e-05 | 2.79e-05 | 2.67e-05 | 3.02e-05 | RV | +39.4% |
| BTC 5m  | 4.46e-05 | 3.85e-05 | 3.85e-05 | 4.05e-05 | EWMA | +25.6% |

YZ is dead last on both markets. The "open-to-prev-close" component of
Yang-Zhang assumes overnight gaps that don't exist on continuously-traded
crypto, so it's been adding noise.

**Wired into the live model:** Yes, via the new `sigma_estimator` field on
`MarketConfig` (default `"yz"`) and the new `_compute_vol` dispatch. Switch
on by adding `sigma_estimator="ewma"` to a `MarketConfig` literal.

**Why default is still YZ:** Ablation
(`validation_runs/ablation_btc_5m.json`,
`validation_runs/ablation_btc.json`):

```
BTC 5m ablation:
  baseline (yz):   316 tr  WR 58.5%  PnL +$863
  +ewma:           285 tr  WR 56.8%  PnL -$3,207
BTC 15m ablation:
  baseline (yz):   113 tr  WR 63.7%  PnL +$3,706
  +ewma:           106 tr  WR 62.3%  PnL +$2,001
```

EWMA gives a *better* σ forecast but a *worse* trading result, because
`edge_threshold`, `kelly_fraction`, and `filtration_threshold` were tuned
against YZ-flavoured σ. Switching σ shifts the entire distribution of z
and breaks downstream calibration.

**Recommendation:** When you want to enable EWMA, do it together with a
re-tune of `edge_threshold` (it should drop) and a re-train of the
filtration model (its features include `sigma` and `vol_regime_ratio`,
both of which change distribution under EWMA).

### 2b — Hawkes intensity

**Built:** `scripts/hawkes.py` provides `HawkesIntensity` (online stateful
tracker), `fit_hawkes_mle()` (offline grid-search MLE for the exponential
kernel), and `detect_jumps()` (turns a price stream + σ into event times
where |z| > k_sigma).

**Wired in:** As an *opt-in published feature*, not as a gate.
`DiffusionSignal` accepts a `hawkes_params=(mu, alpha, beta, k_sigma)`
tuple. When set, it maintains a per-window `HawkesIntensity` instance
fed by the realized return stream and publishes `_hawkes_intensity` and
`_hawkes_n_events` to ctx every tick. Downstream consumers (filtration
model, dashboard, sizing rule) can read it.

**Why default-off:** The user already has filtration features and a
stable baseline. Surfacing a new feature without retraining the
filtration model can't help — it can only push the trained model
out of distribution. Wire it in via the filtration features set the
next time the filtration model is retrained.

**Test:** `tests/test_model_cdf.py::test_hawkes_intensity_decay_and_excitation`
verifies the online recursion (event boosts λ, λ decays toward μ),
plus `test_hawkes_fit_recovers_params_on_synthetic` confirms that
`fit_hawkes_mle` finds parameters that beat the no-excitation baseline
on a Hawkes-generated synthetic stream.

### 2c — HMM regime classifier

**Built:** `regime_classifier.py` wraps an `hmmlearn.GaussianHMM` +
`StandardScaler` + state metadata in a class that mirrors
`FiltrationModel` (load/save semantics, training-to-inference flow).
Features must match `analysis/hmm_regime.py:window_features` exactly:
`[log_sigma, vol_regime, z_early_abs, hour_sin, hour_cos]`. The
classifier exposes `classify_window(features) -> (state_idx, label, kelly_mult)`.

`scripts/train_regime_classifier.py` fits a 2-state HMM on a market's
parquet windows, characterises each state by held-out signal accuracy,
and assigns a `kelly_mult` to each state (high-accuracy state → 1.0,
low-accuracy state → 0.5 by default). Persists to
`regime_classifier_<market>.pkl`.

**Wired in:** `build_diffusion_signal` automatically loads
`regime_classifier_<config.data_subdir>.pkl` if present. The signal
calls `_maybe_compute_regime` once per window when tau crosses
`regime_early_tau_s` (700s for 15m, 200s for 5m), caches the regime
label and kelly_mult in ctx, and multiplies `kelly_fraction` by it at
sizing time.

**Trained models:**

| Market | Holdout state characterisation |
|---|---|
| BTC 15m | state 0: n=450, sig_acc=0.940 → kelly=1.0; state 1: n=570, sig_acc=0.925 → kelly=0.5 |
| BTC 5m  | state 0: n=201, sig_acc=0.731 → kelly=0.5; state 1: n=3235, sig_acc=0.887 → kelly=1.0 |

**Empirical impact on backtest:** Zero. Both ablation rows for "+regime"
are identical to baseline. The regime classifier is loading and firing
(I verified with a context print), but the per-state accuracy gap on
BTC 15m (94.0% vs 92.5%) is too small for the 0.5x penalty to swing
the bet result, AND the BTC 5m low-acc state is so rare (5.8% of
windows) that it almost never affects a fired trade.

**Why this is OK:** The infrastructure is shipped and working. The kelly
multipliers are configurable via the `--high-mult`/`--low-mult` flags on
`scripts/train_regime_classifier.py`. To make the regime classifier
*matter* in backtest, retrain with more aggressive multipliers
(e.g. `--high-mult 1.0 --low-mult 0.0` would zero out all bets in the
low-accuracy state) — but that's a tuning decision the user should make
with their own risk tolerance, not me.

### 2d — Kalman OBI smoothing

**Built:** `_kalman_obi_update` method on `DiffusionSignal` runs an AR(1)
Kalman filter on the raw L2 OBI (β=0.95, Q=0.001, R=0.05) — half-life
~20 ticks, smooths book flicker.

**Wired in:** Opt-in via `use_kalman_obi=True`. Default `False` keeps
the legacy raw-OBI gate. When enabled, the gate consumes the smoothed
value; the raw value is always recorded in ctx as `_obi_raw`.

**Why default-off:** Same lesson as 2a. Smoothing the OBI lets in
trades that the raw gate would have filtered, hurting precision on a
high-precision strategy. The Kalman OBI is the right *direction* —
you don't want a single-tick book flicker to cancel an otherwise good
trade — but downstream params need to be re-tuned to take advantage
of it.

**Test:** `test_kalman_obi_smoother_damps_noise` and
`test_kalman_obi_smoother_tracks_step` verify the smoother behaves
correctly (damps zero-mean noise to 0; converges to >0.3 on a +0.5
constant input).

### 2e — Filtration warmup parity (mid_momentum)

**Built:** Two paths through `_check_filtration` for the `mid_momentum`
feature, selectable via `mid_momentum_parity=False/True`. Default
preserves the legacy "return 0 if len<62" behavior. Opt-in path matches
`train_filtration.py:compute_mid_momentum` exactly: use the earliest
available index up to 60 ticks back, even when len<62.

**Why default-off:** The existing `filtration_model.pkl` was trained
against the broken inference behavior. Flipping the flag on without
retraining the filtration model gives the model an input distribution
it wasn't calibrated for — which hurts.

**To deploy:** Run `train_filtration.py` (no changes needed; the parity
fix is in inference only), then set `mid_momentum_parity=True` in
`build_diffusion_signal`. Verify with the walk-forward filtration test
in 3d.

**Test:** `test_mid_momentum_warmup_parity` verifies that with the flag
on, the inference computation is bit-identical to the training one
across 200 sample positions on a synthetic mid stream.

### Data-driven hourly vol priors

**Built:** `scripts/regen_hourly_priors.py` scans every parquet for a
market, computes the per-second realized σ at each tick using a 90-tick
rolling window, buckets by UTC hour and weekday, and writes the result
to `data/<market>/hourly_priors.json`. `backtest.py:_time_prior_sigma`
loads it lazily on first call (with cache) and falls back to the
hardcoded constants when the JSON isn't present.

**Refresh:** Recommended cadence is weekly, via cron:

```cron
0 4 * * 0  cd /path/to/repo && uv run python scripts/regen_hourly_priors.py --market btc
0 4 * * 0  cd /path/to/repo && uv run python scripts/regen_hourly_priors.py --market btc_5m
```

**Wired in:** Yes. Both BTC markets now have a `hourly_priors.json`
written by this run.

**Result of regenerating** (validation_runs/RESULTS.md memo: "make
hourly priors data-driven; the hardcoded 8.9e-05 mean is stale by ~44%"):

| Market | Hardcoded global σ | Data-driven global σ | Drift |
|---|---:|---:|---|
| BTC 15m | 8.9e-05 | 4.9e-05 | -45% |
| BTC 5m  | 8.9e-05 | 4.9e-05 | -45% |

The constants in `backtest.py:64-74` have been wrong by a factor of
about 1.8x for some time. The data-driven path now corrects this on
every cold-start σ call.

**Backtest impact:** BTC 15m PnL moved +1.2% (+$43 on the same 113
trades) because cold-start σ is now smaller, which makes a few
warmup-period z-scores slightly more aggressive. Small but positive.

---

## Section 3 — Validation tests

All seven tests are built, exercised, and persist their JSON outputs to
`validation_runs/postfix_<market>/`. Findings below.

### 3a — Permutation tests for sequence structure

**Script:** `scripts/validate_permutation_sharpe.py`

The naive "shuffle PnLs and recompute Sharpe" doesn't work because Sharpe
is invariant to ordering. The test instead permutes 2000× and reports
shuffled distributions of *path-dependent* metrics:

- **Max drawdown** (path-dependent — clusters in time look worse)
- **Lag-1 autocorrelation** of PnL (clustering proxy)
- **Wald-Wolfowitz runs test** Z-score (sign-clustering)

**Result:**

| | BTC 5m | BTC 15m |
|---|---|---|
| Actual MDD | 0.476 | 0.290 |
| Shuffled MDD median | 0.444 | 0.276 |
| P(shuffled MDD ≥ actual) | 0.398 | 0.430 |
| Lag-1 autocorr (actual) | -0.029 | -0.133 |
| P(autocorr extreme) | 0.615 | 0.162 |
| Runs Z | +0.625 | -0.972 |
| Verdict | i.i.d. | i.i.d. |

**Both markets pass.** Trades show no statistically significant ordering
structure — losses don't cluster, wins don't cluster, autocorrelation is
not significant. **The post-fix edge is not concentrated in lucky streaks.**

### 3b — Bootstrap Kelly CI

**Script:** `scripts/validate_bootstrap_kelly.py`

Resamples trades with replacement 2000× and computes the empirical
optimal Kelly fraction for each resample. Reports the 5/25/50/75/95
percentiles plus how often the resample gives Kelly = 0 (no edge).

**Result:**

| | BTC 5m | BTC 15m |
|---|---|---|
| Empirical p_win | 0.585 | 0.637 |
| Empirical avg win (per stake) | +0.687 | +0.730 |
| Empirical avg loss (per stake) | -1.000 | -1.000 |
| Point Kelly f* | 0.000 | 0.140 |
| Bootstrap Kelly p05 | 0.000 | 0.000 |
| Bootstrap Kelly p50 | 0.000 | 0.142 |
| Bootstrap Kelly p95 | 0.096 | 0.325 |
| **P(Kelly = 0 in resample)** | **60.85%** | **9.85%** |

**Important nuance:** The point Kelly is 0 for BTC 5m because this is the
*per-stake* return, where every trade contributes equally regardless of
size. In dollar terms (gambler's ruin below), the strategy is favorable
because the high-edge trades happen to be sized larger. Both views are
valid; they answer different questions. The Bootstrap Kelly CI is the
relevant one for "should I trust this for sizing?"

**Verdict:** BTC 5m sample is too small / too marginal for Kelly to be
meaningful. BTC 15m has a positive central tendency but the lower 5%
tile is at 0 — the sample is small enough that 1 in 10 plausible
alternative samples gives no edge. **Recommendation: don't deploy
on either with samples this small.** Wait for more live trades.

### 3c — Ergodicity Monte Carlo

**Script:** `scripts/validate_ergodicity.py`

Bootstraps 2000 wealth paths × 500 trades per path, using the *actual
per-trade pnl/bankroll-at-time* (which preserves the strategy's Kelly
sizing) rather than uniform sizing. Reports ensemble mean vs median
terminal wealth, plus paths-below-initial and ruin-proxy rates.

**Result:**

| | BTC 5m | BTC 15m |
|---|---:|---:|
| Initial bankroll | $10,000 | $10,000 |
| Ensemble mean terminal | $16,527 | $64,653 |
| Median terminal | $11,278 | $39,610 |
| Mean / median ratio | 1.47× | 1.63× |
| Time-avg growth/trade | +0.000232 | +0.002775 |
| P(below initial) | 44.4% | 7.8% |
| P(below 50%) | 18.4% | 1.1% |
| P(above 2×) | 26.0% | 75.8% |
| P(above 10×) | 0.5% | 18.2% |

**BTC 15m is genuinely ergodic and edge-bearing.** The typical trader
(median path) doubles their bankroll. Less than 1% lose half. Time-avg
growth is solidly positive.

**BTC 5m is marginal.** Time-avg growth is positive but tiny, and
nearly *half* of bootstrapped paths end below initial bankroll. Mean
overstates median by 47% — non-ergodicity is real. The +$863 backtest
result is consistent with "narrow positive edge that produces frequent
losing paths."

**This is the critical finding for sizing decisions.** BTC 5m at current
parameters is not safe to trade aggressively even though backtest says
it's positive. Either tighten `edge_threshold` or accept that BTC 5m is
a research market, not a deployment market, until more data accumulates.

### 3d — Walk-forward filtration retrain

**Script:** `scripts/validate_walk_forward_filtration.py`

Properly rolling time-series cross-validation: rolling 21-day train
window (14d for 5m), 1-day embargo, 7-day test window (5d for 5m),
advance 7 days (5d for 5m). For each fold, retrain XGBoost from
scratch on the train window and compute OOS AUC, Brier, lift over
baseline, and kept-percentage on the test window.

**Result:**

| | BTC 15m | BTC 5m |
|---|---|---|
| Folds | 3 | 6 |
| Mean OOS AUC | 0.791 | 0.766 |
| Mean OOS lift | +0.016 | +0.026 |
| Mean IS AUC | 0.878 | 0.867 |
| **OOS / IS AUC ratio** | **0.901** | **0.883** |
| Verdict | ✓ HEALTHY | ✓ HEALTHY |

**Both filtration models pass the overfitting check** (target ≥ 0.85,
both above 0.88). The existing `filtration_model.pkl` is generalising
across rolling folds — it's not just memorising the training period.
OOS lift is positive in *every fold* for both markets.

This is the answer to "is the filter leaking?" — **no**.

### 3e — Stratified calibration

**Script:** `scripts/validate_stratified_calibration.py`

Re-runs the global calibration analysis from
`analysis/calibration_analysis.py`, but stratifies by:
- weekday vs weekend
- vol tercile (low / mid / high realised σ)
- hour-of-day bucket (0-5, 6-11, 12-17, 18-23 UTC)

**Result:**

| Market | global ECE | weekday ECE | **weekend ECE** | low σ ECE | high σ ECE |
|---|---:|---:|---:|---:|---:|
| BTC 15m | 0.028 | 0.031 | **0.026** | 0.036 | 0.059 |
| BTC 5m  | 0.049 | 0.058 | **0.032** | 0.033 | 0.099 |

**The headline finding: weekend ECE is now BETTER than weekday ECE on
both markets.** Pre-fix this would have shown the opposite — weekend was
catastrophic because the Kou drift bug was strongest at low σ. The bug
fix is working in the most diagnostic regime.

**New finding from this test:** the **high-σ tercile** has substantially
worse calibration than low-σ on BTC 5m (0.099 vs 0.033 ECE). The post-fix
model is over-/under-confident in high-volatility regimes. This is a
candidate for a **future improvement** — the model assumes a constant σ
within the prediction window but real σ jumps inside the window. Hawkes
intensity modulation (item 2b, currently published as a feature but not
consumed) could help here.

### 3f — Deflated Sharpe with realistic N

**Script:** `scripts/validate_deflated_sharpe.py`

Uses the same haircut formula as `backtest.py:_compute_metrics` but
sweeps `n_trials` over {1, 10, 100, 1000, 10000}. Reports the deflated
Sharpe at each N.

**Result:**

| Market | Raw Sharpe | N=10 | N=100 | N=1000 | N=10000 |
|---|---:|---:|---:|---:|---:|
| BTC 5m | +0.080 | -1.103 | -1.593 | -1.969 | -2.286 |
| BTC 15m | +0.723 | -1.255 | -2.074 | -2.703 | -3.232 |

**Sobering result:** even raw +0.72 Sharpe on BTC 15m doesn't survive
N=10 multiple-testing penalty. With the project's full history of
parameter sweeps (probably hundreds of configs tried over months),
the honest deflated Sharpe is deeply negative.

**This is consistent with the bootstrap Kelly CI finding** (small samples,
wide CI, low statistical power). Both are saying the same thing: sample
size is the binding constraint, not the model. **The post-fix numbers
are real improvements over the buggy baseline, but they don't constitute
statistical proof of edge given the multiple-testing burden of an active
research project.**

### 3g — Fee-adjusted gambler's ruin

**Script:** `scripts/validate_ruin.py`

Computes `r = (q × L) / (p × W)` from real trade dollar amounts (not
per-stake fractions) and reports gambler's-ruin probability vs target
bankroll multiples.

**Result:**

| | BTC 5m | BTC 15m |
|---|---:|---:|
| p_win | 0.585 | 0.637 |
| avg win ($) | +269.01 | +353.62 |
| avg loss ($) | -373.31 | -531.66 |
| Per-trade EV | +$2.73 | +$32.41 |
| Favorability r = qL/pW | 0.983 | 0.856 |
| Status | FAVORABLE (r<1) | FAVORABLE (r<1) |
| P(ruin → 2× target) | 0.0001 | 0.0000 |
| P(ruin → 10× target) | 0.0001 | 0.0000 |

**Both markets have r<1** in dollar terms, so gambler's ruin is
asymptotically improbable. BTC 5m's r=0.983 is *barely* favorable —
margin of about 1.7% per dollar staked. BTC 15m's r=0.856 has more
breathing room.

**The dollar-EV / per-stake-EV discrepancy:** BTC 5m has a *negative*
per-stake EV (avg pnl_pct = -1.26%) but a *positive* dollar EV
(+$2.73/trade), because the strategy varies stake size with edge
confidence and the larger stakes happen to win more (62% WR for the
"huge" quartile vs 48% for the "small" quartile). This is a real edge
in the *sizing rule*, not in the per-trade signal. It's also fragile —
if the size-edge correlation breaks, the strategy goes negative.

### 3h — Stale-feature gates

**Built:** Three new gates on `DiffusionSignal`, mirroring the existing
`max_book_age_ms`:
- `max_chainlink_age_ms`
- `max_binance_age_ms`
- `max_trade_tape_age_ms`

Each is a hard skip when the corresponding `_age_ms` field in ctx
exceeds the threshold. All four (including the existing book gate) are
no-ops in backtest because `BacktestEngine` doesn't populate the age
fields. They only fire in live trading.

**Bonus fix found while implementing this:** The single-side `decide`
method was missing the `max_book_age_ms` gate that exists in
`decide_both_sides`. So the user's `max_book_age_ms=1000` setting in
btc_5m config was *silently inactive* in FOK mode (which is what the
backtest uses). Both gates now fire in both decision paths.

**Defaults set in `market_config.py`:**

| | BTC 15m | BTC 5m |
|---|---:|---:|
| max_chainlink_age_ms | 60,000 | 30,000 |
| max_binance_age_ms | 2,000 | 1,500 |
| max_trade_tape_age_ms | 10,000 | 8,000 |

5m markets have tighter thresholds because they react faster.

**Test:** `test_stale_feature_gates_skip_when_age_exceeds` verifies the
helper returns `None` when fresh, the right reason string when stale,
and `None` when no age fields are set (backtest mode).

---

## Files added / modified

```
modified:
  backtest.py                       — _model_cdf bug fix preserved + section 2/3 wiring
  market_config.py                  — sigma bounds + stale gates + sigma_estimator
  analysis/calibration_analysis.py  — kou drift fix preserved
  analysis/optimize_kou_5m.py       — kou drift fix + stale baseline fix preserved
  analysis/hmm_regime.py            — early_tau parameterised
  tests/test_model_cdf.py           — 18 regression tests (was 10)

new (Section 2):
  scripts/sigma_estimators.py            — RV/EWMA/GARCH(1,1)
  scripts/hawkes.py                      — HawkesIntensity + fit_hawkes_mle
  scripts/regen_hourly_priors.py         — data-driven priors regen
  scripts/train_regime_classifier.py     — HMM trainer
  regime_classifier.py                   — RegimeClassifier wrapper
  regime_classifier_btc_15m.pkl          — trained 15m HMM
  regime_classifier_btc_5m.pkl           — trained 5m HMM
  data/btc_15m/hourly_priors.json        — fresh priors
  data/btc_5m/hourly_priors.json         — fresh priors

new (Section 3):
  scripts/dump_trades.py                          — backtest trade dumper
  scripts/validate_permutation_sharpe.py          — 3a
  scripts/validate_bootstrap_kelly.py             — 3b
  scripts/validate_ergodicity.py                  — 3c
  scripts/validate_walk_forward_filtration.py    — 3d
  scripts/validate_stratified_calibration.py      — 3e
  scripts/validate_deflated_sharpe.py             — 3f
  scripts/validate_ruin.py                        — 3g
  scripts/validate_sigma_estimators.py            — 2a A/B
  scripts/ablate_improvements.py                  — section 2 ablation harness

new (validation outputs — all under validation_runs/):
  postfix_btc/                  — 113-trade dump + all 3a-g outputs
  postfix_btc_5m/               — 316-trade dump + all 3a-g outputs
  stratified_calibration/       — 3e per-market JSONs and PNGs
  sigma_estimators/             — 2a A/B per-market JSONs
  ablation_btc.json             — section 2 ablation
  ablation_btc_5m.json          — section 2 ablation
  walk_forward_filtration_btc_15m.json   — 3d
  walk_forward_filtration_btc_5m.json    — 3d
  RESULTS_section_2_3.md        — this file
```

## Test summary

```
$ python tests/test_model_cdf.py
PASS  test_hawkes_fit_recovers_params_on_synthetic
PASS  test_hawkes_intensity_decay_and_excitation
PASS  test_kalman_obi_smoother_damps_noise
PASS  test_kalman_obi_smoother_tracks_step
PASS  test_kou_asymmetric_p_up_does_not_distort_z0
PASS  test_kou_cdf_clt_branch_internally_consistent
PASS  test_kou_cdf_reduces_to_normal_when_lambda_is_zero
PASS  test_kou_no_blowup_at_low_sigma
PASS  test_kou_path_equals_normal_path
PASS  test_kou_strictly_monotone_in_z
PASS  test_kou_symmetric_at_extremes
PASS  test_kou_symmetric_returns_half_at_z0
PASS  test_market_config_sigma_bounds_reasonable
PASS  test_mid_momentum_warmup_parity
PASS  test_sigma_estimator_dispatch
PASS  test_sigma_estimator_validation
PASS  test_stale_feature_gates_skip_when_age_exceeds
PASS  test_student_t_unchanged
18/18 tests passed
```

## What's next

Concrete things you can do with what's been built:

1. **Enable EWMA σ for BTC** (item 2a). Set `sigma_estimator="ewma"` in
   the BTC `MarketConfig` literals. Then re-tune `edge_threshold` (start
   by trying `0.05` instead of `0.06`) and re-run `train_filtration.py`
   to retrain the filtration model with the new σ distribution.
   Verify with `dump_trades.py` + the validation suite.

2. **Enable mid_momentum parity** (item 2e). Set
   `mid_momentum_parity=True` AFTER retraining the filtration model.
   Walk-forward retrain test (3d) will tell you if it generalises.

3. **Enable Kalman OBI** (item 2d). Same playbook as 2a — set
   `use_kalman_obi=True` and re-tune `edge_threshold` upward to
   compensate for the looser gate.

4. **Enable Hawkes intensity feature** (item 2b). Add `_hawkes_intensity`
   to `filtration_model.py:extract_features`, retrain, and pass
   `hawkes_params=(0.005, 0.4, 0.025, 2.5)` (sane defaults — fit
   actual params via `fit_hawkes_mle` on jump events from your data).

5. **Refresh hourly priors weekly.** Add the cron entries from the
   "Data-driven hourly vol priors" section above.

6. **Retrain regime classifier with more aggressive multipliers** if
   you want it to actually affect sizing (item 2c). Try
   `--high-mult 1.0 --low-mult 0.0`.

7. **Trust the validation tests when re-running steps 1-4.** Each
   change should re-pass: stratified calibration (3e), walk-forward
   filtration (3d), permutation tests (3a), ergodicity (3c).

The guardrails are now in place. Use them.
