# Bug-fix validation results

Date: 2026-04-07

## What was fixed

| Bug | File | Change |
|---|---|---|
| A — Kou risk-neutral drift in physical-measure prediction | `backtest.py:1374-1389` | `tail_mode="kou"` now returns `norm_cdf(z)` (no drift) |
| A — duplicate in optimizer | `analysis/optimize_kou_5m.py:111-188` | drift_z removed; signature kept for compat, params now inert |
| A — duplicate in calibration | `analysis/calibration_analysis.py:35-48` | model_cdf kou path → `norm_cdf(z)` |
| B — BTC 5m σ pinned to 14% band | `market_config.py:btc_5m` | `min_sigma 7e-5 → 1e-5`, `max_sigma 8e-5 → 2e-4` |
| B — BTC 15m σ band too tight | `market_config.py:btc` | `min_sigma 3e-5 → 1e-5`, `max_sigma 8e-5 → 4e-4` |
| C — stale baseline in optimizer | `analysis/optimize_kou_5m.py:200,320,362` | now reads from `MARKET_CONFIGS["btc_5m"]` |

Empirical motivation for the σ-bound change (90s realized vol on real data):

```
btc_5m  n=9812:  p5=5.8e-6  p50=2.95e-5  p95=9.6e-5  p99=1.5e-4  max=4.4e-4
                 → 88.4% of samples were BELOW the old 7e-5 floor
                 →  8.3% were ABOVE the old 8e-5 cap
                 → only 3.3% of all observations could move the prediction

btc_15m n=106592: p5=9.2e-6  p50=3.9e-5  p95=1.7e-4  p99=4.1e-4  max=4.8e-3
                 → 37.1% below floor, 19.1% above cap
```

## Test 1 — unit tests

```
$ python tests/test_model_cdf.py
PASS  test_kou_asymmetric_p_up_does_not_distort_z0
PASS  test_kou_cdf_clt_branch_internally_consistent
PASS  test_kou_cdf_reduces_to_normal_when_lambda_is_zero
PASS  test_kou_no_blowup_at_low_sigma
PASS  test_kou_path_equals_normal_path
PASS  test_kou_strictly_monotone_in_z
PASS  test_kou_symmetric_at_extremes
PASS  test_kou_symmetric_returns_half_at_z0
PASS  test_market_config_sigma_bounds_reasonable
PASS  test_student_t_unchanged

10/10 tests passed
```

The two key regression tests are `test_kou_no_blowup_at_low_sigma` (specifically tests the BTC 15m σ-floor case where the bias was −7%) and `test_kou_asymmetric_p_up_does_not_distort_z0` (specifically tests the live BTC 5m config).

## Test 2 — direct bias measurement

Pre-fix vs post-fix `_model_cdf` output at the same `z, sigma, tau`:

```
BTC 5m, sigma=7e-5, tau=300, kou_p_up=0.526:
  delta=0.0%   pre-fix p=0.4826   post-fix p=0.5000   bias gone
  delta=+0.05% pre-fix p=0.6438   post-fix p=0.6600   bias gone
  delta=-0.05% pre-fix p=0.3241   post-fix p=0.3400   bias gone

BTC 15m, sigma=3e-5, tau=900 (the σ-floor / weekend case):
  delta=0.0%   pre-fix p=0.4298   post-fix p=0.5000   bias gone (was -7.02%)
  delta=+0.05% pre-fix p=0.6476   post-fix p=0.7107   bias gone (was -6.32%)
  delta=-0.05% pre-fix p=0.2320   post-fix p=0.2893   bias gone (was -5.73%)
```

Post-fix bias is exactly 0.0000% across the entire test grid (sigma ∈ {1e-7, 1e-6, 1e-5, 1e-4, 1e-3}, tau ∈ {60, 300, 900}, z ∈ [-2.5, 2.5]).

## Test 3 — calibration on full dataset

Same `analysis/calibration_analysis.py`, identical inputs, before vs after fix. ECE = expected calibration error (lower is better).

| Market    | Tail mode  | n_obs (before → after) | ECE before | ECE after | ECE Δ      | Brier before | Brier after | LogLoss before | LogLoss after |
|-----------|------------|------------------------|------------|-----------|------------|--------------|-------------|----------------|---------------|
| **BTC 15m** | kou      | 134,819 → 140,489      | **0.0423** | **0.0273** | **−35.5%** | 0.1698       | 0.1681      | 0.5150         | 0.5116        |
| **BTC 5m**  | kou      | 49,244 → 51,925        | **0.0802** | **0.0445** | **−44.5%** | 0.1522       | 0.1513      | 0.4732         | 0.4722        |
| ETH 15m   | student_t | 80,094 → 80,094        | 0.0271     | 0.0271    | (unchanged) | 0.1921       | 0.1921      | 0.5675         | 0.5675        |
| ETH 5m    | student_t | 19,592 → 19,592        | 0.0372     | 0.0372    | (unchanged) | 0.1849       | 0.1849      | 0.5499         | 0.5499        |
| SOL 15m   | normal    | 27,266 → 27,266        | 0.0320     | 0.0320    | (unchanged) | 0.1781       | 0.1781      | 0.5345         | 0.5345        |
| SOL 5m    | normal    | 9,533 → 9,533          | 0.0253     | 0.0253    | (unchanged) | 0.1809       | 0.1809      | 0.5415         | 0.5415        |
| XRP 15m   | normal    | 26,237 → 26,237        | 0.0235     | 0.0235    | (unchanged) | 0.1899       | 0.1899      | 0.5624         | 0.5624        |
| XRP 5m    | normal    | 9,217 → 9,217          | 0.0240     | 0.0240    | (unchanged) | 0.1901       | 0.1901      | 0.5622         | 0.5622        |

Two important observations:
1. **Only the kou markets moved** (BTC 15m, BTC 5m). Student-t and normal markets are byte-identical, confirming the fix is surgical and didn't accidentally change any other CDF path.
2. **n_obs grew on the BTC markets** because realized vol can now go below the old 7e-5 / 3e-5 floors, so windows that were previously rejected as "zero/low vol" are now usable.

### BTC 5m bin-level calibration shift

```
Pred Range       Before:  Avg Pred   Actual UP%    Count    →   After:  Avg Pred   Actual UP%    Count
[0.10, 0.20)              0.156         0.039     9,719              0.160         0.104     15,833
[0.20, 0.30)              0.252         0.183     3,368              0.252         0.350      2,232
[0.40, 0.50)              0.455         0.473     7,969              0.455         0.450      4,503
[0.50, 0.60)              0.546         0.624     6,526              0.545         0.594      4,614
[0.80, 0.90)              0.834         0.952     9,335              0.840         0.891     15,729
```

Look at the extremes. Before the fix, the bottom bin showed `pred=0.156, actual=0.039` — when the model said 15.6% probability of UP, the actual UP rate was only 3.9%. After the fix, the same bin shows `pred=0.160, actual=0.104` — pred is much closer to actual. The top bin moved from `(0.834, 0.952)` to `(0.840, 0.891)`. Both extremes are dramatically less off-center.

## Test 4 — backtest comparison (the one that actually pays bills)

Same seed, same train_frac=0.7, same data, identical CLI invocation. Out-of-sample (test set) metrics:

### BTC 5m

| Metric          | Before fix | After fix | Δ            |
|-----------------|-----------:|----------:|-------------:|
| Trades fired    |        217 |       316 | +99 (+45.6%) |
| Win rate        |      53.5% |     58.5% | **+5.0pp**   |
| Total PnL       |       +$361 |    +$863 | **+$502 (+139%)** |
| Sharpe (annual) |       0.05 |      0.08 | +0.03        |
| Max drawdown    |      42.1% |     47.6% | +5.5pp       |
| Final bankroll  |    $10,361 |   $10,863 | +$502        |
| Verdict         | MARGINAL EDGE | ✓ EDGE DETECTED | flipped |

### BTC 15m  ← **the dramatic one**

| Metric          | Before fix | After fix | Δ            |
|-----------------|-----------:|----------:|-------------:|
| Trades fired    |         79 |       113 | +34 (+43%)   |
| Win rate        |      54.4% |     63.7% | **+9.3pp**   |
| Total PnL       |    **−$2,462** | **+$3,663** | **+$6,125 swing** |
| Sharpe (annual) |     **−0.85** | **+0.72** | **+1.57**    |
| Max drawdown    |      47.5% |     29.0% | **−18.5pp**  |
| Final bankroll  |     $7,538 |   $13,663 | +$6,125      |
| Verdict         | ✗ NO EDGE  | ✓ EDGE DETECTED | flipped |

BTC 15m flipped from a clear net-loser to a clear net-winner on the same data. This is exactly what the analysis predicted — the 15m market was the most affected by Bug A because the σ floor (3e-5) was lowest and the bias `drift_z ∝ 1/σ` was correspondingly the largest. With the floor σ in evidence in the live trade printouts (e.g. baseline trades showing `sig=3.00e-05`), the bug was inflicting a −7% systematic downward bias on every BTC 15m prediction. Removing the bug recovered the edge.

### Notes on the BTC 5m drawdown going up

The 42% → 48% drawdown on BTC 5m is the only metric that didn't improve. Likely cause: the bot is now firing 46% more trades (217 → 316) and sigma is no longer pinned, so position sizing via half-Kelly is sometimes larger than before. Three things to consider next:

1. The total PnL improved much more than the drawdown widened, so risk-adjusted performance still improved.
2. BTC 15m drawdown *fell* by 18.5 percentage points, which is the opposite direction.
3. If the wider drawdown is uncomfortable, lower `kelly_fraction` from 0.25 → 0.15 or tighten `edge_threshold` 0.06 → 0.07. But validate that against backtest first; do not rebuild a fudge factor on top of a now-correct model.

The user's existing memory notes (`project_btc5m_market_blend.md` and `project_btc_5m_weekend_underperform.md`) describe symptoms that this fix addresses at the root rather than working around. Both memory entries should be re-evaluated against post-fix data.

## What was NOT changed

- Half-Kelly sizing math at `backtest.py:1688` — correct as-is.
- Yang-Zhang vol estimator at `backtest.py:257-311` — correct math, just suboptimal for 5s micro-bars; flagged as a future improvement, not a bug.
- All non-kou tail modes (`student_t`, `normal`, `market_adaptive`).
- The full proper `kou_cdf` function at `backtest.py:164-211` — kept intact in case the user later wants to wire it into `_model_cdf` to actually fatten the tails (sigma_per_s assumption would need to be revisited if so).
- Filtration model, microstructure gates, oracle lag handling, fee math, deflated Sharpe, calibration table fusion — all left alone.

## Files touched

```
modified:   backtest.py                       (1 hunk, ~20 lines net)
modified:   market_config.py                  (2 hunks: btc and btc_5m)
modified:   analysis/optimize_kou_5m.py       (4 hunks: 2 functions + 2 main printouts)
modified:   analysis/calibration_analysis.py  (1 hunk: model_cdf kou path)
new file:   tests/test_model_cdf.py           (10 regression tests, ~250 lines)
new file:   validation_runs/RESULTS.md        (this file)
new file:   validation_runs/baseline_*.txt    (pre-fix backtest + calibration logs)
new file:   validation_runs/after_*.txt       (post-fix backtest + calibration logs)
```

## Reproducing the validation

```bash
# unit tests
python tests/test_model_cdf.py

# calibration (uses all data, ~30s)
python analysis/calibration_analysis.py

# backtests (each ~5 min on the full corpus)
python backtest.py --market btc_5m --signal diffusion --train-frac 0.7 --seed 42
python backtest.py --market btc    --signal diffusion --train-frac 0.7 --seed 42
```
