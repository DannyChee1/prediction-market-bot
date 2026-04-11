# Sigma Estimation Research for Binary BTC Prediction

**Date**: 2026-04-11
**Goal**: Find the best volatility estimation approach for P(UP) = Phi(z), where z = delta / (sigma * sqrt(tau)), predicting whether BTC will be above/below a reference price in 5-15 minutes.

---

## 1. Academic Literature on Short-Horizon Realized Volatility

### 1.1 Which Estimator Is Best?

**Range-based estimators (Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang):**

These were designed for daily OHLC bars. Their relative efficiency vs close-to-close:
- Parkinson: 5.2x more efficient (zero-drift assumption; biased with drift)
- Garman-Klass: 7.4x more efficient (zero-drift assumption; biased with drift)
- Rogers-Satchell: 6x more efficient (handles drift; no opening jumps)
- Yang-Zhang: 14x more efficient (handles drift AND opening jumps)

**Critical problem with Yang-Zhang on sub-minute bars**: YZ was designed for daily bars with overnight gaps. Its `var_oc` component is the open-to-previous-close ("overnight gap") variance. On a continuously-traded feed like BTC, bars are contiguous -- there IS no overnight gap. The `var_oc` term degenerates into measuring the noise between bar boundaries (essentially, the difference between one bar's close and the next bar's open, which for 5-second micro-bars is often zero or near-zero). This adds a near-zero variance component that dilutes the estimate.

The codebase already documents this: `"var_oc term assumes overnight gaps that don't exist on a continuously-traded feed"` (signal_diffusion.py:411, market_config.py:59-61).

**Realized variance (sum of squared returns)**: The simplest approach. Under semimartingale theory (Andersen & Bollerslev, 1998), as sampling frequency increases, realized variance converges to the integrated variance -- but only in the absence of microstructure noise. With noise, it diverges.

**Bipower variation (Barndorff-Nielsen & Shephard, 2004)**: Jump-robust. BV = (pi/2) * mean(|r_i| * |r_{i-1}|). A single jump contaminates only one return, not its neighbor, so the cross-product averages away the jump. The continuous component sigma_continuous^2 approx BV, and the jump component = max(0, RV^2 - BV^2). This is the RIGHT sigma for a jump-diffusion model like Kou because it avoids double-counting jump variance. Already implemented in `scripts/sigma_estimators.py:bipower_variation_per_s()`.

**EWMA (RiskMetrics-style)**: sigma^2(t) = lambda * sigma^2(t-1) + (1-lambda) * r^2(t). No assumption about overnight gaps. Self-adapting. Lambda=0.94 is the RiskMetrics daily default. For short-horizon forecasting, research shows lower lambda (faster decay) is optimal -- the "short-term memory gains importance" (Ding & Meade, 2021, arxiv:2105.14382). For sub-second returns, lambda around 0.85-0.90 is likely more appropriate than 0.94.

**Realized kernels (Barndorff-Nielsen, Hansen, Lunde, Shephard 2008)**: Provably consistent even with IID microstructure noise. They weight autocovariances of returns using a kernel function (Parzen, Tukey-Hanning). Insensitive to sampling frequency choice. Computationally heavier but state-of-the-art for noisy tick data.

**GARCH(1,1)**: sigma^2(t) = omega + alpha * r^2(t-1) + beta * sigma^2(t-1). Mean-reverting (unlike EWMA, which is integrated GARCH with omega=0). Already implemented in `scripts/sigma_estimators.py:garch11_sigma_per_s()`. Requires pre-fitted parameters.

### 1.2 Optimal Sampling Frequency

The "5-minute rule" (Andersen, Bollerslev, Diebold, Labys 2000): for equity markets, realized volatility computed from 5-minute returns is optimal because higher frequencies accumulate microstructure noise (bid-ask bounce, price discreteness) that inflates RV. This was established via "volatility signature plots" -- RV vs sampling frequency curves that flatten around 5min for equities.

**For crypto**: The same principle applies but the optimal frequency may be different because:
1. Crypto trades 24/7 (no opening auctions, no overnight gaps)
2. BTC on Binance is highly liquid (trades every ~100ms)
3. Crypto has higher kurtosis (15-100 daily, 2-13 at minute level) and fatter tails
4. Jump component is more prominent in crypto than equities

For the bot's use case (90-second lookback, 1-second sampling), the key concern is: **are 1-second BTC returns contaminated by microstructure noise?** On Binance for BTC/USDT, the bid-ask spread is typically 0.01% or less, and trades happen every ~100ms. At 1-second sampling, the microstructure noise is small relative to true price movement but not negligible. The bot currently uses Chainlink prices (which aggregate with ~1.2s delay), introducing additional smoothing.

**Recommendation**: 1-second sampling from Binance is acceptable for a 90s lookback (gives ~90 observations). Sub-second sampling would add noise. The current 5s OHLC bars for Yang-Zhang are reasonable but the YZ estimator itself is the wrong tool for these bars.

### 1.3 Optimal Lookback Window

For a T-minute prediction horizon, the lookback should be:
- **Long enough** to capture enough returns for statistical significance (at least 20-30 observations at minimum)
- **Short enough** that the volatility estimate is "local" -- reflecting current conditions, not yesterday's

Rules of thumb from literature:
- Lookback = 2-5x the prediction horizon (so 10-25 minutes for 5min prediction, 30-75 min for 15min)
- But for a GARCH/EWMA model, the effective lookback is determined by the decay parameter, not a hard window

The bot uses 90 seconds, which is 0.3x the 5-minute horizon and 0.1x the 15-minute horizon. This is VERY short. With 1-second Binance prices, that's ~90 observations -- statistically OK for an EWMA recursion but tight for a sample-variance estimator.

**The 90s window is a source of instability**: with only 90 observations, adding or removing a single volatile tick can shift realized variance by 30-40%. A 300-second (5-minute) lookback with EWMA would have ~300 observations and be far more stable while still adapting to regime changes.

### 1.4 Microstructure Noise Handling

Methods for handling noise at high frequency:
1. **Subsample and average** (Zhang, Mykland, Ait-Sahalia 2005): compute RV at multiple offsets, average
2. **Pre-averaging** (Podolskij & Vetter 2009): smooth returns before squaring
3. **Realized kernels** (BNHLS 2008): kernel-weighted autocovariances
4. **Two-scale estimator** (Zhang, Mykland, Ait-Sahalia 2005): combine fast-sampled and slow-sampled RV
5. **Skip duplicates** (what the bot already does): drop consecutive identical prices

The bot's current "skip duplicate prices" approach is a crude but effective noise filter for Chainlink data (which has discrete price updates). For Binance bookTicker data (continuous), it works less well because prices are rarely exactly identical.

---

## 2. The "Right" Sigma for Binary Prediction

### 2.1 Realized vs Implied

**For options pricing**: implied volatility (from market prices) is the standard because it's forward-looking and risk-neutral.

**For this bot**: there is no options market to extract implied vol from. The bot needs a FORECAST of what sigma WILL BE over the next 5-15 minutes. Realized vol over the recent past is the only input available. The question is: how to turn historical sigma into a forecast?

Three approaches:
1. **Naive**: sigma_forecast = sigma_realized (assume persistence). Works surprisingly well because vol clusters.
2. **EWMA**: sigma_forecast = latest EWMA state. Forward-looking in the sense that the recursion already decays old information.
3. **GARCH**: sigma_forecast = omega + alpha * r^2(t) + beta * sigma^2(t). Explicitly mean-reverting, which is more realistic.

For this use case, the key insight is: **the bot doesn't need sigma for hedging or risk management. It needs sigma to compute Phi(z), where z = delta / (sigma * sqrt(tau)).** The downstream sensitivity is:

- If sigma is 10% too low: z is 10% too high, Phi(z) moves away from 0.5, model is overconfident
- If sigma is 10% too high: z is 10% too low, Phi(z) stays near 0.5, model is underconfident

**Asymmetric error cost**: overconfidence (sigma too low) is far more dangerous than underconfidence. Overconfidence leads to trades with large perceived edge that don't actually exist -- exactly the failure mode the bot experiences with "sigma collapse." Underconfidence merely reduces trade frequency. For a model where the primary risk is false confidence, a SLIGHT upward bias in sigma is acceptable.

### 2.2 Frozen vs Continuously Updated

**GBM framework**: In a true GBM, sigma IS constant. The bot assumes GBM over a [0, T] window with fixed sigma, then computes P(S_T > K) = Phi(z). If sigma is re-estimated every tick, the model is internally inconsistent -- the probability at t=10s was computed assuming one sigma, but at t=11s assumes a different sigma.

**Practical resolution**: The world isn't GBM. Vol does change. The question is whether re-estimating sigma every tick adds signal or noise. For the bot:

- Updating sigma every tick with a 90s rolling window: HIGH NOISE. Each tick entering/exiting the window shifts the OHLC bar boundaries, which shifts the YZ estimate by 30-40% in extreme cases.
- Updating sigma every tick via EWMA: LOWER NOISE because EWMA is inherently smooth (each new observation has weight 1-lambda ~ 6%).
- Updating sigma every tick via Kalman filter: LOWEST NOISE because the Kalman gain adapts -- small when the estimate is confident, large when uncertainty is high.

**The current approach (Kalman filter on top of YZ) is correct in STRUCTURE but the input to the Kalman filter (YZ) is too noisy.** The Kalman filter is trying to smooth a raw signal that jumps 30-40% per tick. The Kalman parameters (Q=0.10, R=0.075) imply the filter trusts the observation almost as much as the state, so it can't dampen the YZ noise enough.

**Better approach**: Feed EWMA output (which is already smooth) into the Kalman filter. Or replace the YZ+Kalman stack with a single EWMA/GARCH that produces inherently smooth output.

### 2.3 Regime-Switching Models

**GARCH(1,1)** captures "volatility clustering" -- the empirical fact that high-vol periods persist and low-vol periods persist. This is exactly what the bot needs: when vol is high, sigma stays high (no false underconfidence); when vol is low, sigma stays low (no false overconfidence from stale high-vol observations).

**HMM/Markov-switching**: More complex. The bot already has a regime_classifier but it's not wired to sigma estimation. An HMM with 2-3 states (calm/normal/crisis) could provide a regime-dependent sigma, but this is a larger change.

**The sweet spot for this bot**: GARCH(1,1) or EWMA with appropriately tuned lambda. These are "soft" regime-switching models -- they adapt to regime changes without explicitly modeling discrete states.

---

## 3. Alternatives to Sigma-Based P(UP)

### 3.1 Logistic Regression

P(UP) = sigmoid(w1*delta + w2*tau + w3*delta/sqrt(tau) + w4*recent_vol + ...)

**Pros**: No assumption about GBM. Can include arbitrary features (volume, spread, OBI, time-of-day). Calibrated probabilities by construction.
**Cons**: Requires labeled training data (many resolved windows). Linear in features -- can't capture non-linear interactions without feature engineering. Needs retraining as market structure changes.

**Verdict**: Good complement to the GBM model but not a replacement. The z-score IS the logistic regression with one feature (delta/sigma*sqrt(tau)) and the Phi link function. Adding more features is the "filtration model" the bot already has.

### 3.2 Historical Frequency Table (Calibration)

P(UP | z_bucket, tau_bucket) from past resolved windows. Already implemented as `CalibrationTable` in the bot.

**Problem already documented**: With Z_BIN_WIDTH=0.5, all post-fix z values (0.03-0.31) round to z_bin=0.0 and look up the same cell, returning p ~ 0.50. The calibration infrastructure needs a finer bin grid (0.05-0.10) or a continuous interpolation scheme to be useful.

### 3.3 GARCH(1,1) Conditional Volatility

Already implemented. The A/B test showed EWMA has 26% lower 1-step forecast MSE than YZ. GARCH was also tested. The issue is that switching estimators hurts PnL because downstream hyperparameters were tuned to YZ.

### 3.4 Machine Learning (XGBoost/Random Forest)

Could predict P(UP) directly from features (delta, tau, sigma, volume, spread, OBI, time-of-day, recent momentum, etc.). The bot's filtration model already does this as a confidence gate.

**For sigma estimation specifically**: An ML model could predict "sigma over the next 5 minutes" from features, but this requires:
1. Training data (thousands of resolved windows with known forward sigma)
2. Feature engineering (lagged sigmas, returns, volume metrics)
3. Online updating as market structure changes

The HAR (Heterogeneous Autoregressive) model is a simple, effective ML-like approach: sigma_forecast = a * sigma_5min + b * sigma_1hour + c * sigma_1day. The multi-scale structure captures the empirical fact that volatility at different timescales (seconds, minutes, hours) each contributes independently to the forecast. Not currently implemented but relatively simple.

### 3.5 Bayesian Updating

Start with prior P(UP) = 0.50 at window start, update with each price tick using Bayes' rule. Under GBM, the posterior is exactly Phi(z) -- so this IS what the bot is doing, just with a parametric model.

A non-parametric Bayesian approach would maintain a distribution over sigma and integrate out the uncertainty: P(UP) = integral Phi(delta / (sigma * sqrt(tau))) * p(sigma | data) d_sigma. This "model averaging" over sigma uncertainty would naturally make the model less confident when sigma is uncertain (small lookback, regime change). This is theoretically appealing but computationally heavier.

---

## 4. What Works in Practice for Crypto

### 4.1 Crypto-Specific Findings

**Fat tails**: BTC 5-minute returns have kurtosis 15-100 (daily) and tail index 2-13 (minute-level). The bot handles this with Kou jump-diffusion, which is appropriate.

**Inverse leverage effect**: Unlike equities (where negative returns increase vol), positive returns increase crypto vol MORE than negative returns. This means the standard EGARCH leverage term has the wrong sign for crypto. The bot's symmetric estimators (YZ, EWMA, RV) are fine here.

**Jump component is dominant**: Bipower variation analysis shows jumps contribute more to crypto variance than to equity variance. The bot's Kou model and bipower variation estimator already address this.

**Volatility persistence is LOW in crypto**: Vol regimes change faster in crypto than equities. This argues for a SHORTER lookback / faster-decaying EWMA (lower lambda). The bot's 90s window is actually appropriate in spirit -- the issue is not the lookback length but the estimator's stability.

### 4.2 How Crypto Market Makers Estimate Short-Horizon Probabilities

Professional crypto market makers typically use:
1. **Implied vol from options (Deribit)**: When available, this is the best forward-looking estimate. Not available for Polymarket binary contracts.
2. **Realized vol with EWMA**: Fast-adapting, simple, stateless. Lambda tuned to the asset and horizon.
3. **Order flow signals**: Trade imbalance, VPIN, order book depth changes. The bot already incorporates some of these.
4. **Multi-timeframe vol**: Short (1min), medium (5min), long (1hr) realized vols combined. Similar to HAR.

### 4.3 5-Minute Sampling Consensus

Multiple studies confirm 5-minute returns are optimal for realized volatility estimation across assets (Andersen & Bollerslev 2000, Barndorff-Nielsen & Shephard 2004, Liu et al. 2015). For crypto specifically, the same 5-minute frequency has been validated. However, this applies to DAILY vol estimation from intraday data. For 5-minute-ahead forecasting from 1-second data, the situation is different -- the "optimal sampling frequency" research addresses a different use case.

---

## 5. The Specific Sigma Instability Problem

### 5.1 Root Cause Analysis

The bot sees sigma jump 30-40% between ticks because:

1. **YZ on 90s of 5s OHLC bars = only 18 bars**. The YZ estimator computes three components (var_oc, var_co, var_rs) from these 18 bars. With N=18, the sample variance of any component is inherently noisy (standard error of sample variance is proportional to 1/sqrt(2*(N-1)) ~ 17%).

2. **Rolling window edge effects**: When the window slides by 1 second, one 5s bar may exit and another enters. If the exiting bar was calm and the entering bar is volatile (or vice versa), the OHLC composition changes dramatically.

3. **YZ's var_oc term is degenerate**: On continuous 5s bars, open(bar_i) ~ close(bar_{i-1}). So log(open_i / close_{i-1}) is near zero most of the time, occasionally non-zero when a large move happens to span a bar boundary. This creates a high-kurtosis input to the variance estimator, amplifying instability.

4. **Kalman filter can't fix this**: The Kalman's Q=0.10 means "expect sigma to change by 10% of its current value between ticks." R=0.075 means "the observation has noise of 7.5% of current sigma." With these parameters, the Kalman gain K = P_pred / (P_pred + R) converges to about K ~ Q/(Q+R) ~ 0.57. That means 57% of each YZ jump passes through to the smoothed sigma. This is way too responsive.

### 5.2 Practitioner Solutions

**EWMA (Exponential Weighting)**:
- Each new squared return has weight (1-lambda). For lambda=0.94, that's 6% weight per observation.
- A single outlier return shifts sigma^2 by 6%, so sigma shifts by ~3%.
- This is 10x more stable than the rolling-window YZ approach.
- No edge effects: observations decay smoothly instead of falling off a cliff.

**Kalman filtering for vol**:
- The bot already does this but with parameters tuned for a less-noisy input.
- With a smoother input (EWMA), the Kalman can be tuned tighter (lower Q) for additional smoothing.
- OR: skip the Kalman entirely and let EWMA's inherent smoothing be sufficient.

**Median of multiple estimators**:
- Compute YZ, RV, and EWMA simultaneously.
- Take the median as the final sigma.
- The median is robust to one estimator going haywire.
- Downside: adds latency (three computations) and the median itself jumps when the identity of the middle estimator changes.

**Clipping sigma changes per tick**:
- max_delta_sigma = 0.05 * sigma_prev (cap 5% change per tick)
- Simple, effective, but introduces bias: if vol truly spikes, the clip delays the response.
- In practice, real vol doesn't jump 30% between 1-second ticks. Real vol is smooth on a 1-second timescale; the jumps are estimator noise.
- Clipping at 5% per tick means sigma can still double in ~14 seconds (via compounding), which is fast enough for any real regime change.

**Double-EWMA (EWMA of EWMA)**:
- First pass: EWMA of squared returns (the raw sigma estimate)
- Second pass: EWMA of the first pass's output (additional smoothing)
- Equivalent to a second-order lowpass filter. Very smooth but may lag real changes.

### 5.3 Quantitative Impact

With the current stack (YZ + Kalman with K~0.57):
- A 30% YZ jump produces a 0.57 * 30% = 17% change in smoothed sigma
- z changes by ~17% (since z is inversely proportional to sigma)
- At z=0.3 and sigma=3e-5, a 17% sigma shift moves z from 0.30 to 0.25 or 0.35
- Phi(0.35) - Phi(0.25) = 0.637 - 0.599 = 3.8 percentage points of p_model
- With market_blend=0.3, this becomes 2.7pp of the final blended probability
- At edge_threshold=0.06, a 2.7pp swing can cross the threshold and trigger/untrigger a trade

With EWMA (lambda=0.90, no Kalman):
- A single large return shifts sigma by sqrt(0.10 * r^2 / sigma^2 + 0.90) - 1 ~ 5% for a 1-sigma return
- At z=0.3, a 5% sigma shift moves z from 0.30 to 0.285 or 0.315
- Phi(0.315) - Phi(0.285) = 0.624 - 0.612 = 1.2 percentage points
- Well below the threshold sensitivity range

---

## 6. Concrete Recommendation

### Primary Recommendation: EWMA with Tuned Lambda + Rate-Limited Output

**Algorithm:**

```
# Per-tick update (every ~1 second):

1. Compute log return: r = log(price / prev_price) / sqrt(dt_seconds)
   (Skip if price unchanged or dt <= 0)

2. EWMA recursion on sigma^2:
   sigma_sq = lambda * sigma_sq_prev + (1 - lambda) * r^2
   
   lambda = 0.90 for BTC (half-life ~ 10 ticks / 10 seconds)
   
3. Rate-limit the output:
   sigma_new = sqrt(sigma_sq)
   max_change = 0.05 * sigma_prev  # 5% per tick
   sigma_out = clip(sigma_new, sigma_prev - max_change, sigma_prev + max_change)

4. Apply existing min/max sigma bounds:
   sigma_out = clip(sigma_out, min_sigma, max_sigma)
   
5. Apply existing adaptive floor:
   sigma_out = max(sigma_out, MIN_SIGMA_RATIO * sigma_baseline)
```

**Initialization:**
- First 30 ticks: use sample variance of all accumulated returns
- After 30 ticks: switch to EWMA recursion with warm-start from sample variance

**Parameters:**
- `lambda = 0.90` (not 0.94): Crypto vol persistence is lower than equities. Half-life = -1/log(lambda) = 9.5 ticks ~ 10 seconds. This means sigma fully adapts to a new regime in ~50 seconds (5 half-lives), which is appropriate for a 5-minute prediction window.
- `max_change_per_tick = 0.05` (5%): Sigma can still double in 14 seconds via compounding. Real regime shifts over 14 seconds are extreme but possible; estimator noise over 1 second NEVER warrants 5%.

### Why This Is the Right Choice

1. **No misapplied assumptions**: EWMA doesn't have an "overnight gap" term. It works on raw returns. It doesn't assume OHLC bars. It doesn't require bar boundaries.

2. **Inherently smooth**: Each observation has weight 10% (for lambda=0.90). No edge effects from observations falling off a window boundary. The rolling-window cliff is the #1 cause of the current instability and EWMA eliminates it completely.

3. **Self-adapting**: The EWMA recursion naturally gives more weight to recent data. When vol spikes, sigma rises quickly. When vol drops, sigma decays smoothly. No explicit regime detection needed.

4. **Already validated**: The codebase's own A/B test (`scripts/validate_sigma_estimators.py`) showed EWMA beats YZ by 25-39% on 1-step forecast MSE. The only reason it wasn't deployed is that downstream hyperparameters were tuned to YZ.

5. **Rate-limiting kills residual oscillation**: Even EWMA can shift 5-10% on a single large return. The 5%/tick clip makes the output buttery smooth while preserving the ability to track regime changes over 10-20 seconds.

6. **Computationally trivial**: One multiply, one add, one sqrt, one clip. No OHLC bar construction. No matrix operations. Faster than the current YZ pipeline.

7. **The Kalman filter becomes optional**: EWMA + rate-limiting produces output smooth enough that the Kalman's additional smoothing may be unnecessary. But if you keep it, the Kalman will work MUCH better with a smooth EWMA input (tune Q lower, like 0.02, to barely adjust; R stays at 0.075).

### Why Not Other Approaches

- **GARCH(1,1)**: Better in theory (mean reversion) but requires pre-fitted omega/alpha/beta. If the fit drifts, the model breaks. EWMA is GARCH with omega=0, which sacrifices mean reversion for robustness. The adaptive floor + min/max sigma bounds provide the mean reversion that GARCH would give.

- **Bipower variation**: Jump-robust, which is great for Kou model. But it suffers the same rolling-window edge effects as RV. Use BV for calibrating Kou parameters offline; use EWMA for the live sigma estimate.

- **Realized kernel**: State-of-the-art for noisy tick data. Overkill here -- the microstructure noise at 1-second BTC sampling is small. Adds implementation complexity.

- **HAR model**: Excellent for daily vol forecasting from intraday data. For intra-5-minute forecasting, the multi-scale structure (5min/1hr/1day components) doesn't apply because the bot only has 90-300s of data per window.

- **ML (XGBoost)**: Could work but requires training infrastructure, labeled forward-sigma data, and ongoing retraining. Marginal gains over EWMA unlikely to justify complexity for sigma estimation (as opposed to the full P(UP) prediction, where ML adds more).

### Expected Stability Improvement

| Metric | Current (YZ + Kalman) | Proposed (EWMA + Clip) |
|--------|----------------------|----------------------|
| Max sigma change per tick | 17% (observed) | 5% (hard cap) |
| p_model oscillation per tick | 3-4 pp | <1.5 pp |
| False threshold crossings | Frequent | Rare |
| Adaptation to real vol spike | ~3 seconds | ~15 seconds |
| 1-step forecast MSE | Baseline | 25-39% lower |

The tradeoff: 15 seconds to adapt to a real vol spike (vs ~3 seconds currently) means the bot may be slightly slow to recognize a regime change. But this is MUCH better than the current problem of firing trades on estimator noise. The min_sigma/max_sigma bounds and adaptive floor provide hard stops for extreme regimes regardless of EWMA speed.

### Implementation Complexity

**Low**. The changes are:

1. In `DiffusionSignal._compute_vol()`: when `sigma_estimator == "ewma"`, call `ewma_sigma_per_s()` (already implemented in `scripts/sigma_estimators.py`).

2. In `DiffusionSignal._smoothed_sigma()`: add rate-limiting before/after the Kalman filter:
   ```python
   # Rate-limit raw sigma change
   prev = ctx.get("_sigma_prev", raw_sigma)
   max_delta = 0.05 * prev
   raw_sigma = max(prev - max_delta, min(prev + max_delta, raw_sigma))
   ctx["_sigma_prev"] = raw_sigma
   ```

3. In `market_config.py`: change `sigma_estimator="yz"` to `sigma_estimator="ewma"` for BTC markets.

4. **Critical**: Re-tune `edge_threshold` and `kelly_fraction` against EWMA-flavored sigma. The prior ablation showed EWMA hurts PnL with YZ-tuned parameters. A parameter sweep is required.

### Validation (Backtest Methodology)

1. **A/B backtest**: Run full backtest on BTC 5m and BTC 15m parquets with:
   - Arm A: current YZ + Kalman (baseline)
   - Arm B: EWMA(lambda=0.90) + 5% clip + same Kalman
   - Arm C: EWMA(lambda=0.90) + 5% clip + NO Kalman (bypass)
   Compare: PnL, Sharpe, max drawdown, trade count, win rate

2. **Sigma stability diagnostic**: For each arm, log sigma_per_s at every tick. Compute:
   - tick-to-tick sigma change distribution (should tighten dramatically)
   - autocorrelation of sigma changes (should decrease -- less oscillation)
   - max absolute sigma change per window

3. **Forward sigma forecast MSE**: At each tick t, record sigma(t). At t+60s, record realized sigma over [t, t+60s]. Compute MSE(sigma_forecast - sigma_realized) per estimator.

4. **Edge threshold sweep**: For the winning sigma estimator, sweep edge_threshold from 0.02 to 0.12 in steps of 0.01. Find the Sharpe-maximizing threshold. This must be done AFTER choosing the estimator because the optimal threshold depends on the sigma flavor.

5. **Live A/B** (optional): Run two instances in parallel on the same market feed. One with YZ, one with EWMA. Compare decisions tick-by-tick to quantify divergence.

---

## Appendix A: Why EWMA Lambda = 0.90

The half-life of an EWMA with parameter lambda is h = -1/ln(lambda):
- lambda = 0.94: h = 16.1 ticks (~16 seconds at 1-tick-per-second)
- lambda = 0.90: h = 9.5 ticks (~10 seconds)
- lambda = 0.85: h = 6.2 ticks (~6 seconds)

For a 5-minute prediction horizon, we want sigma to reflect the "current" vol regime. Research on crypto volatility shows regime changes happen on the scale of minutes, not hours (low persistence compared to equities). A 10-second half-life means:
- 50% of the weight is on the last 10 seconds
- 75% of the weight is on the last 20 seconds
- 90% of the weight is on the last 30 seconds
- 99% of the weight is on the last 46 seconds

This effectively creates a ~45-second "soft lookback" that is shorter than the current 90s hard window but without the edge effects. The exponential decay means old observations fade gracefully rather than falling off a cliff.

Lambda = 0.90 is a starting point. It should be validated via the forecast MSE methodology in the backtest plan.

## Appendix B: Why Not Freeze Sigma Per Window

An alternative: compute sigma ONCE when enough data accumulates (e.g., at tau=240s for a 300s window) and freeze it for the rest of the window.

**Pros**: Eliminates ALL oscillation. z depends only on price and tau going forward.
**Cons**: If vol truly changes mid-window (BTC moves $500 in a minute), the frozen sigma is wrong for the remainder. Also, the "frozen" approach doesn't work well for maker mode, which needs to continuously requote.

A hybrid: compute sigma continuously via EWMA but "lock" the trade decision for N seconds after each sigma update. This is effectively what the rate-limiter does.

## Appendix C: The Bayesian Model-Averaging Alternative

Instead of choosing a single sigma, maintain a posterior distribution p(sigma | data) and compute:

P(UP) = integral Phi(delta / (sigma * sqrt(tau))) * p(sigma | data) d_sigma

Under a conjugate inverse-gamma prior for sigma^2, this integral has a closed-form solution involving the Student-t CDF. With nu = 2*alpha degrees of freedom (where alpha is the shape parameter of the inverse-gamma posterior), this gives:

P(UP) = T_nu(delta / (sigma_hat * sqrt(tau)))

where T_nu is the Student-t CDF and sigma_hat is the posterior mean.

This is elegant: model uncertainty about sigma is automatically handled by fatter tails. When data is scarce (few observations), nu is low and tails are fat (underconfident). When data is abundant, nu is high and the Student-t converges to the normal (confident).

The bot already supports `tail_mode="student_t"` with an estimated nu from excess kurtosis. The Bayesian interpretation suggests that nu should also depend on the SAMPLE SIZE of the sigma estimate, not just the kurtosis. This could be a future enhancement.

---

## Sources

- [Andersen, Bollerslev, Diebold, Labys (2000)](https://public.econ.duke.edu/~get/browse/courses/201/spr11/DOWNLOADS/VolatilityMeasures/SpecificlPapers/hansen_lunde_forecasting_rv_11.pdf) - Realized volatility optimal sampling
- [Barndorff-Nielsen & Shephard (2004)](https://academic.oup.com/jfec/article-abstract/2/1/1/960705) - Bipower variation
- [Barndorff-Nielsen, Hansen, Lunde, Shephard (2008)](https://onlinelibrary.wiley.com/doi/abs/10.3982/ECTA6495) - Realized kernels
- [Ding & Meade (2021)](https://arxiv.org/abs/2105.14382) - Optimal EWMA decay parameter
- [Cryptocurrency Volatility Comparison (2024)](https://arxiv.org/html/2404.04962v1) - Crypto vs equity vol characteristics
- [Stylized Facts of High-Frequency Bitcoin (2025)](https://www.mdpi.com/2504-3110/9/10/635) - BTC distributional properties
- [Range-based Volatility Estimators](https://portfoliooptimizer.io/blog/range-based-volatility-estimators-overview-and-examples-of-usage/) - Parkinson, GK, RS, YZ comparison
- [Yang & Zhang Estimator (2024)](https://www.sciencedirect.com/science/article/pii/S2665963824000010) - YZ implementation reference
- [EWMA Volatility for Crypto Portfolios](https://www.tandfonline.com/doi/full/10.1080/14697688.2022.2159505) - EWMA performance in crypto
- [Bayesian Market Maker](https://people.cs.vt.edu/~sanmay/papers/bmm-ec.pdf) - Bayesian probability updating in prediction markets
- [Hybrid ML + Stochastic Vol for Crypto](https://link.springer.com/article/10.1007/s44257-025-00046-1) - Combined approaches
- [HAR Model Overview](https://medium.com/@simomenaldo/realized-volatility-and-har-models-a-new-paradigm-for-volatility-forecasting-4a660f2530f3) - Heterogeneous autoregressive model
