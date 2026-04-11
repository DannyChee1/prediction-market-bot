# Sigma Research: What's Wrong and What to Do About It -- 2026-04-11

## Executive Summary

**The original hypothesis was wrong.** The problem is not that sigma is systematically incorrect -- the model is actually *more accurate than the market* when sigma is low and z is high. The real problems are: (1) defensive gates block the best trades, (2) the edge calculation conflates direction accuracy with position sizing, and (3) a direct delta-based fair value (no sigma at all) outperforms the GBM model.

---

## 1. What's Wrong With Current Sigma

### The claimed failure mode
When BTC is quiet for 90s then moves $17:
- sigma = 2e-5 (Yang-Zhang on quiet period)
- z = 0.72 (delta=0.000202 / (2e-5 * sqrt(200)))
- p_up = 0.76
- Market says p_up = 0.58
- Disagreement = 0.18

### What the data actually shows

**Counter-intuitive finding: when sigma is low and |z| is high, the MODEL beats the MARKET on direction.**

From 2,986 backtest windows (tau ~ 150s), verified on large sample:

| Condition | n | Model Direction | Market Direction | Gap |
|-----------|---|-----------------|------------------|-----|
| sigma < 3e-5, |z| > 1.0 | 276 | **79.3%** | 69.6% | **+9.8pp** |
| sigma < 3e-5, |z| > 0.5 | 432 | **75.0%** | 69.7% | +5.3pp |
| All sigma, |z| > 1.0 | 944 | **88.6%** | 85.6% | +3.0pp |
| All sigma, all |z| | 2917 | **76.5%** | 75.2% | +1.2pp |
| Model/market disagree | 327 | **55.7%** | 44.3% | +11.4pp |

The model beats the market on direction accuracy across every regime, with the largest gap (+9.8pp) in the "failure mode" of low sigma and high z. When model and market disagree on direction, the model is right 55.7% of the time -- a statistically meaningful edge.

**Note**: An earlier run on 496 windows showed 97.7% model accuracy in the failure mode. This was a small-sample artifact. The large-sample result of 79.3% is more reliable but still confirms the model's directional superiority.

### Sigma comparison: YZ vs alternatives

| Metric | YZ sigma (current) | Sigma floor 5e-5 | Multi-timescale | Disagree gate |
|--------|-------------------|-------------------|-----------------|---------------|
| Trades (maker, e=0.06) | 238 | 228 | 236 | 198 |
| Win rate | **68.5%** | 65.4% | 66.9% | 63.6% |
| PnL | **$52.20** | $47.92 | $51.55 | $33.72 |
| $/trade | **$0.219** | $0.210 | $0.218 | $0.170 |

**YZ sigma already outperforms every alternative tested.** Raising the floor, using multi-timescale, or adding a disagreement gate all *reduce* performance.

### The sigma-implied paradox

Market-implied sigma = |delta| / (Phi^{-1}(mid_up) * sqrt(tau))

- Median implied/YZ ratio: 0.78 (implied sigma is typically *lower* than YZ)
- Correlation between implied and YZ: **-0.008** (essentially zero)
- YZ off by >2x from implied: 49.2% of windows

The two sigma estimates are essentially uncorrelated. This means:
1. The market is not using a GBM model (no surprise)
2. Market-implied sigma is tautological -- using it to compute fair_value just gives you back the market mid
3. You cannot derive useful forward-looking sigma from the Polymarket book

---

## 2. Alternative Sigma Methods: Analysis

### a) Market-implied sigma
- **How it works**: Derive sigma from Polymarket mid: sigma_impl = |delta| / (Phi^{-1}(mid_up) * sqrt(tau))
- **Solves quiet-then-move?** No. It's tautological -- fair_up from implied sigma = mid_up by construction.
- **Verdict: USELESS.** Using the market to calibrate sigma then using sigma to compute fair_value is circular. The result is always fair_value = market_mid.

### b) EWMA with regime detection
- **How it works**: Lambda=0.94 exponentially-weighted squared returns with regime reset on jump detection
- **Solves quiet-then-move?** Partially. The EWMA reacts faster than YZ to new data (11-tick half-life vs 90-tick window), but it still needs returns to exist. A single $17 jump from silence produces one large return that the EWMA incorporates over ~11 ticks.
- **Backtest evidence**: EWMA has 26% lower 1-step forecast MSE than YZ (validated in scripts/validate_sigma_estimators.py). BUT swapping to EWMA *hurts PnL* by $360-$1,675 because edge_threshold/kelly_fraction were tuned to YZ.
- **Verdict: NEUTRAL.** Better sigma forecast, worse PnL. Would need full parameter re-tune.

### c) Multi-timescale sigma (max of 30s and 90s)
- **How it works**: Blend fast (30s) and slow (90s) sigma, take the max for conservative estimation
- **Solves quiet-then-move?** Only marginally. Both windows are still backward-looking.
- **Backtest evidence**: 236 trades, 66.9% WR, $51.55 PnL. Slightly worse than plain YZ (68.5% WR, $52.20).
- **Verdict: MARGINAL.** No meaningful improvement. More complexity for less performance.

### d) Binance-derived realized vol (high-frequency)
- **How it works**: Use 100ms Binance trade data for higher-resolution vol
- **Current status**: Already in use. The bot uses binance_mid as effective_price when available, which feeds into the YZ/EWMA estimator.
- **Solves quiet-then-move?** No -- if BTC is genuinely quiet on Binance, the vol estimate is legitimately low.
- **Verdict: ALREADY IMPLEMENTED.** Further resolution improvements (10ms vs 100ms) would not change the fundamental issue.

### e) Order book implied vol
- **How it works**: Infer vol from the shape of the Polymarket CLOB (deep books at 0.50 = low vol, thin spread books = high vol)
- **Solves quiet-then-move?** Potentially -- the book should widen when participants expect vol.
- **Empirical evidence**: mid_up + mid_down = 1.0000 always. ask_up + bid_down complement gap = 0.0000 on average. The Polymarket book is **perfectly symmetric and complementary** -- there is no information in the book shape beyond the mid price.
- **Verdict: NOT VIABLE.** The Polymarket CLOB for BTC 5m is too efficient for order-book-implied vol to add information.

### f) Neural network / ML approach
- **How it works**: Train on (delta, tau, binance_price, book_state, trade_flow, time_of_day) -> P(UP) directly.
- **Solves quiet-then-move?** Yes, by design -- it can learn the correct mapping without going through sigma.
- **Analysis**: The empirical delta-based lookup table IS a simplified version of this. With 10 delta bins and 4 tau bins, it achieves 71.3% WR vs 68.5% for the GBM model. A proper ML model with more features could potentially do better.
- **Verdict: PROMISING but requires research investment.** The delta-based lookup shows the approach has merit. A proper ML model is a multi-week research project.

### g) Ensemble: market mid + model direction
- **How it works**: Use market mid as fair_value, only trade when model direction agrees with market
- **Solves quiet-then-move?** Sidesteps it entirely -- no sigma needed.
- **Empirical evidence**:
  - Using min(model_fair, market_fair): 14 trades, 92.9% WR, $5.46 PnL
  - Very selective (14 of 496 windows) but extremely accurate
  - When model and market agree: 11 trades, 90.9% WR
  - When model and market disagree: 3 trades, 100% WR (model is right!)
- **Problem**: Too few trades to be useful. The min(model, market) is very conservative.
- **Verdict: VIABLE as a supplementary filter** but not as primary strategy.

---

## 3. The Direct Delta-Based Approach (No Sigma At All)

### Empirical P(UP) lookup table

Built from 1,500 training windows, tested on 496 hold-out windows.

| Delta from threshold | P(UP) at tau=100s | P(UP) at tau=150s | P(UP) at tau=200s | P(UP) at tau=250s |
|---------------------|-------------------|-------------------|-------------------|-------------------|
| [-$200, -$100) | 0.022 | 0.051 | 0.036 | 0.167 |
| [-$100, -$50) | 0.055 | 0.101 | 0.146 | 0.148 |
| [-$50, -$25) | 0.112 | 0.193 | 0.196 | 0.208 |
| [-$25, -$10) | 0.274 | 0.246 | 0.361 | 0.411 |
| [-$10, $0) | 0.435 | 0.487 | 0.481 | 0.460 |
| [$0, +$10) | 0.626 | 0.609 | 0.554 | 0.539 |
| [+$10, +$25) | 0.764 | 0.672 | 0.710 | 0.742 |
| [+$25, +$50) | 0.853 | 0.833 | 0.754 | 0.695 |
| [+$50, +$100) | 0.951 | 0.901 | 0.822 | 0.750 |
| [+$100, +$200) | 0.964 | 0.962 | 0.923 | 0.895 |

### Key observations from the table:
1. P(UP) is **monotonically increasing with delta** at every tau -- as expected
2. P(UP) is **decreasing with tau at large |delta|** -- the longer you wait, the more time for mean reversion
3. P(UP) is **roughly 0.5 at delta=0** regardless of tau -- the market starts fair
4. At delta=$50 and tau=100s, P(UP) = 0.951 -- very predictable

### Trading simulation results

| Strategy | Trades | Win Rate | PnL | $/trade |
|----------|--------|----------|-----|---------|
| YZ sigma GBM (current, maker e=0.06) | 238 | 68.5% | $52.20 | $0.219 |
| Delta-based lookup (maker e=0.04) | 471 | 71.3% | $66.15 | $0.140 |
| Market mid as fair (maker) | 0 | - | - | - |
| Sigma-weighted blend | 227 | 67.4% | $49.65 | $0.219 |

The delta-based lookup achieves **higher win rate** (71.3% vs 68.5%) with **more trades** (471 vs 238). Total PnL is $66.15 vs $52.20 -- a 27% improvement. The per-trade edge is lower ($0.140 vs $0.219) because the lookup trades more aggressively, but the total PnL is higher.

---

## 4. Brier Score Comparison (Probability Calibration)

Measured at tau ~ 150s on 300 windows:

| Method | Brier Score | Log-Loss |
|--------|-------------|----------|
| **GBM model (YZ sigma)** | **0.1447** | **0.4476** |
| Market mid | 0.1844 | 0.5509 |
| Sigma-weighted blend | 0.1549 | 0.4814 |

**The GBM model is better calibrated than the market mid** overall. The model has a lower Brier score (0.1447 vs 0.1844) and lower log-loss.

### By sigma regime:

| Regime | Model Brier | Market Brier | Winner |
|--------|-------------|--------------|--------|
| Low sigma (< 3e-5) | **0.0811** | 0.2195 | Model by 2.7x |
| Mid sigma (3e-5 to 1e-4) | **0.1635** | 0.1859 | Model by 14% |
| High sigma (> 1e-4) | **0.1575** | 0.1613 | Model by 2% |

**The model is MOST accurate when sigma is LOW** (Brier 0.081 vs market's 0.220). This is the exact opposite of the initial hypothesis. When BTC is quiet and then moves, the model correctly identifies the direction with high confidence, while the market is slow to reprice.

---

## 5. The Real Problems (Not Sigma)

### Problem 1: The disagreement gate blocks the best trades
The `max_model_market_disagreement=0.30` gate filters out every trade where |p_model - mid_up| > 0.30. But these are the trades where the model has a meaningful directional edge (79.3% accuracy vs market's 69.6% when sigma < 3e-5 and |z| > 1.0, and 55.7% model win rate when model/market disagree). The gate was designed to protect against sigma collapse, but the data shows the model retains its directional advantage even in those conditions.

### Problem 2: Edge calculation conflates direction and sizing
When sigma = 2e-5 and z = 3.0 (capped):
- p_model = 0.999
- Edge vs ask at 0.42 = 0.999 - 0.42 = 0.58 (58 cents!)
- Kelly formula sizes a huge position on this "edge"

The direction is right (~79-89% of the time, depending on sigma regime) but the edge is meaninglessly large. The fair_value of 0.999 implies a certainty that doesn't exist at tau=200s. The *direction* information is valid but the *magnitude* is not calibrated for sizing.

### Problem 3: The Kalman smoother delays catch-up
When sigma jumps from quiet to volatile, the Kalman filter takes 60-90s to adapt (documented in kalman_regime_lag findings). During that window, z is inflated and the model looks overconfident relative to the market.

### Problem 4: Live bot has additional gates not in backtest
The live bot applies edge_persistence (10s), max_model_market_disagreement (0.30), vol warmup (60-90s), stale-book gate (5000ms), and other filters that don't appear in backtest. The backtest numbers above don't capture these live-only gates, which further reduce the already-marginal edge.

---

## 6. Concrete Recommendations (Ranked)

### R1: Implement direct delta-based fair value (HIGH IMPACT, LOW EFFORT)

Replace the GBM sigma->z->CDF pipeline with a direct lookup:

```python
# In signal_diffusion.py or a new fair_value module:
DELTA_BINS = [-200, -100, -50, -25, -10, 0, 10, 25, 50, 100, 200]
TAU_BINS = [100, 150, 200, 250]
EMPIRICAL_P_UP = {
    # (delta_bin_idx, tau_bin_idx): p_up
    # Populated from historical backfill data
}

def delta_fair_value(delta_usd: float, tau: float) -> float:
    """Look up P(UP) from empirical (delta, tau) table."""
    # Interpolate between nearest bins
    ...
```

**Evidence**: 71.3% WR vs 68.5% for current model, 27% more total PnL.
**Effort**: 2-4 hours. Build lookup table from existing parquet data, add as `fair_value_mode="delta_lookup"` config option.
**Risk**: The lookup table needs periodic recalibration as BTC volatility regime changes. Use walk-forward recalibration weekly.

### R2: Relax or remove the disagreement gate for stale-quote mode (HIGH IMPACT, 5 MIN)

The `max_model_market_disagreement=0.30` gate filters out the trades where the model is *most* accurate. For stale-quote mode specifically (which is designed to exploit model-market disagreements), this gate is counterproductive.

```python
# In market_config.py for btc_5m stale_quote:
# Either remove the gate:
max_model_market_disagreement=1.0  # disabled
# Or widen it significantly:
max_model_market_disagreement=0.50
```

**Evidence**: On 2,986 windows, model wins 55.7% of all model-market disagreements (n=327). At low sigma with high |z|, model direction accuracy is 79.3% vs market's 69.6%.
**Risk**: During genuinely broken sigma periods (e.g., feed disconnect), removing the gate could cause bad trades. Keep the stale-feature gates (book_age, binance_age) as the safety net.

### R3: Decouple direction from sizing (MEDIUM IMPACT, MEDIUM EFFORT)

Use model for direction signal, market mid for position sizing:

```python
# Direction from model (reliable)
model_direction = "UP" if z > 0 else "DOWN"
model_confidence = abs(z)  # use as filter, not for sizing

# Fair value for edge/sizing from market
fair_up_for_sizing = market_mid_up  # or delta_lookup
edge = fair_up_for_sizing - bid_up  # always half-spread for maker

# Only trade when model is confident AND market has exploitable structure
if model_confidence > min_z and edge > edge_threshold:
    size = kelly(edge, fair_up_for_sizing)  # size on market edge, not model edge
```

**Evidence**: Approach E (model direction + market fair value) achieved 92.9% WR on 14 trades. Very selective but very accurate.
**Effort**: 4-8 hours to implement and test.

### R4: Keep YZ sigma, don't change the estimator (DO NOTHING)

The YZ sigma estimator outperforms EWMA, multi-timescale, and sigma-floor alternatives in trading simulation. The EWMA has better forecast MSE but worse PnL because the downstream parameters are tuned to YZ. Changing the estimator without a full parameter re-tune would be net negative.

**If** a full parameter re-tune is planned anyway, then switch to EWMA and re-optimize edge_threshold, kelly_fraction, and min_entry_z simultaneously.

### R5: Build a proper ML model (HIGH IMPACT, HIGH EFFORT)

The delta-based lookup table is a proof of concept. A proper model would include:
- Features: delta_usd, tau, sigma, mid_up, spread_up, imbalance5_up, binance_mid velocity, Hawkes intensity
- Target: P(UP) (classification) or expected_pnl (regression)
- Method: LightGBM or logistic regression with walk-forward cross-validation
- Calibration: isotonic regression or Platt scaling

**Expected improvement**: The lookup table already beats GBM by 2.8pp WR. A proper ML model with more features could plausibly add another 2-5pp.
**Effort**: 1-2 weeks including validation. Research project, not a parameter tweak.

---

## 7. The Cross-Contract Idea

### Does using one contract's mid to infer the other's fair value work?

**No.** The Polymarket CLOB maintains perfect complementarity:
- mid_up + mid_down = 1.0000 (max gap: 0.005, mean gap: 0.00005)
- ask_up + bid_down gap: mean 0.0000 (perfectly tight)
- bid_up + ask_down gap: max 0.0000

The UP and DOWN books move in lockstep. There is never a persistent dislocation between them. This makes sense: any such dislocation would be instantly arbitraged by buying both sides for < $1 and merging.

**The insight from the prompt was almost right but backwards**: the correct approach is not "use UP mid to trade DOWN" (they're already linked). The correct approach is "use Binance price to trade BOTH sides" (which is what BoneReaper does, and what the stale-quote mode already implements).

---

## 8. What Market-Implied Sigma Actually Tells Us

Market-implied sigma (derived from mid_up via inverse CDF) is **uncorrelated** with YZ sigma (correlation = -0.008). This means:

1. The market is NOT pricing contracts using a GBM model with any recognizable sigma
2. Market prices reflect order flow, sentiment, and other information that has nothing to do with realized volatility
3. There is no "correct" sigma to extract from the market -- the market doesn't think in sigma

This is consistent with the comprehensive strategy review finding: "the market incorporates order flow, cross-asset info, and microstructure that our single-asset GBM cannot see."

---

## 9. The Bigger Picture

The sigma estimation question is a symptom of a deeper architectural choice: using a parametric GBM model for a market that doesn't behave like GBM.

### What the GBM model does well:
- Direction prediction when z is large (>1.0): 89% accuracy (vs market 86%)
- Filtering low-signal periods (z near 0): correctly abstains

### What it does poorly:
- Fair value estimation (the *magnitude* of p_model is miscalibrated)
- Edge calculation (conflates direction confidence with probability precision)
- Interaction with defensive gates (gates designed for bad sigma kill good trades)

### The path forward:
1. **Short term**: Use delta-based lookup or market mid for sizing; use model for direction only
2. **Medium term**: Build a proper ML model that directly predicts P(UP) from features
3. **Long term**: Follow BoneReaper's lead -- monetize Binance-Polymarket dislocations directly without trying to predict the future

---

## 10. Specific Numbers for the Scenario in the Prompt

BTC quiet 90s, then moves $17. sigma=2e-5, tau=200s.

| What? | sigma | z | p_up | Notes |
|-------|-------|---|------|-------|
| Current model | 2e-5 | 0.72 | 0.763 | Direction correct ~79% at low sigma |
| Market mid | - | - | 0.58 | Slow to reprice |
| Sigma floor 3e-5 | 3e-5 | 0.48 | 0.683 | Closer to market |
| Sigma floor 5e-5 | 5e-5 | 0.29 | 0.613 | Very close to market |
| **Sigma floor 7e-5** | **7e-5** | **0.20** | **0.581** | **Almost exactly = market** |
| Delta lookup | - | - | ~0.55-0.61 | From empirical table |
| "Agree" sigma | 7.1e-5 | 0.20 | 0.580 | What makes model = market |

A sigma floor of 7e-5 would make the model agree with the market in this scenario. But the data shows the model (at sigma=2e-5) is *more accurate* than the market. Forcing agreement makes the model *worse*, not better.

**The model saying 0.76 when the market says 0.58 is not a bug. It's the model being right faster than the market.**

The real fix is not to make the model agree with the market. It's to:
1. Trust the model's direction
2. Use a realistic fair value for sizing (not 0.999 from capped z)
3. Remove the gates that prevent trading on this signal

---

## 11. Implementation Plan

### Phase 1: Quick wins (1-2 hours)

1. **Widen disagreement gate in stale-quote mode**: Set `max_model_market_disagreement=0.50` for btc_5m stale-quote config (currently 0.30). This alone lets the highest-edge trades through.

2. **Add edge cap sanity**: The existing `max_edge=0.15` cap (commit c5acb8c) is good. Verify it's applied consistently in both decide_both_sides and decide_stale_quote.

### Phase 2: Delta-based fair value (4-8 hours)

1. Build calibration script: process all parquet files, compute empirical P(UP | delta_bucket, tau_bucket) with walk-forward splitting.
2. Add `fair_value_mode="delta_lookup"` to MarketConfig.
3. In decide_stale_quote, replace `fair_up = self._model_cdf(z, ctx)` with `fair_up = delta_lookup(delta_usd, tau)` when configured.
4. Backtest both modes side-by-side on held-out data.

### Phase 3: Direction-sizing decoupling (1 day)

1. Add `sizing_mode="market"` config: use market mid for Kelly sizing, model for direction filter.
2. Edge = market_mid - bid (maker) or market_mid - ask - fee (taker).
3. Only trade when model agrees on direction (z > min_z in the same direction as market_mid > 0.5).

### Phase 4: ML model (1-2 weeks, research)

1. Feature engineering: delta, tau, sigma, spread, imbalance, Hawkes intensity, time_of_day
2. Target: P(UP) with walk-forward cross-validation
3. Calibration: isotonic regression
4. A/B test against GBM and delta-lookup on live or paper trading
