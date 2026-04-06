# Comprehensive Strategy Review — 2026-04-05

## Executive Summary

Empirical analysis of 4,064 parquet files reveals our GBM diffusion model is **less accurate than the Polymarket order book**. When model disagrees with market, the market wins. Our edge is NOT in model accuracy — it's in **speed at window resolution**. The final 10-60 seconds contain a huge exploitable gap where the oracle reveals the outcome before the market fully reprices.

---

## Finding 1: OUR MODEL LOSES TO THE MARKET (Critical)

When our GBM model disagrees with the market mid-price:

| Condition | n | Win Rate |
|-----------|---|----------|
| Model disagrees, oracle just moved (speed) | 482,500 | 46.0% |
| Model disagrees, market settled (pure model) | 10,567 | **29.8%** |

Win rate by disagreement magnitude (model-only events):
| Staleness | n | WR |
|-----------|---|-----|
| 2-5% | 1,698 | 40.5% |
| 5-10% | 2,313 | 39.6% |
| 10-20% | 3,036 | 32.1% |
| >20% | 3,520 | **16.3%** |

**Conclusion**: The bigger we think our edge is, the more wrong we are. The market incorporates order flow, cross-asset info, and microstructure that our single-asset GBM cannot see. **Do not fight the market on model conviction alone.**

---

## Finding 2: RESOLUTION SNIPE — THE REAL EDGE (Critical)

In the final seconds of each window, the Chainlink oracle already reveals the outcome, but the market takes seconds to reprice.

**BTC 15m — final 60 seconds:**
| Time Remaining | "Obvious" Accuracy | Market Accuracy | Disagree % |
|---------------|-------------------|-----------------|------------|
| 0-10s | **97.6%** | 92.3% | 9.2% |
| 10-20s | 96.6% | 91.1% | 6.6% |
| 20-30s | 96.6% | 89.9% | 8.0% |
| 30-45s | 94.6% | 88.4% | 8.1% |
| 45-60s | 91.4% | 84.2% | 9.9% |

When "obvious" (current price > start) DISAGREES with market (mid > 0.5):
- **BTC 15m**: "Obvious" wins **87.3%** of the time (n=260)
- **BTC 5m**: "Obvious" wins **63.2%** of the time (n=745)

**This is the single largest exploitable edge.** ~9% of windows have a stale market in the final minute. When stale, we know the outcome with 87-97% confidence.

---

## Finding 3: MARKET REPRICING LAG (High Impact)

After an oracle price update, the order book takes:
| Metric | BTC 15m | BTC 5m |
|--------|---------|--------|
| Median reprice | 1.0s | 0.0s |
| p75 reprice | 8.0s | 2.0s |
| p90 reprice | 28.2s | 15.0s |
| <5s | 66.7% | 82.0% |
| <10s | 77.9% | 87.8% |

**22% of repricing events take >10 seconds for 15m markets.** This is the window we can exploit — especially near window end.

---

## Finding 4: CROSS-ASSET CORRELATION (Moderate Impact)

| Pair | Direction Correlation | Same Direction % | n |
|------|----------------------|-------------------|---|
| BTC/ETH 15m | r=0.569 | 78.5% | 390 |
| BTC/ETH 5m | r=0.540 | 77.1% | 879 |
| Prev BTC → Curr ETH 5m | — | 51.8% | 878 |

Same-window correlation is **very strong** (78.5%). Previous-window prediction is useless (51.8%). The cross-asset play: when BTC clearly moving, trade ETH before ETH market reprices. Most useful near window end.

---

## Finding 5: WITHIN-WINDOW MOMENTUM (Moderate Impact)

Early direction → same end direction:
| % Elapsed | BTC 15m Continuation | BTC 5m Continuation |
|-----------|---------------------|---------------------|
| 10% | 67.9% | 58.2% |
| 20% | 69.3% | 53.5% |
| 30% | **49.8%** | 63.7% |
| 50% | 70.0% | 66.6% |

**15m pattern**: momentum at 10-20% elapsed, mean-reversion dip at 30%, then re-establishing at 50%. Entry timing matters — avoid the 30% window.

---

## Finding 6: WINDOW CHOPPINESS (Context)

| Metric | BTC 15m | BTC 5m |
|--------|---------|--------|
| Mean crossovers | 69.7 | 13.7 |
| Straight-shot (0 crossovers) | 9.4% | 13.0% |
| Choppy (>10 crossovers) | **82.7%** | 43.0% |

Most 15-minute windows flip direction many times. Early-window z-scores are noisy. This confirms why our mid-window model trades underperform.

---

## Finding 7: ORACLE LAG IS TINY (Context)

| Metric | Value |
|--------|-------|
| Median Chainlink vs Binance lag | 0.0199% |
| p90 | 0.0271% |
| p99 | 0.0435% |
| >0.1% lag | 0.1% of samples |

The oracle is surprisingly close to Binance (2bp median lag). This means we can't profit much from oracle lag in the middle of windows — only at the critical resolution moment.

---

## Finding 8: SPREADS AND MARKET STRUCTURE (Context)

- UP/DOWN mid prices sum to exactly 1.0000 (no structural arb)
- Typical spread: 1.00-1.10 cents per side (very tight)
- No buy-both-sides or sell-both-sides arb ever observed
- Break-even requires consistent edge of >0.5% as maker, >3% as taker

---

## Finding 9: DRIFT BIAS (Context)

| Asset | n | Actual Up % | Market Implied | Bias |
|-------|---|-------------|----------------|------|
| btc_15m | 556 | 47.5% | 50.0% | -2.6% downtrend |
| btc_5m | 1238 | 47.8% | 50.0% | -2.1% downtrend |
| eth_15m | 390 | 48.7% | 50.2% | -1.5% flat |
| eth_5m | 879 | 47.0% | 50.2% | -3.2% downtrend |

During our data collection period, all assets had a slight downtrend bias. The market always prices near 50% at window start, not accounting for short-term drift.

---

## Research Findings (from literature search)

1. **Kou jump-diffusion** outperforms GBM for BTC option pricing (lower pricing errors). Paper: "Toward Black-Scholes for Prediction Markets" (arxiv 2510.15205) proposes logit jump-diffusion specifically for prediction markets.
2. **HAR model** consistently outperforms GARCH for crypto realized volatility at short horizons.
3. **VPIN** in crypto averages 0.45-0.47 (vs 0.22-0.23 equities). AUC 0.54-0.61 for predicting future price jumps.
4. **Polymarket repricing delay** documented at ~55 seconds in published analysis.
5. **Cross-platform arbitrage**: Kalshi sometimes leads Polymarket during sudden moves. $40M arb profit extracted Apr 2024 - Apr 2025.

---

## Ranked Strategy Recommendations

### Tier 1: Highest Impact — Implement First

**A. Resolution Snipe (NEW STRATEGY)**
- In final 30-60s, if `current_price > start_price` but `mid_up < 0.50` (or vice versa), buy the "obvious" side
- Expected WR: 87-97% (15m), 63% (5m)
- Requires: fast execution (<1s after oracle update), watching the last 60s closely
- This bypasses model accuracy entirely — it's a pure speed play on the known outcome
- Implementation: new mode in `decide_both_sides()` for `time_remaining < 60`

**B. Stop Fighting the Market (MODEL CHANGE)**
- Blend: `p_final = 0.25 * p_model + 0.75 * market_mid`
- Only trade when p_final and market_mid agree on direction
- When they disagree: no trade (current model trades these at 30% WR)
- Implementation: add `market_blend` parameter to DiffusionSignal

### Tier 2: High Impact

**C. Fix Sigma Estimation**
- Use Binance mid for sigma when available (fixes 35% overestimate)
- Fall back to Chainlink-deduped when Binance data unavailable
- Implementation: modify `_compute_vol_deduped()` to prefer `binance_mid` column

**D. Time-Based Entry Gating**
- Avoid entries before 30% elapsed (choppy, 50% continuation at 30% for 15m)
- Focus entries at >50% elapsed (70% continuation)
- Best entries: >70% elapsed with clear direction = resolution snipe
- Implementation: add `min_elapsed_frac` parameter

### Tier 3: Moderate Impact

**E. Cross-Asset Resolution Snipe**
- When BTC clearly going UP at 80%+ elapsed, check if ETH UP is underpriced
- 78.5% correlation = strong confirmation
- Implementation: extend cross_asset_z_lookup for end-of-window confirmation

**F. VPIN-Based Trade Suppression**
- Suppress trades when VPIN > 0.60 (informed flow = someone knows something)
- Implementation: already have VPIN, add threshold gate

**G. Drift Correction**
- Estimate short-term drift from recent windows (not just current window)
- Adjust base probability from 50% to reflect recent trend
- Implementation: track rolling win rate of recent N windows as drift prior

### Not Recommended

- **Cross-platform arb (Kalshi)**: Infrastructure complexity for marginal gain
- **Previous-window prediction**: BTC prev → ETH curr = 51.8%, basically noise

---

## UPDATE: Kou Jump-Diffusion Results (same day)

**Earlier recommendation to NOT use jump-diffusion was WRONG.** Testing revealed:

### Kou Outperforms GBM on BTC 15m
| Model | PnL | WR | Sharpe |
|-------|-----|-----|--------|
| Gaussian | $3,613 | 53.4% | 0.48 |
| Kou (λ=0.007) | $17,167 | 58.3% | 1.31 |

The Kou drift correction acts as a **sigma-adaptive confidence filter** — more conservative when vol is low or early in window (when model is least reliable). This is fundamentally different from static edge thresholds.

### Kou + High min_z Makes BTC 5m Profitable
5m was unprofitable for ALL models at default settings. Kou with aggressive λ=0.100 + min_z≥0.7 flips it to +$7.7k. Static edge thresholds alone never turn 5m profitable — the adaptive dampening is key.

### Monte Carlo Edge Test: STRONG EVIDENCE (p < 0.01)
Permutation test (10,000 shuffles) confirms edge is real for both 15m and 5m:
- BTC 15m: p < 0.0001, WR 56.5%, edge +6.9pp over break-even
- BTC 5m (min_z=0.7): p < 0.0001, WR 42.6%, edge +4.9pp over break-even

### Paper Trading Confirmation
52 live paper trades: 63.5% WR, $679 PnL. |z|≥1.0 bucket: 76.9% WR.

See: `model_comparison_results_2026-04-05.md` and `monte_carlo_edge_test_2026-04-05.md` for full details.
