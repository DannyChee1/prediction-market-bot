# P(UP) Calibration Analysis — Pure Model CDF (No Bayesian Calibration)

**Date:** 2026-04-05
**Method:** Evaluated independent probability estimates across all 3,882 complete parquet windows (314K observations). No calibration table involved — pure CDF only (Kou for BTC, Student-t for ETH, Normal for SOL/XRP).

## Summary Table

| Market | Tail Model | N obs | UP% | Brier | ECE | LogLoss | Verdict |
|--------|-----------|-------|-----|-------|-----|---------|---------|
| SOL 15m | Normal | 27,243 | 49.0% | 0.1783 | 0.0319 | 0.5348 | Best overall |
| XRP 15m | Normal | 26,237 | 49.4% | 0.1899 | 0.0235 | 0.5624 | Best calibrated (lowest ECE) |
| XRP 5m | Normal | 9,214 | 47.1% | 0.1902 | 0.0239 | 0.5623 | Clean |
| SOL 5m | Normal | 9,530 | 47.9% | 0.1810 | 0.0253 | 0.5416 | Clean |
| BTC 15m | Kou | 114,003 | 51.0% | 0.1870 | 0.0373 | 0.5556 | Good — slight under-confidence in tails |
| ETH 5m | Student-t | 19,592 | 52.0% | 0.1849 | 0.0372 | 0.5499 | Good |
| ETH 15m | Student-t | 80,071 | 50.6% | 0.1922 | 0.0270 | 0.5676 | Overconfident on UP side at 0.70-0.80 |
| **BTC 5m** | **Kou** | **27,838** | **51.5%** | **0.2179** | **0.1678** | **0.6226** | **Broken — massive systematic DOWN bias** |

**Baseline:** Brier of always-predict-0.5 = 0.2500. All markets beat this.

## Key Findings

### 1. All markets have real predictive power
Brier scores 0.178-0.218, all well below the 0.25 coin-flip baseline. The GBM-family models genuinely predict outcomes.

### 2. BTC 5m Kou is badly miscalibrated
The Kou parameters for BTC 5m (`kou_lambda=0.100`, 14x the 15m value of 0.007) create an enormous negative drift correction. Every bin is systematically wrong:

| Model Predicts | Actual UP% | Error |
|---------------|-----------|-------|
| 15% | 24% | +9 |
| 25% | 45% | +20 |
| 35% | 58% | +23 |
| 45% | 68% | +23 |
| 55% | 76% | +21 |

ECE of 0.168 vs ~0.03 for all other markets.

### 3. ETH 15m has mild UP-side overconfidence
In the 0.70-0.80 bin, Student-t (nu=13) predicts 0.75 but actual is only 0.62. Fat tails at nu=13 may be too aggressive.

### 4. 15m predictions are bimodal due to max_z=1.0 cap
With 15m windows, z = delta / (sigma * sqrt(900)) saturates at +/-1.0 quickly, clustering predictions at ~0.16 and ~0.84 (= norm_cdf(+/-1)). Not a bug, but limits resolution in the tails.

### 5. Model improves near expiry (as expected)
BTC 15m Brier: 0.221 (600-900s remaining) -> 0.066 (0-150s remaining).

## Time-Remaining Breakdown

### BTC 15m
| Tau Bucket | N | Brier | ECE | UP% |
|-----------|---|-------|-----|-----|
| 0-150s | 4,355 | 0.0662 | 0.1189 | 51.0% |
| 150-300s | 10,610 | 0.1056 | 0.0856 | 51.1% |
| 300-600s | 38,499 | 0.1691 | 0.0359 | 51.1% |
| 600-900s | 60,539 | 0.2214 | 0.0458 | 51.0% |

### BTC 5m
| Tau Bucket | N | Brier | ECE | UP% |
|-----------|---|-------|-----|-----|
| 0-60s | 2,064 | 0.1043 | 0.1165 | 51.5% |
| 60-120s | 4,112 | 0.1524 | 0.1334 | 51.4% |
| 120-180s | 6,151 | 0.1957 | 0.1549 | 51.2% |
| 180-300s | 15,425 | 0.2583 | 0.2001 | 51.3% |

## Recommendations

1. **BTC 5m Kou lambda needs refitting** — biggest single calibration win available
2. **ETH 15m Student-t nu** — consider increasing from 13 to 18-25
3. **max_z cap** — increasing to 1.5-2.0 would add tail resolution but needs ECE verification
4. **SOL, XRP, BTC 15m, ETH 5m** — leave alone, all well-calibrated

## Plots
- `calibration_reliability.png` — reliability diagrams per market
- `calibration_by_tau.png` — calibration stratified by time remaining
- `calibration_distributions.png` — prediction distributions colored by actual outcome

## Methodology Notes
- Volatility: Yang-Zhang realized vol from 5s OHLC micro-bars, 90s lookback
- Z-score: delta / (sigma * sqrt(tau)), capped at +/-1.0
- CDF: Kou jump-diffusion (BTC), Student-t (ETH), Normal (SOL/XRP)
- Sampling: every 30 rows per window after vol warmup
- Completeness: window_end_ms check, start gap < 30s
- NO Bayesian calibration table used — pure independent model CDF only
