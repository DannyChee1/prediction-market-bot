# Distribution Fit Analysis — 2026-04-05

## Question
Does Student-t (nu=3.0, our current model) match the actual window-level return distribution?

## Method
- Loaded all parquet files across btc/eth/sol/xrp × 5m/15m (4,064 files, ~6.7M rows)
- Computed end-of-window log-return z-scores: `z = log(P_end / P_start) / (sigma_per_s * sqrt(elapsed_s))`
- Sigma estimated from non-zero 1-second log-returns within each window
- Fitted Student-t and Gaussian via MLE; compared with KS test

## Key Findings

### 1. Window-level distribution is near-Gaussian, not heavy-tailed

| Asset   | n    | Excess Kurt | Fitted nu       | Reject Gaussian? |
|---------|------|-------------|-----------------|-----------------|
| btc_15m | 515  | 0.05        | 365 (≈ Gaussian)| barely p=0.010  |
| btc_5m  | 1107 | 0.53        | 22.9            | no (p=0.183)    |
| eth_15m | 369  | 1.41        | 12.6            | no (p=0.147)    |
| eth_5m  | 786  | 0.89        | 15.3            | no (p=0.112)    |
| sol_15m | 118  | -0.80       | ≫ 1000 (sub-Gaussian) | no     |
| sol_5m  | 356  | -0.06       | ≫ 1000          | no              |
| xrp_15m | 114  | -1.00       | ≫ 1000 (sub-Gaussian) | no     |
| xrp_5m  | 342  | -0.47       | ≫ 1000          | no              |

SOL/XRP show **negative** kurtosis (platykurtic) — consistent with mean-reversion reducing extreme outcomes.

### 2. Our nu=3.0 Student-t has tails that are far too heavy
- Real data: nu ≈ 13–366 (or effectively Gaussian/sub-Gaussian)
- Model assumption: nu=3.0 (Cauchy-adjacent, infinite-variance adjacent)
- Effect: model **underestimates edge** at every z level vs what the data warrants

### 3. Sigma is overestimated by ~35%
- End-of-window z-score std ≈ 0.68–0.79 instead of 1.0
- Likely cause: estimating sigma from sparse oracle (Chainlink) updates within a single window — a few large discrete jumps inflate the estimate
- Effect: all z-scores are compressed by ~0.73× relative to "true" z
  - Our z=0.5 → true z ≈ 0.68
  - Our z=0.7 → true z ≈ 0.96

### 4. Model is more conservative than intended
Both effects (heavy-tailed distribution + sigma overestimation) independently make the model under-trade and under-price edge. We are leaving edge on the table, not over-fitting.

## Calibration note
`--calibrated` mode partially corrects for sigma overestimation (learns from actual outcomes), but was suspended after finding it inaccurate on short data history. Worth revisiting once more live data accumulates.

## Recommended fixes
1. **Short-term**: Bump `tail_nu_default` from 3.0 → 15.0 (better matches btc/eth empirical nu)
2. **Medium-term**: Switch to `tail_mode="normal"` for sol/xrp (data is sub-Gaussian)
3. **Root cause**: Improve sigma estimation — filter oracle-lag periods more aggressively or use Binance mid-price returns for sigma (much higher frequency, no discrete jumps)
