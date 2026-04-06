# Kalman Sigma Regime Lag — 2026-04-05

## Problem

When volatility regime shifts (e.g., quiet Asia → US open), the Kalman filter lags behind because:
1. The 90s vol lookback still contains old-regime prices
2. The Kalman smoothing adapts gradually, not instantly

## Simulation: 2.3x Vol Regime Shift (quiet → US open)

σ jumps from 0.000060 (10:00 UTC quiet) to 0.000140 (14:00 UTC US open).

| Time After Shift | Kalman σ | True σ | Error | Impact |
|-----------------|----------|--------|-------|--------|
| 0s | 0.000060 | 0.000140 | 57% | z inflated 2.3x |
| 30s | 0.000086 | 0.000140 | **38%** | z inflated 1.6x |
| 60s | 0.000113 | 0.000140 | 19% | z inflated 1.2x |
| 75s | — | — | <10% | Acceptable |
| 90s | 0.000140 | 0.000140 | <1% | Fully adapted |

## Impact by Market

### BTC 15m: SAFE
- maker_warmup = 200s
- Kalman fully adapts by ~90s
- By the time we trade, sigma is accurate
- No action needed

### BTC 5m: VULNERABLE
- maker_warmup = 30s
- At 30s, Kalman sigma is **38% too low**
- z-scores inflated by **1.6x** → model overestimates edge
- Could trigger false entries on the first 5m window after a vol regime shift
- Affects ~2-4 windows per day (at major session transitions)

## Kou Self-Correction (Partial Mitigation)

The Kou drift correction partially compensates:
- `drift_z = -lambda * zeta * sqrt(tau) / sigma`
- When sigma is underestimated, drift_z grows → model becomes MORE conservative
- This dampens the inflated z-scores somewhat
- Not a complete fix but reduces the severity

## Potential Fix (Not Implemented)

Use the hour/day vol lookup table as an **informative prior** for the Kalman, not just a zero-vol fallback. At each regime transition, blend the Kalman state toward the expected vol for the current hour:
```
kalman_prior = alpha * kalman_x + (1 - alpha) * time_prior_sigma(now)
```
This would give the Kalman a "head start" at regime boundaries.

## Practical Risk Assessment

- At $30 bankroll, 1 trade per window: worst case is one bad $2.25 5m entry per day
- At scale ($500+ bankroll): the 1.6x z inflation could cause oversized positions on false signals
- 15m is unaffected due to 200s warmup
- The risk is real but bounded — only matters at exact session transitions
