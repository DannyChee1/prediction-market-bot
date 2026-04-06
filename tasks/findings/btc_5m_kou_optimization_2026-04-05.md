# BTC 5m Kou Parameter Optimization

**Date:** 2026-04-05
**Data:** 1,039 complete BTC 5m windows, 27,838 observations

## Result

| Config | lambda | p_up | eta1 | eta2 | Brier | ECE | LogLoss |
|--------|--------|------|------|------|-------|-----|---------|
| Current | 0.100 | 0.526 | 1254.1 | 1200.5 | 0.2174 | 0.1562 | 0.6211 |
| No jumps (lambda=0) | 0 | - | - | - | 0.1885 | 0.0586 | 0.5569 |
| **Recommended** | **0.005** | **0.526** | **1254.1** | **1200.5** | **0.1888** | **0.0555** | **0.5575** |
| Grid-optimal | 0.008 | 0.480 | 5000 | 3000 | 0.1889 | 0.0482 | 0.5581 |

**ECE reduction: 0.1562 -> 0.0555 (64% improvement)**

## Key Insight

lambda=0.100 was 14x too high. The Kou drift correction was massively over-correcting, creating a systematic DOWN bias in every probability bin.

The grid-optimal (lambda=0.008, eta1=5000, eta2=3000) is marginally better than the simple fix, but the huge eta values make jumps ~0.02% in size — effectively degenerate to normal CDF. The Kou model at these parameters adds almost nothing over plain Normal CDF (ECE 0.0482 vs 0.0586).

## Recommendation

**Set lambda=0.005, keep all other params unchanged.** Rationale:
- Close to BTC 15m's lambda=0.007 (same asset, just different timeframe)
- Minimal param changes = less overfitting risk
- ECE=0.0555 is within noise of the grid-optimal 0.0482
- No need to change eta1/eta2/p_up — they barely matter at low lambda

## Lambda Sweep (full results)

| lambda | ECE | Brier |
|--------|-----|-------|
| 0.000 | 0.0586 | 0.1885 |
| 0.001 | 0.0566 | 0.1885 |
| 0.005 | 0.0555 | 0.1888 |
| 0.007 | 0.0564 | 0.1889 |
| 0.010 | 0.0585 | 0.1892 |
| 0.020 | 0.0629 | 0.1904 |
| 0.050 | 0.0971 | 0.1972 |
| 0.100 | 0.1562 | 0.2174 |
| 0.150 | 0.2167 | 0.2458 |

ECE degrades roughly linearly with lambda above 0.01. The sweet spot is 0.001-0.007.

## Plots
- `kou_5m_optimization.png` — lambda vs ECE curve + reliability diagrams (old vs optimized)
