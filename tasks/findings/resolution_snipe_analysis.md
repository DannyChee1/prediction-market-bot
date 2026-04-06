# Resolution Snipe — Corrected Analysis

## Original Claim (INCORRECT)
"87-97% WR in final 10-60s, $0.42 edge per contract"

## What Actually Happens

The "disagree" events (where current price direction differs from market mid) occur specifically when the **price move is tiny** — the price has barely crossed above/below start. The market correctly treats these as uncertain.

### Key Issue: Less Time = Less Movement = Less Conviction

With 30s remaining in a 15m window:
- Median price move at disagree events: **0.0000%** (price at start price)
- The "obvious" direction is NOT obvious — it's a rounding error
- 53.8% of prices flip after entry — essentially a coin flip on direction

### Corrected Results

| Market | Window | Events | WR | Median Move | Flip Rate | Verdict |
|--------|--------|--------|----|-------------|-----------|---------|
| BTC 15m | 30s | 13 | 100% | 0.000% | 53.8% | Lucky on tiny n |
| BTC 15m | 60s | 23 | 82.6% | 0.000% | 73.9% | Noisy, small n |
| BTC 5m | 30s | 36 | 55.6% | 0.011% | 44.4% | Near coin flip |
| **BTC 5m** | **60s** | **51** | **70.6%** | **0.009%** | **29.4%** | **Genuine edge** |

### Where Edge IS Real

BTC 5m at 60s window, filtered by move size:
- Moves 0.00-0.01%: 68% WR, n=28
- Moves 0.01-0.02%: 69% WR, n=16
- **Moves 0.02-0.05%: 86% WR, n=7** (strongest but tiny sample)

The edge exists when the price has moved enough to be meaningful (>0.01%) but the market hasn't caught up. At avg entry $0.369, break-even is 36.9%, so even 68% WR provides strong edge.

### Why the Earlier Analysis Was Misleading

The "97.6% obvious accuracy at <10s" was measured across ALL observations:
- 91% of ticks: market and obvious AGREE → not tradeable
- 9% of ticks: they DISAGREE → the ambiguous, tiny-move cases
- We can only trade the disagree cases, which are specifically the hardest ones

### Conclusion

Resolution snipe is NOT a standalone strategy yet. It could work as a **supplement** to the main maker strategy:
1. Only trigger when price move > 0.02% (filter coin-flip events)
2. Only on 5m at 60s window (most robust)
3. Need ~100+ more events to confirm the 70% WR

The main strategy (Kou + min_z=0.5 maker) is more reliable.
