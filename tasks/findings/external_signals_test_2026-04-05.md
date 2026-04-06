# External Research Signals — Empirical Test on Our Data

Tested on 609 BTC 15m windows from our parquet data.

## Results Summary

| Signal | Research Claim | Our Data Shows | Actionable? |
|--------|---------------|----------------|-------------|
| Session momentum/reversion | Asia=revert, US=momentum | **OPPOSITE**: Asia 76% continuation, US 68% | No (all high) |
| Consecutive direction | Funding buildup → reversion | **YES**: after 3 UPs, only 42% next UP | **YES** |
| US equity hours | Larger moves, directional | US: 57% up, 1.25x moves | **YES** (regime) |
| Order book imbalance | Bid depth → bullish | **OPPOSITE**: high bid depth = 37% up | **Weak** |
| Spread width | Tight = better signal | 58.5% up at tight spread (n=53) | No (small n) |

---

## 1. Session Directional Bias — OPPOSITE of Research

Research: "Asia = mean-reverting, US = momentum"

Our data (continuation rate at 50% elapsed):
- Asia: **76.2%** continuation (n=206)
- Europe: **76.4%** continuation (n=123)
- US: **68.4%** continuation (n=237)

All sessions show strong momentum (>68%). US actually has the LOWEST continuation — opposite of the "US = momentum" claim. The 95% CIs overlap, so the difference may not be significant.

**Verdict: Not actionable.** Momentum is strong everywhere. No session-specific bias worth trading.

---

## 2. Consecutive Direction (Funding Rate Proxy) — STRONG SIGNAL

After N consecutive UP windows:
| Streak | Next UP % | n |
|--------|-----------|---|
| 1 | 47.8% | 293 |
| 2 | 42.1% | 140 |
| 3 | 42.4% | 59 |
| 4 | 44.0% | 25 |
| 5 | **27.3%** | 11 |

After N consecutive DOWN windows:
| Streak | Next UP % | n |
|--------|-----------|---|
| 1 | 48.4% | 316 |
| 2 | 47.2% | 163 |
| 3 | 51.2% | 86 |
| 4 | 47.6% | 42 |
| 5 | 40.9% | 22 |

**Key finding:** After 2+ consecutive UPs, mean-reversion kicks in (42% vs baseline 48%). After 5 consecutive UPs, only 27% chance of another (n=11, small sample).

DOWN streaks show weaker reversion — approximately 47-48% regardless of streak length.

**This is consistent with funding rate theory:** prolonged uptrends → overleveraged longs → mean-reversion. The asymmetry (UP streaks revert more than DOWN) matches the "sell the rally" dynamic in crypto.

**Verdict: Actionable as regime overlay.** After 2+ consecutive UP windows, bias toward DOWN or reduce confidence in UP signals. Small sample warning on 3+ streaks.

---

## 3. US Equity Hours — Confirmed Larger Moves

| Session | UP Rate | Avg Move | n |
|---------|---------|----------|---|
| US (14-21 UTC) | **57.1%** | 0.181% | 168 |
| Off-hours | 44.6% | 0.145% | 442 |

US hours have **1.25x larger moves** and a **57% bullish bias**. This aligns with ETF-driven institutional buying during NYSE hours.

**Verdict: Actionable.** US hours = more opportunities with larger moves. Could scale position size up during US hours and down during off-hours.

---

## 4. Order Book Imbalance — Counter-Intuitive

| OBI Range | UP Rate | n |
|-----------|---------|---|
| < -0.3 (more asks) | **62.0%** | 71 |
| -0.1 to 0.1 (neutral) | 48.4% | 219 |
| > 0.3 (more bids) | **37.3%** | 59 |

Correlation: **-0.092** (weak negative)

**Counter-intuitive:** More bid depth → MORE likely to go DOWN. More ask depth → MORE likely to go UP.

Explanation: informed traders hit the opposite side. Aggressive buyers deplete ask depth (low ask = bullish signal). Aggressive sellers deplete bid depth (low bid = bearish). The OBI measures RESTING orders, not aggressor flow.

**Verdict: Weak but interesting.** The -0.092 correlation is too weak for a standalone signal. Could be used as a tiebreaker or confirmation. NOTE: our current OBI implementation in the signal uses OBI as a bullish indicator (positive OBI = buy UP) — this data suggests it should be FLIPPED.

---

## 5. Spread Width — Insufficient Data

Tight spread (< 1¢): 58.5% UP (n=53)
Normal spread (1-2¢): 47.2% UP (n=553)

Too few tight-spread windows to draw conclusions. Skip.

---

## Rigorous Statistical Testing (p-values)

### Consecutive Direction: NOT SIGNIFICANT

| Condition | UP Rate | n | 95% CI | p-value | Significant? |
|-----------|---------|---|--------|---------|--------------|
| After 1 UP | 47.8% | 293 | [42.1%, 53.5%] | 0.945 | No |
| After 2 UP | 42.1% | 140 | [33.9%, 50.3%] | 0.162 | No |
| After 3 UP | 42.4% | 59 | [29.8%, 55.0%] | 0.389 | No |
| After 4 UP | 44.0% | 25 | [24.5%, 63.5%] | 0.689 | No |
| After 5 UP | 27.3% | 11 | [1.0%, 53.6%] | 0.169 | No |

Every CI includes the 48% baseline. The "42% after 2 UPs" is noise. Do NOT implement.

### OBI: One Bucket Significant, Borderline

| OBI Range | UP Rate | n | 95% CI | p-value | Significant? |
|-----------|---------|---|--------|---------|--------------|
| < -0.3 (heavy asks) | **62.0%** | 71 | [50.7%, 73.3%] | **0.018** | **Yes** |
| -0.1 to 0.1 (neutral) | 48.4% | 219 | [41.8%, 55.0%] | 0.906 | No |
| > 0.3 (heavy bids) | 37.3% | 59 | [25.0%, 49.6%] | 0.100 | Borderline |

OBI < -0.3 is statistically significant (p=0.018). Heavy ask-side depth on the UP token = 62% UP win rate.

**Why counter-intuitive:** OBI measures resting Polymarket orders, not aggressor flow. Heavy ask depth = market makers comfortably providing UP liquidity (not afraid of upside). Thin bid side = aggressive buyers already took the cheap bids. But n=71 with p=0.018 is borderline — could flip with more data.

Overall OBI-outcome correlation: -0.092 (very weak). Not reliable enough to implement.

## Recommended Actions (revised)

1. **US hours have 1.25x moves and 57% up bias** — already captured by hour-of-day vol prior (implemented). The directional bias is interesting but needs more data.

2. **All other signals: WAIT.** Consecutive direction, OBI, spread width — none are statistically significant at the 95% level (except one OBI bucket at p=0.018, borderline). Keep collecting data and revisit in 1-2 weeks with larger samples.

3. **Session momentum is universal** — All sessions show >68% continuation at midpoint. No session-specific rules needed.
