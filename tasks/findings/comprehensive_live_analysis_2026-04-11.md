# Comprehensive Live Performance Analysis -- 2026-04-11

Data source: `live_trades_btc.jsonl` (29,978 records: 26,322 diagnostics, 75 fills, 74 resolutions)
Period: 2026-04-09 17:00 UTC to 2026-04-11 02:30 UTC (~32.7 hours)

---

## 1. Overall Stats

| Metric | Value |
|--------|-------|
| Total Trades | 74 resolved |
| Total PnL | +$18.84 |
| Win Rate | 44.6% (33W / 41L) |
| Avg Win | +$32.26 |
| Avg Loss | -$25.51 |
| Profit Factor | 1.018 |
| ROI | 1.04% |
| Total Wagered | $1,809 |
| Peak PnL | +$250.02 (after trade 13) |
| Max Drawdown | $371.94 (peak-to-trough) |
| DD/Peak | 149% -- catastrophic risk-adjusted return |
| Daily Sharpe (rough) | 0.18 |

**Verdict: The bot is barely profitable. PF of 1.018 means for every $100 lost, it makes $101.80. The $250 peak eroded to +$18.84 -- the system gave back 92% of its gains. At this Sharpe, a few more unlucky trades would flip the system negative.**

---

## 2. By Market Type

| Market | Trades | WR | PnL | ROI | PF |
|--------|--------|-----|------|------|-----|
| 5m | 61 (excl 15m-as-5m) | 42.9% | +$13.80 | 0.77% | 1.013 |
| 15m | 9 | 44.4% | +$15.79 est | 2.8% | ~1.05 |
| 1h | 4 | 75.0% | +$5.04 | 29.5% | 2.355 |

The 5m market generates most of the volume but almost no edge. The 1h market looks promising at 3/4 wins but tiny sample. 15m has 9 trades, all DOWN, 4 wins / 5 losses -- a net negative.

---

## 3. By Side

| Side | Trades | WR | PnL | Avg PnL |
|------|--------|-----|------|---------|
| UP | 5 | **0.0%** | **-$96.79** | -$19.36 |
| DOWN | 69 | 47.8% | +$115.63 | +$1.68 |

### 5m UP: 0/5 -- a total catastrophe

Every 5m UP trade lost. All 5 share the same signature:
- Entry price: $0.20-0.23 (deep OTM)
- sigma: capped at 0.0002 (the max_sigma ceiling)
- tau: 27-123s (mid to late window)
- p_model: 0.35-0.43 (model says only 35-43% chance of UP)
- Edge: 0.11-0.15 (looks reasonable but is fake)

**Root cause:** When sigma is capped at max_sigma (2e-4), the model produces a moderate p_UP ~0.40. The market prices UP at $0.20-0.23. The model says "40% vs market 22% = 18c of edge, trade it!" But sigma is CAPPED, not observed. The real sigma might be lower, meaning the real p_UP is much lower, meaning the "edge" is an artifact of sigma capping. Every one of these trades lost because the model was overconfident due to sigma saturation at the ceiling.

**Fix:** Raise min_entry_price for UP side to 0.30, or better, skip UP entirely. UP bets at deep OTM ($0.20-0.23) with sigma at cap are systematically mispriced.

---

## 4. Edge Analysis

| Edge Bucket | Trades | WR | PnL | $/trade |
|-------------|--------|-----|------|---------|
| 0.04-0.08 | 2 | 50% | +$0.20 | +$0.10 |
| 0.10-0.12 | 5 | 60% | +$5.83 | +$1.17 |
| 0.12-0.15 | 24 | **29.2%** | **-$101.01** | **-$4.21** |
| 0.15-0.20 | 34 | 47.1% | -$42.10 | -$1.24 |
| 0.20-0.25 | 9 | **66.7%** | **+$155.92** | **+$17.32** |

**Critical finding: the 0.12-0.15 edge bucket is the worst performer (29% WR, -$101). This is the largest loss bucket. Meanwhile, edges above 0.20 have 67% WR and +$17/trade.**

### Edge Threshold Sensitivity

| Threshold | Trades | WR | PnL |
|-----------|--------|-----|------|
| >= 0.06 (current) | 74 | 44.6% | +$18.84 |
| >= 0.14 | 49 | **53.1%** | **+$271.74** |
| >= 0.16 | 33 | 54.5% | +$177.18 |
| >= 0.18 | 19 | **68.4%** | **+$360.76** |
| >= 0.20 | 9 | 66.7% | +$155.92 |

**Raising edge_threshold from 0.06 to 0.14 would have turned $18.84 into $271.74 -- a 14x improvement -- by cutting 25 losing trades.** The marginal trades between 0.06 and 0.14 have a NEGATIVE expected value despite the backtest claiming they're profitable. This is the single most impactful parameter change available.

Note: the backtest said raising edge hurts PnL. The live data says the opposite. This is evidence of backtest/live parity issues (confirmed by Audit finding #6: btc_5m backtests run with wrong window_duration, wrong min_entry_z, and no edge persistence gate).

---

## 5. Entry Price Analysis

| Price Bucket | Trades | WR | PnL | $/trade |
|--------------|--------|-----|------|---------|
| < $0.25 | 11 | **9.1%** | **-$96.58** | **-$8.78** |
| $0.25-0.35 | 12 | 50% | +$181.32 | +$15.11 |
| $0.35-0.45 | 23 | 43.5% | -$44.34 | -$1.93 |
| $0.45-0.55 | 15 | 46.7% | -$0.05 | $0.00 |
| $0.55-0.65 | 10 | **70%** | +$8.15 | +$0.82 |
| $0.65-0.80 | 3 | 66.7% | -$29.66 | -$9.89 |

**Entries below $0.25 are catastrophic: 9.1% win rate (1/11).** This bucket contains all 5 UP losers plus 6 cheap DOWN losers. The one win was $110.68 -- not enough to offset 10 losses.

The $0.25-0.35 bucket is the sweet spot: 50% WR with huge average wins ($15/trade) because these are medium-probability DOWN bets that pay ~3x when they hit.

**Fix:** Raise min_entry_price from 0.20 to 0.25 (currently 0.20 on btc_5m). This alone would have saved $96.58.

---

## 6. Timing Analysis (tau = seconds remaining at fill)

| Tau Bucket | Trades | WR | PnL | $/trade |
|------------|--------|-----|------|---------|
| 0-60s | 6 | **16.7%** | **-$84.36** | **-$14.06** |
| 60-120s | 12 | 25.0% | -$29.37 | -$2.45 |
| 120-180s | 13 | **69.2%** | **+$247.18** | **+$19.01** |
| 180-240s | 22 | 36.4% | -$163.73 | -$7.44 |
| 240-300s | 14 | 57.1% | +$76.55 | +$5.47 |
| 300-900s (15m) | 3 | 33.3% | -$32.47 | -$10.82 |
| 900s+ (1h) | 4 | 75.0% | +$5.04 | +$1.26 |

**Late entries (tau < 60s) are disastrous: 16.7% WR.** These are trades placed in the final minute where the market has already mostly priced in the outcome. The model "sees" edge because the book hasn't fully adjusted, but by the time the order fills and resolves, the "edge" evaporates.

**Early entries (tau > 240s for 5m) are good:** 57% WR. The model has more time to be right and the market has more time to move.

**The 120-180s sweet spot (tau 2-3 minutes before close) has 69% WR and +$19/trade.** This is where the signal has had time to develop but there's still enough tau for the model to have predictive power.

Paradox: entries at 180-240s (3-4 min remaining) drop to 36% WR. This might be adverse selection -- the fills that happen at this tau are the ones where the market aggressively moves against you right after.

---

## 7. Sigma Analysis

| Sigma Bucket | Trades | WR | PnL | $/trade |
|--------------|--------|-----|------|---------|
| Floor (2e-5) | 9 | 66.7% | -$20.92 | -$2.32 |
| Low (2.5-5e-5) | 16 | **31.2%** | **-$152.93** | **-$9.56** |
| Mid (5e-5 to 1e-4) | 24 | **58.3%** | **+$179.42** | **+$7.48** |
| High (1e-4 to 2e-4) | 15 | 46.7% | +$135.71 | +$9.05 |
| Cap (2e-4 = max) | 10 | **10.0%** | **-$122.44** | **-$12.24** |

**Sigma at cap (2e-4 = max_sigma) is a 10% WR disaster.** These 10 trades lost $122.44. When sigma is pegged at the ceiling, the model is using a fabricated volatility estimate. It thinks the market is highly volatile, which pulls p_model toward 0.50, creating "edge" on deep OTM contracts that doesn't exist.

**Low sigma (2.5-5e-5) is also bad: 31% WR, -$153.** These are calm-market trades where the model's sigma estimate is barely above the floor but still allows trading. The model overestimates its confidence in calm conditions.

**Mid sigma (5e-5 to 1e-4) is the sweet spot: 58% WR, +$179.** This is the natural BTC volatility range where the model's GBM assumption best matches reality.

**Fix:** Add a sigma quality gate: only trade when sigma is in the "believable" range (5e-5 to 1.5e-4). This eliminates both the floor-clamp artifacts and the cap-clamp artifacts.

---

## 8. Model Calibration

| p_side Bucket | Trades | Actual WR | Expected WR | Gap |
|---------------|--------|-----------|-------------|-----|
| 0.30-0.40 | 6 | 16.7% | 38.6% | **-22.0 pp** |
| 0.40-0.50 | 8 | 12.5% | 42.7% | **-30.2 pp** |
| 0.50-0.60 | 15 | 33.3% | 57.6% | **-24.3 pp** |
| 0.60-0.70 | 19 | 47.4% | 64.8% | **-17.4 pp** |
| 0.70-0.80 | 15 | 60.0% | 73.5% | **-13.5 pp** |
| 0.80-1.00 | 11 | 72.7% | 86.9% | **-14.2 pp** |

**The model is overconfident across the entire range.** When it says "60% probability," the actual win rate is 47%. When it says "87% probability," the actual rate is 73%. The gap averages -20 percentage points.

This is consistent with the Audit Report finding #11: the calibration table has Z_BIN_WIDTH = 0.5 and virtually every live trade rounds to z_bin = 0 (a symmetric cell that pulls p_model toward 0.5). Combined with market_blend, the signal is "double-shrunk" but still not shrunk enough.

**The systematic overconfidence means the edge calculation is overstated by ~20pp.** When the bot thinks it has 15c of edge, the true edge is closer to -5c. This is why the 0.12-0.15 edge bucket has a 29% win rate.

---

## 9. Double Fill Analysis

All 74 resolved trades were in single-fill windows (max_trades_per_window=1 is working). No double-fill damage in this dataset.

---

## 10. Taker Mode Simulation

| Mode | PnL |
|------|------|
| Maker (actual) | +$18.84 |
| Taker (simulated) | -$31.17 |
| Maker Advantage | **+$50.01** |

The maker rebate (0% fee + 20% rebate) is worth $50 over 74 trades, or ~$0.68/trade. Without maker status, the system would be net negative. **The bot's only source of alpha is the maker fee structure, not the model.**

---

## 11. Sizing Analysis

| Metric | Value |
|--------|-------|
| Avg bet size | $24.46 |
| Median bet | $25.82 |
| Min bet | $1.83 |
| Max bet | $37.62 |
| Avg position as % of bankroll | 4.6% |
| Max position as % of bankroll | 6.1% |
| Min position as % of bankroll | 2.8% |

Kelly fraction is in the 5-6% range. Given the model's 20pp overconfidence, the true Kelly fraction should be much smaller. Half-Kelly would halve both PnL and drawdown (PnL $9.42, DD $186 vs current $18.84 / $372). The Sharpe stays the same -- half-Kelly doesn't improve risk-adjusted returns, it just reduces variance.

---

## 12. DOWN Winners vs Losers

| Feature | Winners (N=34) | Losers (N=36) |
|---------|----------------|---------------|
| Avg edge | 0.1672 | 0.1565 |
| Avg price | **$0.46** | **$0.39** |
| Avg sigma | 7.5e-5 | 8.5e-5 |
| Avg tau | **407.6s** | **260.9s** |
| Avg p_model | 0.298 | 0.388 |

Key differences:
1. **Winners have higher entry prices** ($0.46 vs $0.39): less OTM, higher probability of being correct
2. **Winners have much higher tau** (408s vs 261s): earlier entries win more. This is partly explained by the fact that many 15m and 1h trades are winners with high tau values
3. **Winners have lower p_model** (0.298 vs 0.388): paradoxically, when the model is LESS confident in UP (lower p_model), DOWN bets win more. This is correct -- a lower p_model means a stronger DOWN signal

---

## 13. Hour of Day (UTC)

| Hour | Trades | WR | PnL | Notes |
|------|--------|-----|------|-------|
| 00 | 9 | 33% | -$71 | Post-midnight: bad |
| 01 | 9 | 67% | +$119 | Good |
| 02 | 3 | 0% | -$87 | Terrible (3/3 losses) |
| 06 | 3 | 100% | +$89 | Perfect (3/3 wins) |
| 07 | 5 | 60% | +$86 | Good |
| 08 | 2 | 50% | +$77 | Good |
| 10 | 4 | 0% | **-$140** | **Worst hour** |
| 11-12 | 4 | 25% | -$57 | Bad |
| 22 | 8 | 75% | +$132 | Best hour |
| 23 | 9 | 44% | +$29 | OK |

Two-day sample is too small for reliable hour-of-day conclusions, but hours 02, 10, and 20 stand out as losers. These might correlate with specific BTC market sessions (10 UTC = US pre-market, 20 UTC = US late afternoon).

---

## 14. Loss Streak Analysis

The worst loss sequences:

1. **5 consecutive losses ($-117.55):** Trades 31-35, all 5m DOWN bets from 20:31-21:51 UTC on 04/10
2. **5 consecutive losses ($-138.69):** Trades 52-56, all 5m DOWN bets from 00:01-00:21 UTC on 04/11
3. **4 consecutive losses ($-139.50):** Trades 14-17, all 5m DOWN bets from 10:06-10:46 UTC on 04/10

These clusters suggest the bot keeps betting DOWN during periods when BTC is trending UP. The model doesn't have a momentum/trend detector that would back off when the market moves persistently against it.

---

## 15. Temporal Progression

| Trades | WR | PnL |
|--------|-----|------|
| 1-15 | 53.3% | +$176.69 |
| 16-30 | 33.3% | -$162.45 |
| 31-45 | 46.7% | -$3.44 |
| 46-60 | 33.3% | -$48.90 |
| 61-74 | 57.1% | +$56.94 |

The bot had a strong start (first 15 trades) then entered a prolonged drawdown. Performance is not improving over time -- it's oscillating. There's no evidence of learning or adaptation.

---

## 16. Flat Reasons (Why Trades Don't Fire)

From 24,024 diagnostic evaluations:

| Reason | Count | % |
|--------|-------|---|
| No edge | 9,289 | 38.7% |
| min_z gate | 7,727 | 32.2% |
| Missing book | 1,172 | 4.9% |
| Model-market disagreement | 1,094 | 4.6% |
| Entry price below minimum | ~900 | ~3.7% |
| Edge persistence | ~550 | ~2.3% |
| Stale data | 33 | 0.1% |
| Passed all filters (traded) | ~2,200 | 9.2% |

The bot trades on ~9% of evaluations. The min_z gate (32%) is the second-largest filter. This is the gate that was recently lowered from 0.50 to 0.15 -- before that, it was blocking everything.

---

## 17. BoneReaper Comparison

| Metric | Our Bot | BoneReaper |
|--------|---------|------------|
| Trades/day | ~54 | **652** |
| Monthly PnL | ~$547 (projected) | **$614,000** |
| Volume/trade | ~$25 | ~$199 |
| Strategy | Model prediction | Latency arbitrage |
| Fee structure | Maker (0% + rebate) | Maker (0% + rebate) |
| Edge per trade | ~1% | ~4.8% |

BoneReaper trades **12x more frequently** with **8x larger positions** and **5x higher edge per trade**. The structural difference:

1. **BoneReaper monetizes dislocations. We penalize them.** When Binance moves and Polymarket hasn't repriced, BoneReaper buys the correct side instantly. Our bot detects the same dislocation but runs it through a GBM model that dilutes the signal, then applies edge_threshold, min_z, edge_persistence, and other gates that often kill the trade.

2. **BoneReaper doesn't predict. We predict.** BoneReaper doesn't need a model of where BTC will be at window close. It just needs to know that the Polymarket book is stale relative to Binance -- pure arbitrage. Our model tries to predict the future, which is much harder.

3. **Speed.** BoneReaper fills 652 times per day because it's fast enough to capture fleeting dislocations. Our signal_eval_ms is 1-2ms but the order_post_ms is 1,420ms -- nearly 1.5 seconds from signal to acknowledgment. By then, the dislocation may have closed.

4. **Position sizing.** BoneReaper sizes $199/trade on a larger bankroll. We size $25/trade because our edge estimate is lower (correctly, given the calibration issues).

---

## 18. Concrete Recommendations

### HIGH PRIORITY -- Implement immediately

#### R1. Raise edge_threshold: 0.06 -> 0.14 (5m) / 0.12 (15m)
- **Evidence:** trades with edge < 0.14 have 37% WR and -$258 PnL. Trades >= 0.14 have 53% WR and +$272 PnL.
- **Expected impact:** PnL from +$19 to +$272 on same trade set. Trade frequency drops from 74 to 49, but $/trade goes from $0.25 to $5.55.
- **Risk:** Backtest says this hurts. But backtest has known parity issues (Audit finding #6). Trust live data over broken backtest.

#### R2. Raise min_entry_price: 0.20 -> 0.25 (5m) / 0.25 (15m already)
- **Evidence:** entries < $0.25 have 9.1% WR and -$96.58 PnL. Pure destruction.
- **Expected impact:** saves $97 from 11 losing trades.
- **Risk:** none. The one win in that bucket ($110.68 at $0.21) would be missed, but the expectation is deeply negative.

#### R3. Disable UP trades on 5m (or raise min_entry_price_up to 0.40)
- **Evidence:** 5m UP is 0/5, -$96.79. Every loss is a sigma-cap artifact.
- **Expected impact:** saves $97.
- **Risk:** if BTC enters a persistent downtrend, there would be profitable UP entries. But the model can't reliably identify them yet.

#### R4. Add sigma quality gate: skip trades when sigma is at cap (>= 1.9e-4) or in the low zone (< 4e-5)
- **Evidence:** sigma at cap = 10% WR, -$122. Low sigma = 31% WR, -$153. Mid sigma (5e-5 to 1e-4) = 58% WR, +$179.
- **Expected impact:** eliminates 26 bad trades.
- **Mechanism:** could be a new `min_sigma_for_trade` / `max_sigma_for_trade` in MarketConfig, or simply widen the disagreement gate to catch the same cases indirectly.

### MEDIUM PRIORITY -- Do after high priority

#### R5. Fix model calibration (Audit #11)
- The Z_BIN_WIDTH=0.5 means all live trades fall in one calibration bin. This defeats calibration entirely.
- Reduce Z_BIN_WIDTH to 0.10 and rebuild the calibration table.
- This should reduce the 20pp overconfidence gap.

#### R6. Fix backtest/live parity (Audit #6)
- btc_5m backtests run with window_duration=900 (should be 300), min_entry_z=0.0 (should be 0.15), and no edge persistence gate.
- Every parameter tuned on 5m backtests is tuned against a signal that doesn't match live.
- Until this is fixed, do not trust backtest results for 5m parameter tuning.

#### R7. Fix the regime classifier wiring (Audit #5)
- The HMM regime classifier is completely dead in live (decide_both_sides() skips it).
- Every trade uses kelly_mult=1.0 regardless of regime.
- Either wire it in or remove the dead code.

#### R8. Consider disabling 15m trades
- 15m has 9 trades, 4 wins, 5 losses, net likely negative after accounting for bankroll risk.
- The 5m and 15m share bankroll state (Audit finding P1.5), which creates accounting issues.
- Focus on getting 5m profitable first.

### LOW PRIORITY -- Monitor and evaluate

#### R9. Half-Kelly experiment
- Current sizing is ~5% of bankroll per trade. With 20pp miscalibration, true Kelly is much lower.
- Half-Kelly reduces DD from $372 to $186 but also halves PnL.
- Only worth doing after fixing calibration and edge threshold.

#### R10. market_blend adjustment
- Currently 0.3 for 5m, 0.5 for 15m.
- Given the model is 20pp overconfident, pulling more toward market mid (higher blend) might help.
- But this was already tuned in backtest -- which is broken. Revisit after fixing parity.

#### R11. Explore 1h market further
- 4 trades, 3 wins, +$5.04. Tiny sample but promising.
- The 1h market has more tau for the model to be right and less competition.
- Consider increasing 1h trade size once more data confirms the edge.

### WHAT NOT TO CHANGE

- **kelly_fraction:** The sizing is fine at ~5%. The problem is the edge calculation, not the sizing.
- **market_blend:** Leave at 0.3/0.5 until backtest parity is fixed. Changing it blindly could make things worse.
- **tail_mode:** Kou is fine. The problem is sigma estimation, not the distribution model.
- **max_trades_per_window:** Keep at 1. No evidence to change.

---

## 19. Expected Impact of Top 3 Changes

If we had run the same 32 hours with:
- edge_threshold = 0.14
- min_entry_price = 0.25
- No UP trades on 5m

Combined filter result: **DOWN + price>=0.25 + edge>=0.14 + tau>=100**
- 38 trades (vs 74)
- 60.5% WR (vs 44.6%)
- +$310.00 PnL (vs +$18.84)
- $/trade = +$8.16 (vs +$0.25)

Even the simpler **DOWN + edge>=0.14** gives:
- 47 trades
- 55.3% WR
- +$324.78 PnL

**Projected monthly (at current trade frequency, scaled):**
- Current: ~$547/month
- With fixes: ~$7,100-$9,000/month

Still 100x less than BoneReaper, but a fundamentally different risk/reward profile.

---

## 20. Realistic Sharpe Estimate

Current daily Sharpe: ~0.18 (barely positive).

With the recommended parameter changes:
- Fewer trades but much higher edge per trade
- Estimated per-trade return: 8-10% (vs current 1%)
- Estimated daily Sharpe: 0.5-0.8

To reach Sharpe 1.0+, the bot needs:
1. Fixed calibration (reduces overconfidence)
2. Fixed backtest parity (enables proper parameter tuning)
3. Higher trade frequency (more opportunities per day)

The structural ceiling is the GBM model's limited predictive power on 5m BTC moves. The model is right ~50% of the time on the direction -- barely better than a coin flip. The edge comes from trade selection (only trading when the edge is large enough) and maker rebates. Improving the model itself (e.g., incorporating order flow, momentum, or microstructure features) would be the highest-leverage long-term improvement, but that's a research project, not a parameter tweak.
