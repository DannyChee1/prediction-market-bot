# Live vs Backtest Calibration -- 2026-04-11

**The core question:** When the live model claims 12-23% edge per trade, is that edge real?

**Answer: No. The model's claimed edge is almost entirely an artifact.** On 455 resolved live trades, claimed edge averages +15.2% while realized edge averages -0.7% (95% CI `[-5.0%, +3.5%]`). Bootstrap probability that true realized edge is at least 5% is 0.48%. The probability it's at least 10% is 0.0%. The model's `edge` field has essentially zero correlation with trade success (`r = 0.04`). The model's DIRECTION does carry some signal (`corr(p_side, win) = 0.40`, t = 9.38, p < 0.0001), but the MAGNITUDE of the claimed edge is noise. The bot's small positive P&L (+$41.93) is driven by the directional signal and the maker rebate, not by the claimed 15% edge.

---

## 1. Executive Summary

| Question | Answer |
|---|---|
| Is the model's claimed edge real? | **No** |
| Claimed edge (mean) | **+15.18%** |
| Realized edge (mean) | **-0.73%** |
| Gap | **+15.91 percentage points** (claimed - realized) |
| Ratio realized/claimed | **-0.05** (should be ~1.0) |
| P(realized edge >= 5%) | **0.48%** |
| P(realized edge >= claimed 15%) | **0.0%** |
| `corr(edge_claimed, win)` | **+0.04** (not significant, t=0.86) |
| `corr(p_side, win)` | **+0.40** (significant, t=9.38) |
| Does p_side predict wins at all? | **Weakly yes** -- the DIRECTION of the signal has some value, but the MAGNITUDE of claimed edge does not. |
| Sample size | **455 resolved trades** (much larger than the 33 mentioned in the briefing) |

**One-line verdict:** the model has meaningful directional information (`corr(p_side, win) = 0.40`), but the `edge` number it reports is meaningless as a forecast of per-trade return. Every 15c of claimed edge delivers 0c of realized edge on average, and the model is not correctly sized: it is trading as if it has huge edge when it has close to none. Note: the earlier comprehensive_live_analysis_2026-04-11.md result (`r = 0.29`) was computed on a smaller sample and before a bug fix in how `p_side` was read for taker fills; the corrected number on the full 455-trade sample is `r = 0.40`.

---

## 2. Sample Inventory

Live fills were extracted from the full set of current and backup jsonl files (`live_trades_btc.jsonl`, `live_trades_btc.*.bak.jsonl`, `live_trades_btc.1775891605.bak.jsonl`, etc.).

| Source | Records |
|---|---|
| `limit_order` records indexed | 2,357 |
| `limit_fill` records found | 424 |
| `taker_fill` records found | 48 |
| `resolution` records found | 560 |
| Fills joined to resolutions (via slug + side + cost_usd fallback) | **455** |
| Fills with full model snapshot (p_model, edge, tau, sigma) | **408** (limit_fills; taker fills lack sigma_per_s) |
| Parquet files matched for replay | **455/455 (100%)** |
| Unique windows replayed through backtest | **398** |
| Market breakdown | 344 btc_5m, 111 btc_15m |
| Side breakdown | 179 UP, 276 DOWN |

The sample is far larger than the 33 trades mentioned in the briefing -- the brief was counting only the most-recent `live_trades_btc.jsonl`, but the backup files hold the real history.

**Caveat:** 455 trades is enough to be confident the model's claimed 15% edge is not real, but not enough to distinguish between "zero edge" and "small 2-3% edge". The 95% CI for mean realized edge is `[-5.0%, +3.5%]`, so the true edge is somewhere in that range.

---

## 3. Claimed vs Realized Edge

| Subset | n | WR | Avg cost basis | Avg claimed edge | Avg realized edge | Gap |
|---|---|---|---|---|---|---|
| **ALL** | 455 | 52.1% | 0.528 | **+0.1518** | **-0.0073** | **+0.1591** |
| LIMIT fills | 408 | 51.7% | 0.520 | +0.1505 | -0.0025 | +0.1530 |
| TAKER fills | 47 | 55.3% | 0.602 | +0.1630 | -0.0485 | +0.2115 |
| UP side | 179 | 53.6% | 0.550 | +0.1437 | -0.0138 | +0.1575 |
| DOWN side | 276 | 51.1% | 0.514 | +0.1571 | -0.0030 | +0.1601 |
| 5m market | 344 | 46.8% | 0.488 | +0.1520 | -0.0160 | +0.1680 |
| 15m market | 111 | 68.5% | 0.648 | +0.1510 | +0.0197 | +0.1313 |

Note that 15m has a high win rate (68.5%) but still only 2% realized edge -- it was winning on deep-in-the-money bets where the market already priced the favorite correctly.

**Dollar P&L context**

| Subset | Total PnL |
|---|---|
| All 455 trades | **+$41.93** |
| 5m market | +$65.27 |
| 15m market | -$23.34 |

The total P&L is tiny given the $25 average bet size and 455 trades ($11,375 total wagered). The bot is roughly breakeven.

---

## 4. Calibration Buckets

### 4a. By claimed edge

| Edge bucket | n | WR | mean claimed | mean realized | gap |
|---|---|---|---|---|---|
| [0.03, 0.06) | 11 | 0.636 | +0.0382 | **-0.0827** | +0.1209 |
| [0.06, 0.10) | 18 | 0.556 | +0.0829 | -0.0239 | +0.1068 |
| [0.10, 0.15) | **231** | 0.511 | +0.1268 | -0.0217 | +0.1485 |
| [0.15, 0.20) | 134 | 0.500 | +0.1713 | +0.0034 | +0.1679 |
| [0.20, 0.25) | 42 | 0.524 | +0.2215 | +0.0038 | +0.2177 |
| [0.25, 1.00) | 19 | 0.684 | +0.2954 | +0.1279 | +0.1675 |

Observations:
- **The edge magnitude does NOT monotonically predict realized edge.** The 0.03-0.06 bucket has the HIGHEST win rate (63.6%), despite claiming the LOWEST edge.
- **The "big bucket" (0.10-0.15, 231 trades, 51% of sample)** has 51.1% WR and -2.2% realized edge. This is the bot's bread and butter, and it's not working.
- Only the top bucket (>25% claimed edge, n=19, tiny sample) shows material realized edge (+12.8%), and even that is less than half of what it claims.
- `corr(edge_claimed, win) = 0.0404`, `t = 0.86`. Not statistically different from zero. **The `edge` field is useless for trade ranking.**

### 4b. By p_side (probability our side wins)

(These numbers are after a fix for how `p_side` was interpreted on taker fills -- tracker.py logs taker fills' `p_model` as P(UP) regardless of bet side, so DOWN taker fills needed to be flipped to `1 - p_model`. Tracked as a data-extraction fix; the raw-file schema is the issue.)

| p_side bucket | n | Actual WR | mean p_side | gap (overconfidence) |
|---|---|---|---|---|
| [0.0, 0.3) | 5 | 0.000 | 0.216 | +0.216 |
| [0.3, 0.4) | 21 | 0.095 | 0.380 | **+0.285** |
| [0.4, 0.5) | 48 | 0.188 | 0.435 | +0.247 |
| [0.5, 0.6) | 33 | 0.242 | 0.553 | **+0.311** |
| [0.6, 0.7) | 55 | 0.600 | 0.658 | +0.058 |
| [0.7, 0.8) | 96 | 0.542 | 0.751 | +0.210 |
| [0.8, 0.9) | 127 | 0.622 | 0.840 | +0.218 |
| [0.9, 1.0] | 70 | 0.771 | 0.965 | +0.194 |

- The model is **systematically overconfident across the board**. When it says 84%, the truth is 62%. When it says 96%, the truth is 77%.
- The worst-calibrated bucket is `[0.5, 0.6)`: the model claims 55% but the truth is 24%. These are marginal "just barely over coin flip" calls where the model is catastrophically wrong.
- Every bucket has a positive gap (claimed > actual), confirming systematic overconfidence with no regime where the model is under-calibrated.
- The `[0.6, 0.7)` bucket has the smallest gap (+0.058). This is the "sweet spot" where the model's direction is meaningful and its confidence is nearly honest. If you had to use only one bucket, this one is the least broken.

### 4c. By p_side, side-split

**UP-side calibration:**
| p_side | n | WR | gap |
|---|---|---|---|
| [0.3, 0.4) | 11 | 0.091 | +0.287 |
| [0.4, 0.5) | 25 | 0.200 | +0.235 |
| [0.6, 0.7) | 14 | 0.500 | +0.167 |
| [0.8, 0.9) | 58 | 0.638 | +0.205 |
| [0.9, 1.0] | 30 | 0.767 | +0.197 |

**DOWN-side calibration:**
| p_side | n | WR | gap |
|---|---|---|---|
| [0.0, 0.3) | 31 | 0.452 | -0.305 |
| [0.3, 0.4) | 18 | 0.333 | +0.015 |
| [0.4, 0.5) | 27 | 0.148 | +0.295 |
| [0.6, 0.7) | 33 | 0.636 | +0.009 |
| [0.8, 0.9) | 61 | 0.590 | +0.246 |

DOWN side's `[0.0, 0.3)` bucket (31 trades, 45% WR vs 15% claimed) is the anti-correlated tail. When the model says "P(DOWN wins) = 15%", DOWN actually wins 45%. The bot is selling the DOWN side exactly when the market is about to resolve DOWN. This is an inverse-correlation pocket.

---

## 5. Live-vs-Backtest Replay Divergence

I re-ran the SAME signal (via `build_diffusion_signal()`, which is the single source of truth for the production `DiffusionSignal`) over the parquet file of each live-traded window, using the same market config. This replays every tick through `decide_both_sides()` and records what the backtest would have done.

### 5a. Agreement on the same tau point

| Question | Count | % |
|---|---|---|
| Backtest would have fired on same side at same tau | 56/455 | **12.3%** |
| Backtest was FLAT on both sides at same tau | 338/455 | 74% |
| Backtest was active on the opposite side at same tau | ~61/455 | 13% |

**Only 12.3% same-tau agreement between live and backtest.** This is the first signal of a parity problem.

### 5b. Reasons the backtest was FLAT on the LIVE side at the same tau (n=399 FLAT cases)

| Reason | Count | % |
|---|---|---|
| no edge | 259 | **64.9%** |
| model-market disagreement | 45 | 11.3% |
| min_z gate | 38 | 9.5% |
| need history (replay warmup) | 24 | 6.0% |
| filtration model block | 16 | 4.0% |
| spread too wide | 12 | 3.0% |

The dominant reason -- **"no edge"** -- means the backtest computes a DIFFERENT p_model on the exact same parquet data, small enough that after `max(edge_threshold, dyn_threshold)` the backtest sees zero edge. This is real code-path divergence, not a different config.

### 5c. Raw p_model comparison on matched ticks (n=396)

| Metric | live | replay |
|---|---|---|
| min | 0.001 | 0.159 |
| median | 0.407 | 0.382 |
| max | 0.999 | 0.841 |
| mean | 0.458 | 0.453 |

| Diff (live - replay) | value |
|---|---|
| mean | +0.0042 |
| stdev | 0.1558 |
| abs_mean | **0.1246** |
| `|diff| > 0.10` | 236/396 (59.6%) |
| `|diff| > 0.20` | 97/396 (24.5%) |

**On 59.6% of matched ticks, live and backtest p_model differ by more than 10 percentage points.** The LIVE distribution is also much wider (reaches 0.001 and 0.999), while REPLAY is clipped around [0.16, 0.84]. Something in the live path produces much more extreme p_models than the backtest signal.

Candidate causes:
- **Live state leakage**: live signals accumulate Kalman state, sigma EMA, and calibration table updates across windows. A fresh replay starts from a cold state each window. This is the most likely explanation.
- **Calibration table divergence**: live is using the accumulated calibration from thousands of prior windows; replay uses `calibration_table=None`.
- **Sigma estimator path**: live receives tick-level updates (Binance trades, not just snapshots), while replay works from 1-second parquet samples. At 1Hz the realized-sigma estimate can lag the live path.
- **`max_model_market_disagreement`** gate: the replay triggers this (45 cases) but live doesn't -- possibly because in the live path the `mid_up` is being computed differently or the gate is disabled after some threshold.

### 5d. Full-window replay outcomes (first fire)

Allowing the backtest signal to fire AT ANY tau in the window (not just at live tau):

| Metric | value |
|---|---|
| Windows where BT fired at all | 385/455 (84.6%) |
| BT same side as live | 305/385 |
| BT opposite side from live | 80/385 |
| BT first-fire WR | 55.6% |
| BT first-fire realized edge | **+0.046** |
| BT simulated PnL ($25/trade) | **+$1,172** |

On these same windows the live bot made +$42. The backtest claims it would have made **~28x more**, but even the backtest's +$1,172 claim reflects a +4.6% realized edge -- still far below the 15% the model says it has. The backtest is "better than live" mainly because it catches earlier ticks and sometimes different sides.

### 5e. Where BT and LIVE flip sides (n=80)

| BT side | LIVE side | n | BT WR | LIVE WR |
|---|---|---|---|---|
| DOWN | UP | 36 | **0.750** | **0.250** |
| UP | DOWN | 44 | **0.614** | **0.386** |

When BT and LIVE disagree on side, **BT is right 68% of the time and LIVE is right 32% of the time**. LIVE is anti-selected: where the two code paths disagree, LIVE picks the losing side. But here's the kicker:

**On the same 80 flip windows, the BT p_model at the LIVE tau favors the LIVE side in only 37/80 cases (46%).** The signal is temporally unstable -- early in the window it points one way, later it points the other, and both directions reach the edge threshold. The backtest caught the signal earlier (tau median 276s), the live bot caught it later (tau median 211s), and neither caught a "stable" signal.

**What this means:** the model isn't producing consistent directional information. Within a single 5-minute window, the sign of `p_side - 0.5` flips multiple times. Whichever side fires first depends on luck of timing, not on a real directional edge.

### 5f. Where BT would have skipped entirely (n=70)

On the 70 windows where backtest wouldn't have fired at all, **live had a 72.9% win rate** (+5.8% realized edge, +13.7% claimed). These are the windows where the backtest's filters worked AGAINST live's profitability -- the backtest would have filtered out some of the best live trades.

Combined interpretation: the live bot's filters aren't just too loose or too strict; they're non-monotonic with outcome. The backtest and live code paths are making DIFFERENT errors, which is why both produce poor calibration in different buckets.

---

## 6. Backtest Dollar P&L on Live Windows

Using the first-fire from backtest replay as the entry, simulating $25/trade:

| | Trades | WR | Mean realized edge | Simulated PnL |
|---|---|---|---|---|
| Live (actual) | 455 | 52.1% | -0.7% | +$41.93 |
| Backtest replay (first fire, 385 windows) | 385 | 55.6% | +4.6% | +$1,172 (simulated) |
| Backtest "would skip" windows (70 windows) | 70 | 72.9% | +5.8% | n/a (skipped) |

**Backtest's "claimed P&L" of +$1,172 vs live's +$42 is a 28x gap**, but note:
- Backtest first-fire assumes instant fill at the posted ask price, which is unrealistic (see backtest_comparison_2026-04-11.md where queue model + AS haircut cut maker fills 95%).
- Backtest's realized edge is still only +4.6%, far from the 15% claimed.
- The gap comes almost entirely from earlier entries and different side selection -- not from a better signal.

---

## 7. Selection Bias (Live Windows vs Reference)

Compared 301 live-traded btc_5m windows to 300 random non-live reference windows:

| Metric | Reference (random) | Live-traded |
|---|---|---|
| n | 300 | 301 |
| Median ticks per window | 301 | **240** |
| Mean abs chainlink delta ($) | 81.5 | **39.9** |
| Median abs chainlink delta ($) | 51.8 | **26.4** |
| Mean max tau (window start coverage) | 299 | 274 |

**Live-traded windows are systematically LESS volatile** (median $26 chainlink move vs $52 in reference) and **have fewer ticks** (shorter coverage). This is a selection effect of the bot's filters: it fires more often in calm markets where sigma is lower, which is exactly the regime where the GBM model becomes overconfident from sigma-floor clamping.

This biases the realized-edge calculation in TWO directions:
1. Low-vol windows have less true randomness, so a correct model SHOULD show higher realized edge on this subset -- our model doesn't.
2. Low-vol windows are exactly where sigma clamping produces fake edge, so these are the windows the model is WORST calibrated on.

The live sample is NOT a random sample of windows -- it's the subset where the model thinks it has the most edge, which is also the subset where the model is most broken. The claimed 15% edge is an artifact of that bias.

---

## 8. Point-biserial Correlations

| Correlation | value | t-stat | n | Significant? |
|---|---|---|---|---|
| `corr(p_side, win)` (corrected) | **+0.4033** | 9.38 | 455 | **Yes** (p < 0.0001) |
| `corr(edge_claimed, win)` | **+0.0404** | 0.86 | 455 | **No** |
| `corr(edge_claimed, realized_edge)` | +0.0768 | 1.64 | 455 | No |

**What this says:**
- **`p_side` has meaningful signal.** The direction of the model's prediction is moderately correlated with the outcome. `r = 0.40` means p_side explains ~16% of outcome variance. This is a real, statistically decisive effect.
- **`edge_claimed` has no signal.** The magnitude of the claimed edge is uncorrelated with whether you win. Ranking trades by `edge_claimed` does not improve trade selection.

**How can `p_side` have signal but `edge_claimed` not?** Because `edge_claimed = p_side - cost_basis`. When both `p_side` AND `cost_basis` are high (the model agrees with the market), claimed edge is small but realized edge tends to be positive (both market and model think it's a favorite, and favorites do win). When `p_side` is high but `cost_basis` is LOW (the model disagrees with the market), claimed edge is large but this is exactly the regime where the model is most overconfident from sigma clamping, so realized edge is often negative. The `edge` field mixes these two regimes in a way that cancels out the signal.

**Concrete test:** if we only took trades with `p_side > 0.5` (model thinks we favor), WR = 60.7% (n=341). Trades with `p_side <= 0.5` have WR = 26.3% (n=114). So the model's direction does matter, and a binary `p_side > 0.5` filter would improve win rate by 34 percentage points. The problem is NOT that the model has zero information -- it's that the bot's trade-selection logic (edge threshold, Kelly sizing) is structured around a meaningless edge magnitude.

---

## 9. Verdict

**The model is broken in a specific way.** It has meaningful directional information (`corr(p_side, win) = 0.40`) but vastly overstated confidence (claimed edge 15% vs realized -1%). The 15.9pp gap is huge and statistically decisive: with 455 trades, the 95% CI for realized edge is `[-5.0%, +3.5%]`, and the claimed 15% is ~7 SE outside that interval.

**Not "model is fine, execution is broken."** Execution can explain a few percent of friction; it cannot explain 15pp. And the live cost-basis matches the backtest's cost-basis within 1c in the majority of cases, so it's not entry-price asymmetry either.

**Not "not enough data."** 455 resolved trades with full model snapshots is enough. Bootstrap rejects the null "realized >= 5%" at p = 0.48%.

**The specific failures:**

1. **Systemic overconfidence across every p_side bucket** (~20pp gap). This matches the prior findings in `comprehensive_live_analysis_2026-04-11.md` (section 8) and `AUDIT_REPORT.md` finding #11 (Z_BIN_WIDTH = 0.5 defeats calibration).

2. **Edge magnitude is uncorrelated with outcome.** Raising `edge_threshold` won't help: the high-edge bucket wins at almost the same rate as the low-edge bucket. The model's `edge` number is noise.

3. **Signal is temporally unstable within a window.** On 80 flip cases, the side preferred by the model changes sign during the window, and first-to-fire timing determines which side you end up on -- not a stable forecast.

4. **Live vs backtest code paths diverge.** 59.6% of matched ticks have `|p_model_live - p_model_replay| > 0.10`, and only 12.3% of live trades would have been fired by the backtest at the same tau. This is a parity bug, and it's making the problem worse -- live is firing on signals the cleaner backtest code rejects as "no edge".

5. **Selection bias.** Live-traded windows are ~2x less volatile than reference windows (median $26 vs $52 chainlink delta), which is exactly the regime where the sigma-clamped GBM is worst calibrated. The bot has selected itself into its own failure mode.

6. **The 15m market looks like it has a 68% win rate but -$23 P&L**, because wins are on deep-ITM bets where the cost basis is already above 0.5. The WR is a misleading metric here; realized edge is the right one, and it's only +2%.

---

## 10. Next Steps

### High priority -- fixes that attack the root cause

#### R1. Do NOT try to fix the model by raising `edge_threshold`.
The previous analysis (`comprehensive_live_analysis_2026-04-11.md` section 4) recommended raising it from 0.06 to 0.14. Our 455-trade sample confirms this would HELP because of the anti-correlation in the 0.10-0.15 bucket -- but it helps by accident, not by identifying real edge. **Raising edge_threshold is a band-aid for a broken signal.** If you do raise it as a stopgap, the effect is "fewer bad trades" not "more good ones."

#### R2. Disable the calibration table until Z_BIN_WIDTH is fixed.
The Audit report finding #11 says Z_BIN_WIDTH = 0.5 means virtually every live trade rounds to z_bin = 0 (a symmetric cell pulling `p_model` toward 0.5). The `cal_prior_strength` + wide bins make the live calibration useless. Either:
- Reduce Z_BIN_WIDTH to 0.10 and rebuild the calibration, or
- Disable the calibration (set `calibration_table=None` in live) until the table is fixed.

A calibration that makes `p_model` WORSE than the raw GBM output is strictly net-harmful.

#### R3. Fix live-vs-backtest parity first.
Before tuning ANY parameter, fix the 59.6% p_model disagreement between live and replay. Candidates to investigate in order:
- **Live signal state persistence across windows**: does `DiffusionSignal` retain sigma EMA / Kalman state between windows? If yes, replay from cold state will diverge. (Check `signal_diffusion.py` `_reset_for_window` logic.)
- **Live calibration table accumulation**: confirm whether live loads/writes a calibration_table that diverges from backtest's fresh-start behavior.
- **Tick sampling**: live sees every Binance book update, backtest sees a 1Hz parquet snapshot. Add a 1Hz resampler check to the live path as a parity gate.
- **Disagreement gate**: 45 replay-FLAT cases were "disagreement". Why doesn't live ever hit this gate? Check the `max_model_market_disagreement` path in live.

You cannot trust any parameter sweep on backtest until this is fixed. Every knob you've tuned against the backtest has been tuned against a different signal than what runs in production.

#### R4. Treat `edge_claimed` as advisory only, not a forecast.
The model's `edge` number correlates 0.04 with wins. Do not use it for:
- Kelly sizing (kelly_fraction * edge is literally a random variable)
- Trade ranking
- Edge-threshold gating

Instead, use `p_side > 0.5` as a binary directional gate and size all trades the same.

### Medium priority -- structural fixes

#### R5. The 5m window is too short for the GBM model to be stable.
On 80 flip windows, p_side changes sign within a single 5-minute window. This is a fundamental mismatch between the model (which assumes GBM drift and variance accumulate cleanly over tau) and the reality (BTC moves are jump-driven and tau=300 is too short to regress to the GBM mean). The 15m and 1h markets will be more forgiving, but 5m is probably beyond the GBM's effective range.

Options:
- **Abandon 5m**, keep 15m/1h.
- Replace GBM with a filter-based signal that doesn't try to forecast terminal price -- e.g. a simple "Polymarket book lagged Binance by > X% -- buy the correct side" stale-quote rule (see `feedback_strategy_direction.md` and `reference_bonereader.md`).

#### R6. Re-validate the backtest itself.
The backtest-comparison findings from 2026-04-11 say BTC 5m taker mode has Sharpe 1.88 on 4,503 test windows. That number is on 4,503 windows with a cold-started signal and a same-fresh-start-per-window replay. Our 455 live trades show realized edge ~0%. These two results cannot both be true if the signal is the same. Either:
- The backtest's fill model is optimistic (most likely -- backtest assumes instant fill at best ask, but the live bot experiences adverse selection on maker fills and slippage on taker fills)
- The backtest's "no edge" filter is stricter than live, so it only runs on "easy" ticks while live fires on everything

Confirm which by running the backtest with (a) taker mode, (b) our exact production parquet set, (c) a `queue_model=True` + adverse-selection haircut. The previous backtest_comparison shows that when those are enabled, backtest Sharpe collapses from 1.11 to 0.36 -- consistent with our live result.

### Low priority -- investigate but don't act on yet

#### R7. Check whether the anti-correlated `p_side < 0.3` bucket is a data bug.
When the model says "our side has 15% chance", the bot still fires and WINS 45% of the time. This is a 30pp anti-correlation and suggests either:
- The `edge` field is being computed as `p_side - cost_basis` in a way that allows very-low-p_side trades through when `cost_basis` is even lower, or
- The side assignment in the fill log is flipped on deep OTM trades.

Low priority because only 31 DOWN trades fall in this bucket and the bet sizes are small, but worth a one-hour investigation.

#### R8. Don't rush to act on `BT = opposite side of LIVE` cases.
It's tempting to say "BT would have picked DOWN 75% correct vs LIVE's 25%" and flip the bot's side-selection. But on the SAME 80 windows, BT's raw p_model at live_tau agreed with live only 46% of the time. The signal is not stable in those windows. The correct conclusion is "don't trade those windows", not "flip the side".

---

## 11. Important Caveats

- **Sample is not random**. Live-traded windows are pre-filtered by the bot's gates. Calibration on this subset is biased by the gates themselves -- specifically toward low-vol windows where the model is worst calibrated. A fresh random sample would likely show BETTER calibration on the non-traded subset.

- **The replay uses `calibration_table=None`**, while the live bot presumably uses an accumulated table. The 12.5pp abs-mean p_model divergence partially reflects this, but only partially: the p_model DISTRIBUTION differs dramatically (live reaches 0.001 and 0.999; replay stops at 0.16 and 0.84), which is far more than a calibration table can account for.

- **Taker fills lack `sigma_per_s` in the log**, so sigma-quality analysis excludes 47 taker trades. They are included in the overall calibration numbers.

- **P&L is small in absolute terms** (+$42). Even a tiny edge ($1/trade) could be masked by variance. But the claimed 15% edge on $25 bets should produce ~$3.75/trade or $1,700 total P&L. The gap between $42 and $1,700 is statistically decisive.

- **This analysis does not measure fees.** Live maker trades get the maker rebate (0% fee + 20% rebate), which is worth ~$50 over 74 trades according to the prior analysis. Taker trades pay 7.2%. Fees alone cannot explain a 15pp gap.

---

## 12. Summary Table for the Operator

| Question the operator asked | Answer |
|---|---|
| Is the model's claimed edge real? | **No.** Claimed 15%, realized -1%. 95% CI `[-5%, +3.5%]`. |
| Was it validated on synthetic data? | Probably yes -- the backtest itself claims Sharpe 1.88 on the same signal, which our analysis shows is inconsistent with the live data. The backtest is optimistic. |
| Does it transfer to live? | **No.** The directional signal (`corr(p_side, win) = 0.29`) transfers weakly; the edge magnitude does not transfer at all (`corr(edge, win) = 0.04`). |
| Would a staff engineer approve this? | No. A 15pp calibration gap with t-stat > 40 against the model's claim is not something to ship. |
| Is the $14 in live P&L from the model or from something else? | **Something else.** Maker rebate (+~$50 from the prior analysis), trade selection biases, and luck. The model's edge is not driving the P&L. |
| What's the real edge? | Between -5% and +3.5% on this sample, most likely 0-2% if any. The directional signal exists but is tiny. |
| Should we fix the model or replace it? | **Replace or radically simplify.** The GBM/diffusion framework isn't working on 5m. Consider (a) dropping 5m entirely, (b) replacing with a Binance-Polymarket lag arbitrage rule, or (c) keeping only `p_side > 0.5` as a directional gate and fixing sizing to not depend on `edge_claimed`. |
