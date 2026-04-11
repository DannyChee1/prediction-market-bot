# Model Research Synthesis — 2026-04-11

Synthesized response to the user's questions about Kou vs GBM, model
suitability for binary BTC prediction, the Gu/Kelly/Xiu (2020) paper,
whether neural nets are needed, and a review of three latency-arbitrage
references.

---

## 1. Kou Jump-Diffusion vs GBM — Are They The Same?

**Related but not the same.** Both share continuous Brownian diffusion;
Kou adds Poisson jumps with double-exponentially distributed sizes.

| Model | Captures |
|---|---|
| **GBM (Geometric Brownian Motion)** | Continuous diffusion only. Returns log-normal. Kurtosis = 3 (thin tails). |
| **Kou Jump-Diffusion** | GBM + Poisson(λ) jumps with double-exponential size distribution. Captures fat tails (kurtosis > 3) and crash risk. Reduces to GBM when λ = 0. |
| **What our code actually runs** | "GBM with the label 'kou'." See `signal_diffusion.py:1034-1058` — `tail_mode="kou"` literally returns `norm_cdf(z)`. The actual Kou math runs only under `tail_mode="kou_full"`, which neither btc_5m nor btc_15m configs use. |

The "kou" label is historical baggage from when an earlier code version
had a Kou drift correction that was removed because it produced a
-1.7% to -7% systematic DOWN bias on weekends (visible in the in-code
comment around line 1042-1056). **For btc_5m and btc_15m, we are
running pure Gaussian, NOT Kou jump-diffusion.**

This naming is misleading and worth fixing in a future cleanup. For
analysis purposes, treat "the model" as Gaussian.

---

## 2. Is Pure GBM Suited For Binary BTC Prediction?

**No, demonstrably.** Pure GBM assumes:

- **Constant volatility** (false: BTC has GARCH effects, vol clusters)
- **Log-normal returns** (false: BTC kurtosis is ~6-15 in 1-min windows, GBM expects 3)
- **Independent increments** (false: momentum and mean-reversion at different scales)
- **No regime switching** (false: BTC trades in bursts, calm-burst-calm dominates)
- **Mean drift = 0** (approximately true at 5m, but jumps make this misleading)

Our 2026-04-11 calibration audit measured the consequences directly:

- **Claimed edge: +15.18%** (model's `edge` field at fill time)
- **Realized edge: −0.73%** (actual outcome over 455 trades)
- **Gap: 15.91 percentage points**, concentrated in calm-market windows
  where GBM is most overconfident

The model has SOME directional information (`corr(p_side, win) = 0.40`,
statistically significant), but the magnitude of its claimed edge is
fantasy. This is exactly what one expects from a misspecified model
that captures relative price position but misses jumps, vol regimes,
and feature interactions.

The market_blend parameter (0.3 default for btc) partially corrects
this by anchoring p_model toward the contract mid, but only partially.

---

## 3. Gu, Kelly, Xiu (2020) — "Empirical Asset Pricing via Machine Learning"

**Reference:** Gu, S., Kelly, B., Xiu, D. (2020). Empirical Asset
Pricing via Machine Learning. *Review of Financial Studies*, 33(5),
2223-2273. <https://academic.oup.com/rfs/article/33/5/2223/5758276>

### Their setup
- 30,000+ US stocks
- 60 years of monthly data (1957-2016)
- 94 firm-level characteristics
- 8 macro features
- ~900 feature interactions per stock per month
- Goal: predict next-month excess returns out-of-sample
- Walk-forward time-series cross-validation

### Their model lineup (annualized Sharpe of long-short decile portfolio)

| Model | OOS Sharpe |
|---|---|
| OLS | **negative** (overfits horribly) |
| OLS + Huber + 3 features | 0.79 |
| LASSO/Ridge | 0.93 |
| ElasticNet | 0.93 |
| GLM (additive splines, no interactions) | 0.81 |
| Random Forest | 1.22 |
| Gradient Boosted Trees | 1.30 |
| **NN1 (1 hidden layer)** | 1.32 |
| **NN3 (3 hidden layers)** | **1.35** ⭐ best |
| NN4 | 1.32 |
| NN5 | 1.30 |

### Key findings
1. **Linear models lose** because they can't capture interactions.
2. **GLM (additive splines, no interactions) does NOT beat linear.**
   Critical insight: nonlinearity alone isn't enough — what matters is
   **cross-feature interactions** (momentum × liquidity, volatility ×
   size, etc.).
3. **Trees and shallow NNs both capture interactions** — trees via
   recursive splits, NNs via composition. Both win.
4. **Performance peaks at NN3** then declines. Deeper nets overfit
   even with 22M+ observations.
5. **The best models combine flexibility with regularization.** All
   the winners use early stopping, dropout, or L2 penalties.

### What transfers to our problem
- **Single features have no edge alone.** Our `edge`, `sigma`, `tau`
  by themselves are not predictive. The 2026-04-11 calibration audit
  showed `corr(edge, win) = +0.04` (noise).
- **Cross-interactions are where the signal lives.** Even if you have
  the right features, a linear model won't find the alpha.
- **Trees > linear** is decisive in their data and probably ours.

### What does NOT transfer
- **Sample size:** they have ~22M observations; we have ~500 resolved
  trades. Their statistical power is ~50× ours.
- **Horizon:** they predict monthly returns dominated by fundamentals;
  we predict 5-minute binaries dominated by microstructure.
- **Feature space:** their features are firm characteristics
  (earnings, momentum, valuation); ours are order-book and
  price-process features. Different universes.

---

## 4. Do We Need Neural Nets?

**Short answer: no, but we need a tree model and more features.**

### Sequencing recommendation
1. **Feature engineering first.** Our signal currently sees ~6-8
   features (delta, sigma, tau, edge, market_blend, etc.). Gu/Kelly/Xiu
   show that <30 features can't capture meaningful interactions. Aim
   for 30-50 features. Candidates:
   - Multi-timescale Binance momentum (5s, 30s, 60s, 300s returns)
   - Vol-of-vol (rate of change of σ — regime change detector)
   - Book imbalance at depth (top-1, top-3, top-5 separately)
   - Spread regime (current vs rolling median)
   - Time-of-day encoding (sin/cos of hour, day-of-week)
   - Cross-asset returns at matched horizons (ETH, SOL)
   - TFI (trade flow imbalance) — once recorder is patched
   - Distance from start as fraction of typical move:
     `|delta| / (sigma × sqrt(elapsed))`
   - Recent trade count and size distribution

2. **XGBoost classifier next, NOT a neural net.**
   - **You already have one:** `filtration_model.pkl` is an XGBoost
     trained on 29 features. The OFI agent confirmed it's currently
     INERT in live (only used in backtest path; live never reads it).
   - The right move: augment its feature set, retrain on existing
     parquets, wire it into live as a trade FILTER (not a primary
     signal).
   - Why XGBoost over NN at our scale:
     - Sample efficient (~5-10k samples works fine)
     - Captures interactions natively via splits
     - No GPU required, ~50ms inference
     - Interpretable (SHAP values, feature importance)
     - Less overfitting at small sample sizes

3. **Only if XGBoost shows lift, try shallow NN (NN3).** Per
   Gu/Kelly/Xiu, NN3 is the ceiling — anything deeper hurts. Expect
   maybe +5-10% improvement over XGBoost at our scale. Probably not
   worth the engineering cost until we have much more data.

4. **Don't go beyond NN3.** Their finding was decisive: 3 layers >
   4 > 5. Deeper nets overfit on financial data even with 22M
   observations. With 500-5000 observations we'd be even more limited.

### Why not NN first?
- **Sample efficiency:** trees handle 5-10k samples gracefully; NNs
  need 100k+ to consistently beat trees on tabular data.
- **Interpretability:** when a tree-based model fails, SHAP values
  show which features were responsible. NNs are opaque.
- **Existing infrastructure:** we already have `filtration_model.pkl`
  and the loading logic in `signal_diffusion.py`. No new dependencies.

---

## 5. Latency Arbitrage References — Verdict

### Wah & Wellman (2013) "Latency Arbitrage, Market Fragmentation, and Efficiency: A Two-Market Model"
*Proceedings of ACM EC '13. Updated and superseded by Wah & Wellman 2016 in Algorithmic Finance.*

**Setup:** Single security trades on two exchanges. Aggregate quote
information reaches regular traders with delay D. An "infinitely fast"
arbitrageur sees both venues without delay and profits whenever they
diverge.

**Key findings (genuinely relevant to us):**
1. **The arbitrageur's profit comes entirely from the spread between
   fragmented venues, not from any directional skill.** The arbitrageur
   makes money EVERY time the venues diverge.
2. **Losers are slow regular traders** posting quotes that get picked
   off (i.e., Polymarket makers without latency arb).
3. **Arbitrage edge scales with `volatility × delay × volume_at_stale_price`.**
4. **Strategy is structurally robust** until the venue fights back
   with frequent batch auctions, speed bumps, or cancellation rules.
5. **Replacing continuous-time markets with periodic call markets**
   (frequent batch auctions) eliminates the opportunity entirely.

**Mapping to our setup:**
- Two "venues": Binance (real-time) and Polymarket (1.23s lag via Chainlink RTDS)
- "Slow regular traders": Polymarket makers without latency arb infrastructure
- "Arbitrageur": us, if we ship the latency arb mode
- Polymarket's new dynamic taker fees (up to 3.15% at p=0.5) ARE the
  fight-back. They're attempting to tax the arb. The fee structure
  `7.2% × p × (1−p)` means the fee is small at the wings (~0.5% at
  p=0.20) and large at the mid (~1.8% at p=0.50). **Latency arb on
  wing trades survives; mid trades don't.**

**Our latency arb implementation should:**
- Avoid the 0.40-0.60 mid zone (high fee)
- Trade only at 0.15-0.40 or 0.60-0.85 (acceptable fee)
- Require `Δ_binance > something` so the price move dominates the fee
- Use FOK taker orders (we're not posting, we're crossing the spread)

**Sources:**
- [Wah & Wellman (2013) — Latency Arbitrage, Market Fragmentation, and Efficiency: A Two-Market Model (ACM EC '13)](https://strategicreasoning.org/publications/2013/latency-arbitrage-market-fragmentation-and-efficiency-a-two-market-model/)
- [Wah & Wellman (2016) — Latency arbitrage in fragmented markets: A strategic agent-based analysis (Algorithmic Finance)](https://journals.sagepub.com/doi/10.3233/AF-160060)

### Cube Exchange — Latency Arbitrage explainer
*Mostly textbook material with one useful empirical anchor.*

**Useful number:** LSE study showed ~537 latency races/day on FTSE 100,
modal duration 5-10 microseconds, average 79 microseconds, accounting
for ~22% of FTSE 100 volume and roughly one-third of effective spread.

**Implication for us:** Traditional latency arb is microsecond-scale
and dominated by Citadel, Jump, etc. We're playing in the millisecond
world (Polymarket's 1.23s lag). This means we're competing against
amateurs and slow venues, not Citadel-class HFTs.

**Source:** [Cube Exchange — What is Latency Arbitrage?](https://www.cube.exchange/what-is/latency-arbitrage)

### QuestDB — Latency Arbitrage Models glossary
*Marketing-grade content. Skip.* No math, no actionable insights, no
binary-options or prediction-market specifics.

**Source:** [QuestDB — Latency Arbitrage Models glossary](https://questdb.com/glossary/latency-arbitrage-models/)

---

## 6. Concrete Recommendation — Two Parallel Tracks

### Track A: Latency Arbitrage (already implemented)
- `--latency-arb` flag added in commit `313274e`
- Pure Binance-momentum-based taker mode
- 7/7 smoke tests pass
- **Highest-ROI thing to ship this week**
- Don't complicate it with ML — keep it mechanical
- Test 6 hours dry-run, then small live size

### Track B: ML-Augmented Signal (parallel research, longer timeline)
1. **Engineer 20-30 new features** (1-2 days). See Section 4 above.
2. **Re-train `filtration_model.pkl`** with new features on existing
   parquets (1 day).
3. **Wire it into live** as a trade filter. The model is currently
   loaded only in the backtest path; needs to be loaded in
   `live_trader.py` and consumed by `decide_both_sides()` (it already
   has the consumption logic — it just isn't called in live). 0.5 day.
4. **A/B test** with and without filter for 48h live (2 days).
5. **ONLY if XGBoost shows >2pp WR lift, try shallow NN (NN3).** 1 week.

**Track B total:** ~1 week before any live test, ~2 weeks before
knowing if it helped.

### Which track first?
**Latency arb (Track A) wins per engineering hour**, BUT:
- Latency arb edge will erode as Polymarket adds more countermeasures
- Track B benefits BOTH tracks (TFI features useful for both)
- Diversification across two strategies is the long-term insurance

**Recommended:** Ship latency arb this week (already done in code),
then immediately start Track B feature engineering while latency arb
runs in dry-run / small-size live. The two efforts don't compete for
the same resources.

---

## What NOT To Do

- **Don't build a NN from scratch.** Sample size is too small. NN3
  was Gu/Kelly/Xiu's best with 22M observations; we have 500.
- **Don't add 900 features.** That's their world (broad cross-section),
  not ours (deep time series of one asset). Aim for 30-50 features
  that capture multi-timescale dynamics.
- **Don't believe a backtest** without parity verification. Today's
  whole calibration story started with a backtest running a different
  model than live (the max_z bug from `parity_fixes_applied_2026-04-11.md`).
  Until live-vs-backtest parity is proven on whatever new model we
  build, treat backtest Sharpe as suggestive only.
- **Don't run latency arb at mid-priced contracts** (0.40-0.60). The
  new dynamic taker fees eat the edge. Stick to 0.15-0.40 / 0.60-0.85.
- **Don't fight the fee structure.** Wah & Wellman show that venues
  can eliminate latency arb with the right rules. Polymarket is
  trying. Build the strategy to survive the wings, not depend on the
  mid.

---

## Top-Level Sources

1. [Gu, Kelly, Xiu (2020) — Empirical Asset Pricing via Machine Learning, Review of Financial Studies](https://academic.oup.com/rfs/article/33/5/2223/5758276)
2. [Wah & Wellman (2013) — Latency Arbitrage, Market Fragmentation, and Efficiency: A Two-Market Model](https://strategicreasoning.org/publications/2013/latency-arbitrage-market-fragmentation-and-efficiency-a-two-market-model/)
3. [Wah & Wellman (2016) — Latency arbitrage in fragmented markets: A strategic agent-based analysis (Algorithmic Finance)](https://journals.sagepub.com/doi/10.3233/AF-160060)
4. [Cube Exchange — What is Latency Arbitrage?](https://www.cube.exchange/what-is/latency-arbitrage)
5. [QuestDB — Latency Arbitrage Models glossary](https://questdb.com/glossary/latency-arbitrage-models/) (low value)
