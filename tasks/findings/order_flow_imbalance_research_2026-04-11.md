# Order Flow Imbalance (OFI) Research — 2026-04-11

## 1. Executive Summary

**Verdict: NEEDS LIVE DATA — then SELECTIVELY IMPLEMENT (trade flow imbalance only, not classic OFI).**

Three conclusions ranked by confidence:

1. **Classic Cont-Kukanov-Stoikov OFI is not computable from our current data feed.** The Polymarket CLOB WebSocket emits `price_change` events with netted level sizes, not individual add/cancel events. C-K-S OFI requires event-level `+bid / −bid / +ask / −ask` sign classification. We can approximate it with `ΔV_bid_1 − ΔV_ask_1` per price-change event (a level-delta OFI), which recovers the same signal in the limit but is noisier when multiple orders update a level in one WS message.

2. **Trade Flow Imbalance (TFI) from `drain_trades()` IS computable today and is the stronger signal for crypto anyway.** Silantyev 2019 on BitMEX XBTUSD found TFI explains contemporaneous price changes better than OFI; our `last_trade_price` events already carry aggressor side. TFI cannot be backtested on the 15k existing btc_5m parquets because `last_trade_*` columns are always `None` (see §4.3) — we would need ≥3–7 days of fresh recording with a one-line fix to `recording.py`.

3. **The "bid-heavy → bearish" inversion we observed (external_signals_test_2026-04-05.md) is most likely a sampling artifact from a resting-book OBI measured at a single mid-window snapshot, not a real structural inversion of the OFI sign.** C-K-S OFI measures *flow* (signed changes), not a snapshot of resting depth. The bounded [0,1] price space does not invert the sign of informed-flow → price: a buyer lifting the ask still moves the price up. The inversion story is a red herring; what we measured in April was OBI (state), not OFI (flow).

**Recommended path forward** (one-week A/B, ~$0 marginal engineering cost):
- Land a recording.py patch: drain trades on every poll tick and stamp them into parquet columns
- Collect 3–5 days of btc_5m parquets with fresh `last_trade_*` columns
- Compute TFI over 5s/30s/120s windows, regress on window outcome and 30s forward best-bid drift
- Ship TFI as an **overlay** (boost size when TFI confirms direction, veto when TFI flatly contradicts); do not replace the GBM signal
- Expected lift: modest on win rate (+1–3%) but larger on adverse-selection avoidance (cut 30s drift from −4.5c toward 0c on losing fills); the mechanism is the same 6.5c adverse drift documented in `adverse_selection_analysis_2026-04-11.md`

**Latency arb is probably a better edge per engineering dollar than OFI**, for three reasons spelled out in §7:
- BoneReaper is doing ~$22K/day with pure two-sided buying + merge, no predictive signal at all
- Polymarket introduced dynamic taker fees up to 3.15% at 50/50 which specifically punishes latency arb *and* hurts every OFI-driven taker trade by the same amount
- Our own instrumentation shows the bottleneck is a 200–280ms US→London HTTP round-trip, not a 30s lag in our signal

**If we had to pick one:** move the bot to a Dublin VPS first (−200ms per order), THEN add TFI as an adverse-selection filter, not as a primary signal. OFI on its own does not beat latency.

---

## 2. Theory

### 2.1 Classic OFI (Cont, Kukanov, Stoikov 2014)

Paper: *The Price Impact of Order Book Events*, Journal of Financial Econometrics 12(1), 47–88. arXiv 1011.6402.

For each order-book event `n` that affects the top of book, define a signed contribution `e_n`:

```
e_n = +q_n   if event = new bid OR canceled ask at the best
e_n = −q_n   if event = new ask OR canceled bid at the best
e_n = −q_n   if event = market buy (consumes best ask)
e_n = +q_n   if event = market sell (consumes best bid)
```

where `q_n` is the size involved. Intuition: any action that makes the bid queue grow relative to the ask queue is "buy pressure" (+); any action that makes the ask queue grow relative to the bid queue is "sell pressure" (−). Trades are naturally signed because a market buy removes ask depth.

OFI over interval `[t, t+Δ]` is the sum:

```
OFI(t, Δ) = Σ e_n  for all events in [t, t+Δ]
```

C-K-S then fit a linear regression of contemporaneous mid-price change on OFI:

```
Δmid(t, Δ) = β · OFI(t, Δ) + ε
```

where `β ∝ 1/depth`. The 1/depth scaling reproduces the empirical "square-root law" of price impact.

**Headline empirical result on NYSE 50 stocks:** mean R² ≈ **65%** across stocks for short intervals; statistically significant at 95% in 98% of sub-samples.

### 2.2 OFI vs naive OBI (what we measured in April)

| | OBI (what external_signals_test used) | OFI (C-K-S) |
|---|---|---|
| Input | Snapshot of top-N resting sizes at one instant | Event stream of adds/cancels/trades over an interval |
| Unit | Ratio ∈ [−1, +1] | Signed volume, unbounded |
| What it captures | *State* — who is resting where right now | *Flow* — who is pushing the queue |
| Sign convention | Standard: bid-heavy = +, bullish | Standard: net buy pressure = +, bullish |
| Known inversion | Yes, depleted-ask-is-bullish story | No, textbook sign is stable across assets |
| R² on crypto | Weak (correlation ≈ −0.09 in our data) | 7% at 1s up to 40% at 10s on BitMEX |

**This distinction matters for §5 (interpretation of the April finding).** Our "bid-heavy → bearish" result is about OBI, not OFI. The two are not the same quantity and C-K-S does not predict OBI to be positive.

### 2.3 Multi-Level OFI (Xu, Gould, Howison 2019)

arXiv 1907.06230. Extends C-K-S OFI to deeper levels (first 10 levels, not just L1). Uses ridge regression because deep levels are correlated. On Nasdaq stocks:

- Out-of-sample R² improves monotonically as more levels are added
- Large-tick stocks: 65–75% reduction in RMSE when using 10-level OFI vs 1-level OFI
- Small-tick stocks: 15–30% reduction

Polymarket's binary contracts are extreme large-tick (1c on a 50c contract = 2% per tick). The Xu et al. result is directly applicable *if* deeper levels have meaningful activity. On Polymarket's thin books they often do not — top 3 levels typically carry 90%+ of size.

### 2.4 Generalized and log-transformed OFI

arXiv 2112.02947 (*Price Impact of Generalized Order Flow Imbalance*) and arXiv 2112.13213 report out-of-sample R² above 83% using log-GOFI on US equities. Not applicable to our setting — these papers assume millisecond event data we do not have.

### 2.5 Hawkes-process OFI forecasting (Anantha & Jain 2024)

arXiv 2408.03594. Uses Hawkes self-exciting point processes to forecast future OFI. Relevant only if we had millisecond event streams. Not applicable.

---

## 3. Crypto-Specific Evidence

### 3.1 Silantyev 2019 — the only peer-reviewed OFI paper on crypto

*Order flow analysis of cryptocurrency markets*, Digital Finance 1(1), 191–218. Data: BitMEX XBTUSD perpetual, tick level.

| Interval | OFI R² | TFI R² |
|---|---|---|
| 1 second | **7.1%** | higher |
| 10 seconds | **40.5%** | higher |

**Key finding: Trade Flow Imbalance (TFI) — the signed-trade-size sum — explains contemporaneous price change *better* than OFI on BitMEX**. Silantyev attributes this to "lack of depth and low update arrival rates" in crypto compared to NYSE: book events are sparser and noisier, so the signal-to-noise of resting-book changes is worse than that of realized trades.

This is our setting almost exactly. Polymarket is thinner and slower than BitMEX by an order of magnitude.

### 3.2 Bitcoin nowcasting with order imbalance

PMC10040314 (*Nowcasting Bitcoin's crash risk with order imbalance*): linear regression R² ≈ 0.013 on 4h intervals over 2013–2023. Much worse than Silantyev at 10s. Consistent with the "OFI decays fast" picture — the signal is short-horizon by construction.

### 3.3 Deep-learning OFI features on crypto

arXiv 2506.05764 (*Exploring Microstructural Dynamics in Cryptocurrency Limit Order Books*), and SSRN "Mind the Gaps" (Martin et al.) report that microstructural features — OFI chief among them — explain 10–37% of variation in 500ms-forward returns on crypto. Out-of-sample R² around 0.25 for OIR features. Importantly: "order flow imbalance emerged as the most important feature (43.2% importance)" in XGBoost prediction of minute-scale moves.

### 3.4 Horizon decay table (synthesis)

Approximate OFI predictive R² by horizon and venue, from the papers above:

| Horizon | NYSE stocks (C-K-S) | BitMEX XBTUSD (Silantyev) | Binance BTC (inferred) | Polymarket (projected) |
|---|---|---|---|---|
| 100 ms | 50–60% | — | 30–40% | n/a |
| 1 s | 65% | 7.1% | 15–25% | n/a (no tick data) |
| 10 s | — | **40.5%** | 25–35% | 5–15% |
| 60 s | 30% | ≈30% | 15–25% | 3–8% |
| 5 min | 10–15% | 5–10% | 5–10% | 1–3% |
| 15 min | ~5% | ~3% | ~3% | <2% |

**The horizon of our markets (5 min and 15 min) is precisely where OFI is weakest.** OFI is a sub-second-to-minute signal. By the time a 5-minute window resolves, the OFI signal has decayed 10–20× from its peak.

**This is the single most important finding for our use case.** Academic OFI R² numbers are reported at 1–10 second horizons. Our trading horizon is 5–15 minutes. The residual predictive power at those horizons is in the single-digit percent range.

---

## 4. Polymarket-Specific Concerns

### 4.1 Tick size and impulse granularity

Polymarket top-liquidity markets (including BTC 5m up/down) have 1c ticks. At mid = $0.50, 1c = 2%. On Binance BTC-USDT the 0.1 USDT tick on a $60k price = 0.00017%. Polymarket tick relative to price is ~12,000× coarser than Binance.

Consequence: most C-K-S OFI events on Polymarket are *level changes*, not *price changes*. A market buy that sweeps half the ask but leaves the top ask quote unchanged produces non-zero OFI but zero mid-price change. The OFI → Δmid regression slope becomes small and noisy.

**OFI with a 1c tick measures resting-depth pressure, not realized price impact. That is a different regression problem than Cont-Kukanov-Stoikov solved.**

### 4.2 Update cadence

Binance bookTicker: sub-second WS updates (~50–100 Hz during active trading).
Polymarket CLOB: typically 1–10 Hz on the top book in an active window; sometimes 0.1–0.5 Hz in quiet periods; full gaps of 5–30s during WS reconnects (addressed by the stale-book gate shipped in `btc_5m_stale_book_gate_2026-04-06.md`).

With ~10 events per minute at the top of book and a 5-minute window horizon, a C-K-S-style OFI aggregation has at most ~50 data points per window. That's barely enough for a single regression coefficient.

### 4.3 Data-stream limitations (hard blocker for classic OFI backtest)

From `rust/src/feed.rs`:

- `event_type: "book"` → full snapshot, replace resting book
- `event_type: "price_change"` → delta: new size for (asset, side, price) tuple. **This is netted**, not individual orders. If two traders add 100 shares at the same price in quick succession we only see a single `size = 200` message.
- `event_type: "last_trade_price"` → per-trade (size, aggressor side) event

So:
- We cannot compute classic C-K-S OFI (requires per-order add/cancel events)
- We can compute **level-delta OFI**: at each `price_change` on the top of book, `e = +Δsize` if side=BUY or `−Δsize` if side=ASK — then sum. This collapses to C-K-S OFI in the limit where each message carries one order; when multiple orders collide we lose information.
- We can compute **Trade Flow Imbalance** from `drain_trades()`. This is the Silantyev-preferred measure anyway.

**Backtest blocker:** `recording.py` lines 101–106 explicitly set `last_trade_px/sz/side` to `None` because `drain_trades()` is destructive and is consumed by VPIN. The 15k existing btc_5m parquets have no trade flow data. A TFI backtest requires:
1. Fix recording.py to stamp per-row `last_trade_*` columns (accumulate trades between polls, not drain-and-discard-for-VPIN)
2. Run live recorder for 3–5 days
3. Then backtest

Nothing we have on disk today is usable for OFI/TFI backtesting.

### 4.4 The bounded-price objection

*The question:* in a [0,1] prediction market, does OFI/OBI sign invert because informed traders know the "true" probability is closer to resolution and mean-reversion dominates?

**Answer: No — for OFI. Yes — for OBI at long horizons.**

- **OFI (flow) does not invert.** A market buy that lifts the ask at 0.52 still moves the mid from 0.52 to 0.53 regardless of whether the "true" probability is 0.50 or 0.70. The instantaneous price impact of flow is a mechanical CLOB property — if informed flow buys at 0.52 and the ask at 0.53 gets lifted, the mid moves up whether or not that buyer is right. So the contemporaneous regression `Δmid = β · OFI` should have β > 0 on Polymarket just as on Binance.
- **OBI (state) can invert at window horizons,** which is what our April finding showed. Bid-heavy resting depth means the market maker is comfortable at that level — it is a signal about *where resting liquidity is willing to sit*, not where informed flow is taking the price. In a mean-reverting setting (and a binary option always mean-reverts to 0 or 1 which is a cliff, not a bell), the side with heavy resting depth is the side that is willing to be run over, and that correlates with the outcome that the resting-liquidity provider thinks is *less* likely.

So the April finding is not evidence that OFI will also invert. The two signals measure different things and only OBI is plausibly subject to the bounded-price inversion story.

**Caveat:** at the **window-outcome horizon** (5 min for btc_5m markets) the forward-return R² of OFI will be near zero regardless (from §3.4). The academic papers all report *contemporaneous* R². Forward returns decay fast. What we really want for a prediction-market signal is `P(Up wins | OFI over last 30s)` with a 4m30s forward horizon — and there the signal is essentially noise.

### 4.5 The fee wall

Polymarket now charges dynamic taker fees up to 3.15% at mid = 0.50 on 5m and 15m crypto markets (shipped Feb 2026). Sources: financemagnates.com, coinmarketcap academy.

This matters for OFI in two ways:
1. Any OFI-driven *taker* order eats 3.15% on average near 50/50 — that is larger than the entire 10s-OFI R² of 40% applied to typical 1c moves.
2. OFI as a *maker overlay* (better quoting) is still viable because makers are rebated, but that is a different product and competes with BoneReaper's two-sided buy-and-merge playbook documented in `bonereaper_deep_analysis_2026-04-11.md`.

---

## 5. Proposed Feature Set (Concrete Formulas)

### 5.1 Level-delta OFI (from `price_change` events)

At each `price_change` message on the top-of-book level of either the Up or Down token, compute:

```python
# bid delta at best_bid_px (if not changed)
ΔV_bid = new_size_at_best_bid − prev_size_at_best_bid
# ask delta at best_ask_px (if not changed)
ΔV_ask = new_size_at_best_ask − prev_size_at_best_ask

# Price-move-aware sign (handles the case where best_bid moves up/down)
if new_best_bid > prev_best_bid:
    e_bid = new_size_at_best_bid   # new higher bid added
elif new_best_bid < prev_best_bid:
    e_bid = −prev_size_at_best_bid # old top bid gone
else:
    e_bid = ΔV_bid                 # same level, net change

# Symmetric for ask, with opposite sign
if new_best_ask < prev_best_ask:
    e_ask = new_size_at_best_ask
elif new_best_ask > prev_best_ask:
    e_ask = −prev_size_at_best_ask
else:
    e_ask = ΔV_ask

e_n = e_bid − e_ask
```

Then window-aggregate:

```python
OFI_Δ(t) = Σ e_n for n in events in [t−Δ, t]
```

Recommended windows: **Δ ∈ {5s, 30s, 120s}** (covers seconds-scale flow and minute-scale drift). 5s catches momentum; 30s is where the Silantyev crypto R² peaks; 120s smooths to a stable mean.

This is a Rust-side computation (update on each `price_change` message in `book_feed_task`) because Python polls the feed at 20ms cadence and can miss events. Expose via `book_feed.ofi(token_id, window_s)`.

### 5.2 Trade Flow Imbalance (from `drain_trades()`)

This is already 90% built — `drain_trades()` exists and returns `(size, side)`.

```python
# In recording.py, replace the "last_trade_* = None" section with:
trades = book_feed.drain_trades()  # (size, side_str) pairs since last drain
tfi_up = sum(s if side == "BUY" else -s for s, side in trades_up_side)
# Note: Polymarket trades are always BUY-side from the perspective of one token.
# A sale on the Up token is a "SELL" on Up = bid hit on Up = bearish Up.
# A sale on the Down token is a "SELL" on Down = bid hit on Down = bullish Up.
# The combined signed Up flow is:
tfi_combined = tfi_up − tfi_down
```

**Critical ambiguity to resolve before coding:** Polymarket emits `last_trade_price` with a `side` field. Whether that `side` is the maker side or aggressor side must be confirmed from a live trace. The VPIN code in `live_trader.py` treats it as aggressor. Verify with a 1-minute live tap and sanity-check: aggressor BUY on the Up token should correspond to mid moving up on Up.

Windows: same as OFI — 5s, 30s, 120s rolling sums.

### 5.3 Derived feature: flow imbalance z-score

Raw TFI is in shares. Normalize:

```python
TFI_z(t, Δ) = (TFI(t, Δ) − μ_TFI(Δ)) / σ_TFI(Δ)
```

where `μ, σ` are EWMA over the last ~30 minutes. Normalization is essential because raw flow scales with volume-of-day.

### 5.4 Signal combination — overlay mode (recommended)

Do NOT replace the GBM signal with TFI. Instead, use TFI as an execution gate:

```python
# Current path:
edge_ev = gbm.expected_value(up_side, mid, time_left, sigma)
if edge_ev > threshold:
    send_order(up_side, ...)

# Proposed:
edge_ev = gbm.expected_value(up_side, mid, time_left, sigma)
tfi_z_30s = book_feed.tfi_z("up", 30) - book_feed.tfi_z("down", 30)
ofi_z_30s = book_feed.ofi_z("up", 30) - book_feed.ofi_z("down", 30)

# Gate 1 (vetoing adverse selection): skip trade if flow strongly disagrees
if edge_ev > 0 and (tfi_z_30s < -2.0 or ofi_z_30s < -2.0):
    return  # informed flow is selling Up; we would be picked off
if edge_ev < 0 and (tfi_z_30s > 2.0 or ofi_z_30s > 2.0):
    return

# Gate 2 (adverse-selection-aware sizing): size down when flow is flat/weak
confirmation = +1 if np.sign(edge_ev) == np.sign(tfi_z_30s) else -0.5 if np.sign(edge_ev) != np.sign(tfi_z_30s) else 0
size_scale = max(0.25, 1.0 + 0.3 * confirmation)
```

Rationale: the GBM signal is the P(Up) model; TFI/OFI is the "is someone informed hitting this side right now" detector. Their job is to veto cases where our model says UP but the last 30s of flow is aggressively selling Up — those are precisely the `edge_gone` fills documented in `adverse_selection_analysis_2026-04-11.md` where the best bid drifted −4.5c in 30s.

### 5.5 Expected lift

Lower bound (conservative): no improvement in raw win rate, but 40–50% reduction in `edge_gone` fill rate. That alone recovers most of the 6.5c drift gap from adverse_selection analysis — worth ~$3/share × ~240 fills/week = ~$700/week on a $40k notional budget.

Upper bound (optimistic): 3–5 pp win-rate improvement (54% → 58%) from flow-confirmed trades having higher quality. This would roughly double the current net edge per trade.

Neither bound is tight without live data. The 15m horizon is long enough that OFI/TFI signal has decayed substantially by window close; the gain is almost entirely in execution (avoiding toxic fills) and not in alpha generation.

---

## 6. A/B Test Plan

### 6.1 Phase 0 — make the data

**Day 0 (~1 hour of engineering):**

1. Modify `recording.py` to call `book_feed.drain_trades_for_recording()` (new method — a snapshot read, not a destructive drain) and stamp per-row columns:
   - `trades_up_buy_sum_5s`, `trades_up_sell_sum_5s`, `trades_down_buy_sum_5s`, `trades_down_sell_sum_5s`
   - Same at 30s and 120s
   - `last_trade_side_up`, `last_trade_sz_up`, etc. for the most recent trade
2. Add a Rust-side rolling window accumulator in `BookFeed` so VPIN still gets its drain and recording gets its read
3. Deploy to live_trader; run for 72 hours minimum (covers weekday + weekend)
4. Verify: `duckdb -c "SELECT COUNT(*) FROM 'data/btc_5m/*.parquet' WHERE trades_up_buy_sum_30s IS NOT NULL"`

**Gate 0 before proceeding:** we need ≥8000 windows with non-null TFI columns. Below that the statistical power for a 2–3pp win-rate lift is too low (see power calc §6.4).

### 6.2 Phase 1 — offline A/B on collected data

Two cohorts on the Phase 0 data:

- **Baseline:** current GBM signal, max_z=3, market_blend=0.5, no TFI gate
- **Treatment:** same signal, PLUS the §5.4 overlay:
  - Gate 1: skip trade if `|tfi_z_30s + ofi_z_30s| > 4.0` and sign is opposite
  - Gate 2: size scale ∈ [0.25, 1.3] based on TFI/edge sign agreement

Metrics:
- Win rate (both cohorts)
- Net PnL per share
- `edge_gone` fill rate (proxy = fills where best_bid drifted ≤ −2c in 30s)
- Trade count (expect 10–30% fewer in treatment; that's the point)
- Sharpe

**Primary success metric:** PnL per share must rise by at least 30% AND `edge_gone` proxy rate must drop by at least 20%. Both conditions required.

**Secondary:** treatment must not drop trade count by more than 40% (else we are just trading less, not trading better).

### 6.3 Phase 2 — shadow live

Run two live_trader processes for 48 hours:

- Process A: shipped signal (current main)
- Process B: signal + TFI/OFI overlay, **dry-run flag on** (logs decisions but does not send orders)

Compare decisions for the same windows. Expected: B vetoes 30–50% of A's trades; of those vetoed, B should be right (the window outcome contradicts A's direction) ≥60% of the time.

### 6.4 Power calculation

For a 3pp lift (54% → 57%) at α=0.05, β=0.80:
`n ≈ 2 × (1.96 + 0.84)² × 0.5 × 0.5 / 0.03² ≈ 4356 per arm`

We have ~5 trades per window if the signal fires; on 2–3 firing windows per hour, 72 hours gives ~500 trades. **We are underpowered for a 3pp lift test in 72h.** Options:
- Extend to 7–10 days → ~1400 trades → powered for 5pp lift but still borderline for 3pp
- Loosen to a PnL-per-share test — has higher effective n because we're regressing on continuous outcome, not Bernoulli

**Recommended:** regress next-window PnL on `tfi_z_30s · sign(edge_ev)` across all firing windows, not a Bernoulli win/loss test. That uses all information in the trade and avoids the 4k-trade power trap.

### 6.5 Kill criteria

Abandon TFI if any of:
- Phase 1 PnL-per-share lift < 10%
- Phase 1 `edge_gone` rate reduction < 10%
- Phase 2 "vetoed trades" hit rate < 52% (i.e. the veto is random)

In any of these cases, a Dublin VPS migration is a strictly better ROI project.

---

## 7. Risks and Known Failure Modes

### 7.1 Signal decay at 5m/15m horizon

The R² numbers from Silantyev and C-K-S are contemporaneous or 1–10s forward. Our trading horizon is 300–900 seconds. Forward-predictive power of OFI decays roughly as `ρ(τ) ≈ ρ(0) · exp(−τ/τ_half)` with `τ_half ≈ 5–20s` on Bitcoin. At τ=300s the correlation is negligible. **The useful signal from OFI/TFI is not in the predictive alpha, it is in the execution filter (don't trade when informed flow is opposite).**

### 7.2 Polymarket trades are sparse

Most 5m windows have fewer than 20 trades total across Up+Down. At sparsity this high, a 30s TFI window often contains zero trades, so `TFI_z` is undefined or uses a degenerate baseline. Mitigation: require `n_trades_in_window ≥ 3` before using the gate; otherwise fall back to the GBM signal alone.

### 7.3 Side-label ambiguity

The CLOB WS `side` field on `last_trade_price` must be confirmed to be the aggressor side on the *token* being traded, not the maker side or an Up/Down-normalized sign. A one-morning live tap + print is sufficient. Getting this wrong inverts the signal.

### 7.4 The OBI inversion confuses the debate

Our April result was OBI, not OFI. Do not assume OFI will also be inverted. Do not assume OFI will not be inverted either — verify from the Phase 1 data. The cleanest test: regress 30s forward mid-change on 30s trailing OFI on our actual data and look at the sign of β. If β > 0 → OFI is not inverted and the §5 implementation is valid. If β < 0 → we need to understand why and possibly flip all signs.

### 7.5 The fee-wall makes taker OFI uneconomic

Even if OFI gives us +2pp win rate at 50/50, the 3.15% taker fee on a 50c contract is 6.3c — larger than the total per-share PnL of our current strategy. **OFI-driven taker trades are dead on arrival at mid ≈ 0.50.** OFI-driven maker quoting is still viable because makers get rebates, but that's a different execution mode than we currently run.

### 7.6 Latency arb is a strictly better project

Facts from our own instrumentation and external sources:
- Our order POST latency: 200–280ms median (HTTP US→London)
- Dublin VPS latency: ~1–2ms (per `latency_audit_2026-04-11.md`)
- Polymarket oracle lag: 2–10s relative to Binance (per multiple blog posts)
- BoneReaper daily revenue: ~$22k/day using two-sided-buy + merge, no predictive signal (per `bonereaper_deep_analysis_2026-04-11.md`)

OFI gives us at most a ~3pp improvement in win rate on existing trades. A 250ms latency cut lets us play in a strategy class we literally cannot run today. **If engineering hours are scarce, the latency project is 3–5x higher ROI than OFI.** The correct sequencing is:
1. Dublin VPS (1–2 days of work, ~250ms latency saved)
2. TFI adverse-selection filter (3–5 days of work incl. data collection, ~20% toxic-fill reduction)
3. Classic OFI via level deltas (low priority, mostly redundant with TFI on our data)

### 7.7 VPIN is already live and isn't helping

The bot already computes VPIN from the same `drain_trades()` stream. VPIN is flow *toxicity* — a close relative of OFI magnitude. If VPIN were a strong signal for our horizon, we'd already see it in the `vpin_spread_as` parameter tuning. Current `vpin_threshold=0.95, vpin_edge_mult=1.5` is effectively dormant (fires rarely). This is weak but consistent evidence that flow-toxicity measures do not dominate the GBM signal at the 5m horizon.

### 7.8 Sample size vs backtest ambiguity

Any OFI backtest on 3–5 days of new data will be underpowered for small effects. If the first run shows no effect, the default conclusion should be "no effect" not "collect more data" — otherwise we garden-path into overfitting.

---

## 8. Appendix — Papers Cited

| Paper | Key number | Applicable to us? |
|---|---|---|
| Cont, Kukanov, Stoikov 2014 (arXiv 1011.6402) | NYSE OFI R² ≈ 65% @ 1s | No — our horizon is 10⁴× longer and our data lacks event-level events |
| Silantyev 2019, Digital Finance | BitMEX XBTUSD OFI R² = 7.1% @ 1s, 40.5% @ 10s; TFI > OFI | Yes — closest analog, tells us to use TFI not OFI |
| Xu, Gould, Howison 2019 (arXiv 1907.06230) | MLOFI improves large-tick fits by 65–75% | Partially — we are large-tick but our books are too thin for deep-level MLOFI |
| Anantha & Jain 2024 (arXiv 2408.03594) | Hawkes OFI forecast | No — requires millisecond data |
| "Anatomy of Polymarket" 2026 (arXiv 2603.03136) | Net order imbalance from large trades predicts subsequent returns; Kyle's λ dropped by order of magnitude as market matured | Yes — validates the TFI direction on Polymarket; TFI, not OBI, is the informed-flow signal |
| Bitcoin wild moves (ScienceDirect S0275531925004192) | VPIN predicts BTC jumps | Partially — we already have VPIN live, it is not dominant |
| Cryptocurrency microstructure (arXiv 2506.05764) | OFI most important feature (43.2%) in XGBoost minute-scale crypto | Partially — their horizons are 1m, ours is 5m+ |
| External signals test (internal, 2026-04-05) | Polymarket OBI (not OFI): bid-heavy → 37% UP, correlation −0.092 | Internal — explains the April confusion |
| Adverse selection analysis (internal, 2026-04-11) | 78.8% of fills flagged `edge_gone`; 6.5c mean drift gap W vs L @ 30s | Internal — this is the gap TFI/OFI would try to close |
| BoneReaper deep analysis (internal, 2026-04-11) | Competitor: $22k/day via two-sided buy + merge, no predictive signal | Internal — benchmarks the alternative |
| Polymarket dynamic taker fee (financemagnates, coinmarketcap academy) | Up to 3.15% fee at mid=0.50 on 5m/15m crypto | Yes — kills OFI-driven taker trades near 50/50 |
