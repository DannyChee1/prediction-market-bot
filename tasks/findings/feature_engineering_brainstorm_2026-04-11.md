# Feature Engineering Brainstorm — BTC 5min Polymarket Filter Model

*Compiled 2026-04-11. Goal: find new features for the XGBoost filter that predicts whether the GBM signal's directional call will be correct over the remaining 5-min window. Priority: features that exploit information the GBM model does NOT already use. Latency-arb remains the primary strategy — these features should AUGMENT execution gating, not replace it.*

---

## 0. Context Recap (why most of the current model is a tautology)

The 29-feature model is dominated by `z²` and `|z|` (43% of importance). That means the filter is essentially saying "the GBM signal is more likely to be right when the signal is stronger." That is literally the GBM model's own internal confidence rebadged — **it adds almost nothing**. The only real uplift is coming from the 8 other features that combined get <60% importance, and OBI already contributes 0%.

What that tells us: **the filter has no information the GBM model didn't already have.** The whole point of new features is to inject information the GBM doesn't know about. GBM sees:
- a few binance_mid ticks turned into z, sigma, tau
- nothing about book shape, book flow, cross-asset, adverse selection, volatility decomposition, or regime

So every new feature needs to pass one test: **does this contain information that would change the GBM's prior?** If it's a smoother z-score, skip it.

There is also a hard asymmetry we have to respect: the backtest already showed OBI is useless, and session/streak signals are not statistically significant. That's not because microstructure is broken — it's because we've been computing it from a 1Hz snapshot of the Polymarket L5 book, with no Binance trade tape, no Polymarket trade tape at tick resolution, and no cancel/modify stream. **Most of the highest-signal features in the literature are not computable from our current parquet schema.** Section 5 tracks this explicitly.

---

## 1. Executive Summary — Top 10 Features Ranked

Ranking is by `(expected predictive impact) × (feasibility with current data) × (orthogonality to current features)`. Rank 1 is where I would start tomorrow morning.

| # | Feature | Category | Impact | Data? | Orthogonal? | Why it matters |
|---|---|---|---|---|---|---|
| **1** | **Microprice — mid offset** | Microstructure | HIGH | YES | YES | Stoikov (2018) showed this beats mid as a 3–10s forecaster. Trivial from our L5 snapshot. Directly addresses why raw OBI was useless: microprice is the *non-linear* function of imbalance the literature validated. |
| **2** | **Binance–Polymarket basis (implied P vs book P)** | Fair value disagreement | HIGH | YES | YES | The GBM fair-value P is already a signal; the *residual* between book mid and GBM P is a model-market disagreement — and when the disagreement is stale-book-shaped, it's the latency-arb edge the bot is supposed to harvest. Currently only used as a threshold, not as a feature. |
| **3** | **Time since last book update (polymarket_book_age_ms)** | Stale-quote | HIGH | YES | YES | The single best predictor of "is this fill going to be adversely selected" per the adverse_selection_analysis findings. Bot already tracks this but doesn't feed it to the filter. Zero extra work, massive leverage. |
| **4** | **Binance tick-clock realized variance (last 30/60/300s)** | Vol decomposition | HIGH | YES | YES | HAR-RV (Corsi 2009) style three-timescale vol. The existing sigma is one scalar; HAR-RV says the *ratio* of short/medium/long realized vol is the regime. On BTC 5m the 30s/300s ratio is almost certainly predictive of window outcome, and GBM ignores it. |
| **5** | **Bipower-variation jump indicator (last 5 min)** | Vol decomposition | HIGH | YES | YES | Barndorff-Nielsen & Shephard. Detects whether recent vol is continuous or jump-driven. Jump regimes kill the GBM Gaussian assumption — this tells the filter "don't trust the z-score, it was computed on a jump." |
| **6** | **Polymarket tick-stability counter** | Polymarket microstructure | MED-HIGH | YES | YES | Number of consecutive 1s ticks where best-bid didn't move. Long stability runs = stale market-maker quote = latency-arb edge open. This is the exact inverse of the `edge_gone` pattern in the adverse-selection analysis. |
| **7** | **Window path roughness (|z| crossings & drawdown from peak)** | Window-relative | MEDIUM | YES | YES | GBM is memoryless; it doesn't know the path already went above z=2 and came back. Path features capture "this window already exhausted its move" vs "still trending." Free from existing data. |
| **8** | **BTC–ETH lead-lag z-score (Binance perp)** | Cross-asset | MEDIUM | NO (needs eth feed on btc run) | YES | Weak on 5m-15m in isolation but the literature has it significant at 1-5s leads. The live system doesn't wire ETH into BTC's filter. Cheap add if the feeds worker streams ETH anyway. |
| **9** | **Binance book pressure asymmetry (weighted depth imbalance)** | Microstructure | MEDIUM | NO (need Binance L5 recorder) | YES | Binance depth imbalance ≠ Polymarket depth imbalance. One is the primary reference price's book, the other is secondary. Cont-Kukanov-Stoikov (2014) gold standard; Polymarket OBI proxy is just too thin. |
| **10** | **Window-elapsed-fraction × signed momentum interaction** | Window-relative × time | MEDIUM | YES | PARTIAL | The filter has `|z|*tau` but not `sign(momentum)*elapsed_fraction`. Late-window continuation and early-window reversion are documented regimes — our existing sign-less `tau` feature cannot capture them. |

**One-line strategic answer**: *The two features that will move the needle fastest are (1) microprice offset on Polymarket's own book, and (2) Binance-Polymarket basis as a residual vs the GBM's implied probability.* Both are computable from data you already have on disk. Both are orthogonal to `z²`. Neither has been tried in the current filter.

---

## 2. Feature Catalogue by Category

### 2.1 Multi-timescale momentum / reversal / realized vol

The HAR-RV (Heterogeneous Autoregressive Realized Volatility) framework from Corsi 2009 is the workhorse for intraday vol. The insight: vol at different horizons is heterogeneous and predictive of future vol at *all* horizons, in an additive cascade. We should steal the *feature structure* even without estimating the HAR coefficients.

**F1. Realized variance at 30s, 120s, 300s (HAR-RV triple)**
- Formula: `RV_h = Σ r_i²` over the last `h` seconds of Binance mid returns.
- Why: Corsi 2009 shows these three timescales jointly forecast vol. On our setup, the `RV_30 / RV_300` *ratio* is a real-time regime indicator: ratio >> 1 means a vol burst is still in progress; ratio << 1 means vol has faded.
- Compute: rolling sum of squared 1s log-returns on `binance_mid` from the parquet. Tick-count: 300 points max per window. Cheap.
- Importance: **HIGH**. GBM uses a single sigma estimate that smooths over everything.
- Source: [Corsi 2009, JFE](https://statmath.wu.ac.at/~hauser/LVs/FinEtricsQF/References/Corsi2009JFinEtrics_LMmodelRealizedVola.pdf); [HAR overview (MDPI 2024)](https://www.mdpi.com/2227-9091/12/1/12)

**F2. Log RV ratio (short/long)**
- Formula: `log(RV_30 / RV_300)`
- Why: centers the ratio around 0, symmetric for up/down vol shocks. XGBoost prefers symmetric features.
- Importance: **HIGH**. Directly encodes "volatility regime shift is happening *right now*".

**F3. Realized jump variation (RJV) via bipower variation**
- Formula: `RV_h − BV_h` where `BV_h = (π/2) * Σ |r_i| * |r_{i-1}|` over the same horizon
- Why: Barndorff-Nielsen & Shephard (2004). The bipower variation is robust to jumps; subtracting it from realized variance isolates the jump contribution. High RJV = the recent move was a discontinuity, not a diffusion — exactly when the GBM Gaussian model is wrong.
- Compute: O(N) on 1s returns. Can be computed on 300s or 600s windows.
- Importance: **HIGH**. Directly detects when the GBM prior is mis-specified.
- Source: [Barndorff-Nielsen & Shephard 2004, JFE](https://academic.oup.com/jfec/article-abstract/2/1/1/960705); [threshold bipower variation](https://www.sciencedirect.com/science/article/abs/pii/S0304407610001600)

**F4. Multi-lookback momentum (10s, 60s, 300s)**
- Formula: `mom_h = binance_mid[t] - binance_mid[t-h]`, normalized by sigma_h
- Why: GBM `mid_up_momentum` is only 60s. The literature (Gu-Kelly-Xiu 2020) consistently finds multi-horizon momentum dominates single-horizon. Dimensionless form (ratio to sigma) generalizes across volatility regimes.
- Importance: **MEDIUM-HIGH**. Cheap, orthogonal to `z`.
- Source: [Gu, Kelly, Xiu 2020, RFS](https://academic.oup.com/rfs/article/33/5/2223/5758276)

**F5. Momentum-reversal transition signal**
- Formula: `sign(mom_10) * (1 - sign(mom_60) * sign(mom_300))` — i.e., when short momentum is against longer momentum, flag a reversal
- Why: simple decision-tree-friendly reversal detector. Handles the "bounce off window peak" case.
- Importance: **MEDIUM**

**F6. Fractional differentiation of Binance mid**
- Formula: `fracdiff(binance_mid, d=0.4)` using the López de Prado weight-loss window
- Why: Lopez de Prado (2018) Ch 5. Standard first-differences destroy memory; fractional differencing keeps memory while restoring stationarity. This gives XGBoost a *stationary* price series with predictive content that raw Δlog(p) throws away.
- Compute: O(N × window). The window can be short (60-300 points) for minute-scale.
- Importance: **MEDIUM**. More interesting as a research curiosity, but Lopez de Prado shows it has real out-of-sample lift.
- Source: [Lopez de Prado 2018 Ch 5](https://www.oreilly.com/library/view/advances-in-financial/9781119482086/c05.xhtml); [Hudson & Thames writeup](https://hudsonthames.org/fractional-differentiation/)

### 2.2 Microstructure (the features OBI *should* have been)

The existing `imbalance5_up/down` fields are raw volume ratios at L5 — exactly what Gould & Bonart (2016) call the "queue imbalance" predictor. It has been shown to predict the next tick in equities, but there is one crucial reason it is useless on Polymarket binary markets: the book is thin and one-sided. The informed agents have already eaten the signal by the time a 1-Hz snapshot captures it. The fix is to use *transformations* of the imbalance, not the raw value.

**F7. Microprice offset**
- Formula: `micro_up = (bb_up * size_ask_up + ba_up * size_bid_up) / (size_bid_up + size_ask_up)` — then `micro_offset_up = micro_up - mid_up`
- Why: Stoikov's (2018) microprice. Empirically a better 3–10s forecaster than mid or weighted mid. The *offset* from mid — not the level — is the predictive thing, because it encodes "which way does the book lean?" in units of spread.
- Compute: one line of arithmetic, all inputs already in the parquet.
- Importance: **HIGH**. This is likely the single biggest free win in the microstructure bucket.
- Source: [Stoikov 2018, SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694); [The Pennsylvania State honors thesis empirical](https://honors.libraries.psu.edu/files/final_submissions/9619)

**F8. Microprice drift (1s, 5s, 30s)**
- Formula: `Δ micro_up over last h seconds`
- Why: Changes in the microprice lead changes in the mid because market makers skew the book *before* they re-quote the best. A falling microprice with a stable mid = imminent mid fall.
- Importance: **HIGH**

**F9. Book convexity / L5 shape**
- Formula: Fit a parabola (or use quadratic-fit coefficient) to `[bid_sz_up_1 .. bid_sz_up_5]` vs `[1..5]`. Use the curvature coefficient as a feature.
- Why: A convex book (big deep orders, tiny top-of-book) signals "iceberg/whale hiding" — large hidden size that will absorb moves. A concave book (big top, thin deep) signals "retail froth" that will blow through quickly. Different regimes predict different outcomes.
- Importance: **MEDIUM**. This is a creative one — I have not seen it published for prediction markets — but shape analysis of LOB is a known HFT technique.

**F10. Top-of-book size asymmetry at tick level**
- Formula: `(size_bid_up_1 - size_ask_up_1) / (size_bid_up_1 + size_ask_up_1)`
- Why: Queue imbalance at the *best* level (Gould & Bonart 2016) is separately predictive from L5 imbalance. Large-tick instruments (Polymarket is large-tick — 1¢ vs a $1 price) have especially strong queue signal.
- Compute: free from parquet.
- Importance: **MEDIUM-HIGH**.
- Source: [Gould & Bonart 2016, Quant Finance](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2702117)

**F11. Cross-book imbalance asymmetry (UP vs DOWN token book)**
- Formula: `imbalance5_up + imbalance5_down` (which equals zero only when both books agree there is no pressure)
- Why: In a binary market, if the UP book says "bullish" AND the DOWN book says "bearish", both books agree the probability is rising. If they *disagree*, the tokens are mispriced relative to each other — a put-call-parity style arb. This is categorically different from the existing `imbalance5_up - imbalance5_down` feature.
- Compute: free.
- Importance: **MEDIUM**. The existing model has the difference, not the sum — so this is literally orthogonal.

**F12. Spread compression ratio**
- Formula: `spread_up[t] / median(spread_up[t-60s:t])`
- Why: Spread tightening often precedes directional moves (market makers are confident they know which way it's going). Spread widening precedes vol bursts. The existing model has the raw spread but not the relative-to-recent.
- Importance: **MEDIUM**

### 2.3 Cancel / quote intensity (requires trade-tape patch)

These are the heaviest-hitting microstructure features in the literature, but none of them are computable from the current 1Hz parquet. They need either a raw order book event stream (additions/cancellations) or at least a finer sampling of mid moves. I include them for the wishlist.

**F13. Quote-to-trade ratio**
- Formula: count of distinct (best_bid, best_ask) states divided by count of trades in the last N seconds
- Why: High quote-to-trade = "quote stuffing" or noise; low = informed trading / every quote is a real price. In HFT this is a toxicity proxy.
- Data dependency: need the raw CLOB event stream, not 1Hz samples. **NOT CURRENTLY POSSIBLE.**
- Importance: **HIGH** if we had the data; skip for now.

**F14. Cancel rate**
- Formula: cancellations per second at top of book
- Why: rising cancel rate = makers pulling before a move. Classical adverse-selection proxy.
- Data: requires CLOB event stream. **NOT POSSIBLE today.**
- Importance: **HIGH** if we had it.

**F15. Binance aggressor trade flow imbalance (TFI)**
- Formula: `Σ signed_volume` over last N seconds, where `signed = +volume` on aggressor-buy, `-volume` on aggressor-sell
- Why: Cont-Kukanov-Stoikov (2014) showed near-linear relationship between TFI and next price change. This is the cleanest microstructure feature there is. The slope of TFI → Δprice is inversely proportional to depth.
- Data: needs Binance trade tape recording. **REQUIRES FEEDS PATCH.** Once patched, this is worth every feature on the list combined.
- Importance: **VERY HIGH** (conditional on data).
- Source: [Cont, Kukanov, Stoikov 2014](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1712822)

### 2.4 Order flow toxicity (VPIN family)

**F16. Binance VPIN**
- Formula: partition the last V units of volume into N equal volume-buckets. For each bucket compute `|buy_vol - sell_vol| / V`. Average.
- Why: Easley, Lopez de Prado, O'Hara (2012). The VPIN literature says VPIN spikes precede volatility events by 1–5 min — exactly the right horizon for us. Crypto-specific paper (Sci Direct 2025) confirms VPIN predicts jumps and has 0.45–0.47 baseline levels in crypto vs 0.22 in E-minis.
- Data: needs Binance trade tape. **REQUIRES FEEDS PATCH.**
- Importance: **HIGH** (conditional on data). Note: VPIN predicts *magnitude*, not direction. Use it to gate confidence, not pick a side.
- Source: [Easley, Lopez de Prado, O'Hara 2012](https://stoye.economics.cornell.edu/docs/Easley_ssrn-4814346.pdf); [SciDirect Bitcoin wild moves 2025](https://www.sciencedirect.com/science/article/pii/S0275531925004192)

**F17. Kyle's lambda (price impact coefficient)**
- Formula: Regress Δ mid on signed trade volume over a rolling window; the slope is λ. Or use the simpler form `λ = |r| / volume`.
- Why: Kyle (1985). Higher λ = less liquid = informed traders dominate = predictable next move.
- Data: needs Binance trade tape. **REQUIRES FEEDS PATCH.**
- Importance: **MEDIUM-HIGH** (conditional).
- Source: [Kyle 1985, Econometrica]; [Algoindex crypto microstructure](https://algoindex.org/)

**F18. Amihud illiquidity (fast form)**
- Formula: `|Δ mid| / (volume in last 60s)`
- Why: Cheaper than Kyle's lambda, approximately the same information. Realized-Amihud variants (2023 ScienceDirect) work at 5–10 minute frequencies.
- Data: requires Binance trade tape or at least per-second volume. **REQUIRES FEEDS PATCH.**
- Importance: **MEDIUM** (conditional).
- Source: [Amihud 2002, JFM](https://www.cis.upenn.edu/~mkearns/finread/amihud.pdf)

**F19. Relative-TFI ratio (Hawkes-lite)**
- Formula: `TFI_recent / TFI_background` where background is a longer-window average
- Why: Hawkes processes (arxiv 2312.16190) show trade arrival is self-exciting. A burst of one-sided flow relative to baseline is a 1–30s predictor.
- Data: **REQUIRES FEEDS PATCH.**
- Importance: **MEDIUM-HIGH** (conditional).
- Source: [Hawkes crypto LOB, arxiv 2312.16190](https://arxiv.org/abs/2312.16190)

### 2.5 Cross-asset signals

**F20. BTC–ETH spread z-score**
- Formula: Normalize `log(ETH_price/BTC_price)` by its recent mean/std; use the z-score.
- Why: When the ETH/BTC ratio is at a short-horizon extreme, BTC moves are amplified (leverage effect). Bi-directional causality in the literature, but the *magnitude* signal is real.
- Data: need ETH feed on the BTC worker, which already exists for the ETH bots. **Cheap integration.**
- Importance: **MEDIUM**.
- Source: [Sifat et al. 2019 IRFA](https://www.sciencedirect.com/science/article/abs/pii/S0275531919300522); [BTC-ETH cross-correlation](https://arxiv.org/pdf/2208.01445)

**F21. BTC–ETH short-horizon correlation**
- Formula: 60s rolling Pearson correlation of 1s returns
- Why: Breakdown of correlation during a move often signals an idiosyncratic event (liquidation, news) — a regime change.
- Data: same as F20.
- Importance: **LOW-MEDIUM**. Honest assessment: the 2019 paper says intraday crypto can *barely* exploit this.

**F22. Binance perp funding rate (current + sign-flips)**
- Formula: current funding rate, plus `z-score of funding` vs 30-day history, plus `sign change in last 8h` flag
- Why: Funding > 0.10% per 8h precedes corrections in 5-15% of cases. Not a 5m signal on its own, but a 5m-signal *multiplier* during high-funding regimes (bias mean reversion).
- Data: Binance public funding-rate endpoint; 8h update frequency. Easy REST pull.
- Importance: **MEDIUM** (as a regime overlay).

**F23. Binance perp basis (perp price − spot)**
- Formula: `(perp_mark - spot_mid) / spot_mid`
- Why: Premium = leveraged-long froth = mean-reversion signal. Tracks faster than the 8h funding rate. Real-time.
- Data: Binance perp WebSocket. **Requires feeds patch** (we currently only run spot).
- Importance: **MEDIUM**.

**F24. Open interest change (Δ OI / OI)**
- Formula: 1m change in open interest divided by OI level
- Why: Rising OI + rising price = new money bullish. Falling OI + rising price = short cover (weaker). Classic futures read.
- Data: Binance perp REST (every 1m). **Requires new worker.**
- Importance: **LOW-MEDIUM** at 5m horizon. Save for later.

**F25. Liquidation stream (forced sell / forced buy flow)**
- Formula: aggregated USD of liquidations per side, last 60s; normalize by rolling median
- Why: Liquidation cascades are the single most predictable volatility source in crypto. When you see a $10M long cascade, the next 30s is directional.
- Data: Binance liquidation stream WebSocket. **Requires feeds patch.**
- Importance: **HIGH** (conditional on data). Arguably worth a dedicated strategy on its own.
- Source: [Gate.com derivatives signals 2026](https://www.gate.com/crypto-wiki/article/how-do-crypto-derivatives-market-signals-predict-price-movements-futures-open-interest-funding-rates-liquidation-data-long-short-ratio-and-options-explained-20260129)

**F26. ES/NQ futures tick during US hours**
- Formula: 60s change in ES mid during 13:30–20:00 UTC; 0 otherwise
- Why: Post-ETF BTC has 3-4x NQ beta during US hours. Real-time NQ tick leads BTC by 1–5s on macro shocks.
- Data: needs an equities feed. **REQUIRES PAID FEED** unless we use a free delayed source. **SKIP** per constraints.
- Importance: **MEDIUM** in theory; N/A in practice.

**F27. USDT/USDC premium at Binance**
- Formula: `(USDT/USDC_mid) - 1.0` — basis point premium
- Why: During risk-off, USDT premium spikes (flight to the more liquid stable). 2023 depeg showed 6% premiums. Less useful in calm markets.
- Data: Binance spot pair (USDT/USDC or USDCUSDT). **Requires feeds patch**, but lightweight.
- Importance: **LOW for 5m**, but nearly free — tail hedge feature.

### 2.6 Time-of-day / calendar

**F28. Minutes-until-nearest-macro-event**
- Formula: Minutes until next CPI/NFP/FOMC release (or 0 if past 5m). Positive ahead, negative after, clamp to ±30m.
- Why: Vol compression pre-event, vol expansion post. Known, well-documented.
- Data: manually maintained calendar file; Fed/BLS release dates. Low effort.
- Importance: **MEDIUM** for the ~10 event days per month; NOOP otherwise.

**F29. Minutes-until-Binance-funding-settlement**
- Formula: minutes until next 00:00/08:00/16:00 UTC funding tick
- Why: MMs sometimes skew quotes before funding settlement to push funding direction.
- Data: trivial.
- Importance: **LOW**. Try it as a probe.

**F30. Fraction of day elapsed, sine/cosine encoded**
- Already have hour-of-day sin/cos, but could add minute-of-hour (30m cycle) and minute-of-10min (for 5m window cadence effects).
- Importance: **LOW**. Skip unless the HPO says it helps.

**F31. Day-of-week one-hot**
- Current feature: `is_weekend` (binary).
- Upgrade: one-hot Mon-Sun or at least Sat/Sun/Mon separated. The 2025 data had Tuesday as most volatile day — one-hot lets the model learn it per day.
- Importance: **LOW-MEDIUM**.

### 2.7 Polymarket-specific microstructure

**F32. Time-since-last-book-update (stale book age)**
- Formula: `now - ts_of_last_book_snapshot_change` in ms
- Why: THIS IS THE EXECUTION FEATURE. The adverse-selection analysis found 79% of losing fills happened during stale-book episodes that the bot's own cancel logic flagged as `edge_gone`. If this feature is in the filter, the filter can learn "trust signal only when book is fresh."
- Data: already tracked by the bot (see `max_book_age_ms` stale-book gate commit `b67c7a7`-era). Just needs to be exposed to the feature extractor.
- Importance: **HIGH**. Biggest free lunch in the whole list.

**F33. Time-since-last-polymarket-trade**
- Formula: `now - ts_of_last_contract_trade` in ms, clipped at 60s
- Why: A book without trades is a book that has no new information. Large gaps = illiquid = possibly wider bids worth chasing, but also possibly stale quotes.
- Data: `last_trade_px_up` already in the parquet — we just need to also record the *timestamp* of the last trade, or compute it offline from trade tape history.
- Importance: **MEDIUM-HIGH**.

**F34. Tick-stability count**
- Formula: `count of consecutive 1s ticks where best_bid_up did not change`
- Why: a market-maker quote that hasn't moved in 10s is a stale quote. If the GBM says the fair P has moved but the book hasn't, there's an arbitrage (subject to latency risk).
- Data: 1-pass computation from the parquet.
- Importance: **MEDIUM-HIGH**. Strongly complementary to F32.

**F35. Polymarket implied probability − GBM implied probability**
- Formula: `mid_up - P_gbm(up)` where P_gbm is the model's fair value
- Why: THIS IS THE CORE LATENCY-ARB SIGNAL, currently used only as a threshold (`min_edge`), not as a feature. A filter that sees this residual can learn "trade only when |residual| > 3¢" *conditional on other features*. Right now the residual is trimmed at the threshold, not fed to the filter. Big miss.
- Data: both values are computed every tick already.
- Importance: **HIGH**. This is #2 in the top 10.

**F36. Residual persistence — is the mispricing stable or fleeting?**
- Formula: `std(residual) over last 10s` or `residual[t] - residual[t-5s]`
- Why: A residual that has been persistent for 10s is different from one that appeared in the last 500ms. The persistent ones tend to be stale-quote edges (our edge). The new ones tend to be "Binance just moved and Polymarket is about to catch up, but so are we against 20 other bots." Persistence distinguishes "slow edge" from "race edge."
- Data: rolling window.
- Importance: **HIGH**. Pairs with F35.

**F37. Largest single-order at top of book**
- Formula: `max(size_bid_up_1) — already there; the raw value, not the imbalance`
- Why: A whale resting at top of book signals institutional interest and also a *floor* under price.
- Importance: **LOW-MEDIUM**.

**F38. Number of distinct depth levels with size > threshold**
- Formula: `count([bid_sz_up_1..5] > 50)`
- Why: A book with 5 meaningful levels is different from a book with 1 meaningful level + 4 dust orders. Shallow book = vulnerable to sweeps.
- Importance: **LOW**.

### 2.8 Window-relative path features

The GBM is memoryless — it only knows `z, sigma, tau` at the current instant. Any function of the *path taken to get here* is information the GBM cannot use.

**F39. Drawdown from window-peak z-score**
- Formula: `max(z over window so far) - z[now]` when z[now] > 0; `z[now] - min(z over window so far)` when z[now] < 0
- Why: "We were at +2σ and now we're at +1σ" is a very different state from "we just reached +1σ for the first time." The former has mean-reversion baked in. A decision-tree model will love this feature.
- Importance: **MEDIUM-HIGH**.

**F40. Signed z-score range in window**
- Formula: `max_z - min_z` in the window so far
- Why: A window that has already spanned from -1σ to +1σ has burned most of its volatility budget — the probability of a big further move is lower than Brownian motion would suggest.
- Importance: **MEDIUM**.

**F41. Number of zero-crossings of (p − window_start_price)**
- Formula: count of times `binance_mid` crossed `window_start_price` since window start
- Why: High crossing count = chop = neither side has conviction = mean-reverting regime. Low crossing count = directional.
- Importance: **MEDIUM**.

**F42. Run length of current direction**
- Formula: number of consecutive seconds with same sign of 1s return
- Why: classic momentum vs exhaustion metric. Short runs (1-2s) are noise; long runs (20s+) are either real trends or exhaustion.
- Importance: **MEDIUM**.

**F43. Elapsed fraction × signed |z|**
- Formula: `(1 - tau/full_window) * sign(z) * |z|`
- Why: Interaction between "how much of the window is gone" and "what direction is the signal." Late-window strong signals behave differently from early-window ones because there is less time for reversion. We already have `|z|*tau` but not this signed form.
- Importance: **MEDIUM-HIGH**. Easy win.

### 2.9 Volatility-of-volatility and higher moments

**F44. Realized skewness (intraday, Amaya et al.)**
- Formula: `Σ r_i³ / (N * (RV)^1.5)` over last 60–300s
- Why: Amaya, Christoffersen et al. (2015) showed intraday realized skewness predicts 1-week returns in equities (19bps/week top-bottom decile). Negative skewness = recent right-tail exhaustion; positive skewness = left-tail exhaustion.
- Data: 1s Binance mid returns, free.
- Importance: **MEDIUM-HIGH**. I'd actually put this in the top 15 if I had more space. Orthogonal to everything.
- Source: [Amaya et al. 2015, JFE](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1898735)

**F45. Realized kurtosis**
- Formula: `Σ r_i⁴ / (N * RV²)` over last 60–300s
- Why: Same Amaya paper. Detects fat-tail regime. Weaker signal than skewness but still has lift.
- Importance: **MEDIUM**.

**F46. Vol-of-vol (rolling std of RV_30)**
- Formula: std of last 10 measurements of `RV_30`
- Why: High vol-of-vol = unstable regime = GBM Gaussian assumption is broken.
- Importance: **MEDIUM**.

**F47. Parkinson / Garman-Klass vol on 1m OHLC of binance_mid**
- Formula: `Parkinson: (1/4*ln 2) * (ln(H/L))²`; `GK: 0.5*(ln H/L)² − (2 ln 2 − 1)*(ln C/O)²`
- Why: The existing sigma is close-to-close based. Range estimators are 5–7x more efficient. On BTC with opening jumps, Yang-Zhang is even better (14x).
- Data: resample 1s binance_mid to 1m OHLC, compute the estimator. Free.
- Importance: **MEDIUM-HIGH**.
- Source: [Portfolio Optimizer range vol overview](https://portfoliooptimizer.io/blog/range-based-volatility-estimators-overview-and-examples-of-usage/)

### 2.10 Information theory / complexity

These are speculative but cheap and a couple of them might work.

**F48. Sample entropy of last 60s of 1s returns**
- Formula: `SampEn(returns, m=2, r=0.2*sigma)` — see Richman-Moorman
- Why: Low sample entropy = predictable pattern = momentum regime. High = noise. Bitcoin literature (sci direct 2024) finds antipersistence at short horizons + persistence at longer.
- Importance: **LOW-MEDIUM**. Honest assessment: these usually fail.
- Source: [Bitcoin sample entropy MDPI 2024](https://www.mdpi.com/2504-3110/9/10/635)

**F49. Short-horizon Hurst exponent (300s window)**
- Formula: Hurst estimated on 1s returns over last 5 min via R/S or DFA
- Why: H > 0.5 = persistent (momentum); H < 0.5 = antipersistent (reversion); H = 0.5 = random walk. BTC tends to H < 0.5 at short scales.
- Importance: **LOW**. Noisy at short windows.
- Source: [BTC Hurst 1-6h](https://www.researchgate.net/figure/Hurst-exponent-for-1-to-6-hours-BTC-returns-using-a-sliding-window-of-500-datapoints_fig1_350042242)

**F50. Permutation entropy (order patterns)**
- Formula: PE(returns, order=3) — count permutations of 3-tuples in the return series
- Why: Bandt-Pompe complexity measure. Rising PE = regime transition. Bitcoin study (2024) found it tracks volatility phases.
- Importance: **LOW**.

### 2.11 Regime-detection features (meta)

**F51. Trend vs chop classifier (choppiness index)**
- Formula: `CI = 100 * log10(Σ TR_1s / (max - min)) / log10(n)`
- Why: classic TA indicator that happens to have real statistical content — high = ranging, low = trending.
- Importance: **LOW-MEDIUM**.

**F52. Regime labeling via rolling sigma quantile**
- Formula: `rank(sigma_now) within last 1 day of sigma values` → a value in [0,1]
- Why: Lets the model say "high vol regime" or "low vol regime" in a discretization-friendly form. XGBoost will interact this with other features automatically.
- Importance: **MEDIUM**. Useful and cheap.

---

## 3. Features to AVOID (and why)

I want to be honest about what has been publicly hyped but is unlikely to work for us, because the user has been burned before.

| Avoid | Why it won't help |
|---|---|
| **Raw OBI at L1-L5** | Already tried; the live backtest showed it has the wrong sign on Polymarket and XGBoost assigned 0% importance. The fix is microprice (F7), not another flavor of OBI. |
| **Session directional bias (Asia revert / US momentum)** | Our own backtest (`external_signals_test_2026-04-05`) showed ALL sessions above 68% continuation — no session-specific bias, and p-values not significant. Don't re-add as a feature. |
| **Consecutive-direction streak (N-up reversion)** | Same file: none of the streak buckets are statistically significant (all CIs contain 48%). The 2-UP → 42% effect was noise. |
| **Mempool / on-chain / whale-wallet signals** | 10–60 min latency. By the time the signal arrives, the 5min window is over. Good for daily context, useless for 5m. |
| **Halving / cycle position features** | Multi-month timescale. Irrelevant. |
| **Stablecoin mint/burn** | Coincident at best, often lagging. Daily signal at best. |
| **Twitter/Reddit sentiment** | Even if wired up, the latency and signal/noise makes it a trap. The literature is full of failed replications. |
| **Google Trends** | Daily. Useless at our horizon. |
| **Twitter volume / NLP embeddings of headlines** | Paid data, low signal, expensive infrastructure. Explicitly out of scope per constraints. |
| **End-of-window snipe position** | Per `feedback_no_end_window_snipe.md`, the user has rejected this. Don't add `is_last_30s_of_window` as a trading feature — it signals the wrong strategy. |
| **Weekend-effect dampening** | Post-Kou-fix the effect collapsed to 2.24× and is inverted on BTC 15m. Don't re-introduce as a feature weight. |
| **Raw hour-of-day as a linear feature** | Already there via sin/cos. Linear form would hurt. |
| **Fear & Greed Index** | Daily. Useless. |
| **SOPR / MVRV / any Glassnode chain metric** | Daily. Useless. |
| **Aggregate spot ETF flow** | Daily. Possibly a regime feature, not a 5m feature. |
| **Polymarket "volume in window so far"** | Already confounded with `tau`; low orthogonality. |
| **Deep-learning LOB features (DeepLOB, CNN/LSTM)** | Overkill; overfits; opaque; Explainable Patterns in Crypto Microstructure (arxiv 2602.00776) explicitly warns that better inputs beat deeper nets. Stick with hand-crafted. |
| **VPIN computed on Polymarket trades** | Polymarket trade tape is sparse and bursty; VPIN's volume-clock assumes dense trades. Compute VPIN on Binance, not Polymarket. |

---

## 4. Implementation Difficulty Estimates

### Tier A — Works TODAY, no code changes beyond feature extractor (<1 day each)

| Feature | Effort | LOC |
|---|---|---|
| F1, F2 (HAR-RV triple + log ratio) | 1h | ~20 |
| F3 (bipower RJV) | 1h | ~15 |
| F4, F5 (multi-lookback momentum + reversal) | 1h | ~15 |
| F7, F8 (microprice + microprice drift) | 30m | ~10 |
| F9, F10, F11 (book shape, top queue imbalance, cross-book sum) | 1h | ~20 |
| F12 (spread compression) | 30m | ~5 |
| F32 (stale book age) | 1h | ~5 (just pipe through the existing `book_age_ms`) |
| F34 (tick-stability count) | 1h | ~10 |
| F35, F36 (P residual + persistence) | 1h | ~10 |
| F39-F43 (window-relative path features) | 2h | ~40 |
| F44, F45 (realized skew/kurt) | 1h | ~15 |
| F46 (vol-of-vol) | 30m | ~5 |
| F47 (Parkinson / Garman-Klass) | 1h | ~15 |
| F52 (sigma regime quantile) | 30m | ~10 |

**Tier A total: ~13 hours, all using data already in `data/btc_5m/*.parquet`.**

### Tier B — Needs ETH-feed wiring on BTC worker or new simple REST calls (<1 day each)

| Feature | Effort |
|---|---|
| F20, F21 (BTC-ETH spread/corr) | 2h (ETH feed subscription in BTC worker) |
| F22 (funding rate) | 1h (REST poll every 1h + cache) |
| F27 (USDT/USDC premium) | 1h |
| F28 (macro event calendar) | 2h (maintain a hard-coded calendar file + parser) |
| F31 (day-of-week one-hot) | 15m |

### Tier C — Needs feeds.py patch to record Binance trades (~1 day engineering)

This is the high-leverage unlock: once you have the aggressor-trade tape, you get F13, F14, F15, F16, F17, F18, F19, F25 all at once.

| Feature | Effort |
|---|---|
| **Patch feeds.py to record Binance aggTrade stream to parquet** | ~1 day |
| Then: F15 (TFI), F16 (VPIN), F17 (Kyle λ), F18 (Amihud), F19 (relative-TFI) | 2h each |
| F25 (liquidation stream) requires forceOrder stream | 4h |
| F23 (Binance perp spot basis) | 4h (perp WebSocket add) |

### Tier D — Requires paid feed or not worth it

| Feature | Why skip |
|---|---|
| F26 (ES/NQ futures) | Paid feed |
| F13, F14 (quote intensity / cancels) | Requires Polymarket raw CLOB event stream; not exposed publicly |

---

## 5. Data Dependencies Matrix

| Data source | Status | Features it enables |
|---|---|---|
| `data/btc_5m/*.parquet` (current 1Hz) | HAVE | F1-F12 vol and microstructure, F32-F47 Polymarket and window features, F44-F47 realized moments, F48-F52 entropy/regime |
| Binance ETH spot WebSocket | HAVE (in ETH worker) | F20, F21 (need to fan-out into BTC worker) |
| Binance funding rate REST | HAVE (easy pull) | F22 |
| Binance perp WebSocket (btcusdt-perp) | MISSING | F23, F24 |
| Binance USDT/USDC pair WebSocket | MISSING | F27 |
| **Binance aggTrade WebSocket (trade tape)** | **MISSING** | **F15, F16, F17, F18, F19 — ALL the high-leverage microstructure features** |
| Binance liquidation stream (forceOrder) | MISSING | F25 |
| Polymarket raw CLOB event stream | NOT AVAILABLE | F13, F14 (skip) |
| US equities feed | NOT AVAILABLE | F26 (skip) |
| Macro release calendar (hard-coded JSON) | MISSING | F28 |

**The single highest-ROI data infra work is patching `feeds.py` to record the Binance `btcusdt@aggTrade` stream.** Once we have that, the entire "order flow toxicity" bucket (F15-F19) becomes available, which is where the literature says the real predictive juice lives.

---

## 6. Recommended Starter Set — 24 Features to Try First

This is the 20–30-feature set I would actually implement and retrain the filter with. Every one of these is Tier A or trivial-Tier-B (implementable in <2 days total), none is speculative-beyond-belief, and all are orthogonal to the existing `z²` / `|z|` / OBI bucket.

**Microstructure (6)**
1. F7 — microprice_offset_up
2. F8 — microprice_drift_5s
3. F10 — top-of-book queue imbalance L1
4. F11 — cross-book imbalance sum (UP + DOWN)
5. F12 — spread compression ratio (spread / 60s median spread)
6. F9 — book curvature coefficient (optional if F7/F8 underwhelm)

**Volatility decomposition (5)**
7. F1 — RV_30
8. F1 — RV_120
9. F2 — log(RV_30/RV_300)
10. F3 — bipower RJV indicator
11. F47 — Parkinson sigma on 1m Binance OHLC

**Realized higher moments (2)**
12. F44 — realized skewness (300s)
13. F46 — vol-of-vol

**Window path features (5)**
14. F39 — drawdown from window-peak z
15. F40 — z-range (max_z − min_z) in window
16. F41 — zero-crossings of (p − window_start)
17. F42 — run length of current direction
18. F43 — elapsed_fraction × sign(z) × |z|

**Polymarket-specific (4) — the execution features**
19. F32 — time since last book update (ms)
20. F34 — tick-stability count
21. F35 — residual: Polymarket P − GBM P
22. F36 — residual persistence (std over 10s)

**Regime (1)**
23. F52 — sigma regime quantile (rank within 24h)

**Cross-asset starter (1)**
24. F20 — BTC-ETH spread z-score

That is 24 new features on top of the current 29, for a total of 53. XGBoost on an 800k-row btc_5m dataset will have no problem with that dimensionality. Even if half of them turn out to have zero importance, the other half addresses every orthogonality gap identified in §0.

### What to check after retraining

1. **Does `z²` importance drop below 15%?** If it does not, the new features are not injecting orthogonal information and you have a data problem, not a feature problem.
2. **Do any of F32, F35, F36 land in the top 5 by importance?** They should — they encode the stale-book / fair-value-residual edge the bot's docs keep pointing at.
3. **Does the log-loss on held-out data actually improve?** If the improvement is <0.005 after retraining on the 24 new features, abandon microstructure and invest the time in the Tier C feeds patch instead — the ceiling on this class of features is being hit.
4. **Test on a held-out period that does NOT overlap with the parquet backfill.** The REST-backfilled windows might have different book-age characteristics than live-captured ones, and features like F32 could leak.
5. **Monitor importance on a weekly rolling retrain.** Microstructure features decay fast as the counterparty MM strategies adapt.

---

## 7. Closing Notes

- **The user's instinct is correct that latency arb is the strategic priority.** Nothing in this brainstorm displaces that — the best features on this list (microprice, stale-book-age, P-residual, residual persistence) are *exactly* the features that make the latency-arb strategy sharper by telling the filter "when is the book stale AND the signal fresh." That's the same strategy, better executed.
- **The biggest expected leverage is the free-lunch pile: F7 (microprice offset) and F35 (P residual) alone should dent the `z²` monoculture.** They are 15 minutes of work each.
- **The biggest infrastructure investment is the Binance trade-tape recorder.** It is worth scheduling as a dedicated task — it unlocks an entire feature bucket.
- **Every feature in this list is a candidate, not a commitment.** Train, check importance, drop the dead weight. Do not add features to the live model without SHAP-level evidence they are learning something real. The current model's `z²` dominance is the canary for "we trained on features that were collinear with the signal itself."
- **Don't skip honest ablation.** If the filter model can't beat a pure "GBM signal alone" baseline after these features, the filter architecture itself may be the problem — possibly the answer is a two-head model: one head predicts directional correctness (what we have), the other predicts execution-quality (a regression on realized vs theoretical fill), and the latter uses F32/F34/F35/F36 exclusively. But that is a future investigation, out of scope for this brainstorm.

---

## Sources

**Papers & books**
- Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized Volatility.* Journal of Financial Econometrics. https://statmath.wu.ac.at/~hauser/LVs/FinEtricsQF/References/Corsi2009JFinEtrics_LMmodelRealizedVola.pdf
- Stoikov, S. (2018). *The Microprice: A High-Frequency Estimator of Future Prices.* SSRN 2970694. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694
- Cont, Kukanov, Stoikov (2014). *The Price Impact of Order Book Events.* JFE. https://arxiv.org/abs/1011.6402
- Easley, Lopez de Prado, O'Hara (2012). *Flow Toxicity and Liquidity in a High-Frequency World.* https://www.stern.nyu.edu/sites/default/files/assets/documents/con_035928.pdf
- Barndorff-Nielsen & Shephard (2004). *Power and Bipower Variation with Stochastic Volatility and Jumps.* JFE. https://academic.oup.com/jfec/article-abstract/2/1/1/960705
- Amaya, Christoffersen, Jacobs, Vasquez (2015). *Does Realized Skewness Predict the Cross-Section of Equity Returns?* JFE. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1898735
- Gould & Bonart (2016). *Queue Imbalance as a One-Tick-Ahead Price Predictor.* https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2702117
- Gu, Kelly, Xiu (2020). *Empirical Asset Pricing via Machine Learning.* RFS.
- López de Prado (2018). *Advances in Financial Machine Learning*, Wiley. Ch 5 (fractional differentiation), Ch 2 (information-driven bars).
- Kyle, A. (1985). *Continuous Auctions and Insider Trading.* Econometrica.
- Amihud, Y. (2002). *Illiquidity and Stock Returns.* JFM. https://www.cis.upenn.edu/~mkearns/finread/amihud.pdf
- Bitcoin wild moves: evidence from order flow toxicity and price jumps (2025). Research in International Business and Finance. https://www.sciencedirect.com/science/article/pii/S0275531925004192
- Sifat, Mohamad (2019). *Lead-Lag Relationship between Bitcoin and Ethereum.* IRFA. https://www.sciencedirect.com/science/article/abs/pii/S0275531919300522
- Stylized Facts of High-Frequency Bitcoin Time Series (MDPI 2024). https://www.mdpi.com/2504-3110/9/10/635
- Hawkes-based cryptocurrency forecasting via LOB data (2024). https://arxiv.org/abs/2312.16190
- Explainable Patterns in Cryptocurrency Microstructure. https://arxiv.org/html/2602.00776v1
- The Anatomy of Polymarket: Evidence from the 2024 Presidential Election (2026). https://arxiv.org/html/2603.03136v1

**Existing internal findings referenced**
- `tasks/findings/external_signals_test_2026-04-05.md` (OBI, session, streak ablation — the "features to avoid" list)
- `tasks/findings/crypto_microstructure_research.md` (prior signals prioritization)
- `tasks/findings/adverse_selection_analysis_2026-04-11.md` (F32/F35/F36 motivation)
- `tasks/findings/bonereaper_deep_analysis_2026-04-11.md` (competitor does microprice-style quoting)
- `tasks/findings/latency_audit_2026-04-11.md` (latency arb is the priority)

**Memory references**
- `project_btc5m_stale_book.md` (prior stale-book gate — lines up with F32)
- `project_btc5m_market_blend.md` (existing market-blend work — F35 builds on this)
- `reference_bonereader.md` (BoneReaper competitor profile)
