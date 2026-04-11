# Cross-asset z-scoring and leading indicators — research report

Date: 2026-04-11
Author: research subagent (cross-asset pass)
Script: `analysis/cross_asset_research.py`
Data: `data/btc_5m/*.parquet`, `data/eth_5m/*.parquet` (880 BTC windows × 1111 ETH windows overlapping ≈ 101h of joint coverage)

## 1. Executive summary — verdicts

| Indicator | Claim | Our data shows | Verdict |
|---|---|---|---|
| ETH 60s lagged return → BTC next move | ETH leads BTC | **No.** Whole-window corr is persistence; remaining-move corr = −0.04 | **SKIP** |
| ETH-BTC divergence (60s) mean-revert | Divergence reverts | **No.** Predicts only the half that already happened; remaining-move has no edge | **SKIP** |
| Large ETH move (|r|>10 bp) → BTC direction | Momentum transfer | **Weak.** 58% vs 45% on n=90, z=1.23 (not significant) | **WEAK** |
| BTC–ETH contemporaneous 1s corr | Both move together | **Yes, strongly.** r=0.36 at 1s, r=0.50 at 5s, r=0.58 at 30s | (not a signal; it's the null) |
| Stablecoin USDT/USDC flows (mint events) | Leading regime | Literature supports macro regime, but on-chain lag and cadence don't fit 5m/15m trades | **SKIP** |
| Perp funding rate (Binance/Deribit) | Sentiment extreme → reversal | Literature says binary-style predictive only at extremes (>0.10%/8h); 99%+ of time uninformative | **SKIP** for directional, **WATCH** as regime flag |
| SPX/ES futures lead BTC during US hours | Beta to equities | Literature: 20-day corr 0.5–0.88 swings, but not shown to lead at intraday; Bitcoin often *lags* equities 5–30m | **WEAK** (regime only) |
| DXY inverse lead | Dollar weakness → BTC up | Literature: real at daily-to-weekly; no evidence at 5m | **SKIP** for signals, **WATCH** as regime |
| Coinbase–Binance spread | Regime / liquidity | HFT has compressed this to noise at 1s | **SKIP** |
| USDT peg deviation | Vol leading indicator | No strong 5m signal in literature; peg moves are daily-scale | **SKIP** |

**Top-level conclusion**: on our data and at our horizons (150–300s BTC move), **no cross-asset indicator produces a tradable edge**. The strongest apparent effects (ETH direction → whole-window UP rate) are driven by BTC persistence, not cross-asset prediction. Once you label on the **remaining** BTC move (which is the only thing we can trade), cross-asset correlation collapses to |r|<0.04.

This is a clean **negative result** and it is consistent with the BoneReaper / latency-arb thesis: the edge is in **execution quality against Chainlink lag**, not in predictive modelling.

---

## 2. BTC–ETH lead-lag — empirical results from our parquet data

### 2.1 Setup

Script: `analysis/cross_asset_research.py`

- Concatenated all BTC 5m and ETH 5m parquets where `file_key` falls in the overlap window `[1775580000, 1775944500]` (≈ 101 hours).
- `binance_mid` ticks only. Forward-filled to a 1s uniform grid.
- BTC: 190,540 raw ticks → 364,505 1s samples after grid projection.
- ETH: 246,947 raw ticks → (same grid).

### 2.2 Log-return correlation vs lag k (positive k = BTC(t+k) vs ETH(t))

| return_window | −60s | −30s | −10s | −5s | −1s | 0s | +1s | +5s | +10s | +30s | +60s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1s | +0.036 | +0.001 | +0.002 | +0.009 | +0.094 | **+0.362** | +0.095 | +0.009 | +0.007 | +0.004 | +0.015 |
| 5s | +0.034 | +0.002 | −0.001 | +0.054 | +0.435 | **+0.500** | +0.440 | +0.065 | +0.008 | +0.022 | +0.027 |
| 15s | +0.040 | +0.014 | +0.201 | +0.393 | +0.535 | **+0.557** | +0.539 | +0.404 | +0.214 | +0.038 | +0.032 |
| 30s | +0.056 | +0.043 | +0.404 | +0.500 | +0.570 | **+0.581** | +0.573 | +0.509 | +0.419 | +0.063 | +0.039 |

### 2.3 Asymmetry test (does ETH lead BTC?)

For each |k|, compute `asymmetry = corr(BTC(t+k), ETH(t)) − corr(BTC(t−k), ETH(t))`. Positive ⇒ ETH leads.

| k | 1s-returns | 5s-returns | 15s-returns |
|---:|---:|---:|---:|
| 1s | +0.002 | +0.006 | +0.003 |
| 2s | −0.000 | +0.011 | +0.006 |
| 3s | +0.012 | +0.013 | +0.009 |
| 5s | −0.001 | +0.011 | +0.011 |
| 10s | +0.005 | +0.009 | +0.012 |
| 20s | +0.002 | +0.001 | +0.008 |
| 30s | +0.004 | +0.020 | +0.023 |

**Findings.**
1. **Peak correlation is contemporaneous (lag 0)** at every return window. There is no time-series lag at which ETH or BTC clearly leads on our data.
2. The `ETH-leads-BTC` side of each symmetric pair has a **tiny** positive asymmetry (~+0.01 at k=3–10s on 5s returns). This is consistent with the broader literature's finding of a small "BTC is a shade faster" or "ETH is a shade faster" depending on the sample, but it is **too small to trade** once you subtract transaction costs and 0.23¢ Poly fees.
3. On 15s-return and 30s-return curves, correlations at lag ±1s differ by <0.005 from the lag-0 peak. This means: if BTC moves 1 bp in 1s, ETH has already moved ≈1 bp in the same second. **The cross-asset feed does not give us "extra time."**

Contrast the literature: Katsiampa et al. (2019) and Watorek et al. (2022) report bi-directional causality at hourly timescales, and a small asymmetry (BTC → ETH slightly stronger) on aggregate-volume correlations. None of these findings are strong enough to deliver an edge inside a 5-minute Polymarket window.

Sources:
- [Lead-Lag relationship between Bitcoin and Ethereum: hourly and daily data (Katsiampa, 2019)](https://www.sciencedirect.com/science/article/abs/pii/S0275531919300522)
- [Multifractal Cross-Correlations of Bitcoin and Ether (Watorek et al., 2022)](https://www.mdpi.com/1999-5903/14/7/215)
- [Market uncertainty and correlation between Bitcoin and Ether (2022)](https://www.sciencedirect.com/science/article/abs/pii/S1544612322004214)

---

## 3. Empirical test — does ETH predict BTC's *remaining* move?

### 3.1 Setup

For each of 855 BTC 5m windows with ETH overlap:
- Feature snapshot at `ctx_ts = window_start_ms + 150s` (mid-window, τ ≈ 150s remaining).
- ETH features: `eth_ret_10s`, `eth_ret_30s`, `eth_ret_60s` — log-return of ETH binance_mid at the snapshot tick minus its value at the snapshot tick − {10,30,60}s.
- BTC same-horizon returns as contemporaneous controls.
- Label A (broken): BTC window-end above window-start (includes the half the feature already knows).
- Label B (correct): sign of `btc_ret_remain = log(end_px / mid_px)` — the part we can actually trade.

### 3.2 Broken-label result (demonstrates the trap)

ETH 60s return at mid-window, label = BTC whole-window UP:

| bin | n | up_rate | eth_ret range |
|---:|---:|---:|---|
| 0 (most negative) | 171 | **29.2%** | −0.38% to −0.038% |
| 1 | 171 | 32.7% | −0.038% to −0.010% |
| 2 | 171 | 32.7% | −0.010% to +0.010% |
| 3 | 171 | 50.3% | +0.010% to +0.040% |
| 4 (most positive) | 171 | **65.5%** | +0.040% to +0.675% |

ETH − BTC divergence, label = BTC whole-window UP:

| bin | n | up_rate | div range |
|---:|---:|---:|---|
| 0 (ETH way under BTC) | 171 | **77.2%** | −0.70% to −0.039% |
| 4 (ETH way over BTC) | 171 | **14.6%** | +0.068% to +0.363% |

This **looks like** a 62-point win-rate spread. It is seductive, and it is exactly the kind of plot that fools people. But:

### 3.3 Correct-label result (what we can actually trade)

ETH 60s return at mid-window, label = BTC **mid→end** UP:

| bin | n | remain_up_rate | eth_ret range | remain mean (bp) |
|---:|---:|---:|---|---:|
| 0 | 171 | 51.5% | −0.38% to −0.038% | +0.9 |
| 1 | 171 | 57.3% | −0.038% to −0.010% | +0.2 |
| 2 | 171 | 45.0% | −0.010% to +0.010% | −0.3 |
| 3 | 171 | 48.5% | +0.010% to +0.040% | +0.6 |
| 4 | 171 | 50.9% | +0.040% to +0.675% | +0.5 |

ETH − BTC divergence, label = BTC mid→end UP:

| bin | n | remain_up_rate |
|---:|---:|---:|
| 0 (ETH way under BTC) | 171 | 46.2% |
| 4 (ETH way over BTC) | 171 | 54.4% |

`eth_ret_10s`, `eth_ret_30s`, `eth_ret_60s`, `div_10s`, `div_30s` — **every single one** of these, binned into 5 quantiles, produces win rates in a narrow 44%–57% band. **No monotonic relationship. No tail effect.**

### 3.4 Correlations (n=855)

```
corr(eth_ret_60s,       btc_ret_remain)     = -0.0390
corr(eth_minus_btc_60s, btc_ret_remain)     = -0.0187
corr(eth_ret_60s,       btc_ret_60s_to_mid) = +0.4892   (contemporaneous sanity check)
```

The contemporaneous ETH–BTC correlation is **+0.49** (consistent with the lead-lag table above), but the forward correlation is **−0.04**. The apparent signal is pure persistence of the already-realized segment.

### 3.5 Tail test — large ETH moves

Filter to `|eth_ret_60s| > 10 bp` (top ~10% of moves):

```
n = 90
ETH up 50  → BTC mid→end up rate = 0.580  (k=29, z=+1.13 vs 0.5 baseline)
ETH dn 40  → BTC mid→end up rate = 0.450  (k=18, z=−0.63)
two-sample z = +1.23  (NOT significant at α=0.05)
```

For comparison, the whole-window labels (persistence-contaminated):
```
ETH up 50  → BTC whole UP rate = 0.900
ETH dn 40  → BTC whole UP rate = 0.300
```

A 60-point spread on whole-window vs a 13-point spread on remaining is the clearest possible demonstration that **the signal lives in the past, not the future**.

Tighter threshold (|eth_ret|>20 bp) collapses to n=12 and shows opposite sign (noise).

### 3.6 BTC own-momentum control

To confirm the problem is not specific to ETH, I ran the same test with BTC's own 60s return as the feature:

```
corr(btc_ret_60s_to_mid, btc_ret_remain) = -0.0108

|btc_ret| > 10 bp subset:
  BTC up, n=79,  mid→end up rate = 0.481
  BTC dn, n=112, mid→end up rate = 0.545
```

BTC's own past momentum does **not** predict its remaining move either. This matches the AUDIT_REPORT and `post_fix_revalidation_2026-04-07.md` finding that the Gaussian-on-realized-vol model's "signal" is essentially noise post-Kou fix. The market is nearly Markovian at 2–5 minute horizons, which is exactly why the user's instinct is right: **predictive modelling is not where the edge is**.

---

## 4. Literature review — other proposed leading indicators

### 4.1 Stablecoin flows (USDT/USDC mint and transfer events)

**Claim**: Tether/Circle mints "front-run" BTC rallies; a USDT print at Tether treasury precedes deposit → buy.

**Literature**:
- BIS WP 1270, 1340 ([BIS 1270 PDF](https://www.bis.org/publ/work1270.pdf), [BIS 1340 PDF](https://www.bis.org/publ/work1340.pdf)): stablecoin inflows measurably impact 3-month T-bill yields by 5–8 bp during bill scarcity, but the transmission channel is days-to-weeks, not minutes.
- IMF WP 2025/141 ([IMF WP](https://www.imf.org/-/media/files/publications/wp/2025/english/wpiea2025141-source-pdf.pdf)): stablecoin flows spillover into FX, but again on daily+ cadence.

**Our timeframe reality**:
- Tether mint events happen roughly 0–3 times per day in $100M–$1B chunks. This is **wrong frequency** for a 5m/15m binary.
- On-chain confirmation lag (Ethereum/Tron) is 10–60s+. By the time you see a USDT mint tx confirmed, BTC has already moved.
- No 2-week stretch of our parquet data contains enough mint events to measure the effect ourselves.

**Verdict: SKIP.** Wrong cadence for 5m trades. Possibly useful as a 24h regime flag ("Tether printed $500M this morning → enter bullish regime"), but regime flags are stale relative to the 1.23s Chainlink lag arb opportunity.

### 4.2 Perp funding rates (Binance, Bybit, Deribit)

**Claim**: Extreme positive funding (overleveraged longs) → reversion down; extreme negative funding → reversion up.

**Literature**:
- [MacroMicro perpetual funding](https://en.macromicro.me/charts/49213/bitcoin-perpetual-futures-funding-rate), [Glassnode](https://studio.glassnode.com/charts/derivatives.FuturesFundingRatePerpetual?a=BTC), [AHJ perpetual pricing paper](https://finance.wharton.upenn.edu/~jermann/AHJ-main-10.pdf).
- Search-synthesis finding: funding rates are **sentiment indicators, not predictive signals**. Directional predictive value appears only at extremes (>0.10% per 8h), and even then the payoff is days later, not within the next 5 minutes.
- Strong trends can sustain extreme funding for weeks without reversal.

**Implementation reality**:
- Funding resets every 8 hours (Binance), every 1 hour (Deribit for BTC perps). **Wrong cadence** again.
- Even if funding is currently extreme, the expected reversal horizon (days) dwarfs our 5m window.

**Verdict: SKIP for directional entry signals.** **WATCH** as a 1-per-hour regime overlay (e.g. scale bet size down when funding is >95th-percentile extreme). Worth maybe 1 line of code in `live_trader.py` if/when we fetch Binance funding.

### 4.3 SPX / ES futures as a BTC leader during US overlap hours

**Claim**: Bitcoin has a ~0.5 beta to S&P 500 and will follow ES futures during the 9:30–16:00 NY session.

**Literature**:
- [CoinDesk Jan 2025 — reemerging correlation](https://www.coindesk.com/markets/2025/01/07/correlation-between-bitcoin-and-u-s-stocks-reemerges): BTC-SPX 20-day rolling correlation oscillates between −0.3 and +0.88 in 2025.
- [CME Group — Why Bitcoin moves with equities](https://www.cmegroup.com/insights/economic-research/2025/why-is-bitcoin-moving-in-tandem-with-equities.html): correlation holds in risk-on regimes but breaks during crypto-specific shocks (ETFs, halvings, FTX-style events).
- Our own internal finding `external_signals_test_2026-04-05.md`: US hours have 1.25× larger moves and a 57% UP rate bias vs 44.6% off-hours. This is **already captured** by the hour-of-day vol prior (`_HOURLY_VOL_MULT` in `backtest_core.py`).

**Lead-lag at intraday**:
- None of the sources identify ES as a systematic *leader* of BTC at the 1–30 second horizon. Both are liquid and both respond to the same macro shocks, so contemporaneous correlation should dominate (same story as BTC/ETH).
- In 2025 Q4, cross-asset correlations collapsed to yearly lows as BTC decoupled.

**Verdict: SKIP for directional signals.** The regime effect (US hours = more trading activity) is already in the hourly vol prior.

### 4.4 DXY (dollar index) as an inverse leader

**Literature**:
- [Altrady — DXY impacts crypto](https://www.altrady.com/crypto-trading/macro-and-global-market-insights/us-dollar-index-dxy-impact-crypto-prices), [OSL HK](https://www.osl.com/hk-en/academy/article/the-us-dollar-index-vs-bitcoin-why-the-inverse-correlation-matters): 90-day rolling BTC-DXY correlation runs −0.5 to −0.8.
- [MDPI wavelet analysis (2025)](https://www.mdpi.com/1911-8074/18/5/259): BTC–DXY coherence **vanishes at horizons shorter than ~180 days**. Cross-asset relationship is a macro/regime story.
- Sharp DXY drops (>1% in a week) generally coincide with BTC rallies, but the lead/lag is days, not minutes.

**Verdict: SKIP.** Wrong timescale for 5m/15m binaries. Not investigating further.

### 4.5 Coinbase–Binance BTC spread (regime indicator)

**Claim**: Coinbase–Binance BTC price gap widens during stress → regime signal.

**Literature**:
- [Shu & Zhang 2023 (Accounting & Finance)](https://onlinelibrary.wiley.com/doi/full/10.1111/acfi.13102): Coinbase BTC price is on average above Binance BTC/USDT; spread widens during volatility.
- [Medium — HFT in crypto](https://medium.com/@laostjen/high-frequency-trading-in-crypto-latency-infrastructure-and-reality-594e994132fd): HFT firms have compressed inter-exchange spreads to single-digit basis points most of the time.

**Our context**:
- We trade Polymarket, which uses Chainlink, which ingests feeds including Coinbase and Binance (and Kraken, Bitstamp, etc.). Chainlink's Data Streams protocol already arbitrages this at the source level.
- Polymarket reflects Chainlink with ~1.23s of constant delay (`feed_latency_2026-04-08.md`). The Coinbase–Binance spread cannot give us an edge that the Chainlink feed itself doesn't already express, because Chainlink IS the median of those exchanges.

**Verdict: SKIP** as a leading signal. Worth noting only if the spread *blows out* (rare stress event) as a risk-off regime flag.

### 4.6 USDT peg deviation (USDT/USD)

**Claim**: When USDT trades below $1.000 on spot, fear is rising → BTC down leading.

- [ScienceDirect — stablecoin stability portfolio paper](https://www.sciencedirect.com/science/article/pii/S1572308925000877): peg deviations persist on weekly timescales during stress events.
- Peg deviations below $0.997 are rare (~monthly) and usually follow, not lead, major events.

**Verdict: SKIP.** Wrong cadence.

---

## 5. Implementation plan — recommendation

### 5.1 The honest answer

**Do not add any cross-asset z-score, ETH-BTC divergence gate, or leading indicator to the production signal pipeline.** The empirical tests on our own parquet data say the expected lift is **~0 win-rate points on the remaining BTC move**.

The `cross_asset_z_lookup` path in `signal_diffusion.py` is already disabled (lines 1369–1380 and 1984–2000) with a comment noting it's not wired in on the btc_5m/btc_15m production path. **Leave it disabled.** Specifically:
- Do not re-enable `cross_asset_min_z` gates.
- Do not pass `cross_asset_z_lookup=build_cross_asset_lookup(...)` from `live_trader.py`.
- Delete the `ETH-as-leader` hypothesis from the roadmap.

### 5.2 One possible exception: ETH-confirmation for directional maker sizing

There is a narrow case where cross-asset info might help that I'm NOT claiming the data proves, but which is worth flagging for future testing if the user insists:

> When the model picks a directional side with |z| ≥ 0.5, require that `sign(eth_ret_30s) == sign(model_direction)` OR abstain. This filters against windows where BTC is moving opposite to ETH (e.g. BTC-specific news shock), which historically is a common adverse-selection regime for makers.

I did not test this specific gate on our data because the 855-window overlap is too small to evaluate a gate that would fire on a subset. **Expected lift: unknown, likely <2 wp.** This is <100 lines of code:
```python
# in live_trader.py loop, pseudo:
eth_now = eth_last_trade_px
eth_30s_ago = eth_price_30s_ago  # needs a ring buffer
eth_ret_30s = log(eth_now / eth_30s_ago)
if signed_edge * eth_ret_30s < 0 and abs(eth_ret_30s) > 5e-5:
    skip_trade()
```
Requires a 30s ETH binance_mid ring buffer on the order of <1 kB. Free data via existing Binance WS subscription (`ethusdt@bookTicker`).

**I do not recommend implementing this** until the core signal quality issues from `comprehensive_live_analysis_2026-04-11.md` and `live_pmodel_divergence_root_cause_2026-04-11.md` are resolved. Adding a marginal-maybe filter on top of a buggy signal will only obscure whether the filter works.

### 5.3 Where the real edge is (pointer, not scope)

The user's existing memory files already say this, and my cross-asset research reinforces it:

- `project_lag_backtest.md`, `feed_latency_2026-04-08.md`: Polymarket reflects Chainlink with ~1.23s lag. That is a **deterministic** arbitrage window every single tick.
- `reference_bonereader.md`: the #2 bot on Polymarket runs latency arb, not prediction.
- `tasks/findings/adverse_selection_analysis_2026-04-11.md`, `bonereaper_deep_analysis_2026-04-11.md`: adverse selection against informed taker flow is the actual problem.

Every hour spent trying to build a better forward predictor is an hour not spent shaving RTDS → decision latency, improving order placement timing vs the ~1.23s reorg window, or defending against informed taker adverse selection. **Cross-asset z-scoring does not reduce adverse selection — it just adds a correlated noise source.**

---

## 6. Limitations and failure modes

1. **Sample size**: only 855 mid-window features (≈ 101h of overlap). A true negative result at n=855 with corr = −0.04 has a 95% CI of roughly [−0.11, +0.03], which is tight enough to say "no large effect" but leaves room for a tiny (<3 wp) effect. Rerun if ETH recording runs ≥2 weeks.
2. **Overlap window is recent, calm**: the 101-hour window falls in late trading of the dataset. If crypto volatility regime changes (e.g. spot ETF launch day, FOMC), ETH–BTC lead-lag can briefly spike. I tested average conditions, not tail regimes.
3. **Mid-window snapshot**: I sampled at τ=150s for reproducibility. The live bot decides at various τ depending on fills. A τ-conditional version of this analysis might uncover a small effect at τ close to window start (where ~240s of BTC move is still ahead and persistence matters less); I did NOT test this.
4. **1-second grid**: for the lead-lag study I forward-filled to 1s. If ETH and BTC have sub-second lead-lag of ±200ms, my grid would miss it. A tick-accurate cross-correlation requires per-venue event-time snapshots we don't currently record consistently. Binance WS can deliver this in ~50ms; the operator's home connection introduces ≈50ms jitter in the recording, which would obliterate a sub-second lead anyway. Conclusion: even if it existed, we couldn't trade it.
5. **Binance-only**: we only have `binance_mid`. ETH on Coinbase or Kraken might have different lead-lag. However, Chainlink aggregates all of them so this doesn't matter for Polymarket pricing.
6. **Persistence is the confound**: the single most important lesson from this report. Any cross-asset feature built at mid-window that ignores what BTC has already done will mis-attribute BTC persistence to the cross-asset predictor. Any future researcher must label on `btc_ret_remain`, not `whole_window_up`.
7. **I did not test cross-asset as a VOL (not direction) input**: the existing `cross_asset_z_lookup` uses ETH z as a *disagreement veto*, not a direction predictor. A "when ETH vol is high, BTC vol will be higher next 150s" hypothesis is a different experiment and might merit its own research pass. Given that `sigma_estimation_research_2026-04-11.md` already looked hard at vol estimation and ended on EWMA, I doubt this is where to dig next.

---

## 7. Files and artefacts

- `analysis/cross_asset_research.py` — the analysis script (run-once, no side effects on production).
- `analysis/outputs/cross_asset_features.parquet` — 855 rows × {window_start_ms, mid_ts, eth_ret_{10,30,60}s, btc_ret_{10,30,60}s, btc_ret_remain, label_up}. Reusable for other experiments.
- `tasks/findings/external_signals_test_2026-04-05.md` — predecessor finding (session / OBI / consecutive direction). Same conclusion: most "signals" do not survive honest labelling.

## 8. Sources cited

- [Katsiampa, 2019 — BTC/ETH lead-lag hourly/daily](https://www.sciencedirect.com/science/article/abs/pii/S0275531919300522)
- [Watorek et al., 2022 — Multifractal cross-correlations of BTC/ETH](https://www.mdpi.com/1999-5903/14/7/215)
- [Market uncertainty and BTC-ETH correlation (2022)](https://www.sciencedirect.com/science/article/abs/pii/S1544612322004214)
- [BIS WP 1270 — Stablecoins and safe asset prices](https://www.bis.org/publ/work1270.pdf)
- [BIS WP 1340 — Stablecoin flows and FX spillovers](https://www.bis.org/publ/work1340.pdf)
- [IMF WP 2025/141 — Estimating international stablecoin flows](https://www.imf.org/-/media/files/publications/wp/2025/english/wpiea2025141-source-pdf.pdf)
- [AHJ — Perpetual futures pricing (Wharton WP)](https://finance.wharton.upenn.edu/~jermann/AHJ-main-10.pdf)
- [Glassnode BTC perpetual funding rates](https://studio.glassnode.com/charts/derivatives.FuturesFundingRatePerpetual?a=BTC)
- [CoinDesk Jan 2025 — BTC-SPX correlation reemerges](https://www.coindesk.com/markets/2025/01/07/correlation-between-bitcoin-and-u-s-stocks-reemerges)
- [CME Group — Why Bitcoin moves with equities (2025)](https://www.cmegroup.com/insights/economic-research/2025/why-is-bitcoin-moving-in-tandem-with-equities.html)
- [Altrady — DXY impact on crypto](https://www.altrady.com/crypto-trading/macro-and-global-market-insights/us-dollar-index-dxy-impact-crypto-prices)
- [OSL — DXY vs Bitcoin inverse correlation](https://www.osl.com/hk-en/academy/article/the-us-dollar-index-vs-bitcoin-why-the-inverse-correlation-matters)
- [MDPI wavelet analysis — Bitcoin vs USD](https://www.mdpi.com/1911-8074/18/5/259)
- [Shu & Zhang 2023 — Arbitrage across BTC exchange venues](https://onlinelibrary.wiley.com/doi/full/10.1111/acfi.13102)
- [Keller — HFT in crypto: latency, infrastructure, reality](https://medium.com/@laostjen/high-frequency-trading-in-crypto-latency-infrastructure-and-reality-594e994132fd)
- [CoinAPI.io — Latency in crypto trading](https://www.coinapi.io/blog/crypto-trading-latency-guide)
