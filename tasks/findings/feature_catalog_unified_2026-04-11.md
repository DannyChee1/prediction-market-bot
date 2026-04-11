# Unified Feature Catalog — 2026-04-11

Master list combining the user-facing brainstorm and the research-agent
brainstorm. **No truncation** — every feature from both sources is here.
Truncation will happen empirically after XGBoost training reports
importances.

**Sources:**
- 🤖 = research agent (`feature_engineering_brainstorm_2026-04-11.md`)
- 👤 = main session brainstorm
- ⭐ = both / consensus pick
- 📚 = academic paper citation available

**Tiers:**
- **A**: Computable today from `data/btc_5m/*.parquet`. Pure Python.
- **B**: Needs ETH feed wired into BTC worker, or REST poll (1h cadence).
- **C**: Needs the `feeds.py` patch to record Binance `aggTrade` stream.
- **D**: Needs new feed (Binance perp WS, liquidation stream, USDT/USDC pair).
- **E**: Skip — paid feeds, daily cadence, or empirically dead.

**Status of existing 29-feature model** (from audit):
- Top 2 features (z² + |z|) account for 43% of importance
- All 4 OBI features have 0% importance (XGBoost found NO predictive value)
- Model is mostly a thin wrapper on GBM inputs — needs orthogonal information

---

## Category 1: Multi-Timescale Momentum & Realized Vol (HAR-RV family)

The Corsi 2009 HAR-RV insight: vol at multiple horizons is heterogeneous
and the *ratios* between horizons are themselves regime indicators. The
GBM signal uses a single sigma; we should give the filter ALL the
horizons.

| ID | Feature | Formula | Tier | Source |
|---|---|---|---|---|
| **U1** | `bn_ret_1s` | `log(p_now / p_t-1s)` | A | 👤 |
| **U2** | `bn_ret_5s` | `log(p_now / p_t-5s)` | A | ⭐ |
| **U3** | `bn_ret_10s` | `log(p_now / p_t-10s)` | A | ⭐ |
| **U4** | `bn_ret_30s` | `log(p_now / p_t-30s)` | A | ⭐ |
| **U5** | `bn_ret_60s` | `log(p_now / p_t-60s)` | A | ⭐ |
| **U6** | `bn_ret_120s` | `log(p_now / p_t-120s)` | A | 👤 |
| **U7** | `bn_ret_300s` | `log(p_now / p_t-300s)` | A | ⭐ |
| **U8** | `bn_accel` | `bn_ret_5s − bn_ret_10s` (acceleration) | A | 👤 |
| **U9** | `bn_run_length` | consecutive ticks in same direction | A | 👤 |
| **U10** | `RV_30s` | `Σ r_i² over last 30s of 1s log returns` | A | 🤖 📚 Corsi 2009 |
| **U11** | `RV_120s` | same, 120s window | A | 🤖 📚 |
| **U12** | `RV_300s` | same, 300s window — full HAR-RV triple | A | 🤖 📚 |
| **U13** | `log_RV_ratio_short` | `log(RV_30 / RV_300)` — vol regime change | A | 🤖 📚 |
| **U14** | `log_RV_ratio_long` | `log(RV_300 / RV_1800)` — long-term drift | A | 👤 |
| **U15** | `bipower_variation_300s` | `(π/2) × Σ |r_i|×|r_{i-1}|` — continuous component | A | 🤖 📚 Barndorff-Nielsen 2004 |
| **U16** | `realized_jump_var_300s` | `RV_300s − BV_300s` — pure jump variance | A | 🤖 📚 |
| **U17** | `jump_indicator` | `1 if RJV/RV > 0.3 else 0` — discrete jump regime flag | A | 🤖 📚 |
| **U18** | `mom_zscore_30s` | `bn_ret_30s / sigma_30s` — regime-normalized momentum | A | 🤖 |
| **U19** | `mom_zscore_60s` | `bn_ret_60s / sigma_60s` | A | 🤖 |
| **U20** | `mom_zscore_300s` | `bn_ret_300s / sigma_300s` | A | 🤖 |
| **U21** | `mom_reversal_signal` | `sign(mom_10) × (1 − sign(mom_60)×sign(mom_300))` | A | 🤖 |
| **U22** | `frac_diff_d04` | Lopez de Prado fracdiff(d=0.4) of binance_mid | A | 🤖 📚 Lopez de Prado 2018 Ch 5 |

**Total: 22 features.** Tier A entirely. The HAR-RV triple alone is
likely worth 5+ percentage points of XGBoost importance based on the
literature.

---

## Category 2: Microstructure (the OBI replacement)

The current 4 OBI features have 0% importance. Stoikov's microprice and
Gould-Bonart queue imbalance use the SAME inputs but in non-linear
forms that the literature validated.

| ID | Feature | Formula | Tier | Source |
|---|---|---|---|---|
| **U23** | `microprice_up` | `(bb×ask_sz + ba×bid_sz) / (bid_sz + ask_sz)` for UP token | A | 🤖 📚 Stoikov 2018 |
| **U24** | `microprice_down` | same for DOWN token | A | 🤖 |
| **U25** | `microprice_offset_up` | `microprice_up − mid_up` | A | 🤖 📚 ⭐ |
| **U26** | `microprice_offset_down` | `microprice_down − mid_down` | A | 🤖 |
| **U27** | `microprice_drift_5s` | `Δ microprice_up over last 5s` | A | 🤖 |
| **U28** | `microprice_drift_30s` | `Δ microprice_up over last 30s` | A | 🤖 |
| **U29** | `top_queue_imbalance_up` | `(bid_sz_1 − ask_sz_1) / (bid_sz_1 + ask_sz_1)` for UP | A | 🤖 📚 Gould-Bonart 2016 |
| **U30** | `top_queue_imbalance_down` | same for DOWN | A | 🤖 |
| **U31** | `book_pressure_up` | `log(total_bid_size_5 / total_ask_size_5)` for UP | A | 👤 |
| **U32** | `book_pressure_down` | same for DOWN | A | 👤 |
| **U33** | `cross_book_imbalance_sum` | `imbalance5_up + imbalance5_down` (consensus) | A | 🤖 |
| **U34** | `cross_book_imbalance_diff` | `imbalance5_up − imbalance5_down` (already have, kept for ablation) | A | 👤 |
| **U35** | `book_convexity_up` | quadratic-fit coefficient on `[bid_sz_up_1..5]` | A | 🤖 |
| **U36** | `book_convexity_down` | same for DOWN | A | 🤖 |
| **U37** | `spread_compression_ratio` | `current_spread / median(spread, last 60s)` | A | 🤖 |
| **U38** | `spread_vs_typical` | `spread_now / median_spread_session` | A | 👤 |
| **U39** | `quote_stability_30s` | fraction of last 30s with same best bid/ask | A | 👤 |
| **U40** | `largest_order_ratio` | `max_size_in_book / median_size` | A | 👤 |
| **U41** | `book_depth_total` | sum of sizes top 5 levels both sides | A | 👤 |
| **U42** | `effective_spread` | `2 × |trade_price − mid|` (only when trades present) | A | 🤖 |

**Total: 20 features.** Tier A entirely (all from existing parquet
columns). U25 (microprice offset) is the highest-priority single
addition — Stoikov 2018 showed it beats mid as a 3-10s forecaster.

---

## Category 3: Polymarket-Specific Microstructure

The execution-quality bucket. THESE are the features the bot's docs
keep pointing at but the model never sees.

| ID | Feature | Formula | Tier | Source |
|---|---|---|---|---|
| **U43** | `pm_book_age_ms` | `now − ts_of_last_book_change` | A | 🤖 ⭐ |
| **U44** | `pm_trades_count_60s` | # of contract trades in last 60s | A | 👤 |
| **U45** | `pm_time_since_last_trade` | `now − ts_of_last_contract_trade` | A | 🤖 |
| **U46** | `pm_tick_stability_count` | consecutive 1s ticks where best_bid_up unchanged | A | 🤖 |
| **U47** | `pm_unique_makers_in_book` | distinct LP count from top 5 levels (heuristic from sizes) | A | 👤 |
| **U48** | `pm_largest_ask_size` | max size in any UP ask level | A | 👤 |
| **U49** | `pm_p_residual` | `mid_up − P_gbm(up)` — Polymarket vs GBM disagreement | A | 🤖 ⭐ |
| **U50** | `pm_p_residual_persistence` | `std(p_residual) over last 10s` | A | 🤖 |
| **U51** | `pm_p_residual_drift` | `p_residual[t] − p_residual[t-5s]` | A | 🤖 |
| **U52** | `pm_n_levels_meaningful` | count of levels with size > 50 | A | 🤖 |
| **U53** | `pm_recent_fills_pnl_proxy` | sign of last 5 contract trades direction | A | 👤 |

**Total: 11 features.** Tier A. The combination of U43 (book age) and
U49/U50 (P-residual + persistence) directly encodes the latency-arb
edge as filter inputs.

---

## Category 4: Window-Relative Path Features

The GBM signal is memoryless. Any function of the path TAKEN to get to
the current state is information GBM cannot use. Decision trees love
these.

| ID | Feature | Formula | Tier | Source |
|---|---|---|---|---|
| **U54** | `tau_frac` | `tau / window_duration` | A | 👤 |
| **U55** | `delta_per_typical_move` | `|delta| / (sigma × √elapsed)` — z-score normalized to expected | A | 👤 |
| **U56** | `peak_delta_seen` | max abs(delta) in this window so far | A | 👤 |
| **U57** | `dd_from_peak_z` | drawdown from window peak z-score | A | 🤖 ⭐ |
| **U58** | `signed_z_range` | `max_z − min_z` in window | A | 🤖 |
| **U59** | `n_zero_crossings` | # times delta crossed 0 in this window | A | 🤖 |
| **U60** | `run_length_current` | consecutive seconds with same return sign | A | 🤖 |
| **U61** | `elapsed_frac_x_signed_z` | `(1 − tau/full) × sign(z) × |z|` — interaction | A | 🤖 |
| **U62** | `n_trades_fired_already` | window_trade_count (already in ctx) | A | 👤 |
| **U63** | `n_zero_crossings_normalized` | crossings / sqrt(elapsed_seconds) — choppiness | A | 👤 |

**Total: 10 features.** Tier A.

---

## Category 5: Realized Higher Moments

Amaya et al (2015) showed realized skewness predicts cross-section
returns. We can compute it cheaply on Binance 1s returns.

| ID | Feature | Formula | Tier | Source |
|---|---|---|---|---|
| **U64** | `realized_skew_60s` | `Σr³ / (N × RV_60^1.5)` over last 60s | A | 🤖 📚 Amaya 2015 |
| **U65** | `realized_skew_300s` | same, 300s window | A | 🤖 📚 ⭐ |
| **U66** | `realized_kurt_60s` | `Σr⁴ / (N × RV_60²)` over last 60s | A | 🤖 📚 |
| **U67** | `realized_kurt_300s` | same, 300s window | A | 🤖 📚 |
| **U68** | `vol_of_vol_30s` | std of last 10 measurements of RV_30 | A | 🤖 |
| **U69** | `parkinson_sigma_1m_60` | Parkinson estimator on 1m OHLC over 60min | A | 🤖 📚 |
| **U70** | `garman_klass_sigma_1m_60` | GK estimator on 1m OHLC over 60min | A | 🤖 📚 |
| **U71** | `yang_zhang_sigma_alt` | YZ estimator (alternative to current implementation) | A | 👤 |

**Total: 8 features.** Tier A.

---

## Category 6: Cross-Asset Signals

Even if BTC-ETH correlation is contemporaneous (per the cross-asset
research agent), the cross-asset features may serve as INTERACTION
inputs for the tree model — it can find conditional signal that
isolated correlation can't.

| ID | Feature | Formula | Tier | Source |
|---|---|---|---|---|
| **U72** | `eth_ret_60s` | log return of ETH over 60s | B | ⭐ |
| **U73** | `eth_ret_300s` | log return of ETH over 300s | B | 👤 |
| **U74** | `btc_eth_spread_zscore` | `z(log(ETH/BTC), 60s window)` | B | 🤖 📚 |
| **U75** | `btc_eth_corr_60s` | rolling 60s Pearson corr of 1s returns | B | 🤖 |
| **U76** | `eth_btc_ratio_change` | `Δ(ETH/BTC) over 60s` | B | 👤 |
| **U77** | `binance_funding_rate` | current funding rate (REST 1h cache) | B | 🤖 |
| **U78** | `funding_zscore_30d` | z-score of funding vs 30-day history | B | 🤖 |
| **U79** | `funding_sign_change_8h` | flag for sign flip in last 8h | B | 🤖 |
| **U80** | `binance_perp_basis` | `(perp_mark − spot_mid) / spot_mid` | D | 🤖 |
| **U81** | `usdt_usdc_premium` | `BTC/USDT − BTC/USDC, in bp` | D | 🤖 |
| **U82** | `open_interest_change_1m` | `Δ OI / OI` | D | 🤖 |
| **U83** | `liquidations_60s` | aggregated USD of liquidations per side, last 60s, normalized | D | 🤖 ⭐ |

**Total: 12 features.** U72-U79 are Tier B (1-2 days work). U80-U83
are Tier D (need new WS feeds, postpone).

---

## Category 7: Time-of-Day, Calendar, Macro Events

| ID | Feature | Formula | Tier | Source |
|---|---|---|---|---|
| **U84** | `hour_sin` | `sin(2π × hour/24)` (already have, kept) | A | both |
| **U85** | `hour_cos` | `cos(2π × hour/24)` (already have, kept) | A | both |
| **U86** | `dow_sin` | `sin(2π × dow/7)` | A | 👤 |
| **U87** | `dow_cos` | `cos(2π × dow/7)` | A | 👤 |
| **U88** | `is_weekend` | already have, kept | A | both |
| **U89** | `dow_one_hot_sat`, `..._sun`, `..._mon` | one-hot for Sat/Sun/Mon | A | 🤖 |
| **U90** | `hours_since_us_open` | minutes from 13:30 UTC, signed | A | 👤 |
| **U91** | `hours_since_asia_open` | minutes from 00:00 UTC | A | 👤 |
| **U92** | `is_us_session` | 1 if 13:30-21:00 UTC | A | 👤 |
| **U93** | `is_eu_session` | 1 if 08:00-13:30 UTC | A | 👤 |
| **U94** | `minutes_to_next_macro` | minutes until next CPI/NFP/FOMC release | B | 🤖 |
| **U95** | `is_macro_day` | 1 if today has CPI/NFP/FOMC | B | 🤖 |
| **U96** | `minutes_to_funding_settle` | minutes until next 00/08/16 UTC | A | 🤖 |
| **U97** | `is_options_expiry` | 1 if Friday 08:00 UTC (Deribit) | A | 👤 |
| **U98** | `minute_of_hour` | `now.minute % 60` — top-of-hour effect | A | 🤖 |

**Total: 15 features.** Most Tier A. U94/U95 need a calendar file (1
hour effort).

---

## Category 8: Order Flow Toxicity (REQUIRES feeds.py patch)

These are the highest-leverage features in the academic literature, but
ALL require recording Binance `aggTrade` (the trade tape with aggressor
side). The infrastructure work is ~1 day.

| ID | Feature | Formula | Tier | Source |
|---|---|---|---|---|
| **U99** | `tfi_5s` | `Σ signed_volume over 5s` (Cont-Kukanov-Stoikov) | C | ⭐ 📚 |
| **U100** | `tfi_30s` | same, 30s | C | ⭐ 📚 |
| **U101** | `tfi_300s` | same, 300s | C | 🤖 📚 |
| **U102** | `relative_tfi_ratio` | `tfi_recent / tfi_background` (Hawkes-lite) | C | 🤖 📚 |
| **U103** | `vpin_recent` | volume-bucketed toxicity | C | 🤖 📚 Easley 2012 |
| **U104** | `kyle_lambda_30s` | `|Δ price| / volume` over 30s | C | 🤖 📚 Kyle 1985 |
| **U105** | `amihud_illiq_60s` | `|Δ mid| / volume_60s` | C | 🤖 📚 Amihud 2002 |
| **U106** | `aggression_ratio_30s` | fraction of trades that crossed the spread | C | 👤 |
| **U107** | `trade_size_p90_30s` | p90 of trade sizes in last 30s | C | 👤 |
| **U108** | `trade_count_30s` | # trades in last 30s | C | 👤 |
| **U109** | `median_trade_size_60s` | median size in last 60s | C | 👤 |
| **U110** | `quote_to_trade_ratio` | distinct (bid,ask) pairs / # trades | C | 🤖 |

**Total: 12 features.** All Tier C. **The single highest-leverage data
infra unlock.** The OFI agent estimated Silantyev 2019 found OFI R²
of 40.5% at 10s horizons on BitMEX — these features are where the
literature says the predictive juice lives.

---

## Category 9: Information Theory & Complexity

Speculative but cheap. These usually fail in practice but a couple
might survive.

| ID | Feature | Formula | Tier | Source |
|---|---|---|---|---|
| **U111** | `sample_entropy_60s` | SampEn(returns, m=2, r=0.2σ) — Richman-Moorman | A | 🤖 📚 |
| **U112** | `permutation_entropy_60s` | Bandt-Pompe PE(returns, order=3) | A | 🤖 📚 |
| **U113** | `hurst_exponent_300s` | H via R/S or DFA on 1s returns | A | 🤖 📚 |
| **U114** | `path_efficiency_60s` | `|p_end − p_start| / sum(|p_i+1 − p_i|)` | A | 👤 |
| **U115** | `lyapunov_proxy` | sensitivity to initial conditions on recent prices | A | 👤 |

**Total: 5 features.** Tier A but speculative.

---

## Category 10: Regime Classification

| ID | Feature | Formula | Tier | Source |
|---|---|---|---|---|
| **U116** | `choppiness_index_30s` | `100 × log10(Σ TR / (max−min)) / log10(N)` (Bill Dreiss) | A | ⭐ |
| **U117** | `sigma_regime_quantile_24h` | rank of current σ within last 24h of σ values, [0,1] | A | 🤖 ⭐ |
| **U118** | `vol_regime_label` | discretized: 0=quiet, 1=normal, 2=spike | A | 👤 |
| **U119** | `is_trending` | 1 if `path_efficiency_60s > 0.7` | A | 👤 |
| **U120** | `is_calm_market` | 1 if `RV_300s < quantile_25` | A | 👤 |

**Total: 5 features.** Tier A.

---

## Category 11: Existing Features (KEEP, for ablation reference)

These are the current 29 features. Keep them all so XGBoost can learn
their relative importance vs the new ones.

| ID | Feature | From | Notes |
|---|---|---|---|
| E1-E29 | (existing 29) | filtration_model.py | See `extract_features` |

---

## Master Counts

| Category | Count | Tier-A count |
|---|---|---|
| 1. Multi-timescale momentum & RV | 22 | 22 |
| 2. Microstructure (OBI replacement) | 20 | 20 |
| 3. Polymarket-specific | 11 | 11 |
| 4. Window-relative path | 10 | 10 |
| 5. Realized higher moments | 8 | 8 |
| 6. Cross-asset | 12 | 8 (B) + 4 (D) |
| 7. Time/calendar | 15 | 13 (A) + 2 (B) |
| 8. Order flow toxicity (TFI/VPIN) | 12 | 12 (C, requires patch) |
| 9. Information theory | 5 | 5 |
| 10. Regime classification | 5 | 5 |
| 11. Existing kept | 29 | 29 |
| **TOTAL NEW** | **120** | **101 immediately, 12 after feeds patch, 4 deferred** |
| **GRAND TOTAL** | **149** (existing 29 + 120 new) |

---

## Recommended Implementation Order

The user said "incorporate everything, truncate later." So we ship a
big feature module and let XGBoost decide what's important. But
implementation order still matters because some features unlock others.

### Phase 1: Pure-parquet features (Tier A, ~2 days work)
**Implement all 101 Tier-A features** in `features.py` as pure
functions of:
- Snapshot (current state)
- Rolling history buffer (price, ts)
- Window context (start price, tau)
- Optional: longer-history rolling buffer for 1d quantile features

These split into ~6 sub-modules:
- `features_momentum.py` — Categories 1, 4
- `features_microstructure.py` — Categories 2, 3
- `features_volatility.py` — Categories 1 (HAR-RV), 5
- `features_calendar.py` — Category 7 (no calendar JSON yet)
- `features_information.py` — Categories 9, 10
- `features_extract.py` — main entry point that calls all of the above

### Phase 2: Cross-asset features (Tier B, ~1 day)
- Wire ETH bookticker into BTC worker (lightweight WS subscription)
- Add funding rate REST poller (1h cache)
- Add macro event calendar JSON
- Compute U72-U79 + U94/U95

### Phase 3: Trade tape features (Tier C, ~1.5 days)
- **Patch `feeds.py` to record `btcusdt@aggTrade`** (the unlock)
- Wait 24-48h for fresh data to accumulate
- Compute U99-U110
- These are the highest-leverage features per the literature

### Phase 4: New feed features (Tier D, optional)
- Binance perp WS for U80
- Binance USDT/USDC pair for U81
- Open interest REST for U82
- Liquidation stream for U83

### Phase 5: Train + ablate
1. Train XGBoost v2 on existing parquets with all Phase 1 features
2. Inspect importance — drop features below 0.5% importance
3. Add Phase 2 features, retrain, repeat
4. Add Phase 3 features after data accumulates, retrain, repeat
5. Compare to baseline filtration_model.pkl on the same out-of-sample windows

### Phase 6: A/B in live
- `--filtration --label fm_v2 --no-record` for 24-48h
- Compare to a baseline run via `--label fm_baseline`
- Ship if WR lift > 2pp and realized edge lift > 1pp

---

## Validation Checks (After Retraining)

Per the research agent's recommendation:

1. **Does `z²` importance drop below 15%?** If not, the new features
   aren't injecting orthogonal information — investigate why.
2. **Do any of U43, U49, U50 (PM book age, residual, residual
   persistence) land in the top 5?** They should — they encode the
   latency-arb edge as filter inputs.
3. **Does log-loss on held-out data improve by > 0.005?** If not,
   investing in more features is hitting diminishing returns; shift
   focus to the feeds patch (Tier C).
4. **Test on a held-out period that does NOT overlap with REST-backfill
   windows.** Different book-age characteristics could cause feature
   leakage.
5. **Monitor importance on weekly rolling retrain.** Microstructure
   features decay fast as counterparty MM strategies adapt.

---

## Features To Skip Entirely

From the research agent's "avoid" list, expanded:

| Skip | Reason |
|---|---|
| Raw OBI variants | 0% importance in current model; use microprice instead |
| Session continuation bias | All sessions >68%, no significant difference |
| Consecutive direction streaks | Not significant in our backtest |
| Mempool / on-chain / whale | 10-60min latency, useless at 5m |
| Halving / cycle position | Multi-month timescale |
| Stablecoin mint/burn | Coincident or lagging |
| Twitter/Reddit sentiment | Latency + signal/noise is a trap |
| Google Trends | Daily |
| Fear & Greed | Daily |
| SOPR / MVRV / Glassnode | Daily |
| ETF flow daily | Regime-only, not 5m |
| End-of-window snipe features | User explicit reject |
| Weekend dampening | Post-Kou-fix collapsed |
| DeepLOB / CNN-LSTM | Overfits, opaque, better inputs beat deeper nets |
| ES/NQ futures during US hours | Paid feed required |
| Polymarket trade VPIN | Trade tape too sparse for VPIN's volume clock |
| Linear hour-of-day | Use sin/cos instead |
| Halving cycle | Multi-year |

---

## Top 15 Highest-Priority Picks (Consensus)

If I had to ship just 15 features tomorrow:

1. **U25** `microprice_offset_up` (Stoikov 2018, fixes the OBI hole)
2. **U49** `pm_p_residual` (latency-arb edge as a feature, not just a threshold)
3. **U43** `pm_book_age_ms` (encodes 79% of losing-fill cause)
4. **U13** `log_RV_ratio_short` (regime change detector)
5. **U17** `jump_indicator` (detects when GBM Gaussian breaks down)
6. **U15** `bipower_variation_300s` (continuous component)
7. **U27** `microprice_drift_5s` (leading indicator of mid moves)
8. **U50** `pm_p_residual_persistence` (distinguishes slow from race edges)
9. **U46** `pm_tick_stability_count` (stale market-maker quote detector)
10. **U57** `dd_from_peak_z` (path-aware mean reversion)
11. **U65** `realized_skew_300s` (Amaya 2015, leading indicator of cross-section)
12. **U18** `mom_zscore_30s` (regime-normalized momentum)
13. **U37** `spread_compression_ratio` (MM confidence proxy)
14. **U117** `sigma_regime_quantile_24h` (regime label for tree splits)
15. **U72** `eth_ret_60s` (cross-asset, even if weak alone)

These are the consensus picks where my brainstorm and the agent's
agreed, OR where the citation strength is strongest. **Implement these
first; the other 86 Tier-A features can land in subsequent commits.**

---

## Sources (Combined)

### Papers
- Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility. JFEC. <https://statmath.wu.ac.at/~hauser/LVs/FinEtricsQF/References/Corsi2009JFinEtrics_LMmodelRealizedVola.pdf>
- Stoikov, S. (2018). The Microprice: A High-Frequency Estimator of Future Prices. SSRN 2970694.
- Cont, Kukanov, Stoikov (2014). The Price Impact of Order Book Events. <https://arxiv.org/abs/1011.6402>
- Easley, Lopez de Prado, O'Hara (2012). Flow Toxicity and Liquidity in a High-Frequency World.
- Barndorff-Nielsen & Shephard (2004). Power and Bipower Variation with Stochastic Volatility and Jumps. JFE.
- Amaya, Christoffersen, Jacobs, Vasquez (2015). Does Realized Skewness Predict the Cross-Section of Equity Returns? JFE.
- Gould & Bonart (2016). Queue Imbalance as a One-Tick-Ahead Price Predictor.
- Gu, Kelly, Xiu (2020). Empirical Asset Pricing via Machine Learning. RFS.
- López de Prado (2018). Advances in Financial Machine Learning, Wiley. Ch 5 (fractional differentiation).
- Kyle, A. (1985). Continuous Auctions and Insider Trading. Econometrica.
- Amihud, Y. (2002). Illiquidity and Stock Returns. JFM.
- Sifat, Mohamad (2019). Lead-Lag Relationship between Bitcoin and Ethereum. IRFA.
- Bitcoin wild moves: order flow toxicity and price jumps (2025). Sci Direct.
- The Anatomy of Polymarket: Evidence from the 2024 Election (2026). arXiv 2603.03136.

### Internal findings cross-referenced
- `external_signals_test_2026-04-05.md` (OBI/session/streak ablation)
- `crypto_microstructure_research.md` (prior signals review)
- `adverse_selection_analysis_2026-04-11.md` (motivates U43, U49, U50)
- `bonereaper_deep_analysis_2026-04-11.md` (competitor uses microprice-style)
- `latency_audit_2026-04-11.md` (latency arb is the priority)
- `feed_latency_2026-04-08.md` (1.23s Chainlink lag — feeds the latency arb features)

---

## Closing Note

**Total feature count after this rewrite: 149** (29 existing + 120 new).
That's still less than half of Gu/Kelly/Xiu's 900, but it's enough to
let trees find cross-feature interactions if any exist. If XGBoost
trained on this set still has `z²` as the top feature, the answer isn't
"more features" — it's "the model's job is structurally capped at
filtering, and the alpha lives in execution (latency arb), not
prediction."
