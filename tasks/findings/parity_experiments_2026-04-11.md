# Parity Experiments -- 2026-04-11

Two controlled experiments to decide whether the 15% claimed / -0.7% realized edge gap is caused by the calibration table, the underlying Gaussian model, or something else. Prior audit: `live_vs_backtest_calibration_2026-04-11.md`.

---

## 1. Executive Summary

**The calibration table is NOT the cause of the overconfidence.** Attaching the exact same calibration table live uses (`cal_prior_strength=50, cal_max_weight=0.70`) to the replay produces nearly identical results as running the replay without any calibration table at all. The replay's `p_model` distribution, claimed-vs-realized gap, and correlations are effectively unchanged by toggling the calibration on or off.

What Experiment A revealed instead is much more important: **live and replay disagree even after parity is restored.** On 63.6% of matched ticks, `|live_p_model - replay_p_model| > 0.10` (vs 60.1% without calibration -- barely different). Live `p_model` reaches `[0.001, 0.999]`; replay stops at `[0.10, 0.89]` regardless of calibration setting. On 61 cases where live's `p_model` is extreme (`|p - 0.5| > 0.40`), ZERO had an extreme replay `p_model`.

The source of the overconfidence is NOT in the signal code path the replay exercises. It must be coming from something only the live process does: state accumulation across windows, tick-level updates between parquet samples, or a path in `DiffusionSignal` that fires only on live input streams.

**One-line verdict:** the calibration table is nearly a no-op on this sample. Disabling it will not fix the 15pp gap. The next thing to investigate is why live's `_p_model_trade` at a given parquet tick differs from replay's by ~14 percentage points on average -- that gap is what's producing the fake edge.

| Question | Answer |
|---|---|
| Does disabling the calibration table fix the overconfidence? | **No.** |
| Is the calibration table a no-op on the live sample? | **Effectively yes.** Live-vs-replay p_model gap with cal: 0.1443. Without cal: 0.1375. Difference: 0.007. |
| What IS causing the overconfidence? | Not calibration. Not the "kou"/Gaussian CDF choice. Live path produces `p_model` in `[0.001, 0.999]`; replay path stops at `[0.10, 0.89]`. Source is somewhere live-only. |
| Should we disable the calibration table? | No near-term benefit. The table itself has other problems (Z_BIN_WIDTH=0.5 → only 10-20 cells) but fixing or removing it would not change per-trade realized edge. |

---

## 2. Experimental Setup

- **Sample:** 455 resolved live trades extracted by the prior audit into `/tmp/_analysis_rows.json`. No re-extraction.
- **Replay tool:** `analysis/replay_parity_experiments.py` (new). Takes `--with-calibration` / `--without-calibration` / `--both`. Same window matching and target-tau logic as the prior `replay_live_windows.py`, plus:
  - Captures `ctx['_p_model_trade']` (post-cal, post-market_blend, post-OBI) -- this is what live logs.
  - When `--with-calibration`, builds `build_calibration_table(DATA_DIR / config.data_subdir, vol_lookback_s=90)` once per market key, attaches to `sig.calibration_table`, sets `cal_prior_strength=50.0`, `cal_max_weight=0.70` -- same as `live_trader.py:1219-1225`.
- **Signal factory:** `build_diffusion_signal(market_key, bankroll=100.0, maker=False)`. Same `market_blend` / `min_entry_z` / `min_sigma` / `max_sigma` / `edge_threshold` / `max_model_market_disagreement` / `tail_mode` / ... as live (all inherited from `market_config.get_config(market_key)`).
- **Match strategy:** iterate the parquet, find the row whose `time_remaining_s` is closest to the live fill's `tau`. Mean `dtau` on matched rows = 1.5s, max 51.4s, only 3/455 windows have dtau > 5s. Matching quality is not the issue.
- **Effective Gaussian path confirmed:** `signal_diffusion.py:1034-1058` `_model_cdf` under `tail_mode="kou"` dispatches to `norm_cdf(z)`. The "kou" label is cosmetic. The replay is running the same Gaussian model as live.
- **Calibration-table content (built fresh each run):**
  - `btc_5m`: **10 cells, 93,751 observations**. Every cell has n ≥ 2,095, so `w = min(n/(n+50), 0.70) = 0.700` uniformly. The prior_strength has no effect -- every cell is already saturated at the cap.
  - `btc` (15m): **20 cells, 112,159 observations**. Every cell has n ≥ 314, `w = 0.700` for every cell.
  - Z_BIN_WIDTH=0.5 and only 4 tau bins means the entire |z| > 1 universe collapses into two bins (±1.0). This matches AUDIT_REPORT finding #11.

---

## 3. Experiment A -- Parity Restoration

**Question:** does attaching the live calibration table to the replay shrink the live-vs-replay `p_model` gap?

### 3a. Live-vs-replay p_model diff distribution

| metric                     | **WITH cal (Exp A)** | **WITHOUT cal (prior method)** | prior audit |
|---                         |---:                  |---:                            |---:|
| n matched                  | 415                  | 411                            | 396 |
| mean diff (live − replay)  | -0.0209              | -0.0151                        | +0.0042 |
| stdev diff                 | 0.1684               | 0.1608                         | 0.1558 |
| **abs_mean diff**          | **0.1443**           | **0.1375**                     | 0.1246 |
| `\|diff\| > 0.05`          | 85.1%                | 83.9%                          | n/a |
| **`\|diff\| > 0.10`**      | **63.6%**            | **60.1%**                      | **59.6%** |
| `\|diff\| > 0.20`          | 27.2%                | 23.8%                          | 24.5% |

Attaching the calibration table **does not shrink the gap**. Abs-mean diff increases slightly (0.1375 → 0.1443). The % of ticks with |Δp| > 0.10 actually rises (60.1% → 63.6%). Parity is not restored. The calibration table is either a no-op on this sample or a marginal net-negative.

### 3b. p_model distribution comparison

`p_model` at the matched tau, P(UP). Live reaches the extremes; replay does not.

| stat   | **live** | **replay WITH cal** | **replay WITHOUT cal** |
|---     |---:      |---:                 |---:                    |
| min    | 0.0013   | 0.1017              | 0.1258                 |
| p05    | 0.0374   | 0.1870              | 0.1932                 |
| p25    | 0.2052   | 0.2838              | 0.2719                 |
| p50    | 0.3997   | 0.4343              | 0.4045                 |
| p75    | 0.7257   | 0.6474              | 0.6815                 |
| p95    | 0.9185   | 0.8192              | 0.8175                 |
| max    | 0.9987   | 0.8989              | 0.8859                 |
| mean   | 0.4498   | 0.4707              | 0.4677                 |
| stdev  | 0.2856   | 0.2069              | 0.2169                 |

- Live stdev = 0.2856. Replay stdev = 0.20-0.22. Live p_model varies ~40% more than replay.
- Live min/max reach the floor/ceiling of market_blend clipping (`[0.01, 0.99]`) and ~10x past that via `_p_model_raw`. Replay never gets close to those ends.
- Calibration ON vs OFF **barely matters** for replay's distribution. The shapes are almost identical.

### 3c. Extreme-live cases

On 61 trades where `|live_p_model − 0.5| > 0.40` (n=61/455):

| metric | value |
|---|---|
| n live extreme (|p - 0.5| > 0.40) | 61 |
| n where replay (with cal) is ALSO extreme | **0** |
| max |live − replay| on extreme subset | 0.4139 |
| most extreme pair | live=0.0013, replay=0.4152 |

**On ZERO of 61 extreme-live cases does the replay reproduce an extreme p_model.** The calibration table attached at 0.70 weight cannot explain a divergence that spans from ~0 to ~0.4. The source is upstream of whatever replay computes from the parquet rows -- somewhere in the live input path.

### 3d. Conclusion (Experiment A)

- Parity is **not** restored by attaching the calibration table.
- The |Δp| distribution is statistically indistinguishable across the two replay variants.
- The overconfidence mechanism is **not** in any code path the replay touches.

---

## 4. Experiment B -- Calibration-Table Isolation

**Question:** on the replay, does the calibration table produce a larger claimed-vs-realized gap (the live symptom)?

### 4a. Claimed vs realized edge

Using replay's `p_side` (replay `p_model_trade` flipped for DOWN trades) against the **same** cost_basis live paid. Edge claimed = `p_side − cost_basis`.

| metric | **Live (reference)** | **B1: replay WITH cal** | **B2: replay WITHOUT cal** |
|---|---:|---:|---:|
| n | 455 | 415 | 411 |
| win rate | 52.1% | --- | --- |
| mean live claimed edge | +0.1518 | +0.1546 | +0.1528 |
| **mean replay claimed edge** | --- | **+0.0697** | **+0.0791** |
| mean realized edge | **−0.0073** | −0.0040 | −0.0090 |
| **gap (claimed − realized)** | **+0.1591** | **+0.0737** | **+0.0881** |
| ratio replay/live claimed | 1.0 | 0.45 | 0.52 |

Key observations:
- Replay's own claimed edge is ~7-8% -- roughly **half** of what live claimed on the same windows at the same tau. The other half of the claimed edge is being generated by whatever the live-only path does, not by the signal code the replay exercises.
- Replay still has a **+7pp claimed-vs-realized gap**, much smaller than live's +16pp but still material. The Gaussian model IS somewhat overconfident on its own.
- The claimed-vs-realized gap is **smaller with calibration** (+7.4pp) than without (+8.8pp). The calibration table is marginally beneficial on this sample, contradicting the audit's hypothesis that it's harmful. But the effect size is tiny (1.4pp).

### 4b. Point-biserial correlations

| | **B1 WITH cal** | **B2 WITHOUT cal** | **Live (reference)** |
|---|---:|---:|---:|
| `corr(replay_edge_claimed, win)`  | **+0.0101** (t=0.21) | +0.0140 (t=0.28) | +0.0404 (t=0.86) |
| `corr(replay_p_side, win)`        | **+0.3886** (t=8.57) | +0.3829 (t=8.38) | +0.4033 (t=9.38) |

- Replay's `p_side` has `r ≈ 0.39` with outcome -- **nearly identical to live's `r = 0.40`**. The directional signal survives the code path. Whatever live is doing that replay isn't, it's not adding or removing directional information.
- Replay's `edge_claimed` has `r ≈ 0.01` -- **even worse than live's 0.04**, both statistically zero. The edge magnitude is noise in both.
- **Turning the calibration table on vs off changes nothing meaningful**: 0.3886 vs 0.3829 for p_side correlation, 0.0101 vs 0.0140 for edge correlation. Delta well within noise.

### 4c. Replay p_side calibration buckets

**B1: WITH calibration**
| p_side bucket | n | live_WR | replay p_side mean | gap |
|---|---|---|---|---|
| [0.0, 0.3) | 48 | 0.104 | 0.201 | +0.097 |
| [0.3, 0.4) | 24 | 0.167 | 0.340 | +0.174 |
| [0.4, 0.5) | 23 | 0.391 | 0.459 | +0.068 |
| [0.5, 0.6) | 76 | 0.553 | 0.557 | +0.004 |
| [0.6, 0.7) | 103 | 0.553 | 0.644 | +0.091 |
| [0.7, 0.8) | 109 | 0.679 | 0.749 | +0.070 |
| [0.8, 0.9) | 32 | 0.750 | 0.838 | +0.088 |

**B2: WITHOUT calibration**
| p_side bucket | n | live_WR | replay p_side mean | gap |
|---|---|---|---|---|
| [0.0, 0.3) | 57 | 0.123 | 0.211 | +0.088 |
| [0.3, 0.4) | 20 | 0.200 | 0.340 | +0.140 |
| [0.4, 0.5) | 12 | 0.417 | 0.455 | +0.038 |
| [0.5, 0.6) | 58 | 0.517 | 0.561 | +0.044 |
| [0.6, 0.7) | 111 | 0.559 | 0.656 | +0.097 |
| [0.7, 0.8) | 121 | 0.645 | 0.753 | +0.108 |
| [0.8, 0.9) | 32 | 0.781 | 0.828 | +0.047 |

**Live (reference)**
| p_side bucket | n | live_WR | mean live p_side | gap |
|---|---|---|---|---|
| [0.0, 0.3) | 5 | 0.000 | 0.216 | +0.216 |
| [0.3, 0.4) | 21 | 0.095 | 0.380 | +0.285 |
| [0.4, 0.5) | 48 | 0.188 | 0.435 | +0.247 |
| [0.5, 0.6) | 33 | 0.242 | 0.553 | +0.311 |
| [0.6, 0.7) | 55 | 0.600 | 0.658 | +0.058 |
| [0.7, 0.8) | 96 | 0.542 | 0.751 | +0.210 |
| [0.8, 0.9) | 127 | 0.622 | 0.840 | +0.218 |
| [0.9, 1.0) | 70 | 0.771 | 0.965 | +0.194 |

- **Replay buckets are much better calibrated than live buckets.** Worst replay gap is +0.17 (B1 [0.3, 0.4)); worst live gap is +0.31 (live [0.5, 0.6)).
- Replay NEVER has a bucket above 0.9 (its p_model never exceeds ~0.90), so the replay cannot even express the "90% confidence" regime where live claims it's most sure.
- B1 vs B2 buckets differ by at most 5pp in any bin and mostly by <3pp. **The calibration table barely moves the needle.**

### 4d. Per-trade claimed-edge gap

| metric | WITH cal | WITHOUT cal |
|---|---|---|
| abs_mean(live_edge − replay_edge) | **0.1072** | 0.1016 |
| abs_mean(live_p_side − replay_p_side) | **0.1443** | 0.1375 |
| `|live_edge − replay_edge| > 0.10` | 49.9% | 46.7% |
| `|live_p_side − replay_p_side| > 0.10` | 63.6% | 60.1% |

On nearly half of trades, the replay's claimed edge differs from live's claimed edge by more than 10 percentage points. On 63.6% of trades, the replay's `p_side` differs by more than 10pp. The replay and live are producing DIFFERENT predictions at the same tau in the same parquet window, regardless of calibration.

---

## 5. Which Component Is Broken?

**Not the calibration table.** Experiment B's B1 and B2 are statistically equivalent on every metric that matters (correlations, claimed-vs-realized gap, p_model distribution, bucket calibration). Toggling the calibration table changes the gap by at most 1.4 percentage points on the 415-row sample -- well within noise.

**Partly the Gaussian model.** Even in its clean replay form the Gaussian diffusion overstates edge by about 7-9 percentage points (claimed +7.0-7.9% vs realized -0.4% to -0.9%) and has zero correlation between claimed edge magnitude and wins. So some of the 16pp live gap is intrinsic to the GBM signal. But:

- 7pp of residual gap in replay + 16pp gap in live ≠ the same problem scaled. Live's additional ~9pp gap is ONTO TOP of the Gaussian baseline, and has a completely different source.
- Replay's `p_side` has almost identical directional power (`r = 0.39` vs live's `r = 0.40`). So the extra 9pp of live overconfidence is **pure magnitude noise** -- it's inflating the claimed edge without adding information.

**Mostly something in the live-only path.** The 7pp → 16pp escalation between replay and live happens somewhere in:

1. **Live state accumulation across windows.** `DiffusionSignal` retains `_sigma_ema` and other ctx keys between `decide_both_sides` calls. Replay builds a fresh signal per window (see `replay_parity_experiments.py:96`). If live's EMA has collapsed to a very small number between windows, the resulting z-score explodes and `norm_cdf(z)` goes to 0 or 1. Replay's cold-start sigma from 90s of in-window history is always larger than this collapsed value.
2. **Live tick-level updates between parquet samples.** The parquet is 1Hz-sampled. Live sees every Binance book-ticker update (~10Hz) and every trade-tape entry. The live `_compute_vol` can see sharper moves within the 1s window that the parquet smooths out. This would tend to push live's z higher than replay's.
3. **Live history buffer is longer.** If live's `price_history` spans >> `vol_lookback_s` and a prior window's prices are polluting the current window's sigma estimate, the effect is strongly regime-dependent.

The fact that replay's `p_model` CANNOT reach below 0.10 or above 0.90 (clipped by `min_entry_price=0.20` only applies to cost_basis, not to p_model, so this floor is coming from inside `_model_cdf` through the `z` clamping at ±3 * min_sigma * √tau). Live somehow breaks through this clamp, which means live is using a **smaller sigma or larger delta than replay at the same tick**.

---

## 6. Concrete Recommendation

### Primary: DO NOT disable the calibration table.

Disabling it will not fix the 15pp gap. The calibration table is already a near-no-op on this sample (0.7pp gap difference between B1 and B2). The audit's R2 recommendation was based on the hypothesis that Z_BIN_WIDTH=0.5 was "pulling p_model toward 0.5" -- but the table actually uses n/(n+50) weighted blending, and every cell has so much data that the weight is saturated at 0.70, which on THIS sample produces marginally LESS overconfidence than no calibration at all.

If anything, `cal_max_weight` could be raised to 1.0 and re-evaluated, but the effect size will remain tiny (<2pp). Not worth the risk to trade without any corrective step at all.

### Primary: FIX LIVE-VS-REPLAY PARITY by investigating live-only state.

This is the real root cause. Concrete next steps:

1. **Instrument live `DiffusionSignal` to log `_p_model_raw`, `_p_model_trade`, and `_sigma_per_s` at every decision**, and compare to a simultaneous replay on the tick-by-tick live input. The live bot already has diagnostic logging at `tracker.py:1464`. Add `_p_model_raw` and `_sigma_per_s` to that log row permanently.
2. **Cold-start the live signal per window**: at the start of every new market window, call `sig._reset_for_window()` or equivalent. If this doesn't exist, add it. If after this the p_model distribution tightens to match replay's, state leakage is the culprit.
3. **Check the live feed ingestion code path** (`feeds.py`, `signal_ticker.py`) for any place where `ctx['_p_model_raw']` is written from a DIFFERENT code path than `decide_both_sides`. The replay only ever exercises `decide_both_sides`. If live has a second write location, replay will never reproduce it.
4. **Log live's sigma_per_s on every decision** and cross-check against replay's sigma on the same parquet row. If live's sigma is systematically smaller (say, <50% of replay's), that explains the blown-up z-score and the extreme p_model.

### Secondary: the underlying GBM IS partially overconfident (7% gap even in clean replay).

After parity is fixed, there will still be a residual 7-8pp gap. That needs a different remedy:

- `edge_threshold` raise (currently 0.06) would not help since `corr(edge_claimed, win) ≈ 0` in both replay variants. The high-edge bucket is not more likely to win.
- The real fix is structural: replace the "forecast terminal probability" model with one that doesn't rely on `z/σ` terms that blow up when sigma collapses. Options discussed in the prior audit R5: drop 5m, replace with a Binance-Polymarket lag rule, or size by `p_side > 0.5` as a binary directional gate.
- The 7pp residual gap is consistent with `corr(p_side, win) ≈ 0.39` -- i.e. p_side explains ~15% of outcome variance, enough to produce real but smaller edge than the model claims.

### Tertiary: Z_BIN_WIDTH tuning is a side quest.

AUDIT finding #11 is correct that Z_BIN_WIDTH=0.5 is far too wide (only 10 cells total for 5m!). Narrowing to 0.10 would produce ~50 cells and might let the calibration table actually do something. But our data shows that on the CURRENT calibration table, fixing it or removing it both have essentially zero effect on realized edge. Retuning Z_BIN_WIDTH is only worthwhile IF the live-vs-replay parity is first fixed -- without parity, a narrower table will also get bypassed by whatever mechanism is producing the extreme live p_models.

---

## 7. Next Steps (Operator-Facing)

Priority ordered, with scope and expected effort:

1. **[1h] Add `_p_model_raw`, `_p_model_trade`, `_sigma_per_s`, `ctx['_p_display']` to `tracker._log_diagnostic`**. They already exist in ctx. Just expose them in the log row. This gives tick-level live visibility for parity debugging.
2. **[2h] Read `signal_diffusion.py` for any place `_p_model_raw` is set OUTSIDE `_p_model()`**. The replay only hits the `decide_both_sides` path. Confirm there is no alternate live-only code path. See signal_diffusion.py lines 1336, 1641, 1945, 2323.
3. **[2h] Write a "live replay parity" test**: given a recorded live trade at tau T on slug S, feed the stored parquet row-by-row through `decide_both_sides` and assert that the p_model at the matched tau equals the p_model in the live jsonl within some tolerance (say 5pp). If it doesn't, dump `ctx` and compare field-by-field against the jsonl's model snapshot.
4. **[4h] Investigate state leakage**: log `len(price_history)`, `ts_history[0]`, `ts_history[-1]`, `_sigma_per_s` at every decision. Check whether live's price_history accumulates across windows. If yes, add `_reset_for_window()` and re-test.
5. **[1h, last resort] Disable the calibration table in live**. Set `cal_table = None` in `live_trader.py:1223`. Expected effect: realized edge changes by <1pp (and not in an obvious direction). This is the only action the audit recommended; our data shows it won't help but it also won't hurt.

**Do not tune `edge_threshold` or any Kelly parameter against the current backtest** until the parity bug is fixed. Every knob tuned against the replay's signal is tuned against a different model than what runs in production.

---

## 8. Artifacts

- `/Users/dannychee/Desktop/prediction-market-bot/analysis/replay_parity_experiments.py` -- new replay harness with `--with-calibration` / `--without-calibration` / `--both` flags. Uses `build_diffusion_signal` + attaches the same calibration table live uses. Writes `/tmp/_parity_with_cal.json` and `/tmp/_parity_without_cal.json`.
- `/Users/dannychee/Desktop/prediction-market-bot/analysis/analyze_parity.py` -- experiment A+B analyzer. Prints all tables in sections 3-4 above.
- `/Users/dannychee/Desktop/prediction-market-bot/analysis/analyze_parity_extras.py` -- calibration-table inspection, extreme-case divergence, per-bucket calibration for both replay variants.
- `/tmp/_parity_with_cal.json` -- 455 replays with calibration table attached (40 rows missing replay p_model because window warmup didn't complete).
- `/tmp/_parity_without_cal.json` -- 455 replays with calibration_table=None (44 rows missing).
- `/tmp/_analysis_rows.json` -- 455 live trade rows extracted by the prior audit. Not modified.
- `/tmp/_replays.json` -- the prior audit's replay output. Kept for reference.
