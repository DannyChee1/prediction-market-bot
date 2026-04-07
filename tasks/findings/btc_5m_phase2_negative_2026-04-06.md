# BTC 5m Phase 2 — Negative Results

## TL;DR

Three "quick win" candidates investigated; **none ship**. Phase 1 (`market_blend=0.3`) is the entire current edge. Real next-step improvements require Phase 3 work (DVOL feed, snipe mode, latency profiling, ML uplift).

## Candidates investigated

### A1. `chainlink_blend_s = 60` for BTC 5m (was 120)

**Hypothesis.** A 120s blend window for a 300s BTC 5m window means 40% of the window pulls effective price toward Chainlink. For a 5m window where the snipe edge is in the final ~60s, blending earlier than that fights `market_blend=0.3`. Tighten to 60s (last 20% of window).

**Result.** Bit-identical PnL/WR/Sharpe across 5 different time slices vs the unchanged config. Investigation:

- Current trade tau distribution (latest 200 windows): min=57, p10=69, p25=78, **median=114**, p75=178, p90=249.
- 50% of trades fire at tau > 120 → chain_blend has zero effect on them regardless of value.
- For trades at tau ∈ [60, 120], the blend weight is `1 - tau/120` = 0% to 50%. At tau=100 (a typical entry), it's 17% Chainlink. Combined with the typical near-zero Binance−Chainlink gap, the effective price shift is below the threshold needed to flip an edge decision.

**Conclusion.** A1 is a no-op in the current oracle regime. Don't ship. If Binance and Chainlink ever diverge meaningfully (e.g., during a fast move), revisit.

---

### A2. Wire the XGBoost filtration model for BTC 5m

**Hypothesis.** `filtration_model.pkl` exists, was trained with explicit `asset_id=4` for BTC 5m, has the gate code at `backtest.py:1191-1268` (`_check_filtration`), but is never instantiated in `live_trader.py` (zero references confirmed by grep). Wiring it should add a quality gate.

**Result.** Identical trade set across thresholds 0.45, 0.50, 0.55, 0.60, 0.65. The model is approving every trade that reaches it.

**Why it's a no-op.** The check has an early return at `abs(z) < 0.10` that skips most low-z ticks. By the time z is large enough for the check to actually run, the trade has already passed several other gates (min_z, edge, min_entry_price), and the model — trained on a more directional regime — happily approves the strong-z setups.

**Conclusion.** Don't ship. The model needs retraining on current-regime data before it can add value, OR a different threshold strategy. Adding to Phase 3 backlog: re-train filtration model on last ~30 days of BTC 5m parquets and re-test.

---

### A3. `obi_weight = 0.0` for BTC 5m

**Hypothesis.** `external_signals_test_2026-04-05.md` shows the OBI signal has correlation −0.092 (very weak) and may even be inverted (the doc itself says "may be FLIPPED"). Disabling it for BTC 5m removes a noise source.

**Result. Regime-fragile.** Multi-slice Sharpe comparison vs Phase 1 baseline:

| Slice | P1 Sharpe | P1+A3 Sharpe | Δ |
|---|---:|---:|---:|
| latest 200    | 0.79 | 0.92 | **+0.13** ✓ |
| 200-500 ago   | 0.71 | 1.04 | **+0.33** ✓ |
| 500-900 ago   | 1.78 | **0.53** | **−1.25** ✗ |
| 900-1300 ago  | 4.31 | 3.43 | **−0.88** ✗ |
| latest 800    | 1.96 | 1.23 | **−0.73** ✗ |

A3 helps the most recent 500 windows but hurts the 500-1300 ago slices. The OBI signal was actually positive on older data and is currently noise. Disabling it is **regime-current** but not **regime-robust**.

**Conclusion.** Don't ship without a regime-detection gate. The user's Sharpe-first preference argues against shipping a change that drops Sharpe by 37% on the broader sample even if it helps the latest 200.

---

## What's still on the table (Phase 3)

In priority order:

1. **DVOL regime gate.** `recorder.py:312-369` already collects Deribit DVOL but `live_trader.py` doesn't see it. Plumb it through `feeds.py` and use as a regime filter (skip entries when DVOL > X%). Requires backtest with DVOL column populated.
2. **Resolution snipe supplement mode.** Per `resolution_snipe_analysis.md`: 70.6% WR for BTC 5m at tau≤60s when filtered by move > 0.01% (n=51). New decision branch in `decide_both_sides()` for the late-window edge.
3. **Re-train filtration model on current regime.** Pull last 30 days of BTC 5m parquets, retrain `filtration_model.pkl`, validate against forward windows. The model trained on older data approves everything in current regime.
4. **VPIN data fix.** `last_trade_side_up/down` parquet columns are 100% null in current data. Either fix the recording pipeline or remove the dead VPIN code path.
5. **Latency profiling.** `signal_eval_ms` is logged in `live_trades_*.jsonl`. Distribution analysis to find and fix p95 outliers.
6. **ML uplift model.** Train a small XGBoost on (features → realized PnL) instead of (features → win) to learn entry quality. Stack on top of blended `p_model`.
7. **Time-of-day vol prior.** Already partially captured by `time_prior_sigma`; could be a per-hour `kelly_fraction` multiplier per `external_signals_test`.

## Process note

This negative result is itself the deliverable for Phase 2. The Phase 1 fix (market_blend) extracted the easy edge. Going further requires either new data sources, retraining, or architectural work — all of which need their own planning cycles.
