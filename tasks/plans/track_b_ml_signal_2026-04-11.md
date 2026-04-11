# Track B — ML-Augmented Signal Plan (2026-04-11)

Parallel to Track A (latency arb, already shipping). Goal: improve the
existing GBM/maker signal by adding feature engineering + tree-based
filter, NOT by replacing the model wholesale. Inspired by Gu, Kelly,
Xiu (2020) — the lift comes from cross-feature interactions, which
trees capture natively.

**Existing infrastructure to reuse:**
- `filtration_model.pkl` already exists in the project root
- `filtration_model.py` has the load/predict logic
- `signal_diffusion.py` has the consumption hook (`_check_filtration`)
  but it's currently inert in live (only runs in backtest)
- `tasks/findings/filtration_regression_2026-04-08.md` documents the
  prior model's performance and feature set

**Decision principle:** the model is a TRADE FILTER, not a primary
signal. The GBM signal still decides direction; the model decides
whether the GBM signal is reliable enough this tick.

---

## Step 0: Verify the Existing Filtration Model (1 hour)

Before doing any new work, understand what we already have.

**Tasks:**
1. Read `filtration_model.py` — feature list, target variable, model type
2. Load the existing pkl: `import pickle; model = pickle.load(open('filtration_model.pkl', 'rb'))`
3. Inspect: `model.feature_names_`, `model.feature_importances_`, accuracy
4. Read `tasks/findings/filtration_regression_2026-04-08.md`
5. Find where it's called in `signal_diffusion.py:_check_filtration`
6. Find why it's not loaded in `live_trader.py` (the OFI agent confirmed live never reads it)

**Output:** a 1-pager `tasks/findings/filtration_baseline_audit_2026-04-11.md`
listing what features it has, how it was trained, and what it's missing.

---

## Step 1: Engineer 20-30 New Features (1-2 days)

Augment the feature set. The goal is enough features to capture
multi-timescale interactions, but not so many that overfitting takes over.

### Multi-timescale Binance momentum (4 features)
- `bn_ret_5s`: log return over last 5 seconds
- `bn_ret_30s`: log return over last 30 seconds
- `bn_ret_60s`: log return over last 60 seconds
- `bn_ret_300s`: log return over last 5 minutes

### Volatility regime (4 features)
- `sigma_30s`: realized vol over last 30s
- `sigma_300s`: realized vol over last 300s
- `vol_ratio`: `sigma_30s / sigma_300s` (regime change indicator)
- `vol_quantile_session`: where current σ ranks within today's distribution

### Order book microstructure (5 features)
- `obi_top1`: imbalance at best level
- `obi_top3`: imbalance over top 3 levels
- `obi_top5`: imbalance over top 5 levels
- `spread_ratio`: current spread vs rolling median spread
- `book_depth_total`: total size on top 5 levels

### Time encoding (4 features)
- `hour_sin`: `sin(2π × hour_utc / 24)`
- `hour_cos`: `cos(2π × hour_utc / 24)`
- `dow`: 0-6 day-of-week
- `is_us_hours`: 1 if 13:00-21:00 UTC else 0 (per existing finding that
  US hours have 1.25× larger moves and 57% UP bias)

### Cross-asset (3 features — defensive, may not help)
- `eth_ret_60s`: ETH 60-second return at matched horizon
- `eth_btc_ratio_change`: change in ETH/BTC ratio over 60s
- `cross_asset_z`: z-score of (BTC return − ETH return) over 60s

### Window-relative (4 features)
- `tau_frac`: `time_remaining / window_duration` (already implicit but worth adding)
- `delta_per_sigma`: `delta / (sigma × √elapsed)` (z-score of current move vs typical)
- `move_persistence`: sign-agreement of last 10 sub-second returns
- `recent_max_drawdown`: largest drawdown from peak in last 30s

### Trade tape (3 features — REQUIRES recorder patch first)
- `tfi_5s`: trade flow imbalance over last 5 seconds
- `tfi_30s`: trade flow imbalance over last 30 seconds
- `trade_count_30s`: number of trades in last 30 seconds

### Existing GBM features (already there, KEEP)
- `p_model`, `p_side`, `edge`, `sigma_per_s`, `tau`, `cost_basis`,
  `dyn_threshold`, `oracle_lag`, `vpin`, `toxicity`, `mid_up`,
  `bid_up`, `ask_up`, `spread_up`

**Total: ~27 new features + ~14 existing = ~41 features.** Comparable
to filtration_model.pkl's 29 baseline + augmentation.

### Feature ETL pipeline
- Build a `features.py` module that exposes
  `compute_features(snapshot, ctx, history_buffer) -> dict[str, float]`
- Each feature is a pure function of available state — no future
  leakage
- Unit test each feature on synthetic input

**Effort:** 1-2 days. Mostly pure Python; the trade-tape features
require the recorder patch (defer until later).

---

## Step 2: Re-train filtration_model.pkl (1 day)

### Data prep
- Iterate over all `data/btc_5m/*.parquet` files
- For each window, sample 5-10 ticks at decision-time taus (e.g.
  240, 180, 120, 60s remaining)
- For each sample, compute the new feature vector
- Label = the eventual window outcome (UP=1, DOWN=0)
- Output: a single training parquet `train_features_2026-04-11.parquet`

### Train
- 80/20 walk-forward split (not random — preserve temporal order)
- XGBoost with conservative hyperparameters:
  - `n_estimators=200`
  - `max_depth=4` (shallow trees prevent overfit on small data)
  - `learning_rate=0.05`
  - `subsample=0.8, colsample_bytree=0.8`
  - `early_stopping_rounds=20`
- Track: AUC, accuracy at p>0.5, calibration plot
- Save as `filtration_model_v2_2026-04-11.pkl` (don't overwrite original)

### Sanity checks
- AUC > 0.55 on out-of-sample (anything less is no better than random)
- Calibration plot is roughly diagonal
- Feature importances make physical sense
- SHAP values for top 10 features
- **Compare to filtration_model.pkl baseline** — if v2 doesn't beat v1
  by at least 1pp AUC, the new features aren't helping

### Output
- `filtration_model_v2_2026-04-11.pkl`
- `tasks/findings/filtration_v2_training_2026-04-11.md` with metrics

**Effort:** 1 day.

---

## Step 3: Wire into Live (0.5 day)

Currently `filtration_model.pkl` is loaded only in `backtest.py`. Live
never reads it. Two changes:

### a) Load the model in live_trader.py
Find where the signal kwargs are built and add:
```python
# Load filtration model for live use (parity with backtest)
try:
    fm_path = ROOT / "filtration_model_v2_2026-04-11.pkl"
    if fm_path.exists():
        from filtration_model import FiltrationModel
        signal_kw["filtration_model"] = FiltrationModel.load(fm_path, ...)
        signal_kw["filtration_threshold"] = args.filtration_threshold
except Exception as exc:
    print(f"[WARN] filtration model load failed: {exc}")
```

### b) Add a `--filtration` CLI flag
Default off. When enabled, the live signal calls `_check_filtration`
in `decide_both_sides` (the consumption hook already exists per
`signal_diffusion.py:1983`).

### c) Use the existing `filtration_mode` (size_mult or gate)
- `gate` = binary skip if confidence < threshold
- `size_mult` = always pass, multiply Kelly by confidence
- Recommend `size_mult` initially — softer than a hard gate

**Effort:** 0.5 day.

---

## Step 4: A/B Test in Live (2 days)

### Phase 1: Dry-run for 24h
```bash
caffeinate -i .venv/bin/python live_trader.py --market btc \
    --filtration --label fm_v2 --no-record \
    --max-trades-per-window 10 --max-positions 5 --dry-run \
    --bankroll 100 2>err_fm.log | tee stdout_fm.log
```
Compare to a baseline run (same flags but no `--filtration`) in
parallel using the `--label` mechanism we just added.

### Phase 2: Compare metrics
- Trades fired (each instance)
- Win rate
- Realized edge
- Calibration gap (claimed vs realized edge)
- Sharpe (if enough trades)

### Phase 3: Ship-up criteria
The filtration v2 model ships to live (replacing the inert one) IF:
- AUC out-of-sample > 0.55 in training
- Live A/B WR lift > 2pp over baseline
- Live A/B realized edge lift > 1pp over baseline
- No regressions on calibration (gap doesn't widen)

If any of those fail, iterate on features (not on the model
hyperparameters — they're not the bottleneck).

**Effort:** 2 days (mostly waiting for live data).

---

## Step 5: Optional — NN3 Comparison (1 week, only if XGBoost wins)

If XGBoost shows clear lift, try a shallow neural net per Gu/Kelly/Xiu's
NN3 (3 hidden layers).

### Setup
- PyTorch
- Architecture: input → 64 → 32 → 16 → sigmoid
- AdamW optimizer, lr=1e-3, weight_decay=1e-4
- Dropout 0.2 between layers
- Early stopping on validation AUC, patience=10
- Same feature set as XGBoost

### Comparison
Train XGBoost and NN3 on the same data, evaluate on the same OOS split.
Expect NN3 to outperform XGBoost by maybe 1-3pp AUC IF interactions are
dense. Likely not worth the engineering cost for a small lift, but
we'll know.

### Verdict criterion
NN3 ships only if it beats XGBoost by >2pp AUC AND the live A/B
shows >1pp WR lift over XGBoost. Otherwise stay with XGBoost.

**Effort:** 1 week. This step is OPTIONAL.

---

## Total Timeline

| Step | Effort | Calendar |
|---|---|---|
| 0. Audit existing filtration model | 1 hr | day 0 |
| 1. Feature engineering | 1-2 days | day 1-2 |
| 2. Re-train + validate | 1 day | day 3 |
| 3. Wire into live | 0.5 day | day 3 |
| 4. A/B test (dry + live) | 2 days | day 4-5 |
| 5. NN3 comparison (optional) | 1 week | day 6-12 |

**Track B time-to-decision:** ~5 days for XGBoost ship/no-ship.
**Track B time-to-NN3 verdict:** ~12 days if XGBoost wins.

---

## Why This Path (Not Others)

### Why not start with NNs?
Sample size. Gu/Kelly/Xiu had 22M observations and NN3 was their
ceiling. We have ~5k labeled windows with the new feature set.
XGBoost handles 5k samples comfortably; NNs need 100k+ to consistently
beat trees on tabular data.

### Why not just engineer features for the GBM model directly?
Because the GBM model is LINEAR in (delta, sigma, tau). It can't
combine features in nonlinear ways. The Gu/Kelly/Xiu finding was
that linear models don't beat random even with 900 features —
it's the cross-interactions that matter.

### Why XGBoost over Random Forest?
Both work well in their paper. XGBoost was slightly better on Sharpe
(1.30 vs 1.22). XGBoost also has better feature importance and faster
inference. Random Forest is fine if XGBoost is unavailable.

### Why a filter and not a replacement signal?
Risk management. The GBM signal has known directional alpha
(`corr(p_side, win) = 0.40`). Replacing it wholesale loses that.
Layering a filter on top means: in the worst case the filter
contributes nothing; in the best case it removes the calm-market
losing trades that the calibration audit identified.

### Why not just raise edge_threshold to filter?
Because edge is uncorrelated with outcomes (`corr(edge, win) = 0.04`).
Tightening a noise threshold doesn't help. The filter needs to use
DIFFERENT features (multi-timescale momentum, vol regime, etc.) than
the ones the GBM model already uses.

---

## Risks and Failure Modes

1. **Overfitting on small data.** Mitigation: walk-forward CV, shallow
   trees, early stopping, dropout (for NN), feature importance audit.

2. **Live-vs-backtest divergence.** We have a known parity issue
   (max_z drift, etc.) that today's commits partially fixed. Before
   running ANY filtration A/B in live, re-verify with the tick
   debugger that the same window produces the same model output in
   live and replay.

3. **Lookhead bias in feature ETL.** Every new feature MUST be
   computable from data available at decision time. No future
   prices, no end-of-window outcomes. Unit tests should explicitly
   check this.

4. **The new features don't help.** This is a real possibility. The
   calibration audit suggested the model has a structural ceiling
   ~5% real edge that no feature engineering will fix. If XGBoost
   v2 doesn't beat the baseline by >2pp WR, that's evidence for the
   ceiling hypothesis and the right move is to lean harder on
   latency arb instead.

5. **The filtration model becomes a crutch.** If the filter trims
   trade volume too aggressively, the bot becomes too conservative
   and misses opportunity cost. Use `size_mult` mode (soft scaling)
   not `gate` mode (hard skip) to mitigate.

---

## Dependencies

- Track A (latency arb) is independent and can ship in parallel.
- This plan does NOT need the recorder patch — TFI features can be
  added later when the recorder is patched. Initial v2 ships with
  ~38 features (no TFI), with TFI added in v3 once data is available.
- Dublin VPS deployment is independent and benefits both tracks.

---

## Out of Scope

- Reinforcement learning (not enough data, action space is too large)
- LLM-based features (no clear use case at this timescale)
- Sentiment data from social media (latency too high for 5m windows)
- On-chain features (latency too high, mostly noise at our timescale)
- Options-implied vol from Deribit (interesting but expensive to integrate)
