# Filtration Regression Target — 2026-04-08 (negative result)

## Hypothesis

The current filtration model is an XGBoost classifier with label
"was the z-score signal direction correct?" When used as a Kelly
sizing modifier (P10.1), the model's `predict_proba(features)` returns
P(direction correct), which is then mapped linearly to a multiplier
in `[0, 1]`.

The hypothesis: train a regressor on **realized PnL per dollar invested**
instead. This is closer to what we actually care about (expected return,
not classification accuracy) and should give a better sizing signal:

- For each (z, tau) checkpoint, compute the trade we WOULD have placed
  at the current ask price.
- Realized PnL/$ = `(1 - cost - fee) / cost` if direction correct,
  else `-1.0 - fee/cost`.
- Use as XGBRegressor target (no calibration needed — output is already
  a return).
- Map predicted EV → Kelly multiplier directly (>0 → scale linearly to
  the ceiling, <0 → zero size).

## What I shipped (infrastructure stays)

| File | Change |
|---|---|
| `filtration_model.py` | New `RegressionWrapper` class with `target_type="regression"` and `predict_proba(X)[:, 1] = predicted_pnl_per_dollar`. `FiltrationModel.target_type` introspects the underlying wrapper. Backwards compatible — `target_type` defaults to `"classification"` for legacy models. |
| `train_filtration.py` | New `--target {classification, regression}` flag. Regression mode computes per-row PnL/$ using `best_ask_up`/`best_ask_down` from the parquet, trains XGBRegressor (no calibration), and saves to `filtration_model_pnl.pkl` by default. RMSE/R²/decile-PnL diagnostics replace the classification report. |
| `signal_diffusion.py` | `_filtration_size_multiplier` now branches on `target_type`. Regression branch maps predicted EV linearly from 0.0 (mult=0) to `filtration_ev_full` (mult=1). Below 0 → zero size. New `filtration_ev_full: float = 0.50` parameter. |
| `backtest.py` | New `--filtration-model-path` CLI flag (defaults to `filtration_model.pkl`). New `--filtration-ev-full` flag for the regression ceiling. Plumbs through `build_diffusion_signal`. |

## Training results

Regression model trained on 36,530 train + 15,657 val samples across all
markets at MIN_Z_SIGNAL=0.10:

- **RMSE: 0.5786** vs label std 0.5740 — slightly worse than predicting the mean
- **R²: -0.0161** — confirms the model doesn't beat baseline
- **Mean prediction: +0.0321** vs actual mean **+0.0109** — slight upward bias
- **Prediction std: 0.1071** vs actual std **0.5740** — 5× compression

The decile breakdown of realized PnL by predicted-EV decile:

| decile | n | pred_ev | actual_pnl |
|---:|---:|---:|---:|
| 1 | 1566 | -0.1744 | -0.1080 |
| 2 | 1566 | -0.0502 | +0.0111 |
| 3 | 1565 | -0.0134 | +0.0068 |
| 4 | 1566 | +0.0053 | +0.0274 |
| 5 | 1566 | +0.0200 | +0.0086 |
| 6 | 1565 | +0.0357 | +0.0303 |
| 7 | 1566 | +0.0569 | +0.0351 |
| 8 | 1565 | +0.0854 | +0.0359 |
| 9 | 1566 | +0.1289 | +0.0544 |
| **10** | 1566 | **+0.2265** | **+0.0070** ← collapse |

So the model correctly identifies the **bottom decile** (predicted ≈ -0.17,
actual ≈ -0.11). Middle deciles (4-9) show monotonic actual PnL growth.
But **decile 10 collapses** — the highest-predicted-EV trades realize
LESS than middle deciles. Classic overfit at the extremes.

Top features (similar to classification model):
`tau, log_vol_regime, vol_regime_ratio, z, z_x_vol_regime, log_sigma`.

## Backtest A/B (50d, walk-forward 70/30, seed 42)

| Mode | btc_5m PnL | Sharpe | DD | btc 15m PnL | Sharpe | DD |
|---|---:|---:|---:|---:|---:|---:|
| **Classification** (current default) | **$26,577** | **1.51** | 4.0% | **$3,918** | **1.34** | 6.3% |
| Regression ev_full=0.50 | $20,906 | 1.39 | 5.2% | $2,635 | 0.95 | 6.0% |
| Regression ev_full=0.10 | $22,326 | 1.41 | 5.2% | $2,907 | 1.02 | 6.1% |
| Regression ev_full=0.05 | $22,511 | 1.41 | 5.2% | $2,953 | 1.03 | 6.0% |

**Regression loses on both markets at every ceiling tested:**
- btc_5m: -$4,066 (-15%) PnL, -0.10 Sharpe, +1.2pp DD
- btc 15m: -$965 (-25%) PnL, -0.31 Sharpe vs classification

Tightening the ceiling (50¢ → 5¢) improves things by ~$1k on btc_5m and
~$300 on btc 15m, but never recovers to classification baseline.

## Why it fails

1. **PnL/$ is bimodal, not smooth.** The label is essentially "lose ~1.0"
   or "win 0.5 to 2.0". A regressor's MSE objective predicts the mean of
   each cluster, which for our positively-skewed dataset is just slightly
   above zero. The model can't learn fine-grained discrimination of WHICH
   trades land in the high-win cluster — only that the MEAN is positive.

2. **Prediction variance is far below label variance.** Std of predictions
   is 0.107 vs 0.574 for labels — 5× compression. The model is essentially
   outputting "everything looks like ~+0.03 EV" with small wiggles. With
   the default ceiling of 0.50, that puts almost all trades at mult ≈ 0.06.
   The whole strategy gets shrunk to ~6% of normal Kelly.

3. **The decile-10 collapse is a smoking gun.** When the model is MOST
   confident about a high-EV trade, it's overfitted on training noise.
   The trades it labels "best" turn out to be average-or-worse. This is
   the same lesson as the Hawkes filtration experiment (2026-04-07):
   classification metrics don't predict portfolio metrics.

4. **The classification model already encodes most of the useful signal.**
   When the classification gate (P direction correct) is well-calibrated,
   the size_mult mapping (P10.1) already captures the same structure
   that the regressor was trying to predict. Adding a separate model on
   top doesn't add information.

## What stays

- `RegressionWrapper` class in `filtration_model.py` — useful for future
  experiments with different regression targets (e.g., Sharpe-aware,
  risk-adjusted return).
- `--target regression` flag in `train_filtration.py` — can be re-run
  with different label formulations.
- `filtration_model_pnl.pkl` — kept on disk (gitignored via the existing
  `filtration_model.pkl.*` pattern... wait, that pattern only matches
  `.pkl.X`, not `_pnl.pkl`. Adding to gitignore explicitly).
- `--filtration-model-path` CLI flag — handy for A/B testing alternate
  models without renaming files.
- `filtration_ev_full` parameter in `DiffusionSignal` — already plumbed
  through, no-op for classification models.

## What didn't ship

- The regression model is not made the default. `filtration_model.pkl`
  (classification, P10.1 size_mult mode) remains the production model.

## Next experiments worth trying (if revisiting)

1. **Classification on a richer label.** Instead of "won/lost", train on
   "this trade was in the top half by PnL/$ given the same z-bin and
   tau-bin". This filters out near-break-even trades from genuinely
   profitable ones, while keeping the binary structure XGBoost handles
   well.

2. **Tweedie regression.** PnL/$ has a point mass at -1 and a continuous
   positive tail — that's a Tweedie distribution (compound Poisson-Gamma).
   `xgb.XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.5)`
   handles this directly and should fit better than squared error.

3. **Quantile regression.** Predict the 25th/50th/75th percentiles of
   PnL/$ rather than the mean. The 25th percentile (worst-case-typical)
   is what we'd actually want for risk-aware sizing.

4. **Cost-sensitive classification.** Train the existing classifier with
   sample weights = abs(PnL/$). Trades that matter more for total PnL
   get more weight in the loss function.

None of these are urgent. P10.1 (size_mult on the classification model)
already gives a measurable +Sharpe on btc 15m. The marginal upside of
further filtration tuning is small compared to the cost of the
experiments.
