# Hawkes Self-Exciting Jump Intensity → Filtration Feature — 2026-04-07

## Hypothesis

`scripts/hawkes.py` (built earlier under Quant Guild #94) provides a univariate
exponential Hawkes process for modeling jump clustering in returns. The infrastructure
already published `_hawkes_intensity` and `_hawkes_n_events` to ctx as opt-in
features, but no downstream consumer was reading them. The natural place to put
them is the XGBoost filtration model, which already pulls features from ctx.

**Hypothesis:** Surfacing Hawkes intensity to the filtration model would let it
recognize "we just had a cluster of jumps, the next 10-20s of signal is unreliable"
and filter those trades out, improving Sharpe.

## What I shipped

1. **Hawkes parameter fits per market** via `/tmp/fit_hawkes.py`:
   - Walks 1000 parquets per market
   - Computes per-window σ via bipower variation (jump-robust)
   - `detect_jumps(prices, ts, sigma, k_sigma=3.0)` for jump events
   - Concatenates events with cumulative time offset, fits via grid-search MLE
   - Results (k=3.0σ, both stationary):
     - **btc_5m**: μ=0.0237, α=0.02, β=0.05, branching ratio 0.40, half-life 13.9s
     - **btc 15m**: μ=0.0116, α=0.03, β=0.05, branching ratio 0.60, half-life 13.9s

2. **`hawkes_params` field in `MarketConfig`** + populated for both markets.

3. **Plumbed `config.hawkes_params` → `DiffusionSignal` via `build_diffusion_signal`**.
   The signal already had `_maybe_publish_hawkes()` written; this just turns it on.

4. **`decide_both_sides` parity fix**: it was missing the `_maybe_publish_hawkes`
   call that `decide()` had. Maker mode would have seen `_hawkes_intensity=0`
   even after the rest of the wiring was in. Now both paths publish.

5. **Filtration features** (`extract_features` + `FEATURE_NAMES`): added 4 hawkes
   features:
   - `hawkes_intensity` (current λ(t))
   - `log_hawkes_intensity` (log scale)
   - `hawkes_n_events` (cumulative jumps in window)
   - `hawkes_intensity_x_z_abs` (interaction)

6. **Retrained filtration** on 51,805 samples (14k BTC 5m + 4.7k BTC 15m + small
   ETH/SOL/XRP). Training pipeline (`train_filtration.py`) computed Hawkes features
   for each tau-checkpoint row by simulating the live signal's "seed" path:
   spawn a fresh `HawkesIntensity`, call `detect_jumps` on `prices[:best_idx+1]`,
   add events, snapshot `intensity_at(ts[best_idx])`.

## Offline training results

The retrained model showed Hawkes IS a discriminative feature:

- **AUC**: 0.7574 (vs 0.7671 for the pre-hawkes baseline — slightly down)
- **Per-tau lift improved**:
  - tau=750s: +1.0pp (was ~+1pp)
  - tau=600s: +1.4pp
  - tau=450s: +1.0pp
  - tau=300s: +1.2pp
  - tau=150s: **+3.1pp** (vs ~+1pp before)
  - tau=60s: **+2.2pp**
- **`hawkes_n_events` ranked #4** in feature importance (0.054), beating
  `log_sigma`, `vol_regime_ratio`, `avg_spread`, and `tau`. So the XGBoost
  model definitely uses the signal.

By all the offline classification metrics, this looked like a win.

## Walk-forward backtest A/B (THE FAIL)

| Market | filtration | Trades (test) | WR | PnL | Sharpe | MaxDD |
|---|---|---:|---:|---:|---:|---:|
| btc_5m | WITH hawkes | 3529 | 62.3% | $26,144 | 1.50 | 4.0% |
| btc_5m | WITHOUT     | 3598 | 62.2% | $27,308 | 1.54 | 3.9% |
| btc 15m| WITH hawkes |  612 | 59.2% |  $3,715 | 1.21 | 5.8% |
| btc 15m| WITHOUT     |  609 | 59.3% |  $3,834 | 1.26 | 5.8% |

Δ (WITH minus WITHOUT):

- btc_5m: −69 trades (−1.9%), **−$1,164 PnL (−4.3%)**, −0.04 Sharpe
- btc 15m: +3 trades, **−$119 PnL (−3.1%)**, −0.05 Sharpe

For comparison, the pre-Hawkes filtration model was a slightly weaker no-op:

- btc_5m old filt vs no filt: −22 trades, −$711 PnL, −0.04 Sharpe
- btc 15m old filt vs no filt: −5 trades, +$13 PnL, +0.02 Sharpe

So the new (hawkes-aware) model is **MORE selective AND SLIGHTLY WORSE** on PnL
than both no-filtration and the old filtration.

## Why the offline win didn't translate

1. **Classification metrics ≠ portfolio metrics.** AUC and per-tau lift measure
   "would the filtered trades have been correct?" — but the trades the model
   filters out can still be EV-positive even when their *classification* is wrong.
   At our 53%+ break-even win rate (after fees), the filter needs to filter
   trades whose true win rate is *below* 53%, not just below the model's
   confidence threshold.

2. **Hawkes intensity might be a confound.** Recent jumps mean recent volatility
   spikes — and our other vol-regime features (`vol_regime_ratio`, `log_sigma`)
   already pick those up. The XGBoost model uses `hawkes_n_events` because it's
   slightly more discriminative for the *classification* task, but in the actual
   trade set this just filters out vol-regime trades that we already trade
   profitably with the existing edge_threshold and Kelly sizing.

3. **The gate is filtering the wrong trades.** Even with much better lift at
   short tau (where most trades happen), the actual PnL goes down. This is the
   classic story: "the model can predict win-rate well, but the trades it
   identifies as low-win-rate happen to be the ones with the best risk-adjusted
   payouts."

## What I shipped vs. reverted

| | Status |
|---|---|
| `scripts/hawkes.py` | **Kept** (was already shipped) |
| `MarketConfig.hawkes_params` field | **Kept** |
| BTC 5m / BTC 15m hawkes_params populated | **Kept** |
| `build_diffusion_signal` plumbs `config.hawkes_params` | **Kept** |
| `_maybe_publish_hawkes` call in `decide_both_sides` (parity fix) | **Kept** |
| `extract_features(hawkes_intensity=, hawkes_n_events=)` kwargs | **Kept** (silently ignored) |
| `FEATURE_NAMES` hawkes entries | **Reverted** |
| `extract_features` actually using hawkes args | **Reverted** |
| `train_filtration.py` Hawkes feature computation | **Reverted** |
| New `filtration_model.pkl` (33-feature, hawkes-aware) | **Saved as `.with_hawkes_features`**, NOT default |
| `filtration_model.pkl` | **Restored from `.pre_hawkes` backup** |

The kept items mean:
- `_hawkes_intensity` / `_hawkes_n_events` are now published to ctx in BOTH
  `decide()` and `decide_both_sides()` for both BTC markets (live + backtest).
- A future downstream consumer (a sizing modifier, a regime gate, a different
  ML model) can read those ctx fields with no further plumbing.
- `extract_features()` accepts the kwargs but ignores them, so the alternate
  33-feature model can be loaded by callers that opt in.

## Next experiments worth trying (if revisiting)

1. **Hawkes as a sizing modifier, not a gate.** Halve Kelly when
   `hawkes_intensity > steady-state * 1.5`. Doesn't filter trades, just shrinks
   exposure during clustering episodes.

2. **Hawkes-conditional edge threshold.** `dyn_threshold *= 1.0 + α·excess_intensity`
   where excess_intensity is normalized λ above steady-state. Same intuition as
   the toxicity / VPIN excess paths already in `decide()`.

3. **Two-market Hawkes coupling.** Detect jumps on the *other* timeframe (BTC 5m
   gets Hawkes from BTC 15m and vice versa) — cross-timeframe self-excitation.

4. **MLE refinement instead of grid search.** The `fit_hawkes_mle` function uses
   coarse grids; an L-BFGS or Nelder-Mead refinement step would give better
   parameter estimates. Minor — both fits already landed in stable interior
   regions of the grid at k=3.0σ.

None of these are urgent. The main lesson is that **classification AUC for
filtration features doesn't predict portfolio PnL** — you need the full
backtest A/B before believing any new filtration feature.
