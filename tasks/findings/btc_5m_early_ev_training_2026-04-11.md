# BTC 5m Early EV Training — 2026-04-11

## Goal

Build a dedicated **btc_5m early-tau** trainer on **full live windows
only**, and compare targets that better match the real decision:

> "Are we buying a contract cheaper than it was actually worth?"

This is narrower than the generic v2 filtration trainer:

- market: `btc_5m` only
- windows: full live only (`filter_live`, so no backfill, no partials)
- taus: early only
- labels: tied to the **contemplated GBM trade**, not generic window-UP

## New script

Created:

- `train_filtration_btc5_early.py`

It:

1. samples only full live `btc_5m` windows
2. defaults to early taus `240,180`
3. computes the raw GBM `p_gbm` at the checkpoint
4. picks the contemplated trade the live signal would prefer:
   - `edge_up = p_gbm - (ask_up + fee_up)`
   - `edge_down = (1-p_gbm) - (ask_down + fee_down)`
   - choose the larger edge
5. trains on that contemplated trade only

### Targets tested

1. `direction`
   Binary target: did the contemplated trade resolve in the money?

2. `value_weighted_direction`
   Same binary target, but XGBoost loss weighted by `abs(realized edge/share)`.

3. `edge_share`
   Regression target:

   `realized_edge_share = resolved_value - ask - fee`

   This is the cleanest "cheapness" target because it directly measures
   how much value per share we captured, and avoids the heavy-tailed
   `PnL/$` label from the older regression experiment.

## Why `edge_share` instead of `PnL/$`

The previous repo regression experiment used:

`PnL/$ = (resolved_value - ask - fee) / ask`

That target is noisy and heteroskedastic:

- same correct trade can be `+0.25x` or `+4.0x` depending on price
- wrong trades are clustered near `-1.0`
- MSE regresses to the mean and collapses the tails

`edge_share` is better aligned and better behaved:

- bounded in roughly `[-1, +1]`
- directly answers "did we buy cheaper than worth?"
- not exploded by dividing through a tiny ask

## Dataset

Full-window-only `btc_5m` live windows:

- `565` windows kept
- default early taus `[240, 180]`
- extracted contemplated-trade rows: `991`

For `tau=240` only:

- extracted rows: `501`
- train/test split: `400 / 101`

## Results

### A. Early taus `[240, 180]`

Validation set characteristics:

- raw GBM edge mean: `+0.2466`
- realized edge/share mean: `+0.2760`
- realized PnL/$ mean: `+0.6080`
- contemplated-side hit rate: `79.4%`

This held-out tail was unusually favorable, so absolute numbers are
optimistic. The comparison across targets is still informative.

#### 1. `direction`

- AUC: `0.7657`
- logloss: `0.4811`
- top score bucket actual edge/share: `+0.1089`
- overall validation edge/share: `+0.2760`

Verdict:

- good at ranking **hit probability**
- bad at ranking **value**
- highest-confidence bucket was worse than the average validation trade

That means "predict the winner" is the wrong objective for this use case.

#### 2. `edge_share`

- RMSE: `0.4173`
- corr(pred, actual edge/share): `0.0859`
- top score bucket actual edge/share: `+0.3928`
- bottom score bucket actual edge/share: `+0.1525`
- overall validation edge/share: `+0.2760`

Verdict:

- noisy, but materially better aligned with the objective
- top bucket had the best realized edge/share
- bottom bucket was clearly worse than average

The deciles were not monotonic, so this is **not** ready for continuous
sizing. But it is the best of the tested targets for ranking value.

### B. `tau=240` only

This is the cleaner early decision point for btc_5m.

Validation set characteristics:

- raw GBM edge mean: `+0.2593`
- realized edge/share mean: `+0.2735`
- realized PnL/$ mean: `+0.5879`
- contemplated-side hit rate: `79.2%`

Again, this tail is favorable, so absolute levels are optimistic.

#### 1. `direction`

- AUC: `0.7500`
- top score bucket actual edge/share: `+0.2482`
- overall validation edge/share: `+0.2735`
- bottom bucket actual edge/share: `+0.1063`

Verdict:

- still not a value-ranking model
- top bucket is not actually the best edge bucket

#### 2. `value_weighted_direction`

- AUC: `0.7292`
- top score bucket actual edge/share: `+0.3113`
- overall validation edge/share: `+0.2735`
- bottom bucket actual edge/share: `+0.2256`

Verdict:

- better aligned than plain direction
- still only modest spread between top and bottom buckets
- useful hybrid, but not the strongest value sorter

#### 3. `edge_share`

- RMSE: `0.4039`
- corr(pred, actual edge/share): `0.0742`
- top score bucket actual edge/share: `+0.3976`
- overall validation edge/share: `+0.2735`
- bottom bucket actual edge/share: `+0.1882`

Verdict:

- best match to the desired objective
- top bucket clearly better than the average trade
- still noisy / not monotonic, so use as a coarse ranker, not a smooth EV curve

## Interpretation

1. **The plain direction target is wrong for this problem.**
   It can predict winners, but the highest-probability trades are not the
   highest-value trades.

2. **The best target tested is `edge_share`, not win/loss.**
   That is the closest approximation to "buying cheaper than actually worth."

3. **The edge-share regressor is only a coarse filter right now.**
   It separates strong from weak trades somewhat, but not smoothly enough
   for continuous sizing.

4. **Value-weighted classification is a reasonable fallback.**
   If the regression target proves too unstable out of sample, this is the
   best compromise tested so far.

5. **The held-out slice itself was unusually strong.**
   A `79%` contemplated-trade hit rate and `+0.27` edge/share mean on the
   validation tail means this period is easy. The ranking conclusions are
   useful; the absolute numbers are optimistic.

## Recommendation

### Best next step

Use the new trainer with:

- market: `btc_5m`
- tau: `240`
- target: `edge_share`

But use it as a **rank-and-gate** model, not a sizing curve.

Meaning:

- score contemplated trades
- trade only the top bucket / top quantile / score above a threshold
- do **not** map the raw regressor output linearly into Kelly size yet

### Why not use plain direction?

Because it optimizes hit rate, not cheapness. In the experiments above,
its highest-score bucket was not the highest-value bucket.

### Why not jump to NN?

Still the wrong move. Sample is far too small, and the tabular tree model
already picks up meaningful structure. The remaining issue is target
formulation and stability, not model depth.

## Concrete next experiments

1. **Replay/backtest the `tau=240 edge_share` model as a hard gate**
   - trade only if predicted score is above a threshold
   - compare realized PnL, edge/share, and trade count vs the raw GBM trader

2. **Sweep score thresholds / top-quantile gates**
   - top 10%
   - top 20%
   - score > 0.20
   - score > 0.30

3. **Compare against `value_weighted_direction`**
   - if regression ranking is unstable across windows, weighted
     classification may generalize better

4. **Do not use the current regression output for continuous Kelly sizing**
   - deciles are not monotonic enough
   - use it as a yes/no gate first

## Commands run

```bash
.venv/bin/python -u train_filtration_btc5_early.py --target direction
.venv/bin/python -u train_filtration_btc5_early.py --target edge_share
.venv/bin/python -u train_filtration_btc5_early.py --target direction --taus 240
.venv/bin/python -u train_filtration_btc5_early.py --target edge_share --taus 240
.venv/bin/python -u train_filtration_btc5_early.py --target value_weighted_direction --taus 240
```
