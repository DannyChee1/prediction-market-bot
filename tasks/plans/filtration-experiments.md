# F5 + F6 — Filtration model variants

## Background

P10.1 (commit 6aed181) made size_mult mode the default — confidence
from the XGBoost classifier scales Kelly instead of being a binary
gate. Ships ≥ baseline on btc_5m, beats baseline on btc 15m.

P10.3 (commit 8145b11) tried regression on PnL/$ as the target. Lost
on both markets. Negative result, infrastructure kept.

These follow-ups try smarter target formulations for the same
filtration model.

## F5 — Tweedie regression

### Why retry regression at all

Past PnL/$ regression failed because the target distribution is
**bimodal** (lose ~1.0, or win 0.5..2.0). MSE regression collapsed
to predicting the mean (~+0.03), with std 0.107 vs label std 0.574 —
5× compression. The model couldn't discriminate.

Tweedie is the right family for this distribution — it's a compound
Poisson-Gamma that explicitly handles a point mass at one value
(losses) plus a continuous positive tail (wins). XGBoost has native
support: `objective="reg:tweedie"`.

### Concrete change

`train_filtration.py` already has `--target regression` from P10.3.
Add `--objective {squarederror, tweedie}` flag. When tweedie:
```python
model = xgb.XGBRegressor(
    objective="reg:tweedie",
    tweedie_variance_power=1.5,  # 1.0 = Poisson, 2.0 = Gamma; 1.5 = compound
    ...
)
```

### Success criteria

- Prediction std at least 50% of label std (vs 5× compression in P10.3)
- Decile 10 actual_pnl > decile 9 actual_pnl (no overfit at the top)
- Backtest Sharpe ≥ classification baseline on btc_5m AND btc 15m

### Estimated effort

~3 hours (one flag, retrain, A/B).

### ROI

Small — bounded by how much information the FEATURES contain about
EV. Tweedie can't extract signal that isn't there. But the
class-balance issue that killed plain regression should be fixable,
so this is worth one shot.

---

## F6 — Cost-sensitive classification (sample weights)

### Idea

Keep the existing classifier (predicts P(direction correct)), but
weight samples by `abs(realized_pnl)` during training. Trades that
matter more for total PnL get more weight in the loss function.

### Why it might work

The current classifier optimizes log-loss equally over all trades.
But a wrong call on a $10 trade matters 5× more than a wrong call on
a $2 trade. Weighting by realized PnL aligns the training objective
with portfolio PnL more directly than plain accuracy.

### Concrete change

In `train_filtration.py:build_dataset`, add a `weight` field to each
row equal to `abs(realized_pnl_per_dollar)`. Pass `sample_weight=`
to `model.fit()`.

```python
weights = np.array([r["weight"] for r in train_rows], dtype=np.float32)
model.fit(X_train, y_train, sample_weight=weights, verbose=False)
```

The realized PnL is already computed in P10.3's regression branch —
just reuse that calculation in classification mode.

### Success criteria

- AUC unchanged or slightly worse (expected — accuracy isn't the
  optimization target anymore)
- Backtest Sharpe ≥ classification baseline by ≥ +0.02 on at least
  one BTC market
- No worse than -0.05 Sharpe on the other market

### Estimated effort

~2 hours (same as F5, one branch in train_filtration.py).

### ROI

Small-medium. Probably the highest-ROI filtration variant left to
try, because it directly addresses the criticism in
`tasks/findings/hawkes_filtration_2026-04-07.md` — "classification
metrics don't predict portfolio metrics." Sample-weighting is the
canonical fix.

---

## Don't try (already known to fail)

- Plain MSE regression on PnL/$ (P10.3, commit 8145b11)
- Hawkes intensity as filtration feature (commit 46d7f8c)
- Higher filtration_threshold to filter more (P10.1 swept it,
  marginal trades are profitable)

## Order to try

1. F6 (cost-sensitive classification) — highest expected value, lowest effort
2. F5 (Tweedie regression) — backup if F6 doesn't beat baseline

If both fail, accept that the current classification + size_mult is
the best we can do with the current feature set, and focus on adding
new features instead (Hawkes-as-sizing in F7, oracle lead-lag in F4).
