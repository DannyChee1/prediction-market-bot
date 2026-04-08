# F4 — Oracle lead-lag as a profit signal

## Motivation

The bot currently uses the Binance↔Chainlink **price gap** as a safety
widener (`oracle_lag_threshold=0.002`, `oracle_lag_mult=2.0` widen
`dyn_threshold` proportionally). It's purely defensive — when the two
disagree, require more edge to enter.

But the gap is also **predictive**. Chainlink Data Streams update
roughly every minute (with low-noise pushes faster than that under
volatility). Binance updates many times per second. When Binance has
moved 0.3% but Chainlink hasn't propagated yet, the next Chainlink
push will likely close the gap toward Binance — that's a fact about
how the oracle works, not a model assumption.

If we trade UP/DOWN based on the *expected* future Chainlink price
(after the oracle catches up to Binance) rather than the current
Chainlink, we get a small but real edge. The Polymarket window
resolves on Chainlink, so the relevant question is "where will
Chainlink be at window end" — and Binance is the best predictor of
where Chainlink is heading.

## Hypothesis

When `binance_mid > chainlink_price` by more than X% AND the gap has
persisted for Y seconds (not just a flash), the next chainlink update
is more likely UP than DOWN. Conversely for negative gaps.

The bot should:
1. Convert this directional belief into a small p_model bias.
2. Or use it as an additional Kelly multiplier (boost size when the
   model and the lead-lag agree, shrink when they disagree).

## Design sketch

Three increasingly complex variants:

### Variant A — gap-as-feature (safest, lowest reward)

Add `oracle_lag_signed` (signed binance − chainlink, normalized) as a
filtration model feature. Let the model learn whether high gap →
better trade. Requires retraining filtration, which would also need
F1 (instrumentation) for honest p99 measurement.

### Variant B — gap-as-bias (medium reward, medium risk)

Modify p_model directly:
```
gap_signed = (binance_mid - chainlink_price) / chainlink_price
p_model_biased = p_model + alpha * clip(gap_signed / threshold, -1, 1)
```
where `alpha ~ 0.05`. Threshold = `oracle_lag_threshold` (already 0.002).

This says: a 0.2% positive gap shifts our P(UP) belief by +5pp.
Half-Kelly sizing absorbs this naturally; over-confident bets get
clipped by the existing `max_z` cap and `market_blend`.

### Variant C — gap-as-direction (highest reward, highest risk)

Use the gap as an INDEPENDENT signal. When |gap| > threshold AND signal
direction agrees with gap direction → boost Kelly. When they disagree
→ skip.

```
if gap > oracle_lag_threshold and signal_says_up:
    kelly_mult *= 1 + boost
elif gap > oracle_lag_threshold and signal_says_down:
    return FLAT  # gap and signal disagree → skip
elif gap < -oracle_lag_threshold and signal_says_down:
    kelly_mult *= 1 + boost
else:
    return FLAT
```

This is more aggressive and would need backtest validation.

## Prerequisites

- **F1 (feed instrumentation)** — without honest measurement of how
  often the gap actually predicts a chainlink move (and by how much),
  we're guessing. The first thing this plan needs is a measurement
  campaign: log `(binance_event_ts, binance_mid, chainlink_event_ts,
  chainlink_price)` for 48h, compute the empirical lead-lag
  cross-correlation, then decide which variant to ship.
- The gap_signed feature already exists in `_compute_oracle_lag` (it's
  computed but only the abs value is used). One-line change to expose
  the signed value.

## Success criteria

Backtest A/B (50d, walk-forward 70/30):
- Sharpe improvement ≥ +0.05 on btc_5m OR btc 15m
- No worse than -0.02 Sharpe on the other market
- Drawdown not increased

If neither variant beats the baseline by ≥ +0.05 Sharpe, document as
negative result and move on.

## Risks

1. **Look-ahead bug**: easy to accidentally use future Chainlink price
   in feature construction. Backtest pipeline must compute the gap
   from data available AT decision time only.
2. **Non-stationarity**: Chainlink update cadence has changed in the
   past (analysis2.md mentions). The lead-lag relationship may not be
   stable across regimes.
3. **Conflicts with market_blend**: market_blend already pulls p_model
   toward contract mid, which already absorbs some of the
   binance-leading-chainlink information indirectly. The lead-lag bias
   may double-count.

## Estimated effort

- Variant A: ~4 hours (extract feature, retrain, A/B)
- Variant B: ~2 hours (one-line p_model bias, A/B)
- Variant C: ~3 hours (gate logic + A/B)
- Plus instrumentation prerequisite: ~6 hours

## Estimated ROI

Hard to predict without measurement. Hand-wave: if the gap leads
chainlink by ~50% of the time and the average informational content
is ~5% directional accuracy, that's a +1pp win-rate boost = ~+$0.05
EV per trade × 50 trades/day = ~$2.50/day on the current $40 bankroll.
Scales with bankroll. Not life-changing but real.

The biggest unknown is whether market_blend already captures most of
this. Need F1 to disentangle.
