# F7 — Hawkes intensity as sizing modifier

## Background

P8.3 (commit 46d7f8c) tried adding Hawkes self-exciting jump
intensity as filtration MODEL features. Negative result on both
markets — the feature ranked #4 in XGBoost importance but the
model overfit at the top decile and the gate filtered the wrong
trades.

What's left from that work:
- `MarketConfig.hawkes_params` populated for both BTC markets
- `_maybe_publish_hawkes` writes `_hawkes_intensity` and
  `_hawkes_n_events` to ctx every tick in both `decide()` and
  `decide_both_sides()`
- Per-market parameters fitted via `/tmp/fit_hawkes.py` at
  k_sigma=3.0 (btc_5m: branching 0.40, half-life 13.9s; btc 15m:
  branching 0.60)

The infrastructure is wired. Just no consumer.

## Hypothesis

Use `_hawkes_intensity` as a Kelly multiplier instead of a feature.

When Hawkes intensity is HIGH (recent jump cluster), the market is in
an unstable regime — recent moves are more likely to continue/reverse
chaotically. Reduce position size during these moments.

When Hawkes intensity is LOW (calm regime), the diffusion model is
more reliable. Use full Kelly.

## Concrete change

Add a multiplier function similar to `_filtration_size_multiplier`:
```python
def _hawkes_size_multiplier(self, ctx: dict) -> float:
    """Dampen Kelly sizing when jump intensity is elevated."""
    if self.hawkes_params is None:
        return 1.0
    intensity = ctx.get("_hawkes_intensity")
    if intensity is None or intensity <= 0:
        return 1.0
    mu, alpha, beta, _ = self.hawkes_params
    # Steady-state intensity: mu / (1 - alpha/beta)
    steady = mu / max(1e-6, 1.0 - alpha / beta)
    excess_ratio = intensity / steady
    # Dampen smoothly: at steady-state → 1.0, at 3x steady → 0.5,
    # at 5x steady → 0.3
    if excess_ratio <= 1.0:
        return 1.0
    return float(max(0.2, 1.0 / (1.0 + 0.5 * (excess_ratio - 1.0))))
```

Apply it in the Kelly step alongside `regime_mult` and `filt_mult`:
```python
hawkes_mult = self._hawkes_size_multiplier(ctx)
kelly_fraction_adj = self.kelly_fraction * regime_mult * filt_mult * hawkes_mult
```

## Success criteria

Backtest A/B (50d, walk-forward 70/30, seed 42):
- Sharpe ≥ baseline on btc_5m AND btc 15m (no regression)
- Drawdown reduced by ≥ 0.5pp on at least one market (the whole
  point is to dampen exposure during volatility clusters)
- PnL within ±$500 of baseline (small loss is acceptable for the
  drawdown reduction)

## Risks

- **Same lesson as P8.3**: maybe the markets we backtest on don't
  have enough cluster events for this to matter. The fitted
  branching ratios (0.4 / 0.6) suggest moderate clustering — not
  rare, not constant.
- **Conflicts with regime_classifier and vol_regime gate**: both
  already do something similar (downsize during high-vol periods).
  May be triple-counting volatility. Need to verify the multipliers
  don't compound to zero size.

## Estimated effort

~3 hours (helper function, plumb in both decide paths, A/B test).

## Estimated ROI

Small. The drawdown reduction is the main goal, not Sharpe. If the
backtest shows -1pp DD with neutral PnL, ship it. Otherwise document
as second negative result and remove the Hawkes infra entirely.

## Decision criterion

If F7 fails too, **delete the Hawkes infrastructure**. Two negative
results from the same feature set means the dataset doesn't support
it. Removing dead code is more valuable than keeping speculative
plumbing.
