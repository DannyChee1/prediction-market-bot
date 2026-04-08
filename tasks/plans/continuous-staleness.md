# F9 — Continuous staleness penalty (replace hard gates)

## Current behavior

Stale-feature gates are binary skips:
```python
if book_age_ms > self.max_book_age_ms:
    return Decision("FLAT", ...)
```

`max_book_age_ms = 1000` for btc_5m. At 999ms we trade full size; at
1001ms we skip entirely. That's wasteful at the boundary and doesn't
distinguish "1.5s stale" from "60s disconnected."

## Idea

Replace the hard gate with a continuous penalty on either:
- **Sigma inflation**: `sigma_per_s *= 1 + age_ms / max_age_ms`
  (signal auto-dampens when data is late)
- **Edge penalty**: `edge -= k * log(1 + age_ms / ref_ms)`
- **Kelly multiplier**: `kelly_fraction *= max(0, 1 - age_ms / max_age_ms)`

Soft transitions, no edge cliffs.

## Variant comparison

| Approach | Pros | Cons |
|---|---|---|
| Sigma inflation | model becomes naturally less confident | might not be enough during real disconnects |
| Edge penalty | linear, easy to tune | doesn't reduce sizing aggressively enough |
| Kelly multiplier | aggressive sizing reduction | still trades during disconnects (just smaller) |

Recommend a HYBRID: Kelly multiplier (gradual reduction) +  retain a
hard cap at 5× max_age_ms (refuse to trade during real disconnects).

## Concrete change

```python
def _staleness_size_multiplier(self, ctx: dict) -> float:
    """Smooth Kelly multiplier from feed staleness."""
    book_age = ctx.get("_book_age_ms")
    if book_age is None or self.max_book_age_ms is None:
        return 1.0
    if book_age > 5 * self.max_book_age_ms:
        return 0.0  # hard cap
    if book_age <= self.max_book_age_ms:
        return 1.0
    # Linear from full size at threshold to zero at 5x threshold
    excess = (book_age - self.max_book_age_ms) / (4 * self.max_book_age_ms)
    return float(max(0.0, 1.0 - excess))
```

Apply at the Kelly step alongside `regime_mult`, `filt_mult`,
`hawkes_mult` (F7).

## Success criteria

Backtest A/B (50d, walk-forward 70/30):
- Trade count INCREASES (capturing trades at the boundary that the
  hard gate currently skips)
- Sharpe stays within ±0.05 of baseline
- Max DD doesn't increase

## Effort

~2 hours.

## ROI

Modest. Backtest never populates `_book_age_ms` so this is
**live-only** behavior change. Real impact only measurable in live
A/B (parallel run with old gates vs new penalty).

Most likely to help during periods when book WS is intermittently
slow (50% of the time it's fine, 50% it's at 1.2-2x threshold). The
hard gate currently throws away half the trading windows in those
periods; the penalty would still trade them at smaller size.
