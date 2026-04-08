# F8 — Z-confirmation rate limit (second-layer spike defense)

## Status

**Likely not needed** — measurement shows zero p_model spikes in the
post-P9.2 sigma-floor era. Keep this as a fallback only if the
operator observes a NEW kind of spike that the existing defenses
miss.

## Existing defenses (already in place)

1. **P9.2 adaptive sigma floor** (`backtest.py:MIN_SIGMA_RATIO=0.5`):
   the short-window σ cannot drop below 50% of the 300s baseline.
   Prevents `z = delta/(σ·√τ)` from blowing up when σ briefly
   collapses during quiet sub-periods.
2. **`max_z` cap** (1.0 for btc_5m, 3.0 for btc 15m): clamps any
   z-score regardless of cause.
3. **Delta velocity check** (`signal_diffusion.py:1342`): rejects
   BUY_UP when the OLS slope of last 30s prices is negative, and
   vice versa. So even if a fast p_model spike says BUY_UP, if the
   underlying price is actually falling, the trade is blocked.

## Hypothetical second layer (only if F8 is needed)

Require the model's BUY decision to **persist for K consecutive
ticks** before firing the order. Single-tick spikes get filtered.

```python
# In tracker.py decision loop
if dec.action != "FLAT":
    if self._last_decision_action == dec.action:
        self._consecutive_decisions += 1
    else:
        self._consecutive_decisions = 1
    self._last_decision_action = dec.action
    if self._consecutive_decisions < self.confirm_ticks:
        continue  # not enough confirmation yet
```

With `confirm_ticks=3` and 50ms tick rate, requires 100ms of
sustained signal before a trade fires. Spike-trades are eliminated
by construction.

## When to ship

- Operator reports a new spike-trade that wasn't blocked by the
  existing 3 layers, OR
- Backtest shows the existing defenses miss real cases (run an
  audit on the diagnostic JSONL looking for any |Δp_model| > 0.20
  in 1 tick that resulted in a fill — if zero, F8 is not needed)

## Effort

~1 hour if needed (5 lines of code + a config flag + restart).
