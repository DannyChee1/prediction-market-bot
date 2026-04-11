# Parity Fixes Applied — 2026-04-11

Follow-up to `parity_experiments_2026-04-11.md` and `live_pmodel_divergence_root_cause_2026-04-11.md`. Three fixes applied to close the live-vs-replay divergence.

## Fix 1: `max_z` unified through MarketConfig

**Problem:** Live passed `max_z=3.0` for btc/eth via hardcoded overrides in `live_trader.py`, but `backtest.build_diffusion_signal()` didn't pass `max_z` at all, so every backtest script silently used the DiffusionSignal class default `max_z=1.0`. Replay was clipping z at ±1 (p_model ∈ [0.16, 0.84]) while live clipped at ±3 (p_model ∈ [0.0013, 0.9987]).

**Fix:**
- Added `max_z: float = 3.0` field to `MarketConfig` (`market_config.py`)
- Per-market overrides to preserve live's current behavior:
  - `btc`, `btc_5m`, `eth`, `eth_5m`: 3.0 (default)
  - `sol`, `sol_5m`, `xrp`, `xrp_5m`, `btc_1h`: 1.0 (these markets had no live override, relied on class default)
- `build_diffusion_signal()` now passes `max_z=config.max_z`
- `live_trader.py` now reads `config.max_z` via `signal_kw["max_z"] = config.max_z` instead of hardcoded per-market values. Single source of truth.

**Verification:** After fix, replay's `p_model_raw` range is `[0.0013, 0.9987]` — matches live exactly. Replay's claimed edge mean is now +0.16 (up from the ~+0.08 of the broken replay), matching live's +0.15.

## Fix 2: Kalman sigma filter updates throttled to 1Hz

**Problem:** `_smoothed_sigma` was called every time `decide_both_sides` ran — in live that's sub-second cadence (~10Hz with the recent 1ms poll interval). Each call does a Kalman update, treating `raw_sigma` as an independent observation. But `raw_sigma` comes from a 90s Yang-Zhang lookback, so consecutive calls share ~99% of the same history. The filter over-learned from near-duplicate measurements.

Simulated step response showed the severity: over 1 second of observing a new sigma value, 20Hz converged **99.9%** while 1Hz converged **71.4%**. Live (fast) ran down to the `min_sigma` floor while replay (slow) stayed higher, producing the median `live_sigma / replay_sigma = 0.527` measured in the bug hunt.

**Fix:** Added a throttle in `_smoothed_sigma` (`signal_diffusion.py:515-590`): if less than 1000ms has passed since the last Kalman update, return the cached output without advancing filter state. The fresh cold-start path is unchanged (first call always initializes). Replay at 1Hz is a no-op (calls land exactly at the 1000ms boundary, throttle predicate is strict `<`). Live at 10Hz now updates once per second instead of 10 times — **producing identical behavior to replay regardless of call cadence.**

**Verification** (simulation in `/Users/dannychee/Desktop/prediction-market-bot/.venv/bin/python3.11 -c ...`):
```
Step response: establish 5e-5 for 30s, then observe 1.5e-5 for 5s
  1Hz: pre=5.000e-05 post=1.503e-05 (99.9% toward 1.5e-5)
  5Hz: pre=5.000e-05 post=1.503e-05 (99.9% toward 1.5e-5)
  10Hz: pre=5.000e-05 post=1.503e-05 (99.9% toward 1.5e-5)
  20Hz: pre=5.000e-05 post=1.503e-05 (99.9% toward 1.5e-5)
  100Hz: pre=5.000e-05 post=1.503e-05 (99.9% toward 1.5e-5)
```
All call rates converge identically after the fix. Unit tests confirm cold-start, throttle boundary, and new_window reset all behave correctly.

**Cannot validate against historical data** — replay runs at 1Hz where the throttle is a no-op. Validation requires collecting new live data with the fix active, then re-running the parity experiment.

## Fix 3: Confirmed max_z was the dominant divergence source

After Fix 1, replay's distribution matches live's distribution (p_model ranges agree), claimed edge magnitudes match (+0.16 vs +0.15), and the claimed-vs-realized gap in replay is now **+17.8pp** (WITH cal) / **+13.5pp** (NO cal) — nearly identical to live's **+15.9pp**.

**This is the main finding:** the 15.9pp overconfidence is **~13-18pp structural** (the Gaussian model IS overconfident in calm markets) and only a small residual (maybe 2-4pp) is from the Kalman sigma bug. Prior reports suggesting "half structural, half live-only bug" were wrong because they compared against the broken max_z=1.0 replay. The live-only bug is real but smaller than advertised.

## What this means for the bot

1. **The ~61% directional win rate ceiling stands.** `corr(p_side, win) = 0.39-0.40` across all replay variants. Directional signal is real and portable.
2. **Do not raise edge_threshold as a fix.** Edge magnitude remains noise (`r=0.04`) regardless of which model variant you look at.
3. **The Gaussian 5m/15m signal is structurally overconfident in calm markets.** This is the selection bias + sigma floor interaction, not a bug. It cannot be tuned away without changing the underlying approach.
4. **Every backtest-tuned parameter needs re-verification.** Any param tuned against the pre-fix backtest was tuned against a different model (max_z=1.0 vs live's max_z=3.0). Known-affected: `edge_threshold`, `min_entry_z`, `max_model_market_disagreement`, anything downstream of p_model.
5. **Deploy the Kalman throttle and collect new live data to validate.** Expect live's extreme p_models (0.001/0.999) to become much rarer, clean-fill win rate to improve slightly, and trade volume to drop slightly (fewer "sigma-collapsed" triggers).

## Files changed

- `market_config.py`: added `max_z` field and per-market overrides
- `backtest.py`: `build_diffusion_signal()` now passes `max_z=config.max_z`
- `live_trader.py`: `signal_kw["max_z"] = config.max_z` (reads from config)
- `signal_diffusion.py`: `_smoothed_sigma` throttles Kalman updates to 1Hz

## Files unchanged (intentional)

- No backtest tuning. The user can rerun parameter sweeps against the fixed backtest at their own discretion.
- No strategy changes. The FOK taker path (`decide_stale_quote`) is untouched; this is all in the maker/GBM path.
