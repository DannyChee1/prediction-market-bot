# Filtration v2 Live-Only Retrain — 2026-04-11

## Goal

Continue the `train_filtration_v2.py` work on the **real recorder data
only**, excluding:

- REST-backfilled parquets (`bid_depth5_up` all NaN)
- partial live parquets (joined mid-window)

The user explicitly wanted **full windows only**.

## Real-data cutoff

Tail scan of both directories shows the first contiguous full live
windows start at:

- `btc_5m`: `2026-04-07 16:45:00 UTC`
- `btc_15m`: `2026-04-07 16:45:00 UTC`

Everything before that boundary in the tail scan was backfilled. After
that boundary, files are a mix of full live and partial live windows.

## Dataset actually used

Full-window-only counts at retrain time:

| Market | Total parquets | Full live | Backfill | Partial |
|---|---:|---:|---:|---:|
| `btc_5m` | 15,230 | 565 | 14,300 | 365 |
| `btc_15m` | 5,090 | 220 | 4,771 | 99 |

The current trainer already matches the requested policy because it uses
`parquet_kind.filter_live(...)`, which keeps only `classify(...) == "live"`.

## Training setup

- Script: `train_filtration_v2.py`
- Features available: `90`
- Leakage features dropped: `8`
- Final model inputs: `82` features
- Sampling:
  - `btc_5m`: taus `[240, 180, 120, 60]`
  - `btc_15m`: taus `[840, 720, 600, 480, 360, 240, 120]`
- Split: chronological 80/20 walk-forward
- Learner: XGBoost classifier

Training rows extracted:

| Market | Rows |
|---|---:|
| `btc_5m` | 2,221 |
| `btc_15m` | 1,488 |
| **Total** | **3,709** |

## Leakage features removed

These are excluded from fit time:

- `elapsed_frac_x_signed_z`
- `dd_from_peak_z`
- `peak_delta_seen`
- `signed_z_range`
- `run_length_current`
- `n_zero_crossings`
- `n_zero_crossings_normalized`
- `n_trades_fired_already`

## Held-out results

### Overall

Raw XGBoost on the held-out tail:

- `AUC = 0.848`
- `logloss = 0.4267`
- `brier = 0.1350`

Train/test base-rate drift is still large:

- train UP rate: `48.9%`
- test UP rate: `24.5%`

The logistic calibration head was **worse** than raw XGB on the held-out
tail:

- raw logloss `0.4267` vs calibrated `0.6111`
- raw brier `0.1350` vs calibrated `0.1548`

So the ranking signal is useful, but the current calibration layer
should not be trusted as-is.

### By market

| Market | n | UP rate | AUC | Logloss | Brier |
|---|---:|---:|---:|---:|---:|
| `btc_15m` | 315 | 31.1% | 0.818 | 0.5286 | 0.1693 |
| `btc_5m` | 427 | 19.7% | 0.863 | 0.3516 | 0.1096 |

### By market × target tau

#### BTC 15m

| Tau | n | UP rate | AUC |
|---:|---:|---:|---:|
| 840 | 45 | 31.1% | 0.588 |
| 720 | 45 | 31.1% | 0.696 |
| 600 | 45 | 31.1% | 0.758 |
| 480 | 45 | 31.1% | 0.839 |
| 360 | 45 | 31.1% | 0.885 |
| 240 | 45 | 31.1% | 0.869 |
| 120 | 45 | 31.1% | 0.968 |

#### BTC 5m

| Tau | n | UP rate | AUC |
|---:|---:|---:|---:|
| 240 | 107 | 19.6% | 0.731 |
| 180 | 107 | 19.6% | 0.821 |
| 120 | 107 | 19.6% | 0.899 |
| 60 | 106 | 19.8% | 0.980 |

## Interpretation

1. **Dropping the explicit leakage block did NOT kill the useful early
   signal.**
   - `btc_5m @ tau=240`: `AUC 0.731`
   - `btc_15m @ tau=840`: `AUC 0.588`
   This is weaker than the blended headline but still above noise.

2. **The headline `0.848` AUC is still materially inflated by late-window
   samples.**
   The closer the market is to resolution, the more the path itself
   becomes nearly the answer:
   - `btc_5m @ tau=60`: `0.980`
   - `btc_15m @ tau=120`: `0.968`
   These rows are useful for research, but they should not drive the
   product decision.

3. **Microstructure is genuinely carrying the model.**
   Top features were:
   - `microprice_down`
   - `microprice_up`
   - `pm_n_levels_meaningful`
   - `pm_unique_makers_proxy`
   - `bn_ret_300s`
   - `pm_p_residual`

4. **The research synthesis conclusion still stands: stay with XGBoost,
   not NN-first.**
   With only `3,709` full-window training rows, this is still squarely a
   small-tabular-data regime where boosted trees are the right default.

5. **Calibration is now the main modeling problem.**
   The classifier ranks windows well, but the probability mapping drifts
   badly across time because the class balance shifted hard in the held-out
   tail.

## Recommendation

1. Keep the live-only, full-window-only policy.
2. Judge the model on **early tau**:
   - `btc_5m @ 240`
   - `btc_15m @ 840`
3. Do **not** switch to a neural net yet.
4. Next modeling step:
   - either save **raw XGB scores** as the gate signal, or
   - refit calibration with a proper time-split calibration slice
5. Only after that, wire the model into live for an A/B gate test.

## Command used

```bash
.venv/bin/python -u train_filtration_v2.py --market both
```
