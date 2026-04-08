# Kou Full (Bipower + Physical Measure) A/B Results — 2026-04-07

## Setup

- 14k BTC 5m + 4.7k BTC 15m post-fix REST-backfilled parquets.
- Walk-forward 70/30 split, seed 42.
- `--tail-mode kou_full` vs `--tail-mode kou` (which is `norm_cdf(z)` post-Kou-drift-bug-fix), same market_blend, same every other param.
- `kou_full` path calls `kou_cdf()` with:
  - `sigma=bipower_variation_per_s(hist)` — continuous-component σ, jump-robust
  - `mu_override=0.0` — physical measure, zero drift
  - Kou params moment-matched from 2M+ returns on the post-fix data, k=3σ jump threshold

## Moment-matched Kou params (new)

| | old (pre-fix) | new (post-fix, k=3σ) |
|---|---:|---:|
| BTC 5m | λ=0.007, p_up=0.526, η1=1254, η2=1200 | **λ=0.076, p_up=0.5014, η1=4885, η2=4868** |
| BTC 15m | (defaults: λ=0.007, p_up=0.51, η1=η2=1100) | **λ=0.068, p_up=0.5013, η1=4504, η2=4510** |

Old values were fit against the buggy Kou drift (which had a ~1-7% downward bias on `p_UP` that the moment-match was absorbing asymmetrically via `p_up=0.526`). Post-fix, the jump distribution is essentially symmetric (`p_up ≈ 0.501`).

## Results

### BTC 5m (4,307 test windows)

| tail_mode | trades | WR | PnL | Sharpe | MaxDD |
|---|---:|---:|---:|---:|---:|
| `kou` *(norm_cdf post-fix)* | **3605** | **62.2%** | **+$27,197** | **1.53** | 3.9% |
| `kou_full` | 2324 | 59.4% | +$17,659 | 1.51 | 4.6% |

`kou_full` is **strictly worse** on 5m: −36% trades, −3pp WR, −35% PnL, flat Sharpe, higher DD.

### BTC 15m (1,439 test windows)

| tail_mode | trades | WR | PnL | Sharpe | MaxDD |
|---|---:|---:|---:|---:|---:|
| `kou` *(norm_cdf post-fix)* | 611 | 59.2% | +$3,854 | 1.26 | 5.8% |
| `kou_full` | **132** | 59.9% | +$1,227 | **1.87 (+48%)** | **2.9% (−50%)** |

`kou_full` has a **real Sharpe improvement and halves drawdown** on 15m, but at the cost of −78% trade count and −68% absolute PnL. **132 test trades is borderline for reliable Sharpe estimation** (bootstrap CI would be wide).

## Interpretation

Two things are happening, and they pull in opposite directions:

1. **Jump-variance fattening** widens the predicted distribution → less conviction per prediction → fewer trades pass the edge threshold → survivors are higher quality.
2. **Small-n concern**: with 132 trades, the Sharpe 1.87 estimate could easily be 1.2 or 2.3 in the true population. Walk-forward is 50 days, so this is 2.6 trades/day — few enough that day-of-week or regime variance could distort it.

On 5m, the "quality filtering" isn't strong enough to offset the loss of trade count — because 5m edges are smaller per trade and we need volume to get meaningful Sharpe out of them. On 15m, each trade is already higher-conviction, and further filtering concentrates into a smaller set of very-high-conviction entries.

## Diagnostics: is the moment-matched Kou even right?

Sanity check from the fit:
```
BTC 5m: 151,414 jumps detected at k=3σ
        across 554.6 hours of observations.
        λ = 151,414 / (554.6 * 3600) = 0.0758 jumps/sec
        λ·τ for 5m window = 0.0758 * 300 = 22.7 expected jumps/window
```

22.7 jumps per 5-minute window is **a lot**. Under the CLT branch of `kou_cdf` (triggered when `λτ > 5`), the decomposition becomes:
```
total_var = sigma_continuous²·τ + λτ·ej2
```
Where `ej2 = 2·p/η1² + 2·q/η2² ≈ 8.4e-8`. So jump variance per second = `λ·ej2 ≈ 6.4e-9`.

Compare to realized total variance per second ≈ `(7e-5)² = 4.9e-9`. The fitted jump variance is **larger than the total realized variance** — which is impossible if jumps are a subset of total variation. That means the k=3σ threshold is **too loose**: some of what's being detected as "jumps" is really just Gaussian tail events from the continuous process.

In other words: **the moment-matched λ is overestimated by ~2-3×**, and `kou_full` is adding too much jump variance on top of an already-correct continuous σ. That's why it kills edges on 5m and filters aggressively on 15m.

## Recommendation: do not ship on either market

- Keep `tail_mode="kou"` on both markets (which is `norm_cdf(z)` post-fix).
- The infrastructure (`kou_full` branch, `bipower_variation_per_s`, `mu_override` on `kou_cdf`, `--tail-mode` CLI) stays in place for future experiments.
- The next iteration would need **either**:
  - A stricter jump detection threshold (k=4σ or k=5σ) and re-moment-match, **or**
  - A proper MLE fit of the Kou params on the observed returns (don't assume the k=σ threshold is the right separator), **or**
  - Fold the Hawkes self-exciting intensity in (Option C from PLAN.md) — clustered jumps might be what's actually happening, and treating λ as constant is too crude.

None of these are a small project. Before spending 1-2 more days on this, the bug fix already captured the main ECE improvement (BTC 5m 0.0802 → 0.0200, BTC 15m 0.0423 → 0.0450). The marginal Kou jump-modeling upside after that fix is small and we have higher-leverage levers (retraining filtration on post-fix data — already done today; retuning threshold/kelly; watching live fills).

## Code shipped (infrastructure stays)

| File | Change |
|---|---|
| `scripts/sigma_estimators.py` | `bipower_variation_per_s()` and `jump_variance_per_s()` — jump-robust σ estimators with unit test passing (Brownian + jump-contaminated synthetic) |
| `backtest.py` `kou_cdf()` | `mu_override: float \| None` parameter — when set, uses the override instead of Q-measure formula `-σ²/2 - λ·ζ`. Lets callers pass `mu_override=0.0` for physical-measure binary prediction |
| `backtest.py` `_model_cdf()` | New `tail_mode="kou_full"` branch. Reads `_sigma_continuous_per_s` + `_delta_log` from ctx, calls `kou_cdf(-delta_log, sigma_cont, λ, p_up, η1, η2, τ, mu_override=0.0)` |
| `backtest.py` `decide()` + `decide_both_sides()` | Populate `ctx["_sigma_continuous_per_s"]` (via bipower variation) and `ctx["_delta_log"]` when `tail_mode == "kou_full"` |
| `backtest.py` CLI | New `--tail-mode` arg with choices `{normal, kou, kou_full, student_t, market_adaptive}` — overrides `config.tail_mode` for A/B tests |
| `backtest.py` `build_diffusion_signal()` | Accepts `tail_mode_override` parameter, plumbs it into the constructor |
| `market_config.py` `btc_5m` | Kou params updated to moment-matched values: `kou_lambda=0.0758, kou_p_up=0.5014, kou_eta1=4884.8, kou_eta2=4867.7` (inert under `tail_mode="kou"`, used under `kou_full`) |
| `market_config.py` `btc` | Kou params added: `kou_lambda=0.0684, kou_p_up=0.5013, kou_eta1=4504.3, kou_eta2=4509.6` |

All changes are backward compatible — default behavior is unchanged because `tail_mode="kou"` still routes through the `norm_cdf(z)` branch.

## Also shipped today (alongside Option B)

- **HMM retrained** with `--high-mult 1.0 --low-mult 0.0`:
  - BTC 5m: state 0 = high_acc 0.882 (kelly_mult=1.0), state 1 = low_acc 0.694 (**kelly_mult=0.0 — zeroes out bets in this state**)
  - BTC 15m: state 0 = high_acc 0.923 (kelly_mult=1.0), state 1 = low_acc 0.854 (**kelly_mult=0.0**)
  - Previous version had both states at kelly_mult=1.0 (no-op). New version will actually affect sizing.
- **Filtration model retrained** on full 14k post-fix data:
  - AUC 0.7671, Brier 0.1392, accuracy 0.7964 at threshold 0.55
  - Keeps 91.9% of trades (filters out 8.1% low-confidence)
  - Positive lift across all tau checkpoints (+1-4pp over baseline)
  - Top features: z_sq, z_abs, sigma, log_vol_regime, log_sigma
  - NOT yet wired into `build_diffusion_signal()` — still opt-in, needs explicit plumbing
