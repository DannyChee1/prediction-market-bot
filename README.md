# Prediction Market Bot

Automated trading bot for Polymarket BTC / ETH / SOL / XRP "Up or Down"
binary markets (5-minute and 15-minute windows). Posts CLOB limit orders
from a diffusion-model-based signal with Bayesian calibration, market
blending, and microstructure gating.

Currently in **dry-run** for ETH / SOL / XRP and **real-money-eligible**
for BTC (5m + 15m) after the post-Kou-fix re-validation.

## Quickstart

```bash
# install deps
uv sync

# live trading (dry-run)
uv run python live_trader.py --market btc --dry-run --bankroll 500

# dashboard (web UI at localhost:8000)
uv run python dashboard.py

# walk-forward backtest
uv run python backtest.py --market btc_5m --signal diffusion --train-frac 0.7

# data recorder (captures per-second parquet snapshots)
uv run python recorder.py --market btc

# historical backfill (Polymarket REST + Binance klines)
uv run python analysis/polymarket_rest_backfill.py \
    --interval 5m --start 2026-02-17 --end 2026-04-08
```

## Repo Layout

### Production code (root)

| File | Purpose |
|---|---|
| `live_trader.py` | Main live-trading entrypoint. Window lifecycle loop, feed wiring, trade execution. |
| `backtest.py` | Offline walk-forward backtest engine + `DiffusionSignal` class (the main signal). |
| `market_config.py` | Per-market parameters (σ bounds, thresholds, `market_blend`, stale-feature gates). |
| `tracker.py` | Live trade state machine — order placement, fills, resolutions, bankroll. |
| `feeds.py` | WebSocket feeds: CLOB order book, Chainlink RTDS prices, Binance book ticker. |
| `recording.py` / `recorder.py` | Per-second parquet snapshot writer (standalone and embedded). |
| `orders.py` | Order placement mixin: limit orders, cancel / replace, fill polling. |
| `redemption.py` | On-chain CTF redemption of winning positions on Polygon. |
| `market_api.py` | Gamma API + CLOB REST client (market discovery, balance, resolution). |
| `dashboard.py` + `dashboard_signal_worker.py` | FastAPI web dashboard showing live signal + paper trades. |
| `display.py` | Terminal UI for `live_trader.py`. |
| `filtration_model.py` + `.pkl` | XGBoost confidence gate (not wired into live signal — opt-in). |
| `regime_classifier.py` + `regime_classifier_*.pkl` | HMM regime classifier (auto-loaded; currently no-op in all regimes). |
| `tick_backtest.py` | Alternate tick-level backtest harness (used by some `analyze_*.py` scripts). |
| `train_filtration.py` | Training script for the XGBoost filtration model. |
| `clean_data.py` | One-off data cleanup utilities. |

### Supporting directories

| Dir | Contents |
|---|---|
| `data/btc_5m/`, `data/btc_15m/`, `data/eth_5m/`, ... | Per-second parquet snapshots, one file per window. Gitignored. |
| `analysis/` | Offline analysis scripts (`analyze_*.py`, `calibration_analysis.py`, `polymarket_rest_backfill.py`, etc.) |
| `analysis/outputs/` | PNG plots from calibration and analysis scripts. |
| `analysis/notebooks/` | Jupyter notebooks for exploratory work. |
| `scripts/` | Validation scripts (`validate_*.py`), training scripts (`train_regime_classifier.py`), σ estimators, Hawkes tools. |
| `tests/` | `pytest`-compatible unit tests (`tests/test_model_cdf.py`). |
| `validation_runs/` | Output dumps from validation runs: trade parquets, metrics JSONs, ergodicity plots, `RESULTS*.md` reports. |
| `tasks/findings/` | Dated markdown findings from each investigation. |
| `rust/` | Experimental Rust WebSocket client (not currently used in the hot path). |
| `live_state_*.json`, `live_trades_*.jsonl`, `live_redemption_queue_*.json` | Runtime state files — read / written by `tracker.py` at repo root. |

## Signal pipeline

Inside `DiffusionSignal.decide_both_sides()`:

1. **Missing-book + invalid-asks gates** — reject if any side is missing.
2. **Stale-feature gates** — `max_book_age_ms`, `max_chainlink_age_ms`, `max_binance_age_ms`, `max_trade_tape_age_ms`. Backtest no-ops; live-only. BTC 5m gates more aggressively than 15m.
3. **Volatility estimation** — Yang-Zhang on 5-second OHLC bars (default). Opt-in EWMA / realized variance / GARCH via `sigma_estimator` field.
4. **z-score** — `z = (chainlink_price − window_start_price) / (sigma · √τ)`, capped at `±max_z`.
5. **`p_model`** — `Φ(z)` (after the Kou drift bug fix, `tail_mode="kou"` collapses to plain Normal CDF because our σ estimator already absorbs jump variance).
6. **Bayesian calibration fusion** — `p = w · p_cal + (1 − w) · p_model`, where `w` grows with the per-bin observation count in the calibration table built during walk-forward train.
7. **Market blend** — `p_model_final = (1 − market_blend) · p_model + market_blend · mid_up`. BTC 5m uses 0.3, BTC 15m uses 0.5 (validated post-Kou-fix on 19k windows).
8. **OBI nudge + reversion discount** — small adjustments based on book imbalance and mean-reversion.
9. **Edge + sizing** — `edge = p − bid − spread_penalty − fees`. Fractional Kelly sized by bankroll × configurable fraction, further multiplied by an optional HMM regime multiplier.
10. **Entry gates** — `min_entry_z`, `min_entry_price`, `edge_threshold`, momentum majority, spread gate, toxicity / VPIN thresholds. Any gate failure → `FLAT`.

The whole pipeline is the same code in both `backtest.py` and the live path — enforced by a single `build_diffusion_signal()` factory function.

## Current shipped params (`market_config.py`)

| Market | `tail_mode` | `market_blend` | `min_entry_z` | `min_entry_price` | `max_book_age_ms` |
|---|---|---:|---:|---:|---:|
| BTC 15m (`btc`) | kou | **0.5** | 0.5 | 0.25 | – |
| BTC 5m (`btc_5m`) | kou | **0.3** | 0.0 | 0.20 | **1000** |
| ETH 15m (`eth`) | student_t (ν=13) | 0.0 | 0.5 | 0.25 | – |
| ETH 5m (`eth_5m`) | student_t (ν=15) | 0.0 | 0.5 | 0.25 | – |
| SOL / XRP 15m / 5m | normal | 0.0 | 0.5 | 0.25 | – |

## Historical findings

Chronological record of investigations and shipped changes —
see `tasks/findings/` for full write-ups:

- `comprehensive_strategy_review_2026-04-05.md` — the "GBM loses to market, snipe is the edge" thesis (partially superseded by later findings).
- `btc_5m_market_blend_fix_2026-04-06.md` — initial `market_blend=0.3` ship for BTC 5m.
- `btc_5m_stale_book_gate_2026-04-06.md` — `max_book_age_ms=1000` ship after finding 5% of live trades happened during WS disconnects.
- `btc_5m_phase2_negative_2026-04-06.md` — investigated and *rejected* 3 proposed quick wins (chainlink_blend tune, filtration wire, obi disable).
- `polymarket_rest_backfill_2026-04-07.md` — built the historical backfill pipeline, grew sample to 14k BTC 5m windows.
- `post_fix_revalidation_2026-04-07.md` — re-ran everything after the research merge fixed the Kou drift bug. **btc.market_blend bumped 0.0 → 0.5**, backtest plumbing fix to actually read `config.market_blend`.
- `validation_runs/RESULTS.md` + `RESULTS_section_2_3.md` — full research branch write-ups (Kou fix, sigma bounds, stale gates, Hawkes, HMM, sigma estimators).

## Opt-in features (not wired into default signal)

All of these are smoke-tested and load cleanly, but each hurts backtest
PnL when swapped in unilaterally because downstream thresholds were
tuned to the legacy paths. Before enabling any one of them, re-tune
`edge_threshold` / `kelly_fraction` / `filtration_threshold` in lockstep.

- **EWMA / realized-variance / GARCH σ** — `scripts/sigma_estimators.py`. Opt in with `sigma_estimator="ewma"` in `MarketConfig`. Wins σ-forecast MSE by 26-39% but costs PnL until downstream re-tune.
- **HMM regime classifier** — `regime_classifier_<market>.pkl`. Auto-loads if present. Multiplies `kelly_fraction` by a per-state multiplier. Currently always returns `1.0` in the live regime → effectively a no-op. Retrain with `scripts/train_regime_classifier.py --high-mult 1.0 --low-mult 0.0` to make it matter.
- **Hawkes self-exciting jump intensity** — `scripts/hawkes.py`. Published as `_hawkes_intensity` ctx feature when `hawkes_params` is set; no downstream consumer reads it yet. Fold into the filtration features on the next retrain.
- **Filtration model (XGBoost confidence gate)** — `filtration_model.pkl` exists but is not wired into `build_diffusion_signal()`. Model was trained pre-Kou-fix, so retrain before shipping.
- **Kou jump diffusion (proper)** — `kou_cdf()` function at `backtest.py:220` is left intact but dead. Wiring it in requires a jump-robust σ estimator (bipower variation) so jump variance isn't double-counted, and a `mu_override=0.0` path so we use physical-measure drift instead of Q-measure. ~3-6h project.

## Calibration plots

See `analysis/outputs/`:

- `calibration_reliability.png` — reliability diagrams per market (pure GBM model, pre-Bayesian-fusion). Current sample: BTC 15m ECE=0.0450, BTC 5m ECE=0.0200. ETH/SOL/XRP cells are mostly empty because those markets haven't been backfilled.
- `calibration_by_tau.png` — reliability curves sliced by time-remaining bucket. BTC 5m / 15m show good calibration across tau; non-BTC panels are empty for the same reason.
- `calibration_distributions.png` — histograms of predicted `p(UP)` colored by actual outcome. BTC markets show the expected U-shape (predictions cluster near 0 and 1 as windows resolve); non-BTC panels are sparse.

Regenerate with:
```bash
uv run python analysis/calibration_analysis.py
```

## Running tests

```bash
uv run python tests/test_model_cdf.py       # 18 regression tests on _model_cdf
```

## Data backfill

If `data/btc_5m/` or `data/btc_15m/` is missing or stale, regenerate
from Polymarket REST + Binance historical klines:

```bash
# BTC 5m, 50 days back (~3h wall clock, ~14k windows)
uv run python analysis/polymarket_rest_backfill.py \
    --interval 5m --start 2026-02-17 --end 2026-04-08 --workers 4

# BTC 15m, same range (~2h)
uv run python analysis/polymarket_rest_backfill.py \
    --interval 15m --start 2026-02-17 --end 2026-04-08 --workers 2
```

Known limitations of REST-backfilled parquets:
- Chainlink price is approximated by Binance 1s klines (median ~2 bp error).
- L2 depth is synthesized from best bid/ask at size=100 (not real depth).
- Tick density is 1 Hz (vs ~3 Hz for live recordings).
- Phase 1 backtests (anything using bid/ask/mid) work fine. Strategies that depend on real depth (OBI gradient, VPIN with real trade tape) do not.

## Project instructions

See `CLAUDE.md` for workflow rules (plan-mode-first, use subagents, verify before done, keep findings in `tasks/findings/`).
