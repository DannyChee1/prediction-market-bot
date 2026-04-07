# BTC 5m Market-Blend Fix — 2026-04-06

## Problem

BTC 5m live trading was bleeding money. From the live trade log
(`live_trades_btc.jsonl`, 2026-04-05 19:11 → 2026-04-06 20:46):

- 77 resolved trades, 57.1% WR, **+2.65% ROI** (BTC 15m on the same window: 5.03%)
- DOWN side broken: 25 trades, 44.0% WR, **−16.8% ROI** (−$11.38 PnL)
- 0% WR on 6 trades that entered at price < $0.15

## Root cause

The Kou jump-diffusion model fights the market consensus on 5m windows.
When the contract trades at $0.10 (market says P(UP)≈10%), the model says
"BTC moved $X, so P(UP)≈40% → BUY UP at $0.10, edge=0.30". The market is
right; the model is wrong. Lambda was tuned for calibration (lambda=0.007
in `optimize_kou_5m.py`), but calibration alone doesn't capture this
"don't fight the consensus" effect — the model has near-zero edge over
the market on 5m windows. The strategy review memo (Apr 2026) already
flagged this: *"GBM model loses to market"*.

## Fix

Blend `p_model` with the contract market mid right after the model
returns it, before edge computation:

```python
if self.market_blend > 0:
    mid_up = (bid_up + ask_up) / 2.0
    p_model = (1.0 - self.market_blend) * p_model + self.market_blend * mid_up
    p_model = max(0.01, min(0.99, p_model))
```

`market_blend=0.0` (default) is no-op for ETH/SOL/XRP/BTC 15m. BTC 5m
gets `market_blend=0.3` plus relaxed entry filters (the blend itself
is now the main "don't enter" gate).

**New BTC 5m config** (`market_config.py`):

| Field | Old | New |
|---|---:|---:|
| `kou_lambda`        | 0.007  | 0.007 (unchanged) |
| `min_entry_z`       | 0.7    | **0.0**  |
| `min_entry_price`   | 0.10   | **0.20** |
| `edge_threshold`    | 0.06   | 0.06 (unchanged) |
| `market_blend`      | —      | **0.30** |

## Backtest results

### Lambda sweep is a dead end
A pure lambda sweep over 1159 BTC 5m windows shows the model is
basically equivalent to Normal CDF for any reasonable lambda — Kou drift
correction is degenerate at calibrated η values. Lambda is not the
problem.

### Blend sweep (1159 windows, p≥$0.10, z≥0)

| `market_blend` | Trades | WR | PnL | ROI | Sharpe |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 1738 | 41.0% | +43.0 |  +5.0% | 1.81 |
| 0.1 | 1727 | 43.4% | +56.4 |  +6.6% | 2.45 |
| 0.2 | 1632 | 44.0% | +60.7 |  +7.6% | 2.85 |
| **0.3** | **1738** | **45.1%** | **+66.6** |  **+9.3%** | **3.43** |
| 0.5 | 1489 | 42.9% | +67.1 | +11.7% | 3.77 |
| 0.7 |  845 | 38.9% | +38.1 | +13.1% | 2.94 |

### Most-recent-250-window slice (matches the live regime)

| Config | Trades | WR | ROI |
|---|---:|---:|---:|
| current cfg (live) | 160 | 28.1% | **−10.4%** ← matches live |
| blend=0.3 p≥0.10 z=0 | 407 | 41.0% | **+10.0%** |
| blend=0.5 p≥0.30 z=0 | 304 | 38.8% | −2.8% (overconservative) |

### 5-fold CV stability of `blend=0.3 p≥0.20 z=0`

ROIs across 5 folds: `[2.6, 11.2, 10.0, 8.9, 7.5]`
**mean +8.0%, std 3.0%** — most stable across folds.

Compare current cfg: `[4.4, 19.7, 12.4, 3.5, −10.0]`
mean +6.0%, std **9.9%** (3× more variance).

### End-to-end verification through real `DiffusionSignal`

Ran the actual `decide_both_sides()` code path on 291 recent windows,
with all live filters active (toxicity, OBI, oracle lag, max_spread,
sigma bounds, etc.):

| Config | Trades | WR | PnL | ROI | Sharpe | UP | DOWN |
|---|---:|---:|---:|---:|---:|---|---|
| OLD live cfg          |  92 | 47.8% |  +$5.48 | +14.2% | 1.69 | +$2.59 | +$2.89 |
| **NEW cfg (blend=0.3)** | **124** | 46.0% | **+$10.87** | **+23.6%** | **2.16** | **+$4.37** | **+$6.50** |

ROI nearly doubles, Sharpe +28%, DOWN side PnL +124%.

## Files changed

- `market_config.py` — added `market_blend` field; updated `btc_5m`
- `backtest.py` — added `market_blend` param to `DiffusionSignal`;
  inserted blend block in both `decide()` and `decide_both_sides()`
  after `_p_model()` and before edge computation
- `live_trader.py` — `_build_tracker` plumbs `config.market_blend`
- `dashboard_signal_worker.py` — `_signal_kwargs` plumbs
  `config.market_blend`

Other markets (BTC 15m, ETH, SOL, XRP) are unaffected because
`market_blend` defaults to 0.0.

## Rollback trigger

If after 30+ live BTC 5m resolutions DOWN-side ROI < −10%, either:
- Set `market_blend=0.5` (more conservative blend), OR
- Revert `market_blend=0.0` and disable BTC 5m via `--no-5m`

## Out of scope (intentionally not changed)

- Kou parameters (calibration is fine; the issue is model-vs-market)
- ETH/SOL/XRP markets
- BTC 15m (already profitable at +5% ROI live)
- DiffusionSignal architecture
