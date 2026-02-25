# Prediction Market Bot

Automated trading bot for Polymarket 15-minute (and 5-minute) crypto Up/Down binary markets. Places limit orders on the CLOB using a GBM diffusion model with Bayesian calibration.

## File Structure

| File | Purpose |
|------|---------|
| `live_trader.py` | Entry point. CLI args, startup config, window lifecycle loop. |
| `tracker.py` | Core trading state machine. Signal evaluation, order management, circuit breakers, position tracking, diagnostics logging. |
| `backtest.py` | Offline backtesting engine. Signal classes (DiffusionSignal), VPIN/toxicity computation, Kelly sizing, calibration table builder. |
| `feeds.py` | WebSocket feeds. CLOB order book stream, RTDS Chainlink price stream, trade event capture for VPIN bars. |
| `display.py` | Terminal dashboard. Renders live book, signal state, positions, toxicity/VPIN indicators, session stats. |
| `orders.py` | Order placement mixin. Limit order creation, cancel/replace logic, fill polling via CLOB API. |
| `recorder.py` | Data recorder. Captures 1-second snapshots to parquet files for backtesting. OrderBook class for L2 book management. |
| `market_api.py` | API helpers. CLOB client construction, market discovery, USDC balance queries, resolution polling. |
| `market_config.py` | Market definitions. Per-asset config (BTC, ETH, 5m variants) -- thresholds, symbols, durations. |
| `redemption.py` | On-chain CTF redemption. Redeems winning positions on Polygon via conditional token framework. |

## Signal Pipeline

1. Chainlink oracle price via RTDS WebSocket
2. Realized volatility (EMA-smoothed, regime-filtered)
3. z-score = delta / (sigma * sqrt(tau)), capped at +/-1.5
4. p(UP) via Bayesian fusion of GBM prior + calibration table
5. Edge = p_model - bid_price - spread_penalty
6. Filters: spread gate, toxicity, VPIN, vol kill switch, momentum
7. Sizing: fractional Kelly with inventory skew

## Usage

```
# Live trading (maker mode)
py -3 live_trader.py --calibrated --early-exit

# Dry run (no real orders)
py -3 live_trader.py --dry-run

# Backtest
py -3 backtest.py --maker --calibrated

# Record data
py -3 recorder.py
```
