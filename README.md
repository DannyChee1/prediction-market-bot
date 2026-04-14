# BTC Latency Arbitrage Bot

  Fires FOK taker orders when Binance BTC moves significantly in 2s and
  Polymarket's order book is still stale.

  ## Strategy

  Binance is the price-discovery venue; Polymarket market makers lag.
  When Binance moves ≥ $30 in 2s and Polymarket's book is 300-5000ms
  stale, we cross the ask on the side of the move. Chainlink is the
  settlement oracle (not our signal).

  No probability model. No Kelly sizing. Pure momentum + venue latency.

  ## Architecture

      Binance WS ──┐
                   │
      Polymarket ──┼──> signal_ticker ──> decide_latency_arb ──> FOK taker
      CLOB WS      │         │
                   │         ├─ Tau gate (≥30s)
      Chainlink ───┘         ├─ Cooldown (4s)
                             ├─ |Binance Δ| ≥ $30 over 2s
                             ├─ book_age in [300, 5000] ms
                             ├─ ask in [0.15, 0.85]
                             └─ Fire: side = sign(Δ)

  ## Files

  - `live_trader.py` — entry point, runs feeds + signal loop
  - `signal_diffusion.py` — `decide_latency_arb()` at ~L2762
  - `tracker.py` — order execution, fill logging, bankroll
  - `feeds.py` — snapshot builder from Rust BookFeed
  - `market_api.py` — Polymarket market discovery via Gamma API
  - `market_config.py` — BTC 5m + 15m parameters
  - `orders.py` — `OrderClient` (signs and posts CLOB orders)
  - `recorder.py` — `OrderBook` parser (required by snapshots)
  - `display.py` — terminal dashboard

  ## Running

      python live_trader.py --market btc --latency-arb --dry-run \
          --bankroll 100 --arb-delta-usd 30 --arb-book-stale-ms 300 \
          2>err.log

  ## Parameters

  | Flag | Default | Meaning |
  |---|---|---|
  | `--arb-delta-usd` | 30 | Binance move threshold |
  | `--arb-window-s` | 2.0 | Lookback window for delta |
  | `--arb-book-stale-ms` | 300 | Min book age to qualify as stale |
  | `--arb-cooldown-s` | 4.0 | Wait between fires |
  | `--arb-min-ask` / `--arb-max-ask` | 0.15 / 0.85 | Entry price band |
  | `--arb-min-tau-s` | 30 | Don't fire within 30s of window end |
  | `--arb-size-usd` | 10 | Fixed bet size |