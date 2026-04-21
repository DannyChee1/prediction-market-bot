# BTC Latency Arbitrage Bot (Rust)

Fires FOK taker orders when Binance BTC moves significantly in 2s and
Polymarket's order book is still stale.

## Strategy

Binance is the price-discovery venue; Polymarket market makers lag.
When Binance moves ≥ $30 in 2s and Polymarket's book is 600-5000ms
stale, we cross the ask on the side of the move. Chainlink is the
settlement oracle (not our signal).

No probability model. No Kelly sizing. Pure momentum + venue latency.

## Architecture

    Binance WS ──┐
                 │
    Polymarket ──┼──> signal loop ──> decide_latency_arb ──> FOK taker
    CLOB WS      │         │
                 │         ├─ Tau gate (≥30s)
    Chainlink ───┘         ├─ Cooldown (4s)
                           ├─ |Binance Δ| ≥ $30 over 2s
                           ├─ book_age in [600, 5000] ms
                           ├─ ask in [0.15, 0.85]
                           └─ Fire: side = sign(Δ)

## Crate layout (`rust/src/`)

- `main.rs`        — CLI (clap), runtime, per-market signal loop
- `signal.rs`      — `decide_latency_arb`
- `config.rs`      — `MarketConfig` (BTC 15m + 5m), `ArbParams`
- `feed.rs`        — `BookFeed`, `PriceFeed`, `BinanceFeed` (WS + reconnect)
- `book.rs`        — `BookSnapshot`
- `client.rs`      — `OrderClient` wrapping polyfill-rs (signs + posts CLOB orders)
- `market_api.rs`  — Polymarket Gamma `find_market` + resolution polling
- `tracker.rs`     — bankroll, cooldown, fill log
- `display.rs`     — terminal dashboard
- `redemption.rs`  — CTF redemption constants (implementation stubbed)
- `types.rs`       — `f64_to_decimal`, `now_s`, `now_ms`

## Running

    cd rust
    cargo run --release -- --market btc --dry-run \
        --bankroll 100 --arb-delta-usd 30 --arb-book-stale-ms 600 \
        2>err.log

Requires `.env` with `PRIVATE_KEY`, `POLY_FUNDER`, `POLY_API_KEY`,
`POLY_API_SECRET`, `POLY_PASSPHRASE` once `--dry-run` is dropped.

## Parameters

| Flag                   | Default   | Meaning                                 |
| ---                    | ---       | ---                                     |
| `--market`             | `btc`     | `btc` (15m+5m) / `btc_15m` / `btc_5m`   |
| `--bankroll`           | `100`     | Paper or live bankroll                  |
| `--dry-run`            | off       | Log signals, don't post orders          |
| `--arb-delta-usd`      | `30`      | Binance move threshold                  |
| `--arb-window-s`       | `2.0`     | Lookback window for delta               |
| `--arb-book-stale-ms`  | `600`     | Min book age to qualify as stale        |
| `--arb-cooldown-s`     | `4.0`     | Wait between fires                      |
| `--arb-min-ask`        | `0.15`    | Entry price floor                       |
| `--arb-max-ask`        | `0.85`    | Entry price ceiling                     |
| `--arb-min-tau-s`      | `30`      | Don't fire within 30s of window end     |
| `--arb-size-usd`       | `10`      | Fixed bet size                          |
