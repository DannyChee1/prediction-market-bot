# TODO

## DONE: ETH support (2026-04-18)

Implemented multi-asset latency arb — BTC + ETH under one bankroll.

### Changes
- [x] `config.rs`: `Asset` enum (Btc/Eth) with `binance_symbol`/`coinbase_product`/`chainlink_symbol` helpers. Added `ETH_15M`/`ETH_5M`. Dropped per-config symbol fields.
- [x] `resolve()`: new `eth`, `eth_15m`, `eth_5m`, `all` options.
- [x] `main.rs`: per-asset feed bundles (`HashMap<Asset, Arc<AssetFeeds>>`). BTC 15m + BTC 5m share one Binance/Coinbase/Chainlink feed; likewise for ETH. Per-asset Chainlink history for resolution lookup.
- [x] `tracker.rs`: `Position` and `FireTicket` carry an `asset`. `PersistedPosition.asset` uses `#[serde(default)]` → BTC so existing `bot_state.json` files migrate cleanly.
- [x] `tracker.rs`: dropped global `last_fire_ms`. Cooldown is now per-market in the signal loop. Fixes a latent bug where any fire on any market froze ALL markets for 4s.
- [x] `main.rs`: pipelined order firing via `tokio::spawn` — signal loop no longer blocks 50-100ms on each HTTP round-trip. Per-market cooldown set optimistically before the spawn so we can't double-fire.
- [x] `display.rs`: generic "Latency Arb" header.
- [x] CLI: `--log-path` and `--state-path` flags; default log path renamed to `live_trades_arb.jsonl` (both BTC + ETH rows carry `slug` + `asset`).

### Verification
- `cargo check` / `cargo build --release`: clean, no warnings.
- 15s dry-run smoke test with `--market all`: all four markets discovered on Gamma (`btc-updown-15m-…`, `btc-updown-5m-…`, `eth-updown-15m-…`, `eth-updown-5m-…`), BTC + ETH feeds independently populated (BTC σ_2s≈$2.2, ETH σ_2s≈$0.1), dashboard renders correctly.

### Known tuning gap
- Used `delta_floor=$25` for ETH per user request. At ETH ≈ $2360 this is a 1.06% move in 2s — about 21× more restrictive than the equivalent BTC ratio. ETH will fire rarely until the floor is re-tuned. Since `k×σ_2s` on ETH is ~$0.1, the floor is the binding constraint in quiet regimes.
- Suggested starting point once the user tunes: `delta_floor_eth ≈ delta_floor_btc × (eth_price / btc_price)` ≈ $25 × (2360/75900) ≈ **$0.78**. That's 2σ-3σ on current ETH σ_2s. User asked to ship with $25 first; CLI `--arb-delta-floor` lets them test from a second process.

## Previous TODO

## P1: Book depth check

Before firing, verify the ask level has enough size to fill the intended
notional without slipping to the next level.

## P3: Chainlink confirmation

We have `snapshot.chainlink_price` but don't use it for the arb gate. If
Chainlink has already moved in the same direction as Binance, signal is much
stronger (settlement oracle is confirming).
