# F11 — Redundant WS connections (p99 hedge)

## Idea

Run 2-3 concurrent WebSocket connections per feed:
- Different DNS resolutions (different exchange edge IPs)
- Different endpoints where available (e.g., Binance has
  `stream.binance.com`, `data-stream.binance.vision`)
- Take the earliest-arriving message by `event_time`
- Deduplicate by sequence number

This collapses tails from any single connection's hiccups (NIC
glitches, transient WS protocol errors, brief network jitter) without
any new infrastructure.

## Why this matters

ChatGPT's analysis2.md cites a benchmark: Binance from AWS Tokyo is
~4ms median, ~13ms p99. The p50/p99 ratio is ~3x.

For a single-connection feed from residential Toronto, the ratio is
likely 5-10x (one TCP retransmit dominates). Hedging across 2-3
parallel connections drops the effective p99 close to the p50 of the
luckiest connection — a free 30-50% tail latency cut.

## Concrete change

In `rust/src/feed.rs`, generalize each Feed (BookFeed, PriceFeed,
BinanceFeed, UserFeed) to maintain N parallel WS connections:

```rust
struct BinanceFeed {
    connections: Vec<BinanceConnection>,  // N=2 or N=3
    seq_dedup: HashSet<u64>,              // dedup by Binance update_id
    latest: Mutex<Option<(u64, f64)>>,    // (event_time_ms, mid)
}

impl BinanceFeed {
    async fn ingest(&self, evt: BinanceEvent) {
        let mut state = self.latest.lock().unwrap();
        if state.is_none() || evt.event_time > state.unwrap().0 {
            *state = Some((evt.event_time, evt.mid));
        }
    }
}
```

Spawn one tokio task per connection. They all push into the shared
`latest` slot, only the freshest by event_time wins.

## Cost / benefit

- **CPU**: 2-3x parsing overhead per feed. JSON @bookTicker is ~100
  bytes, parsing is microseconds. Negligible.
- **Bandwidth**: 2-3x. Still tiny (~kbps total).
- **Connections**: more open sockets. Some exchanges rate-limit
  total connections per IP — Binance is generous (300+), Coinbase
  has tighter limits.

## Prerequisites

- F1 (instrumentation) to measure p99 BEFORE and AFTER. Otherwise we
  can't tell if it actually helped.
- F3 (event-time staleness) so the dedup-by-event-time logic works.

## Success criteria

Per-feed p99 staleness drops by ≥30% in F1 measurement after F11
ships.

Live PnL impact is unmeasurable — this is a tail latency fix, not a
direct edge fix. The benefit shows up indirectly via fewer
stale-feature gate trips during bursts.

## Effort

~6 hours per feed type. Start with the most critical (PriceFeed for
Chainlink RTDS, since that's the resolution oracle).

## ROI

Modest in steady state, large during bursts. Worth doing AFTER F1
confirms p99 tails are actually a problem from the bot's location.
