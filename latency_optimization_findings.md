# Latency Optimization Findings

Date: 2026-04-19
Scope: polybot-arb latency arb bot (Rust, Tokio, taker mode, BTC/ETH up-down markets)

## 0. TL;DR

**Headline network finding:** Polymarket CLOB origin is **AWS eu-west-2 (London)** — triangulated via Cloudflare behavior and confirmed by practitioner analysis. Your Ireland VPS (eu-west-1) is ~8-12ms from Polymarket origin. **This is near-optimal. Don't move to Tokyo or US East.** The only real geographic lever is Binance (Tokyo AWS, fixed at ~98ms from Ireland).

**Headline code findings:**
- `TCP_NODELAY` is **not set** on any WebSocket connection. Could be up to 40ms Nagle tax on outbound frames (currently only impacts Polymarket WS writes — PING heartbeats and book-sub messages).
- `tokio-tungstenite` pinned at **v0.21** (current stable is >=0.26). Minor perf improvements and bugfixes in-between. Free upgrade.
- All Mutex usage is `tokio::sync::Mutex` — even for critical sections that don't cross `.await`. `parking_lot::Mutex` would be ~2× faster uncontended. Biggest impact on hot path: the main signal loop's `rt.lock()`.
- JSON parsing via `serde_json`. `sonic-rs` (ByteDance) is 3-4× faster on tiny messages like Binance bookTicker. μs-level win per message.

**Ranked action plan** (details below):
1. Add `set_nodelay(true)` to every WebSocket stream — up to 40ms potential p99 win, 30 min work.
2. Swap `tokio::sync::Mutex` → `parking_lot::Mutex` on `MarketRuntime` and `tracker`. ~300ns/tick, 1 hour.
3. Parallel Binance WS connections (N=2-3), dedupe by `u` sequence id — cuts p99 tail jitter 2-8ms, 4 hours.
4. Bump tokio-tungstenite 0.21 → ≥0.26. Free. 30 min.
5. `RUSTFLAGS="-C target-cpu=native"` on the VPS build. 5-20% on hot math, 10 min.
6. Replace serde_json with sonic-rs for Binance feed decode. ~1.5μs/msg, 3 hours.

Total estimated critical-path latency savings: **20-50ms p99** (dominated by Nagle fix if it's hitting). Tier 2 items (SBE, fastwebsockets, io_uring, PGO, Tokyo VPS) all rejected on cost/benefit — see §5.

---

## 1. Hot-path map

### Signal path (Binance tick → Polymarket order)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ t=0      Binance bookTicker tick on data-stream.binance.vision              │
│          (Tokyo AWS ap-northeast-1)                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ +98ms    Ireland VPS NIC receives TLS bytes                                 │
│          [NETWORK — dominated by geographic RTT Tokyo→Ireland]              │
├─────────────────────────────────────────────────────────────────────────────┤
│ +98.x    tokio-tungstenite decodes WS frame (tiny, ~100 bytes)              │
│          [CODE — est. 5-10μs on this version]                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ +98.x    serde_json parses {"b":...,"a":...}                                │
│          [CODE — est. 3μs with serde_json, ~1μs with sonic-rs]              │
├─────────────────────────────────────────────────────────────────────────────┤
│ +98.x    Atomic store of mid price to AtomicU64                             │
│          wake.notify_waiters()                                              │
│          [CODE — ~100ns, lock-free]                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ +98.x    Main signal loop wakes from tokio::select!                         │
│          acquires rt.lock() — tokio::sync::Mutex                            │
│          [CODE — 500-1500ns on tokio::sync::Mutex, 200-500ns parking_lot]   │
├─────────────────────────────────────────────────────────────────────────────┤
│ +98.x    Pushes sample to VecDeque ring, runs decide_latency_arb            │
│          [CODE — pure Rust, sub-μs in release]                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ +98.x    prepare_fire (tracker Mutex acquire, check bounds, return ticket)  │
│          [CODE — ~μs]                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ +98.x    tokio::spawn order-submit task                                     │
│          polyfill-rs::create_market_order (EIP-712 signing)                 │
│          [CODE — likely 1-5ms incl JSON serde + keccak256 + secp256k1]     │
├─────────────────────────────────────────────────────────────────────────────┤
│ +~100    HTTP POST to clob.polymarket.com (Cloudflare eu-west POP)          │
│          Cloudflare → Polymarket origin (AWS eu-west-2 London)              │
│          [NETWORK — ~10ms Ireland VPS → London CF POP → London origin]     │
├─────────────────────────────────────────────────────────────────────────────┤
│ +~110    Polymarket CLOB processes match                                    │
│          [PM internal — ~20-50ms per practitioner data]                     │
├─────────────────────────────────────────────────────────────────────────────┤
│ +~160    HTTP 200 response back to Ireland VPS                              │
│          record_fire reconciles into tracker state                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Bottleneck breakdown (estimated):**
- Binance network ingress: **98ms** (geographic, unavoidable without Tokyo VPS)
- Ireland VPS → Polymarket origin: **~10ms** (near-optimal for London-origin CLOB)
- Code processing on VPS: **~2-10ms** (signing is the biggest in-VPS chunk)
- Polymarket internal matching: **~20-50ms** (external, out of our control)

**→ The code processing is 2-10ms of a 180ms total round trip. The geographic 98ms Binance hop is ~54% of the budget.**

### Feed path (supporting streams)

- **Coinbase WS** (`ws-feed.exchange.coinbase.com`): similar geometry, also cross-ocean. Only used for cross-venue consensus.
- **Polymarket CLOB WS** (`ws-subscriptions-clob.polymarket.com`): Cloudflare-terminated, ~10ms from Ireland. Push-based book updates.
- **Polymarket RTDS** (`ws-live-data.polymarket.com`): Chainlink price stream, ~10ms from Ireland.

Feed staleness budget governed by `book_stale_ms` (600ms currently). We require Polymarket's book to be ≥600ms stale to fire — so feed path latency doesn't eat into edge, as long as Binance is faster (98ms << 600ms).

---

## 2. What's working (don't touch)

- **`tokio::sync::Notify`-based signal loop wake** is exactly right. The 100ms safety sleep is the only "poll" cost, and only on quiet periods.
- **Per-asset feed bundles + per-market rings** correctly isolates state.
- **Pipelined order firing** via `tokio::spawn` means the 80ms HTTP round-trip doesn't block subsequent signals.
- **HTTP keep-alive** enabled by default in reqwest; polyfill-rs pre-warms tick_size and neg_risk.
- **Atomic price storage** (AtomicU64 with f64 bit-pattern) for cross-thread shared mid price is lock-free and correct.
- **Persisted redeem queue + 3-strike drop** (added last session) prevents poison-pill infinite loops.

---

## 3. Ranked optimizations

### Tier 1 — Do these now (cheap, high-value, low-risk)

#### 3.1 TCP_NODELAY on every WS connection
- **Savings:** up to 40ms p99 (Nagle holds small frames for 40ms default). Real impact depends on how often we send tiny frames (PINGs, book subs).
- **Where:** `feed.rs` — every `connect_async` creates a `WebSocketStream` that wraps a `TcpStream`. After connect, grab `tcp.set_nodelay(true)`.
- **Cost:** ~30 min of work. Four connection sites (Binance, Coinbase, PM CLOB book, PM RTDS).
- **Risk:** None. This is universally applied in HFT contexts.
- **Verify after:** compare order-send p50 before/after via `strace -e trace=sendto` on the VPS or via tracked timestamps.

#### 3.2 parking_lot::Mutex for non-await critical sections
- **Savings:** ~300-1200ns per lock acquisition under low contention (Tokio maintainers themselves recommend this pattern when the critical section has no await points).
- **Where:** `Arc<Mutex<MarketRuntime>>`, `Arc<Mutex<Tracker>>`, `Arc<Mutex<VecDeque>>` for chainlink histories, `Arc<Mutex<Vec<QueuedRedemption>>>` for redeem queue.
  - Most are locked without spanning `.await`. They can all switch.
  - Exception: any code holding a lock across `await` needs tokio::sync::Mutex or refactor.
- **Cost:** ~1-2 hours. Mostly mechanical `use tokio::sync::Mutex` → `use parking_lot::Mutex` + `.lock()` (blocking, but microseconds).
- **Risk:** Low. Audit for any `.await` inside lock scopes before swapping (blocking in async = bad).
- **Aggregate impact:** At ~10 ticks/sec with ~3 locks per tick, savings of ~3μs/sec. Small in aggregate but removes a subtle yield-point footgun.

#### 3.3 Parallel Binance WS connections (N=2-3 with dedup)
- **Savings:** 2-8ms p99 tail jitter reduction. Binance WS frames have high inter-arrival variance (the earlier probe showed max=301ms on stream.binance.com, max=62ms on data-stream.binance.vision). Multiple connections take "first to arrive" per sequence ID and smooth this.
- **Where:** `feed.rs::BinanceFeed::new` — spawn N=2 or 3 parallel subscriptions, dedupe by `u` (update ID in message).
- **Cost:** ~4 hours. Non-trivial but contained.
- **Risk:** Medium. Binance allows 300 connections per IP per 5min, so 2-3 fine. Must ensure all N see the same `btcusdt@bookTicker` stream; if they diverge we'd get conflicting bids/asks.
- **Verify after:** log arrival jitter histogram before/after.

#### 3.4 Upgrade tokio-tungstenite 0.21 → ≥0.26
- **Savings:** Sub-ms but includes bugfixes and minor decode optimizations.
- **Where:** `Cargo.toml`. Might also need `tungstenite`, `futures-util` version bumps.
- **Cost:** ~30 min. Potential breaking-change audit.
- **Risk:** Low. Mainstream library with stable semantics.

#### 3.5 target-cpu=native
- **Savings:** 5-20% on SHA256, keccak256, SIMD-able pure Rust (price ring scans).
- **Where:** Build script or `.cargo/config.toml` on the VPS.
- **Cost:** 10 min.
- **Risk:** Binary tied to VPS CPU family.

### Tier 2 — Worth doing after Tier 1 is live

#### 3.6 sonic-rs for Binance JSON
- **Savings:** ~1.5μs per bookTicker message (from ~3μs serde_json → ~1μs sonic-rs). At 100 msg/sec per market × 4 markets = ~600 msg/sec = ~900μs/sec saved. Small.
- **Where:** `feed.rs::binance_feed_task`. Coinbase too if we care.
- **Cost:** ~3 hours. Add dep; sonic-rs parses directly to struct (no tape stage).
- **Risk:** Low. sonic-rs is pure-Rust, production-grade at ByteDance.
- **Note:** Do NOT use simd-json — requires `&mut [u8]` which complicates WS streaming.

#### 3.7 Lock-free SPSC ring for price samples
- **Savings:** ~200ns per push/pop. Only worth it if you're chasing tail microseconds after parking_lot is in.
- **Where:** binance_ring / coinbase_ring inside MarketRuntime.
- **Cost:** ~6 hours. `crossbeam::queue::ArrayQueue` or `rtrb` crate.
- **Risk:** Medium. Ring size + overflow semantics differ from VecDeque. Must match existing capacity cap behavior.

### Tier 3 — Explicitly rejected (bad ROI)

| Idea | Why not |
|---|---|
| **Binance SBE (binary protocol)** | 50ms cadence vs 100ms on depth streams. But `@bookTicker` is already push-on-change. Our signal window is 2s — 50ms cadence is <3% of budget. Integration cost ~20 hours. Revisit only at sub-500ms signal windows. |
| **fastwebsockets rewrite** | Shown 2-5ms faster vs old tokio-tungstenite, but vs 0.26+ the delta is <1ms. Plus API differs (raw frames). Not worth 8 hours for <1ms. |
| **io_uring / glommio / monoio** | Higher throughput but worse p99/max latencies than epoll at low connection counts (<10). We have ~5 WS connections. 6+ months rewrite for zero clear win. |
| **Kernel bypass (DPDK/AF_XDP)** | Relevant at microsecond HFT. We're at 80ms HTTP-POST scale. Overkill by 3 orders of magnitude. |
| **PGO (profile-guided optimization)** | 5-15% on synthetic CPU benches. Our hot path is network-bound. PGO can't touch the 98ms Binance hop. |
| **Pre-signed nonce pool** | Polymarket nonces are one-use, server-tracked. secp256k1 signing is ~50μs (not the 1-5ms estimated — that's likely JSON serde + keccak of static fields). Noise against 80ms RTT. Caching the domain separator hash might save ~50μs — worth doing IF you benchmark and confirm. |
| **Tokyo VPS** | Binance 10ms (save 88ms) but Polymarket origin is London (not US East as first assumed). Tokyo → CF Tokyo POP → CF London → London origin = 100-140ms transpacific. Net LOSS of ~30-70ms on the full round trip. |
| **US East VPS** | Binance ~90ms (similar to Ireland). Polymarket: CF US-East POP → London origin transatlantic = ~80ms. Net ~LOSS. |
| **mimalloc/jemalloc** | Few small allocs in hot path. <5% win. |
| **Multi-region (Tokyo ingest + London execute) with inter-region link** | Inter-region AWS backbone is ~100-140ms transpacific. Not faster than direct. Only helps if we could move the DECISION to Tokyo. But then order still has to go Tokyo → London origin = worse than direct. |

---

## 4. Network topology: why Ireland is right

Earlier in the conversation I incorrectly assumed Polymarket's origin was US-East. The research subagent triangulated via William Entriken's Cloudflare analysis that **the CLOB origin is AWS eu-west-2 (London)**. This changes everything:

| VPS region | Binance Tokyo RTT | Polymarket origin RTT | Total critical path |
|---|---|---|---|
| **Ireland eu-west-1 (current)** | ~98ms | **~10ms** (inter-region AWS) | **~108ms** |
| US East us-east-1 | ~90ms | ~80ms (transatlantic) | ~170ms |
| Tokyo ap-northeast-1 | ~10ms | ~130ms (transpacific) | ~140ms |
| Waterloo / Canada | ~172ms | ~70ms | ~242ms |

**Ireland is the single optimal point.** Any move loses on one hop more than it gains on the other.

---

## 5. What to NOT do

- **Do not move the VPS.**
- **Do not try to go maker.** (Already covered last session; maker economics don't beat taker net of adverse selection + fill rate. Confirmed by BoneReader being taker-primary.)
- **Do not rewrite on io_uring / glommio / monoio.** Wrong tool for this scale.
- **Do not attempt Binance SBE integration** until signal window is ≤500ms. Not currently the bottleneck.
- **Do not chase microseconds** below the Tier 2 level until the 98ms Binance geographic hop is addressed — and the only way to address it is a Tokyo ingress pod, which is already ruled out.

---

## 6. Open questions / things I didn't prove

1. **Polymarket internal processing time** (~20-50ms assumption from research). Never directly measured. Would need matched timestamps on outgoing POST and response.
2. **polyfill-rs create_market_order breakdown** (1-5ms assumption). Not profiled. Worth running a Criterion bench locally to see if it's signing, JSON encoding, or something else that dominates.
3. **Cloudflare Argo smart routing** on Polymarket's account — if enabled, inter-region CF paths could be faster than geographic estimates. No public data.
4. **Binance WS arrival jitter histogram on the Ireland VPS** (we have probe numbers but only for connection/first-msg, not per-tick arrival).

### Suggested micro-benchmarks to run locally (no VPS needed)

- `cargo bench` on serde_json vs sonic-rs decode of a sample bookTicker frame. Expected: ~3μs → ~1μs.
- Criterion bench on `polyfill-rs::create_market_order` to break down signing vs serialization cost.
- Benchmark `tokio::sync::Mutex` vs `parking_lot::Mutex` for the specific `MarketRuntime` usage pattern.

---

## 7. Not recommended but worth calling out for completeness

- **Co-located hardware at AWS eu-west-2** (same region as Polymarket origin) would shave ~5-10ms off the Polymarket round trip vs eu-west-1. Marginal. Worth experimenting IF you hit a ceiling and need the last milliseconds.
- **`SO_BUSY_POLL` + `SO_INCOMING_CPU`**: kernel-level NIC interrupt tuning. Reduces interrupt latency ~100μs. Relevant on dedicated hardware, not standard VPS.
- **NUMA-aware thread pinning**: only matters on multi-socket bare metal.

---

## Sources referenced

- [QuantVPS: Polymarket servers location](https://www.quantvps.com/blog/polymarket-servers-location)
- William Entriken's CLOB triangulation to AWS eu-west-2
- [Binance WS streams docs](https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams)
- [Binance SBE streams docs](https://developers.binance.com/docs/binance-spot-api-docs/sbe-market-data-streams)
- [sonic-rs (ByteDance)](https://github.com/cloudwego/sonic-rs)
- [Tokio mutex discussion #2599](https://github.com/tokio-rs/tokio/issues/2599)
- [reqwest TCP_NODELAY default](https://github.com/seanmonstar/reqwest/commit/1f83471)
- [nnethercote Rust perf book](https://nnethercote.github.io/perf-book/build-configuration.html)
- [atomic_queue SPSC benchmarks](https://max0x7ba.github.io/atomic_queue/html/benchmarks.html)
