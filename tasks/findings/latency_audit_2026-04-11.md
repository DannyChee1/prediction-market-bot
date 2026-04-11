# Latency Audit & Optimization Report — 2026-04-11

## Current System Latency Profile

Measured from `live_trades_btc_1h.jsonl` (21 orders with latency instrumentation):

```
                        min    median    mean     p95      max
order_post_ms           199    206       232      292      433    ← HTTP round-trip US→London
signal_to_post_ms       0      23        40       112      180    ← wake + eval + construct
signal_to_ack_ms        200    248       272      404      434    ← total signal→ack
signal_eval_ms          0      1         98       209      220    ← Python model compute
decision_total_ms       0      169       126      232      302
chainlink_age_ms        43     449       523      1042     1116   ← Chainlink is SLOW
binance_age_ms          0      36        91       350      614
book_age_ms             0      81        74       122      180
signal_trigger_age_ms   0      22        39       112      179
```

Trigger sources: binance=10, book=9, chainlink=2

---

## Critical Path Breakdown

```
                                    Time (ms)
                               min    typical    p95
                              ─────  ─────────  ─────
① Feed WS → Rust memory       ~0      ~0         ~0     (native WS, lock-free atomics)
② Python polls Rust feed       0      10         20     (asyncio.sleep(0.02))
③ wake_event → signal_ticker   0       1          5     (asyncio event dispatch)
④ Signal evaluation (Python)   0       1        209     (decide_both_sides: vol, z, edge)
⑤ Order construction (Python)  0       1          1     (price lookup, size calc, logging)
⑥ py.allow_threads → Rust      0       0          1     (GIL release overhead)
⑦ create_order (sign)          ?       5?        20?    (ECDSA + order builder, in-process)
⑧ HTTP POST to CLOB            ?     195        280     (TLS + network US→London round trip)
                              ─────  ─────────  ─────
TOTAL                          ~200   ~215       ~530
```

**Dominant bottleneck: Step ⑧ (HTTP POST) = 85-95% of total latency**

On Dublin VPS, step ⑧ drops to ~1-2ms. Then the new bottleneck becomes steps ②+④ (polling + Python eval).

---

## Architecture: What Runs Where

### Currently in Rust (polybot_core via PyO3)
- `BookFeed` — CLOB order book WS, BTreeMap-based book, RwLock
- `PriceFeed` — Chainlink RTDS WS, AtomicU64 lock-free reads
- `BinanceFeed` — Binance WS (JSON or SBE), AtomicU64
- `UserFeed` — user trade feed WS
- `OrderClient` — HTTP/2 order placement via polyfill-rs ClobClient
  - `place_order` (GTC limit)
  - `place_market_order` (FOK taker)
  - `place_orders` (batch up to 15)
  - `cancel`, `cancel_all`, `cancel_orders`
  - `warmup` (connection pre-warm + OrderOptions cache)

### Currently in Python (critical path)
- `signal_ticker` — asyncio loop, polls feeds, calls evaluate
- `_poll_price_feed` / `_poll_binance_feed` / `_poll_book_feed` — asyncio polling bridges
- `tracker.evaluate()` → `signal.decide_both_sides()` — ALL signal logic
  - Vol estimation (_compute_vol: Yang-Zhang or tick-based)
  - Z-score computation
  - Calibration table lookup (_p_model)
  - VPIN computation
  - Toxicity computation
  - Oracle lag computation
  - Edge calculation
  - Dynamic threshold
  - Book staleness gates
- `_place_limit_order` — order construction + calls Rust OrderClient

### Tokio Runtime
- 2 worker threads (shared for HTTP + all WS feeds)
- `RUNTIME.block_on()` wraps all async calls from Python
- `py.allow_threads()` releases GIL during HTTP round-trips

---

## Python-Rust Boundary Crossings (per decision tick)

| Crossing | Direction | Frequency | Overhead |
|----------|-----------|-----------|----------|
| `price_feed.price()` | Rust→Python | Every 20ms | ~1µs (atomic load) |
| `price_feed.last_update_ts()` | Rust→Python | Every 20ms | ~1µs (atomic load) |
| `binance_feed.mid()` | Rust→Python | Every 20ms | ~1µs (atomic load) |
| `book_feed.snapshot(token)` | Rust→Python | Every 20ms × 2 tokens | ~10µs (read lock + copy) |
| `book_feed.drain_trades()` | Rust→Python | Every 20ms | ~5µs (write lock + drain) |
| `client.place_order()` | Python→Rust | Per trade | ~200ms (network) |

Total PyO3 overhead per tick: **~30µs** — negligible.
The problem is not the boundary itself, it's the **polling architecture** on the Python side.

---

## Optimization Plan (Ranked by Impact)

### Tier 1: Dublin VPS (~200ms saved, 85% of total latency)

**Impact: order_post_ms drops 200→2ms**

- AWS eu-west-2 Dublin or London region
- Vultr High Frequency London: $12/month (6 GB RAM, 2 vCPU, NVMe)
- Or AWS EC2 t3.medium eu-west-2: ~$30/month
- Polymarket CLOB server is in AWS eu-west-2 (London)
- Dublin→London: ~1ms network
- This alone reduces total latency from ~250ms to ~50ms

### Tier 2: Push-Based Feed Notification (~10-20ms saved)

**Impact: eliminate 0-20ms polling jitter**

Current: Rust feeds → Python polls every 20ms → detects change → evaluates signal
Proposed: Rust feeds → callback directly notifies Python → immediate evaluation

Two approaches:
1. **Rust-side callback via PyO3**: When WS message arrives in Rust, call a Python callback directly. Problem: requires acquiring GIL from Rust, which can block.
2. **Shared-memory signaling**: Rust writes to a shared atomic flag + epoch counter. Python uses a tight spin (or OS-level eventfd) instead of asyncio.sleep. Already partially done with `wake_event`.

The current `wake_event` mechanism is close, but the 20ms `poll_interval_s` in the feed pollers adds the jitter. Reducing to 1ms or switching to a Rust→Python callback would help.

**Quick win: reduce poll_interval_s from 0.02 to 0.001 (1ms)**
- Cost: slightly more CPU usage
- Benefit: 10-20ms shaved from every signal

### Tier 3: Move Signal Hot Path to Rust (~1-200ms saved)

**Impact: eliminate Python compute from critical path**

The signal evaluation (`decide_both_sides`) is pure math:
- Vol estimation (log returns, Yang-Zhang OHLC)
- Z-score = delta / (sigma * sqrt(tau))
- `norm_cdf(z)` → probability
- Calibration table lookup
- VPIN from trade bars
- Edge = p_side - cost_basis
- Threshold comparison

All of this could be a Rust function that takes (price_history, book_state, config) and returns (action, edge, size). The Rust WS handler could call it directly when a new price arrives — zero polling, zero GIL.

**Effort: Medium-High.** The signal_diffusion.py is ~2300 lines, but most is Python-specific scaffolding. The core math is ~500 lines.

**Recommended approach: keep Python for strategy development, compile hot path to Rust once stable.** You're still iterating on the model — rewriting to Rust now locks you in.

### Tier 4: Taker (FOK) Orders (not latency, but execution quality)

Switching from GTC maker → FOK taker:
- **Eliminates adverse selection** (your core problem: 39% win rate)
- **Guaranteed fills** — no more "order placed but not filled"
- **Price improvement** — FOK at 0.99 fills at actual ask price
- **Cost: ~0.5-2% taker fee** (varies by price level)
- At mid-prices (0.40-0.60), fee is ~2% × p × (1-p) ≈ 0.5%
- With 6% edge, you still keep ~4-5.5% net

### Tier 5: Concurrency & Parallelism

**Already parallelized:**
- WS feeds run as separate tokio tasks (true parallelism)
- `py.allow_threads` releases GIL during HTTP
- 5m + 15m trackers run in same asyncio loop

**Opportunities:**
1. **Parallel signal eval for 5m + 15m**: Currently sequential in asyncio. Could use `asyncio.gather()` for both trackers simultaneously.
2. **Parallel order signing**: When placing batch orders, sign all in parallel (tokio tasks).
3. **Dedicated signal thread**: Instead of asyncio coroutine, run signal eval in a dedicated thread with its own priority.
4. **Separate tokio runtimes**: Feed WS vs. HTTP order placement on different runtimes. Currently both share 2 threads — a burst of HTTP calls could starve WS reads.

### Tier 6: Low-Level Optimizations (diminishing returns on Polymarket)

These matter for exchange HFT (nanoseconds) but Polymarket's matching engine is not that fast:

| Technique | Saves | Worth it? |
|-----------|-------|-----------|
| CPU pinning (isolcpus + taskset) | ~1-5µs | Maybe on VPS |
| NUMA-aware allocation | ~100ns | No (single-socket VPS) |
| Kernel bypass (DPDK/io_uring) | ~10-50µs | No (not the bottleneck) |
| Lock-free order book (crossbeam) | ~1-5µs | No (RwLock is fine for 1 reader) |
| SIMD JSON parsing | ~1ms on 480KB | Already done by polyfill-rs |
| `#[repr(C)]` struct layout | ~100ns | Marginal |
| Pre-allocated memory pools | ~1-5µs | No (not allocation-heavy) |

---

## Should We Rewrite Everything in Rust?

### Current Python overhead on critical path (post-Dublin):
- Polling: 0-20ms (fixable by reducing interval)
- Signal eval: 1ms median, 200ms p95 (fixable by caching/optimizing Python)
- GIL release: ~1µs (negligible)
- PyO3 crossings: ~30µs (negligible)

### Full Rust rewrite would save:
- ~10-20ms typical (polling + Python overhead)
- ~200ms in worst case (p95 signal eval spikes)

### Full Rust rewrite would cost:
- 2-4 weeks of development
- Loss of rapid Python iteration for strategy changes
- Harder to add new features (calibration tables, regime filters, etc.)
- Need to port: DiffusionSignal, vol estimation, VPIN, toxicity, Kou model

### Verdict: **Not yet. Hybrid is correct for now.**

The Python-Rust boundary adds ~20ms worst case. On Polymarket, the matching engine processes orders in ~5-50ms server-side. Your competition (BoneReaper) is profitable at ~652 trades/day — they're not running FPGA-level latency, they're running Dublin VPS + smart execution.

**When to reconsider full Rust:**
- After Dublin VPS is deployed and you can measure actual server-side matching latency
- If you find the matching engine is consistently faster than your signal (sub-5ms)
- If you're leaving measurable money on the table due to Python compute time
- When your strategy is stable and you've stopped iterating on the model

---

## Recommended Implementation Order

```
Step 1: Dublin AWS VPS                    [~200ms saved]  ← do this first
Step 2: FOK taker orders                  [fix adverse selection]
Step 3: Reduce poll_interval_s to 1ms     [~10ms saved]
Step 4: Separate tokio runtimes           [prevent WS starvation]
Step 5: Parallel signal eval (5m + 15m)   [better throughput]
Step 6: Rust signal hot path              [~20-200ms saved, only when strategy is stable]
```

---

## Additional Findings from Deep Research

### Worst-Case Latency is Much Worse Than Median
From the full trade log analysis, order_post_ms has a **long tail**:
- p50: ~400-600ms
- p99: ~1000ms
- max observed: **4684ms**

The 425 "service not ready" backoff triggers a 10s pause. These tail latencies are what really kill you in a race.

### polyfill-rs Already Implements Key Optimizations
- SIMD-accelerated JSON parsing (1.77x faster than serde_json on 480KB payloads)
- Zero-allocation hot paths for order book updates (159.6µs/1000 ops)
- Spread/mid calculations in 70ns (14.3M ops/sec)
- HTTP/2 with 512KB stream windows optimized for typical payload size
- Connection pre-warming: 70% faster subsequent requests

### Current Rust Module Architecture is Well-Designed
- PriceFeed + BinanceFeed use lock-free AtomicU64 (sub-microsecond reads)
- BookFeed uses async RwLock (minimal contention: 1 writer, 1 reader)
- GIL properly released on all network calls (2026-04-09 fix)
- OrderOptions cache eliminates 2 HTTP GETs per order (~100-200ms saved)

### What NOT to Do (Agent Analysis)
- Don't move order management (tracker.py) to Rust — complex mutable state, frequent changes
- Don't add rayon/parallel crates — we're IO-bound, not CPU-bound
- Don't increase tokio worker threads beyond 2-4 — more threads = more context-switch overhead for IO workloads
- Don't rewrite the signal until it's proven profitable and stable

---

## Relevant CppCon Techniques (Carl Cook, Optiver, 2017)

1. **Hot/cold path separation**: Keep the trading path free of logging, allocation, error handling. Our `_place_limit_order` does logging in the hot path — should be deferred.
2. **Template over branches**: Compile-time decisions (side=UP/DOWN) instead of runtime checks. Applicable if we move to Rust (generics).
3. **Cache line awareness**: Group frequently-accessed data (price, sigma, z, edge) in a single struct. Currently scattered across Python dict.
4. **Avoid multithreading on hot path**: Single-threaded is faster than any lock. Our signal_ticker is already single-threaded.
5. **Memory pool allocation**: Pre-allocate order objects. Our Rust `OrderOptions` cache already does this.
6. **Cache warming**: Execute dummy orders periodically to keep TLS connections and CPU caches warm. Our `warmup()` + `start_keepalive()` already does the network part.

## CppCon 2024: David Gross (Optiver) — "When Nanoseconds Matter"

Additional techniques from the 2024 keynote (Gross benchmarked ~30 order book implementations):

7. **Contiguous memory beats trees**: `std::vector` with `lower_bound` beats `BTreeMap`/`std::map` for small-to-moderate N due to cache prefetching. Our Rust BookFeed uses BTreeMap — fine for our scale but worth noting.
8. **Branchless binary search**: Replace conditional branches with `cmov` instructions to eliminate branch misprediction during price level lookup.
9. **Price-indexed arrays**: For bounded price ranges (like Polymarket 0.00-1.00 with 0.0001 tick = 10,000 slots), direct array indexing is O(1) and cache-perfect.
10. **Single-writer principle**: Each piece of data has exactly one writer thread. Eliminates cache coherency traffic. Our architecture already follows this (Rust WS tasks write, Python reads).
11. **Pre-fault pages + mlock**: Pin memory to prevent page faults during trading. A single page fault costs ~1-10µs.
12. **Disable hyperthreading on trading core**: Sibling hyperthread shares execution resources, causing interference.
13. **Profile-Guided Optimization (PGO)**: Compile with runtime profiling data for optimal branch/cache layout.

## HFT Design Patterns (arXiv 2309.04259)

14. **Cache warming was the single most impactful technique** across all benchmarks in this paper. We do connection warming but not CPU cache warming.
15. **LMAX Disruptor pattern**: Pre-allocated bounded ring buffer with cache-line-padded sequence counters. 50ns latency, 25M+ messages/sec. Overkill for Polymarket but the pattern (pre-allocated ring buffer + single-writer) is sound.
16. **Semi-static conditions**: Runtime binary patching to eliminate branches entirely. Interesting for compile-time-known paths (BUY/SELL side) but too exotic for our use case.

## What Applies to Us vs. What Doesn't

| Technique | Applies? | Why |
|-----------|----------|-----|
| Dublin VPS (co-location) | **YES** | 200ms → 1ms, biggest single win |
| Connection pre-warming | **Already done** | warmup() + keepalive |
| GIL release during IO | **Already done** | py.allow_threads() |
| OrderOptions cache | **Already done** | Saves 100-200ms/order |
| Reduce poll interval | **YES** | 20ms → 1ms, easy win |
| FOK taker orders | **YES** | Fixes adverse selection |
| Lock-free atomics | **Already done** | PriceFeed/BinanceFeed |
| Kernel bypass (DPDK) | No | Polymarket matching is 5-50ms, not µs |
| FPGA | No | Way overkill for prediction markets |
| CPU pinning | Maybe | On VPS, small win (~1-5µs) |
| Cache warming (code) | Maybe | Only if we move signal to Rust |
| Branchless search | No | Book is small (<20 levels) |
| PGO compilation | Maybe | Easy to enable for Rust build |

---

## Sources

- [Polymarket Server Locations & Latency Guide](https://newyorkcityservers.com/blog/polymarket-server-location-latency-guide)
- [polyfill-rs: Fastest Polymarket Rust Client](https://github.com/floor-licker/polyfill-rs)
- [Polymarket Rate Limits Guide](https://agentbets.ai/guides/polymarket-rate-limits-guide/)
- [Low Latency Trading Systems 2026 Guide](https://www.tuvoc.com/blog/low-latency-trading-systems-guide/)
- [Rust vs C++ for Trading Systems (Databento)](https://databento.com/blog/rust-vs-cpp)
- [Rust in HFT Systems](https://dasroot.net/posts/2026/02/rust-high-frequency-trading-systems/)
- [CppCon 2017: Carl Cook — When a Microsecond Is an Eternity](https://github.com/CppCon/CppCon2017/blob/master/Presentations/When%20a%20Microsecond%20Is%20an%20Eternity/)
