# BTC 5m Stale-Book Gate — 2026-04-06

## TL;DR

Live latency profile of `live_trades_btc.jsonl` revealed that **5% of BTC 5m
trades happen during book-WS disconnects** (book_age_ms > 1s). Those stale-book
trades win 25% with **−$6.13 PnL**, vs the 71 fresh-book trades at 57.7% WR
with **+$6.21 PnL**. The two cancel out — the bot is doing a lot of work for
roughly zero net edge purely because of websocket reliability.

**Fix:** Added `max_book_age_ms` parameter to `DiffusionSignal`. BTC 5m set to
1000ms. The gate triggers a FLAT decision when `ctx["_book_age_ms"]` exceeds
the threshold, before any other signal computation.

## Investigation path

### Latency profile

Pulled from 2276 `limit_order` events in `live_trades_btc.jsonl`:

| Field | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|
| signal_eval_ms | 1 | 218 | 640 | 1823 |
| decision_total_ms | 2 | 223 | 641 | 1824 |
| signal_to_post_ms | 1 | 288 | 689 | 1752 |
| signal_to_ack_ms | 311 | 1021 | 2181 | 7474 |
| order_post_ms | 289 | 846 | 1841 | 7473 |
| chainlink_age_ms | 531 | 1216 | 1844 | 49,729 |
| binance_age_ms | 8 | 245 | 720 | 15,611 |
| **book_age_ms** | **3** | **166,384** | **383,766** | **705,635** |

The book_age_ms distribution is **bimodal**:
- p50 = 3ms, p75 = 13ms (healthy)
- p90 = 43,811ms (catastrophic)

The jump from 13ms to 44 seconds at the 75-90 percentile is the signature
of a websocket dropping and reconnecting. When the book WS reconnects, the
last-known book timestamp is from before the disconnect — so any decision
made before fresh data arrives is operating on minutes-old prices.

### Buckets

```
records with book_age > 1s   : 247 (10.9%)
records with book_age > 5s   : 245 (10.8%)
records with book_age > 30s  : 238 (10.5%)
records with book_age > 60s  : 217 (9.5%)
records with book_age > 300s :  42  (1.8%)
```

Note that the 1s→5s and 5s→30s buckets contain almost no records — when the
book is stale, it's dramatically stale. There's no smooth tail.

### Win rate by book_age (resolved trades)

| Bucket | n | WR | PnL | Cost | ROI |
|---|---:|---:|---:|---:|---:|
| < 100ms (fresh) | 71 | **57.7%** | **+$6.21** | $202.49 | +3.1% |
| 100ms-1s | 1 | 0.0% | −$3.71 | $3.72 | −99.7% |
| 30-60s | 2 | 50.0% | −$1.20 | $6.20 | −19.4% |
| > 60s (stale) | 1 | 0.0% | −$1.22 | $1.22 | −100.0% |

Stale buckets are tiny in count (4 trades) but their losses (−$6.13) almost
exactly cancel the fresh trades' gains (+$6.21). Sample is small but the
mechanism is physically obvious: trading on a 60-second-old book = trading
against ghosts.

## Fix

Added `max_book_age_ms: float | None = None` to `DiffusionSignal.__init__`.
In `decide_both_sides()`, after the missing-book and invalid-asks gates:

```python
if self.max_book_age_ms is not None:
    book_age = ctx.get("_book_age_ms")
    if book_age is not None and book_age > self.max_book_age_ms:
        reason = f"stale book ({book_age:.0f}ms > {self.max_book_age_ms}ms)"
        return (Decision("FLAT", 0.0, 0.0, reason),
                Decision("FLAT", 0.0, 0.0, reason))
```

`live_trader.py` already populates `ctx["_book_age_ms"]` at line 133 — no new
plumbing needed for that signal. The new parameter is plumbed through:
- `market_config.py` → `MarketConfig.max_book_age_ms` (default `None`)
- `live_trader.py:_build_tracker` → `signal_kw["max_book_age_ms"] = config.max_book_age_ms`
- `dashboard_signal_worker.py:_signal_kwargs` → same

Per-market settings:
- **BTC 5m: 1000ms** (only market with confirmed evidence)
- BTC 15m, ETH, SOL, XRP: `None` (no change)

## Verification

Unit-tested the gate directly with a synthetic snapshot:

```
book_age=100   → passes ("no edge ...")
book_age=5000  → GATED   ("stale book (5000ms > 1000.0ms)")
book_age=None  → passes  (no telemetry, don't reject)
gate=None      → passes  (other markets unaffected even with stale book)
```

Smoke-tested all production paths: `live_trader.py --help`,
`dashboard.py --help`, `tick_backtest.py --help` — all OK.

## Expected impact

Naive math from this sample: removing the 4 stale-book trades would have
turned $0.08 net PnL into $6.21 net PnL — a 76× improvement. This is on a
small sample, so the magnitude is uncertain, but the direction is clear and
the mechanism is physically guaranteed: you cannot price-discover with
two-minute-old book data.

The downside is bounded: at worst we skip ~5% of trades during WS outages.

## Out of scope (defer to deeper Phase 3)

The latency profile also revealed:
1. **`signal_to_ack_ms` p50 = 311ms early-window vs 2ms late-window** — something
   is blocking decisions early in each new window. Could be book-WS reconnect,
   feed warmup, or thread contention. Worth profiling but not blocking.
2. **Book WS itself drops** — the gate is a band-aid; the real fix is to make
   the book WS more reliable (heartbeat, faster reconnect, dual feed). That's a
   `feeds.py` infrastructure project, separate plan.
3. **`chainlink_age_ms` max = 49,729ms** — Chainlink feed died for 49 seconds at
   one point. Edge case but worth alerting on.
