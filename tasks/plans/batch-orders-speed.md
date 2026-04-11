# Batch Orders & Speed Improvements

## Context

Current order flow: signal evaluation runs every ~5-25ms. When the signal says
"cancel old order, place new one", we do 2 sequential HTTP calls (cancel + post)
plus a 3rd verify GET. Each Polymarket CLOB round-trip is ~100-300ms. The cancel
also creates a ~200ms race window where the old order can fill at a stale price.

BoneReaper trades ~36 times per 5m window with ~$47 average size, placing both
UP and DOWN sides in 64% of windows. They appear to be a maker (84% prices on
cent boundaries) with sub-5s reaction to window open. Their speed advantage
likely comes from fewer HTTP round-trips per decision cycle.

---

## Phase 1: Parallel cancel + place in Rust (tokio::join!)

**Problem**: When edge shifts from one side to another (cancel UP, place DOWN),
two sequential HTTP calls cost 200-600ms total. During that window the bot is
blind to fills on the cancelled order.

**Solution**: Add a `cancel_and_place` method to `OrderClient` that runs
`cancel_orders` and `post_orders` concurrently via `tokio::join!`.

### Changes

**`rust/src/client.rs`** — Add new method after `place_orders` (~line 418):

```rust
/// Atomically cancel old orders and place new ones in parallel.
///
/// Both HTTP calls run concurrently via tokio::join!, cutting
/// total latency from ~400ms (sequential) to ~200ms (parallel).
/// Returns (cancel_response, Vec<place_responses>).
#[pyo3(signature = (cancel_ids, new_orders, order_type="GTC", fee_rate_bps=0))]
fn cancel_and_place(
    &self,
    py: Python<'_>,
    cancel_ids: Vec<String>,
    new_orders: Vec<(String, f64, f64, String)>,  // [(token_id, price, size, side)]
    order_type: &str,
    fee_rate_bps: u32,
) -> PyResult<PyObject> {
    // ... sign new orders, then:
    // let (cancel_resp, place_resp) = tokio::join!(
    //     client.cancel_orders(&cancel_ids),
    //     client.post_orders(signed, ot),
    // );
}
```

**`orders.py:~596-634`** — In the `_evaluate_maker` cancel+replace path:
Currently the code cancels on line 596/603, then on the NEXT 1s tick, places
a new order on line 634. This means cancel and place are always in different
evaluation cycles (1s apart). To use `cancel_and_place`:

1. When a cancel reason triggers AND the opposite/same side has a pending
   placement, batch them together.
2. In `_evaluate_maker`, collect `(cancels_needed, places_needed)` in a first
   pass, then execute as a single `cancel_and_place` call.

**`tracker.py:~526-634`** — Refactor the for-loop to two passes:
- Pass 1: Collect cancel decisions and place decisions into lists
- Pass 2: Execute via single `cancel_and_place` if both lists non-empty,
  or `cancel_orders`/`place_orders` individually if only one list

**Expected savings**: ~150-200ms per cancel+replace cycle (from sequential
~400ms to parallel ~200ms). This matters most when the bot is requoting.

**Priority**: HIGH — most impactful single change

---

## Phase 2: Batch dual-side placement

**Problem**: When both UP and DOWN have edge simultaneously (which BoneReaper
does in 64% of windows), we currently place them as 2 separate HTTP calls.

**Solution**: Use `place_orders` to post both in a single call.

### Changes

**`tracker.py:~526-634`** — After the Phase 1 refactor to two-pass:
- The "places_needed" list naturally collects both sides
- `place_orders` already accepts `Vec<(token_id, price, size, side)>`
- Post all pending placements in one batch call

**`orders.py:355-588`** — Refactor `_place_limit_order`:
- Extract the order-preparation logic (price validation, share sizing,
  bankroll check, model snapshot) into `_prepare_limit_order` that returns
  an order dict without actually posting
- Add `_execute_batch_orders(prepared_orders: list[dict])` that calls
  `self.client.place_orders(...)` once, then processes all responses
- Keep `_place_limit_order` as a convenience wrapper that prepares + executes
  a single order (for non-batch paths like sell-side)

**Python-side changes needed**:

```python
# orders.py — new method
def _prepare_limit_order(self, snapshot, decision, token_id, side_label) -> dict | None:
    """Prepare order params without posting. Returns None if invalid."""
    # ... all validation + sizing logic from _place_limit_order lines 362-425
    # Returns: {token_id, price, shares, side, cost_est, model_snapshot, ...}

def _execute_batch_orders(self, prepared: list[dict], snapshot) -> None:
    """Post multiple orders in a single batch HTTP call."""
    tuples = [(o["token_id"], o["price"], o["shares"], "BUY") for o in prepared]
    responses = self.client.place_orders(tuples, "GTC", 1000)
    for order_params, resp in zip(prepared, responses):
        # ... process each response (same logic as lines 487-588)
```

**`tracker.py:~625-634`** — Replace individual `_place_limit_order` calls:
```python
# Collect all placements
placements = []
for dec, token, side_label in [(up_dec, up_token, "UP"), (down_dec, down_token, "DOWN")]:
    if dec.action != "FLAT" and dec.size_usd > 0 ...:
        prepared = self._prepare_limit_order(snapshot, dec, token, side_label)
        if prepared:
            placements.append(prepared)
if placements:
    self._execute_batch_orders(placements, snapshot)
```

**Expected savings**: ~150-200ms when placing dual-sided (skip one full HTTP
round-trip). Enables BoneReaper-style dual-quoting without latency penalty.

**Priority**: MEDIUM — only helps when both sides have edge simultaneously.
Depends on Phase 1 refactor.

---

## Phase 3: Batch end-of-window cancels

**Problem**: `_cancel_open_orders` (line 951-969) and `_cancel_open_sell_orders`
(line 1283-1289) loop through orders and cancel them one at a time. With 2-4
open orders, this is 2-4 sequential HTTP calls (400-1200ms).

### Changes

**`orders.py:951-969`** — Replace loop with batch:
```python
def _cancel_open_orders(self):
    if not self.open_orders:
        return
    order_ids = [o["order_id"] for o in self.open_orders
                 if not o.get("_verify_pending")]
    if order_ids and not self.dry_run:
        self.client.cancel_orders(order_ids)
    # Then verify each (still sequential — needed for fill detection)
    for order in list(self.open_orders):
        self._verify_and_reconcile(order)
```

**Problem**: The verify step after cancel is the real latency hog. Each
`get_order` call is ~100-200ms. For 3 orders that's 300-600ms of verification.

**Solution**: Add a `get_orders_batch` method to Rust that fetches multiple
order statuses. polyfill-rs doesn't have a batch get_order, but we can use
`tokio::join!` to run them concurrently:

**`rust/src/client.rs`** — Add:
```rust
fn get_orders_batch(&self, py: Python<'_>, order_ids: Vec<String>) -> PyResult<PyObject> {
    let client = self.inner.clone();
    let results = py.allow_threads(move || {
        RUNTIME.block_on(async move {
            let futs: Vec<_> = order_ids.iter()
                .map(|id| client.get_order(id))
                .collect();
            futures::future::join_all(futs).await
        })
    });
    // ... convert to Python list of dicts
}
```

**Expected savings**: End-of-window cleanup goes from O(n)*200ms to ~200ms
constant time. For 3 orders: ~400ms saved.

**Priority**: MEDIUM-LOW — only fires once per window (at withdrawal time),
not on hot path. But reduces the window-transition blackout period.

---

## Phase 4: Eliminate asyncio.to_thread overhead

**Problem**: `live_trader.py:185` wraps `tracker.evaluate()` in
`asyncio.to_thread()`. This spawns a thread-pool task, acquires the GIL,
runs the synchronous method, then returns. The thread dispatch adds ~0.5-2ms
per evaluation cycle.

**Solution**: Since `tracker.evaluate()` is CPU-bound Python (signal math)
plus blocking Rust calls (which already release the GIL via `py.allow_threads`),
the `to_thread` wrapper is correct for not blocking the event loop. However:

1. The Rust `place_order` call inside `_place_limit_order` already releases
   the GIL. The `to_thread` is redundant for the HTTP portion — it only
   helps for the ~1ms of signal computation.

2. We could move the blocking Rust calls to their own `to_thread` calls
   and keep signal computation synchronous. But this adds complexity for
   minimal gain.

**Recommendation**: Leave `to_thread` as-is. The ~1ms overhead is negligible
compared to the 100-300ms HTTP savings from batching.

**Priority**: LOW — <2ms savings

---

## Phase 5: Pre-sign orders during signal evaluation

**Problem**: Order signing (ECDSA) takes ~1-3ms. Currently it happens inside
`place_order` / `place_orders` after the decision is made.

**Solution**: Sign orders speculatively during signal evaluation, before the
decision is final. If the signal says "likely BUY UP at 0.53", pre-sign
that order while finishing the evaluation. If the decision confirms, post
immediately; if not, discard.

### Changes

**`rust/src/client.rs`** — Add `create_signed_order` that returns a serialized
signed order without posting:
```rust
fn create_signed_order(&self, py: Python<'_>, token_id: &str, price: f64,
    size: f64, side: &str) -> PyResult<PyObject> {
    // create_order without post_order
}

fn post_presigned_orders(&self, py: Python<'_>, signed_orders: Vec<PyObject>,
    order_type: &str) -> PyResult<PyObject> {
    // post_orders with pre-signed data
}
```

**Complexity**: HIGH — requires passing serialized signed orders through Python,
and the signature includes a nonce that may expire. Not worth it for 1-3ms.

**Priority**: LOW — minimal savings, high complexity

---

## Phase 6: Signal wake optimization

**Problem**: `signal_ticker` (live_trader.py:65-80) waits for `wake_event` with
a 10ms timeout (`signal_idle_s=0.01`). When a feed ticks, it sets `wake_event`
which wakes the signal loop. But the `wake_event.wait()` → `evaluate()` path
still has 5-25ms of minimum interval (`signal_min_interval_s=0.005`).

The minimum interval prevents CPU spin but also delays reaction to fast
Binance price moves. BoneReaper likely runs at <5ms reaction time.

**Solution**: Reduce `signal_min_interval_s` to 0 and instead rate-limit
only the ORDER PLACEMENT (not the signal evaluation). The signal should run
on every feed tick, but only place/cancel orders every N ms.

### Changes

**`live_trader.py:60`** — Change:
```python
FAST_SIGNAL_MIN_INTERVAL_S = 0.005  # currently 5ms minimum between evals
# Change to:
FAST_SIGNAL_MIN_INTERVAL_S = 0.0    # evaluate on every tick
```

**`orders.py`** — Add placement rate limiter:
```python
def _place_limit_order(self, ...):
    now = _time.time()
    if now - self._last_place_ts < 0.050:  # 50ms cooldown between placements
        return
    self._last_place_ts = now
    # ... rest of placement logic
```

**Expected savings**: ~5-15ms faster reaction to Binance dislocations.

**Priority**: MEDIUM — improves signal freshness without increasing HTTP load.

---

## Implementation Priority Order

| # | Phase | Savings | Effort | Priority |
|---|-------|---------|--------|----------|
| 1 | Parallel cancel+place (tokio::join!) | 150-200ms | Medium | HIGH |
| 2 | Batch dual-side placement | 150-200ms | Medium | MEDIUM |
| 3 | Batch end-of-window cancels | 200-400ms | Low | MEDIUM-LOW |
| 6 | Signal wake optimization | 5-15ms | Low | MEDIUM |
| 4 | Eliminate asyncio.to_thread | <2ms | Low | LOW |
| 5 | Pre-sign orders | 1-3ms | High | LOW |

**Phase 1 is the highest-impact change**: it turns every cancel+replace from
~400ms sequential to ~200ms parallel, and the code path (cancel old, place new)
is the most common order lifecycle event.

**Phase 2 enables BoneReaper-style dual-quoting**: post UP and DOWN in one call.
Combined with Phase 1, a full "cancel 2 old + place 2 new" operation goes from
~800ms (4 sequential calls) to ~200ms (1 parallel batch).

**Total potential savings**: 500-800ms per decision cycle on the critical path.
At BoneReaper's 36 trades/window rate, that's 18-29 seconds of cumulative
latency saved per 5-minute window.
