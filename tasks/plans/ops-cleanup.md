# F12-F16 — Operational cleanup batch

Small defensive items collected from operator feedback and code
review. Each is independent and self-contained — pick whichever is
most annoying right now.

## F12 — Suppress / route Rust feed eprintln spam

### Problem

Operator: "I'm getting WebSocket protocol errors sometimes" and
"flash read error at the bottom" cluttering the display. They come
from `rust/src/feed.rs`:
- `[BookFeed] read error: {e}` — line 228
- `[PriceFeed] read error: {e}` — line 525
- `[BinanceFeed] read error: {e}` — line 793
- `[UserFeed] read error: {e}` — line 1056

Each feed has a 30s stale watchdog that auto-reconnects. The errors
aren't a correctness issue — they're noise from the reconnect path.

### Options

A. **Operator-side workaround** (zero code): document that operators
   should run `uv run python live_trader.py ... 2>err.log` to route
   stderr to a file. The display stays clean.

B. **Quiet the Rust prints** (rebuild required): add a verbosity
   flag in `rust/src/feed.rs` and only log if `RUST_LOG=debug`.
   Default = silent reconnect with metrics-only counters.

C. **Capture in Python** (works without rebuild): wrap the entire
   process in a stderr redirect at startup, log to a rotating file.

Recommend (A) for now (zero work), (B) when next touching the Rust
extension for any reason.

## F13 — Persist `pending_fills` across restart

### Problem

`tracker.pending_fills` (filled but not yet resolved positions) lives
in memory only. When the bot restarts mid-window, those positions
exist on Polymarket but the bot has forgotten about them. They get
flagged as "unresolved fill" by the next `new_window()` (just a
warning print) and never resolve in the bot's accounting.

### Fix

In `save_state()`, serialize `pending_fills` (similar to how P12.4
serializes `all_results`). On restart, restore them and let the
window-end resolution path handle them normally.

Catch: pending_fills reference market_slug + condition_id which may
have changed if Polymarket recreated the market under a new id.
Need a defensive check on restore: query Gamma API for each
pending fill's market and confirm it still exists; drop entries that
don't.

### Effort

~3 hours (serialize/deserialize + Gamma sanity check on restore).

## F14 — Atomic file rotation for live_trades JSONL

### Problem

`live_trades_btc.jsonl` is currently append-only and grows forever
(~17MB after a few days). On restart it's re-read for `scan_unredeemed_wins`
which scales with file size. Eventually startup gets slow.

### Fix

When file size exceeds 50MB:
1. Rename to `live_trades_btc.YYYY-MM-DD.jsonl.gz` and gzip
2. Start a fresh `live_trades_btc.jsonl`
3. `scan_unredeemed_wins` only reads the recent (uncompressed) file
4. Old files are read on demand if needed for analysis

### Effort

~2 hours.

## F15 — Per-feed health histograms (p50/p95/p99) on the display

### Idea

Once F1 is in place, surface the rolling p50/p95/p99 staleness for
each feed in the display header instead of the current single
"Vol(20s)" line. Operator gets immediate feedback on feed health.

### Effort

~1 hour after F1.

## F16 — Unit tests for cancel_verify_failed path

### Problem

P11.1 fix (commit 2298ca2) prevented the false-refund bug in
`_cancel_single_order`. But there's no regression test — a future
refactor could re-introduce the bug silently.

### Fix

Add `tests/test_cancel_verify.py`:
- Mock the `client.cancel()` to succeed
- Mock `client.get_order()` to raise `Exception("parse error")`
- Call `_cancel_single_order(order, "test")` and assert:
  - Returns `False`
  - Order STAYS in `open_orders`
  - Order has `_verify_pending=True`
  - Bankroll is UNCHANGED (no refund)
- Then mock get_order to return MATCHED on the next call
- Call `_poll_open_orders` and assert the order moves to pending_fills
  with the right cost

### Effort

~3 hours (set up the test fixture for OrderMixin in isolation is the
hard part — it currently depends on `LiveTradeTracker` state).

---

## Order to do these

1. F12 (option A) — zero work, just document
2. F16 (test for P11.1) — locks in the most critical fix
3. F13 (pending_fills persistence) — closes a real correctness gap
4. F14 (file rotation) — eventual-must-have, not urgent
5. F15 (health histograms) — depends on F1
