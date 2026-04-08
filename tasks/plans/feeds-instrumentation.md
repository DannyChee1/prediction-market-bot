# F1 + F3 — Feed latency instrumentation campaign

## Why this is the highest priority

Every other feed-related plan (F2, F4, F10, F17, F18) depends on
knowing the **actual** distribution of staleness for each feed. The
bot has zero per-feed latency telemetry today — only point-in-time
`*_age_ms` gates that compare `now - last_local_recv_ts`. That's
silently corrupted by:

1. **Buffering bursts** — 10 messages arrive in one batch after a 2s
   pause. The last message has age=0 locally but the earliest price
   in the batch is 2s stale.
2. **Silent stalls** — WS connection held open, last local message
   recent, but server stopped publishing. Local age looks fine; the
   server-side timestamp tells the truth.
3. **Network jitter** — p99 tail dominates PnL, but we only ever look
   at the most recent point.

Without measurement, every other optimization is guesswork.

## Concrete plan

### Phase 0 — non-invasive (no behavior change)

1. **Add event_time extraction in `rust/src/feed.rs`** for each
   feed. Each WS message already has a server-side timestamp:
   - Binance @bookTicker: `E` field (ms by default; add
     `?timeUnit=MICROSECOND` to URL for µs precision)
   - RTDS chainlink: `payload.timestamp` (already extracted as
     `last_update_ts` per scripts/rebroadcast notes)
   - Polymarket CLOB book: check if `event_time` exists in the
     message envelope; if not, fall back to local recv only with a
     loud comment.
   - Polymarket CLOB user feed: `timestamp` field if present.
2. **Surface event_time to Python**. Each feed already exposes a
   `last_update_ts()` method or equivalent — extend to also report
   `last_event_time_ms()`.
3. **Bridge the metric in `_poll_*_feed`**. Where we set
   `price_state["last_update_ts"]`, also set
   `price_state["last_event_time_ms"]` and compute
   `event_to_recv_delta_ms = local_recv_ts*1000 - event_time_ms`.
4. **Log to a rotating file**. Append per-tick records to
   `feed_latency_<market>.jsonl`:
   ```json
   {"ts": ..., "feed": "binance", "event_ms": ..., "recv_ms": ...,
    "delta_ms": ..., "gap_to_prev_ms": ...}
   ```
   Keep the last 7 days, rotate daily.

### Phase 1 — analysis script

`scripts/analyze_feed_latency.py`:
- Reads the rotating log
- Computes per-feed p50/p95/p99/p99.9 of `delta_ms`
- Computes inter-message gap distribution (detects silent stalls)
- Cross-correlates Binance event_time vs Chainlink event_time
  (how often does Binance lead by N ms? what's the average lead?)
- Outputs a summary table to stdout and a `feed_latency_summary.json`

### Phase 2 — gate switch (F3)

Once we have honest event_time data, replace the local-recv staleness
gates with event-time gates:

```python
# Old: now - last_local_recv_ts
# New: now - max(event_time_seen)
chainlink_age_ms = current_time_ms - price_state["last_event_time_ms"]
```

This catches silent stalls. Backward compatible — feeds that don't
expose event_time fall back to local recv with a deprecation warning.

### Phase 3 — dynamic gates (extends F9)

Once histograms are populated, gates can be adaptive instead of fixed:
```
max_chainlink_age_ms = max(static_min, p99_chainlink * 1.5)
```
The gate auto-tightens when p99 is good and loosens when bursts
happen. Avoids both false-skip during normal jitter AND unsafe-trade
during real outages.

## Outputs (what to put in tasks/findings/ when done)

| Metric | btc_5m | btc 15m |
|---|---:|---:|
| Chainlink RTDS p50 latency | ?ms | ?ms |
| Chainlink RTDS p99 latency | ?ms | ?ms |
| Binance @bookTicker p50 | ?ms | ?ms |
| Binance @bookTicker p99 | ?ms | ?ms |
| CLOB book p50 | ?ms | ?ms |
| CLOB book p99 | ?ms | ?ms |
| Binance leads Chainlink avg | ?ms | ?ms |
| Empirical max staleness in 7 days | ?ms | ?ms |

These numbers either confirm or reject every plan in tier 1/2.

## Estimated effort

- Phase 0: ~6 hours (Rust + Python plumbing for 4 feeds)
- Phase 1: ~3 hours (analysis script + plotting)
- Phase 2 (event-time gates): ~2 hours
- Phase 3 (dynamic gates): ~4 hours

Total: ~15 hours for the full campaign. Phase 0+1 alone (~9 hours) is
enough to make data-driven decisions about F2, F4, F17.

## Risks

- **Clock skew**: bot's local clock vs exchange clocks. Phase 0 logs
  raw event_ms — analysis script must apply NTP-corrected `now`. If
  clock skew is >50ms the latency numbers are noise. Use chrony or
  set `time.monotonic()` correctly.
- **Rust extension rebuild**: Phase 0 requires `cargo build` of the
  polybot_core extension. Test in dev before live.
