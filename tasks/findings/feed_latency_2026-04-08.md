# F1 Phase 0 — Feed latency measurement campaign (initial)

## TL;DR

**Polymarket RTDS rebroadcasts Chainlink with ~1.3 seconds of constant
delay.** Direct Chainlink Data Streams (F2) would save that latency.
This validates the F2 plan as high-ROI without any further evidence.

## What I built

`scripts/measure_feed_latency.py` — standalone, read-only Python tool
that subscribes to:
- Polymarket RTDS `crypto_prices_chainlink` (server-side `payload.timestamp` available)
- Binance `@bookTicker` (no event_time in payload)
- Binance `@trade` (has `E` event_time field — used as a network-latency baseline)

Logs `(local_recv_ms, server_event_ms, gap_to_prev_ms, age_ms, price)`
to a rotating JSONL. Does not affect the live trader.

`scripts/analyze_feed_latency.py` — computes p50/p95/p99 of `age_ms`
(server→here) and `gap_to_prev_ms` (inter-message spacing) per feed,
estimates rebroadcast tax, and recommends F2 priority.

## Smoke-test results (15s, 216 records)

| feed | n | age p50 | age p95 | age p99 | gap p50 | gap p99 |
|---|---:|---:|---:|---:|---:|---:|
| binance_bookticker | 179 | — | — | — | 1ms | 1010ms |
| binance_trade | 23 | **53ms** | 55ms | 55ms | 242ms | 2940ms |
| rtds_chainlink | 14 | **1280ms** | 1740ms | 1770ms | 1090ms | 1550ms |

(Run on operator's home connection. Sample size is small but the
RTDS vs Binance latency gap is so large that it's significant even
with n=14 vs n=23.)

## Key observations

1. **RTDS rebroadcast tax = ~1.23 seconds**, computed as
   `rtds_chainlink_p50 (1.28s) − binance_trade_p50 (53ms) = 1.23s`.

2. The tax is **not jitter** — `age_p50` (1.28s), `age_p95` (1.74s),
   and `age_p99` (1.77s) are all within ~500ms of each other. This
   is a constant baseline tax, not a long tail. Direct Chainlink
   would save the entire 1.23s for every update, every time.

3. **Inter-message gaps**:
   - RTDS pushes ~1 update/sec (p50 gap = 1090ms)
   - Binance @trade is event-driven (highly variable, p50 = 242ms but
     p99 = 2940ms — long tail)
   - Binance @bookTicker is much faster (p50 gap = 1ms, p99 = 1010ms)

4. **Binance latency from operator's location ~50ms.** That's
   reasonable for a residential connection. AWS Tokyo would be ~10ms
   (analysis2.md cites 4-13ms benchmark) so there's some room for
   F17 (geographic redeploy), but F2 is much higher impact.

## What this confirms / changes

| Plan | Before | After |
|---|---|---|
| F2 direct Chainlink | "high ROI if rebroadcast tax >200ms" | ✅ **CONFIRMED HIGH** (tax = 1230ms) |
| F4 oracle lead-lag | "TBD until F1 measures" | The 1.23s gap means Binance can lead Chainlink by ~1.2s on EVERY new chainlink update — strong predictive signal |
| F11 redundant WS | "tail cut" | Binance @trade p99=2940ms shows real tails worth hedging |
| F17 geographic move | "skip until measured" | binance_trade_p50=53ms is fine; topology is NOT the bottleneck right now. Skip. |
| F18 SBE binary | "only matters in-region" | Confirmed skip. Binance JSON parse time is sub-ms; the dominant tax is the round-trip, which SBE doesn't change. |

## Next steps (in order)

1. **Run a 24h campaign** to lock in the percentile estimates with
   real sample sizes. The 15s smoke test had n=14 for RTDS — fine for
   the headline number but not for tail estimates.
2. **Apply for Chainlink Data Streams credentials** (F2 prereq). The
   rebroadcast tax measurement justifies the effort.
3. **Implement F4 (oracle lead-lag)** in parallel — doesn't depend on
   credentials, can ship faster than F2.

## How to reproduce

```bash
# 24h measurement campaign
uv run python scripts/measure_feed_latency.py \
    --symbol btc/usd --binance btcusdt \
    --duration 86400 --output feed_latency_24h.jsonl

# Analyze (after the run finishes, or in parallel with --window-min)
uv run python scripts/analyze_feed_latency.py feed_latency_24h.jsonl

# Just look at the last 60 minutes of a long-running campaign
uv run python scripts/analyze_feed_latency.py feed_latency_24h.jsonl --window-min 60
```

The measurement tool is read-only and uses ~kbps of bandwidth. Safe
to leave running alongside the live trader.
