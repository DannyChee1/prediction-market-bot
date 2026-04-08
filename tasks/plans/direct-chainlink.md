# F2 — Direct Chainlink Data Streams subscription

## Premise

Polymarket BTC Up/Down markets resolve on **Chainlink Data Streams**
(BTC/USD feed). The bot currently consumes a Polymarket *rebroadcast*
of that feed at `wss://ws-live-data.polymarket.com` (topic
`crypto_prices_chainlink`). Polymarket's own UI states "live data may
be delayed by a few seconds."

If the rebroadcast tax is significant (>200ms p50), subscribing
**directly** to Chainlink Data Streams gives the bot a strict edge
over every Polymarket trader who reads the rebroadcast — including
all the bots competing on the CLOB.

This is the highest single-shot ROI item in the entire backlog. It
also has the highest activation cost.

## Prerequisites (must do first)

1. **F1 — feed instrumentation campaign**. Without measurement, we
   don't know whether the rebroadcast tax is 50ms (not worth doing)
   or 2000ms (drop everything and do this). The measurement defines
   the ROI.

2. **Chainlink Data Streams credentials**. Endpoint:
   `wss://ws.dataengine.chain.link/api/v1/ws?feedIDs=...`. Auth: HMAC
   with API key + ed25519. Need to:
   - Find pricing/approval path (Chainlink may require KYC,
     enterprise contracts, or the feed may be free for low-volume
     consumers — unknown without contacting them).
   - Generate the credentials.
   - Provision a clock-discipline solution (chrony or AWS NTP) — the
     auth enforces ≤5s clock skew.

3. **Find the BTC/USD feed ID** Polymarket actually uses. Documented
   in the Polymarket market metadata, but worth confirming directly
   with Polymarket support — there might be edge cases (different
   feed for low-volume markets, fallbacks).

## Design

### Architecture

- New Rust feed: `rust/src/chainlink_direct.rs`. Mirrors the existing
  `PriceFeed` interface (poll-able, exposes `price()`,
  `last_event_time_ms()`, `last_update_ts()`).
- Python wrapper bridges to a new key in `price_state`:
  `"price_direct": ..., "price_direct_event_ts": ...`
- Existing RTDS-based `price_state["price"]` stays as a **fallback**
  and as a measurement baseline (rebroadcast tax = direct - rtds).
- Both feeds run concurrently. The signal layer reads
  `price_state["price_direct"]` first; falls back to
  `price_state["price"]` if direct is stale or unavailable.
- All staleness gates apply to direct first, RTDS second.

### Data plane

- Direct Chainlink emits "reports" — each report has the raw
  observation timestamp + a signed payload. The bot only needs the
  observation_ts and the price. No on-chain verification required
  (the operator trusts Chainlink as the publisher).
- Bandwidth: similar to existing RTDS. Cost difference: probably an
  API fee or rate limit (TBD on credentials).

### Resolution path

`tracker.resolve_window` already walks `price_history` for the last
update at-or-before `end_ts_ms` (P9.1 fix). Once direct chainlink is
authoritative, this should walk **direct** history, not RTDS history.
Add a separate history buffer for direct:
```python
price_state["price_history_direct"]: deque
```

## Success criteria

- **Empirical rebroadcast tax** measured at >200ms p50 (from F1). If
  smaller, this whole effort is low-ROI; document and skip.
- **Backtest A/B**: not directly possible (the bot's parquets only
  contain RTDS-derived chainlink_price). Need to capture the FIRST 7
  days of dual-feed data, then re-run a windowed backtest using the
  direct timestamps as "as-of" inputs to the signal.
- **Live A/B**: run direct + RTDS in parallel for 7 days, log every
  trade decision twice (one based on direct, one based on RTDS), and
  compute the divergence. Trades that would have been different
  measure the actual edge.

## Risks

- **Cost & approval**: Chainlink Data Streams pricing is opaque. If
  it's $5k/month, this is dead. If it's free for low-volume, ship.
  Likely somewhere in between.
- **Existing oracle-lag edge weakens**: the bot's
  `_compute_oracle_lag` (Binance vs RTDS-Chainlink) currently
  exploits the rebroadcast lag indirectly. If we eliminate the lag
  by going direct, that signal weakens. We'd be trading a small
  delta-arbitrage edge for a large look-ahead edge — should be net
  positive but not free.
- **Clock skew enforcement**: Chainlink's ≤5s skew rule means a
  flaky NTP setup will cause auth failures and dropped messages.
  Need chrony/AWS NTP before this goes live.

## Estimated effort

- Credentials + KYC: 1-3 weeks (calendar time, mostly waiting)
- Rust feed implementation: 2-3 days
- Python integration + history wiring: 1 day
- Live A/B logging + analysis: 1 week of parallel running + 1 day analysis
- Total dev time: ~5 days, plus weeks of waiting

## Estimated ROI

Strictly bounded by the empirical rebroadcast tax measured in F1.
Hand-wave: if RTDS is 500ms slower than direct, and the BTC price
moves an average of $10/sec at our trading times, we'd see prices
$5 ahead of every other Polymarket bot — that's directly equivalent
to a few % win rate boost on the marginal edge trades. Could be
+0.1-0.3 Sharpe.

If RTDS is <100ms slower, ROI is near zero — skip.
