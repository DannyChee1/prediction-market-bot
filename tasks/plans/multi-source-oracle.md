# F10 — Multi-source oracle disagreement

## Premise

Currently the bot has 2 oracle sources for BTC:
- **Binance** (`bookTicker`, fast, watchdog only)
- **Chainlink** (via Polymarket RTDS, slow, used as truth)

Add 2 more reference exchanges:
- **Coinbase** (`wss://ws-feed.exchange.coinbase.com`)
- **Kraken** (`wss://ws.kraken.com/v2`)

Now we have 4 independent BTC price views. The DISAGREEMENT pattern
becomes a feature.

## Why this matters

1. **All 4 agree, Chainlink lags** → strong directional belief that
   Chainlink will catch up. (Same idea as F4 oracle lead-lag, but
   with 4 sources instead of 1.)
2. **3 agree, 1 disagrees** → outlier; the consensus is probably
   right.
3. **All 4 disagree** → market is uncertain, reduce sizing.
4. **Chainlink LEADS the consensus** → weird, investigate.

## Use cases

### As a toxicity feature

Variance across the 4 log-prices = a continuous "disagreement"
metric. When variance is high, market is unstable; downsize. Similar
in spirit to existing `_toxicity` and `_vpin` but more direct.

### As a Kelly multiplier

```python
disagreement = stdev([binance, coinbase, kraken, chainlink])
# Normalize by typical level (rolling window p95)
ratio = disagreement / typical
mult = exp(-ratio)  # 1.0 when ratio=0, 0.37 when ratio=1
```

### As a directional signal (when combined with F4)

If 3-of-4 fast exchanges agree on direction, the lone slow Chainlink
will likely follow. Boost p_model in the consensus direction.

## Concrete plan

### Phase 1 — passive logging (depends on F1)

Add Coinbase + Kraken as Rust feeds in `rust/src/feed.rs` (mirrors
the existing `BinanceFeed` structure). Forward to Python ctx as
`_coinbase_mid` and `_kraken_mid`. **Don't use them in any
decision yet.** Just log alongside existing feeds.

After 1-2 weeks of logging, analyze:
- How often do they disagree with Binance? By how much?
- Is Chainlink always the laggard? (Expected yes.)
- Does any of them ever LEAD Binance? (Probably no — they're all
  centralized exchanges with similar latency from our location.)

### Phase 2 — feature integration

If the analysis shows disagreement is a meaningful signal,
integrate as either:
- New filtration model feature
- New Kelly multiplier
- Both

A/B test on backtest before shipping live.

## Success criteria

Phase 1 must show:
- p99 of `stdev(binance, coinbase, kraken)` > 0.001 (some
  disagreement happens, otherwise the feature is constant)
- Disagreement events are correlated with subsequent volatility
  (so the feature has predictive power)

Phase 2 backtest A/B:
- Sharpe ≥ baseline + 0.05 on at least one BTC market

## Effort

- Phase 1: ~6 hours (2 new Rust feeds + Python wiring + analysis script)
- Phase 2: ~4 hours (feature engineering + retrain + A/B)

## ROI

Medium. The biggest win comes from F4 (lead-lag); F10 is the
generalization. If F4 is positive, F10 is the natural extension. If
F4 is negative, F10 is unlikely to win on its own.

## Dependencies

- F1 (instrumentation) for honest measurement
- Either F4 first (cleaner test of the lead-lag hypothesis with 1
  source) or F10 first (richer feature space, but also harder to
  diagnose)
