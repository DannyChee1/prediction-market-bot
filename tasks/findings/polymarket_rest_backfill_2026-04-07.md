# Polymarket REST Backfill — 2026-04-07

## TL;DR

Built `analysis/polymarket_rest_backfill.py` to backfill historical BTC
up/down 5m parquets directly from Polymarket REST APIs. Added **937 new
windows** to `data/btc_5m/` in 12.5 minutes (1662 → 2599 = +56% sample).
**Phase 1 (`market_blend=0.3`) validates strongly on the bigger sample**:
ROI 16.9% → 33.8%, Sharpe 1.96 → 3.48 vs the OLD config on the same
data.

## What we tried first (and discarded)

1. **`kagglehub` `marvingozo/polymarket-tick-level-orderbook-dataset`**
   (38 GB Kaggle dataset). Promising on paper — full L2 ticks for 21
   days — but actual yield was poor:
   - Only **223 BTC up/down markets** in the 2026-03-26 day file (~10%
     of dollar volume)
   - **Top markets are MISSING** (e.g., highest-volume 9:00AM-9:05AM ET
     window with $592k vol is absent)
   - **Only one side per market** in the file (yes OR no token, not
     both — would need to merge)
   - Selection bias is unexplained
   - Total realistic yield ≈ 1500-2500 windows for 38 GB download
   Kept 13/21 days (~26 GB) for sanity-check / supplemental fill, but
   not used as the primary path.

2. **`/prices-history`** endpoint (CLOB). Works but max granularity is
   `fidelity=2` (2-min bars). Only 3 data points per 5m window — too
   coarse for tick-level backtest.

## What worked: REST trade tape + Binance kline proxy

Three free public endpoints:
- `gamma-api.polymarket.com /events?slug={slug}` — market metadata
  (conditionId, clobTokenIds, endDate, outcomes)
- `data-api.polymarket.com /trades?market={cid}&limit=500&offset=N&takerOnly=true`
  — full per-market trade tape (BUY/SELL, price, size, ts, outcome).
  Paginated; offset cap is ~3500. Returns ~1-3k trades for typical 5m
  window.
- `api.binance.com /api/v3/klines?symbol=BTCUSDT&interval=1s&startTime/endTime`
  — 1-second BTCUSDT OHLC (Chainlink proxy).

### Bid/ask reconstruction

The trade tape gives us BUY/SELL prices but no resting book state. We
approximate per-second bid/ask from rolling latest trades:

```
BUY trade  on Up at price X → ask_up ≈ X (taker hit the ask)
SELL trade on Up at price Y → bid_up ≈ Y (taker hit the bid)
```

With cross-side fallback `bid_up ≈ 1 − ask_down` when one side is silent.
Median spread on the converted parquets matches live (~$0.01).

### Pipeline (per market)

```
slug → fetch_event() → cid + token_up + token_dn   (~75ms)
        ↓
      ┌─ fetch_trades() — paginate ≤8 pages, cap 4000 trades  (~1.5s)
      └─ fetch_binance_klines() — 1s OHLC, 301 rows           (~200ms)
        ↓ ThreadPoolExecutor in parallel
      build_parquet() → per-second snapshot DataFrame         (~1s)
        ↓
      *.parquet (75 cols, depth-5 columns NaN since unavailable)
```

## Throughput

Tested with 4 workers + 429 backoff:
- **288 markets in 258 sec = 1.1 markets/sec** sequential equivalent
- Single-day 1-day backfill: ~5 minutes
- 7-day backfill (2016 slugs): **752 sec / 12.5 min**
  - 937 written, 854 skipped (already existed), 225 no_data, 0 errors

Bottlenecks resolved:
- Initial **HTTP 400 on offset > 3500**: data-api hard cap. Fixed by
  catching the 400 in `fetch_trades` and stopping pagination cleanly.
- **HTTP 429 rate limits with 8 workers**: dropped to 4 workers and
  added exponential backoff. 0 errors with 4 workers + 6-retry backoff.

## Schema parity vs live recordings

| | REST backfill | Live recording |
|---|---|---|
| Columns | 75 | 74 (older) / 75 (newer) |
| Tick density | 1 row/sec (Binance kline interval) | ~3 rows/sec |
| `chainlink_price` | Binance close (proxy) | actual Chainlink WS |
| `binance_mid` | Binance close | Binance bookTicker |
| `best_bid_*` / `best_ask_*` | reconstructed from trade tape | live book WS |
| `bid_px_*_N` / `ask_px_*_N` (depth-5) | NaN (unavailable) | live L2 depth |
| `last_trade_*` | from trade tape | live WS |
| `size_bid_*` / `size_ask_*` | NaN | live |

The depth-5 columns are NaN — Polymarket exposes no historical L2 depth
endpoint. Our Phase 1 (`market_blend`) doesn't use depth, so this is
acceptable for validation. **A backtest that depends on `bid_depth5_*`
or per-level book state will not work on REST-backfilled parquets.**

## Validation results

Re-ran `verify_blend.py` on the latest 297 windows (mix of new REST and
existing live):

| Config | Trades | WR | PnL | ROI | Sharpe |
|---|---:|---:|---:|---:|---:|
| Phase 0 (no blend) | 111 | 47.7% | +$7.65 | +16.9% | 1.96 |
| **Phase 1 (blend=0.3)** | **176** | **48.3%** | **+$21.47** | **+33.8%** | **3.48** |

Compare against the original 291-window verify (before backfill):
- Original: $10.87 PnL, 23.6% ROI, Sharpe 2.16
- After backfill: $21.47 PnL, 33.8% ROI, Sharpe 3.48 (**+61% Sharpe**)

Multi-slice CV (5 slices) confirms Phase 1 is positive everywhere. The
latest-200 slice shows the strongest improvement: 1.49 → 3.47 Sharpe
(+135%).

## Out of scope

- **30-day backfill**: 30 × 288 × 0.4 sec/market ≈ 2 hours wall clock.
  Easy to run. Deferred so the current sample (2599 windows) can be
  validated against live performance over a few days first.
- **15m backfill**: same script, just `--interval 15m`. Should yield
  ~600-800 new 15m windows. Deferred.
- **ETH/SOL/XRP backfill**: needs minor refactor to take a market_key
  parameter and use the right Binance symbol + slug prefix. Deferred.
- **L2 depth columns**: Polymarket has no historical depth endpoint.
  Either record forward or accept depth-free backtests.
- **Kaggle merger**: 13 days of Kaggle orderbook downloaded but not yet
  converted. Lower priority because REST already covers the same
  windows + more, with cleaner schema mapping.

## Files added

- `analysis/polymarket_rest_backfill.py` — the converter script
- `data/btc_5m/btc-updown-5m-*.parquet` — 937 new files (Apr 1-7 range)
- `data/kaggle_raw/` (gitignored) — 13 Kaggle orderbook days + labels
  (~26 GB, kept for sanity checks)

## How to extend

```bash
# 30-day BTC 5m backfill (~2 hours)
uv run python analysis/polymarket_rest_backfill.py \
  --interval 5m --start 2026-03-08 --end 2026-04-07 --workers 4

# BTC 15m backfill
uv run python analysis/polymarket_rest_backfill.py \
  --interval 15m --start 2026-04-01 --end 2026-04-07 --workers 4

# Single-slug debug
uv run python analysis/polymarket_rest_backfill.py \
  --slug btc-updown-5m-1775508300 --out-dir /tmp/debug
```
