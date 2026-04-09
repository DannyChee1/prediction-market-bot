# F4 — Oracle lead-lag as profit signal (backtest negative)

## TL;DR

The 1.22s rebroadcast tax F1 measured is real, but **using it as a
direct p_model bias does not improve backtest PnL** at any bias
value tested (0.0, 0.05, 0.10) on either btc_5m or btc 15m.

The likely reason: `market_blend=0.3` (default for both BTC markets)
already absorbs the lead-lag signal indirectly via the Polymarket CLOB
mid, and the bias gets diluted to ~70% after market_blend mixes it.

Infrastructure shipped as default-off opt-in. No production behavior
change. Live A/B remains the only honest test of whether the live
CLOB has un-priced lead-lag.

## Hypothesis

From `tasks/plans/oracle-lead-lag.md`:

> When binance_mid > chainlink_price by more than X%, the next
> chainlink update is more likely UP than DOWN. The bot should
> convert this directional belief into a p_model bias.

Variant B (gap-as-bias):
```
gap = (binance_mid - chainlink) / chainlink
bias = oracle_lead_bias * clip(gap / oracle_lag_threshold, -1, 1)
p_model_biased = clip(p_model + bias, 0.01, 0.99)
```

At gap = oracle_lag_threshold (0.2%), bias = oracle_lead_bias.
Default `oracle_lead_bias=0.0` (off). Tested values 0.05, 0.10.

## Implementation

- **`signal_diffusion.py`**: new `_oracle_lead_bias()` helper, applied
  in both `decide()` and `decide_both_sides()` BEFORE filtration so
  the filtration model's z/p features see the bias-adjusted view.
  Stored in ctx as `_oracle_lead_bias` for diagnostics.
- **`backtest.py`**: new `oracle_lead_bias` kwarg in
  `build_diffusion_signal`, new `--oracle-lead-bias` CLI flag.
- **`live_trader.py`**: new `--oracle-lead-bias` CLI flag, plumbed
  via `signal_kw["oracle_lead_bias"]`.

All paths default to 0.0 (off). Opt-in via CLI flag only.

## Backtest A/B (50d, walk-forward 70/30, seed 42)

| Mode | btc_5m PnL | Sharpe | DD | btc 15m PnL | Sharpe | DD |
|---|---:|---:|---:|---:|---:|---:|
| bias=0.00 (off) | $18,191 | 2.39 | 5.4% | $1,390 | 1.80 | 2.6% |
| bias=0.05 | $18,191 | 2.39 | 5.4% | $1,390 | 1.80 | 2.6% |
| bias=0.10 | $18,378 | 2.41 | 5.4% | $1,390 | 1.80 | 2.6% |

The +0.10 case nudges btc_5m by +$187 PnL and +0.02 Sharpe — within
backtest noise. btc 15m is **byte-identical** at all three bias
values.

## Why backtest is essentially blind to F4

1. **market_blend dilutes the bias.** The bias is added BEFORE
   `market_blend`, then market_blend mixes p_model with the contract
   mid: `p_model = (1−blend) · p_model + blend · mid_up`. With
   blend=0.3 (default for both BTC markets), a +0.05 bias becomes
   +0.035 effective. With blend=0.5 it becomes +0.025. The CLOB mid
   already reflects what other Polymarket bots have priced in about
   the lead-lag, so adding our own bias on top is largely redundant
   in backtest.

2. **The parquet recorder anchors both feeds to the same local
   time.** F1 measured the 1.22s tax as `local_recv_time -
   server_event_time` for RTDS. But the parquet stores
   `binance_mid[t]` and `chainlink_price[t]` at the same local
   timestamp t. The TIME-lag arbitrage opportunity (Binance is
   reporting newer info than Chainlink at any given live moment)
   isn't preserved in the parquet — only the PRICE gap is, and the
   existing `_compute_oracle_lag` already exploits that as a safety
   widener.

3. **btc 15m has very few non-zero gap events.** With only 154 test
   trades total, most of them happen during quiet windows where
   `|gap| < oracle_lag_threshold (0.002)` and the bias is zero.

## Why live could still differ

In live:
- The bot reads chainlink with the 1.22s rebroadcast tax (real)
- Other Polymarket bots have the SAME tax — they're all reading
  the rebroadcast
- CLOB mid reflects all of those bots' lagged views
- IF the bot can act on Binance directly while other bots wait for
  the next chainlink push, it has a real edge

In backtest:
- The parquet's CLOB mid was recorded by the same bot at the same
  local time as the chainlink/binance prices
- It already reflects the recorder's view of "Polymarket consensus
  given lagged data"
- There's no separate "live CLOB" that hasn't absorbed the lag

So backtest captures the static price-gap signal but NOT the
"Binance lets us see the future of CLOB consensus" signal. The
latter only exists in live.

## What stays

- `_oracle_lead_bias()` helper in `signal_diffusion.py`
- `oracle_lead_bias` parameter on `DiffusionSignal` (default 0.0)
- `--oracle-lead-bias` CLI flag in both `backtest.py` and
  `live_trader.py` (default 0.0)
- ctx tracking of `_oracle_lead_bias` for live diagnostics

## What didn't ship

- Default value remains 0.0. Not turned on for production.
- No backtest evidence justifies turning it on by default.

## Future test (live A/B)

The only honest test is parallel live runs:
- Bot A: `--oracle-lead-bias 0.0` (control)
- Bot B: `--oracle-lead-bias 0.05` (treatment)
- Same bankroll, same market, run for ≥1000 trades each
- Compare realized PnL and Sharpe

If treatment beats control by ≥+0.1 Sharpe, ship the bias as default.
Otherwise document the second negative result and consider removing.

## Lessons

Same lesson as P10.3 (filtration regression target): a measurement
campaign (F1) can confirm a phenomenon exists, but that doesn't mean
exploiting it as a signal will improve PnL — there might be existing
indirect mechanisms (here: market_blend) that already handle it.
Always run the full backtest A/B before celebrating.
