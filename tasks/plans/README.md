# Future Plans / Ideas to Explore

Each entry: status, ROI guess, effort, dependencies. Detailed plans
in sibling files. Items from the analysis2.md brainstorm (in
`temporary/`) are folded in. Negative results from past experiments
live in `tasks/findings/` — check there before re-running anything.

## Tier 1 — high leverage, do first

| ID | Idea | Effort | ROI | Status | Detail |
|---|---|---|---|---|---|
| F1 | Feed latency instrumentation campaign | low | unlocks everything else | **phase 0 done** — see [findings](../findings/feed_latency_2026-04-08.md) | [feeds-instrumentation.md](feeds-instrumentation.md) |
| F2 | Direct Chainlink Data Streams subscription | medium | huge if rebroadcast tax >200ms | **CONFIRMED HIGH** (1.23s tax measured) | [direct-chainlink.md](direct-chainlink.md) |
| F3 | Event-time staleness (not local-recv) | low | correctness | not started | [feeds-instrumentation.md](feeds-instrumentation.md) |
| F4 | Oracle lead-lag as profit signal | medium | **negative in backtest**, infra ON | shipped default-off; live A/B pending; see [findings](../findings/f4_oracle_lead_lag_2026-04-08.md) | [oracle-lead-lag.md](oracle-lead-lag.md) |

## Tier 2 — incremental improvements

| ID | Idea | Effort | ROI | Status | Detail |
|---|---|---|---|---|---|
| F5 | Filtration: Tweedie regression on PnL/$ | medium | small | proven negative for plain regression (P10.3); Tweedie distribution might fit better | [filtration-experiments.md](filtration-experiments.md) |
| F6 | Filtration: cost-sensitive classification | low | small-medium | not started | [filtration-experiments.md](filtration-experiments.md) |
| F7 | Hawkes intensity as sizing modifier | low | small | infra exists (commit 46d7f8c); need different consumer than filtration features | [hawkes-sizing.md](hawkes-sizing.md) |
| F8 | Z-confirmation rate-limit (2nd layer spike defense) | low | only matters if sigma-floor + delta-velocity miss a spike; currently 0 spikes/session | [z-confirmation.md](z-confirmation.md) |
| F9 | Continuous staleness penalty (replace hard gates) | low | modest | not started | [continuous-staleness.md](continuous-staleness.md) |
| F10 | Multi-source oracle (Coinbase + Kraken refs) | medium | toxicity feature | not started | [multi-source-oracle.md](multi-source-oracle.md) |
| F11 | Redundant WS connections (p99 hedge) | low | tail cut | not started | [redundant-ws.md](redundant-ws.md) |

## Tier 3 — defensive/operational

| ID | Idea | Effort | ROI | Status |
|---|---|---|---|---|
| F12 | Suppress / route Rust feed eprintln spam | low | UX | known issue (operator complaint about WS read errors flashing); see [ops-cleanup.md](ops-cleanup.md) |
| F13 | Persist `pending_fills` across restart | low | correctness | not started; currently any in-flight position is "lost" on restart |
| F14 | Atomic file rotation for live_trades JSONL | low | data safety | not started |
| F15 | Per-feed health histograms (p50/p95/p99) | low | observability | not started |
| F16 | Unit tests for cancel_verify_failed path | low | regression safety | P11.1 fix not test-covered |

## Tier 4 — infra (skip until measured-needed)

| ID | Idea | Effort | ROI | Status |
|---|---|---|---|---|
| F17 | Move feed collectors to AWS Tokyo (Binance) | high | huge if topology is binding constraint | skip until F1 measures p99 from Toronto |
| F18 | Binance SBE binary mode | medium | only matters in-region | infra exists (`rust/src/feed.rs:735`); flag-gated |
| F19 | Chrony/AWS NTP clock discipline | low | measurement prerequisite for F1, F2 | skip until needed |
| F20 | Event-driven Python feed (no 5ms poll) | medium | 1-5ms savings | skip until F1 confirms polling is bottleneck |

## Tier 5 — strategy R&D

| ID | Idea | Effort | ROI | Status |
|---|---|---|---|---|
| F21 | Add new markets (eth/sol/xrp 5m re-tune) | medium | scaling | data exists but not tuned for live |
| F22 | Position-aware Kelly (drawdown-conditional) | medium | risk control | not started |
| F23 | Ensemble: GBM + Kou + market_blend with learned weights | high | research | speculative |

## Tier 6 — refactors

| ID | Idea | Effort | ROI | Status |
|---|---|---|---|---|
| F24 | Make BacktestEngine read ALL config knobs (not just max_trades_per_window + window_duration) | low | parity | mostly done in P12.1; lurking ones may exist |
| F25 | Add unit tests for orders.py fill bookkeeping | medium | regression safety | none today |

---

## Things explicitly NOT worth pursuing

(From negative experiments — see `tasks/findings/`)

- **Hawkes intensity as filtration FEATURE** — tried, lost on both BTC markets (commit 46d7f8c). Could still work as a sizing modifier (F7).
- **End-of-window resolution snipes** — operator rejected the strategy in past sessions despite backtest evidence (memory: feedback_no_end_window_snipe).
- **PnL regression filtration target** — proven worse than classification (commit 8145b11). Tweedie variant (F5) is the only retry worth trying.
- **Higher btc_5m edge_threshold** — backtest sweep shows lower threshold = higher Sharpe (P11.5 commit, log shows edge=0.04 best).
- **Filtration as binary gate** — replaced by sizing modifier in P10.1.
- **Weekend skip on btc/btc_5m** — memory: project_btc_5m_weekend_underperform; was inverted on 15m, don't ship.

---

## How to add a new plan

1. Add a row to the right tier table above with status `not started`.
2. If the idea needs >1 paragraph, create a sibling `<id>-name.md` file
   with: motivation, design sketch, success criteria, dependencies,
   estimated effort/ROI.
3. When started: change status to `in_progress`. When done (positive or
   negative result): move the writeup to `tasks/findings/` and update
   the row to point at it.
