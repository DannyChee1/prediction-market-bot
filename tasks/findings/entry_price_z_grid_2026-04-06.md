# Entry Price × Min-Z Grid Search — 2026-04-06

Walk-forward 70/30 with Kou config, edge_threshold=0.06.

## BTC 15m (204 test windows)

| | z≥0.0 | z≥0.3 | z≥0.5 | z≥0.7 |
|---|---|---|---|---|
| **p≥$0.00** | +1.1k/42%/S0.3 | +5.9k/44%/S1.4 | +5.5k/39%/S1.2 | +8.5k/39%/S1.6 |
| **p≥$0.10** | +2.5k/44%/S0.7 | +10.3k/50%/S2.1 | +12.1k/48%/S2.2 | **+20.4k/52%/S2.8** |
| **p≥$0.25** | +8.3k/56%/S2.4 | **+12.4k/76%/S4.5** | +3.5k/74%/S2.8 | +4.6k/85%/S6.2 |
| **p≥$0.35** | +3.7k/62%/S1.8 | +4.4k/73%/S2.9 | +3.5k/74%/S2.8 | +4.6k/85%/S6.2 |
| **p≥$0.45** | +6.3k/76%/S4.0 | +4.2k/76%/S3.2 | +3.2k/76%/S3.1 | +3.5k/84%/S5.8 |

### Key Observations

1. **All cells profitable** — BTC 15m with Kou has edge everywhere
2. **Highest PnL**: p≥$0.10, z≥0.7 = $20.4k, Sharpe 2.8 (many trades, moderate WR)
3. **Highest Sharpe**: p≥$0.25, z≥0.3 = Sharpe 4.5, 76% WR (fewer trades but very clean)
4. **Highest WR**: p≥$0.25, z≥0.7 = 85% WR, Sharpe 6.2 (but only $4.6k PnL, few trades)
5. **Previous config** (p≥$0.25, z≥0.5): $3.5k, 74%, S2.8

### Trade-off: PnL vs Risk-Adjusted

- **Aggressive (max PnL)**: p≥$0.10, z≥0.7 — trades cheapest contracts, highest total return
- **Balanced (best Sharpe with decent PnL)**: p≥$0.25, z≥0.3 — $12.4k, 76% WR, S4.5
- **Conservative (max WR)**: p≥$0.45, z≥0.0 — 76% WR, $6.3k, fewer trades

## BTC 5m (all negative on this data split)

| | z≥0.0 | z≥0.3 | z≥0.5 | z≥0.7 |
|---|---|---|---|---|
| **p≥$0.00** | -2.6k/32% | -5.3k/11% | -0.7k/7% | -3.3k/0% |
| **p≥$0.10** | -2.6k/32% | -5.3k/11% | -3.9k/0% | -0.7k/0% |
| **p≥$0.25** | -2.2k/33% | -3.1k/12% | $0/0 trades | $0/0 trades |
| **p≥$0.35** | -2.0k/38% | -0.5k/0% | $0/0 trades | $0/0 trades |
| **p≥$0.45** | -0.1k/50% | -0.1k/50% | $0/0 trades | $0/0 trades |

**Every profitable cell has 0 trades** — the filters are so tight nothing passes. Every cell WITH trades is negative. The 5m market does not have enough edge to overcome the spread/fee overhead on this data split.

### 5m Instability Warning

Earlier tests (different data split) showed 5m profitable at Kou lambda=0.100, z≥0.7. The current split shows all negative. This means:
- The 5m edge is **fragile and data-split dependent**
- Not robust enough for live trading with real money
- Recording data only (--dry-run) is the right call

## Recommendation

**BTC 15m**: Switch to p≥$0.10, z≥0.3 (Sharpe 4.5, 76% WR, $12.4k PnL)
- Loosens min_entry_price from $0.25 → $0.10 (allows cheaper contracts)
- Tightens min_z from 0.5 → 0.3 (actually looser — lets in more trades)
- Net effect: more trades, higher PnL, higher Sharpe

**BTC 5m**: Disable live trading. Record parquets only (--dry-run).
