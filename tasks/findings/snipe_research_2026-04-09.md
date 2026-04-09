# Resolution Snipe Strategy Research — 2026-04-09

## Strategy
Buy the winning side of a BTC 5m market at 99c near resolution, pocket 1c/share when it settles at $1.00.

## Key Finding: **Strategy is conditionally viable, but narrow**

### Flip Rates by Entry Timing (14,631 windows)

At 99c, you need >99% win rate. The overall flip rate is too high (~3.5%), but when filtered by BTC delta magnitude, there's a viable subset:

| Delta from threshold | Flip rate at tau=1s | Break-even price | Windows (% of total) |
|---------------------|--------------------|-----------------|--------------------|
| $0-$10              | 16.6%              | 83c             | 10.5%              |
| $10-$25             | 3.88%              | 96c             | 16.5%              |
| $25-$50             | 0.20%              | 99.8c           | 19.9%              |
| **$50-$100**        | **0.08%**          | **99.9c**       | **24.8%**          |
| **$100+**           | **0.00%**          | **100c**        | **28.2%**          |

**At delta >= $50 and tau=1s, the strategy is profitable at 99c** (0.08% flip rate << 1% threshold).

### Liquidity at 99c (delta >= $50)
- 63% of qualifying windows have asks at 0.99
- Median ask size: 100 shares (~$99)
- 61% have best_ask at 0.99 or below
- Almost no liquidity at lower prices (98c: 1.3%, 97c: 0.6%)

### Estimated Daily PnL
```
Qualifying windows/day:  288 × 53% = ~153
With liquidity at 99c:   153 × 63% = ~96 fills/day
Net EV per fill:         $0.273 (at 99.92% win rate)
Daily EV:                ~$26/day at $30/trade
```

### Critical Caveats
1. **Competition**: Many bots targeting the same 99c asks. Actual fill rate << 63%
2. **Chainlink oracle lag**: ~30s update frequency means "tau=1s" delta may be 15s stale
3. **Dynamic taker fees**: Polymarket charges higher fees for crypto short-term markets
4. **CLOB availability post-close**: Unknown if orders accepted after `closed=True`

### Risk Model Observation
The erfc-based flip risk model is severely miscalibrated (predicts 0.001% risk when actual is 3.5%). Root cause: sigma estimation from stale chainlink prices underestimates true volatility. **Use delta magnitude directly, not the flip risk model.**

### Sell-Side Variant (More Promising)
If the bot already holds a winning position:
- Sell at 99c immediately vs wait 60-120s for on-chain redemption
- Cost: 1c/share
- Benefit: instant liquidity, skip redemption failures/rate-limiting
- No flip risk (already won)
- This should be tested first as it's lower risk and faster to implement

## Verdict

The "buy at 99c" strategy only works when:
1. BTC has moved $50+ from threshold
2. There's 1-5 seconds left
3. There's liquidity at 99c
4. You can beat other bots to the fill

Expected edge is ~$26/day at $30/trade, scaling linearly with size. But competition likely erodes this significantly. The sell-side variant (sell winning positions at 99c for instant liquidity) is more promising.

## Next Steps
- [ ] Deploy observer script to collect live book + resolution timing data
- [ ] CLOB acceptance test: can orders be placed after `closed=True`?
- [ ] If yes, the risk-free post-resolution variant dominates
- [ ] Test sell-side variant: sell winning positions at 99c immediately
