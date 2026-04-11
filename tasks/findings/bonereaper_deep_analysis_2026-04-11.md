# BoneReaper Deep Strategy Analysis -- 2026-04-11

## Profile Summary

- **Address**: `0xeebde7a0e019a63e6b476eb425505b7b3e6eba30`
- **Joined**: March 25, 2026 (~16 days of activity)
- **Realized PnL**: +$357,782
- **Volume**: $39,457,464
- **Predictions**: 10,506
- **Edge per dollar**: $357,782 / $39,457,464 = **0.907%** (~0.9 cents per dollar traded)
- **Avg trade size**: $39.4M / 10,506 = **$3,755** in volume per prediction
- **Daily volume**: ~$2.47M/day (16 days)
- **Daily PnL**: ~$22,361/day

## Dataset

Extracted ~900 trades from the Polymarket data-api (limit=3500, but API returns
the most recent trades first). Dataset covers primarily April 10, 2026 with some
earlier records. All analysis below is based on this sample.

---

## 1. CRITICAL FINDING: BoneReaper is a Two-Sided BUYER Who Exits via MERGE

### Evidence A: ZERO sell trades -- they ONLY buy

Across the entire dataset (~900+ records), **every single TRADE has side="BUY"**.
There are literally zero SELL trades. BoneReaper never sells contracts on the CLOB.
They exit positions exclusively through on-chain MERGE (combine Up+Down -> USDC)
and REDEEM (cash in winning side after resolution).

This eliminates the traditional "market maker posting bid+ask" hypothesis. They
are NOT quoting two-sided limit orders. Instead, they are aggressively BUYING
both Up and Down contracts, then MERGEing the matched pairs for profit.

### Evidence B: They buy BOTH sides of every window

In virtually every 5-minute window observed, BoneReaper buys **both** Up and Down
contracts. This is the defining signature of a market maker, not a directional bettor.

Example from window `btc-updown-5m-1775879100` (a single 5m window):
```
t+5s:   BUY Down 0.50 (78 shares, $40.50)
t+5s:   BUY Down 0.51 (78 shares, $41.31)
t+9s:   BUY Up   0.52 (79 shares, $42.64)
t+9s:   BUY Down 0.50 (77 shares, $40.00)
t+9s:   BUY Up   0.50 (78 shares, $40.50)
t+11s:  BUY Up   0.55 (78 shares, $44.71)
t+13s:  BUY Down 0.55 (66 shares, $37.38)
t+13s:  BUY Down 0.56 (78 shares, $45.36)
t+31s:  BUY Down 0.69 (30 shares, $20.70)
t+31s:  BUY Down 0.61 (76 shares, $46.07)
t+33s:  BUY Down 0.70 (95 shares, $68.18)
t+35s:  BUY Up   0.33 (84 shares, $30.41)
t+35s:  BUY Up   0.35 (47 shares, $16.03)
t+47s:  BUY Down 0.62 (85 shares, $52.70)
t+51s:  BUY Up   0.43 (78 shares, $34.83)
t+59s:  BUY Up   0.52 (77 shares, $41.60)
... (continues with many more trades on both sides)
```

**In this single 5-minute window, they have 90+ trades, buying both Up and Down.**

### Evidence B: They settle via MERGE, not REDEEM

Throughout the dataset, I found MERGE operations between windows:
```
1775879125 | btc-updown-15m-1775877300 | MERGE | $1,661.02
1775879125 | btc-updown-5m-1775878200  | MERGE | $1,574.91
1775879121 | btc-updown-5m-1775877900  | MERGE | $1,158.37
1775879121 | btc-updown-5m-1775877600  | MERGE | $1,524.06
```

**MERGE** = combining equal quantities of Up + Down tokens to recover the $1 backing.
This is how a market maker exits: you hold N Up shares and N Down shares, and you
merge them back into N USDC (minus the 2% spread you captured).

REDEEM = cashing in the winning side after resolution. Some REDEEMs also appear,
meaning they sometimes have NET directional exposure that resolves.

**The MERGE-to-REDEEM ratio is the smoking gun.** A pure directional trader would
only REDEEM. BoneReaper primarily MERGEs, meaning their core strategy is:

> Buy Up at 0.48, buy Down at 0.48. Each pair costs $0.96. Merge recovers $1.00.
> Profit: $0.04 per pair (before fees). At maker-fee = 0%, this is pure profit.

### Evidence C: Price analysis confirms spread capture

Examining the price distribution across all ~900 trades:

**BUY prices cluster around the midpoint:**
- Prices 0.40-0.60: ~45% of trades (near-50/50 odds = wide spread)
- Prices 0.60-0.80: ~30% of trades (moderate conviction, but still buying BOTH sides)
- Prices 0.80-1.00: ~15% of trades (high conviction, near resolution)
- Prices 0.00-0.40: ~10% of trades (contrarian / far OTM)

**They are NOT buying at 0.95+ to snipe resolution.** The bulk of trading is at
0.40-0.70, which is classic market-making territory.

### Evidence D: Fractional prices confirm TAKER fills sweeping maker resting orders

Examining price precision:
- Many prices are on exact cent boundaries: 0.45, 0.48, 0.50, 0.52, 0.55, 0.61, 0.68
- Many prices are at fractional values: 0.5487604938, 0.6572939460, 0.7052261307

The fractional prices are the result of **sweeping multiple resting limit orders in
a single market order**. Example:
```
BUY Down at 0.6900000538770104 (12.06 shares)
BUY Down at 0.69                (30 shares)
BUY Down at 0.6900000164300008  (32.26 shares)
BUY Down at 0.6900001188333869  (3.87 shares)
BUY Down at 0.61                (75.53 shares)
```

The tiny deviations from 0.69 (like 0.6900000538) are the result of the CLOB
matching engine splitting a market order across multiple resting limit orders at
slightly different prices. **This proves BoneReaper is placing market orders (taker),
not limit orders.**

Wait -- but market makers place limit orders, not market orders. What's going on?

### THE ACTUAL STRATEGY: Aggressive Two-Sided Accumulation

BoneReaper is doing something more nuanced than pure passive market-making:

1. **Early in the window (t+0 to t+60s)**: Buy BOTH Up and Down aggressively via
   market orders, sweeping available liquidity on both sides of the book.

2. **Throughout the window**: Continue accumulating on whichever side has cheaper
   prices, maintaining roughly balanced exposure.

3. **Near resolution (t+200 to t+300s)**: If the price has moved strongly in one
   direction (e.g., Up is now 0.85), they buy more of the winning side to increase
   their net directional exposure.

4. **After resolution**: MERGE the balanced portion (Up + Down pairs -> $1 each),
   REDEEM the net directional exposure on the winning side.

---

## 2. Timing Analysis

### Trades by seconds into the 5m window:

Examining the `btc-updown-5m-*` trades, computing `trade_ts - window_start_ts`:

```
0-10s:    Heavy cluster of trades (both sides)
10-30s:   Very heavy trading
30-60s:   Heavy trading
60-120s:  Moderate trading
120-180s: Moderate trading
180-240s: Increasing (approaching resolution)
240-300s: Very heavy (final minute accumulation)
```

**Key insight**: Trading is distributed throughout the entire window, not
concentrated at the end. The first trades appear within **3-5 seconds** of
window open. This is NOT resolution sniping -- it's continuous market-making.

### Entry latency

Earliest trades in several windows appear at t+3s to t+5s after window open.
This suggests:
- They detect the new market quickly (~1-2s after open)
- They can place orders within ~1-3s of detection
- Total latency: ~3-5s from market open to first fill

This is NOT sub-second HFT. Our ~300ms order latency is MORE than sufficient
to match their entry speed.

---

## 3. Position Management Within a Window

### Pattern: Dual-Sided Accumulation

For window `btc-updown-5m-1775878500`, tracking the net position over time:

```
Time    Action        Net UP shares (cumulative)
t+7s    BUY Up 0.54   +30
t+7s    BUY Up 0.54   +110
t+7s    BUY Up 0.50   +188
t+13s   BUY Down 0.56 -80 (net = +108)
t+17s   BUY Down 0.58 -80 (net = +28)
t+23s   BUY Down 0.58 +/-  (net oscillates)
t+29s   BUY Up 0.60   +60
t+35s   BUY Up 0.64   +87
t+37s   BUY Up 0.71   +105
t+39s   BUY Up 0.71   +99
t+41s   BUY Up 0.76   +111
t+63s   BUY Down 0.39 (rebalancing)
... continues
```

**They oscillate between buying Up and Down, keeping net exposure moderate,
but consistently letting the net position drift toward whichever side has
better odds.** When Up goes from 0.50 to 0.70, they buy more Up (riding
the trend). When it reverses, they buy Down to rebalance.

### MERGE sizes confirm balanced positions

MERGE operations show sizes of $1,158 to $1,661 per window. At ~$40 average
trade size, that's ~30-40 pairs being merged. This means they have 30-40
"matched" pairs (Up+Down) plus some net directional leftover.

---

## 4. Fee Analysis: Why This Works

### Polymarket fee structure for BTC 5m/15m markets

- **Taker fee**: 7.2% of min(price, 1-price)
  - At p=0.50: fee = 7.2% * 0.50 = 3.6 cents per share
  - At p=0.30: fee = 7.2% * 0.30 = 2.16 cents per share
- **Maker fee**: 0%

### But BoneReaper is taking (crossing the spread)!

If they're paying 3.6 cents taker fee on each trade, and buying both sides:
- Buy Up at 0.48: cost = $0.48 + $0.0346 fee = $0.5146
- Buy Down at 0.48: cost = $0.48 + $0.0346 fee = $0.5146
- Total cost per pair: $1.0292
- Merge recovery: $1.00
- **Loss**: -$0.0292 per pair

**This doesn't work with taker fees.** So either:

1. **They have a special fee arrangement** (whitelisted maker or reduced fees)
2. **They ARE placing limit orders** and the fractional prices are just the
   CLOB matching engine's rounding
3. **Their edge comes from the DIRECTIONAL component**, not the spread capture

### Corrected interpretation: BUY-only taker with MERGE exit

Since ALL trades are BUY-side, the fractional prices (0.6900000538) are definitively
the result of market orders crossing multiple resting limit orders at slightly
different prices. BoneReaper is a TAKER on every trade.

But wait -- if they pay 7.2% taker fees, does the math work?

**Scenario: Buy Up@0.48 + Buy Down@0.48, then MERGE**

Taker fee on Up: 7.2% * min(0.48, 0.52) = 7.2% * 0.48 = $0.03456/share
Taker fee on Down: 7.2% * min(0.48, 0.52) = 7.2% * 0.48 = $0.03456/share
Total cost per pair: $0.48 + $0.03456 + $0.48 + $0.03456 = $1.02912
MERGE recovery: $1.00
**Net: -$0.0291 per pair (LOSS)**

So pure symmetric buying-and-merging at 50/50 prices is a LOSER after taker fees.

**Scenario: Buy Up@0.35 + Buy Down@0.35, then MERGE**

This only works if both sides are priced below $0.50 -- which happens when the
book is mispriced or during brief dislocations.
Total cost: $0.35 + $0.0252 + $0.35 + $0.0252 = $0.7504
MERGE: $1.00
**Net: +$0.2496 per pair (HUGE WIN)**

But you can't normally buy both sides below 0.50 simultaneously (they should
add to ~1.00). This only works if:
1. There's a brief moment when both sides have asks below 0.50
2. They buy at different times and the prices average below 0.50 per side

**Scenario: Directional taker with MERGE hedge**

The most likely explanation:
- Buy Up aggressively when they believe Up will win (informed by Binance)
- Also buy some Down as a hedge (smaller size, near-50/50 prices)
- If Up wins: the Up position pays $1/share, the Down position is worthless
  but they've hedged some risk
- MERGE the matched portion (min of Up shares, Down shares) to recover capital
- Net profit = directional gain on the unmatched portion

Example from the data:
Window btc-updown-5m-1775878800:
- Bought ~400 Down shares at avg ~0.20 (cheap, far OTM) = ~$80 spent
- Bought ~350 Up shares at avg ~0.85 (expensive, near ITM) = ~$297 spent  
- Total spent: ~$377 + ~$29 fees = ~$406
- If Up wins: Up shares pay $350, Down shares = $0, MERGE 0 pairs
  Net: $350 - $406 = -$56 (LOSS on this window)
- If Down wins: Down shares pay $400, Up shares = $0  
  Net: $400 - $406 = -$6 (small loss)

Hmm, this doesn't add up either. Let me reconsider.

**The real answer: They're paying taker fees but making it up on PRICE IMPROVEMENT**

Looking at the actual data more carefully: they buy Up at 0.43 early in the window,
then buy Up at 0.86 late in the window. The SAME Up contract. Early cheap buys
represent "correct" fair value before the market catches up. Late expensive buys
represent resolution conviction.

Their edge is NOT in the spread between Up and Down prices. Their edge is:
1. **Buying early at depressed prices** before the market prices in BTC movements
2. **Informed taker**: using Binance feed to know fair value before the book adjusts
3. **Resolution accumulation**: piling into the winning side near the end

The MERGE operation is just capital-efficient risk management -- reclaiming the
cost of the losing side's hedge.

At their actual daily PnL of ~$22,361 on ~$2.47M volume, they need ~0.9% edge.
On a 50/50 market where they know the direction (say) 60% of the time:
- 60% of the time: buy winning side at avg 0.50, pays $1 = $0.50 profit
- 40% of the time: buy losing side at avg 0.50, pays $0 = $0.50 loss
- Net per trade: 0.60*$0.50 - 0.40*$0.50 = $0.10 edge minus fees

With ~$40 avg trade, fee = ~$1.44 per trade (7.2% * $20).
Net per trade: $0.10 * 80 shares - $1.44 = $6.56 per trade.
At 657 trades/day: $6.56 * 657 = $4,309/day.

This underestimates their actual PnL, suggesting their win rate or their
price improvement is better than this naive model. Likely they have a
MUCH better than 60% directional accuracy thanks to the Binance feed.

---

## 5. The Three Pillars of BoneReaper's Edge

### Pillar 1: Informed Directional Taker (Primary Edge)

- They watch Binance BTC price in real-time (or an equivalent exchange feed)
- When BTC moves, they know the fair value of Up/Down BEFORE the Polymarket
  book catches up
- They aggressively BUY the correct side, sweeping available liquidity
- This is "picking off stale quotes" -- the resting limit orders on the book
  are from slower participants who haven't updated their prices yet
- At BTC $84,000 with a threshold at $84,050, a $100 BTC move changes fair
  value of Up from ~0.50 to ~0.35. If resting asks for Up are still at 0.45,
  they're mispriced by 10 cents. BoneReaper buys Down (equivalent to selling
  Up) to capture that dislocation.

### Pillar 2: Two-Sided Accumulation with MERGE Hedge

- They buy BOTH Up and Down throughout the window, but with NET directional bias
- The losing side is hedged by MERGEing matched pairs (Up+Down -> $1)
- Example: buy 100 Up shares at avg 0.55 and 80 Down shares at avg 0.40
  - MERGE 80 pairs: recover $80, cost was $80*0.55 + $80*0.40 = $76 -> profit $4
  - Net 20 Up shares held to resolution: cost $20*0.55 = $11
  - If Up wins: $20 payout - $11 cost = $9 directional profit
  - Total window: $13 profit before fees

### Pillar 3: Resolution Timing (Late-Window Conviction)

- With 30-60 seconds left, they KNOW (high confidence) which side will win
- BTC is $150 above threshold -> Up is ~0.95 fair value
- They buy Up at whatever price is available (0.79, 0.84, 0.87) 
- These shares pay $1.00 at resolution: guaranteed 13-21 cents per share
- This is visible in the data: clusters of high-price buys at t+230s to t+295s

### How the pillars interact:

1. **Early window (0-60s)**: Informed taker. Buy both sides near 50/50 when
   mispriced relative to Binance. Accumulate matched pairs for MERGE.
2. **Mid window (60-180s)**: Directional drift. As BTC moves, buy more of
   the winning side. Still buy some of the losing side cheaply for hedging.
3. **Late window (180-300s)**: Maximum directional conviction. Pile into the
   winning side at 0.75-0.90. These are near-certain wins at resolution.
4. **After resolution**: MERGE the matched pairs, REDEEM the winning net
   directional exposure. Deploy freed capital to next window.

---

## 6. Volume and Scale

### How they deploy $39.4M in volume over 16 days

- 10,506 "predictions" (market-side entries)
- Average $3,755 per prediction
- But each "prediction" may consist of multiple fills (the API shows individual
  fills, and they report 10,506 predictions vs our ~900 individual fills visible)
  
### Per-window deployment

From the data, in a single 5m window they deploy:
- 40-90 individual fills
- $1,000-$2,500 total USDC across both sides
- Net directional exposure: typically $200-$500 after netting Up vs Down

### Capital efficiency

They don't need $2,500 sitting idle per window. They:
1. Deploy capital at window open
2. MERGE matched pairs immediately after resolution (or even during the window
   if they're confident about the spread)
3. Redeploy the freed capital to the next window

With 288 five-minute windows per day, if they cycle $2,000 per window and
MERGE after each, their actual capital requirement is only ~$2,000-$5,000
(enough for 1-2 concurrent windows).

---

## 7. Can We Replicate This?

### What we'd need:

| Requirement | BoneReaper | Us | Gap |
|---|---|---|---|
| Order latency | ~3-5s to first fill | ~300ms per order | We're FASTER |
| Taker fee | 7.2% | 7.2% | No gap |
| BTC price feed | Binance or similar | Binance (10ms) | No gap |
| Book visibility | WebSocket book feed | WebSocket book feed | No gap |
| Fair value model | Likely simple (delta) | Diffusion model | We may be better |
| MERGE capability | Yes (on-chain) | Not implemented | MODERATE GAP |
| Bankroll | ~$5,000+ cycling | $100 | CRITICAL GAP |
| Concurrent windows | Multiple 5m+15m+1h | Single market pair | MODERATE GAP |
| Bot sophistication | Custom, battle-tested | Python+Rust | MODERATE GAP |

### Bankroll is the binding constraint, but...

At $100 bankroll, we can deploy ~$25 per trade. BoneReaper deploys ~$40-80
per trade. Our per-trade economics are similar, just scaled down.

The critical question is NOT "can we match their volume?" but "can we find
enough stale-quote dislocations to make $10-50/day?"

### The competition problem

BoneReaper is clearly racing against other informed takers for the same
stale quotes. When BTC moves $50 and there are 100 shares of stale Down
asks at 0.50 (fair value 0.35), multiple bots are trying to buy those
shares simultaneously. The fastest bot gets the fill.

At 300ms latency from home, we're actually competitive -- BoneReaper's
first fills appear at t+3-5s after window open, suggesting they're NOT
particularly fast. The book-staleness windows likely last 2-10 seconds
as passive market makers slowly update their quotes.

But in a race with other bots at 15ms latency (Dublin/AWS), our 300ms
is a disadvantage. We'd miss ~20x more fill opportunities than a
co-located bot.

---

## 8. Proposed Strategy: "Mini-BoneReaper"

### Core concept

Informed directional taker on BTC 5m windows, using Binance price feed for
real-time fair value calculation. Buy the underpriced side when the Polymarket
book lags Binance. MERGE matched pairs for capital recovery.

### Strategy A: Stale-Quote Sniper (Simplest to implement)

**When**: The Polymarket book hasn't caught up to a BTC price move.
**What**: Buy the underpriced side as a taker.
**How**:

1. Continuously compute fair_value_up from Binance BTC price + our diffusion model
2. Read Polymarket book: best_ask_up, best_ask_down
3. If best_ask_up < fair_value_up - fee_cost:
   - BUY Up shares (taker), up to $25 per fill
   - This is "stale ask" -- someone left a limit order that's now too cheap
4. If best_ask_down < (1 - fair_value_up) - fee_cost:
   - BUY Down shares (taker)
5. Hold all shares to resolution. Winners pay $1, losers pay $0.

Fee-adjusted threshold:
- Taker fee at p=0.50: 7.2% * 0.50 = $0.036/share
- Need edge > $0.036 per share to be profitable
- fair_value_up - best_ask_up > $0.036 means the book is mispriced by > 3.6 cents
- This happens every time BTC moves $20+ and the book takes >2s to adjust

### Strategy B: Two-Sided Accumulator (BoneReaper clone)

**When**: Throughout each 5m window.
**What**: Buy both Up and Down, but with informed bias toward the winning side.
**How**:

1. At window open, buy small amounts of both Up and Down near 0.50
2. As BTC moves, buy more of the favored side
3. In the last 60s, if direction is clear (|delta from threshold| > $30),
   pile into the winning side
4. After resolution: MERGE the min(up_shares, down_shares), REDEEM the winner

### Strategy C: Late-Window Resolution Play (Most conservative)

**When**: Last 60 seconds of each 5m window.
**What**: Buy the winning side when outcome is nearly certain.
**How**:

1. Wait until t+240s (60 seconds remaining)
2. If BTC is $50+ above threshold: buy Up at whatever's available
3. If BTC is $50+ below threshold: buy Down
4. Hold 60 seconds, resolve for $1.00 per share
5. Skip if delta < $50 (too risky -- might flip)

Expected PnL from snipe research: ~$26/day, but competition reduces this.

### Which strategy to implement FIRST:

**Strategy A (Stale-Quote Sniper)** because:
- Simplest to implement (just add fair-value computation to existing bot)
- Doesn't require MERGE (hold to resolution)
- Can test with existing infrastructure
- Edge is clear and measurable (Binance vs Polymarket price delta)
- Works at our bankroll ($25 per trade, 4 trades per window max)

### Expected performance at $100 bankroll (Strategy A)

Conservative:
- Dislocations per window where edge > fee: ~30% of windows = 86/day
- Avg edge per dislocation: 5 cents/share (book is 5c stale)
- Shares per trade: 50 ($25 at $0.50/share)
- Gross profit per dislocation: $2.50
- Taker fee per trade: $0.036 * 50 = $1.80
- Net per trade: $0.70
- But 40% of "stale" quotes aren't actually stale (we're wrong): -$25 * 0.40
- Effective daily: 86 * 0.60 * $0.70 - 86 * 0.40 * $1.80 = -$25.80

Wait -- this is negative. The problem is that when we're wrong about the
direction, we lose the full cost. Let me recalculate:

- Per dislocation: buy 50 Up shares at $0.50 = $25 cost + $1.80 fee = $26.80
- If we're right (60%): shares worth $50. Profit = $50 - $26.80 = $23.20
- If we're wrong (40%): shares worth $0. Loss = -$26.80
- Expected per trade: 0.60 * $23.20 - 0.40 * $26.80 = $14.20 - $10.72 = $2.20

But we're not betting on 50/50 markets. We're betting when the book is STALE.
If the book says Up=0.50 but Binance says Up should be 0.70, then:
- Buy Up at 0.50, fair value 0.70
- If fair value is correct (80% of the time): expected value = 0.70 * $1 = $0.70
- Cost = $0.50 + $0.036 = $0.536
- Expected profit: 0.70 - 0.536 = $0.164/share = $8.20 on 50 shares
- If the book was right and we were wrong (20%): loss = $26.80
- Expected per trade: 0.80 * $8.20 - 0.20 * $26.80 = $6.56 - $5.36 = $1.20

At 86 trades/day: $1.20 * 86 = **$103/day** (103% daily return on $100)

This is highly optimistic. Realistically, stale-quote opportunities at this
magnitude are rare with competition. More conservative: 10-20 opportunities/day
at smaller edge: $1.20 * 15 = **$18/day** (18% daily return).

Still excellent if achievable. The key question is: how often are Polymarket
books actually stale relative to Binance, and can we detect it faster than
BoneReaper and other competitors?

---

## 9. What's Different From What We Currently Do

### Current strategy (directional)
- Wait for diffusion model to predict a side (Up or Down)
- Buy that side if edge > 6%
- Hold to resolution
- 1 trade per window, ~$5 per trade
- Result: +$19 on 74 trades (~$0.26 per trade)

### BoneReaper's strategy (market-making + directional overlay)
- Buy BOTH sides continuously throughout the window
- Use Binance feed to estimate fair value
- Cancel stale quotes when BTC moves
- MERGE matched pairs for riskless profit
- Layer directional bets on top when conviction is high
- 40-90 trades per window, ~$40 per trade
- Result: +$22,361/day

### Key differences
1. **Volume**: They trade 50-100x more per window
2. **Strategy**: Two-sided (market-making) vs one-sided (directional)
3. **Exit**: MERGE (riskless) vs hold-to-resolution (risky)
4. **Fee regime**: Maker (0%) vs unclear in our case
5. **Capital**: ~$2,000/window vs ~$5/trade

---

## 10. Implementation Priority

### Phase 0: Measure Stale-Quote Frequency (REQUIRED FIRST)

Before building anything, answer: "How often does the Polymarket book lag
Binance by more than 4 cents on BTC 5m?"

Method:
1. Use existing parquet data (Binance prices + Polymarket book snapshots)
2. For each timestamp: compute fair_value_up from Binance price + diffusion model
3. Compare to best_ask_up from the book snapshot
4. Count instances where best_ask_up < fair_value_up - 0.04
5. If this happens < 10 times/day: Strategy A won't work at scale
6. If > 50 times/day: huge opportunity

**This analysis can be done TODAY with existing data. No new code needed.**

### Phase 1: Stale-Quote Sniper (Strategy A)

If Phase 0 shows sufficient dislocations:
1. Modify existing bot to compute Binance-derived fair value in real-time
2. When dislocation > fee threshold: place BUY market order on underpriced side
3. Hold to resolution (existing logic already handles this)
4. No MERGE needed -- just directional taker bets with better signals

This is essentially what our bot already does (directional betting), but with
the signal source changed from "diffusion model" to "Binance vs book price
dislocation."

### Phase 2: Two-Sided Accumulation (Strategy B)

Requires:
1. MERGE capability (new code: interact with CTF contract on Polygon)
2. Inventory tracking (how many Up vs Down shares do we hold?)
3. Position sizing (how much to buy on each side)
4. Resolution timing (when to MERGE vs REDEEM)

### Phase 3: Late-Window Resolution (Strategy C)

Layer on top of Phase 1/2:
1. In last 60s, check BTC delta from threshold
2. If |delta| > $50: buy winning side aggressively
3. This amplifies the directional component

### Phase 4: Scale and Optimize

- Increase bankroll to $500-$1,000
- Move to Dublin for 15ms latency (helps win stale-quote races)
- Add 15m and 1h markets (larger windows = more time for dislocations)
- Add ETH, SOL, XRP (more markets = more opportunities per hour)

---

## 11. Risks and Honest Assessment

### Why this might NOT work for us:

1. **Competition**: BoneReaper is one of several sophisticated market makers.
   The spreads visible in the book are already tight (2-4 cents). With more
   competition, spreads compress to zero and nobody profits.

2. **Adverse selection**: Our latency (300ms) means we're ~10x slower to cancel
   stale quotes than a co-located bot. Every time BTC moves, we're the last
   to react and the first to get picked off.

3. **Bankroll**: At $100, each adverse-selection hit is a 2-5% bankroll loss.
   We can't survive a bad streak like BoneReaper can with their larger capital.

4. **MERGE complexity**: We've never implemented MERGE. There may be gas costs,
   timing constraints, or other operational risks we haven't considered.

5. **The API view is incomplete**: We're seeing ~900 of their ~10,506 predictions.
   Their actual strategy may be more nuanced than what we can infer from this
   sample.

### What we should do FIRST:

1. **Observe, don't trade**: Run a paper-trading market-making bot for 24 hours.
   Log every quote, fill, and theoretical P&L. See if the math works before
   risking real capital.

2. **Measure adverse selection**: On our existing book data, compute how often
   our resting quotes would have been adversely selected (filled just before
   BTC moved against us).

3. **Verify MERGE economics**: Can we MERGE on Polygon with acceptable gas costs?
   What's the latency? Can we MERGE during a window or only after resolution?

4. **Benchmark against BoneReaper's timing**: Can we match their 3-5s entry
   speed? Our existing infrastructure suggests yes, but test it.

---

## Summary

### What BoneReaper actually does

BoneReaper is an **informed directional taker** who uses exchange price feeds
(likely Binance) to identify when Polymarket book prices are stale. They:

1. **BUY both Up and Down** throughout each 5m/15m window (zero SELL trades in
   the entire dataset -- confirmed via raw JSON inspection)
2. **Bias toward the winning side** based on real-time BTC price vs threshold
3. **MERGE matched pairs** (Up+Down -> $1) to recover capital from the losing side
4. **REDEEM net directional exposure** on the winning side for profit
5. Pay **7.2% taker fees** on every trade but overcome them with 5-20 cent price
   improvements from stale-quote exploitation

### Their edge (0.9 cents per dollar) comes from:

- **Latency arbitrage**: Binance updates in 10ms; Polymarket book quotes update
  in 2-10 seconds. That gap is profit.
- **Resolution conviction**: In the last 60s, buying the winning side at 0.80-0.90
  for a $1 payout is 10-20% return in 60 seconds.
- **Capital efficiency via MERGE**: The losing side's cost is partially recovered,
  reducing effective loss on wrong-way trades.

### What we should do

**Immediate (today)**: Run the stale-quote frequency analysis on existing parquet
data. Answer: "How many times per day does the Polymarket book lag Binance by
more than 4 cents?" This determines whether any BoneReaper-inspired strategy is
viable for us.

**If stale quotes are frequent (>20/day)**: Implement Strategy A (stale-quote
sniper). This requires minimal code changes -- essentially swapping the signal
source from "diffusion model" to "Binance vs book dislocation" in our existing
bot.

**If stale quotes are rare (<10/day)**: The opportunity is too competitive for
our latency and bankroll. Focus on improving our existing directional strategy
instead.

### Key corrected assumptions from previous analysis

1. WRONG: "BoneReaper is a high-frequency taker sweeping dislocations"
   CORRECTED: They ARE a taker (BUY-only, zero SELLs), but they're more accurately
   an "informed accumulator" who buys both sides and exits via MERGE+REDEEM.

2. WRONG: "84% of prices on cent boundaries implies maker"
   CORRECTED: Many fills ARE on cent boundaries, but many are also fractional
   (0.6900000538). ALL trades are BUY-side taker fills. The cent boundaries
   simply mean they're hitting resting limit orders posted at round prices.

3. WRONG: "They might be market-making with 0% maker fees"
   CORRECTED: They pay 7.2% taker fees on every trade. Their edge must overcome
   ~3.6 cents per share in fees. This is why they need 5-20 cent dislocations to
   be profitable -- small-spread market-making would be unprofitable after fees.
