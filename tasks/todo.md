 TODO

  ## P0: Multi-timeframe confirmation

  **Problem**: Current arb fires on any 2s Binance move, even when it contradicts
  the 60s trend. In a 2-day uptrend sample:
  - UP trades: 39/49 (80% WR)
  - DOWN trades: 15/35 (43% WR)
  DOWN arbs were fighting the trend and losing $106.

  **Proposed fix**: require 60s direction to agree with 2s direction before firing.

  **Implementation sketch**:
  1. `live_trader.py`: maintain a 60s Binance ring alongside the 10s ring (or
     widen the existing ring's `maxlen`)
  2. `signal_diffusion.py:decide_latency_arb()`: after the 2s delta check, compute
     the 60s delta from the long ring
  3. Gate: if `sign(60s_Δ) != sign(2s_Δ) and abs(60s_Δ) > arb_confirm_threshold`,
     return FLAT with reason "trend disagreement"
  4. Add `--arb-confirm-window-s` (default 60) and `--arb-confirm-threshold-usd`
     (default 15) flags

  **Acceptance**: replay the 84 fills in `live_trades_btc_arb.jsonl` (from the
  main worktree) and confirm DOWN WR improves from 43% → 60%+ without tanking
  UP WR.

  ## P1: Book depth check

  Before firing, verify the ask level has enough size to fill the intended
  notional without slipping to the next level.

  ## P2: Volatility-adaptive threshold

  Replace fixed `arb_delta_usd` with z-score-based threshold (fire when 2s move
  exceeds N·σ of recent 2s moves). Adapts automatically to regime.

  ## P3: Chainlink confirmation

  We have `snapshot.chainlink_price` but don't use it for the arb gate. If
  Chainlink has already moved in the same direction as Binance, signal is much
  stronger (settlement oracle is confirming).

  ## P4: Signal-strength sizing

  Scale `arb_size_usd` with `|Δ| / arb_delta_usd` ratio, capped at 3x base.