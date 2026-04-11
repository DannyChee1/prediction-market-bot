# Latency Arbitrage Implementation Plan — 2026-04-11

## 1. Executive Summary

**Recommendation: SHIP a dedicated `decide_latency_arb()` function behind a
new `--latency-arb` CLI flag, running in parallel with (not replacing)
`--stale-quote`.** The central asset is the confirmed 1.23s Polymarket RTDS
rebroadcast tax; the existing `decide_stale_quote` is a GBM fair-value
calculator with a stale-book side effect, not a true latency capture.

A pure delta-trigger signal ("Binance moved +$X in the last W seconds → buy
that side at ask") needs less code than the current implementation, has no
σ sensitivity, fires 10-30× more often, and survives the calm-market filter
for free because it only fires when σ is spiking locally.

**Expected trade frequency (BTC 5m+15m combined, pre-Dublin, 24h):**
~35-80 fires/day at the baseline thresholds below. Post-Dublin, fill rate
goes up and we expect 60-150 trades/day. Not BoneReaper's 652, but 10-50×
our current ~2/day.

**Expected per-trade edge:** 3-8 cents/share gross, 1-3 cents net after fees.
The wide range reflects honest uncertainty — backtest can't reproduce the
1.23s lag (see F4 Oracle Lead-Lag failure), so the real number will be
measured in Phase 2.

**Implementation cost:** ~180 lines of new Python in `signal_diffusion.py`
(new `decide_latency_arb` method + two small helpers in `live_trader.py` to
maintain a Binance-only price ring buffer). Zero Rust changes. No new state
files.

---

## 2. Strategy Overview

### The edge (one sentence)
When Binance moves by more than `delta_usd_threshold` within `window_s`,
Polymarket's CLOB is guaranteed to be at least 1.23s behind on pricing.
Any resting ask on the side that just became "correct" is stale. Cross the
spread at ask; realize the convergence when Chainlink catches up.

### Pipeline

```
Rust BinanceFeed (WS, sub-ms)
  └→ _poll_binance_feed wakes Python (every 1ms)
      └→ signal_ticker (wake_event fires)
          └→ decide_latency_arb(snapshot, ctx)          ┐
              1. Read binance_ring_buffer (last 2s)     │  ≤1ms
              2. Δ_usd = bn_now - bn_t0                 │
              3. |Δ_usd| < threshold ? → FLAT           │
              4. last_fire_ts within cooldown? → FLAT   │
              5. Pick side (SIGN of Δ)                  │
              6. Validate ask sanity                    │
              7. Compute stale_ok (book hasn't caught up)
              8. Fire BUY at ask, tag reason            ┘
          └→ place_market_order (FOK taker) via Rust
              └→ HTTP POST to CLOB                     ~200ms (home), ~2ms (Dublin)
```

### Timing budget (Dublin VPS target)

| Stage                          | Budget | Notes |
|--------------------------------|-------:|-------|
| Binance tick → Python wake     |   1 ms | `_poll_binance_feed.poll_interval_s=0.001` |
| signal_ticker dispatch         |   1 ms | wake_event |
| decide_latency_arb compute     | <1 ms  | ring-buffer linear scan, ≤40 ops |
| Order construction + sign      |   5 ms | existing Rust `place_market_order` |
| HTTP POST to CLOB (Dublin)     |   2 ms | |
| **Total signal→ack**           | **~10 ms** | |

Today from the US home connection the HTTP RTT is ~200ms, so the trade
will race — some will lose to BoneReaper and co-located bots. That's
acceptable during Phase 2 validation. The Dublin move is assumed.

---

## 3. Concrete Signal Definition

### 3a. State (per tracker ctx)

Two new ctx keys, maintained by an augmented `signal_ticker`:

```
ctx["_bn_ring_mid"]  : deque[(ts_ms, mid_price)]   # last 3s of Binance mids only
ctx["_last_arb_fire_ts_ms"] : int                  # most recent fill wall-time
```

The **ring is Binance-only** (not mixed with Chainlink). This is important:
the existing `price_history` in `decide_stale_quote` is polluted with
Chainlink fallbacks when Binance is stale, which would destroy the Δ signal.

Ring size cap: **150 entries** (3 seconds at 50 Hz; Binance bookTicker
bursts well above 50 Hz but we only push on mid-change, so 150 is enough
for the worst-case ~50ms/tick path).

### 3b. Parameters (all tunable, start values in bold)

| Param               | Default | Range to sweep | Rationale |
|---------------------|--------:|---------------:|-----------|
| `delta_usd_thresh`  | **$25** | 15 / 25 / 40   | $25 at BTC=84k ≈ 3.0 bps; one std of the 1-sec realized move during active hours. Below $15 = noise. |
| `window_s`          | **2.0** | 1.0 / 2.0 / 3.0 | Must exceed the 1.23s Chainlink tax so the book is guaranteed stale; 2.0s is the comfortable midpoint. |
| `cooldown_s`        | **4.0** | 3.0 / 4.0 / 6.0 | Prevents re-firing on the same move while allowing a follow-through. Cooldown > window_s by design. |
| `max_ask_up`        | **0.85** | 0.80-0.90    | Never buy into the resolution tail; edge collapses past 0.85 because taker fee still scales with p*(1-p). |
| `min_ask_up`        | **0.15** | 0.10-0.20    | Symmetric OTM floor. |
| `stale_book_ms`     | **600** | 400/600/1000 | Only fire when the Polymarket book hasn't updated in ≥600ms. Same-tick book updates mean someone is already repricing. |
| `require_book_stale`| **True** | bool         | Toggle for the "both" variant (see §4). |
| `min_fire_tau_s`    | **30** | 15/30/60      | Avoid the last 30s where resolution dynamics dominate (per user memo on end-of-window snipes). |
| `max_fire_tau_s`    | **None** | None / 240   | Optional: only fire in the first 4 min of a 5m window. |

### 3c. Pseudocode

```python
def decide_latency_arb(self, snapshot, ctx):
    flat = Decision("FLAT", 0.0, 0.0, "")
    now_ms = int(time.time() * 1000)

    # 0. Tau gates
    tau = snapshot.time_remaining_s
    if tau is None or tau <= 0:
        return flat, flat
    if tau < self.arb_min_fire_tau_s:
        return flat, Decision("FLAT", 0.0, 0.0,
                              f"tau<{self.arb_min_fire_tau_s}")
    if self.arb_max_fire_tau_s and tau > self.arb_max_fire_tau_s:
        return flat, Decision("FLAT", 0.0, 0.0,
                              f"tau>{self.arb_max_fire_tau_s}")

    # 1. Cooldown
    last_fire = ctx.get("_last_arb_fire_ts_ms", 0)
    if now_ms - last_fire < int(self.arb_cooldown_s * 1000):
        return flat, Decision("FLAT", 0.0, 0.0, "cooldown")

    # 2. Feed freshness — reuse existing _check_stale_features
    stale_reason = self._check_stale_features(ctx)
    if stale_reason:
        return flat, Decision("FLAT", 0.0, 0.0, stale_reason)

    # 3. Ring buffer lookup
    ring = ctx.get("_bn_ring_mid")
    if not ring or len(ring) < 2:
        return flat, Decision("FLAT", 0.0, 0.0, "ring warmup")

    bn_now_ts, bn_now_mid = ring[-1]
    cutoff_ms = now_ms - int(self.arb_window_s * 1000)
    bn_t0 = None
    for ts, mid in ring:                # linear scan, <150 entries
        if ts >= cutoff_ms:
            bn_t0 = (ts, mid)
            break
    if bn_t0 is None or bn_now_mid <= 0 or bn_t0[1] <= 0:
        return flat, Decision("FLAT", 0.0, 0.0, "ring empty")

    delta_usd = bn_now_mid - bn_t0[1]
    if abs(delta_usd) < self.arb_delta_usd_thresh:
        return flat, Decision("FLAT", 0.0, 0.0,
            f"Δ=${delta_usd:+.1f} < ${self.arb_delta_usd_thresh:.0f}")

    ctx["_arb_delta_usd"] = delta_usd
    ctx["_arb_window_s"]  = (bn_now_ts - bn_t0[0]) / 1000.0

    # 4. Book sanity
    ask_up, ask_down = snapshot.best_ask_up, snapshot.best_ask_down
    if ask_up is None or ask_down is None:
        return flat, Decision("FLAT", 0.0, 0.0, "no book")
    if not (0 < ask_up < 1 and 0 < ask_down < 1):
        return flat, Decision("FLAT", 0.0, 0.0, "bad asks")

    # 5. Side selection by Δ sign
    if delta_usd > 0:
        side, ask, token_label = "BUY_UP", ask_up, "UP"
    else:
        side, ask, token_label = "BUY_DOWN", ask_down, "DOWN"

    # 6. Entry price band
    if ask > self.arb_max_ask:
        return flat, Decision("FLAT", 0.0, 0.0,
            f"ask {ask:.3f} > max {self.arb_max_ask:.2f}")
    if ask < self.arb_min_ask:
        return flat, Decision("FLAT", 0.0, 0.0,
            f"ask {ask:.3f} < min {self.arb_min_ask:.2f}")

    # 7. Book-stale gate (variant C, recommended default)
    if self.arb_require_book_stale:
        book_age = ctx.get("_book_age_ms") or 0.0
        if book_age < self.arb_stale_book_ms:
            return flat, Decision("FLAT", 0.0, 0.0,
                f"book fresh ({book_age:.0f}ms)")

    # 8. Edge-estimate (OPTIONAL, used only for sizing + telemetry)
    #    Use a naive linear projection: 1 bps Binance → Δp ≈ 1 cent
    #    at p=0.5, scaled by the current side's gamma (1/(2σ√τ)).
    #    This is NOT a trade gate — the gate is delta_usd threshold.
    est_edge = self._estimate_arb_edge(delta_usd, ask, snapshot, side)
    fee = poly_fee(ask)
    net_edge = est_edge - fee
    ctx["_arb_est_edge"] = est_edge
    ctx["_arb_net_edge"] = net_edge

    # 9. Size (fixed — see §5)
    size_usd = min(self.arb_size_usd, self.bankroll * self.arb_max_frac)
    min_usd  = self.min_order_shares * ask
    if size_usd < min_usd:
        if self.bankroll >= min_usd:
            size_usd = min_usd
        else:
            return flat, Decision("FLAT", 0.0, 0.0, "bankroll<min")

    reason = (f"LAT_ARB Δ=${delta_usd:+.1f} w={self.arb_window_s:.1f}s "
              f"ask={ask:.3f} est={est_edge:.3f} net={net_edge:.3f} "
              f"book_age={ctx.get('_book_age_ms',0):.0f}ms ${size_usd:.0f}")
    fill = Decision(side, max(net_edge, 0.01), size_usd, reason)
    if side == "BUY_UP":
        return fill, flat
    return flat, fill
```

Note 1: the decision is **not gated on positive `net_edge`**. The
delta_usd threshold is the entry gate; the edge estimate is advisory
and used only by `tracker._execute_taker` for size bookkeeping / logs.
The reason: the fair-value-to-edge translation depends on σ, and σ
estimation is the exact failure mode that broke the existing
`decide_stale_quote` during calm-market drift. We don't want that
bug back.

Note 2: `_last_arb_fire_ts_ms` is written by `tracker._execute_taker`
on a successful fill, NOT by `decide_latency_arb`. This avoids
firing the cooldown on rejected fills (max_positions hit, etc.).

---

## 4. Direction Logic — Three Variants

| Variant | Entry rule | Expected fire rate | Expected edge/trade |
|---------|-----------|-------------------:|--------------------:|
| **A. Pure impulse** | `|Δ_bn| > $25` in 2s | ~120/day | 2-3c gross |
| **B. Pure convergence (stale-book only)** | Book age > 600ms AND best_ask stale vs binance | ~20/day | 5-8c gross |
| **C. Both (AND)** | A ∧ B | ~35-80/day | 4-6c gross |

### Recommendation: Variant C (default) with Variant A as fallback flag

Reasoning:
- A alone is momentum-following — half the fires are "BTC just moved, book
  already moved with it, we're late and get no edge." Fee-negative in practice.
- B alone has the highest edge but only fires ~20/day — below BoneReaper's
  volume floor.
- C captures the intersection: Binance ran, book hasn't caught up yet. This
  is the literal definition of latency arbitrage.

Ship C as the default. Expose `--arb-variant {impulse,conv,both}` so we
can A/B A vs. C during Phase 2 to confirm C's edge advantage pays off
despite the ~2× lower fire rate.

The "convergence-only" variant B is intentionally not offered — it
requires computing a fair value from σ, which reintroduces the GBM
dependency we're trying to kill.

---

## 5. Sizing

### Fixed size, not Kelly.

- `arb_size_usd = $10` per trade default (scales to `$5-25` via CLI).
- Hard cap: `bankroll * arb_max_frac` with `arb_max_frac = 0.15`
  (prevents single-trade ruin even at $100 bankroll).

### Why not Kelly?
- Kelly sizing on `p_model` is what got us into trouble with
  `decide_stale_quote` during σ collapse (confident fractional bets on
  garbage p_model).
- The latency arb edge is **uncorrelated with Binance Δ magnitude** past
  the threshold. A $30 move and a $200 move produce the same fair-value
  shift relative to the stale book — the book was stale either way. So
  Kelly has nothing to scale on.
- Equal-sized bets give cleaner WR/edge telemetry during Phase 2.

### Optional magnitude-tilt (Phase 3 only)
If Phase 2 confirms positive edge, consider:
```
size_usd = base * (1 + min(|Δ_usd| / delta_usd_thresh - 1, 2.0))
```
Caps at 3× base at |Δ|=3×threshold. Worth ~10-20% extra PnL on paper;
deferred until Phase 2 validates the base case.

---

## 6. Interaction with Existing Filters

### 6a. Test #3 calm-market filter (`min_trade_sigma=2.5e-5`)
**Confirmed non-conflicting.** At BTC=84k, a $25 move in 2s implies a
per-sec log-return magnitude of ~1.5e-4, which is **6× above** the 2.5e-5
floor. The calm filter only refuses when the realized vol has been <$2 in
the last 2 seconds — exactly the regime latency arb does not fire in.

We still keep the stale-feature gate (`_check_stale_features`) because
it catches WS disconnects, not quiescence. Add no new sigma checks.

### 6b. `max_positions` (tracker default 5)
Latency arb is bursty — a single $80 BTC impulse can fire 3 buys on the
same side in 10 seconds if cooldown is bypassed by a fresh impulse. Cap
interaction:
- Current behavior: `tracker.evaluate` returns FLAT with reason
  `max_positions` when hit. Fine — the next impulse simply doesn't fire.
- **No change needed.** But log the skip so we can measure opportunity cost.

### 6c. `max_trades_per_window`
Current live config: `--max-trades-per-window 10` on BTC 5m (per user).
For latency arb this is too low — at peak we expect 3-8 fires per 5m
window during active hours. **Raise to 20 for latency arb mode** (see
§7 step 5).

### 6d. `_last_taker_ts` 30s taker cooldown (tracker.py:524)
**Currently blocks latency arb.** This 30s inter-trade cooldown was added
for the slow diffusion signal; latency arb needs to fire every 4s max.
Fix: make the cooldown mode-aware by reading
`getattr(self.signal, "latency_arb_mode", False)` and using
`self.arb_cooldown_s` instead of hardcoded 30.0.

### 6e. Same-price stacking guard (tracker.py:515-521)
Keep. It prevents double-entering the same level on repeated impulses,
which is what we want.

### 6f. Existing `decide_stale_quote`
**Don't touch.** The new `decide_latency_arb` lives as a sibling method.
Routing happens in `tracker.evaluate`:
```python
if getattr(self.signal, "latency_arb_mode", False):
    up, dn = self.signal.decide_latency_arb(snapshot, self.ctx)
elif getattr(self.signal, "stale_quote_mode", False):
    up, dn = self.signal.decide_stale_quote(snapshot, self.ctx)
else:
    up, dn = self.signal.decide_both_sides(snapshot, self.ctx)
```

---

## 7. Implementation Steps

### Step 1 — Add Binance-only ring buffer to `signal_ticker`
**File:** `live_trader.py:143-149`
Insert after the current `_binance_mid` freshness check:
```python
# Maintain a Binance-only 3s ring for latency arb (don't mix with
# Chainlink fallback — that destroys the Δ signal on WS disconnects).
bn_mid = tracker.ctx.get("_binance_mid")
if bn_mid is not None:
    ring = tracker.ctx.setdefault("_bn_ring_mid", collections.deque(maxlen=150))
    if not ring or ring[-1][1] != bn_mid:
        ring.append((int(_time.time() * 1000), float(bn_mid)))
```
Add `import collections` at the top.

### Step 2 — Add `latency_arb_mode` constructor flag + tunables
**File:** `signal_diffusion.py:293-302` (DiffusionSignal `__init__` kwargs)
Append new parameters:
```python
latency_arb_mode: bool = False,
arb_delta_usd_thresh: float = 25.0,
arb_window_s: float = 2.0,
arb_cooldown_s: float = 4.0,
arb_max_ask: float = 0.85,
arb_min_ask: float = 0.15,
arb_stale_book_ms: float = 600.0,
arb_require_book_stale: bool = True,
arb_min_fire_tau_s: float = 30.0,
arb_max_fire_tau_s: float | None = None,
arb_size_usd: float = 10.0,
arb_max_frac: float = 0.15,
arb_variant: str = "both",   # "impulse" | "both"
```
Store each on `self.` in the body near `self.stale_threshold = ...` (line
~408).

### Step 3 — Add `decide_latency_arb` method
**File:** `signal_diffusion.py:2253` (just above `decide_stale_quote`)
Insert the full pseudocode from §3c above. Estimated 120-150 lines
including the `_estimate_arb_edge` helper. The helper can be a 5-line
approximation:
```python
def _estimate_arb_edge(self, delta_usd, ask, snapshot, side):
    wsp = snapshot.window_start_price or snapshot.chainlink_price
    if not wsp:
        return 0.0
    # Project Binance delta onto the side's implied price shift.
    # rough: Δp = Δ_usd / (2 * wsp * max_ask_spread_approximation)
    # using τ-independent linear approx (not scale-sensitive)
    approx = abs(delta_usd) / (4.0 * wsp)  # ~1c per $40 BTC move
    return approx
```
The estimate is advisory; do not gate on it.

### Step 4 — Route in `tracker.evaluate`
**File:** `tracker.py:429-432`
Replace the if/else block with the 3-way branch from §6f.

### Step 5 — Mode-aware taker cooldown
**File:** `tracker.py:523-525`
Replace:
```python
if hasattr(self, '_last_taker_ts') and (now - self._last_taker_ts) < 30.0:
    return Decision("FLAT", 0.0, 0.0, "taker_cooldown")
```
with:
```python
cooldown_s = (self.signal.arb_cooldown_s
              if getattr(self.signal, "latency_arb_mode", False)
              else 30.0)
if hasattr(self, '_last_taker_ts') and (now - self._last_taker_ts) < cooldown_s:
    return Decision("FLAT", 0.0, 0.0, "taker_cooldown")
```
And update `_last_arb_fire_ts_ms` alongside `_last_taker_ts` on successful
fill (tracker.py:577, add one line inside the `_execute_taker` success path).

### Step 6 — CLI flag wiring
**File:** `live_trader.py:1387` (where `signal_kw["stale_quote_mode"]` is
assigned) and the argparse block above it.
Add:
```python
parser.add_argument("--latency-arb", action="store_true",
    help="Enable latency arbitrage mode (Binance Δ trigger).")
parser.add_argument("--arb-delta-usd", type=float, default=25.0)
parser.add_argument("--arb-window-s", type=float, default=2.0)
parser.add_argument("--arb-cooldown-s", type=float, default=4.0)
parser.add_argument("--arb-variant", choices=["impulse", "both"], default="both")
parser.add_argument("--arb-size-usd", type=float, default=10.0)
parser.add_argument("--arb-dry-run", action="store_true",
    help="Log would-fire events without placing orders.")
```
Then:
```python
signal_kw["latency_arb_mode"]      = args.latency_arb
signal_kw["arb_delta_usd_thresh"]  = args.arb_delta_usd
signal_kw["arb_window_s"]          = args.arb_window_s
signal_kw["arb_cooldown_s"]        = args.arb_cooldown_s
signal_kw["arb_require_book_stale"] = (args.arb_variant == "both")
signal_kw["arb_size_usd"]          = args.arb_size_usd
```
`--arb-dry-run` turns on a would-fire logger described in §8 Phase 1.

### Step 7 — Raise max_trades_per_window under latency arb
**File:** `live_trader.py` near the CLI default assignment — not the
market_config.py permanent value (we want this scoped to the flag):
```python
if args.latency_arb and args.max_trades_per_window is None:
    args.max_trades_per_window = 20
```

### Step 8 — Would-fire telemetry (used in Phase 1)
**File:** `signal_diffusion.py`, inside `decide_latency_arb` just before
the final return statements, and symmetrically on the FLAT rejection path
when `_arb_delta_usd` was set.

Append a jsonl line to `latency_arb_shadow.jsonl` via a helper
`self._shadow_log({...})` that batches writes every 100 events. Fields:
`ts_ms, delta_usd, window_s, ask_up, ask_down, book_age_ms, decision,
reason, tau, binance_mid, would_fire_side`.

This gives us a 24h shadow dataset before touching real money.

---

## 8. Live A/B Test Plan

### Phase 1 — Shadow logging (24h, zero risk)
```
python3 live_trader.py btc btc_5m \
  --latency-arb --arb-dry-run \
  --arb-delta-usd 25 --arb-window-s 2.0 --arb-variant both \
  --dry-run  # no real orders anyway, belt-and-suspenders
```
Parallel: run another tracker with `--arb-variant impulse` and `--arb-delta-usd 15`
to sweep sensitivity.

**Exit criteria before Phase 2:**
- `latency_arb_shadow.jsonl` contains at least **20 would-fires** across
  BTC 5m + 15m in 24h with variant=both.
- Median `book_age_ms` at fire time is >600ms (confirms the staleness
  asset is real at our measurement point).
- Fires cluster around ~$25-$150 |Δ|, not tightly around $25 (confirms
  threshold is in the right zone, not a cliff).

**Abort criteria:**
- <10 fires/24h → threshold too tight; retry at $15.
- >500 fires/24h → threshold too loose; retry at $40.
- Median `book_age_ms` < 200ms at fire time → the book is not meaningfully
  stale; the Chainlink lag isn't translating into CLOB lag. Stop and
  investigate.

### Phase 2 — Live with tiny bankroll (48h, capped risk)
```
python3 live_trader.py btc btc_5m \
  --latency-arb \
  --arb-delta-usd 25 --arb-window-s 2.0 \
  --arb-size-usd 5 \
  --bankroll 100 --max-positions 5 --max-trades-per-window 20
```
Target: 70-160 trades across 48h. Stop conditions:
- Bankroll < $70 (30% drawdown) — halt and review.
- 40+ trades with realized WR < 45% — halt; the signal is wrong.
- After 48h with ≥30 resolved trades: compute WR, avg edge/share, Sharpe.

**Ship-up criteria (Phase 3):**
- Realized WR ≥ 55%, avg edge/share ≥ $0.015, PnL > $0 after fees.
- P-value from bootstrap of per-trade edge > 0 below 10%.

If these are met, scale to `--arb-size-usd 25 --bankroll 500` and enable
on BTC 15m simultaneously.

### Phase 3 — Scale & widen (after Dublin VPS is live)
- Raise `arb_size_usd` to $25-50 as bankroll grows.
- Move to `BTC 5m + BTC 15m + ETH 5m` concurrently (each tracker is
  independent; they already share feeds).
- Optionally enable the magnitude-tilt sizing from §5.
- Consider loosening `arb_delta_usd_thresh` to $15 once Dublin cuts RTT
  to 2ms — the edge per trade drops, but at 5× the volume the aggregate
  PnL may win.

---

## 9. Failure Modes & Monitoring

### Failure modes (ranked by expected probability)

1. **Order gets filled AFTER the book repriced.** Our FOK fires at ask,
   but by the time the CLOB matches we get either (a) the fill at a
   worse level in the book (if it stepped up multiple ticks in-flight)
   or (b) FOK rejection. This is the RTT cost. The 200ms home RTT means
   the book moves in ~40% of cases (p50 Binance tick-to-tick ~50-100ms).
   **Dublin mandatory before scaling beyond Phase 2.**

2. **Adverse shadow — the Δ is real but Binance IS the market and the
   move reverses.** In practice ~20% of $25 impulses reverse within 2s.
   That's fine, it shows up in the WR. The cooldown (§3b) guards against
   doubling down on the retrace.

3. **Calm-market whipsaw: book is quiet, single tick pushes through
   the threshold, we fire, book unmoved because nobody cared about $26.**
   The `arb_require_book_stale=True` gate (variant C) is the specific
   defense. Monitor: fraction of fires with `book_age_ms < 600` should
   be 0 under variant C.

4. **Binance WS stalls.** `_check_stale_features` via existing
   `max_binance_age_ms=1500` already handles this. Ring buffer will
   simply not receive updates; no false fires because `bn_now_ts` comes
   from the most recent ring entry, which ages out on its own.

5. **Ring buffer polluted by Chainlink fallback.** Avoided by design —
   we only push to `_bn_ring_mid` when `_binance_mid` is not None (i.e.
   Binance freshness gate passed).

6. **Resolution snipe competition.** We fire well before the last 30s
   (`min_fire_tau_s=30`). User's explicit directive: no end-of-window
   plays. Enforced.

7. **Over-fires on correlated 5m+15m.** Same BTC move fires both
   trackers simultaneously, using 2× capital. Acceptable (they resolve
   at different times, uncorrelated resolution risk), but tracked.

### Live monitoring (add to dashboard)

- **`arb_fires_24h`** by market (count)
- **`arb_hit_rate`** — fires where we got a fill (vs. FOK rejects)
- **`arb_book_stale_pct`** — fires where book_age_ms > 600 at fire
- **`arb_delta_p50 / p90`** — magnitude distribution
- **`arb_edge_net_p50`** — realized edge per share, once positions resolve
- **`arb_reverse_rate`** — fraction of trades where Binance mid at
  `fire_ts + 10s` moved AGAINST our side (leading indicator of WR)

Add a "LATENCY ARB" subsection to `display.py` between the existing STALE-QUOTE
and POSITIONS panels, rendering the above six metrics. ~40 lines.

### Logs to grep post-mortem

- `tracker.ctx["_arb_delta_usd"]` on every decide → sparsifies in logs.
- `_execute_taker` event string: look for `[TAKER-FOK]` with reason
  `LAT_ARB` to filter latency-arb fills.
- `latency_arb_shadow.jsonl` — Phase 1 full history, retained permanently.

---

## 10. What to explicitly NOT do

1. **Don't add a σ estimator.** The whole point is avoiding the GBM
   failure mode. If your instinct is "but we need to know the magnitude
   of the fair-value shift" — no, we don't. The Δ threshold IS the shift
   detector. Anything further is reintroducing the bug.

2. **Don't gate on `_p_model_raw`.** Not computed in this path. Stays
   unset; the display code has to handle `None` (it already does for the
   first second of a window).

3. **Don't add a filtration model call.** Out of scope for latency arb.

4. **Don't couple the Phase 2 test with stale-quote mode running.** One
   at a time. Comparing WR between two modes requires them to be exclusive
   (same bankroll, same windows).

5. **Don't refactor `decide_stale_quote` "while we're here."** It's a
   working signal; the calm-market filter keeps it safe. Leave it alone.
   Ship latency arb beside it, let the two compete on Sharpe, keep both
   available via flags.

6. **Don't pre-optimize the ring buffer.** A deque(maxlen=150) is fine.
   Don't reach for numpy or a lock-free Rust ring until Phase 3 after
   Dublin shows Python-side overhead actually matters. Current median
   `signal_eval_ms` is 1ms with way more work happening.

---

## 11. Estimated Code Diff

| File | Lines added | Lines changed |
|------|------------:|--------------:|
| `signal_diffusion.py` | ~150 | 0 |
| `live_trader.py`      |  ~35 | ~5 |
| `tracker.py`          |  ~10 | ~5 |
| `display.py`          |  ~40 | 0 |
| **Total**             | **~235** | **~10** |

Comfortably under the 300-line budget. Zero Rust changes.

---

## 12. First Commit Should Be

Just steps 1-3 (ring buffer + method + signature). No CLI flag, no
routing. A unit test that constructs a DiffusionSignal with
`latency_arb_mode=True`, synthesizes a `_bn_ring_mid` with a $50 impulse,
and asserts `decide_latency_arb` returns a BUY_UP decision. Run it. Ship.

Then the plumbing (steps 4-7). Then the shadow logger (step 8) — the
minute this is live, Phase 1 starts collecting data and we have a real
number to argue about in 24 hours.
