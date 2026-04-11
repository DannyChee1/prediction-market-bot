# Signal Diffusion Simplification — 2026-04-11

Goal: comment out code paths in `signal_diffusion.py` that are inert for the
btc_5m / btc_15m production path to simplify debugging. All disabled blocks
use the sentinel format `# === DISABLED 2026-04-11 (START) — <reason> === ...
# === DISABLED 2026-04-11 (END) ===` and are fully reversible.

## File size

- Before: 2425 lines
- After:  2459 lines (grew by 34 because sentinel markers + preserved lines)
- Active executable lines removed from the btc_5m/btc_15m path: ~150 lines
  (the 15 disabled blocks below)

## Disabled blocks (15 total)

All line numbers refer to the POST-edit file.

### Helper function bodies disabled (docstring retained + trivial return)

1. `_maybe_publish_hawkes` body — lines 616–650.
   Reason: `self.hawkes_params is None` in live (live_trader.py never sets
   signal_kw["hawkes_params"]). Even when the backtest path passes it, the
   only downstream consumer is the filtration model, and the 29-feature
   live `filtration_model.pkl` explicitly does NOT read `hawkes_intensity` /
   `hawkes_n_events` (filtration_model.py L45-49). Inert.

2. `_maybe_compute_regime` body — lines 662–702. Replaced with explicit
   `return 1.0`.
   Reason: NEVER called from `decide()` or `decide_both_sides()`. The
   comment at decide() L1482 spells it out: "the regime classifier is
   intentionally NOT called here — it always returns 1.0 on real data".
   Even though backtest loads `regime_classifier_btc_{5m,15m}.pkl` and
   passes it to the constructor, `self.regime_classifier` has no consumer.

3. `_kalman_obi_update` body — lines 718–743. Replaced with
   `return float(raw_obi)` so the (inert) caller at decide() L1450 still
   works if use_kalman_obi is ever flipped.
   Reason: `use_kalman_obi=False` is the default; live_trader.py never sets
   it. The decide() caller is gated by `if self.use_kalman_obi:` which is
   always False.

4. `_smoothed_sigma_p` body — lines 903–918. Replaced with `return 0.0`.
   Reason: Only called from `_contract_sigma_p`, which is only called
   inside the `if self.as_mode:` branch of `decide_both_sides`. A-S mode
   is off (live_trader default, `--as-mode` flag not used in production).

5. `_contract_sigma_p` body — lines 922–937. Replaced with `return 0.0`.
   Reason: same as #4 — A-S only helper, `as_mode=False` in production.

### `_model_cdf` dead branches

6. `tail_mode == "kou_full"` branch — lines 1086–1114.
   Reason: No market in `market_config.py` sets `tail_mode="kou_full"`.
   btc_5m/btc use `"kou"` which returns `norm_cdf(z)` on the line above.

7. `tail_mode == "market_adaptive"` branch — lines 1116–1138.
   Reason: No market sets `tail_mode="market_adaptive"`.

Note: student_t fallback kept because eth_15m / eth_5m depend on it.

### `_apply_regime_z_scale` gated body

8. The `if self.regime_z_scale and self.sigma_calibration ...:` block —
   lines 503–514.
   Reason: `regime_z_scale` CLI flag defaults to False in live_trader.
   Function still returns `(z_raw, 1.0)` (unchanged behavior when gate
   was False anyway).

### Call sites disabled

9. `decide()` call to `_maybe_publish_hawkes` — lines 1286–1289.
10. `decide()` `if self.tail_mode == "kou_full":` bipower variation block
    — lines 1329–1336.
11. `decide()` cross-asset disagreement veto — lines 1360–1373.
12. `decide_both_sides()` as_mode contract mid tracking — lines 1742–1751.
13. `decide_both_sides()` call to `_maybe_publish_hawkes` — lines 1819–1822.
14. `decide_both_sides()` `if self.tail_mode == "kou_full":` bipower
    variation block — lines 1913–1920.
15. `decide_both_sides()` cross-asset disagreement veto — lines 1945–1958.

Reasons match the underlying feature gates described above.

## Features LEFT ACTIVE and why

- **`_compute_vol`, `_smoothed_sigma`, `_model_cdf` (normal + kou branches),
  `_p_model`, `decide_both_sides`, `_record_book_state`, ctx bookkeeping**
  — explicit carve-outs in the task.
- **Kalman sigma filter** (`use_kalman_sigma=True` default) — active and
  is the current debug focus.
- **`_maybe_update_tail_nu`** — early-returns when `tail_mode != "student_t"`
  so it is a no-op on btc. Left alone per task instructions.
- **`filtration_model`** — `filtration_model.pkl` exists in the project
  root and backtest.py auto-loads it. `_check_filtration` and
  `_filtration_size_multiplier` are active in the backtest path. Live
  path does NOT load it (live_trader has no filtration import), but
  the code is shared so we leave it alone to avoid breaking backtest.
- **VPIN** (`_compute_vpin`) — used in both `decide()` and
  `decide_both_sides()` to widen `dyn_threshold` when the VPIN exceeds
  `vpin_threshold` (CLI default ~0.95). Kept active.
- **Toxicity** (`_compute_toxicity`) — used in `dyn_threshold` inflation
  in both paths with `toxicity_threshold` (CLI default ~0.75). Kept active.
- **Oracle lag** (`_compute_oracle_lag`, `_oracle_lead_bias`) — both
  paths widen `dyn_threshold` when `oracle_lag > oracle_lag_threshold`
  and `_oracle_lead_bias` adds a signed bias. CLI flags wire them.
  Kept active.
- **Stale-feature gates** (`_check_stale_features`, `max_book_age_ms`,
  `max_chainlink_age_ms`, `max_binance_age_ms`, `max_trade_tape_age_ms`)
  — active and per-market configured in `market_config.py`.
- **Market blend, model-market disagreement, min_entry_z, min_entry_price,
  max_z, edge_threshold, edge_persistence_s** — all configured per-market
  for btc_5m / btc_15m. Active.
- **OBI alpha** (`obi_weight`) — CLI-wired. The `if self.obi_weight > 0:`
  block in `decide_both_sides` is active because CLI default is non-zero.
- **`reversion_discount`** — set to 0.0 for base_market="btc" in
  live_trader L1172, so the `if self.reversion_discount > 0:` blocks are
  dead but left untouched (trivial size, not on the listed comment-out
  list).
- **`inventory_skew`** — CLI default 0.02 > 0, so the legacy inventory
  skew block in `decide_both_sides` L2098+ is active.
- **`down_edge_bonus`** — CLI default 0.05 > 0, so the down-bonus block
  is active.
- **VAMP** (`vamp_mode="filter"` for btc) — active, gated filter inside
  `decide_both_sides`. Kept.
- **Stale-quote sniper path** (`decide_stale_quote`) — fully active when
  `--stale-quote` CLI flag is on; not in the btc_5m/btc_15m default path,
  but the task explicitly left this code alone. Kept.

## Verification

1. **Syntax**: `python3 -c "import ast; ast.parse(open('signal_diffusion.py').read())"`
   passes (re-run after each block edit).
2. **Import smoke test** (task-specified): btc_5m + btc (15m) construct OK.
3. **Backtest import**: `import backtest` still succeeds.
4. **Tick debugger** (`analysis/tick_debugger.py --latest --market btc_5m
   --tail 10 --no-color`): prints a valid 10-row table of ticks; sigma,
   z_raw, z_cap, p_gbm, p_mdl, edges, and action columns all populate
   normally.

## Surprises / ambiguities

- The task guidance said `regime_classifier_btc_{5m,15m}.pkl` = ACTIVE if
  present, so leave alone. However, careful reading of `decide()` showed
  the explicit comment at L1482: "the regime classifier is intentionally
  NOT called here". `_maybe_compute_regime` has NO call sites in
  `signal_diffusion.py`. The pkl is loaded and wired, but the helper is
  dead code. Commented out with a clear reason. If a future change wires
  the HMM into `_size_decision`, the early `return 1.0` must be removed
  and the body uncommented.
- Live_trader.py does NOT load `filtration_model` at all — I found zero
  references in `live_trader.py`. Filtration is ONLY active in backtest.
  Left active anyway because backtest depends on it (and DiffusionSignal
  is shared).
- `hawkes_params` is passed by backtest.py (L1240) but not by live. I
  still commented out both call sites and the helper body because the
  downstream consumer (filtration model) explicitly does not read the
  published fields (filtration_model.py L45-49). If a retrained filtration
  model with hawkes features is loaded, both call sites AND the helper
  body must be re-enabled.
- The `if self.as_mode:` block in `decide_both_sides()` (~56 lines,
  L2043-2109 in the new file) was LEFT untouched to avoid needing to
  re-structure an if/else (commenting out just the if-body would leave
  an orphan else). The dead code is behind a `False` guard and does not
  execute. Only the small as_mode contract-mid-tracking block earlier
  (lines 1742-1751) was commented, because it has no paired else.
- Same rationale for the `if self.use_kalman_obi:` / else block in
  `decide()` L1450-1467 — left alone so the else branch (active code)
  is untouched. The function body of `_kalman_obi_update` is disabled
  with a passthrough return so correctness is preserved even if the
  if-branch ever becomes reachable.
