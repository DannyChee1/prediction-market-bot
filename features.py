"""
Feature extraction for the v2 filtration model.

Implements the unified feature catalog from
tasks/findings/feature_catalog_unified_2026-04-11.md (~120 features).

Phase 1 (this file): all Tier-A features that work today from the
existing parquet schema. ~101 features.

Design principles:
  - Pure functions: no side effects, no global state
  - Train/inference parity: features computed identically in backtest
    and live (no look-ahead bugs)
  - NaN-safe: every feature handles missing inputs gracefully and
    returns 0.0 (or a documented sentinel) when computation is impossible
  - Robust to thin history: don't error on the first 5 ticks of a window
  - Cheap: each feature is O(1) or O(history_window) — no hidden N²
  - Documented IDs (U1-U120) match the catalog so XGBoost importance
    output can be cross-referenced

Caller pattern:
    from features import compute_features
    feats = compute_features(snapshot, ctx, history_buf=ctx.get("price_history", []),
                              ts_history=ctx.get("ts_history", []),
                              window_duration_s=300.0,
                              p_gbm=ctx.get("_p_model_raw"))
    # feats is a dict[str, float] with ~100 entries

Train/inference parity caveat:
    - Backtest replay calls this with parquet ts_ms (1Hz)
    - Live calls it with sub-second binance_mid history
    - The features that depend on cadence (HAR-RV, vol-of-vol) will
      give slightly different values in live vs backtest because the
      input history is at different sampling rates. This is a KNOWN
      limitation; it's the same parity issue documented in
      parity_fixes_applied_2026-04-11.md.

Sources for individual features: see feature_catalog_unified_2026-04-11.md
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional, Sequence

import numpy as np


# ── Numerical helpers ─────────────────────────────────────────────────────────

def _safe_log(x: float, floor: float = 1e-12) -> float:
    """log(x) with a small floor to avoid -inf on zero/negative inputs."""
    return math.log(max(x, floor))


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    """num/den with a default when den is too small."""
    if abs(den) < 1e-12:
        return default
    return num / den


def _zscore(x: float, mu: float, sd: float) -> float:
    """Standardize x; return 0 if sd is degenerate."""
    return _safe_div(x - mu, sd, 0.0)


def _stdev(values: Sequence[float]) -> float:
    """Sample std with ddof=1; returns 0 if fewer than 2 samples."""
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def _slice_by_time(
    history: Sequence[float],
    ts_history: Sequence[int],
    lookback_s: float,
    end_ts_ms: Optional[int] = None,
) -> tuple[list[float], list[int]]:
    """Return (prices, ts) within the last `lookback_s` seconds.

    end_ts_ms defaults to the last entry in ts_history. The slice is
    inclusive of any sample whose ts is >= (end_ts_ms - lookback_ms).
    """
    if not history or not ts_history:
        return [], []
    if end_ts_ms is None:
        end_ts_ms = ts_history[-1]
    cutoff = end_ts_ms - int(lookback_s * 1000)
    out_p, out_t = [], []
    # Linear walk from end (history is short, this is fine)
    for i in range(len(ts_history) - 1, -1, -1):
        if ts_history[i] < cutoff:
            break
        out_p.append(history[i])
        out_t.append(ts_history[i])
    out_p.reverse()
    out_t.reverse()
    return out_p, out_t


def _log_returns(prices: Sequence[float]) -> list[float]:
    """Log returns over consecutive non-duplicate price ticks."""
    rets = []
    for i in range(1, len(prices)):
        if prices[i] > 0 and prices[i - 1] > 0 and prices[i] != prices[i - 1]:
            rets.append(math.log(prices[i] / prices[i - 1]))
    return rets


def _realized_variance(prices: Sequence[float]) -> float:
    """Sum of squared log returns. Per-window total, not per-second."""
    return float(sum(r * r for r in _log_returns(prices)))


# ─────────────────────────────────────────────────────────────────────────────
# Category 1: Multi-timescale momentum & realized vol (HAR-RV family)
# Features U1-U22
# ─────────────────────────────────────────────────────────────────────────────

def _features_momentum_and_vol(
    history: Sequence[float],
    ts_history: Sequence[int],
    sigma_per_s: float,
) -> dict[str, float]:
    """Categories 1 (multi-timescale momentum & RV)."""
    out: dict[str, float] = {}

    # Multi-lookback returns
    for label, lb in [("1s", 1), ("5s", 5), ("10s", 10), ("30s", 30),
                      ("60s", 60), ("120s", 120), ("300s", 300)]:
        slice_p, _ = _slice_by_time(history, ts_history, lb)
        if len(slice_p) >= 2 and slice_p[0] > 0 and slice_p[-1] > 0:
            out[f"bn_ret_{label}"] = math.log(slice_p[-1] / slice_p[0])
        else:
            out[f"bn_ret_{label}"] = 0.0

    # Acceleration: 5s return − 10s return (positive = accelerating up)
    out["bn_accel"] = out["bn_ret_5s"] - out["bn_ret_10s"]

    # Run length: consecutive ticks in same direction (looking back through hist)
    rets_recent = _log_returns(history[-30:]) if len(history) >= 2 else []
    if rets_recent:
        last_sign = 1 if rets_recent[-1] > 0 else (-1 if rets_recent[-1] < 0 else 0)
        run = 0
        for r in reversed(rets_recent):
            r_sign = 1 if r > 0 else (-1 if r < 0 else 0)
            if r_sign == last_sign and r_sign != 0:
                run += 1
            else:
                break
        out["bn_run_length"] = float(run * (1 if last_sign > 0 else -1))
    else:
        out["bn_run_length"] = 0.0

    # Realized variance at multiple horizons (HAR-RV triple)
    rv30 = _realized_variance(_slice_by_time(history, ts_history, 30)[0])
    rv120 = _realized_variance(_slice_by_time(history, ts_history, 120)[0])
    rv300 = _realized_variance(_slice_by_time(history, ts_history, 300)[0])
    rv1800 = _realized_variance(_slice_by_time(history, ts_history, 1800)[0])
    out["RV_30s"] = rv30
    out["RV_120s"] = rv120
    out["RV_300s"] = rv300
    out["log_RV_ratio_short"] = _safe_log(rv30) - _safe_log(rv300) if rv300 > 0 else 0.0
    out["log_RV_ratio_long"] = _safe_log(rv300) - _safe_log(rv1800) if rv1800 > 0 else 0.0

    # Bipower variation (Barndorff-Nielsen 2004) — robust to jumps
    rets_300 = _log_returns(_slice_by_time(history, ts_history, 300)[0])
    if len(rets_300) >= 2:
        bv = (math.pi / 2.0) * sum(
            abs(rets_300[i]) * abs(rets_300[i - 1])
            for i in range(1, len(rets_300))
        )
    else:
        bv = 0.0
    out["bipower_variation_300s"] = bv

    # Realized jump variation = RV - BV (positive = jump-driven)
    rjv = max(rv300 - bv, 0.0)
    out["realized_jump_var_300s"] = rjv
    # Jump indicator: discrete flag when jumps dominate the variance
    out["jump_indicator"] = 1.0 if (rv300 > 0 and rjv / rv300 > 0.3) else 0.0

    # Regime-normalized momentum (z-score of momentum vs sigma)
    # Use the smoothed sigma_per_s as a denominator proxy
    sigma_floor = max(sigma_per_s, 1e-7)
    out["mom_zscore_30s"] = _safe_div(out["bn_ret_30s"], sigma_floor * math.sqrt(30))
    out["mom_zscore_60s"] = _safe_div(out["bn_ret_60s"], sigma_floor * math.sqrt(60))
    out["mom_zscore_300s"] = _safe_div(out["bn_ret_300s"], sigma_floor * math.sqrt(300))

    # Momentum-reversal signal: simple decision-tree-friendly form
    s10 = 1 if out["bn_ret_10s"] > 0 else (-1 if out["bn_ret_10s"] < 0 else 0)
    s60 = 1 if out["bn_ret_60s"] > 0 else (-1 if out["bn_ret_60s"] < 0 else 0)
    s300 = 1 if out["bn_ret_300s"] > 0 else (-1 if out["bn_ret_300s"] < 0 else 0)
    out["mom_reversal_signal"] = float(s10 * (1 - s60 * s300))

    # Fractional differentiation (Lopez de Prado 2018, simplified)
    # We use a fixed-window approximation with d=0.4
    out["frac_diff_d04"] = _frac_diff_simplified(history[-60:], d=0.4)

    return out


def _frac_diff_simplified(prices: Sequence[float], d: float = 0.4) -> float:
    """Single-point fractional difference at the last sample.

    Lopez de Prado's full implementation uses fixed-width window weights;
    this is a simplified version that computes d-th order finite differences
    over the last few prices. Good enough for a single feature value.
    """
    n = len(prices)
    if n < 4 or any(p <= 0 for p in prices[-4:]):
        return 0.0
    log_p = [math.log(p) for p in prices[-4:]]
    # Approximation: weighted sum of recent log differences
    # Weights for d=0.4: w0=1, w1=-d, w2=d(d-1)/2!, w3=-d(d-1)(d-2)/3!
    w = [1.0, -d, d * (d - 1) / 2, -d * (d - 1) * (d - 2) / 6]
    return sum(w[i] * log_p[-1 - i] for i in range(4))


# ─────────────────────────────────────────────────────────────────────────────
# Category 2: Microstructure (OBI replacement: microprice, queue imbalance,
# book pressure, book shape)
# Features U23-U42
# ─────────────────────────────────────────────────────────────────────────────

def _features_microstructure(snapshot, ctx) -> dict[str, float]:
    """Order-book microstructure features. Replaces the dead OBI features."""
    out: dict[str, float] = {}

    # ── Microprice (Stoikov 2018) ─────────────────────────────────────
    # microprice = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)
    # The "wrong" weighting (ask weighted by bid_size) is correct: it
    # reflects the side that's about to crack. When bid_size >> ask_size,
    # the microprice tilts toward the ask (because the ask is the side
    # most likely to lift).
    def _micro(bb, ba, sb, sa):
        if bb is None or ba is None or sb is None or sa is None:
            return None
        denom = sb + sa
        if denom <= 0:
            return None
        return (bb * sa + ba * sb) / denom

    micro_up = _micro(snapshot.best_bid_up, snapshot.best_ask_up,
                      snapshot.size_bid_up, snapshot.size_ask_up)
    micro_down = _micro(snapshot.best_bid_down, snapshot.best_ask_down,
                        snapshot.size_bid_down, snapshot.size_ask_down)
    mid_up = (snapshot.best_bid_up + snapshot.best_ask_up) / 2.0 \
        if snapshot.best_bid_up is not None and snapshot.best_ask_up is not None else None
    mid_down = (snapshot.best_bid_down + snapshot.best_ask_down) / 2.0 \
        if snapshot.best_bid_down is not None and snapshot.best_ask_down is not None else None

    out["microprice_up"] = micro_up if micro_up is not None else 0.0
    out["microprice_down"] = micro_down if micro_down is not None else 0.0
    out["microprice_offset_up"] = (micro_up - mid_up) if (micro_up and mid_up) else 0.0
    out["microprice_offset_down"] = (micro_down - mid_down) if (micro_down and mid_down) else 0.0

    # Microprice drift requires history — we read from ctx
    micro_hist = ctx.setdefault("_microprice_up_hist", [])
    micro_ts = ctx.setdefault("_microprice_up_ts", [])
    if micro_up is not None:
        micro_hist.append(micro_up)
        micro_ts.append(snapshot.ts_ms)
        # Keep last 60s
        cutoff = snapshot.ts_ms - 60_000
        while micro_ts and micro_ts[0] < cutoff:
            micro_hist.pop(0)
            micro_ts.pop(0)
    sl5_p, _ = _slice_by_time(micro_hist, micro_ts, 5, end_ts_ms=snapshot.ts_ms)
    sl30_p, _ = _slice_by_time(micro_hist, micro_ts, 30, end_ts_ms=snapshot.ts_ms)
    out["microprice_drift_5s"] = (micro_up - sl5_p[0]) if (micro_up is not None and sl5_p) else 0.0
    out["microprice_drift_30s"] = (micro_up - sl30_p[0]) if (micro_up is not None and sl30_p) else 0.0

    # ── Top queue imbalance (Gould-Bonart 2016) ───────────────────────
    def _top_qi(bid_sz, ask_sz):
        if bid_sz is None or ask_sz is None:
            return 0.0
        denom = bid_sz + ask_sz
        if denom <= 0:
            return 0.0
        return (bid_sz - ask_sz) / denom

    out["top_queue_imbalance_up"] = _top_qi(snapshot.size_bid_up, snapshot.size_ask_up)
    out["top_queue_imbalance_down"] = _top_qi(snapshot.size_bid_down, snapshot.size_ask_down)

    # ── Book pressure: log of total depth ratio ──────────────────────
    def _book_pressure(bids, asks):
        bid_total = sum(s for _, s in bids[:5])
        ask_total = sum(s for _, s in asks[:5])
        if bid_total <= 0 or ask_total <= 0:
            return 0.0
        return math.log(bid_total / ask_total)

    out["book_pressure_up"] = _book_pressure(snapshot.bid_levels_up, snapshot.ask_levels_up)
    out["book_pressure_down"] = _book_pressure(snapshot.bid_levels_down, snapshot.ask_levels_down)

    # ── Cross-book features ──────────────────────────────────────────
    # Compute imbalance5 from book levels (the parquet may not always have it)
    def _imbalance5(bids, asks):
        bid_total = sum(s for _, s in bids[:5])
        ask_total = sum(s for _, s in asks[:5])
        denom = bid_total + ask_total
        if denom <= 0:
            return 0.0
        return (bid_total - ask_total) / denom

    imb_up = _imbalance5(snapshot.bid_levels_up, snapshot.ask_levels_up)
    imb_down = _imbalance5(snapshot.bid_levels_down, snapshot.ask_levels_down)
    out["cross_book_imbalance_sum"] = imb_up + imb_down  # consensus
    out["cross_book_imbalance_diff"] = imb_up - imb_down  # current model has this

    # ── Book convexity (quadratic fit on top 5 levels) ───────────────
    def _convexity(levels):
        if len(levels) < 5:
            return 0.0
        sizes = [s for _, s in levels[:5]]
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        y = np.array(sizes, dtype=np.float64)
        try:
            # Quadratic fit y = a*x² + b*x + c; return a
            coeffs = np.polyfit(x, y, 2)
            return float(coeffs[0])
        except Exception:
            return 0.0

    out["book_convexity_up"] = _convexity(snapshot.bid_levels_up)
    out["book_convexity_down"] = _convexity(snapshot.bid_levels_down)

    # ── Spread features ──────────────────────────────────────────────
    spread_up = (snapshot.best_ask_up - snapshot.best_bid_up) \
        if (snapshot.best_ask_up is not None and snapshot.best_bid_up is not None) else 0.0
    spread_down = (snapshot.best_ask_down - snapshot.best_bid_down) \
        if (snapshot.best_ask_down is not None and snapshot.best_bid_down is not None) else 0.0

    # Rolling spread history for the compression ratio
    spread_hist = ctx.setdefault("_spread_up_hist", [])
    spread_ts = ctx.setdefault("_spread_up_ts", [])
    spread_hist.append(spread_up)
    spread_ts.append(snapshot.ts_ms)
    cutoff = snapshot.ts_ms - 600_000  # keep 10 minutes
    while spread_ts and spread_ts[0] < cutoff:
        spread_hist.pop(0)
        spread_ts.pop(0)

    sl60_p, _ = _slice_by_time(spread_hist, spread_ts, 60, end_ts_ms=snapshot.ts_ms)
    if sl60_p:
        med60 = float(np.median(sl60_p))
        out["spread_compression_ratio"] = _safe_div(spread_up, med60, 1.0)
    else:
        out["spread_compression_ratio"] = 1.0

    sl_session_p, _ = _slice_by_time(spread_hist, spread_ts, 600, end_ts_ms=snapshot.ts_ms)
    if sl_session_p:
        out["spread_vs_typical"] = _safe_div(spread_up, float(np.median(sl_session_p)), 1.0)
    else:
        out["spread_vs_typical"] = 1.0

    # Quote stability: fraction of last 30s where best bid/ask didn't change
    bb_hist = ctx.setdefault("_bb_up_hist", [])
    bb_ts = ctx.setdefault("_bb_up_ts", [])
    if snapshot.best_bid_up is not None:
        bb_hist.append(snapshot.best_bid_up)
        bb_ts.append(snapshot.ts_ms)
        cutoff = snapshot.ts_ms - 60_000
        while bb_ts and bb_ts[0] < cutoff:
            bb_hist.pop(0)
            bb_ts.pop(0)
    sl30_bb, _ = _slice_by_time(bb_hist, bb_ts, 30, end_ts_ms=snapshot.ts_ms)
    if len(sl30_bb) >= 2:
        unchanged = sum(1 for i in range(1, len(sl30_bb)) if sl30_bb[i] == sl30_bb[i - 1])
        out["quote_stability_30s"] = unchanged / (len(sl30_bb) - 1)
    else:
        out["quote_stability_30s"] = 0.0

    # ── Largest order ratio ──────────────────────────────────────────
    all_sizes_up = [s for _, s in snapshot.bid_levels_up[:5]] + [s for _, s in snapshot.ask_levels_up[:5]]
    if all_sizes_up:
        med = float(np.median(all_sizes_up)) if any(s > 0 for s in all_sizes_up) else 1.0
        out["largest_order_ratio"] = _safe_div(max(all_sizes_up), med, 1.0)
        out["book_depth_total"] = float(sum(all_sizes_up))
    else:
        out["largest_order_ratio"] = 0.0
        out["book_depth_total"] = 0.0

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Category 3: Polymarket-specific (book age, residual, tick stability)
# Features U43-U53
# ─────────────────────────────────────────────────────────────────────────────

def _features_polymarket(snapshot, ctx, p_gbm: Optional[float]) -> dict[str, float]:
    """Polymarket-specific microstructure: stale book, P residual, tick stability."""
    out: dict[str, float] = {}

    # Book staleness in ms (already tracked by the bot, just expose to filter)
    book_age = ctx.get("_book_age_ms")
    out["pm_book_age_ms"] = float(book_age) if book_age is not None else 0.0

    # Time since last contract trade (need to track this)
    # The recorder doesn't capture trade ts directly — we approximate from
    # ctx if the bot tracked it during this session
    last_trade_ts = ctx.get("_pm_last_trade_ts_ms", 0)
    if last_trade_ts > 0:
        out["pm_time_since_last_trade_ms"] = float(snapshot.ts_ms - last_trade_ts)
    else:
        out["pm_time_since_last_trade_ms"] = 60_000.0  # sentinel: 60s if unknown

    # Tick stability: count consecutive ticks where best_bid_up unchanged
    last_bb = ctx.get("_pm_last_bb_up", None)
    stab_count = ctx.get("_pm_tick_stability_count", 0)
    if snapshot.best_bid_up is not None:
        if last_bb is not None and snapshot.best_bid_up == last_bb:
            stab_count += 1
        else:
            stab_count = 0
        ctx["_pm_last_bb_up"] = snapshot.best_bid_up
        ctx["_pm_tick_stability_count"] = stab_count
    out["pm_tick_stability_count"] = float(stab_count)

    # P residual: book mid - GBM model probability
    if (snapshot.best_bid_up is not None and snapshot.best_ask_up is not None
            and p_gbm is not None):
        mid_up = (snapshot.best_bid_up + snapshot.best_ask_up) / 2.0
        residual = mid_up - p_gbm
    else:
        residual = 0.0
        mid_up = None
    out["pm_p_residual"] = residual

    # Residual persistence: stdev over last 10s
    res_hist = ctx.setdefault("_pm_residual_hist", [])
    res_ts = ctx.setdefault("_pm_residual_ts", [])
    res_hist.append(residual)
    res_ts.append(snapshot.ts_ms)
    cutoff = snapshot.ts_ms - 30_000
    while res_ts and res_ts[0] < cutoff:
        res_hist.pop(0)
        res_ts.pop(0)
    sl10_res, _ = _slice_by_time(res_hist, res_ts, 10, end_ts_ms=snapshot.ts_ms)
    out["pm_p_residual_persistence"] = _stdev(sl10_res)
    sl5_res, _ = _slice_by_time(res_hist, res_ts, 5, end_ts_ms=snapshot.ts_ms)
    if sl5_res:
        out["pm_p_residual_drift"] = residual - sl5_res[0]
    else:
        out["pm_p_residual_drift"] = 0.0

    # Number of meaningful levels (size > 50)
    levels_with_size = sum(1 for _, s in snapshot.bid_levels_up[:5] if s > 50)
    out["pm_n_levels_meaningful"] = float(levels_with_size)

    # Largest single ask size
    if snapshot.ask_levels_up:
        out["pm_largest_ask_size"] = float(max(s for _, s in snapshot.ask_levels_up[:5]))
    else:
        out["pm_largest_ask_size"] = 0.0

    # Number of distinct depth levels
    out["pm_unique_makers_proxy"] = float(len(snapshot.bid_levels_up[:5]))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Category 4: Window-relative path features
# Features U54-U63
# ─────────────────────────────────────────────────────────────────────────────

def _features_window_path(
    snapshot,
    ctx,
    history: Sequence[float],
    ts_history: Sequence[int],
    window_duration_s: float,
    sigma_per_s: float,
) -> dict[str, float]:
    """Path-aware features the GBM signal can't see (it's memoryless)."""
    out: dict[str, float] = {}

    tau = snapshot.time_remaining_s
    elapsed = max(0.0, window_duration_s - tau)
    out["tau_frac"] = _safe_div(tau, window_duration_s, 0.0)

    # Delta and z-score normalized to expected move size
    if snapshot.window_start_price > 0:
        eff_price = ctx.get("_binance_mid") or snapshot.chainlink_price
        if eff_price and eff_price > 0:
            delta_log = math.log(eff_price / snapshot.window_start_price)
        else:
            delta_log = 0.0
    else:
        delta_log = 0.0

    sigma_floor = max(sigma_per_s, 1e-7)
    expected = sigma_floor * math.sqrt(max(elapsed, 1.0))
    out["delta_per_typical_move"] = _safe_div(abs(delta_log), expected, 0.0)

    # Window-history of log deltas (relative to start) maintained in ctx
    win_deltas = ctx.setdefault("_window_log_deltas", [])
    win_deltas.append(delta_log)
    if len(win_deltas) > 1500:  # cap memory
        del win_deltas[:-1500]

    if win_deltas:
        out["peak_delta_seen"] = float(max(abs(d) for d in win_deltas))

        # z-score of current vs typical
        z_now = _safe_div(delta_log, expected, 0.0)
        # The peak z seen so far in the window
        max_pos_z = 0.0
        max_neg_z = 0.0
        for d in win_deltas:
            zi = _safe_div(d, expected, 0.0)
            if zi > max_pos_z:
                max_pos_z = zi
            if zi < max_neg_z:
                max_neg_z = zi
        # Drawdown from window peak z-score
        if z_now > 0:
            out["dd_from_peak_z"] = max(0.0, max_pos_z - z_now)
        else:
            out["dd_from_peak_z"] = max(0.0, z_now - max_neg_z)
        out["signed_z_range"] = max_pos_z - max_neg_z
    else:
        out["peak_delta_seen"] = 0.0
        out["dd_from_peak_z"] = 0.0
        out["signed_z_range"] = 0.0

    # Number of zero crossings of (price - window_start_price)
    if len(win_deltas) >= 2:
        crossings = sum(1 for i in range(1, len(win_deltas))
                        if (win_deltas[i] > 0) != (win_deltas[i - 1] > 0))
        out["n_zero_crossings"] = float(crossings)
        out["n_zero_crossings_normalized"] = _safe_div(
            crossings, math.sqrt(max(elapsed, 1.0)), 0.0
        )
    else:
        out["n_zero_crossings"] = 0.0
        out["n_zero_crossings_normalized"] = 0.0

    # Run length: consecutive seconds in same direction (already in momentum)
    # Reuse from win_deltas
    if len(win_deltas) >= 2:
        last_dir = 1 if win_deltas[-1] > win_deltas[-2] else (-1 if win_deltas[-1] < win_deltas[-2] else 0)
        run = 0
        for i in range(len(win_deltas) - 1, 0, -1):
            d = 1 if win_deltas[i] > win_deltas[i - 1] else (-1 if win_deltas[i] < win_deltas[i - 1] else 0)
            if d == last_dir and d != 0:
                run += 1
            else:
                break
        out["run_length_current"] = float(run * (1 if last_dir > 0 else -1))
    else:
        out["run_length_current"] = 0.0

    # Elapsed fraction × signed |z|
    z = _safe_div(delta_log, expected, 0.0)
    out["elapsed_frac_x_signed_z"] = (1 - out["tau_frac"]) * (1 if z > 0 else -1) * abs(z)

    # Trades fired count (already in ctx from tracker)
    out["n_trades_fired_already"] = float(ctx.get("window_trade_count", 0))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Category 5: Realized higher moments
# Features U64-U71
# ─────────────────────────────────────────────────────────────────────────────

def _features_higher_moments(
    history: Sequence[float],
    ts_history: Sequence[int],
) -> dict[str, float]:
    out: dict[str, float] = {}

    # Realized skewness (Amaya 2015)
    def _realized_skew(rets: list[float]) -> float:
        if len(rets) < 4:
            return 0.0
        rv = sum(r * r for r in rets)
        if rv <= 0:
            return 0.0
        n = len(rets)
        return _safe_div(sum(r * r * r for r in rets), n * (rv ** 1.5), 0.0)

    def _realized_kurt(rets: list[float]) -> float:
        if len(rets) < 4:
            return 0.0
        rv = sum(r * r for r in rets)
        if rv <= 0:
            return 0.0
        n = len(rets)
        return _safe_div(sum(r ** 4 for r in rets), n * (rv * rv), 0.0)

    rets_60 = _log_returns(_slice_by_time(history, ts_history, 60)[0])
    rets_300 = _log_returns(_slice_by_time(history, ts_history, 300)[0])
    out["realized_skew_60s"] = _realized_skew(rets_60)
    out["realized_skew_300s"] = _realized_skew(rets_300)
    out["realized_kurt_60s"] = _realized_kurt(rets_60)
    out["realized_kurt_300s"] = _realized_kurt(rets_300)

    # Vol-of-vol: stdev of recent RV_30 measurements
    # We need 10 recent RV_30 values; they're not stored. Approximate
    # by computing RV in 30s sliding windows over the last 300s.
    rv_buckets = []
    if ts_history:
        end = ts_history[-1]
        for k in range(10):
            slice_start = end - (k + 1) * 30_000
            slice_end = end - k * 30_000
            sl = [history[i] for i in range(len(ts_history))
                  if slice_start <= ts_history[i] < slice_end]
            if len(sl) >= 2:
                rv_buckets.append(_realized_variance(sl))
    out["vol_of_vol_30s"] = _stdev(rv_buckets)

    # Parkinson and Garman-Klass on 1m OHLC
    # Requires bucketing into 1-minute bars; expensive. Compute once per call.
    bars = _build_minute_ohlc(history, ts_history, max_bars=60)
    if len(bars) >= 2:
        out["parkinson_sigma_1m_60"] = _parkinson_vol(bars)
        out["garman_klass_sigma_1m_60"] = _garman_klass_vol(bars)
    else:
        out["parkinson_sigma_1m_60"] = 0.0
        out["garman_klass_sigma_1m_60"] = 0.0

    return out


def _build_minute_ohlc(prices, ts_ms, max_bars: int = 60):
    """Group prices into 1-minute OHLC bars. Returns list of (o,h,l,c)."""
    if not prices or not ts_ms:
        return []
    bars = []
    cur_minute = None
    cur = []
    for p, t in zip(prices, ts_ms):
        m = t // 60_000
        if cur_minute is None:
            cur_minute = m
        if m != cur_minute:
            if cur:
                bars.append((cur[0], max(cur), min(cur), cur[-1]))
            cur = []
            cur_minute = m
        cur.append(p)
    if cur:
        bars.append((cur[0], max(cur), min(cur), cur[-1]))
    return bars[-max_bars:]


def _parkinson_vol(bars) -> float:
    """Parkinson estimator: σ² = (1 / 4 ln 2) × mean(ln(H/L))²"""
    if not bars:
        return 0.0
    s = 0.0
    n = 0
    for _, h, l, _ in bars:
        if h > 0 and l > 0 and h > l:
            s += math.log(h / l) ** 2
            n += 1
    if n == 0:
        return 0.0
    var = s / (n * 4 * math.log(2))
    return math.sqrt(var) if var > 0 else 0.0


def _garman_klass_vol(bars) -> float:
    """Garman-Klass: 0.5 × (ln H/L)² − (2 ln 2 − 1) × (ln C/O)²"""
    if not bars:
        return 0.0
    s = 0.0
    n = 0
    k = 2 * math.log(2) - 1
    for o, h, l, c in bars:
        if h > 0 and l > 0 and o > 0 and c > 0 and h >= l:
            s += 0.5 * (math.log(h / l) ** 2) - k * (math.log(c / o) ** 2)
            n += 1
    if n == 0:
        return 0.0
    var = s / n
    return math.sqrt(max(var, 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Category 7: Time, calendar, macro events
# Features U84-U98
# ─────────────────────────────────────────────────────────────────────────────

def _features_time(snapshot) -> dict[str, float]:
    out: dict[str, float] = {}
    dt = datetime.fromtimestamp(snapshot.ts_ms / 1000, tz=timezone.utc)

    out["hour_sin"] = math.sin(2 * math.pi * dt.hour / 24)
    out["hour_cos"] = math.cos(2 * math.pi * dt.hour / 24)
    out["dow_sin"] = math.sin(2 * math.pi * dt.weekday() / 7)
    out["dow_cos"] = math.cos(2 * math.pi * dt.weekday() / 7)
    out["is_weekend"] = 1.0 if dt.weekday() >= 5 else 0.0
    out["dow_one_hot_sat"] = 1.0 if dt.weekday() == 5 else 0.0
    out["dow_one_hot_sun"] = 1.0 if dt.weekday() == 6 else 0.0
    out["dow_one_hot_mon"] = 1.0 if dt.weekday() == 0 else 0.0

    # Hours since US open (13:30 UTC), wrap into [-12, +12]
    minutes_today = dt.hour * 60 + dt.minute
    us_open_min = 13 * 60 + 30
    delta = minutes_today - us_open_min
    if delta > 720:
        delta -= 1440
    elif delta < -720:
        delta += 1440
    out["hours_since_us_open"] = delta / 60.0

    asia_open_min = 0
    delta_asia = minutes_today - asia_open_min
    if delta_asia > 720:
        delta_asia -= 1440
    out["hours_since_asia_open"] = delta_asia / 60.0

    out["is_us_session"] = 1.0 if 13 * 60 + 30 <= minutes_today <= 21 * 60 else 0.0
    out["is_eu_session"] = 1.0 if 8 * 60 <= minutes_today <= 13 * 60 + 30 else 0.0

    # Minutes to next funding settlement (00, 08, 16 UTC)
    funding_hours = [0, 8, 16, 24]
    h = dt.hour + dt.minute / 60.0
    next_funding = next(fh for fh in funding_hours if fh > h)
    out["minutes_to_funding_settle"] = (next_funding - h) * 60

    # Options expiry: Fridays at 08:00 UTC
    out["is_options_expiry"] = 1.0 if (dt.weekday() == 4 and dt.hour == 8) else 0.0

    out["minute_of_hour"] = float(dt.minute)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Category 9: Information theory & complexity
# Features U111-U115 (skipping the most expensive ones)
# ─────────────────────────────────────────────────────────────────────────────

def _features_information(history: Sequence[float], ts_history: Sequence[int]) -> dict[str, float]:
    out: dict[str, float] = {}

    # Path efficiency: net move / total path length
    sl60_p, _ = _slice_by_time(history, ts_history, 60)
    if len(sl60_p) >= 2 and sl60_p[0] > 0 and sl60_p[-1] > 0:
        net = abs(math.log(sl60_p[-1] / sl60_p[0]))
        total = sum(abs(math.log(sl60_p[i] / sl60_p[i - 1]))
                    for i in range(1, len(sl60_p))
                    if sl60_p[i] > 0 and sl60_p[i - 1] > 0)
        out["path_efficiency_60s"] = _safe_div(net, total, 0.0)
    else:
        out["path_efficiency_60s"] = 0.0

    # Permutation entropy of return signs (order=3 = 6 unique permutations)
    rets = _log_returns(sl60_p)
    if len(rets) >= 6:
        order = 3
        from collections import Counter
        triples = []
        for i in range(len(rets) - order + 1):
            triple = rets[i:i + order]
            # Encode as the sorted-permutation pattern
            sorted_idx = sorted(range(order), key=lambda j: triple[j])
            triples.append(tuple(sorted_idx))
        counts = Counter(triples)
        n = sum(counts.values())
        if n > 0:
            pe = -sum((c / n) * math.log(c / n) for c in counts.values() if c > 0)
            out["permutation_entropy_60s"] = pe / math.log(math.factorial(order))
        else:
            out["permutation_entropy_60s"] = 0.0
    else:
        out["permutation_entropy_60s"] = 0.0

    # Sample entropy: too expensive for every tick — approximate via
    # variance ratio of overlapping windows (cheap proxy)
    if len(rets) >= 8:
        half = len(rets) // 2
        var1 = float(np.var(rets[:half]))
        var2 = float(np.var(rets[half:]))
        out["sample_entropy_proxy"] = _safe_log(var2) - _safe_log(var1)
    else:
        out["sample_entropy_proxy"] = 0.0

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Category 10: Regime classification
# Features U116-U120
# ─────────────────────────────────────────────────────────────────────────────

def _features_regime(
    history: Sequence[float],
    ts_history: Sequence[int],
    sigma_per_s: float,
    ctx: dict,
) -> dict[str, float]:
    out: dict[str, float] = {}

    # Choppiness Index (Bill Dreiss)
    sl30_p, _ = _slice_by_time(history, ts_history, 30)
    if len(sl30_p) >= 4:
        # Sum of true ranges (use 1s diffs as approximation)
        tr_sum = sum(abs(sl30_p[i] - sl30_p[i - 1]) for i in range(1, len(sl30_p)))
        rng = max(sl30_p) - min(sl30_p)
        if rng > 0 and tr_sum > 0:
            out["choppiness_index_30s"] = 100 * math.log10(tr_sum / rng) / math.log10(len(sl30_p))
        else:
            out["choppiness_index_30s"] = 0.0
    else:
        out["choppiness_index_30s"] = 0.0

    # Sigma regime quantile (rank within session sigma history)
    sigma_session = ctx.setdefault("_sigma_session_history", [])
    if sigma_per_s > 0:
        sigma_session.append(sigma_per_s)
        if len(sigma_session) > 5000:
            del sigma_session[:-5000]
    if len(sigma_session) >= 10:
        sorted_s = sorted(sigma_session)
        rank = sum(1 for s in sorted_s if s <= sigma_per_s)
        out["sigma_regime_quantile_session"] = rank / len(sorted_s)
    else:
        out["sigma_regime_quantile_session"] = 0.5

    # Discrete vol regime label (0/1/2)
    if len(sigma_session) >= 10:
        q33 = float(np.quantile(sigma_session, 0.33))
        q66 = float(np.quantile(sigma_session, 0.66))
        if sigma_per_s < q33:
            out["vol_regime_label"] = 0.0
        elif sigma_per_s < q66:
            out["vol_regime_label"] = 1.0
        else:
            out["vol_regime_label"] = 2.0
    else:
        out["vol_regime_label"] = 1.0

    # Trending flag: high path efficiency
    out["is_trending"] = 0.0  # set later via interaction in caller if path_efficiency known

    # Calm market flag: sigma below 25th percentile
    if len(sigma_session) >= 10:
        q25 = float(np.quantile(sigma_session, 0.25))
        out["is_calm_market"] = 1.0 if sigma_per_s < q25 else 0.0
    else:
        out["is_calm_market"] = 0.0

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(
    snapshot,
    ctx: dict,
    *,
    history: Optional[Sequence[float]] = None,
    ts_history: Optional[Sequence[int]] = None,
    window_duration_s: float = 300.0,
    sigma_per_s: Optional[float] = None,
    p_gbm: Optional[float] = None,
) -> dict[str, float]:
    """Compute the full Tier-A feature set for one tick.

    Args:
        snapshot: backtest_core.Snapshot — current book state
        ctx: live signal context dict (mutable, used for rolling buffers)
        history: list of recent prices (binance_mid or chainlink). Defaults
                 to ctx['price_history'] if not given.
        ts_history: list of timestamps in ms matching `history`
        window_duration_s: window length in seconds (300 for 5m, 900 for 15m)
        sigma_per_s: smoothed sigma from the GBM signal. Defaults to
                     ctx['_sigma_per_s'] if not given.
        p_gbm: GBM probability (P_up). Defaults to ctx['_p_model_raw'].

    Returns:
        dict[str, float] with ~95-100 features. Every value is a float;
        missing/invalid inputs return 0.0.
    """
    if history is None:
        history = ctx.get("price_history", [])
    if ts_history is None:
        ts_history = ctx.get("ts_history", [])
    if sigma_per_s is None:
        sigma_per_s = ctx.get("_sigma_per_s", 0.0) or 0.0
    if p_gbm is None:
        p_gbm = ctx.get("_p_model_raw")

    feats: dict[str, float] = {}
    feats.update(_features_momentum_and_vol(history, ts_history, sigma_per_s))
    feats.update(_features_microstructure(snapshot, ctx))
    feats.update(_features_polymarket(snapshot, ctx, p_gbm))
    feats.update(_features_window_path(snapshot, ctx, history, ts_history,
                                        window_duration_s, sigma_per_s))
    feats.update(_features_higher_moments(history, ts_history))
    feats.update(_features_time(snapshot))
    feats.update(_features_information(history, ts_history))
    feats.update(_features_regime(history, ts_history, sigma_per_s, ctx))

    return feats


def feature_names() -> list[str]:
    """Return the canonical ordered list of feature names this module produces.

    Used by the training script to lock in column ordering across train
    and inference. If a feature is missing in compute_features, this list
    is the source of truth — always pass `feats.get(name, 0.0)` for safety.
    """
    return [
        # Category 1: momentum + RV
        "bn_ret_1s", "bn_ret_5s", "bn_ret_10s", "bn_ret_30s",
        "bn_ret_60s", "bn_ret_120s", "bn_ret_300s",
        "bn_accel", "bn_run_length",
        "RV_30s", "RV_120s", "RV_300s",
        "log_RV_ratio_short", "log_RV_ratio_long",
        "bipower_variation_300s", "realized_jump_var_300s", "jump_indicator",
        "mom_zscore_30s", "mom_zscore_60s", "mom_zscore_300s",
        "mom_reversal_signal", "frac_diff_d04",
        # Category 2: microstructure
        "microprice_up", "microprice_down",
        "microprice_offset_up", "microprice_offset_down",
        "microprice_drift_5s", "microprice_drift_30s",
        "top_queue_imbalance_up", "top_queue_imbalance_down",
        "book_pressure_up", "book_pressure_down",
        "cross_book_imbalance_sum", "cross_book_imbalance_diff",
        "book_convexity_up", "book_convexity_down",
        "spread_compression_ratio", "spread_vs_typical",
        "quote_stability_30s",
        "largest_order_ratio", "book_depth_total",
        # Category 3: Polymarket
        "pm_book_age_ms",
        "pm_time_since_last_trade_ms",
        "pm_tick_stability_count",
        "pm_p_residual", "pm_p_residual_persistence", "pm_p_residual_drift",
        "pm_n_levels_meaningful", "pm_largest_ask_size", "pm_unique_makers_proxy",
        # Category 4: window path
        "tau_frac", "delta_per_typical_move",
        "peak_delta_seen", "dd_from_peak_z", "signed_z_range",
        "n_zero_crossings", "n_zero_crossings_normalized",
        "run_length_current", "elapsed_frac_x_signed_z",
        "n_trades_fired_already",
        # Category 5: higher moments
        "realized_skew_60s", "realized_skew_300s",
        "realized_kurt_60s", "realized_kurt_300s",
        "vol_of_vol_30s",
        "parkinson_sigma_1m_60", "garman_klass_sigma_1m_60",
        # Category 7: time/calendar
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
        "dow_one_hot_sat", "dow_one_hot_sun", "dow_one_hot_mon",
        "hours_since_us_open", "hours_since_asia_open",
        "is_us_session", "is_eu_session",
        "minutes_to_funding_settle", "is_options_expiry",
        "minute_of_hour",
        # Category 9: information theory
        "path_efficiency_60s", "permutation_entropy_60s", "sample_entropy_proxy",
        # Category 10: regime
        "choppiness_index_30s",
        "sigma_regime_quantile_session", "vol_regime_label",
        "is_trending", "is_calm_market",
    ]
