use crate::config::ArbParams;

#[derive(Clone, Debug)]
pub struct Snapshot {
    pub ts_ms: i64,
    pub time_remaining_s: f64,
    pub best_ask_up: Option<f64>,
    pub best_ask_down: Option<f64>,
    pub window_start_price: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Side {
    Flat,
    BuyUp,
    BuyDown,
}

#[derive(Clone, Debug)]
pub struct Decision {
    pub side: Side,
    pub size_usd: f64,
    pub reason: String,
}

impl Decision {
    pub fn flat(reason: impl Into<String>) -> Self {
        Self { side: Side::Flat, size_usd: 0.0, reason: reason.into() }
    }
}

pub struct ArbState {
    pub last_fire_ms: i64,
    pub bankroll: f64,
    /// EWMA σ_2s in USD, updated every ~2s from Binance ring. Persistent
    /// across ticks — doesn't recompute from scratch each evaluation.
    pub sigma_2s: f64,
    /// Sum of `Position::slots()` for positions already open on THIS market's
    /// current window. Used to inflate the Binance-delta threshold per
    /// successive fire: each slot already filled makes the next fire require
    /// a stronger signal (`ramp_per_slot`). At slots_used=0 (no existing
    /// fire) threshold is unchanged.
    pub slots_used: f64,
}

/// Polymarket dynamic taker fee model — currently 7.2% * p * (1-p).
/// See research note 2026-04-14: Polymarket's live 15m fee may peak at
/// ~3.15% (k≈0.126). Verify against a real fill before treating this as
/// accurate; underestimating fees inflates edge_proxy.
fn poly_fee(ask: f64) -> f64 {
    0.072 * ask * (1.0 - ask)
}

/// Delta over a lookback window ending at the latest ring sample.
/// Returns (delta_usd, window_age_s, cur_price). None if not enough samples.
fn ring_delta(ring: &[(i64, f64)], window_ms: i64) -> Option<(f64, f64, f64)> {
    if ring.len() < 2 {
        return None;
    }
    let (cur_ts, cur_price) = *ring.last().unwrap();
    let cutoff_ms = cur_ts - window_ms;
    let (old_ts, old_price) = ring.iter().find(|(ts, _)| *ts >= cutoff_ms).copied()?;
    Some((cur_price - old_price, (cur_ts - old_ts) as f64 / 1000.0, cur_price))
}

/// Pure Binance + Coinbase consensus latency arb.
///
/// Gates (in order):
///   1. Tau (time remaining in window).
///   2. Cooldown.
///   3. Binance 2s delta vs threshold.
///   4. Book-staleness band.
///   5. NEW: trend confirmation — 15s Binance delta same sign + >=50% magnitude.
///   6. NEW: cross-venue consensus — Coinbase 2s delta same sign + >=50% threshold.
///   7. Ask in entry band.
///   8. Size sanity.
pub fn decide_latency_arb(
    snap: &Snapshot,
    book_age_ms: Option<f64>,
    binance_ring: &[(i64, f64)],
    coinbase_ring: &[(i64, f64)],
    arb: &ArbParams,
    state: &ArbState,
) -> Decision {
    // ── 1. Tau gate ──────────────────────────────────────────────
    let tau = snap.time_remaining_s;
    if tau <= arb.min_tau_s {
        return Decision::flat(format!("tau {tau:.0}s <= min {:.0}s", arb.min_tau_s));
    }

    // ── 2. Cooldown ──────────────────────────────────────────────
    if state.last_fire_ms > 0 {
        let elapsed_s = (snap.ts_ms - state.last_fire_ms) as f64 / 1000.0;
        if elapsed_s < arb.cooldown_s {
            return Decision::flat(format!("arb cooldown {:.1}s", arb.cooldown_s - elapsed_s));
        }
    }

    // ── 3. Binance 2s delta ──────────────────────────────────────
    let window_ms = (arb.window_s * 1000.0) as i64;
    let (delta_usd, delta_age_s, cur_price) = match ring_delta(binance_ring, window_ms) {
        Some(v) => v,
        None => return Decision::flat("no binance sample in window"),
    };
    // Adaptive threshold: max(floor, k × σ_2s), then inflated by the
    // per-slot ramp if there are already fires on this window.
    //
    // σ_2s is EWMA-estimated (λ=0.94), updated every 2s in MarketRuntime.
    // Spike-resistant: single $30 move contributes only 6% to variance.
    //
    // Ramp: `threshold_effective = base × (1 + ramp_per_slot × slots_used)`.
    //   - slots_used = 0 (no existing fire): threshold = base
    //   - slots_used = 1 (one full fill): threshold = base × (1 + ramp)
    //   - slots_used = 2 (two full fills): threshold = base × (1 + 2·ramp)
    // Each successive fire within the same window thus requires a stronger
    // signal, reflecting that subsequent correlated fires have diminishing
    // marginal edge vs. the first.
    let sigma_2s = state.sigma_2s;
    let vol_thresh = arb.sigma_k * sigma_2s;
    let base = arb.delta_floor.max(vol_thresh);
    let ramp_factor = 1.0 + arb.ramp_per_slot * state.slots_used;
    let adaptive = base * ramp_factor;
    if delta_usd.abs() < adaptive {
        return Decision::flat(format!(
            "|delta|=${:.2} < ${:.2} (base=${:.2} ramp×{:.2} slots={:.2} σ=${:.2} k×σ=${:.2} floor=${:.2} window {:.1}s)",
            delta_usd.abs(),
            adaptive,
            base,
            ramp_factor,
            state.slots_used,
            sigma_2s,
            vol_thresh,
            arb.delta_floor,
            delta_age_s
        ));
    }

    // ── 4. Book staleness ────────────────────────────────────────
    let book_age = book_age_ms.unwrap_or(0.0);
    if book_age < arb.book_stale_ms {
        return Decision::flat(format!(
            "book fresh ({book_age:.0}ms < {:.0}ms)",
            arb.book_stale_ms
        ));
    }
    if book_age > 5000.0 {
        return Decision::flat(format!("book too stale ({book_age:.0}ms > 5000ms)"));
    }

    // ── 5. Trend confirmation (15s Binance) ──────────────────────
    // Rationale: a 2s pop against a 15s trend is often a retracement
    // wick. Real arbs line up across horizons.
    let trend_window_ms = (arb.trend_window_s * 1000.0) as i64;
    match ring_delta(binance_ring, trend_window_ms) {
        Some((trend_delta, _, _)) => {
            if trend_delta.signum() != delta_usd.signum() {
                return Decision::flat(format!(
                    "trend disagree: 2s={:+.1} vs {:.0}s={:+.1}",
                    delta_usd, arb.trend_window_s, trend_delta
                ));
            }
            let needed = arb.trend_min_ratio * delta_usd.abs();
            if trend_delta.abs() < needed {
                return Decision::flat(format!(
                    "trend weak: |{:.0}s|={:.1} < {:.1}*|2s|={:.1}",
                    arb.trend_window_s, trend_delta.abs(), arb.trend_min_ratio, needed
                ));
            }
        }
        None => return Decision::flat("no binance sample in trend window"),
    }

    // ── 6. Cross-venue consensus (Coinbase 2s) ───────────────────
    // Binance-specific wicks that Coinbase doesn't confirm are often
    // where makers intentionally leave stale quotes as traps.
    //
    // Consensus threshold scales off `arb.delta_floor` (per-asset) rather
    // than the legacy `arb.delta_usd` constant, so ETH doesn't inherit
    // BTC's $12.50 Coinbase requirement (which on $2.4k ETH is a 0.5%
    // move in 2s — effectively impossible, silently blocking every
    // single ETH fire).
    match ring_delta(coinbase_ring, window_ms) {
        Some((cb_delta, _, _)) => {
            if cb_delta.signum() != delta_usd.signum() {
                return Decision::flat(format!(
                    "coinbase disagree: bn={:+.2} cb={:+.2}",
                    delta_usd, cb_delta
                ));
            }
            let needed = arb.crossvenue_min_ratio * arb.delta_floor;
            if cb_delta.abs() < needed {
                return Decision::flat(format!(
                    "coinbase weak: |cb|={:.2} < {:.2}*floor={:.2}",
                    cb_delta.abs(), arb.crossvenue_min_ratio, needed
                ));
            }
        }
        None => return Decision::flat("no coinbase sample in window"),
    }

    // ── 7. Ask band ──────────────────────────────────────────────
    let (side, ask) = if delta_usd > 0.0 {
        (Side::BuyUp, snap.best_ask_up)
    } else {
        (Side::BuyDown, snap.best_ask_down)
    };
    let ask = match ask {
        Some(a) if a > 0.0 && a < 1.0 => a,
        _ => return Decision::flat("invalid ask"),
    };
    if ask < arb.min_ask || ask > arb.max_ask {
        return Decision::flat(format!(
            "ask {ask:.3} outside arb band [{:.2}, {:.2}]",
            arb.min_ask, arb.max_ask
        ));
    }

    let fee = poly_fee(ask);
    let spot = if cur_price > 0.0 { cur_price } else { snap.window_start_price.max(1.0) };
    let delta_bps = delta_usd.abs() / spot * 10_000.0;
    let edge_proxy = (1.0 - ask) - fee;

    // ── 8. Z-score proportional sizing ──────────────────────────
    // Stronger signals (higher z = |Δ|/σ) get larger bets. Scales
    // from 1.0× base at threshold to 1.5× base at 2× threshold.
    // Capped by bankroll frac. Respects min 5 shares.
    let z = if sigma_2s > 0.01 { delta_usd.abs() / sigma_2s } else { 1.0 };
    let z_scale = (z / arb.sigma_k).clamp(1.0, 1.5);
    let size_usd = (arb.size_usd * z_scale).min(state.bankroll * arb.max_bankroll_frac);
    if size_usd < arb.min_order_shares * ask {
        return Decision::flat(format!(
            "size ${size_usd:.2} below min (z={z:.1} scale={z_scale:.2})"
        ));
    }

    let sign = if delta_usd >= 0.0 { "+" } else { "-" };
    let reason = format!(
        "LATENCY_ARB delta={sign}${:.2} ({sign}{:.1}bp) in {:.1}s book_age={book_age:.0}ms \
         ask={ask:.3} fee={:.2}% edge={edge_proxy:.4} z={z:.1} size=${size_usd:.2}",
        delta_usd.abs(),
        delta_bps,
        delta_age_s,
        fee * 100.0,
    );

    Decision { side, size_usd, reason }
}
