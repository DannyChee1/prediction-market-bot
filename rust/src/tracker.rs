use std::collections::{HashMap, VecDeque};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde_json::json;

use crate::client::{OrderClient, OrderResponse};
use crate::config::{ArbParams, Asset};
use crate::market_api::Market;
use crate::signal::{Decision, Side};

const RECENT_TRADES_MAX: usize = 5;

fn default_asset_btc() -> Asset {
    Asset::Btc
}

/// Persisted bot state — survives restarts via `bot_state.json`.
///
/// The `last_fire_ms` field used to live here but is now per-market only
/// (on the signal loop's ArbState). Old state files that still carry it
/// just get ignored via `#[serde(default)]` on the new fields.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct BotState {
    pub bankroll: f64,
    pub starting_bankroll: f64,
    pub fills: u64,
    pub wins: u64,
    pub losses: u64,
    pub realized_pnl: f64,
    pub open_positions: Vec<PersistedPosition>,
}

/// Minimal position representation for persistence (avoids DateTime serde issues).
#[derive(serde::Serialize, serde::Deserialize)]
pub struct PersistedPosition {
    pub token_id: String,
    pub side: Side,
    pub avg_price: f64,
    pub shares: f64,
    pub entered_at_ms: i64,
    pub window_end_ms: i64,
    pub window_duration_s: i64,
    pub slug: String,
    pub market_name: String,
    pub condition_id: String,
    pub entry_price: Option<f64>,
    /// Defaults to BTC so state files written before ETH support round-trip cleanly.
    #[serde(default = "default_asset_btc")]
    pub asset: Asset,
    /// Intended USD notional at fire time (before FAK partial-fill truncation).
    /// Used for fractional slot accounting — a partial fill at 70% consumes
    /// 0.7 of a position slot rather than 1.0. Defaults to 0.0 in legacy
    /// state files, in which case Position::slots() falls back to 1.0
    /// (original count-based semantics).
    #[serde(default)]
    pub intended_size_usd: f64,
}

/// An open taker position.
#[allow(dead_code)]
pub struct Position {
    pub token_id: String,
    pub side: Side,
    pub avg_price: f64,
    pub shares: f64,
    pub entered_at: DateTime<Utc>,
    pub window_end: DateTime<Utc>,
    pub window_duration_s: i64,
    pub slug: String,
    pub market_name: String,
    pub condition_id: String,
    /// Underlying price at fire time (Chainlink). Approximates window-start price.
    pub entry_price: Option<f64>,
    pub asset: Asset,
    /// USD notional we TRIED to take (`decision.size_usd` at fire time). The
    /// `shares * avg_price` product is the USD we actually spent after FAK
    /// partial-fill truncation. `cost / intended_size_usd` is the fraction
    /// of a "position slot" this fill consumed.
    pub intended_size_usd: f64,
}

impl Position {
    /// Fraction of a `max_positions_per_window` slot this position occupies.
    ///
    /// - Full FOK fill: `cost == intended_size_usd` → 1.0
    /// - FAK 70% partial: → 0.7
    /// - Legacy Position without `intended_size_usd` (pre-FAK state file):
    ///   fall back to 1.0 so old positions still cap the window correctly.
    pub fn slots(&self) -> f64 {
        let cost = self.shares * self.avg_price;
        if self.intended_size_usd > 0.0 {
            (cost / self.intended_size_usd).max(0.0)
        } else {
            1.0
        }
    }
}

/// One row of the "last resolved trades" dashboard section.
#[derive(Clone)]
#[allow(dead_code)]
pub struct ResolvedTrade {
    pub market_name: String,
    pub window_start: DateTime<Utc>,
    pub window_end: DateTime<Utc>,
    pub side: Side,
    pub entry_price: Option<f64>,
    pub final_price: Option<f64>,
    pub trade_pnl: f64,
    pub won: bool,
    pub resolved_at: DateTime<Utc>,
}

/// Captures everything needed to fire an order AND later record its outcome
/// without holding the tracker lock across the HTTP round-trip.
#[derive(Clone)]
pub struct FireTicket {
    pub token_id: String,
    pub side: Side,
    pub size_usd: f64,
    pub ask: f64,
    pub slug: String,
    pub market_name: String,
    pub condition_id: String,
    pub window_end: DateTime<Utc>,
    pub window_duration_s: i64,
    pub entry_price: Option<f64>,
    pub reason: String,
    pub asset: Asset,
}

#[allow(dead_code)]
pub struct Tracker {
    pub bankroll: f64,
    pub starting_bankroll: f64,
    pub client: Option<Arc<OrderClient>>,
    pub dry_run: bool,
    pub fills: u64,
    pub wins: u64,
    pub losses: u64,
    pub realized_pnl: f64,
    pub open_positions: Vec<Position>,
    pub log_path: PathBuf,
    pub recent_trades: VecDeque<ResolvedTrade>,
    /// Rolling counters of why FLAT decisions were returned. Useful for
    /// post-hoc analysis of how aggressive each gate is. Not displayed.
    pub filter_counts: HashMap<String, u64>,
}

impl Tracker {
    pub fn new(bankroll: f64, client: Option<Arc<OrderClient>>, dry_run: bool, log_path: PathBuf) -> Self {
        Self {
            bankroll,
            starting_bankroll: bankroll,
            client,
            dry_run,
            fills: 0,
            wins: 0,
            losses: 0,
            realized_pnl: 0.0,
            open_positions: Vec::new(),
            log_path,
            recent_trades: VecDeque::with_capacity(RECENT_TRADES_MAX + 1),
            filter_counts: HashMap::new(),
        }
    }

    /// Increment a flat-reason counter. Categorisation is done in main.rs to
    /// keep this method cheap and lock-friendly.
    pub fn note_filter(&mut self, category: &'static str) {
        *self.filter_counts.entry(category.to_string()).or_insert(0) += 1;
    }

    /// Check if we can open another position on this market window.
    ///
    /// `max_per_window` is interpreted as a FRACTIONAL NOTIONAL CAP, not a
    /// count cap. Each existing position contributes `slots() = cost /
    /// intended_size_usd` (1.0 for full fills, <1.0 for FAK partials).
    /// Allow a new fire iff the sum of slots is below the cap.
    ///
    /// Rationale: under FAK, partial fills are common and a "2 positions
    /// per window" cap should mean "≤2× base-notional committed," not "≤2
    /// fill events (regardless of how little each filled)." Otherwise
    /// two 30%-filled positions would exhaust the window even though
    /// we've only used 0.6 slots of our 2.0 intended notional.
    ///
    /// Direction-block (no hedging) is unchanged: all existing positions on
    /// this slug must already be on `side`, else reject.
    pub fn can_enter(&self, market: &Market, side: &Side, max_per_window: usize) -> bool {
        let existing: Vec<&Position> = self
            .open_positions
            .iter()
            .filter(|p| p.slug == market.slug)
            .collect();
        // Block opposite-direction entries (guaranteed 2× fees, ~0 edge).
        if !existing.iter().all(|p| p.side == *side) {
            return false;
        }
        // Fractional slot cap. A 70%-filled position uses 0.7 slots, freeing
        // 0.3 for a future fire. The intent of `max_per_window` is preserved
        // (notional ceiling = max × base_size) while allowing more fills
        // when earlier ones were partial.
        let slot_sum: f64 = existing.iter().map(|p| p.slots()).sum();
        slot_sum < max_per_window as f64
    }

    /// Fast, synchronous check + reserve. Returns None if we shouldn't fire.
    ///
    /// Cooldown is enforced per-market by the signal loop — we do not touch
    /// shared cooldown state here. Lock is held only for the duration of this
    /// call (microseconds); the caller releases it before the order POST.
    pub fn prepare_fire(
        &mut self,
        decision: &Decision,
        market: &Market,
        ask: f64,
        arb: &ArbParams,
        entry_price: Option<f64>,
        market_name: String,
        window_duration_s: i64,
        asset: Asset,
    ) -> Option<FireTicket> {
        if decision.side == Side::Flat {
            return None;
        }
        if !self.can_enter(market, &decision.side, arb.max_positions_per_window) {
            return None;
        }
        let token_id = match decision.side {
            Side::BuyUp => market.up_token.clone(),
            Side::BuyDown => market.down_token.clone(),
            Side::Flat => return None,
        };
        let size_usd = decision.size_usd.min(self.bankroll * arb.max_bankroll_frac);
        if size_usd <= 0.0 {
            return None;
        }

        Some(FireTicket {
            token_id,
            side: decision.side.clone(),
            size_usd,
            ask,
            slug: market.slug.clone(),
            market_name,
            condition_id: market.condition_id.clone(),
            window_end: market.end_time,
            window_duration_s,
            entry_price,
            reason: decision.reason.clone(),
            asset,
        })
    }

    /// Reconcile the order result into tracker state. Call after the HTTP
    /// round-trip completes, with the lock re-acquired.
    pub fn record_fire(&mut self, ticket: FireTicket, result: anyhow::Result<OrderResponse>) {
        match result {
            Ok(resp) if resp.success && resp.status == "MATCHED" => {
                let shares = resp
                    .taking_amount
                    .as_ref()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(ticket.size_usd / ticket.ask.max(0.0001));
                let cost = resp
                    .making_amount
                    .as_ref()
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(ticket.size_usd);
                self.fills += 1;
                self.bankroll -= cost;
                let pos = Position {
                    token_id: ticket.token_id.clone(),
                    side: ticket.side.clone(),
                    avg_price: if shares > 0.0 { cost / shares } else { ticket.ask },
                    shares,
                    entered_at: Utc::now(),
                    window_end: ticket.window_end,
                    window_duration_s: ticket.window_duration_s,
                    slug: ticket.slug.clone(),
                    market_name: ticket.market_name.clone(),
                    condition_id: ticket.condition_id.clone(),
                    entry_price: ticket.entry_price,
                    asset: ticket.asset,
                    intended_size_usd: ticket.size_usd,
                };
                // Fill rate = actual cost / intended cost. For FOK this is
                // always 1.0 (or the order is REJECT). For FAK, <1.0 means
                // the book had less depth than we asked for; the remainder
                // was cancelled. Logged for visibility — track over time
                // to know if you're hitting capacity ceilings.
                let fill_rate = if ticket.size_usd > 0.0 {
                    cost / ticket.size_usd
                } else {
                    1.0
                };
                eprintln!(
                    "  [FILL] {} {:?} shares={:.2} cost=${:.2}/${:.2} ({:.0}% fill, {:.2} slot) bankroll=${:.2}",
                    ticket.market_name,
                    ticket.side,
                    shares,
                    cost,
                    ticket.size_usd,
                    fill_rate * 100.0,
                    pos.slots(),
                    self.bankroll,
                );
                self.log_fill(&pos, &resp.order_id, &ticket.reason);
                self.open_positions.push(pos);
            }
            Ok(resp) => {
                eprintln!(
                    "  [REJECT] {} {:?} status={} err={:?}",
                    ticket.market_name, ticket.side, resp.status, resp.error_msg
                );
            }
            Err(e) => {
                eprintln!(
                    "  [ORDER_ERR] {} {:?} {e}",
                    ticket.market_name, ticket.side
                );
            }
        }
    }

    /// Apply a resolution outcome to one open position. Credits bankroll
    /// with the payout, updates counters/PnL, removes from open_positions,
    /// and appends a row to recent_trades for the dashboard.
    ///
    /// Returns (condition_id, won) when a position was actually resolved —
    /// caller uses this to fire auto-redemption on wins.
    pub fn resolve_position(
        &mut self,
        slug: &str,
        up_won: u8,
        final_price: Option<f64>,
    ) -> Option<(String, bool)> {
        let idx = self.open_positions.iter().position(|p| p.slug == slug)?;
        let pos = self.open_positions.remove(idx);
        let pos_condition_id = pos.condition_id.clone();
        let won = (up_won == 1 && pos.side == Side::BuyUp)
            || (up_won == 0 && pos.side == Side::BuyDown);
        let payout = if won { pos.shares } else { 0.0 };
        let cost = pos.shares * pos.avg_price;
        let trade_pnl = payout - cost;

        self.bankroll += payout;
        self.realized_pnl += trade_pnl;
        if won {
            self.wins += 1;
        } else {
            self.losses += 1;
        }

        eprintln!(
            "  [RESOLVE] {slug} {} side={:?} shares={:.2} cost=${:.2} payout=${:.2} \
             trade_pnl={:+.2} bankroll=${:.2} w/l={}/{}",
            if up_won == 1 { "UP" } else { "DOWN" },
            pos.side,
            pos.shares,
            cost,
            payout,
            trade_pnl,
            self.bankroll,
            self.wins,
            self.losses,
        );

        let record = json!({
            "ts": Utc::now().to_rfc3339(),
            "event": "RESOLVE",
            "slug": slug,
            "asset": pos.asset,
            "side": format!("{:?}", pos.side),
            "winner": if up_won == 1 { "UP" } else { "DOWN" },
            "shares": pos.shares,
            "cost": cost,
            "payout": payout,
            "trade_pnl": trade_pnl,
            "bankroll_after": self.bankroll,
            "wins": self.wins,
            "losses": self.losses,
            "entry_price": pos.entry_price,
            "final_price": final_price,
        });
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&self.log_path) {
            let _ = writeln!(f, "{}", record);
        }

        let window_start = pos.window_end - chrono::Duration::seconds(pos.window_duration_s);
        self.recent_trades.push_back(ResolvedTrade {
            market_name: pos.market_name,
            window_start,
            window_end: pos.window_end,
            side: pos.side,
            entry_price: pos.entry_price,
            final_price,
            trade_pnl,
            won,
            resolved_at: Utc::now(),
        });
        while self.recent_trades.len() > RECENT_TRADES_MAX {
            self.recent_trades.pop_front();
        }

        Some((pos_condition_id, won))
    }

    /// Save bot state to disk. Called on clean shutdown and periodically.
    pub fn save_state(&self, path: &std::path::Path) {
        let state = BotState {
            bankroll: self.bankroll,
            starting_bankroll: self.starting_bankroll,
            fills: self.fills,
            wins: self.wins,
            losses: self.losses,
            realized_pnl: self.realized_pnl,
            open_positions: self.open_positions.iter().map(|p| PersistedPosition {
                token_id: p.token_id.clone(),
                side: p.side.clone(),
                avg_price: p.avg_price,
                shares: p.shares,
                entered_at_ms: p.entered_at.timestamp_millis(),
                window_end_ms: p.window_end.timestamp_millis(),
                window_duration_s: p.window_duration_s,
                slug: p.slug.clone(),
                market_name: p.market_name.clone(),
                condition_id: p.condition_id.clone(),
                entry_price: p.entry_price,
                asset: p.asset,
                intended_size_usd: p.intended_size_usd,
            }).collect(),
        };
        match serde_json::to_string_pretty(&state) {
            Ok(json) => {
                if let Err(e) = std::fs::write(path, json) {
                    eprintln!("  [STATE] save failed: {e}");
                }
            }
            Err(e) => eprintln!("  [STATE] serialize failed: {e}"),
        }
    }

    /// Restore bot state from disk. Returns None if file doesn't exist.
    pub fn load_state(path: &std::path::Path) -> Option<BotState> {
        let data = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&data).ok()
    }

    /// Apply a loaded state snapshot to this tracker instance.
    pub fn apply_state(&mut self, state: BotState) {
        self.bankroll = state.bankroll;
        self.starting_bankroll = state.starting_bankroll;
        self.fills = state.fills;
        self.wins = state.wins;
        self.losses = state.losses;
        self.realized_pnl = state.realized_pnl;
        self.open_positions = state.open_positions.into_iter().map(|p| {
            use chrono::TimeZone;
            Position {
                token_id: p.token_id,
                side: p.side,
                avg_price: p.avg_price,
                shares: p.shares,
                entered_at: Utc.timestamp_millis_opt(p.entered_at_ms).unwrap(),
                window_end: Utc.timestamp_millis_opt(p.window_end_ms).unwrap(),
                window_duration_s: p.window_duration_s,
                slug: p.slug,
                market_name: p.market_name,
                condition_id: p.condition_id,
                entry_price: p.entry_price,
                asset: p.asset,
                intended_size_usd: p.intended_size_usd,
            }
        }).collect();
        eprintln!(
            "  [RESTORE] bankroll=${:.2} fills={} w/l={}/{} realized=${:+.2} open={}",
            self.bankroll, self.fills, self.wins, self.losses, self.realized_pnl,
            self.open_positions.len(),
        );
    }

    fn log_fill(&self, pos: &Position, order_id: &str, reason: &str) {
        let record = json!({
            "ts": Utc::now().to_rfc3339(),
            "slug": pos.slug,
            "asset": pos.asset,
            "side": format!("{:?}", pos.side),
            "token_id": pos.token_id,
            "avg_price": pos.avg_price,
            "shares": pos.shares,
            "order_id": order_id,
            "bankroll_after": self.bankroll,
            "reason": reason,
        });
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&self.log_path) {
            let _ = writeln!(f, "{}", record);
        }
    }
}
