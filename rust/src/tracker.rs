use std::collections::{HashMap, VecDeque};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde_json::json;

use crate::client::{OrderClient, OrderResponse};
use crate::config::ArbParams;
use crate::market_api::Market;
use crate::signal::{Decision, Side};

const RECENT_TRADES_MAX: usize = 5;

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
    /// BTC price at fire time (Chainlink). Approximates window-start price.
    pub entry_price: Option<f64>,
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
}

#[allow(dead_code)]
pub struct Tracker {
    pub bankroll: f64,
    pub starting_bankroll: f64,
    pub last_fire_ms: i64,
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
            last_fire_ms: 0,
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

    /// Cap on open positions (defensive — we do not double-enter on the same window).
    pub fn can_enter(&self, market: &Market) -> bool {
        !self.open_positions.iter().any(|p| p.slug == market.slug)
    }

    /// Fast, synchronous check + reserve. Optimistically updates `last_fire_ms`
    /// so the cooldown starts *immediately* (before the order round-trip
    /// completes). Returns None if we shouldn't fire.
    ///
    /// Lock is held only for the duration of this call (microseconds); the
    /// caller releases it before the await on the order POST.
    pub fn prepare_fire(
        &mut self,
        decision: &Decision,
        market: &Market,
        ask: f64,
        arb: &ArbParams,
        entry_price: Option<f64>,
        market_name: String,
        window_duration_s: i64,
    ) -> Option<FireTicket> {
        if decision.side == Side::Flat {
            return None;
        }
        if !self.can_enter(market) {
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

        self.last_fire_ms = Utc::now().timestamp_millis();

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
                };
                eprintln!(
                    "  [FILL] {:?} shares={:.2} cost=${:.2} bankroll=${:.2}",
                    ticket.side, shares, cost, self.bankroll
                );
                self.log_fill(&pos, &resp.order_id, &ticket.reason);
                self.open_positions.push(pos);
            }
            Ok(resp) => {
                eprintln!(
                    "  [REJECT] {:?} status={} err={:?}",
                    ticket.side, resp.status, resp.error_msg
                );
            }
            Err(e) => {
                eprintln!("  [ORDER_ERR] {:?} {e}", ticket.side);
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

    fn log_fill(&self, pos: &Position, order_id: &str, reason: &str) {
        let record = json!({
            "ts": Utc::now().to_rfc3339(),
            "slug": pos.slug,
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
