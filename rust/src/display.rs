use chrono::{DateTime, FixedOffset, Utc};

use crate::book::BookSnapshot;
use crate::signal::Side;
use crate::tracker::{Position, ResolvedTrade};

/// Eastern time. Hardcoded UTC-4 (EDT). Switch to chrono-tz if you need
/// proper DST handling — for live dashboards this is fine.
fn et() -> FixedOffset {
    FixedOffset::west_opt(4 * 3600).unwrap()
}

fn fmt_et_time(t: DateTime<Utc>) -> String {
    t.with_timezone(&et()).format("%-I:%M %p").to_string()
}

fn fmt_price(p: Option<f64>) -> String {
    p.map(|v| format!("${v:>7.0}")).unwrap_or_else(|| "      —".into())
}

pub struct Row<'a> {
    pub name: &'a str,
    pub binance_mid: Option<f64>,
    pub chainlink: Option<f64>,
    pub up_snap: &'a BookSnapshot,
    pub down_snap: &'a BookSnapshot,
    pub window_end: DateTime<Utc>,
    pub ring_len: usize,
    pub delta_2s: Option<f64>,
    pub last_reason: &'a str,
}

pub fn print_dashboard(
    rows: &[Row<'_>],
    bankroll: f64,
    fills: u64,
    pnl: f64,
    wins: u64,
    losses: u64,
    open_positions: &[Position],
    recent: &[ResolvedTrade],
) {
    let now = Utc::now();
    let now_et = now.with_timezone(&et());
    print!("\x1b[2J\x1b[H");
    println!(
        "━━━ BTC Latency Arb · {} ET · bankroll ${:.2} fills={} w/l={}/{} realized=${:+.2} ━━━",
        now_et.format("%-I:%M:%S %p"),
        bankroll,
        fills,
        wins,
        losses,
        pnl,
    );
    for r in rows {
        let tau = (r.window_end - now).num_seconds();
        let bid_up = r.up_snap.best_bid.map(|v| format!("{v:.3}")).unwrap_or_else(|| "—".into());
        let ask_up = r.up_snap.best_ask.map(|v| format!("{v:.3}")).unwrap_or_else(|| "—".into());
        let bid_dn = r.down_snap.best_bid.map(|v| format!("{v:.3}")).unwrap_or_else(|| "—".into());
        let ask_dn = r.down_snap.best_ask.map(|v| format!("{v:.3}")).unwrap_or_else(|| "—".into());
        let binance = r.binance_mid.map(|v| format!("${v:.0}")).unwrap_or_else(|| "—".into());
        let chainlink = r.chainlink.map(|v| format!("${v:.0}")).unwrap_or_else(|| "—".into());
        let delta = r.delta_2s.map(|v| format!("{v:+.2}")).unwrap_or_else(|| "—".into());
        println!(
            "  {:<8} tau={:>4}s  BIN={} CL={} UP={}/{} DN={}/{} Δ2s={} ring={}",
            r.name, tau, binance, chainlink, bid_up, ask_up, bid_dn, ask_dn, delta, r.ring_len
        );
        if !r.last_reason.is_empty() {
            println!("           └─ {}", r.last_reason);
        }
    }

    if !open_positions.is_empty() {
        println!();
        println!(
            "  ── open positions ({}) ────────────────────────────────────",
            open_positions.len()
        );
        for p in open_positions {
            let window_start = p.window_end - chrono::Duration::seconds(p.window_duration_s);
            let cost = p.shares * p.avg_price;
            let tau = (p.window_end - now).num_seconds().max(0);
            let side_str = match p.side {
                Side::BuyUp => "UP  ",
                Side::BuyDown => "DOWN",
                Side::Flat => "    ",
            };
            println!(
                "    {:<7} {}-{} ET  bet {}  @{:.3}  cost ${:5.2}  shares {:6.2}  entry {}  τ={:>4}s",
                p.market_name,
                fmt_et_time(window_start),
                fmt_et_time(p.window_end),
                side_str,
                p.avg_price,
                cost,
                p.shares,
                fmt_price(p.entry_price),
                tau,
            );
        }
    }

    if !recent.is_empty() {
        println!();
        println!("  ── last {} resolved ──────────────────────────────────────", recent.len());
        for t in recent {
            let result = if t.won { "WIN " } else { "LOSS" };
            // Compare full-precision f64s; display values are rounded but the
            // direction must reflect the actual settlement comparison.
            let arrow = match (t.entry_price, t.final_price) {
                (Some(e), Some(f)) if f > e => "↑",
                (Some(e), Some(f)) if f < e => "↓",
                (Some(_), Some(_)) => "=",
                _ => "?",
            };
            let side_str = match t.side {
                Side::BuyUp => "UP  ",
                Side::BuyDown => "DOWN",
                Side::Flat => "    ",
            };
            println!(
                "    {:<7} {}-{} ET  bet {}  {} {} {}  pnl={:+6.2}  ({})",
                t.market_name,
                fmt_et_time(t.window_start),
                fmt_et_time(t.window_end),
                side_str,
                fmt_price(t.entry_price),
                arrow,
                fmt_price(t.final_price),
                t.trade_pnl,
                if t.won { result } else { result },
            );
        }
    }
}
