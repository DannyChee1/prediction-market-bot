mod book;
mod client;
mod config;
mod display;
mod feed;
mod market_api;
mod redemption;
mod signal;
mod tracker;
mod types;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use tokio::sync::Mutex;

use crate::client::{OrderClient, OrderResponse};
use crate::config::{resolve, ArbParams, MarketConfig};
use crate::display::{print_dashboard, Row};
use crate::feed::{BinanceFeed, BookFeed, CoinbaseFeed, PriceFeed};
use crate::market_api::{find_market, poll_resolution, Market, CHAIN_ID, CLOB_HOST};
use crate::signal::{decide_latency_arb, ArbState, Side, Snapshot};
use crate::tracker::Tracker;
use crate::types::{now_ms, now_s};

#[derive(Parser, Debug, Clone)]
#[command(name = "polybot-arb", about = "BTC latency-arb bot")]
struct Args {
    #[arg(long, default_value = "btc")]
    market: String,

    #[arg(long, default_value_t = 100.0)]
    bankroll: f64,

    #[arg(long, default_value_t = false)]
    dry_run: bool,

    #[arg(long, default_value_t = 30.0)]
    arb_delta_usd: f64,

    #[arg(long, default_value_t = 2.0)]
    arb_window_s: f64,

    #[arg(long, default_value_t = 4.0)]
    arb_cooldown_s: f64,

    #[arg(long, default_value_t = 600.0)]
    arb_book_stale_ms: f64,

    #[arg(long, default_value_t = 0.15)]
    arb_min_ask: f64,

    #[arg(long, default_value_t = 0.85)]
    arb_max_ask: f64,

    #[arg(long, default_value_t = 30.0)]
    arb_min_tau_s: f64,

    #[arg(long, default_value_t = 10.0)]
    arb_size_usd: f64,
}

impl Args {
    fn arb_params(&self) -> ArbParams {
        ArbParams {
            delta_usd: self.arb_delta_usd,
            window_s: self.arb_window_s,
            cooldown_s: self.arb_cooldown_s,
            book_stale_ms: self.arb_book_stale_ms,
            min_ask: self.arb_min_ask,
            max_ask: self.arb_max_ask,
            min_tau_s: self.arb_min_tau_s,
            size_usd: self.arb_size_usd,
            max_bankroll_frac: 0.05,
            min_order_shares: 5.0,
            trend_window_s: 15.0,
            trend_min_ratio: 0.5,
            crossvenue_min_ratio: 0.5,
        }
    }
}

/// Ring capacity: ~20s of Binance samples at 100 msg/s, ~60s of Coinbase
/// at 10 msg/s. Trend-check looks back 15s, so 2000/600 covers with headroom.
const BINANCE_RING_CAP: usize = 2000;
const COINBASE_RING_CAP: usize = 600;

/// A single market's runtime state: book feed, discovered market, price rings,
/// and the last flat reason (for the dashboard).
struct MarketRuntime {
    cfg: MarketConfig,
    market: Market,
    book: Arc<BookFeed>,
    binance_ring: VecDeque<(i64, f64)>,
    coinbase_ring: VecDeque<(i64, f64)>,
    last_binance_ts: f64,
    last_coinbase_ts: f64,
    last_reason: String,
}

impl MarketRuntime {
    fn push_binance_sample(&mut self, binance: &BinanceFeed) {
        let ts = binance.last_update_ts();
        if ts <= 0.0 || ts == self.last_binance_ts {
            return;
        }
        self.last_binance_ts = ts;
        if let Some(mid) = binance.mid() {
            let ts_ms = (ts * 1000.0) as i64;
            self.binance_ring.push_back((ts_ms, mid));
            while self.binance_ring.len() > BINANCE_RING_CAP {
                self.binance_ring.pop_front();
            }
        }
    }

    fn push_coinbase_sample(&mut self, coinbase: &CoinbaseFeed) {
        let ts = coinbase.last_update_ts();
        if ts <= 0.0 || ts == self.last_coinbase_ts {
            return;
        }
        self.last_coinbase_ts = ts;
        if let Some(mid) = coinbase.mid() {
            let ts_ms = (ts * 1000.0) as i64;
            self.coinbase_ring.push_back((ts_ms, mid));
            while self.coinbase_ring.len() > COINBASE_RING_CAP {
                self.coinbase_ring.pop_front();
            }
        }
    }
}

/// Map a verbose flat reason string into a short, stable category for the
/// filter-counts histogram. Order of checks matches return-frequency.
fn categorize_flat_reason(reason: &str) -> &'static str {
    if reason.starts_with("|delta|=") { "delta_below_threshold" }
    else if reason.starts_with("book fresh") { "book_fresh" }
    else if reason.starts_with("book too stale") { "book_too_stale" }
    else if reason.starts_with("tau ") { "tau" }
    else if reason.starts_with("arb cooldown") { "cooldown" }
    else if reason.starts_with("trend disagree") { "trend_disagree" }
    else if reason.starts_with("trend weak") { "trend_weak" }
    else if reason.starts_with("no binance sample in trend") { "trend_warmup" }
    else if reason.starts_with("coinbase disagree") { "coinbase_disagree" }
    else if reason.starts_with("coinbase weak") { "coinbase_weak" }
    else if reason.starts_with("no coinbase sample") { "coinbase_warmup" }
    else if reason.starts_with("no binance") { "binance_warmup" }
    else if reason.starts_with("ask ") { "ask_band" }
    else if reason.starts_with("invalid ask") { "invalid_ask" }
    else if reason.starts_with("size ") { "size_min" }
    else if reason.is_empty() { "ok" }
    else { "other" }
}

async fn market_name_for_idx(
    runtimes: &[Arc<Mutex<MarketRuntime>>],
    idx: usize,
) -> String {
    runtimes[idx].lock().await.cfg.display_name.to_string()
}

async fn window_duration_for_idx(
    runtimes: &[Arc<Mutex<MarketRuntime>>],
    idx: usize,
) -> i64 {
    runtimes[idx].lock().await.cfg.window_duration_s
}

fn build_client_from_env() -> Result<OrderClient> {
    let private_key =
        std::env::var("PRIVATE_KEY").context("PRIVATE_KEY not set in .env")?;
    let funder = std::env::var("POLY_FUNDER").context("POLY_FUNDER not set in .env")?;
    let sig_type: u8 = std::env::var("SIGNATURE_TYPE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let api_key = std::env::var("POLY_API_KEY").unwrap_or_default();
    let api_secret = std::env::var("POLY_API_SECRET").unwrap_or_default();
    let passphrase = std::env::var("POLY_PASSPHRASE").unwrap_or_default();

    if api_key.is_empty() || api_secret.is_empty() || passphrase.is_empty() {
        anyhow::bail!(
            "POLY_API_KEY/SECRET/PASSPHRASE missing in .env; run credential bootstrap first"
        );
    }

    Ok(OrderClient::new(
        CLOB_HOST,
        &private_key,
        CHAIN_ID,
        &api_key,
        &api_secret,
        &passphrase,
        sig_type,
        Some(&funder),
    ))
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();
    let args = Args::parse();
    let arb = args.arb_params();

    let configs = resolve(&args.market);
    if configs.is_empty() {
        anyhow::bail!("unknown market '{}' (use btc | btc_15m | btc_5m)", args.market);
    }

    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()?;

    eprintln!(
        "  [INIT] market='{}' bankroll=${:.2} dry_run={} arb.delta=${} window={}s stale={}-5000ms tau>={}s",
        args.market, args.bankroll, args.dry_run, arb.delta_usd, arb.window_s, arb.book_stale_ms, arb.min_tau_s
    );

    // One shared BinanceFeed, CoinbaseFeed, PriceFeed — BTC 15m + 5m use the same.
    let binance_symbol = configs[0].binance_symbol.to_string();
    let chainlink_symbol = configs[0].chainlink_symbol.to_string();
    let binance = Arc::new(BinanceFeed::new(binance_symbol.clone()));
    let coinbase = Arc::new(CoinbaseFeed::new("BTC-USD".to_string()));
    let price = Arc::new(PriceFeed::new(chainlink_symbol.clone()));

    eprintln!("  [FEEDS] Binance={binance_symbol} Coinbase=BTC-USD Chainlink={chainlink_symbol}");

    // Discover each market and spin up a book feed.
    let mut runtimes: Vec<Arc<Mutex<MarketRuntime>>> = Vec::new();
    let mut all_tokens: Vec<String> = Vec::new();
    for cfg in &configs {
        let market = find_market(&http, cfg).await.with_context(|| {
            format!("find_market failed for {}", cfg.display_name)
        })?;
        eprintln!(
            "  [MARKET] {} slug={} end={} up={} down={}",
            cfg.display_name,
            market.slug,
            market.end_time.format("%H:%M:%S"),
            &market.up_token[..market.up_token.len().min(10)],
            &market.down_token[..market.down_token.len().min(10)],
        );
        all_tokens.push(market.up_token.clone());
        all_tokens.push(market.down_token.clone());
        let book = Arc::new(BookFeed::new(vec![
            market.up_token.clone(),
            market.down_token.clone(),
        ]));
        runtimes.push(Arc::new(Mutex::new(MarketRuntime {
            cfg: cfg.clone(),
            market,
            book,
            binance_ring: VecDeque::with_capacity(BINANCE_RING_CAP),
            coinbase_ring: VecDeque::with_capacity(COINBASE_RING_CAP),
            last_binance_ts: 0.0,
            last_coinbase_ts: 0.0,
            last_reason: String::new(),
        })));
    }

    // Order client (only when not dry-running).
    let order_client: Option<Arc<OrderClient>> = if args.dry_run {
        None
    } else {
        let c = build_client_from_env()?;
        c.warmup(&all_tokens).await?;
        if let Some(addr) = c.address() {
            eprintln!("  [CLIENT] wallet={addr}");
        }
        match c.get_balance().await {
            Ok(b) => eprintln!("  [BALANCE] ${b:.2}"),
            Err(e) => eprintln!("  [BALANCE] query failed: {e}"),
        }
        Some(Arc::new(c))
    };

    let tracker = Arc::new(Mutex::new(Tracker::new(
        args.bankroll,
        order_client.clone(),
        args.dry_run,
        PathBuf::from("live_trades_btc_arb.jsonl"),
    )));

    // Rolling Chainlink price history. We capture every update; on resolution
    // we look up the price at the market's window_end (not "now"), so the
    // displayed final_price matches the actual settlement reference.
    // 2048 samples × ~1s/update ≈ 34 min of history — covers 5m and 15m.
    let chainlink_history: Arc<Mutex<VecDeque<(i64, f64)>>> =
        Arc::new(Mutex::new(VecDeque::with_capacity(2048)));
    {
        let history = chainlink_history.clone();
        let price = price.clone();
        tokio::spawn(async move {
            let mut last_ts: f64 = 0.0;
            loop {
                tokio::time::sleep(Duration::from_millis(200)).await;
                let ts = price.last_update_ts();
                if ts > last_ts {
                    last_ts = ts;
                    if let Some(p) = price.price() {
                        let mut h = history.lock().await;
                        h.push_back(((ts * 1000.0) as i64, p));
                        while h.len() > 2048 {
                            h.pop_front();
                        }
                    }
                }
            }
        });
    }

    // Spawn a single resolution watcher. Every 30s, walks open_positions and
    // polls Gamma for any whose window has ended (+ 30s buffer for UMA).
    // Applies payouts to bankroll and logs a RESOLVE row. Works identically
    // in dry-run and live — in dry-run you see realistic PnL accrue.
    {
        let tracker = tracker.clone();
        let http = http.clone();
        let history = chainlink_history.clone();
        let dry_run_flag = args.dry_run;
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(30)).await;
                let now = chrono::Utc::now();
                let ready: Vec<(String, i64)> = {
                    let t = tracker.lock().await;
                    t.open_positions
                        .iter()
                        .filter(|p| now > p.window_end + chrono::Duration::seconds(30))
                        .map(|p| (p.slug.clone(), p.window_end.timestamp_millis()))
                        .collect()
                };
                for (slug, window_end_ms) in ready {
                    if let Some(up_won) = poll_resolution(&http, &slug, 6, 5.0).await {
                        let final_price = {
                            let h = history.lock().await;
                            h.iter()
                                .rev()
                                .find(|(ts, _)| *ts <= window_end_ms)
                                .map(|(_, p)| *p)
                        };
                        let redeem_info = {
                            let mut t = tracker.lock().await;
                            t.resolve_position(&slug, up_won, final_price)
                        };
                        // Auto-redeem on wins. Skipped in dry-run. Spawned
                        // so the resolve loop keeps moving through other
                        // markets while the relayer dance takes ~5-30s.
                        if let Some((condition_id, won)) = redeem_info {
                            if won && !dry_run_flag {
                                let http2 = http.clone();
                                tokio::spawn(async move {
                                    match crate::redemption::redeem_position(
                                        &http2,
                                        &condition_id,
                                    )
                                    .await
                                    {
                                        Ok(tx_hash) => eprintln!(
                                            "  [REDEEM] OK cid={}.. tx={tx_hash}",
                                            &condition_id[..condition_id.len().min(10)]
                                        ),
                                        Err(e) => eprintln!(
                                            "  [REDEEM] FAIL cid={}.. {e}",
                                            &condition_id[..condition_id.len().min(10)]
                                        ),
                                    }
                                });
                            }
                        }
                    } else {
                        eprintln!("  [RESOLVE] {slug}: not yet finalized, will retry");
                    }
                }
            }
        });
    }

    // Periodic filter-stats logger. Every 10 minutes, dump the filter_counts
    // histogram to stderr so it lands in err.log for later analysis. User
    // never sees this on the dashboard — it's for post-hoc tuning.
    {
        let tracker = tracker.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(600)).await;
                let snapshot: Vec<(String, u64)> = {
                    let t = tracker.lock().await;
                    t.filter_counts.iter().map(|(k, v)| (k.clone(), *v)).collect()
                };
                let mut entries = snapshot;
                entries.sort_by(|a, b| b.1.cmp(&a.1));
                let summary = entries
                    .iter()
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                eprintln!("  [FILTER_STATS] {}", summary);
            }
        });
    }

    // Spawn a rotation task per market. Every 20s, checks whether the current
    // market is within 60s of end; if so, calls find_market and swaps in the
    // next window's market + a fresh BookFeed. Old BookFeed drops → its WS
    // task aborts via Drop impl on BookFeed.
    for rt in &runtimes {
        let rt = rt.clone();
        let http = http.clone();
        let oc = order_client.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(20)).await;
                let (cfg, tau_s, current_slug) = {
                    let r = rt.lock().await;
                    let tau =
                        (r.market.end_time - chrono::Utc::now()).num_seconds();
                    (r.cfg.clone(), tau, r.market.slug.clone())
                };
                if tau_s > 60 {
                    continue; // market still comfortably active
                }
                match find_market(&http, &cfg).await {
                    Ok(new_market) if new_market.slug != current_slug => {
                        // Warm up new tokens on the order client (cached
                        // tick_size + neg_risk, HTTP keep-alive) so the
                        // first fire on the new market isn't slowed by
                        // synchronous lookups.
                        if let Some(client) = &oc {
                            let tokens = vec![
                                new_market.up_token.clone(),
                                new_market.down_token.clone(),
                            ];
                            if let Err(e) = client.warmup(&tokens).await {
                                eprintln!(
                                    "  [ROTATE] warmup failed for {}: {e}",
                                    cfg.display_name
                                );
                            }
                        }
                        let new_book = Arc::new(BookFeed::new(vec![
                            new_market.up_token.clone(),
                            new_market.down_token.clone(),
                        ]));
                        let mut r = rt.lock().await;
                        eprintln!(
                            "  [ROTATE] {}: {} -> {} (end {} UTC)",
                            r.cfg.display_name,
                            r.market.slug,
                            new_market.slug,
                            new_market.end_time.format("%H:%M:%S"),
                        );
                        r.book = new_book; // old Arc drops → old WS aborts
                        r.market = new_market;
                    }
                    Ok(_) => {
                        // Gamma returned the same slug — next window hasn't
                        // been published yet. Keep polling; it'll show up.
                    }
                    Err(e) => {
                        eprintln!(
                            "  [ROTATE] find_market failed for {}: {e}",
                            cfg.display_name
                        );
                    }
                }
            }
        });
    }

    // Signal loop: poll each market every 10ms, evaluate arb, execute.
    let cancel = Arc::new(tokio::sync::Notify::new());
    let cancel_sig = cancel.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        eprintln!("\n  [SHUTDOWN] Ctrl-C received");
        cancel_sig.notify_waiters();
    });

    // Per-market arb-state (cooldown).
    let mut arb_states: Vec<ArbState> = (0..runtimes.len())
        .map(|_| ArbState { last_fire_ms: 0, bankroll: args.bankroll })
        .collect();

    let mut last_render = 0u64;
    loop {
        tokio::select! {
            _ = cancel.notified() => break,
            _ = tokio::time::sleep(Duration::from_millis(10)) => {}
        }

        let now_s_val = now_s();

        // Update each market
        for (idx, rt) in runtimes.iter().enumerate() {
            // Scope the rt lock so we release it before the order POST.
            let (decision, market, up_ask, down_ask) = {
                let mut rt = rt.lock().await;
                rt.push_binance_sample(&binance);
                rt.push_coinbase_sample(&coinbase);

                let up_snap = rt.book.snapshot(&rt.market.up_token);
                let down_snap = rt.book.snapshot(&rt.market.down_token);

                // Book age: staler of the two sides (either gate is a problem).
                let book_age_ms = {
                    let up_age = up_snap.age_ms(now_s_val);
                    let down_age = down_snap.age_ms(now_s_val);
                    match (up_age, down_age) {
                        (Some(a), Some(b)) => Some(a.max(b)),
                        (Some(a), None) | (None, Some(a)) => Some(a),
                        (None, None) => None,
                    }
                };

                let tau = (rt.market.end_time - chrono::Utc::now()).num_milliseconds() as f64
                    / 1000.0;
                let window_start_price = price.price().unwrap_or(0.0);
                let snap = Snapshot {
                    ts_ms: now_ms(),
                    time_remaining_s: tau.max(0.0),
                    best_ask_up: up_snap.best_ask,
                    best_ask_down: down_snap.best_ask,
                    window_start_price,
                };

                // Sync bankroll + cooldown from tracker
                {
                    let t = tracker.lock().await;
                    arb_states[idx].bankroll = t.bankroll;
                    arb_states[idx].last_fire_ms =
                        arb_states[idx].last_fire_ms.max(t.last_fire_ms);
                }

                let bn_ring: Vec<(i64, f64)> = rt.binance_ring.iter().copied().collect();
                let cb_ring: Vec<(i64, f64)> = rt.coinbase_ring.iter().copied().collect();
                let decision = decide_latency_arb(
                    &snap,
                    book_age_ms,
                    &bn_ring,
                    &cb_ring,
                    &arb,
                    &arb_states[idx],
                );
                rt.last_reason = decision.reason.clone();

                (decision, rt.market.clone(), up_snap.best_ask, down_snap.best_ask)
            }; // rt lock released here

            if decision.side == Side::Flat {
                let cat = categorize_flat_reason(&decision.reason);
                let mut t = tracker.lock().await;
                t.note_filter(cat);
                continue;
            }

            // Phase 1: reserve the fire (brief lock, no await held).
            let ticket = {
                let mut t = tracker.lock().await;
                let ask = match decision.side {
                    Side::BuyUp => up_ask.unwrap_or(0.0),
                    Side::BuyDown => down_ask.unwrap_or(0.0),
                    Side::Flat => 0.0,
                };
                let entry_price = price.price();
                t.prepare_fire(
                    &decision,
                    &market,
                    ask,
                    &arb,
                    entry_price,
                    market_name_for_idx(&runtimes, idx).await,
                    window_duration_for_idx(&runtimes, idx).await,
                )
            };

            let Some(ticket) = ticket else { continue };

            eprintln!(
                "  [FIRE] {:?} token={} size=${:.2} ask={:.4} reason={}",
                ticket.side,
                &ticket.token_id[..ticket.token_id.len().min(10)],
                ticket.size_usd,
                ticket.ask,
                ticket.reason
            );

            // Phase 2: fire the order WITHOUT holding any lock.
            let result: anyhow::Result<OrderResponse> = if args.dry_run {
                Ok(OrderResponse {
                    success: true,
                    order_id: "DRY".to_string(),
                    status: "MATCHED".to_string(),
                    making_amount: Some(ticket.size_usd.to_string()),
                    taking_amount: Some((ticket.size_usd / ticket.ask.max(0.0001)).to_string()),
                    error_msg: None,
                    transaction_hashes: vec![],
                })
            } else if let Some(client) = &order_client {
                // 1000 bps (10%) max-fee commitment in the signed order.
                // Polymarket charges the market's actual dynamic fee, capped
                // at this value. Declaring 0 causes 400 "invalid fee rate".
                client
                    .place_market_order(&ticket.token_id, ticket.size_usd, "BUY", 1000)
                    .await
            } else {
                Err(anyhow::anyhow!("no order client"))
            };

            // Phase 3: reconcile result into tracker state (brief lock).
            {
                let mut t = tracker.lock().await;
                t.record_fire(ticket, result);
                arb_states[idx].last_fire_ms = t.last_fire_ms;
            }
        }

        // Render ~2x/sec
        let now_ms_u = now_ms() as u64;
        if now_ms_u - last_render >= 500 {
            last_render = now_ms_u;
            let mut rows_data: Vec<_> = Vec::new();
            for rt in runtimes.iter() {
                let r = rt.lock().await;
                let up_snap = r.book.snapshot(&r.market.up_token);
                let down_snap = r.book.snapshot(&r.market.down_token);
                let delta_2s = if r.binance_ring.len() >= 2 {
                    let (cur_ts, cur_px) = *r.binance_ring.back().unwrap();
                    let cutoff = cur_ts - 2000;
                    r.binance_ring
                        .iter()
                        .find(|(ts, _)| *ts >= cutoff)
                        .map(|(_, p)| cur_px - *p)
                } else {
                    None
                };
                rows_data.push((
                    r.cfg.display_name.to_string(),
                    binance.mid(),
                    price.price(),
                    up_snap,
                    down_snap,
                    r.market.end_time,
                    r.binance_ring.len(),
                    delta_2s,
                    r.last_reason.clone(),
                ));
            }
            let rows: Vec<Row<'_>> = rows_data
                .iter()
                .map(|(name, bm, cl, up, dn, end, rlen, d2s, reason)| Row {
                    name,
                    binance_mid: *bm,
                    chainlink: *cl,
                    up_snap: up,
                    down_snap: dn,
                    window_end: *end,
                    ring_len: *rlen,
                    delta_2s: *d2s,
                    last_reason: reason,
                })
                .collect();
            let t = tracker.lock().await;
            let recent: Vec<_> = t.recent_trades.iter().cloned().collect();
            print_dashboard(
                &rows,
                t.bankroll,
                t.fills,
                t.realized_pnl,
                t.wins,
                t.losses,
                &t.open_positions,
                &recent,
            );
        }
    }

    eprintln!("  [DONE] clean shutdown");
    Ok(())
}
