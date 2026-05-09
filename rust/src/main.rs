mod book;
mod client;
mod config;
mod display;
mod feed;
mod market_api;
mod redeem;
mod redemption;
mod signal;
mod tracker;
mod types;

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::Parser;
use parking_lot::Mutex;
use tokio::sync::Notify;

use crate::client::{OrderClient, OrderResponse};
use crate::config::{resolve, ArbParams, Asset, MarketConfig};
use crate::display::{print_dashboard, Row};
use crate::feed::{BinanceFeed, BookFeed, CoinbaseFeed, PriceFeed};
use crate::market_api::{find_market, poll_resolution, Market, CHAIN_ID, CLOB_HOST};
use crate::signal::{decide_latency_arb, ArbState, Side, Snapshot};
use crate::tracker::Tracker;
use crate::types::{now_ms, now_s};

#[derive(Parser, Debug, Clone)]
#[command(name = "polybot-arb", about = "Polymarket latency-arb bot (BTC + ETH up/down)")]
struct Args {
    /// Which markets to run. `all` = BTC 15m + BTC 5m + ETH 15m + ETH 5m under one bankroll.
    #[arg(long, default_value = "btc")]
    market: String,

    #[arg(long, default_value_t = 100.0)]
    bankroll: f64,

    #[arg(long, default_value_t = false)]
    dry_run: bool,

    /// Legacy fixed threshold (used for cross-venue consensus base).
    #[arg(long, default_value_t = 25.0)]
    arb_delta_usd: f64,

    /// Adaptive threshold floor for BTC 5m — never fire below this.
    /// threshold = max(floor, 2.5 × σ_2s). Scaled for BTC's ~$75k price point.
    /// BTC 5m historical data: z2-3 bucket is profitable (+27% ROI), keep low.
    #[arg(long, default_value_t = 25.0)]
    arb_delta_floor: f64,

    /// Adaptive threshold floor for BTC 15m. Reverted 2026-04-22 from
    /// $35 back to $25 (same as 5m). The $35 raise did not improve 15m
    /// performance in live trading and removing the split lets us use a
    /// single well-understood threshold until we re-evaluate with more
    /// data. Flag retained so the per-15m override is easy to re-enable.
    #[arg(long, default_value_t = 25.0)]
    arb_delta_floor_btc_15m: f64,

    /// Adaptive threshold floor for ETH 5m — never fire below this.
    /// σ-matched to BTC's ~13σ calm-regime tail (Block Scholes 2025,
    /// SSRN 4814346).
    #[arg(long, default_value_t = 1.40)]
    arb_delta_floor_eth: f64,

    /// Adaptive threshold floor for ETH 15m. Kept equal to the 5m default
    /// until we revisit ETH; the ETH-specific filter fixes belong in a
    /// different place (time-of-day gate, z-cap) not threshold bumps.
    #[arg(long, default_value_t = 1.40)]
    arb_delta_floor_eth_15m: f64,

    /// Adaptive threshold cap (unused in current max-based formula, kept for compat).
    #[arg(long, default_value_t = 100.0)]
    arb_delta_cap: f64,

    /// Multiplier on σ_2s for the volatility-adjusted branch of the threshold:
    /// threshold = max(floor, sigma_k × σ_2s). Lower k fires more during vol
    /// (2.5 → 2.0 drops vol-regime threshold by 20%); higher k fires less.
    /// Applies to both BTC and ETH.
    #[arg(long, default_value_t = 2.5)]
    arb_sigma_k: f64,

    /// Max concurrent positions per BTC market window. Reverted to 2
    /// on 2026-04-22 from 3 — slot-3 fires had no historical baseline
    /// and allowing them appeared to change slot-2 dynamics negatively
    /// (slot 2 ROI dropped from +18% historical to -17% after the
    /// max=3 deploy, on a small but consistent sample).
    #[arg(long, default_value_t = 2)]
    max_positions_per_window_btc: usize,

    /// Max concurrent positions per ETH market window. Kept lower than
    /// BTC because ETH's edge is still being debugged; fewer parallel
    /// slots caps total exposure until calibration is proven.
    #[arg(long, default_value_t = 2)]
    max_positions_per_window_eth: usize,

    /// Per-slot ramp on the Binance-delta threshold within a window.
    /// `threshold_effective = base × (1 + ramp × slots_used)`.
    ///   - 0.0 (default): ramping disabled — all fires use same threshold.
    ///     Analysis showed slot 2 at flat threshold was already profitable
    ///     on BTC (68% WR, +21% ROI), so ramping would have filtered out
    ///     good fires without evidence of benefit.
    ///   - 0.5: slot 2 needs 1.5× base, slot 3 needs 2× base. Use this
    ///     if you later decide to require progressively stronger signals
    ///     for successive fires on the same window.
    #[arg(long, default_value_t = 0.0)]
    arb_ramp_per_slot: f64,

    #[arg(long, default_value_t = 2.0)]
    arb_window_s: f64,

    #[arg(long, default_value_t = 4.0)]
    arb_cooldown_s: f64,

    /// Minimum Polymarket book staleness for BTC fires. Below this the
    /// book is still fresh — no real arb gap.
    #[arg(long, default_value_t = 600.0)]
    arb_book_stale_ms: f64,

    /// ETH book staleness floor. Historical data showed ETH fires with
    /// <1000ms book age were LOSING; 2000-3000ms bucket was the clear
    /// winner. Raising ETH's floor to 1500ms rejects the thin-lag window
    /// that's noise on ETH but a genuine arb gap on BTC.
    #[arg(long, default_value_t = 1500.0)]
    arb_book_stale_ms_eth: f64,

    /// ETH trading hours — START of allowed UTC window (inclusive).
    /// Default 12 UTC = 8am ET. Before this hour, ETH fires were heavy
    /// losers (thin liquidity, overnight adverse selection).
    #[arg(long, default_value_t = 12)]
    eth_hours_start_utc: u8,

    /// ETH trading hours — END of allowed UTC window (exclusive).
    /// Default 18 UTC = 2pm ET = covers 12z–17z inclusive. Historical
    /// WR in this 6-hour window was ~100%.
    #[arg(long, default_value_t = 18)]
    eth_hours_end_utc: u8,

    /// ETH z-score cap. Reject fires whose Binance 2s delta exceeds
    /// this many σ. Default 10.0: below z=10 ETH behaves reasonably;
    /// above it, Binance spoofs dominate and WR drops to 36%.
    #[arg(long, default_value_t = 10.0)]
    arb_z_cap_eth: f64,

    #[arg(long, default_value_t = 0.15)]
    arb_min_ask: f64,

    #[arg(long, default_value_t = 0.85)]
    arb_max_ask: f64,

    #[arg(long, default_value_t = 30.0)]
    arb_min_tau_s: f64,

    #[arg(long, default_value_t = 10.0)]
    arb_size_usd: f64,

    /// Per-fill JSONL log path. Defaults to `live_trades_arb.jsonl` (shared
    /// across BTC and ETH — each row carries slug + asset).
    #[arg(long, default_value = "live_trades_arb.jsonl")]
    log_path: PathBuf,

    /// Persisted state path (bankroll + open positions).
    #[arg(long, default_value = "bot_state.json")]
    state_path: PathBuf,

    /// Persisted redemption queue. Survives restarts so pending winning
    /// positions don't get stranded in CTF when the bot is killed.
    #[arg(long, default_value = "redeem_queue.json")]
    redeem_queue_path: PathBuf,
}

impl Args {
    /// Build an `ArbParams` for a specific market config. Several fields are
    /// per-(asset, window-length) now:
    ///   - `delta_floor`: BTC 15m and BTC 5m use different floors ($35/$25),
    ///     ETH 15m/5m likewise.
    ///   - `max_positions_per_window`: BTC=3 (with ramp), ETH=2.
    /// Everything else is shared.
    fn arb_params_for(&self, cfg: &MarketConfig) -> ArbParams {
        let delta_floor = match (cfg.asset, cfg.window_duration_s) {
            (Asset::Btc, 900) => self.arb_delta_floor_btc_15m,
            (Asset::Btc, _)   => self.arb_delta_floor,
            (Asset::Eth, 900) => self.arb_delta_floor_eth_15m,
            (Asset::Eth, _)   => self.arb_delta_floor_eth,
        };
        let max_positions = match cfg.asset {
            Asset::Btc => self.max_positions_per_window_btc,
            Asset::Eth => self.max_positions_per_window_eth,
        };
        debug_assert!(
            delta_floor.is_finite() && delta_floor > 0.0,
            "delta_floor must be finite and positive; got {delta_floor} for {}",
            cfg.display_name,
        );
        // ETH filters (hour gate, z-cap, ETH-specific stale-ms) were
        // reverted on 2026-04-22 during the post-regression bisection.
        // All assets now share the global book_stale_ms and run without
        // hour or z-score gating — same behavior as pre-ETH-filter state.
        // CLI flags `--eth-hours-start-utc`, `--eth-hours-end-utc`,
        // `--arb-z-cap-eth`, `--arb-book-stale-ms-eth` are retained but
        // ignored; re-enable by restoring the asset-match block below.
        let _ = (
            self.arb_book_stale_ms_eth,
            self.eth_hours_start_utc,
            self.eth_hours_end_utc,
            self.arb_z_cap_eth,
        );
        let book_stale_ms = self.arb_book_stale_ms;
        let allowed_hours: Option<(u8, u8)> = None;
        let z_cap: Option<f64> = None;
        ArbParams {
            delta_usd: self.arb_delta_usd,
            window_s: self.arb_window_s,
            cooldown_s: self.arb_cooldown_s,
            book_stale_ms,
            min_ask: self.arb_min_ask,
            max_ask: self.arb_max_ask,
            min_tau_s: self.arb_min_tau_s,
            size_usd: self.arb_size_usd,
            max_bankroll_frac: 0.04,
            min_order_shares: 5.0,
            trend_window_s: 15.0,
            trend_min_ratio: 0.5,
            crossvenue_min_ratio: 0.5,
            delta_floor,
            delta_cap: self.arb_delta_cap,
            sigma_k: self.arb_sigma_k,
            max_positions_per_window: max_positions,
            ramp_per_slot: self.arb_ramp_per_slot,
            allowed_hours_utc: allowed_hours,
            z_cap,
        }
    }
}

/// Ring capacity: ~20s of Binance samples at 100 msg/s, ~60s of Coinbase
/// at 10 msg/s. Trend-check looks back 15s, so 2000/600 covers with headroom.
const BINANCE_RING_CAP: usize = 2000;
const COINBASE_RING_CAP: usize = 600;
const CHAINLINK_HISTORY_CAP: usize = 2048; // ~34 min @ 1s updates

/// One asset's feed bundle — shared across every market on the same asset
/// (BTC 15m + BTC 5m share one BinanceFeed; likewise for ETH).
struct AssetFeeds {
    binance: Arc<BinanceFeed>,
    coinbase: Arc<CoinbaseFeed>,
    chainlink: Arc<PriceFeed>,
}

impl AssetFeeds {
    fn spawn(asset: Asset, wake: Arc<Notify>) -> Self {
        let binance = Arc::new(BinanceFeed::new(
            asset.binance_symbol().to_string(),
            wake.clone(),
        ));
        let coinbase = Arc::new(CoinbaseFeed::new(
            asset.coinbase_product().to_string(),
            wake.clone(),
        ));
        let chainlink = Arc::new(PriceFeed::new(asset.chainlink_symbol().to_string()));
        Self { binance, coinbase, chainlink }
    }
}

/// A single market's runtime state: book feed, discovered market, price rings,
/// and the last flat reason (for the dashboard). Feeds live on AssetFeeds and
/// are looked up by `cfg.asset` each tick.
struct MarketRuntime {
    cfg: MarketConfig,
    market: Market,
    book: Arc<BookFeed>,
    binance_ring: VecDeque<(i64, f64)>,
    coinbase_ring: VecDeque<(i64, f64)>,
    last_binance_ts: f64,
    last_coinbase_ts: f64,
    last_reason: String,
    /// EWMA volatility state — persistent, updates every ~2s.
    /// λ=0.94 (RiskMetrics standard): half-life ~22s, matches the old
    /// prediction-market-bot's proven EWMA estimator.
    ewma_var: f64,
    ewma_last_ts: i64,
    ewma_last_price: f64,
}

const EWMA_LAMBDA: f64 = 0.94;
const EWMA_UPDATE_MS: i64 = 2000; // update every 2s (non-overlapping windows)

impl MarketRuntime {
    fn update_sigma(&mut self) {
        if self.binance_ring.is_empty() {
            return;
        }
        let (cur_ts, cur_price) = *self.binance_ring.back().unwrap();

        if self.ewma_last_ts == 0 {
            // First sample — initialize anchor, no variance estimate yet
            self.ewma_last_ts = cur_ts;
            self.ewma_last_price = cur_price;
            return;
        }

        if cur_ts - self.ewma_last_ts < EWMA_UPDATE_MS {
            return; // not enough time for a non-overlapping 2s window
        }

        let ret = cur_price - self.ewma_last_price;
        if self.ewma_var == 0.0 {
            // Bootstrap: first observation seeds the variance
            self.ewma_var = ret * ret;
        } else {
            self.ewma_var = EWMA_LAMBDA * self.ewma_var + (1.0 - EWMA_LAMBDA) * ret * ret;
        }
        self.ewma_last_ts = cur_ts;
        self.ewma_last_price = cur_price;
    }

    fn sigma_2s(&self) -> f64 {
        self.ewma_var.sqrt()
    }

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
    else if reason.starts_with("off-hours") { "off_hours" }
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
    else if reason.starts_with("z ") { "z_cap" }
    else if reason.starts_with("size ") { "size_min" }
    else if reason.is_empty() { "ok" }
    else { "other" }
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

    let configs = resolve(&args.market);
    if configs.is_empty() {
        anyhow::bail!(
            "unknown market '{}' (use btc | btc_15m | btc_5m | eth | eth_15m | eth_5m | all)",
            args.market
        );
    }

    // Build one ArbParams per market (keyed by slug_prefix — unique per
    // (asset, window-length) combination). BTC 15m and BTC 5m get different
    // thresholds; ETH 15m/5m likewise.
    let mut arb_by_market: HashMap<&'static str, Arc<ArbParams>> = HashMap::new();
    for cfg in &configs {
        arb_by_market
            .entry(cfg.slug_prefix)
            .or_insert_with(|| Arc::new(args.arb_params_for(cfg)));
    }

    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()?;

    eprintln!(
        "  [INIT] market='{}' ({} configs) bankroll=${:.2} dry_run={} window={}s stale={}-5000ms tau>={}s",
        args.market, configs.len(), args.bankroll, args.dry_run,
        args.arb_window_s, args.arb_book_stale_ms, args.arb_min_tau_s
    );
    for cfg in &configs {
        if let Some(ap) = arb_by_market.get(cfg.slug_prefix) {
            let hours = match ap.allowed_hours_utc {
                Some((s, e)) => format!(" hours=[{s},{e})"),
                None => String::new(),
            };
            let zcap = match ap.z_cap {
                Some(c) => format!(" z_cap={c:.1}"),
                None => String::new(),
            };
            eprintln!(
                "  [PARAMS] {:<8}: delta_floor=${:.2} sigma_k={:.1} size=${:.2} max_slots={} ramp={:.2} stale>={:.0}ms{}{}",
                cfg.display_name,
                ap.delta_floor,
                ap.sigma_k,
                ap.size_usd,
                ap.max_positions_per_window,
                ap.ramp_per_slot,
                ap.book_stale_ms,
                hours,
                zcap,
            );
        }
    }

    // Single wake notifier — any Binance/Coinbase update wakes the signal loop.
    let wake = Arc::new(Notify::new());

    // Build one AssetFeeds per unique asset in the config list. BTC 15m + BTC 5m
    // share BTC feeds; ETH 15m + ETH 5m share ETH feeds.
    let mut asset_feeds: HashMap<Asset, Arc<AssetFeeds>> = HashMap::new();
    for cfg in &configs {
        asset_feeds
            .entry(cfg.asset)
            .or_insert_with(|| Arc::new(AssetFeeds::spawn(cfg.asset, wake.clone())));
    }
    for (asset, _) in asset_feeds.iter() {
        eprintln!(
            "  [FEEDS] {}: Binance={} Coinbase={} Chainlink={}",
            asset.short_name(),
            asset.binance_symbol(),
            asset.coinbase_product(),
            asset.chainlink_symbol(),
        );
    }

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
            ewma_var: 0.0,
            ewma_last_ts: 0,
            ewma_last_price: 0.0,
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

    let state_path = args.state_path.clone();
    let mut tracker_inst = Tracker::new(
        args.bankroll,
        order_client.clone(),
        args.dry_run,
        args.log_path.clone(),
    );
    // Restore state from previous run if available (cron restart survival).
    if let Some(state) = Tracker::load_state(&state_path) {
        tracker_inst.apply_state(state);
    }
    let tracker = Arc::new(Mutex::new(tracker_inst));

    // Per-asset Chainlink price history. On resolution, we look up the price at
    // the position's window_end for display — each asset has its own history so
    // BTC and ETH positions resolve against the correct reference series.
    let chainlink_histories: HashMap<Asset, Arc<Mutex<VecDeque<(i64, f64)>>>> = asset_feeds
        .keys()
        .map(|a| (*a, Arc::new(Mutex::new(VecDeque::with_capacity(CHAINLINK_HISTORY_CAP)))))
        .collect();

    for (&asset, feeds) in asset_feeds.iter() {
        let history = chainlink_histories[&asset].clone();
        let price = feeds.chainlink.clone();
        tokio::spawn(async move {
            let mut last_ts: f64 = 0.0;
            loop {
                tokio::time::sleep(Duration::from_millis(200)).await;
                let ts = price.last_update_ts();
                if ts > last_ts {
                    last_ts = ts;
                    if let Some(p) = price.price() {
                        let mut h = history.lock();
                        h.push_back(((ts * 1000.0) as i64, p));
                        while h.len() > CHAINLINK_HISTORY_CAP {
                            h.pop_front();
                        }
                    }
                }
            }
        });
    }

    // Polymarket enabled native auto-redemption in April 2026 — their
    // platform now auto-settles winning positions into USDC without any
    // client-side action. The internal queue/relayer loop previously
    // lived here (see redeem.rs, still intact for emergency re-enable).
    //
    // We still load any pre-existing queue from disk so legacy pending
    // cids from before auto-redeem can be one-shot drained if Polymarket
    // happened to miss any. After one drain on startup, the queue stays
    // empty — nothing new pushes to it (see resolve watcher below).
    let redeem_queue_path = args.redeem_queue_path.clone();
    let redeem_queue: Arc<Mutex<Vec<redeem::QueuedRedemption>>> = Arc::new(Mutex::new(
        redeem::load_queue(&redeem_queue_path),
    ));
    {
        let pending = redeem_queue.lock().len();
        if pending > 0 {
            eprintln!(
                "  [REDEEM] {pending} legacy pending redemptions loaded; will drain once then go quiet (PM auto-redeem handles new wins)"
            );
            // One-shot drain loop (guarded to not loop forever).
            let http = http.clone();
            let queue = redeem_queue.clone();
            let qp = redeem_queue_path.clone();
            tokio::spawn(async move {
                redeem::run_redeem_loop(http, queue, qp).await;
            });
        }
    }

    // Resolution watcher: every 30s, walk open_positions and poll Gamma for
    // any whose window has ended (+30s buffer for UMA). Applies payouts to
    // bankroll and logs a RESOLVE row. Each position knows its asset so we
    // pick the right Chainlink history for final_price lookup.
    //
    // We no longer queue the conditionId for redemption — Polymarket's
    // native auto-redeem handles it. We still call `resolve_position` which
    // credits bankroll and writes the RESOLVE log row for PnL tracking.
    {
        let tracker = tracker.clone();
        let http = http.clone();
        let histories = chainlink_histories.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(30)).await;
                let now = chrono::Utc::now();
                let ready: Vec<(String, Asset, i64)> = {
                    let t = tracker.lock();
                    t.open_positions
                        .iter()
                        .filter(|p| now > p.window_end + chrono::Duration::seconds(30))
                        .map(|p| (p.slug.clone(), p.asset, p.window_end.timestamp_millis()))
                        .collect()
                };
                for (slug, asset, window_end_ms) in ready {
                    if let Some(up_won) = poll_resolution(&http, &slug, 6, 5.0).await {
                        let final_price = if let Some(history) = histories.get(&asset) {
                            let h = history.lock();
                            h.iter()
                                .rev()
                                .find(|(ts, _)| *ts <= window_end_ms)
                                .map(|(_, p)| *p)
                        } else {
                            None
                        };
                        {
                            let mut t = tracker.lock();
                            let _ = t.resolve_position(&slug, up_won, final_price);
                        }
                    } else {
                        eprintln!("  [RESOLVE] {slug}: not yet finalized, will retry");
                    }
                }
            }
        });
    }

    // Periodic state persistence — every 60s, save bankroll/PnL/positions.
    // Redeem queue save is gone since Polymarket auto-redeems; the queue
    // only exists for legacy-drain on startup and doesn't grow any more.
    {
        let tracker = tracker.clone();
        let sp = state_path.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(60)).await;
                let t = tracker.lock();
                t.save_state(&sp);
            }
        });
    }

    // Periodic filter-stats logger. Every 10 minutes, dump the filter_counts
    // histogram to stderr for post-hoc tuning.
    {
        let tracker = tracker.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(600)).await;
                let snapshot: Vec<(String, u64)> = {
                    let t = tracker.lock();
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
    // next window's market + a fresh BookFeed.
    for rt in &runtimes {
        let rt = rt.clone();
        let http = http.clone();
        let oc = order_client.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(20)).await;
                let (cfg, tau_s, current_slug) = {
                    let r = rt.lock();
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
                        let mut r = rt.lock();
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

    // Signal loop: wake on price updates, evaluate every market, fire async.
    //
    // Shutdown via AtomicBool. Using `Notify::notify_waiters` alone has a
    // race: if Ctrl+C arrives while the main loop is in its work phase
    // (not currently awaiting `notified()`), the signal is dropped on the
    // floor — `notify_waiters` only wakes current waiters, it doesn't
    // latch a permit. The AtomicBool + wake notify combo is bulletproof:
    // the flag is checked at the top of every iteration, and wake.notify
    // ensures we don't have to wait for the 100ms safety sleep.
    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let shutdown = shutdown.clone();
        let wake = wake.clone();
        tokio::spawn(async move {
            let _ = tokio::signal::ctrl_c().await;
            eprintln!("\n  [SHUTDOWN] Ctrl-C received");
            shutdown.store(true, Ordering::Release);
            wake.notify_waiters();
        });
    }

    // Per-market arb state (cooldown + sigma + bankroll view). Lives entirely
    // inside the main signal loop — no shared ownership needed. Cooldown is set
    // optimistically after prepare_fire; spawned order tasks never touch this.
    let mut arb_states: Vec<ArbState> = (0..runtimes.len())
        .map(|_| ArbState {
            last_fire_ms: 0,
            bankroll: args.bankroll,
            sigma_2s: 0.0,
            slots_used: 0.0,
        })
        .collect();

    let mut last_render = 0u64;
    loop {
        if shutdown.load(Ordering::Acquire) {
            break;
        }
        // Wake on: (a) any Binance/Coinbase price update, (b) 100ms safety
        // timeout so the dashboard/render still ticks and cooldowns/book_age
        // stay fresh even during quiet feed periods. Shutdown is handled by
        // the AtomicBool check above plus a wake notify from the signal
        // handler — this avoids the notify_waiters race where a Ctrl+C during
        // the loop body gets silently dropped.
        tokio::select! {
            _ = wake.notified() => {}
            _ = tokio::time::sleep(Duration::from_millis(100)) => {}
        }
        if shutdown.load(Ordering::Acquire) {
            break;
        }

        let now_s_val = now_s();

        for (idx, rt) in runtimes.iter().enumerate() {
            let (asset, slug_prefix, current_slug) = {
                let r = rt.lock();
                (r.cfg.asset, r.cfg.slug_prefix, r.market.slug.clone())
            };
            let feeds = asset_feeds.get(&asset).expect("asset feeds missing");
            let arb = arb_by_market.get(slug_prefix).expect("arb params missing").clone();

            // Count slots already used on this market's current window so the
            // ramped threshold in `decide_latency_arb` can apply.
            let slots_used: f64 = {
                let t = tracker.lock();
                t.open_positions
                    .iter()
                    .filter(|p| p.slug == current_slug)
                    .map(|p| p.slots())
                    .sum()
            };

            let (decision, market, up_ask, down_ask, up_asks_top3, down_asks_top3, market_name, window_duration_s) = {
                let mut rt_guard = rt.lock();
                rt_guard.push_binance_sample(&feeds.binance);
                rt_guard.push_coinbase_sample(&feeds.coinbase);
                rt_guard.update_sigma();

                let up_snap = rt_guard.book.snapshot(&rt_guard.market.up_token);
                let down_snap = rt_guard.book.snapshot(&rt_guard.market.down_token);

                // Book age: staler of the two sides.
                let book_age_ms = {
                    let up_age = up_snap.age_ms(now_s_val);
                    let down_age = down_snap.age_ms(now_s_val);
                    match (up_age, down_age) {
                        (Some(a), Some(b)) => Some(a.max(b)),
                        (Some(a), None) | (None, Some(a)) => Some(a),
                        (None, None) => None,
                    }
                };

                let tau = (rt_guard.market.end_time - chrono::Utc::now())
                    .num_milliseconds() as f64 / 1000.0;
                let window_start_price = feeds.chainlink.price().unwrap_or(0.0);
                let snap = Snapshot {
                    ts_ms: now_ms(),
                    time_remaining_s: tau.max(0.0),
                    best_ask_up: up_snap.best_ask,
                    best_ask_down: down_snap.best_ask,
                    window_start_price,
                };

                // Refresh arb state: own per-market cooldown + sigma; bankroll
                // is the tracker's current view (shared across all markets).
                arb_states[idx].sigma_2s = rt_guard.sigma_2s();
                arb_states[idx].bankroll = tracker.lock().bankroll;
                arb_states[idx].slots_used = slots_used;

                let bn_ring: Vec<(i64, f64)> =
                    rt_guard.binance_ring.iter().copied().collect();
                let cb_ring: Vec<(i64, f64)> =
                    rt_guard.coinbase_ring.iter().copied().collect();
                let decision = decide_latency_arb(
                    &snap,
                    book_age_ms,
                    &bn_ring,
                    &cb_ring,
                    &arb,
                    &arb_states[idx],
                );
                rt_guard.last_reason = decision.reason.clone();

                // Capture top-3 ask levels on each side for capacity analysis
                // (logged with [FIRE]). Used later by analyze_wl.py to check
                // whether trade sizes are eating into L2/L3 on the book.
                let up_asks_top3: Vec<(f64, f64)> =
                    up_snap.asks.iter().take(3).copied().collect();
                let down_asks_top3: Vec<(f64, f64)> =
                    down_snap.asks.iter().take(3).copied().collect();

                (
                    decision,
                    rt_guard.market.clone(),
                    up_snap.best_ask,
                    down_snap.best_ask,
                    up_asks_top3,
                    down_asks_top3,
                    rt_guard.cfg.display_name.to_string(),
                    rt_guard.cfg.window_duration_s,
                )
            };

            if decision.side == Side::Flat {
                let cat = categorize_flat_reason(&decision.reason);
                let mut t = tracker.lock();
                t.note_filter(cat);
                continue;
            }

            // Phase 1: reserve the fire (brief tracker lock, no await held).
            let ticket = {
                let mut t = tracker.lock();
                let ask = match decision.side {
                    Side::BuyUp => up_ask.unwrap_or(0.0),
                    Side::BuyDown => down_ask.unwrap_or(0.0),
                    Side::Flat => 0.0,
                };
                let entry_price = feeds.chainlink.price();
                t.prepare_fire(
                    &decision,
                    &market,
                    ask,
                    &arb,
                    entry_price,
                    market_name,
                    window_duration_s,
                    asset,
                )
            };

            let Some(ticket) = ticket else { continue };

            // Per-market cooldown reservation: set NOW, before the async order
            // starts, so the next tick sees the cooldown and we don't double-fire
            // while this order is in flight.
            arb_states[idx].last_fire_ms = now_ms();

            // Log top-3 book levels on the side we're taking. Shape:
            //   [(price, size), ...]
            // `size` is shares, so dollar-denominated-available = sum(price*size).
            // Useful for downstream capacity / slippage analysis — if our
            // `size_usd` consistently exceeds price*size of L1, we're eating
            // into deeper levels and real edge is lower than the posted ask
            // suggests.
            let hit_asks = match ticket.side {
                Side::BuyUp => &up_asks_top3,
                Side::BuyDown => &down_asks_top3,
                Side::Flat => &up_asks_top3,
            };
            let depth_str: String = hit_asks
                .iter()
                .map(|(p, s)| format!("({p:.3}×{s:.1})"))
                .collect::<Vec<_>>()
                .join(",");

            eprintln!(
                "  [FIRE] {} {:?} token={} size=${:.2} ask={:.4} book=[{}] reason={}",
                ticket.market_name,
                ticket.side,
                &ticket.token_id[..ticket.token_id.len().min(10)],
                ticket.size_usd,
                ticket.ask,
                depth_str,
                ticket.reason
            );

            // Phase 2+3: fire the order async and reconcile in a spawned task.
            // The main signal loop continues immediately — we don't block 50-100ms
            // on the HTTP round-trip. Each market's cooldown already prevents
            // same-market double-fire; bankroll is re-checked in prepare_fire on
            // the next iteration.
            let tracker_ref = tracker.clone();
            let client_ref = order_client.clone();
            let dry_run = args.dry_run;
            tokio::spawn(async move {
                let result: anyhow::Result<OrderResponse> = if dry_run {
                    Ok(OrderResponse {
                        success: true,
                        order_id: "DRY".to_string(),
                        status: "MATCHED".to_string(),
                        making_amount: Some(ticket.size_usd.to_string()),
                        taking_amount: Some(
                            (ticket.size_usd / ticket.ask.max(0.0001)).to_string(),
                        ),
                        error_msg: None,
                        transaction_hashes: vec![],
                    })
                } else if let Some(client) = client_ref {
                    // 1000 bps (10%) max-fee commitment in the signed order.
                    // Polymarket charges the market's actual dynamic fee, capped
                    // at this value. Declaring 0 causes 400 "invalid fee rate".
                    client
                        .place_market_order(&ticket.token_id, ticket.size_usd, "BUY", 1000)
                        .await
                } else {
                    Err(anyhow::anyhow!("no order client"))
                };

                let mut t = tracker_ref.lock();
                t.record_fire(ticket, result);
            });
        }

        // Render ~2x/sec
        let now_ms_u = now_ms() as u64;
        if now_ms_u - last_render >= 500 {
            last_render = now_ms_u;
            let mut rows_data: Vec<_> = Vec::new();
            for rt in runtimes.iter() {
                let r = rt.lock();
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
                let feeds = &asset_feeds[&r.cfg.asset];
                rows_data.push((
                    r.cfg.display_name.to_string(),
                    feeds.binance.mid(),
                    feeds.chainlink.price(),
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
            let t = tracker.lock();
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

    // Save state + redemption queue on clean shutdown so cron restarts
    // preserve PnL/bankroll AND don't strand pending winning positions.
    {
        let t = tracker.lock();
        t.save_state(&state_path);
    }
    eprintln!(
        "  [DONE] clean shutdown, state saved to {}",
        state_path.display(),
    );
    Ok(())
}
