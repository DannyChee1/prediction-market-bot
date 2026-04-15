//! Cross-feed price comparison: Binance vs Coinbase vs Chainlink.
//!
//! Runs all three WS feeds in parallel for a fixed duration, records every
//! mid-price tick with local receive timestamp, then reports:
//!
//!   1. Per-feed cadence (msg count, inter-arrival p50/p95/p99).
//!   2. "True value" picture: pairwise price bias (mean), noise (std),
//!      and tail divergence (p99) at common timestamps.
//!   3. Lead/lag analysis: when Binance crosses a new price level, how
//!      long until Coinbase and Chainlink reach the same level? This is
//!      the actual "edge window" you'd lose by switching signal source.
//!   4. Rolling % moves over 10s / 60s windows per feed — shows which
//!      feed is noisiest vs quiet in a normal market.
//!
//! Usage:
//!   cargo run --release --bin feed_compare -- --duration 180

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use tokio::time::sleep;
use tokio_tungstenite::{connect_async, tungstenite::Message};

#[derive(Clone, Copy, Debug)]
struct Sample {
    recv_ms: i64,  // local wall clock, ms since epoch
    price: f64,
}

struct FeedBuf {
    samples: Mutex<Vec<Sample>>,
}

impl FeedBuf {
    fn new() -> Self {
        Self { samples: Mutex::new(Vec::with_capacity(100_000)) }
    }
    fn push(&self, s: Sample) {
        self.samples.lock().unwrap().push(s);
    }
    fn snapshot(&self) -> Vec<Sample> {
        self.samples.lock().unwrap().clone()
    }
}

fn now_ms() -> i64 {
    (std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64() * 1000.0) as i64
}

// ── Feed tasks ──────────────────────────────────────────────────────────────

async fn binance_task(buf: Arc<FeedBuf>) {
    let url = "wss://data-stream.binance.vision/ws/btcusdt@bookTicker";
    loop {
        let Ok((ws, _)) = connect_async(url).await else {
            sleep(Duration::from_secs(2)).await;
            continue;
        };
        let (_, mut read) = ws.split();
        while let Some(Ok(Message::Text(t))) = read.next().await {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&t) {
                let bid = v.get("b").and_then(|x| x.as_str()).and_then(|s| s.parse::<f64>().ok());
                let ask = v.get("a").and_then(|x| x.as_str()).and_then(|s| s.parse::<f64>().ok());
                if let (Some(b), Some(a)) = (bid, ask) {
                    buf.push(Sample { recv_ms: now_ms(), price: (b + a) / 2.0 });
                }
            }
        }
    }
}

async fn coinbase_task(buf: Arc<FeedBuf>) {
    let url = "wss://ws-feed.exchange.coinbase.com";
    let sub = r#"{"type":"subscribe","product_ids":["BTC-USD"],"channels":["ticker"]}"#;
    loop {
        let Ok((ws, _)) = connect_async(url).await else {
            sleep(Duration::from_secs(2)).await;
            continue;
        };
        let (mut write, mut read) = ws.split();
        let _ = write.send(Message::Text(sub.to_string())).await;
        while let Some(Ok(Message::Text(t))) = read.next().await {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&t) {
                if v.get("type").and_then(|x| x.as_str()) != Some("ticker") {
                    continue;
                }
                let bid = v.get("best_bid").and_then(|x| x.as_str()).and_then(|s| s.parse::<f64>().ok());
                let ask = v.get("best_ask").and_then(|x| x.as_str()).and_then(|s| s.parse::<f64>().ok());
                if let (Some(b), Some(a)) = (bid, ask) {
                    buf.push(Sample { recv_ms: now_ms(), price: (b + a) / 2.0 });
                }
            }
        }
    }
}

async fn chainlink_task(buf: Arc<FeedBuf>) {
    let url = "wss://ws-live-data.polymarket.com";
    let sub = r#"{"action":"subscribe","subscriptions":[{"topic":"crypto_prices_chainlink","type":"*"}]}"#;
    loop {
        let Ok((ws, _)) = connect_async(url).await else {
            sleep(Duration::from_secs(2)).await;
            continue;
        };
        let (mut write, mut read) = ws.split();
        let _ = write.send(Message::Text(sub.to_string())).await;
        // Heartbeat PING every 5s
        let hb = tokio::spawn(async move {
            loop {
                sleep(Duration::from_secs(5)).await;
                if write.send(Message::Text("PING".to_string())).await.is_err() {
                    break;
                }
            }
        });
        while let Some(Ok(msg)) = read.next().await {
            let Message::Text(t) = msg else { continue };
            if t == "PONG" || t.is_empty() {
                continue;
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(&t) else { continue };
            let payload = v.get("payload").cloned().unwrap_or(v.clone());
            if payload.get("symbol").and_then(|x| x.as_str()) != Some("btc/usd") {
                continue;
            }
            let price = payload
                .get("data").and_then(|d| d.as_array()).and_then(|a| a.last())
                .and_then(|e| e.get("value"))
                .and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
                .or_else(|| payload.get("value").and_then(|v| {
                    v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok()))
                }));
            if let Some(p) = price {
                buf.push(Sample { recv_ms: now_ms(), price: p });
            }
        }
        hb.abort();
    }
}

// ── Analysis ────────────────────────────────────────────────────────────────

fn pct(v: &[f64], q: f64) -> f64 {
    if v.is_empty() { return f64::NAN; }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((s.len() as f64 - 1.0) * q).round() as usize;
    s[idx.min(s.len() - 1)]
}

fn report_cadence(name: &str, samples: &[Sample]) {
    if samples.len() < 2 {
        println!("  {name:<10} n={} (too few)", samples.len());
        return;
    }
    let gaps: Vec<f64> = samples
        .windows(2)
        .map(|w| (w[1].recv_ms - w[0].recv_ms) as f64)
        .collect();
    let duration_s = (samples.last().unwrap().recv_ms - samples.first().unwrap().recv_ms) as f64 / 1000.0;
    let rate = samples.len() as f64 / duration_s.max(0.001);
    println!(
        "  {name:<10} n={:>5}  {:>5.1} msg/s  gap p50={:>5.1}ms p95={:>6.1}ms p99={:>6.1}ms",
        samples.len(), rate,
        pct(&gaps, 0.50), pct(&gaps, 0.95), pct(&gaps, 0.99),
    );
}

/// For each tick in `ref_`, find the nearest prior sample in `other`; report
/// mean/std/p99 of (ref_price - other_price).
fn pairwise_bias(ref_name: &str, ref_: &[Sample], other_name: &str, other: &[Sample]) {
    if other.is_empty() || ref_.is_empty() { return; }
    let mut diffs = Vec::new();
    let mut j = 0usize;
    for r in ref_ {
        while j + 1 < other.len() && other[j + 1].recv_ms <= r.recv_ms {
            j += 1;
        }
        if other[j].recv_ms <= r.recv_ms {
            diffs.push(r.price - other[j].price);
        }
    }
    if diffs.is_empty() { return; }
    let mean: f64 = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let var: f64 = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / diffs.len() as f64;
    let std = var.sqrt();
    let abs: Vec<f64> = diffs.iter().map(|d| d.abs()).collect();
    println!(
        "  {ref_name} - {other_name:<9}  mean={:+7.2}  std={:6.2}  |diff| p50={:5.2} p95={:6.2} p99={:6.2} max={:7.2}",
        mean, std,
        pct(&abs, 0.50), pct(&abs, 0.95), pct(&abs, 0.99),
        abs.iter().cloned().fold(0.0_f64, f64::max),
    );
}

/// For each large move in `leader` (>= move_thresh dollars within `window_ms`),
/// measure how long until `follower` reaches the new price level.
fn lead_lag(
    leader_name: &str,
    leader: &[Sample],
    follower_name: &str,
    follower: &[Sample],
    move_thresh: f64,
    window_ms: i64,
) {
    if leader.len() < 2 || follower.is_empty() {
        println!("  {leader_name} -> {follower_name}: insufficient data");
        return;
    }
    let mut lags = Vec::new();
    let mut i = 0usize;
    while i < leader.len() {
        // find latest sample <= leader[i].recv_ms - window_ms
        let t_now = leader[i].recv_ms;
        let t_old = t_now - window_ms;
        let mut j = i;
        while j > 0 && leader[j - 1].recv_ms >= t_old {
            j -= 1;
        }
        let delta = leader[i].price - leader[j].price;
        if delta.abs() >= move_thresh {
            let target = leader[i].price;
            // find first follower sample AFTER t_now with price crossing target
            let sign = delta.signum();
            let mut k = follower.iter().position(|s| s.recv_ms > t_now);
            let mut found: Option<i64> = None;
            if let Some(mut idx) = k.take() {
                while idx < follower.len() {
                    let p = follower[idx].price;
                    if (sign > 0.0 && p >= target) || (sign < 0.0 && p <= target) {
                        found = Some(follower[idx].recv_ms - t_now);
                        break;
                    }
                    idx += 1;
                }
            }
            if let Some(lag) = found {
                lags.push(lag as f64);
            }
            // advance past this event so we don't double-count the same move
            i += 5;
        } else {
            i += 1;
        }
    }
    if lags.is_empty() {
        println!(
            "  {leader_name:<8} -> {follower_name:<8}  no >=${move_thresh:.0}/{window_ms}ms moves in sample"
        );
        return;
    }
    println!(
        "  {leader_name:<8} -> {follower_name:<8}  n={:>3} events  lag p50={:>6.0}ms p95={:>7.0}ms p99={:>7.0}ms max={:>7.0}ms",
        lags.len(),
        pct(&lags, 0.50), pct(&lags, 0.95), pct(&lags, 0.99),
        lags.iter().cloned().fold(0.0_f64, f64::max),
    );
}

/// Rolling max |% change| over `window_s` — characterises feed "noise".
fn rolling_moves(name: &str, samples: &[Sample], window_s: i64) {
    if samples.len() < 2 { return; }
    let window_ms = window_s * 1000;
    let mut ring: VecDeque<Sample> = VecDeque::new();
    let mut moves = Vec::new();
    for s in samples {
        ring.push_back(*s);
        while let Some(front) = ring.front() {
            if s.recv_ms - front.recv_ms > window_ms {
                ring.pop_front();
            } else {
                break;
            }
        }
        if ring.len() >= 2 {
            let (lo, hi) = ring.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| {
                (lo.min(x.price), hi.max(x.price))
            });
            if lo > 0.0 {
                moves.push((hi - lo) / lo * 10_000.0); // bps
            }
        }
    }
    if moves.is_empty() { return; }
    println!(
        "  {name:<10} {}s window  range p50={:>5.1}bp p95={:>6.1}bp p99={:>6.1}bp max={:>7.1}bp",
        window_s,
        pct(&moves, 0.50), pct(&moves, 0.95), pct(&moves, 0.99),
        moves.iter().cloned().fold(0.0_f64, f64::max),
    );
}

// ── Main ────────────────────────────────────────────────────────────────────

#[derive(Debug)]
struct Args {
    duration_s: u64,
}

fn parse_args() -> Args {
    let mut duration_s = 120u64;
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--duration" | "-d" => {
                if let Some(v) = args.get(i + 1).and_then(|s| s.parse().ok()) {
                    duration_s = v;
                }
                i += 2;
            }
            _ => i += 1,
        }
    }
    Args { duration_s }
}

#[tokio::main]
async fn main() {
    let args = parse_args();
    println!(
        "feed_compare  duration={}s  {}",
        args.duration_s,
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
    );

    let binance = Arc::new(FeedBuf::new());
    let coinbase = Arc::new(FeedBuf::new());
    let chainlink = Arc::new(FeedBuf::new());

    let b = binance.clone();
    let c = coinbase.clone();
    let ch = chainlink.clone();
    let h1 = tokio::spawn(binance_task(b));
    let h2 = tokio::spawn(coinbase_task(c));
    let h3 = tokio::spawn(chainlink_task(ch));

    // Live progress
    let start = Instant::now();
    let progress_binance = binance.clone();
    let progress_coinbase = coinbase.clone();
    let progress_chainlink = chainlink.clone();
    let progress = tokio::spawn(async move {
        loop {
            sleep(Duration::from_secs(10)).await;
            let b = progress_binance.samples.lock().unwrap().len();
            let c = progress_coinbase.samples.lock().unwrap().len();
            let cl = progress_chainlink.samples.lock().unwrap().len();
            let t = start.elapsed().as_secs();
            eprintln!("  [{t}s] binance={b} coinbase={c} chainlink={cl}");
        }
    });

    sleep(Duration::from_secs(args.duration_s)).await;
    h1.abort();
    h2.abort();
    h3.abort();
    progress.abort();

    let b_s = binance.snapshot();
    let c_s = coinbase.snapshot();
    let cl_s = chainlink.snapshot();

    println!("\n── Cadence ────────────────────────────────────────────────");
    report_cadence("binance", &b_s);
    report_cadence("coinbase", &c_s);
    report_cadence("chainlink", &cl_s);

    println!("\n── Pairwise price bias ($) ────────────────────────────────");
    pairwise_bias("binance  ", &b_s, "coinbase ", &c_s);
    pairwise_bias("binance  ", &b_s, "chainlink", &cl_s);
    pairwise_bias("coinbase ", &c_s, "chainlink", &cl_s);

    println!("\n── Lead/lag on >= $5 / 2s moves ──────────────────────────");
    lead_lag("binance", &b_s, "coinbase", &c_s, 5.0, 2000);
    lead_lag("binance", &b_s, "chainln ", &cl_s, 5.0, 2000);
    lead_lag("coinbase", &c_s, "binance", &b_s, 5.0, 2000);
    lead_lag("coinbase", &c_s, "chainln ", &cl_s, 5.0, 2000);

    println!("\n── Lead/lag on >= $2 / 2s moves ──────────────────────────");
    lead_lag("binance", &b_s, "coinbase", &c_s, 2.0, 2000);
    lead_lag("coinbase", &c_s, "binance", &b_s, 2.0, 2000);

    println!("\n── Rolling price range (bps) ──────────────────────────────");
    rolling_moves("binance", &b_s, 10);
    rolling_moves("coinbase", &c_s, 10);
    rolling_moves("binance", &b_s, 60);
    rolling_moves("coinbase", &c_s, 60);

    println!("\nInterpretation:");
    println!("  * cadence: Binance sends every book-tick (~200ms typical), Coinbase every ticker event (usually slower), Chainlink every oracle round (~30s).");
    println!("  * bias mean: persistent $ offset (Coinbase usually 1-5 above/below Binance from fees/funding).");
    println!("  * |diff| p99: tail of price divergence — if huge, the two exchanges occasionally diverge and your signal would be misleading.");
    println!("  * lead/lag: 'binance -> coinbase' = if you watched Binance, how long till Coinbase confirms. If <50ms, Coinbase is a valid signal. >200ms: Coinbase is a laggard; bad substitute.");
    println!("  * rolling range: 10s p99 in bps is your typical arb opportunity window size.");
}
