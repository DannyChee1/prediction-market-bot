//! VPS location probe.
//!
//! Measures round-trip latency to every endpoint the arb bot touches.
//! Run the exact same binary from candidate VPS locations and compare.
//!
//! Usage:
//!   cargo run --release --bin loc_probe
//!   # copy target/release/loc_probe to each VPS and run there.
//!
//! Reports for each endpoint:
//!   * DNS-resolved IPs (so you can see which AWS region you landed in)
//!   * TCP connect time  (n=20) — pure network RTT + TCP handshake
//!   * TLS + WS upgrade  (n=10) — full "time to usable connection"
//!   * First data message (WS only) — measures subscribe-to-data turnaround
//!
//! All times in milliseconds. Lower p50 = closer. Low p99/max spread = low jitter.

use std::net::ToSocketAddrs;
use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpStream;
use tokio::time::timeout;
use tokio_tungstenite::{connect_async, tungstenite::Message};

struct Target {
    name: &'static str,
    host: &'static str,
    port: u16,
    /// None = TCP-only (HTTPS endpoint we POST orders to).
    /// Some((url, subscribe_json)) = WS; probe also measures first data message.
    ws: Option<(&'static str, Option<&'static str>)>,
}

const CLOB_SUB: &str = r#"{"assets_ids":["1"],"type":"market","custom_feature_enabled":true}"#;

const RTDS_SUB: &str =
    r#"{"action":"subscribe","subscriptions":[{"topic":"crypto_prices_chainlink","type":"*"}]}"#;

const TARGETS: &[Target] = &[
    Target {
        name: "binance-ws    ",
        host: "data-stream.binance.vision",
        port: 443,
        ws: Some((
            "wss://data-stream.binance.vision/ws/btcusdt@bookTicker",
            None, // URL subscribes automatically
        )),
    },
    Target {
        name: "polymkt-clob  ",
        host: "ws-subscriptions-clob.polymarket.com",
        port: 443,
        ws: Some((
            "wss://ws-subscriptions-clob.polymarket.com/ws/market",
            Some(CLOB_SUB),
        )),
    },
    Target {
        name: "polymkt-rtds  ",
        host: "ws-live-data.polymarket.com",
        port: 443,
        ws: Some(("wss://ws-live-data.polymarket.com", Some(RTDS_SUB))),
    },
    Target {
        name: "clob-http     ",
        host: "clob.polymarket.com",
        port: 443,
        ws: None, // this is the order-POST endpoint
    },
    Target {
        name: "gamma-http    ",
        host: "gamma-api.polymarket.com",
        port: 443,
        ws: None,
    },
];

async fn tcp_sample(host: &str, port: u16) -> Option<f64> {
    let addr = format!("{host}:{port}");
    let t0 = Instant::now();
    let conn = timeout(Duration::from_secs(5), TcpStream::connect(&addr)).await;
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    match conn {
        Ok(Ok(_)) => Some(ms),
        _ => None,
    }
}

/// Returns (ws_open_ms, first_msg_ms_from_start).
async fn ws_sample(url: &str, sub: Option<&str>) -> Option<(f64, Option<f64>)> {
    let t0 = Instant::now();
    let (ws, _) = timeout(Duration::from_secs(10), connect_async(url))
        .await
        .ok()?
        .ok()?;
    let open_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let (mut write, mut read) = ws.split();

    if let Some(s) = sub {
        let _ = write.send(Message::Text(s.to_string())).await;
    }

    // Wait up to 5s for first data frame we'd actually use.
    let first_ms = timeout(Duration::from_secs(5), async {
        loop {
            match read.next().await? {
                Ok(Message::Text(t)) if !t.is_empty() && t != "PONG" => {
                    return Some(t0.elapsed().as_secs_f64() * 1000.0);
                }
                Ok(Message::Binary(b)) if !b.is_empty() => {
                    return Some(t0.elapsed().as_secs_f64() * 1000.0);
                }
                Ok(_) => continue,
                Err(_) => return None,
            }
        }
    })
    .await
    .ok()
    .flatten();

    Some((open_ms, first_ms))
}

fn pct(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = (((sorted.len() as f64 - 1.0) * q).round() as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn summary(label: &str, samples: &[f64]) {
    if samples.is_empty() {
        println!("  {label}: ALL SAMPLES FAILED");
        return;
    }
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = s.first().copied().unwrap();
    let max = s.last().copied().unwrap();
    let avg: f64 = s.iter().sum::<f64>() / s.len() as f64;
    println!(
        "  {label} n={:>2}  min={:>6.1}  avg={:>6.1}  p50={:>6.1}  p95={:>6.1}  p99={:>6.1}  max={:>6.1}  (ms)",
        s.len(),
        min,
        avg,
        pct(&s, 0.50),
        pct(&s, 0.95),
        pct(&s, 0.99),
        max,
    );
}

fn resolve(host: &str, port: u16) -> Vec<std::net::IpAddr> {
    format!("{host}:{port}")
        .to_socket_addrs()
        .map(|it| it.map(|sa| sa.ip()).collect())
        .unwrap_or_default()
}

#[tokio::main]
async fn main() {
    let hostname = std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    println!("=== loc_probe  host='{hostname}'  {}  ===",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"));

    // DNS first (tells you which region your VPS resolved to — AWS publishes IP ranges)
    println!("\n── DNS ─────────────────────────────────────────────────────");
    for t in TARGETS {
        let ips = resolve(t.host, t.port);
        let ip_list = if ips.is_empty() {
            "RESOLVE FAILED".to_string()
        } else {
            ips.iter()
                .take(4)
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        println!("  {} {} -> {ip_list}", t.name, t.host);
    }

    println!("\n── TCP connect (n=20, raw network + SYN/ACK) ──────────────");
    for t in TARGETS {
        let mut samples = Vec::new();
        for _ in 0..20 {
            if let Some(ms) = tcp_sample(t.host, t.port).await {
                samples.push(ms);
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        summary(t.name, &samples);
    }

    println!("\n── WS handshake (n=10, TCP + TLS + HTTP upgrade) ──────────");
    let mut first_msg_results: Vec<(&'static str, Vec<f64>)> = Vec::new();
    for t in TARGETS {
        let Some((url, sub)) = t.ws else { continue };
        let mut opens = Vec::new();
        let mut firsts = Vec::new();
        for _ in 0..10 {
            if let Some((open_ms, first_ms)) = ws_sample(url, sub).await {
                opens.push(open_ms);
                if let Some(f) = first_ms {
                    firsts.push(f);
                }
            }
            tokio::time::sleep(Duration::from_millis(300)).await;
        }
        summary(t.name, &opens);
        first_msg_results.push((t.name, firsts));
    }

    println!("\n── WS subscribe -> first data message ─────────────────────");
    for (name, samples) in &first_msg_results {
        summary(name, samples);
    }

    println!("\nTips:");
    println!("  * p50 ~ geographic distance. p99 - p50 ~ jitter.");
    println!("  * Polymarket (CLOB + RTDS) runs on AWS us-east-1 — expect ~1-5ms there, ~70-80ms Dublin.");
    println!("  * Binance WS has global edges; numbers should be similar in either VPS region.");
    println!("  * Arb edge lives on the STALE SIDE: low Polymarket latency + decent Binance = winning combo.");
}
