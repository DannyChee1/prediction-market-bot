
use std::net::ToSocketAddrs;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use tokio::net::TcpStream;
use tokio::time::timeout;
use tokio_tungstenite::{connect_async, tungstenite::Message};

struct Target {
    name: &'static str,
    host: &'static str,
    port: u16,
    ws_url: &'static str,
}

const TARGETS: &[Target] = &[
    Target {
        name: "stream.binance.com:9443     ",
        host: "stream.binance.com",
        port: 9443,
        ws_url: "wss://stream.binance.com:9443/ws/btcusdt@bookTicker",
    },
    Target {
        name: "data-stream.binance.vision  ",
        host: "data-stream.binance.vision",
        port: 443,
        ws_url: "wss://data-stream.binance.vision/ws/btcusdt@bookTicker",
    },
    Target {
        name: "fstream.binance.com         ",
        host: "fstream.binance.com",
        port: 443,
        ws_url: "wss://fstream.binance.com/ws/btcusdt@bookTicker",
    },
];

fn resolve(host: &str, port: u16) -> Vec<std::net::IpAddr> {
    format!("{host}:{port}")
        .to_socket_addrs()
        .map(|it| it.map(|sa| sa.ip()).collect())
        .unwrap_or_default()
}

fn guess_region(ip: &std::net::IpAddr) -> &'static str {
    let s = ip.to_string();
    // These are heuristic, based on AWS public IP ranges.
    if s.starts_with("18.181.") || s.starts_with("13.230.") || s.starts_with("54.65.")
        || s.starts_with("52.68.") || s.starts_with("52.196.") || s.starts_with("13.115.")
    {
        "ap-northeast-1 Tokyo"
    } else if s.starts_with("3.248.") || s.starts_with("34.240.") || s.starts_with("52.208.")
        || s.starts_with("54.72.") || s.starts_with("52.16.") || s.starts_with("18.200.")
    {
        "eu-west-1 Ireland"
    } else if s.starts_with("54.") || s.starts_with("3.") || s.starts_with("18.")
        || s.starts_with("52.") || s.starts_with("34.")
    {
        "AWS (unclassified)"
    } else {
        "non-AWS / CDN"
    }
}

async fn tcp_connect(host: &str, port: u16) -> Option<f64> {
    let t0 = Instant::now();
    match timeout(Duration::from_secs(5), TcpStream::connect(format!("{host}:{port}"))).await {
        Ok(Ok(_)) => Some(t0.elapsed().as_secs_f64() * 1000.0),
        Ok(Err(e)) => {
            eprintln!("      tcp error: {e}");
            None
        }
        Err(_) => {
            eprintln!("      tcp timeout after 5s");
            None
        }
    }
}

async fn probe_target(t: &Target) {
    println!("\n=== {} ===", t.name.trim());

    // DNS
    let ips = resolve(t.host, t.port);
    if ips.is_empty() {
        println!("  DNS: RESOLVE FAILED");
        return;
    }
    println!("  DNS: {}", ips.iter().take(4).map(|i| format!("{i} ({})", guess_region(i)))
        .collect::<Vec<_>>().join(", "));

    // TCP RTT x 10
    let mut tcps = Vec::new();
    for _ in 0..10 {
        if let Some(ms) = tcp_connect(t.host, t.port).await {
            tcps.push(ms);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    if tcps.is_empty() {
        println!("  TCP: BLOCKED / UNREACHABLE");
        return;
    }
    let (min, p50, max) = stats(&tcps);
    println!("  TCP  n={} min={:.1} p50={:.1} max={:.1} ms", tcps.len(), min, p50, max);

    // WS open + read 20 messages
    let t_ws = Instant::now();
    let conn = timeout(Duration::from_secs(10), connect_async(t.ws_url)).await;
    let ws_open_ms;
    let ws = match conn {
        Ok(Ok((ws, resp))) => {
            ws_open_ms = t_ws.elapsed().as_secs_f64() * 1000.0;
            println!("  WS:  opened in {ws_open_ms:.1} ms  (HTTP {})", resp.status());
            ws
        }
        Ok(Err(e)) => {
            let msg = format!("{e}");
            println!("  WS:  OPEN FAILED — {msg}");
            if msg.contains("451") || msg.contains("403") || msg.contains("Unavailable For Legal") {
                println!("       ^ geofenced / compliance-blocked from this IP");
            }
            return;
        }
        Err(_) => {
            println!("  WS:  OPEN TIMEOUT (10s) — likely filtered at the network layer");
            return;
        }
    };

    let (_, mut read) = ws.split();
    let start = Instant::now();
    let mut intervals = Vec::new();
    let mut last_rx: Option<Instant> = None;
    let mut first_msg_ms: Option<f64> = None;
    let mut n = 0usize;
    let collect = timeout(Duration::from_secs(10), async {
        while n < 20 {
            match read.next().await {
                Some(Ok(Message::Text(_))) => {
                    let now = Instant::now();
                    if first_msg_ms.is_none() {
                        first_msg_ms = Some((now - start).as_secs_f64() * 1000.0);
                    }
                    if let Some(prev) = last_rx {
                        intervals.push((now - prev).as_secs_f64() * 1000.0);
                    }
                    last_rx = Some(now);
                    n += 1;
                }
                Some(Ok(Message::Ping(_))) | Some(Ok(Message::Pong(_))) => {}
                Some(Ok(_)) => {}
                Some(Err(e)) => {
                    println!("  WS:  read error after {n} msgs: {e}");
                    return;
                }
                None => {
                    println!("  WS:  stream closed after {n} msgs");
                    return;
                }
            }
        }
    }).await;

    if collect.is_err() {
        println!("  WS:  timed out after {n} msgs in 10s (stream idle — may be geofenced read)");
    }

    if let Some(f) = first_msg_ms {
        println!("  WS:  first message at {f:.0} ms from connect");
    }
    if !intervals.is_empty() {
        let (imin, ip50, imax) = stats(&intervals);
        println!("  WS:  {} msgs  inter-arrival  min={:.1} p50={:.1} max={:.1} ms",
            intervals.len() + 1, imin, ip50, imax);
    }
    if n > 0 {
        println!("  -> READ-ONLY ACCESS: OK");
    } else {
        println!("  -> READ-ONLY ACCESS: NO DATA (connected but silent — unusual)");
    }
}

fn stats(samples: &[f64]) -> (f64, f64, f64) {
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = s[s.len() / 2];
    (*s.first().unwrap(), p50, *s.last().unwrap())
}

#[tokio::main]
async fn main() {
    let hostname = std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    println!("binance_probe  host='{hostname}'  {}",
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"));

    for t in TARGETS {
        probe_target(t).await;
    }

    println!("\nLegend:");
    println!("  * TCP p50  = pure geographic RTT");
    println!("  * WS open  = TCP + TLS + HTTP upgrade (~2x TCP RTT expected)");
    println!("  * inter-arrival = how fast bookTicker updates flow once connected");
    println!("  * If any endpoint shows 'OPEN FAILED 451' -> IP-geofenced, skip it");
}
