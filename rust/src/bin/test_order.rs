//! Tests whether a limit BUY at (best_bid + 1 tick) — which equals best_ask when
//! the spread is 1 tick — is treated as taker or maker by Polymarket.
//!
//! Finds a btc-updown-5m market with a favorite side (ask $0.65–$0.85) with a
//! tight spread, places 5 shares at `bid + tick_size`, and reports the fill:
//!   - taking_amount > 0, status=MATCHED → crossed the book = TAKER (paid fee)
//!   - taking_amount = 0, status=LIVE     → rested on book = MAKER (no fee)
//!
//! Uses a persistent WebSocket subscription (same as the main bot) for real
//! book data. Polls every 100ms for criteria match, fires immediately.

use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Duration as ChDuration, Timelike, Utc};
use futures_util::{SinkExt, StreamExt};
use polyfill_rs::orders::SigType;
use polyfill_rs::types::{
    ApiCreds, ExtraOrderArgs, OrderOptions, OrderType as PolyOrderType, Side,
};
use polyfill_rs::{ClobClient, OrderArgs};
use rust_decimal::prelude::*;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio_tungstenite::{connect_async, tungstenite::Message};

const GAMMA: &str = "https://gamma-api.polymarket.com";
const CLOB: &str = "https://clob.polymarket.com";
const CLOB_WS: &str = "wss://ws-subscriptions-clob.polymarket.com/ws/market";
const FAVORITE_MIN: f64 = 0.65;
const FAVORITE_MAX: f64 = 0.85;
const SHARES: f64 = 5.0;

// ── BTreeMap ordered-float key (same pattern as main bot's feed.rs) ────────
mod of {
    #[derive(Clone, Copy, PartialEq)]
    pub struct OF(pub f64);
    impl Eq for OF {}
    impl PartialOrd for OF {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            self.0.partial_cmp(&o.0)
        }
    }
    impl Ord for OF {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            self.partial_cmp(o).unwrap_or(std::cmp::Ordering::Equal)
        }
    }
}
use of::OF;

#[derive(Default)]
struct TokenBook {
    bids: BTreeMap<OF, f64>,
    asks: BTreeMap<OF, f64>,
}

impl TokenBook {
    fn best_bid(&self) -> Option<f64> {
        self.bids.iter().next_back().map(|(k, _)| k.0)
    }
    fn best_ask(&self) -> Option<f64> {
        self.asks.iter().next().map(|(k, _)| k.0)
    }
}

type SharedBook = Arc<Mutex<HashMap<String, TokenBook>>>;

// ── Market discovery (same logic as main bot) ─────────────────────────────

struct Market {
    slug: String,
    up_token: String,
    down_token: String,
    end_time: DateTime<Utc>,
}

fn ensure_list(v: &Value) -> Value {
    match v {
        Value::String(s) => serde_json::from_str(s).unwrap_or(Value::Null),
        other => other.clone(),
    }
}

fn parse_market(event: &Value) -> Result<Market> {
    let m = event
        .get("markets")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .ok_or_else(|| anyhow!("no markets"))?;
    let slug = event.get("slug").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let end_str = m.get("endDate").and_then(|v| v.as_str()).unwrap_or_default();
    let end_time = DateTime::parse_from_rfc3339(&end_str.replace('Z', "+00:00"))?
        .with_timezone(&Utc);
    let ids_val = m.get("clobTokenIds").map(ensure_list).unwrap_or(Value::Null);
    let ids: Vec<String> = ids_val
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    if ids.len() < 2 {
        return Err(anyhow!("need 2 token ids"));
    }
    let outs_val = m.get("outcomes").map(ensure_list).unwrap_or(Value::Null);
    let outs: Vec<String> = outs_val
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let up_idx = outs
        .iter()
        .position(|o| o.eq_ignore_ascii_case("Up"))
        .unwrap_or(0);
    Ok(Market {
        slug,
        up_token: ids[up_idx].clone(),
        down_token: ids[1 - up_idx].clone(),
        end_time,
    })
}

async fn find_5m_market(http: &reqwest::Client) -> Result<Market> {
    let now = Utc::now();
    let minute = (now.minute() as i64 / 5) * 5;
    let window_start = now
        .with_minute(minute as u32).unwrap()
        .with_second(0).unwrap()
        .with_nanosecond(0).unwrap();
    for offset in [0i64, -5, 5, -10, 10] {
        let ts = (window_start + ChDuration::minutes(offset)).timestamp();
        let slug = format!("btc-updown-5m-{}", ts);
        let Ok(resp) = http
            .get(format!("{GAMMA}/events"))
            .query(&[("slug", slug.as_str())])
            .send().await
        else { continue };
        let Ok(data) = resp.json::<Value>().await else { continue };
        let Some(event) = data.as_array().and_then(|a| a.first()) else {
            continue;
        };
        let Ok(market) = parse_market(event) else { continue };
        if now < market.end_time {
            return Ok(market);
        }
    }
    Err(anyhow!("no active 5m market"))
}

// ── Persistent WS book-feed task ──────────────────────────────────────────

async fn ws_book_task(tokens: Vec<String>, state: SharedBook) {
    loop {
        let Ok((ws, _)) = connect_async(CLOB_WS).await else {
            tokio::time::sleep(Duration::from_secs(2)).await;
            continue;
        };
        let (mut write, mut read) = ws.split();
        let sub = serde_json::json!({
            "assets_ids": tokens,
            "type": "market",
        });
        if write.send(Message::Text(sub.to_string().into())).await.is_err() {
            continue;
        }

        while let Some(msg) = read.next().await {
            let Ok(Message::Text(t)) = msg else { continue };
            if t == "PONG" || t.is_empty() {
                continue;
            }
            let Ok(payload) = serde_json::from_str::<Value>(&t) else { continue };
            let msgs = if payload.is_array() {
                payload.as_array().cloned().unwrap_or_default()
            } else {
                vec![payload]
            };
            let mut books = state.lock().unwrap();
            for m in msgs {
                let asset = m.get("asset_id").and_then(|v| v.as_str()).unwrap_or("");
                match m.get("event_type").and_then(|v| v.as_str()) {
                    Some("book") => {
                        let book = books.entry(asset.to_string()).or_default();
                        book.bids.clear();
                        book.asks.clear();
                        for (side, dst) in
                            [("bids", &mut book.bids), ("asks", &mut book.asks)]
                        {
                            if let Some(arr) = m.get(side).and_then(|v| v.as_array()) {
                                for lvl in arr {
                                    let p = lvl.get("price")
                                        .and_then(|v| v.as_str())
                                        .and_then(|s| s.parse::<f64>().ok());
                                    let sz = lvl.get("size")
                                        .and_then(|v| v.as_str())
                                        .and_then(|s| s.parse::<f64>().ok());
                                    if let (Some(p), Some(sz)) = (p, sz) {
                                        if sz > 0.0 {
                                            dst.insert(OF(p), sz);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Some("price_change") => {
                        if let Some(changes) = m.get("price_changes").and_then(|v| v.as_array()) {
                            for ch in changes {
                                let ch_asset = ch.get("asset_id").and_then(|v| v.as_str()).unwrap_or(asset);
                                let ch_side = ch.get("side").and_then(|v| v.as_str()).unwrap_or("");
                                let p = ch.get("price").and_then(|v| v.as_str()).and_then(|s| s.parse::<f64>().ok());
                                let sz = ch.get("size").and_then(|v| v.as_str()).and_then(|s| s.parse::<f64>().ok());
                                if let (Some(p), Some(sz)) = (p, sz) {
                                    let book = books.entry(ch_asset.to_string()).or_default();
                                    let dst = if ch_side == "BUY" { &mut book.bids } else { &mut book.asks };
                                    if sz > 0.0 {
                                        dst.insert(OF(p), sz);
                                    } else {
                                        dst.remove(&OF(p));
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        eprintln!("  [ws] disconnected, reconnecting");
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

// ── Main ──────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()?;

    // 1. Find the current 5m market.
    println!("Finding current btc-updown-5m market...");
    let market = loop {
        match find_5m_market(&http).await {
            Ok(m) => {
                let tau = (m.end_time - Utc::now()).num_seconds();
                if tau > 60 {
                    break m;
                }
                eprintln!("  only {tau}s left — waiting for next window");
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
            Err(e) => {
                eprintln!("  {e} — retrying");
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        }
    };
    println!("  market:  {}", market.slug);
    println!("  tau:     {}s", (market.end_time - Utc::now()).num_seconds());
    println!("  tokens:  UP={} DOWN={}", &market.up_token[..12], &market.down_token[..12]);

    // 2. Start persistent WS book feed.
    let book_state: SharedBook = Arc::new(Mutex::new(HashMap::new()));
    {
        let tokens = vec![market.up_token.clone(), market.down_token.clone()];
        let s = book_state.clone();
        tokio::spawn(async move { ws_book_task(tokens, s).await });
    }

    // 3. Wait for first book snapshot
    println!("Waiting for first book snapshot...");
    loop {
        tokio::time::sleep(Duration::from_millis(100)).await;
        let books = book_state.lock().unwrap();
        let up_ready = books.get(&market.up_token).map_or(false, |b| b.best_ask().is_some());
        let down_ready = books.get(&market.down_token).map_or(false, |b| b.best_ask().is_some());
        if up_ready && down_ready {
            break;
        }
    }
    println!("  books ready.");

    // 4. Get tick size (needed for bid+1 computation).
    let private_key = std::env::var("PRIVATE_KEY").context("PRIVATE_KEY")?;
    let funder = std::env::var("POLY_FUNDER").context("POLY_FUNDER")?;
    let api_key = std::env::var("POLY_API_KEY").context("POLY_API_KEY")?;
    let api_secret = std::env::var("POLY_API_SECRET").context("POLY_API_SECRET")?;
    let passphrase = std::env::var("POLY_PASSPHRASE").context("POLY_PASSPHRASE")?;
    let clob = Arc::new(ClobClient::with_l2_headers(
        CLOB,
        &private_key,
        137,
        ApiCreds { api_key, secret: api_secret, passphrase },
        Some(SigType::PolyProxy),
        funder.parse().ok(),
    ));
    let up_tick = clob.get_tick_size(&market.up_token).await.map_err(|e| anyhow!("tick: {e}"))?;
    let neg_risk = clob.get_neg_risk(&market.up_token).await.map_err(|e| anyhow!("neg: {e}"))?;
    println!("  tick_size: {up_tick}  neg_risk: {neg_risk}");
    let tick_f64 = up_tick.to_string().parse::<f64>().unwrap_or(0.01);

    // 5. Poll book every 100ms until a favorite emerges in [0.65, 0.85].
    println!();
    println!("Watching for favorite (ask $0.65-$0.85)...");
    let (selected_token, side_name, best_bid, best_ask) = 'outer: loop {
        tokio::time::sleep(Duration::from_millis(100)).await;
        // Bail if window is ending without a favorite — let caller re-run.
        let tau = (market.end_time - Utc::now()).num_seconds();
        if tau < 30 {
            eprintln!();
            eprintln!("  ⚠  window ending in {tau}s with no favorite in range.");
            eprintln!("  ⚠  BTC stayed flat — neither side crossed into [$0.65, $0.85].");
            eprintln!("  ⚠  Re-run the script to try the next window.");
            return Ok(());
        }
        let books = book_state.lock().unwrap();
        let up = books.get(&market.up_token);
        let down = books.get(&market.down_token);
        let up_ask = up.and_then(|b| b.best_ask());
        let up_bid = up.and_then(|b| b.best_bid());
        let down_ask = down.and_then(|b| b.best_ask());
        let down_bid = down.and_then(|b| b.best_bid());

        if let Some(a) = up_ask {
            if (FAVORITE_MIN..=FAVORITE_MAX).contains(&a) {
                break 'outer (market.up_token.clone(), "UP", up_bid.unwrap_or(a - tick_f64), a);
            }
        }
        if let Some(a) = down_ask {
            if (FAVORITE_MIN..=FAVORITE_MAX).contains(&a) {
                break 'outer (market.down_token.clone(), "DOWN", down_bid.unwrap_or(a - tick_f64), a);
            }
        }
        drop(books);
        // Live status every ~2s
        if Utc::now().timestamp() % 2 == 0 {
            eprintln!(
                "  [tau={tau}s] UP {:.3?}/{:.3?}  DOWN {:.3?}/{:.3?}",
                up_bid, up_ask, down_bid, down_ask
            );
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    };

    let order_price = (best_bid + tick_f64).min(best_ask);
    println!();
    println!("═══ Selected ═══");
    println!("  side:          {side_name}");
    println!("  best_bid:      ${:.4}", best_bid);
    println!("  best_ask:      ${:.4}", best_ask);
    println!("  tick:          ${:.4}", tick_f64);
    println!("  order price:   ${:.4} (bid + 1 tick)", order_price);
    println!(
        "  interpretation: {}",
        if (order_price - best_ask).abs() < 1e-9 {
            "== ask (will cross as taker)"
        } else {
            "< ask (will rest as maker)"
        }
    );

    // 6. Place the order.
    let opts = OrderOptions {
        tick_size: Some(up_tick),
        neg_risk: Some(neg_risk),
        fee_rate_bps: None,
    };
    let price_dec = Decimal::from_f64(order_price).ok_or_else(|| anyhow!("dec price"))?;
    let size_dec = Decimal::from_f64(SHARES).ok_or_else(|| anyhow!("dec size"))?;
    let order_args = OrderArgs::new(&selected_token, price_dec, size_dec, Side::BUY);
    let extras = Some(ExtraOrderArgs { fee_rate_bps: 1000, ..Default::default() });

    println!();
    println!("Placing GTC BUY {SHARES} @ ${order_price:.4}...");
    let signed = clob
        .create_order(&order_args, None, extras, Some(&opts))
        .await
        .map_err(|e| anyhow!("create: {e}"))?;
    let resp = clob
        .post_order(signed, PolyOrderType::GTC)
        .await
        .map_err(|e| anyhow!("post: {e}"))?;

    println!();
    println!("═══ Response ═══");
    println!("  success:       {}", resp.success);
    println!("  order_id:      {}", resp.order_id);
    if let Some(e) = &resp.error_msg {
        println!("  error:         {e}");
    }
    println!("  making_amount: {:?}  (USD spent)", resp.making_amount);
    println!("  taking_amount: {:?}  (shares received)", resp.taking_amount);

    if !resp.success {
        return Err(anyhow!("rejected"));
    }

    let taking = resp.taking_amount.unwrap_or_default();
    let making = resp.making_amount.unwrap_or_default();

    println!();
    println!("═══ Classification ═══");
    if taking.is_zero() {
        println!("  ⟂ MAKER — order rests on book. No fee yet.");
        println!("    Sitting at ${order_price:.4}, will fill if someone sells into it.");
        println!("    Cancel via polymarket.com or wait for resolution.");
    } else {
        let effective = making / taking;
        println!("  ⟂ TAKER — order crossed and matched.");
        println!("    cost:            ${making}  (making_amount)");
        println!("    shares:          {taking}  (taking_amount)");
        println!("    effective price: ${effective}");
        println!("    your limit:      ${order_price:.4}");
        if effective > price_dec {
            let delta = effective - price_dec;
            let pct = (delta / price_dec) * Decimal::new(100, 0);
            println!("    → fee added on top: +${delta} ({pct:.2}%)");
        } else if effective < price_dec {
            let delta = price_dec - effective;
            println!("    → filled BELOW limit by ${delta} (got extra shares)");
        } else {
            println!("    → filled at exact limit price (no visible fee adjustment)");
        }
    }

    // 7. Wait for window close and resolution.
    println!();
    println!("Waiting for window close (tau countdown)...");
    loop {
        let remaining = (market.end_time - Utc::now()).num_seconds();
        if remaining <= 0 {
            break;
        }
        if remaining % 30 == 0 || remaining <= 10 {
            println!("  tau = {remaining}s");
        }
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
    println!("Window closed. Polling for resolution...");
    for attempt in 0..18 {
        tokio::time::sleep(Duration::from_secs(10)).await;
        let resp = http
            .get(format!("{GAMMA}/events"))
            .query(&[("slug", market.slug.as_str())])
            .send().await?;
        let data: Value = resp.json().await?;
        let Some(event) = data.as_array().and_then(|a| a.first()) else { continue };
        let Some(m) = event
            .get("markets")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
        else { continue };
        let closed = m.get("closed").and_then(|v| v.as_bool()).unwrap_or(false);
        let uma = m.get("umaResolutionStatus").and_then(|v| v.as_str()).unwrap_or("");
        println!("  attempt {}: closed={closed} uma={uma}", attempt + 1);
        if closed && uma == "resolved" {
            let prices_val = m.get("outcomePrices").map(ensure_list).unwrap_or(Value::Null);
            let prices: Vec<f64> = prices_val.as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().and_then(|s| s.parse().ok())).collect())
                .unwrap_or_default();
            let outs_val = m.get("outcomes").map(ensure_list).unwrap_or(Value::Null);
            let outs: Vec<String> = outs_val.as_array()
                .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default();
            if prices.len() == 2 && outs.len() == 2 {
                let up_idx = outs.iter().position(|o| o.eq_ignore_ascii_case("Up")).unwrap_or(0);
                let up_won = prices[up_idx] >= 0.5;
                let winner = if up_won { "UP" } else { "DOWN" };
                let we_won = winner == side_name;
                println!();
                println!("═══ Resolution ═══");
                println!("  winner:  {winner}");
                println!("  we bet:  {side_name}");
                println!("  result:  {}", if we_won { "WON ✓" } else { "LOST ✗" });
                let cost = making.to_f64().unwrap_or(order_price * SHARES);
                let shares = taking.to_f64().unwrap_or(SHARES);
                if we_won && !taking.is_zero() {
                    println!("  cost:    ${:.4}", cost);
                    println!("  payout:  ${:.4}", shares);
                    println!("  profit:  ${:+.4}", shares - cost);
                } else if taking.is_zero() {
                    println!("  (order didn't fill — no position, neither profit nor loss)");
                } else {
                    println!("  loss:    ${:.4}", cost);
                }
                return Ok(());
            }
        }
    }
    println!("Resolution timeout.");
    Ok(())
}
