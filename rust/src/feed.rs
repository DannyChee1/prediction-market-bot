use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use crate::book::BookSnapshot;
use crate::types::now_s;

// ── Helpers ─────────────────────────────────────────────────────────────────

fn store_f64(a: &AtomicU64, v: f64) {
    a.store(v.to_bits(), Ordering::Release);
}
fn load_f64(a: &AtomicU64) -> f64 {
    f64::from_bits(a.load(Ordering::Acquire))
}

mod ordered_float {
    #[derive(Clone, Copy, PartialEq)]
    pub struct OrderedFloat<T>(pub T);
    impl<T: PartialOrd> Eq for OrderedFloat<T> {}
    impl<T: PartialOrd> PartialOrd for OrderedFloat<T> {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            self.0.partial_cmp(&other.0)
        }
    }
    impl<T: PartialOrd> Ord for OrderedFloat<T> {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
        }
    }
}
use ordered_float::OrderedFloat;

// ── BookFeed ────────────────────────────────────────────────────────────────

#[derive(Default)]
struct SimpleBook {
    bids: BTreeMap<OrderedFloat<f64>, f64>,
    asks: BTreeMap<OrderedFloat<f64>, f64>,
    /// Any event touching this book (for WS-alive signalling).
    last_update_ts: f64,
    /// Timestamp of the last change to the best_bid OR best_ask VALUE.
    /// This is the "arb staleness" clock — tracks top-of-book repricing,
    /// ignores deep-level churn and same-level size updates.
    last_best_change_ts: f64,
    /// Previously-observed top-of-book values, so we can detect changes.
    last_best_bid: Option<f64>,
    last_best_ask: Option<f64>,
}

impl SimpleBook {
    fn current_best_bid(&self) -> Option<f64> {
        self.bids.iter().next_back().map(|(p, _)| p.0)
    }
    fn current_best_ask(&self) -> Option<f64> {
        self.asks.iter().next().map(|(p, _)| p.0)
    }
    /// Call after any mutation to bids/asks. Only bumps the staleness
    /// clock if the best-level value changed; size-only updates and
    /// deep-level changes don't reset it.
    fn recompute_best(&mut self, now: f64) {
        let new_bid = self.current_best_bid();
        let new_ask = self.current_best_ask();
        if new_bid != self.last_best_bid || new_ask != self.last_best_ask {
            self.last_best_change_ts = now;
            self.last_best_bid = new_bid;
            self.last_best_ask = new_ask;
        }
    }
}

struct BookState {
    books: HashMap<String, SimpleBook>,
    last_update_ts: f64,
}

pub struct BookFeed {
    state: Arc<RwLock<BookState>>,
    /// Aborts the background WS task when this BookFeed is dropped.
    /// Prevents a WS-connection leak when we rotate to a new market.
    task: Option<tokio::task::AbortHandle>,
}

impl Drop for BookFeed {
    fn drop(&mut self) {
        if let Some(h) = self.task.take() {
            h.abort();
        }
    }
}

impl BookFeed {
    pub fn new(tokens: Vec<String>) -> Self {
        let url = "wss://ws-subscriptions-clob.polymarket.com/ws/market".to_string();
        let state = Arc::new(RwLock::new(BookState {
            books: tokens.iter().map(|t| (t.clone(), SimpleBook::default())).collect(),
            last_update_ts: 0.0,
        }));
        let state_clone = state.clone();
        let handle = tokio::spawn(async move {
            book_feed_task(url, tokens, state_clone).await;
        });
        Self { state, task: Some(handle.abort_handle()) }
    }

    pub fn snapshot(&self, token_id: &str) -> BookSnapshot {
        let state = self.state.read().unwrap();
        if let Some(book) = state.books.get(token_id) {
            let bids: Vec<(f64, f64)> = book
                .bids
                .iter()
                .rev()
                .take(10)
                .map(|(p, s)| (p.0, *s))
                .collect();
            let asks: Vec<(f64, f64)> = book
                .asks
                .iter()
                .take(10)
                .map(|(p, s)| (p.0, *s))
                .collect();
            // Arb staleness uses top-of-book change time, not any-event time.
            // Falls back to last_update_ts (any event) and then state-wide
            // timestamp for fresh subscriptions still warming up.
            let ts = if book.last_best_change_ts > 0.0 {
                book.last_best_change_ts
            } else if book.last_update_ts > 0.0 {
                book.last_update_ts
            } else {
                state.last_update_ts
            };
            BookSnapshot {
                best_bid: bids.first().map(|(p, _)| *p),
                best_ask: asks.first().map(|(p, _)| *p),
                bids,
                asks,
                timestamp: ts,
            }
        } else {
            BookSnapshot::default()
        }
    }
}

async fn book_feed_task(url: String, tokens: Vec<String>, state: Arc<RwLock<BookState>>) {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::connect_async;
    use tokio_tungstenite::tungstenite::Message;

    let mut backoff = 2u64;

    loop {
        let (ws, _) = match connect_async(&url).await {
            Ok(c) => {
                backoff = 2;
                c
            }
            Err(e) => {
                eprintln!("[BookFeed] connect error: {e}");
                tokio::time::sleep(std::time::Duration::from_secs(backoff.min(60))).await;
                backoff = (backoff * 2).min(60);
                continue;
            }
        };

        let (mut write, mut read) = ws.split();
        let sub_msg = serde_json::json!({
            "assets_ids": tokens,
            "type": "market",
            "custom_feature_enabled": true,
        });
        if write.send(Message::Text(sub_msg.to_string())).await.is_err() {
            continue;
        }

        let hb_handle = tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                if write.send(Message::Text("PING".to_string())).await.is_err() {
                    break;
                }
            }
        });

        let read_timeout = std::time::Duration::from_secs(30);
        loop {
            let msg = match tokio::time::timeout(read_timeout, read.next()).await {
                Ok(Some(m)) => m,
                Ok(None) => break,
                Err(_) => {
                    eprintln!("[BookFeed] no data for 30s, reconnecting");
                    break;
                }
            };
            let text = match msg {
                Ok(Message::Text(t)) => t,
                Ok(_) => continue,
                Err(e) => {
                    eprintln!("[BookFeed] read error: {e}");
                    break;
                }
            };

            if text == "PONG" || text.is_empty() {
                let mut s = state.write().unwrap();
                let now = now_s();
                s.last_update_ts = now;
                for book in s.books.values_mut() {
                    book.last_update_ts = now;
                }
                continue;
            }

            let Ok(payload) = serde_json::from_str::<serde_json::Value>(&text) else {
                continue;
            };
            let msgs = if payload.is_array() {
                payload.as_array().cloned().unwrap_or_default()
            } else {
                vec![payload]
            };

            let mut s = state.write().unwrap();
            let now = now_s();
            s.last_update_ts = now;

            for msg in msgs {
                let etype = msg.get("event_type").and_then(|v| v.as_str());
                let asset_id = msg.get("asset_id").and_then(|v| v.as_str()).unwrap_or("");

                match etype {
                    Some("book") => {
                        if let Some(book) = s.books.get_mut(asset_id) {
                            book.bids.clear();
                            book.asks.clear();
                            for (side_key, dst) in
                                [("bids", &mut book.bids), ("asks", &mut book.asks)].iter_mut()
                            {
                                if let Some(levels) = msg.get(*side_key).and_then(|v| v.as_array())
                                {
                                    for level in levels {
                                        let p = level
                                            .get("price")
                                            .and_then(|v| v.as_str())
                                            .and_then(|s| s.parse::<f64>().ok());
                                        let sz = level
                                            .get("size")
                                            .and_then(|v| v.as_str())
                                            .and_then(|s| s.parse::<f64>().ok());
                                        if let (Some(p), Some(sz)) = (p, sz) {
                                            if sz > 0.0 {
                                                dst.insert(OrderedFloat(p), sz);
                                            }
                                        }
                                    }
                                }
                            }
                            book.last_update_ts = now;
                            book.recompute_best(now);
                        }
                    }
                    Some("price_change") => {
                        if let Some(changes) =
                            msg.get("price_changes").and_then(|v| v.as_array())
                        {
                            for ch in changes {
                                let ch_asset = ch
                                    .get("asset_id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or(asset_id);
                                let ch_side =
                                    ch.get("side").and_then(|v| v.as_str()).unwrap_or("");
                                let p = ch
                                    .get("price")
                                    .and_then(|v| v.as_str())
                                    .and_then(|s| s.parse::<f64>().ok());
                                let sz = ch
                                    .get("size")
                                    .and_then(|v| v.as_str())
                                    .and_then(|s| s.parse::<f64>().ok());
                                if let (Some(p), Some(sz)) = (p, sz) {
                                    if let Some(tb) = s.books.get_mut(ch_asset) {
                                        let levels = if ch_side == "BUY" {
                                            &mut tb.bids
                                        } else {
                                            &mut tb.asks
                                        };
                                        let key = OrderedFloat(p);
                                        if sz > 0.0 {
                                            levels.insert(key, sz);
                                        } else {
                                            levels.remove(&key);
                                        }
                                        tb.last_update_ts = now;
                                        tb.recompute_best(now);
                                    }
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Clear stale books before reconnecting
        {
            let mut s = state.write().unwrap();
            for book in s.books.values_mut() {
                book.bids.clear();
                book.asks.clear();
                book.last_update_ts = 0.0;
            }
            s.last_update_ts = 0.0;
        }

        hb_handle.abort();
        tokio::time::sleep(std::time::Duration::from_secs(backoff.min(30))).await;
        backoff = (backoff * 2).min(60);
    }
}

// ── PriceFeed (Chainlink via RTDS) ──────────────────────────────────────────

#[allow(dead_code)]
pub struct PriceFeed {
    price: Arc<AtomicU64>,
    last_update_ts: Arc<AtomicU64>,
}

impl PriceFeed {
    pub fn new(symbol: String) -> Self {
        let price = Arc::new(AtomicU64::new(0));
        let ts = Arc::new(AtomicU64::new(0));
        let p = price.clone();
        let t = ts.clone();
        tokio::spawn(async move {
            price_feed_task(symbol, p, t).await;
        });
        Self { price, last_update_ts: ts }
    }

    pub fn price(&self) -> Option<f64> {
        let v = load_f64(&self.price);
        if v == 0.0 {
            None
        } else {
            Some(v)
        }
    }

    #[allow(dead_code)]
    pub fn last_update_ts(&self) -> f64 {
        load_f64(&self.last_update_ts)
    }
}

async fn price_feed_task(
    symbol: String,
    price: Arc<AtomicU64>,
    last_update_ts: Arc<AtomicU64>,
) {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::connect_async;
    use tokio_tungstenite::tungstenite::Message;

    let url = "wss://ws-live-data.polymarket.com";
    let mut backoff = 2u64;

    loop {
        let (ws, _) = match connect_async(url).await {
            Ok(c) => {
                backoff = 2;
                c
            }
            Err(e) => {
                eprintln!("[PriceFeed] connect error: {e}");
                tokio::time::sleep(std::time::Duration::from_secs(backoff.min(60))).await;
                backoff = (backoff * 2).min(60);
                continue;
            }
        };

        let (mut write, mut read) = ws.split();
        let sub = serde_json::json!({
            "action": "subscribe",
            "subscriptions": [{"topic": "crypto_prices_chainlink", "type": "*"}],
        });
        let _ = write.send(Message::Text(sub.to_string())).await;

        let hb = tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                if write.send(Message::Text("PING".to_string())).await.is_err() {
                    break;
                }
            }
        });

        let read_timeout = std::time::Duration::from_secs(30);
        loop {
            let msg = match tokio::time::timeout(read_timeout, read.next()).await {
                Ok(Some(m)) => m,
                Ok(None) => break,
                Err(_) => {
                    eprintln!("[PriceFeed] no data for 30s, reconnecting");
                    break;
                }
            };
            let text = match msg {
                Ok(Message::Text(t)) => t,
                Ok(_) => continue,
                Err(e) => {
                    eprintln!("[PriceFeed] read error: {e}");
                    break;
                }
            };
            if text == "PONG" || text.is_empty() {
                continue;
            }
            let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) else {
                continue;
            };
            let payload = parsed.get("payload").cloned().unwrap_or(parsed.clone());
            let msg_symbol = payload.get("symbol").and_then(|v| v.as_str());
            if msg_symbol != Some(&symbol) {
                continue;
            }
            let p = payload
                .get("data")
                .and_then(|d| d.as_array())
                .and_then(|arr| arr.last())
                .and_then(|e| e.get("value"))
                .and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
                .or_else(|| {
                    payload
                        .get("value")
                        .and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
                });
            if let Some(px) = p {
                store_f64(&price, px);
                store_f64(&last_update_ts, now_s());
            }
        }

        hb.abort();
        tokio::time::sleep(std::time::Duration::from_secs(backoff.min(30))).await;
        backoff = (backoff * 2).min(60);
    }
}

// ── BinanceFeed (JSON bookTicker) ───────────────────────────────────────────

pub struct BinanceFeed {
    mid: Arc<AtomicU64>,
    last_update_ts: Arc<AtomicU64>,
}

impl BinanceFeed {
    pub fn new(symbol: String) -> Self {
        let mid = Arc::new(AtomicU64::new(0));
        let ts = Arc::new(AtomicU64::new(0));
        let m = mid.clone();
        let t = ts.clone();
        tokio::spawn(async move {
            binance_feed_task(symbol, m, t).await;
        });
        Self { mid, last_update_ts: ts }
    }

    pub fn mid(&self) -> Option<f64> {
        let v = load_f64(&self.mid);
        if v == 0.0 {
            None
        } else {
            Some(v)
        }
    }

    pub fn last_update_ts(&self) -> f64 {
        load_f64(&self.last_update_ts)
    }
}

// ── CoinbaseFeed (ticker channel) ───────────────────────────────────────────

pub struct CoinbaseFeed {
    mid: Arc<AtomicU64>,
    last_update_ts: Arc<AtomicU64>,
}

impl CoinbaseFeed {
    /// product_id in Coinbase format, e.g. "BTC-USD"
    pub fn new(product_id: String) -> Self {
        let mid = Arc::new(AtomicU64::new(0));
        let ts = Arc::new(AtomicU64::new(0));
        let m = mid.clone();
        let t = ts.clone();
        tokio::spawn(async move {
            coinbase_feed_task(product_id, m, t).await;
        });
        Self { mid, last_update_ts: ts }
    }

    pub fn mid(&self) -> Option<f64> {
        let v = load_f64(&self.mid);
        if v == 0.0 { None } else { Some(v) }
    }

    pub fn last_update_ts(&self) -> f64 {
        load_f64(&self.last_update_ts)
    }
}

async fn coinbase_feed_task(
    product_id: String,
    mid: Arc<AtomicU64>,
    last_update_ts: Arc<AtomicU64>,
) {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::connect_async;
    use tokio_tungstenite::tungstenite::Message;

    let url = "wss://ws-feed.exchange.coinbase.com";
    let sub = serde_json::json!({
        "type": "subscribe",
        "product_ids": [product_id],
        "channels": ["ticker"],
    });
    let mut backoff = 2u64;

    loop {
        let (ws, _) = match connect_async(url).await {
            Ok(c) => { backoff = 2; c }
            Err(e) => {
                eprintln!("[CoinbaseFeed] connect error: {e}");
                tokio::time::sleep(std::time::Duration::from_secs(backoff.min(60))).await;
                backoff = (backoff * 2).min(60);
                continue;
            }
        };
        let (mut write, mut read) = ws.split();
        if write.send(Message::Text(sub.to_string())).await.is_err() {
            continue;
        }
        let read_timeout = std::time::Duration::from_secs(30);
        loop {
            let msg = match tokio::time::timeout(read_timeout, read.next()).await {
                Ok(Some(m)) => m,
                Ok(None) => break,
                Err(_) => {
                    eprintln!("[CoinbaseFeed] no data for 30s, reconnecting");
                    break;
                }
            };
            match msg {
                Ok(Message::Text(t)) => {
                    let Ok(v) = serde_json::from_str::<serde_json::Value>(&t) else { continue };
                    if v.get("type").and_then(|x| x.as_str()) != Some("ticker") {
                        continue;
                    }
                    let bid = v.get("best_bid")
                        .and_then(|x| x.as_str())
                        .and_then(|s| s.parse::<f64>().ok());
                    let ask = v.get("best_ask")
                        .and_then(|x| x.as_str())
                        .and_then(|s| s.parse::<f64>().ok());
                    if let (Some(b), Some(a)) = (bid, ask) {
                        store_f64(&mid, (b + a) / 2.0);
                        store_f64(&last_update_ts, now_s());
                    }
                }
                Ok(Message::Ping(data)) => {
                    let _ = write.send(Message::Pong(data)).await;
                }
                Ok(Message::Close(_)) => break,
                Ok(_) => {}
                Err(e) => {
                    eprintln!("[CoinbaseFeed] read error: {e}");
                    break;
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_secs(backoff.min(30))).await;
        backoff = (backoff * 2).min(60);
    }
}

async fn binance_feed_task(
    symbol: String,
    mid: Arc<AtomicU64>,
    last_update_ts: Arc<AtomicU64>,
) {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::connect_async;
    use tokio_tungstenite::tungstenite::Message;

    let url = format!(
        "wss://data-stream.binance.vision/ws/{}@bookTicker",
        symbol.to_lowercase()
    );
    let mut backoff = 2u64;

    loop {
        let (ws, _) = match connect_async(&url).await {
            Ok(c) => {
                backoff = 2;
                c
            }
            Err(e) => {
                eprintln!("[BinanceFeed] connect error: {e}");
                tokio::time::sleep(std::time::Duration::from_secs(backoff.min(60))).await;
                backoff = (backoff * 2).min(60);
                continue;
            }
        };
        let (mut write, mut read) = ws.split();
        let read_timeout = std::time::Duration::from_secs(30);
        loop {
            let msg = match tokio::time::timeout(read_timeout, read.next()).await {
                Ok(Some(m)) => m,
                Ok(None) => break,
                Err(_) => {
                    eprintln!("[BinanceFeed] no data for 30s, reconnecting");
                    break;
                }
            };
            match msg {
                Ok(Message::Text(t)) => {
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&t) {
                        let bid = parsed
                            .get("b")
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.parse::<f64>().ok());
                        let ask = parsed
                            .get("a")
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.parse::<f64>().ok());
                        if let (Some(b), Some(a)) = (bid, ask) {
                            store_f64(&mid, (b + a) / 2.0);
                            store_f64(&last_update_ts, now_s());
                        }
                    }
                }
                Ok(Message::Ping(data)) => {
                    let _ = write.send(Message::Pong(data)).await;
                }
                Ok(Message::Close(_)) => break,
                Ok(_) => {}
                Err(e) => {
                    eprintln!("[BinanceFeed] read error: {e}");
                    break;
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_secs(backoff.min(30))).await;
        backoff = (backoff * 2).min(60);
    }
}
