use pyo3::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::book::BookSnapshot;
use crate::RUNTIME;

// ── BookFeed ────────────────────────────────────────────────────────────────

/// CLOB order book WebSocket feed wrapping polyfill-rs streaming.
///
/// Spawns a background tokio task that connects to the Polymarket CLOB WS,
/// subscribes to market channels, and maintains thread-safe order books
/// via polyfill-rs's OrderBookImpl / OrderBookManager.
///
/// Python reads book snapshots via .snapshot(token_id) which acquires
/// a read lock — fast and non-blocking for the WS task.
#[pyclass]
pub struct BookFeed {
    books: Arc<RwLock<BookState>>,
    _cancel: Arc<tokio::sync::Notify>,
}

struct BookState {
    books: std::collections::HashMap<String, SimpleBook>,
    last_update_ts: f64,
    /// Trade events buffer for VPIN: (size, side_str). Drained by Python.
    trade_buffer: std::collections::VecDeque<(f64, String)>,
}

#[derive(Default)]
struct SimpleBook {
    bids: std::collections::BTreeMap<ordered_float::OrderedFloat<f64>, f64>,
    asks: std::collections::BTreeMap<ordered_float::OrderedFloat<f64>, f64>,
}

// We use a simple ordered-float wrapper since we can't depend on the crate.
// Using raw f64 keys with BTreeMap requires Ord, so we wrap in a newtype.
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

#[pymethods]
impl BookFeed {
    /// Create a BookFeed and start streaming order book data.
    ///
    /// Args:
    ///     tokens: List of token IDs to subscribe to (e.g. [up_token, down_token])
    ///     ws_url: WebSocket URL (default: Polymarket CLOB WS)
    #[new]
    #[pyo3(signature = (tokens, ws_url=None))]
    fn new(tokens: Vec<String>, ws_url: Option<String>) -> PyResult<Self> {
        let url = ws_url.unwrap_or_else(|| {
            "wss://ws-subscriptions-clob.polymarket.com/ws/market".to_string()
        });

        let books = Arc::new(RwLock::new(BookState {
            books: tokens
                .iter()
                .map(|t| (t.clone(), SimpleBook::default()))
                .collect(),
            last_update_ts: 0.0,
            trade_buffer: std::collections::VecDeque::with_capacity(1024),
        }));
        let cancel = Arc::new(tokio::sync::Notify::new());

        let books_clone = books.clone();
        let cancel_clone = cancel.clone();
        let tokens_clone = tokens.clone();

        RUNTIME.spawn(async move {
            book_feed_task(url, tokens_clone, books_clone, cancel_clone).await;
        });

        Ok(Self {
            books,
            _cancel: cancel,
        })
    }

    /// Drain accumulated trade events for VPIN calculation.
    ///
    /// Returns list of (size, side_str) tuples and clears the buffer.
    /// Call this periodically from Python to feed VPIN bar accumulation.
    fn drain_trades(&self) -> Vec<(f64, String)> {
        let mut state = RUNTIME.block_on(async { self.books.write().await });
        state.trade_buffer.drain(..).collect()
    }

    /// Read the current order book snapshot for a token.
    ///
    /// Returns a BookSnapshot with best_bid, best_ask, top bids/asks.
    /// This is a fast read-lock operation (~1µs).
    fn snapshot(&self, token_id: &str) -> BookSnapshot {
        let state = RUNTIME.block_on(async { self.books.read().await });

        if let Some(book) = state.books.get(token_id) {
            let bids: Vec<(f64, f64)> = book
                .bids
                .iter()
                .rev() // highest price first
                .take(10)
                .map(|(p, s)| (p.0, *s))
                .collect();

            let asks: Vec<(f64, f64)> = book
                .asks
                .iter() // lowest price first
                .take(10)
                .map(|(p, s)| (p.0, *s))
                .collect();

            BookSnapshot {
                best_bid: bids.first().map(|(p, _)| *p),
                best_ask: asks.first().map(|(p, _)| *p),
                bids,
                asks,
                timestamp: state.last_update_ts,
            }
        } else {
            BookSnapshot {
                best_bid: None,
                best_ask: None,
                bids: Vec::new(),
                asks: Vec::new(),
                timestamp: 0.0,
            }
        }
    }
}

/// Background task: connect to CLOB WS, parse messages, update books.
async fn book_feed_task(
    url: String,
    tokens: Vec<String>,
    books: Arc<RwLock<BookState>>,
    cancel: Arc<tokio::sync::Notify>,
) {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::connect_async;

    let mut backoff = 2u64;

    loop {
        let result = connect_async(&url).await;
        let (mut ws, _) = match result {
            Ok(conn) => {
                backoff = 2;
                conn
            }
            Err(e) => {
                eprintln!("[BookFeed] connect error: {e}");
                tokio::time::sleep(std::time::Duration::from_secs(backoff.min(60))).await;
                backoff = (backoff * 2).min(60);
                continue;
            }
        };

        // Subscribe to market channel
        let sub_msg = serde_json::json!({
            "assets_ids": tokens,
            "type": "market",
            "custom_feature_enabled": true,
        });
        if let Err(e) = ws
            .send(tokio_tungstenite::tungstenite::Message::Text(
                sub_msg.to_string(),
            ))
            .await
        {
            eprintln!("[BookFeed] subscribe error: {e}");
            continue;
        }

        // Heartbeat task
        let (mut write, mut read) = ws.split();

        let cancel_clone = cancel.clone();
        let hb = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(std::time::Duration::from_secs(10)) => {
                        let _ = write
                            .send(tokio_tungstenite::tungstenite::Message::Text(
                                "PING".to_string(),
                            ))
                            .await;
                    }
                    _ = cancel_clone.notified() => break,
                }
            }
            write
        });

        // Read messages
        while let Some(msg) = read.next().await {
            let text = match msg {
                Ok(tokio_tungstenite::tungstenite::Message::Text(t)) => t,
                Ok(_) => continue,
                Err(_) => break,
            };

            if text == "PONG" || text.is_empty() {
                continue;
            }

            // Parse and apply book updates
            if let Ok(payload) = serde_json::from_str::<serde_json::Value>(&text) {
                let msgs = if payload.is_array() {
                    payload
                        .as_array()
                        .cloned()
                        .unwrap_or_default()
                } else {
                    vec![payload]
                };

                let mut state = books.write().await;
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64();
                state.last_update_ts = now;

                for msg in msgs {
                    let etype = msg.get("event_type").and_then(|v| v.as_str());
                    let asset_id = msg
                        .get("asset_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");

                    match etype {
                        Some("book") => {
                            // Full snapshot: replace book
                            if let Some(book) = state.books.get_mut(asset_id) {
                                book.bids.clear();
                                book.asks.clear();
                                if let Some(bids) = msg.get("bids").and_then(|v| v.as_array()) {
                                    for level in bids {
                                        if let (Some(p), Some(s)) = (
                                            level
                                                .get("price")
                                                .and_then(|v| v.as_str())
                                                .and_then(|s| s.parse::<f64>().ok()),
                                            level
                                                .get("size")
                                                .and_then(|v| v.as_str())
                                                .and_then(|s| s.parse::<f64>().ok()),
                                        ) {
                                            if s > 0.0 {
                                                book.bids
                                                    .insert(ordered_float::OrderedFloat(p), s);
                                            }
                                        }
                                    }
                                }
                                if let Some(asks) = msg.get("asks").and_then(|v| v.as_array()) {
                                    for level in asks {
                                        if let (Some(p), Some(s)) = (
                                            level
                                                .get("price")
                                                .and_then(|v| v.as_str())
                                                .and_then(|s| s.parse::<f64>().ok()),
                                            level
                                                .get("size")
                                                .and_then(|v| v.as_str())
                                                .and_then(|s| s.parse::<f64>().ok()),
                                        ) {
                                            if s > 0.0 {
                                                book.asks
                                                    .insert(ordered_float::OrderedFloat(p), s);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Some("price_change") => {
                            // Incremental updates — each change has its own asset_id
                            if let Some(changes) =
                                msg.get("price_changes").and_then(|v| v.as_array())
                            {
                                for ch in changes {
                                    let ch_asset = ch
                                        .get("asset_id")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or(asset_id);
                                    let ch_side = ch
                                        .get("side")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("");
                                    let ch_price = ch
                                        .get("price")
                                        .and_then(|v| v.as_str())
                                        .and_then(|s| s.parse::<f64>().ok());
                                    let ch_size = ch
                                        .get("size")
                                        .and_then(|v| v.as_str())
                                        .and_then(|s| s.parse::<f64>().ok());

                                    if let (Some(p), Some(s)) = (ch_price, ch_size) {
                                        if let Some(tb) = state.books.get_mut(ch_asset) {
                                            let levels = if ch_side == "BUY" {
                                                &mut tb.bids
                                            } else {
                                                &mut tb.asks
                                            };
                                            let key = ordered_float::OrderedFloat(p);
                                            if s == 0.0 {
                                                levels.remove(&key);
                                            } else {
                                                levels.insert(key, s);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Some("last_trade_price") => {
                            // Trade event for VPIN — buffer (size, side)
                            let size = msg
                                .get("size")
                                .and_then(|v| {
                                    v.as_str()
                                        .and_then(|s| s.parse::<f64>().ok())
                                        .or_else(|| v.as_f64())
                                });
                            let trade_side = msg
                                .get("side")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_uppercase();
                            if let Some(sz) = size {
                                if sz > 0.0
                                    && (trade_side == "BUY" || trade_side == "SELL")
                                {
                                    state.trade_buffer.push_back((sz, trade_side));
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Reconnect
        let _ = hb.await;
        tokio::time::sleep(std::time::Duration::from_secs(backoff.min(30))).await;
        backoff = (backoff * 2).min(60);
    }
}

// ── PriceFeed ───────────────────────────────────────────────────────────────

/// RTDS Chainlink price feed via WebSocket.
///
/// Spawns a background tokio task that connects to the Polymarket RTDS WS
/// and filters for the specified symbol (e.g. "btc/usd").
/// Stores latest price in an AtomicU64 (bit-cast f64) for lock-free reads.
#[pyclass]
pub struct PriceFeed {
    price: Arc<AtomicU64>,
    last_update_ts: Arc<AtomicU64>,
    _cancel: Arc<tokio::sync::Notify>,
}

fn store_f64(atomic: &AtomicU64, val: f64) {
    atomic.store(val.to_bits(), Ordering::Release);
}

fn load_f64(atomic: &AtomicU64) -> f64 {
    f64::from_bits(atomic.load(Ordering::Acquire))
}

#[pymethods]
impl PriceFeed {
    /// Create a PriceFeed and start streaming Chainlink prices.
    ///
    /// Args:
    ///     symbol: Chainlink symbol to filter for (e.g. "btc/usd")
    #[new]
    fn new(symbol: String) -> Self {
        let price = Arc::new(AtomicU64::new(0));
        let last_update_ts = Arc::new(AtomicU64::new(0));
        let cancel = Arc::new(tokio::sync::Notify::new());

        let p = price.clone();
        let ts = last_update_ts.clone();
        let c = cancel.clone();

        RUNTIME.spawn(async move {
            price_feed_task(symbol, p, ts, c).await;
        });

        Self {
            price,
            last_update_ts,
            _cancel: cancel,
        }
    }

    /// Get the latest Chainlink price, or None if no data yet.
    fn price(&self) -> Option<f64> {
        let v = load_f64(&self.price);
        if v == 0.0 {
            None
        } else {
            Some(v)
        }
    }

    /// Timestamp (seconds since epoch) of the last price update.
    fn last_update_ts(&self) -> f64 {
        load_f64(&self.last_update_ts)
    }
}

async fn price_feed_task(
    symbol: String,
    price: Arc<AtomicU64>,
    last_update_ts: Arc<AtomicU64>,
    cancel: Arc<tokio::sync::Notify>,
) {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::connect_async;

    let url = "wss://ws-live-data.polymarket.com";
    let mut backoff = 2u64;

    loop {
        let result = connect_async(url).await;
        let (mut ws, _) = match result {
            Ok(conn) => {
                backoff = 2;
                conn
            }
            Err(e) => {
                eprintln!("[PriceFeed] connect error: {e}");
                tokio::time::sleep(std::time::Duration::from_secs(backoff.min(60))).await;
                backoff = (backoff * 2).min(60);
                continue;
            }
        };

        // Subscribe to Chainlink prices
        let sub_msg = serde_json::json!({
            "action": "subscribe",
            "subscriptions": [{
                "topic": "crypto_prices_chainlink",
                "type": "*",
            }],
        });
        let _ = ws
            .send(tokio_tungstenite::tungstenite::Message::Text(
                sub_msg.to_string(),
            ))
            .await;

        let (mut write, mut read) = ws.split();

        // Heartbeat
        let cancel_clone = cancel.clone();
        let hb = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => {
                        let _ = write.send(
                            tokio_tungstenite::tungstenite::Message::Text("PING".to_string())
                        ).await;
                    }
                    _ = cancel_clone.notified() => break,
                }
            }
            write
        });

        while let Some(msg) = read.next().await {
            let text = match msg {
                Ok(tokio_tungstenite::tungstenite::Message::Text(t)) => t,
                Ok(_) => continue,
                Err(_) => break,
            };

            if text == "PONG" || text.is_empty() {
                continue;
            }

            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) {
                let payload = parsed.get("payload").cloned().unwrap_or(parsed.clone());
                let msg_symbol = payload.get("symbol").and_then(|v| v.as_str());

                if msg_symbol != Some(&symbol) {
                    continue;
                }

                // Try data array format first
                let p = payload
                    .get("data")
                    .and_then(|d| d.as_array())
                    .and_then(|arr| arr.last())
                    .and_then(|entry| entry.get("value"))
                    .and_then(|v| v.as_f64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
                    // Fallback: direct value field
                    .or_else(|| {
                        payload
                            .get("value")
                            .and_then(|v| {
                                v.as_f64()
                                    .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
                            })
                    });

                if let Some(px) = p {
                    store_f64(&price, px);
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs_f64();
                    store_f64(&last_update_ts, now);
                }
            }
        }

        let _ = hb.await;
        tokio::time::sleep(std::time::Duration::from_secs(backoff.min(30))).await;
        backoff = (backoff * 2).min(60);
    }
}

// ── BinanceFeed ─────────────────────────────────────────────────────────────

/// Binance bookTicker WebSocket feed for oracle lag detection.
///
/// Stores mid price = (best_bid + best_ask) / 2 in an AtomicU64 for
/// lock-free reads.
#[pyclass]
pub struct BinanceFeed {
    mid: Arc<AtomicU64>,
    last_update_ts: Arc<AtomicU64>,
    _cancel: Arc<tokio::sync::Notify>,
}

#[pymethods]
impl BinanceFeed {
    /// Create a BinanceFeed and start streaming bookTicker data.
    ///
    /// Args:
    ///     symbol: Binance symbol in lowercase (e.g. "btcusdt")
    #[new]
    fn new(symbol: String) -> Self {
        let mid = Arc::new(AtomicU64::new(0));
        let last_update_ts = Arc::new(AtomicU64::new(0));
        let cancel = Arc::new(tokio::sync::Notify::new());

        let m = mid.clone();
        let ts = last_update_ts.clone();
        let c = cancel.clone();

        RUNTIME.spawn(async move {
            binance_feed_task(symbol, m, ts, c).await;
        });

        Self {
            mid,
            last_update_ts,
            _cancel: cancel,
        }
    }

    /// Get the latest Binance mid price, or None if no data yet.
    fn mid(&self) -> Option<f64> {
        let v = load_f64(&self.mid);
        if v == 0.0 {
            None
        } else {
            Some(v)
        }
    }

    /// Timestamp (seconds since epoch) of the last update.
    fn last_update_ts(&self) -> f64 {
        load_f64(&self.last_update_ts)
    }
}

async fn binance_feed_task(
    symbol: String,
    mid: Arc<AtomicU64>,
    last_update_ts: Arc<AtomicU64>,
    _cancel: Arc<tokio::sync::Notify>,
) {
    use futures_util::StreamExt;
    use tokio_tungstenite::connect_async;

    let url = format!(
        "wss://stream.binance.com:9443/ws/{}@bookTicker",
        symbol.to_lowercase()
    );
    let mut backoff = 2u64;

    loop {
        let result = connect_async(&url).await;
        let (ws, _) = match result {
            Ok(conn) => {
                backoff = 2;
                conn
            }
            Err(e) => {
                eprintln!("[BinanceFeed] connect error: {e}");
                tokio::time::sleep(std::time::Duration::from_secs(backoff.min(60))).await;
                backoff = (backoff * 2).min(60);
                continue;
            }
        };

        let (_write, mut read) = ws.split();

        while let Some(msg) = read.next().await {
            let text = match msg {
                Ok(tokio_tungstenite::tungstenite::Message::Text(t)) => t,
                Ok(_) => continue,
                Err(_) => break,
            };

            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) {
                let best_bid = parsed
                    .get("b")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok());
                let best_ask = parsed
                    .get("a")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<f64>().ok());

                if let (Some(bid), Some(ask)) = (best_bid, best_ask) {
                    let mid_val = (bid + ask) / 2.0;
                    store_f64(&mid, mid_val);
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs_f64();
                    store_f64(&last_update_ts, now);
                }
            }
        }

        tokio::time::sleep(std::time::Duration::from_secs(backoff.min(30))).await;
        backoff = (backoff * 2).min(60);
    }
}
