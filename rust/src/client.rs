use pyo3::prelude::*;
use once_cell::sync::Lazy;
use std::sync::Arc;
use tokio::runtime::Runtime;

use std::collections::HashMap;

use polyfill_rs::ClobClient;
use polyfill_rs::OrderArgs;  // simple struct from client module (token_id, price, size, side)
use polyfill_rs::types::{ApiCreds, AssetType, BalanceAllowanceParams, ExtraOrderArgs, MarketOrderArgs, OrderOptions, Side, OrderType as PolyfillOrderType};
use polyfill_rs::orders::SigType;

use crate::types;

// Global tokio runtime shared across all Rust components.
// 2 worker threads is enough for HTTP + WS; the hot path is IO-bound.
pub(crate) static RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime")
});

/// High-performance order client wrapping polyfill-rs ClobClient.
///
/// All methods are synchronous from Python's perspective — they block on
/// the internal tokio runtime. The hot-path methods (place_order,
/// place_market_order, cancel, cancel_all, get_order) wrap the
/// `RUNTIME.block_on(...)` call in `py.allow_threads(...)` so the Python
/// GIL is released during the HTTP roundtrip. Without this, even calling
/// these from `asyncio.to_thread` would still block other Python coroutines
/// because the worker thread holds the GIL throughout the network IO.
///
/// 2026-04-09 fix: previously the GIL was held during ~200-500ms HTTP
/// posts, causing the OTHER market's signal loop to freeze every time a
/// trade was placed. py.allow_threads decouples them.
#[pyclass]
pub struct OrderClient {
    inner: Arc<ClobClient>,
    /// Cached OrderOptions (tick_size + neg_risk) per token_id.
    /// These values are static for the lifetime of a market, so we
    /// fetch them once during warmup and reuse on every place_order
    /// call, saving 2 HTTP GETs (~100-200ms) per trade.
    options_cache: std::sync::Mutex<HashMap<String, OrderOptions>>,
}

fn parse_side(s: &str) -> PyResult<Side> {
    match s.to_uppercase().as_str() {
        "BUY" => Ok(Side::BUY),
        "SELL" => Ok(Side::SELL),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "side must be 'BUY' or 'SELL'",
        )),
    }
}

fn parse_order_type(s: &str) -> PolyfillOrderType {
    match s.to_uppercase().as_str() {
        "GTC" => PolyfillOrderType::GTC,
        "FOK" => PolyfillOrderType::FOK,
        "GTD" => PolyfillOrderType::GTD,
        _ => PolyfillOrderType::GTC,
    }
}

fn map_err(e: impl std::fmt::Display) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

#[pymethods]
impl OrderClient {
    /// Create a new OrderClient with L2 API-key authentication.
    ///
    /// Args:
    ///     host:           CLOB API endpoint (e.g. "https://clob.polymarket.com")
    ///     private_key:    Hex-encoded Ethereum private key
    ///     chain_id:       Blockchain network ID (137 for Polygon)
    ///     api_key:        Polymarket API key
    ///     api_secret:     Polymarket API secret
    ///     api_passphrase: Polymarket API passphrase
    ///     sig_type:       Signature type (0=EOA, 1=PolyProxy, 2=PolyGnosisSafe)
    ///     funder:         Proxy wallet address (hex string), or None for EOA
    #[new]
    #[pyo3(signature = (host, private_key, chain_id, api_key, api_secret, api_passphrase, sig_type=1, funder=None))]
    fn new(
        host: &str,
        private_key: &str,
        chain_id: u64,
        api_key: &str,
        api_secret: &str,
        api_passphrase: &str,
        sig_type: u8,
        funder: Option<&str>,
    ) -> PyResult<Self> {
        let creds = ApiCreds {
            api_key: api_key.to_string(),
            secret: api_secret.to_string(),
            passphrase: api_passphrase.to_string(),
        };

        let sig = match sig_type {
            0 => None,
            1 => Some(SigType::PolyProxy),
            2 => Some(SigType::PolyGnosisSafe),
            _ => None,
        };

        // Parse funder hex address. Address is from alloy-primitives.
        let funder_addr: Option<alloy_primitives::Address> =
            funder.and_then(|f| f.parse().ok());

        let client = ClobClient::with_l2_headers(
            host,
            private_key,
            chain_id,
            creds,
            sig,
            funder_addr,
        );

        let inner = Arc::new(client);

        Ok(Self {
            inner,
            options_cache: std::sync::Mutex::new(HashMap::new()),
        })
    }

    /// Pre-fetch and cache OrderOptions (tick_size, neg_risk) for a list
    /// of token_ids, and establish HTTP keep-alive connections.
    ///
    /// Call this once at startup with all token_ids the bot will trade.
    /// Eliminates 2 HTTP GETs per subsequent place_order/place_market_order.
    /// The GIL is released during all network calls.
    fn warmup(&self, py: Python<'_>, token_ids: Vec<String>) -> PyResult<()> {
        let client = self.inner.clone();
        let ids = token_ids.clone();

        let results = py.allow_threads(move || {
            RUNTIME.block_on(async {
                // 1. Prewarm the HTTP connection pool (TLS handshake, etc.)
                if let Err(e) = client.prewarm_connections().await {
                    eprintln!("[OrderClient] prewarm_connections failed: {}", e);
                }

                // 2. Start keep-alive pings (every 30s) to prevent idle drops
                client
                    .start_keepalive(std::time::Duration::from_secs(30))
                    .await;

                // 3. Fetch tick_size + neg_risk for each token_id
                let mut results = Vec::with_capacity(ids.len());
                for tid in &ids {
                    let tick = client.get_tick_size(tid).await;
                    let neg = client.get_neg_risk(tid).await;
                    results.push((tid.clone(), tick, neg));
                }
                results
            })
        });

        // Populate the cache with successful results
        let mut cache = self.options_cache.lock().unwrap();
        for (tid, tick_result, neg_result) in results {
            match (tick_result, neg_result) {
                (Ok(tick_size), Ok(neg_risk)) => {
                    cache.insert(
                        tid.clone(),
                        OrderOptions {
                            tick_size: Some(tick_size),
                            neg_risk: Some(neg_risk),
                            fee_rate_bps: None,
                        },
                    );
                    eprintln!(
                        "[OrderClient] warmup OK for {}: tick_size={}, neg_risk={}",
                        tid, tick_size, neg_risk
                    );
                }
                (Err(e), _) => {
                    eprintln!(
                        "[OrderClient] warmup failed for {} (tick_size): {}",
                        tid, e
                    );
                }
                (_, Err(e)) => {
                    eprintln!(
                        "[OrderClient] warmup failed for {} (neg_risk): {}",
                        tid, e
                    );
                }
            }
        }

        Ok(())
    }

    /// Place a limit order (create + sign + post in one call).
    ///
    /// Returns a dict matching py-clob-client response format:
    ///   {success, orderID, status, makingAmount?, takingAmount?, errorMsg?}
    ///
    /// order_type: "GTC" (default), "FOK", "FAK", or "GTD"
    /// expiration: Unix timestamp for GTD orders (0 = no expiration)
    #[pyo3(signature = (token_id, price, size, side, order_type="GTC", fee_rate_bps=0, expiration=0))]
    fn place_order(
        &self,
        py: Python<'_>,
        token_id: &str,
        price: f64,
        size: f64,
        side: &str,
        order_type: &str,
        fee_rate_bps: u32,
        expiration: u64,
    ) -> PyResult<PyObject> {
        let side_enum = parse_side(side)?;
        let ot = parse_order_type(order_type);

        let price_dec = types::f64_to_decimal(price)?;
        let size_dec = types::f64_to_decimal(size)?;
        let args = OrderArgs::new(
            token_id,
            price_dec,
            size_dec,
            side_enum,
        );

        let extras = if fee_rate_bps > 0 {
            Some(ExtraOrderArgs { fee_rate_bps, ..ExtraOrderArgs::default() })
        } else {
            None
        };

        // Look up cached OrderOptions to skip 2 HTTP GETs per order.
        // Falls back to None (polyfill-rs fetches on demand) if not warmed.
        let cached_opts = {
            let cache = self.options_cache.lock().unwrap();
            cache.get(token_id).cloned()
        };
        let opts_ref = cached_opts.as_ref();

        let client = self.inner.clone();
        // Release the GIL during the HTTP roundtrip so other Python
        // coroutines can run while we wait. Without this, asyncio.to_thread
        // would still block the event loop.
        let resp = py
            .allow_threads(move || {
                RUNTIME.block_on(async move {
                    let exp = if expiration > 0 { Some(expiration) } else { None };
                    let signed = client
                        .create_order(&args, exp, extras, opts_ref)
                        .await?;
                    client.post_order(signed, ot).await
                })
            })
            .map_err(map_err)?;

        types::post_response_to_py(py, &resp)
    }

    /// Place a FOK market sell order (for position exit).
    ///
    /// Returns a dict matching py-clob-client response format.
    #[pyo3(signature = (token_id, amount, side, fee_rate_bps=0))]
    fn place_market_order(
        &self,
        py: Python<'_>,
        token_id: &str,
        amount: f64,
        side: &str,
        fee_rate_bps: u32,
    ) -> PyResult<PyObject> {
        let side_enum = parse_side(side)?;

        let amount_dec = types::f64_to_decimal(amount)?;
        let args = MarketOrderArgs {
            token_id: token_id.to_string(),
            amount: amount_dec,
            side: side_enum,
        };

        let extras = if fee_rate_bps > 0 {
            Some(ExtraOrderArgs { fee_rate_bps, ..ExtraOrderArgs::default() })
        } else {
            None
        };

        // Look up cached OrderOptions to skip 2 HTTP GETs per order.
        let cached_opts = {
            let cache = self.options_cache.lock().unwrap();
            cache.get(token_id).cloned()
        };
        let opts_ref = cached_opts.as_ref();

        let client = self.inner.clone();
        let resp = py
            .allow_threads(move || {
                RUNTIME.block_on(async move {
                    let signed = client
                        .create_market_order(&args, extras, opts_ref)
                        .await?;
                    client.post_order(signed, PolyfillOrderType::FOK).await
                })
            })
            .map_err(map_err)?;

        types::post_response_to_py(py, &resp)
    }

    /// Cancel a single order by ID.
    ///
    /// Returns a dict with cancel response.
    fn cancel(&self, py: Python<'_>, order_id: &str) -> PyResult<PyObject> {
        let client = self.inner.clone();
        let oid = order_id.to_string();
        let resp = py
            .allow_threads(|| {
                RUNTIME.block_on(async move { client.cancel(&oid).await })
            })
            .map_err(map_err)?;

        types::serialize_to_py(py, &resp)
    }

    /// Cancel all open orders.
    ///
    /// Returns a dict with cancel response.
    fn cancel_all(&self, py: Python<'_>) -> PyResult<PyObject> {
        let client = self.inner.clone();
        let resp = py
            .allow_threads(|| {
                RUNTIME.block_on(async move { client.cancel_all().await })
            })
            .map_err(map_err)?;

        types::serialize_to_py(py, &resp)
    }

    /// Batch cancel multiple orders in a single HTTP call.
    ///
    /// Returns a dict with cancel response. Up to ~100 order IDs per call.
    /// Saves one HTTP round-trip vs cancelling individually.
    fn cancel_orders(&self, py: Python<'_>, order_ids: Vec<String>) -> PyResult<PyObject> {
        let client = self.inner.clone();
        let resp = py
            .allow_threads(move || {
                RUNTIME.block_on(async move { client.cancel_orders(&order_ids).await })
            })
            .map_err(map_err)?;

        types::serialize_to_py(py, &resp)
    }

    /// Batch place multiple signed orders in a single HTTP call.
    ///
    /// Takes a list of (token_id, price, size, side) tuples.
    /// Signs all orders, then POSTs them as one batch (up to 15 per call).
    /// Returns a list of response dicts.
    #[pyo3(signature = (orders, order_type="GTC", fee_rate_bps=0))]
    fn place_orders(
        &self,
        py: Python<'_>,
        orders: Vec<(String, f64, f64, String)>,  // [(token_id, price, size, side), ...]
        order_type: &str,
        fee_rate_bps: u32,
    ) -> PyResult<PyObject> {
        if orders.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err("orders list is empty"));
        }
        if orders.len() > 15 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max 15 orders per batch (Polymarket limit)"
            ));
        }

        let ot = parse_order_type(order_type);
        let extras = if fee_rate_bps > 0 {
            Some(ExtraOrderArgs { fee_rate_bps, ..ExtraOrderArgs::default() })
        } else {
            None
        };

        // Pre-resolve OrderOptions from cache for each unique token_id
        let cache = self.options_cache.lock().unwrap();
        let mut order_args_list = Vec::with_capacity(orders.len());
        for (token_id, price, size, side) in &orders {
            let side_enum = parse_side(side)?;
            let price_dec = types::f64_to_decimal(*price)?;
            let size_dec = types::f64_to_decimal(*size)?;
            let args = OrderArgs::new(token_id, price_dec, size_dec, side_enum);
            let opts = cache.get(token_id.as_str()).cloned();
            order_args_list.push((args, opts, extras.clone()));
        }
        drop(cache);

        let client = self.inner.clone();
        let responses = py
            .allow_threads(move || {
                RUNTIME.block_on(async move {
                    // Sign all orders
                    let mut signed = Vec::with_capacity(order_args_list.len());
                    for (args, opts, ext) in &order_args_list {
                        let s = client
                            .create_order(args, None, ext.clone(), opts.as_ref())
                            .await?;
                        signed.push(s);
                    }
                    // Post as a single batch
                    client.post_orders(signed, ot).await
                })
            })
            .map_err(map_err)?;

        // Convert Vec<PostOrderResponse> to a Python list of dicts
        let py_list = pyo3::types::PyList::empty(py);
        for resp in &responses {
            let py_dict = types::post_response_to_py(py, resp)?;
            py_list.append(py_dict)?;
        }
        Ok(py_list.into_any().unbind())
    }

    /// Get order status by ID.
    ///
    /// Returns a dict with fields: status, size_matched, original_size, etc.
    fn get_order(&self, py: Python<'_>, order_id: &str) -> PyResult<PyObject> {
        let client = self.inner.clone();
        let oid = order_id.to_string();
        let order = py
            .allow_threads(|| {
                RUNTIME.block_on(async move { client.get_order(&oid).await })
            })
            .map_err(map_err)?;

        types::serialize_to_py(py, &order)
    }

    /// Get USDC (collateral) balance.
    ///
    /// Returns balance as float. Handles the wei-to-USD conversion if needed
    /// (same logic as the existing query_usdc_balance Python function).
    /// Releases the GIL during the HTTP roundtrip.
    fn get_balance(&self, py: Python<'_>) -> PyResult<f64> {
        let client = self.inner.clone();
        let params = BalanceAllowanceParams {
            asset_type: Some(AssetType::COLLATERAL),
            token_id: None,
            signature_type: None,
        };
        let resp = py
            .allow_threads(|| {
                RUNTIME.block_on(async move { client.get_balance_allowance(Some(params)).await })
            })
            .map_err(map_err)?;

        let raw: f64 = resp
            .get("balance")
            .and_then(|v| {
                v.as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .or_else(|| v.as_f64())
            })
            .unwrap_or(0.0);

        // The API may return balance in micro-units (> 1M means raw wei)
        if raw > 1_000_000.0 {
            Ok(raw / 1e6)
        } else {
            Ok(raw)
        }
    }

    /// Get the wallet address derived from the private key.
    ///
    /// Returns hex-encoded address string, or None if not available.
    fn address(&self) -> Option<String> {
        self.inner.get_address()
    }
}
