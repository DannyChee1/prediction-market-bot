use pyo3::prelude::*;
use once_cell::sync::Lazy;
use std::sync::Arc;
use tokio::runtime::Runtime;

use polyfill_rs::ClobClient;
use polyfill_rs::OrderArgs;  // simple struct from client module (token_id, price, size, side)
use polyfill_rs::types::{ApiCreds, AssetType, BalanceAllowanceParams, ExtraOrderArgs, MarketOrderArgs, Side, OrderType as PolyfillOrderType};
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
/// the internal tokio runtime. This is safe because the Python callers
/// run these in asyncio.to_thread(), so blocking doesn't stall the event loop.
#[pyclass]
pub struct OrderClient {
    inner: Arc<ClobClient>,
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

        Ok(Self {
            inner: Arc::new(client),
        })
    }

    /// Place a GTC limit order (create + sign + post in one call).
    ///
    /// Returns a dict matching py-clob-client response format:
    ///   {success, orderID, status, makingAmount?, takingAmount?, errorMsg?}
    #[pyo3(signature = (token_id, price, size, side, order_type="GTC", fee_rate_bps=0))]
    fn place_order(
        &self,
        py: Python<'_>,
        token_id: &str,
        price: f64,
        size: f64,
        side: &str,
        order_type: &str,
        fee_rate_bps: u32,
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

        let client = self.inner.clone();
        let resp = RUNTIME
            .block_on(async move {
                let signed = client
                    .create_order(&args, None, extras, None)
                    .await?;
                client.post_order(signed, ot).await
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

        let client = self.inner.clone();
        let resp = RUNTIME
            .block_on(async move {
                let signed = client
                    .create_market_order(&args, extras, None)
                    .await?;
                client.post_order(signed, PolyfillOrderType::FOK).await
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
        let resp = RUNTIME
            .block_on(async move { client.cancel(&oid).await })
            .map_err(map_err)?;

        types::serialize_to_py(py, &resp)
    }

    /// Cancel all open orders.
    ///
    /// Returns a dict with cancel response.
    fn cancel_all(&self, py: Python<'_>) -> PyResult<PyObject> {
        let client = self.inner.clone();
        let resp = RUNTIME
            .block_on(async move { client.cancel_all().await })
            .map_err(map_err)?;

        types::serialize_to_py(py, &resp)
    }

    /// Get order status by ID.
    ///
    /// Returns a dict with fields: status, size_matched, original_size, etc.
    fn get_order(&self, py: Python<'_>, order_id: &str) -> PyResult<PyObject> {
        let client = self.inner.clone();
        let oid = order_id.to_string();
        let order = RUNTIME
            .block_on(async move { client.get_order(&oid).await })
            .map_err(map_err)?;

        types::serialize_to_py(py, &order)
    }

    /// Get USDC (collateral) balance.
    ///
    /// Returns balance as float. Handles the wei-to-USD conversion if needed
    /// (same logic as the existing query_usdc_balance Python function).
    fn get_balance(&self) -> PyResult<f64> {
        let client = self.inner.clone();
        let params = BalanceAllowanceParams {
            asset_type: Some(AssetType::COLLATERAL),
            token_id: None,
            signature_type: None,
        };
        let resp = RUNTIME
            .block_on(async move { client.get_balance_allowance(Some(params)).await })
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
