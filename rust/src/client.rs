use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use polyfill_rs::orders::SigType;
use polyfill_rs::types::{
    ApiCreds, AssetType, BalanceAllowanceParams, ExtraOrderArgs, MarketOrderArgs, OrderOptions,
    OrderType as PolyOrderType, Side,
};
use polyfill_rs::{ClobClient, OrderArgs, PostOrderResponse};

use crate::types::f64_to_decimal;

pub struct OrderClient {
    inner: Arc<ClobClient>,
    options_cache: Mutex<HashMap<String, OrderOptions>>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct OrderResponse {
    pub success: bool,
    pub order_id: String,
    pub status: String,
    pub making_amount: Option<String>,
    pub taking_amount: Option<String>,
    pub error_msg: Option<String>,
    pub transaction_hashes: Vec<String>,
}

fn response_from(resp: &PostOrderResponse) -> OrderResponse {
    let status = if !resp.success {
        "REJECTED"
    } else if resp.taking_amount.map_or(false, |v| !v.is_zero()) {
        "MATCHED"
    } else {
        "LIVE"
    }
    .to_string();
    OrderResponse {
        success: resp.success,
        order_id: resp.order_id.clone(),
        status,
        making_amount: resp.making_amount.map(|v| v.to_string()),
        taking_amount: resp.taking_amount.map(|v| v.to_string()),
        error_msg: resp.error_msg.clone(),
        transaction_hashes: resp.transaction_hashes.clone(),
    }
}

impl OrderClient {
    pub fn new(
        host: &str,
        private_key: &str,
        chain_id: u64,
        api_key: &str,
        api_secret: &str,
        api_passphrase: &str,
        sig_type: u8,
        funder: Option<&str>,
    ) -> Self {
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

        Self {
            inner: Arc::new(client),
            options_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Pre-fetch tick_size + neg_risk for every token and open HTTP keep-alive.
    /// Without this, each place_order costs 2 extra HTTP GETs (~150ms).
    ///
    /// Collects results before taking the cache lock so the future stays
    /// `Send` — the rotation task spawns this onto tokio, which requires it.
    pub async fn warmup(&self, token_ids: &[String]) -> Result<()> {
        if let Err(e) = self.inner.prewarm_connections().await {
            eprintln!("[OrderClient] prewarm failed: {e}");
        }
        self.inner
            .start_keepalive(std::time::Duration::from_secs(30))
            .await;

        let mut results = Vec::with_capacity(token_ids.len());
        for tid in token_ids {
            let tick = self.inner.get_tick_size(tid).await;
            let neg = self.inner.get_neg_risk(tid).await;
            results.push((tid.clone(), tick, neg));
        }

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
                        "[OrderClient] warmup OK for {tid}: tick={tick_size}, neg_risk={neg_risk}"
                    );
                }
                (Err(e), _) | (_, Err(e)) => {
                    eprintln!("[OrderClient] warmup failed for {tid}: {e}");
                }
            }
        }
        Ok(())
    }

    /// FOK market-buy for taker arb entries.
    ///
    /// `fee_rate_bps` is a MAX-fee commitment in the signed order, NOT a fee
    /// we pay. Polymarket charges the market's actual dynamic fee (~1-3%
    /// typically), bounded above by this declared max. Setting it too low
    /// (< current market's taker fee) rejects the order with a 400. 1000 bps
    /// (10%) is a safe cap that covers all current Polymarket taker fees.
    pub async fn place_market_order(
        &self,
        token_id: &str,
        amount: f64,
        side: &str,
        fee_rate_bps: u32,
    ) -> Result<OrderResponse> {
        let side_enum = match side.to_uppercase().as_str() {
            "BUY" => Side::BUY,
            "SELL" => Side::SELL,
            _ => return Err(anyhow!("side must be BUY or SELL")),
        };
        let amount_dec = f64_to_decimal(amount)?;
        let args = MarketOrderArgs {
            token_id: token_id.to_string(),
            amount: amount_dec,
            side: side_enum,
        };
        let opts = {
            let cache = self.options_cache.lock().unwrap();
            cache.get(token_id).cloned()
        };
        let extras = if fee_rate_bps > 0 {
            Some(ExtraOrderArgs { fee_rate_bps, ..ExtraOrderArgs::default() })
        } else {
            None
        };
        let signed = self
            .inner
            .create_market_order(&args, extras, opts.as_ref())
            .await
            .map_err(|e| anyhow!("create_market_order: {e}"))?;
        let resp = self
            .inner
            .post_order(signed, PolyOrderType::FOK)
            .await
            .map_err(|e| anyhow!("post_order: {e}"))?;
        Ok(response_from(&resp))
    }

    /// Limit order with configurable order type (default GTC).
    #[allow(dead_code)]
    pub async fn place_order(
        &self,
        token_id: &str,
        price: f64,
        size: f64,
        side: &str,
        order_type: &str,
    ) -> Result<OrderResponse> {
        let side_enum = match side.to_uppercase().as_str() {
            "BUY" => Side::BUY,
            "SELL" => Side::SELL,
            _ => return Err(anyhow!("side must be BUY or SELL")),
        };
        let ot = match order_type.to_uppercase().as_str() {
            "FOK" => PolyOrderType::FOK,
            "GTD" => PolyOrderType::GTD,
            _ => PolyOrderType::GTC,
        };
        let price_dec = f64_to_decimal(price)?;
        let size_dec = f64_to_decimal(size)?;
        let args = OrderArgs::new(token_id, price_dec, size_dec, side_enum);
        let opts = {
            let cache = self.options_cache.lock().unwrap();
            cache.get(token_id).cloned()
        };
        let extras: Option<ExtraOrderArgs> = None;
        let signed = self
            .inner
            .create_order(&args, None, extras, opts.as_ref())
            .await
            .map_err(|e| anyhow!("create_order: {e}"))?;
        let resp = self
            .inner
            .post_order(signed, ot)
            .await
            .map_err(|e| anyhow!("post_order: {e}"))?;
        Ok(response_from(&resp))
    }

    pub async fn get_balance(&self) -> Result<f64> {
        let params = BalanceAllowanceParams {
            asset_type: Some(AssetType::COLLATERAL),
            token_id: None,
            signature_type: None,
        };
        let resp = self
            .inner
            .get_balance_allowance(Some(params))
            .await
            .map_err(|e| anyhow!("get_balance_allowance: {e}"))?;
        let raw: f64 = resp
            .get("balance")
            .and_then(|v| {
                v.as_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .or_else(|| v.as_f64())
            })
            .unwrap_or(0.0);
        Ok(if raw > 1_000_000.0 { raw / 1e6 } else { raw })
    }

    pub fn address(&self) -> Option<String> {
        self.inner.get_address()
    }
}
