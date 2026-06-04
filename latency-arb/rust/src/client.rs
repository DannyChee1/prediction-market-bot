use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Result};
use polyfill_rs::types::{
    ApiCredentials, AssetType, BalanceAllowanceParams, ClientConfig, CreateOrderOptions,
    MarketOrderArgs, OrderType as PolyOrderType, PostOrderOptions, Side,
};
use polyfill_rs::{ClobClient, OrderArgs};

use crate::types::f64_to_decimal;

pub struct OrderClient {
    inner: Arc<ClobClient>,
    options_cache: Mutex<HashMap<String, CreateOrderOptions>>,
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

fn response_from(resp: &polyfill_rs::types::PostOrderResponse) -> OrderResponse {
    let making = if resp.making_amount.is_empty() { None } else { Some(resp.making_amount.clone()) };
    let taking = if resp.taking_amount.is_empty() { None } else { Some(resp.taking_amount.clone()) };
    let err = if resp.error_msg.is_empty() { None } else { Some(resp.error_msg.clone()) };
    OrderResponse {
        success: resp.success,
        order_id: resp.order_id.clone(),
        status: resp.status.clone(),
        making_amount: making,
        taking_amount: taking,
        error_msg: err,
        transaction_hashes: resp.transactions_hashes.clone(),
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
    ) -> Result<Self> {
        let creds = ApiCredentials {
            api_key: api_key.to_string(),
            secret: api_secret.to_string(),
            passphrase: api_passphrase.to_string(),
        };
        let config = ClientConfig {
            base_url: host.to_string(),
            chain: chain_id,
            private_key: Some(private_key.to_string()),
            api_credentials: Some(creds),
            signature_type: Some(sig_type),
            funder: funder.map(|s| s.to_string()),
            ..ClientConfig::default()
        };
        let client = ClobClient::from_config(config)
            .map_err(|e| anyhow!("ClobClient::from_config: {e}"))?;
        Ok(Self {
            inner: Arc::new(client),
            options_cache: Mutex::new(HashMap::new()),
        })
    }

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
                        CreateOrderOptions {
                            tick_size: Some(tick_size),
                            neg_risk: Some(neg_risk),
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

    pub async fn place_market_order(
        &self,
        token_id: &str,
        amount: f64,
        side: &str,
        _fee_rate_bps: u32,
    ) -> Result<OrderResponse> {
        let side_enum = match side.to_uppercase().as_str() {
            "BUY" => Side::BUY,
            "SELL" => Side::SELL,
            _ => return Err(anyhow!("side must be BUY or SELL")),
        };
        let amount_dec = f64_to_decimal(amount)?;
        let args = MarketOrderArgs::new(token_id, amount_dec, side_enum, PolyOrderType::FOK);
        let opts = {
            let cache = self.options_cache.lock().unwrap();
            cache.get(token_id).cloned()
        };
        let signed = self
            .inner
            .create_market_order(&args, opts.as_ref())
            .await
            .map_err(|e| anyhow!("create_market_order: {e}"))?;
        let post_opts = PostOrderOptions { order_type: PolyOrderType::FOK, ..PostOrderOptions::default() };
        let resp = self
            .inner
            .post_order(signed, Some(&post_opts))
            .await
            .map_err(|e| anyhow!("post_order: {e}"))?;
        Ok(response_from(&resp))
    }

    pub async fn place_limit_fok(
        &self,
        token_id: &str,
        max_price: f64,
        amount: f64,
        side: &str,
    ) -> Result<OrderResponse> {
        let side_enum = match side.to_uppercase().as_str() {
            "BUY" => Side::BUY,
            "SELL" => Side::SELL,
            _ => return Err(anyhow!("side must be BUY or SELL")),
        };
        let amount_dec = f64_to_decimal(amount)?;
        let limit_dec = f64_to_decimal(max_price.clamp(0.01, 0.99))?;
        let mut args = MarketOrderArgs::new(token_id, amount_dec, side_enum, PolyOrderType::FOK);
        args.price_limit = Some(limit_dec);
        let opts = {
            let cache = self.options_cache.lock().unwrap();
            cache.get(token_id).cloned()
        };
        let signed = match self.inner.create_market_order(&args, opts.as_ref()).await {
            Ok(s) => s,
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("price_limit") || msg.contains("violates") {
                    return Ok(OrderResponse {
                        success: false,
                        order_id: String::new(),
                        status: "PRICE_LIMIT_REJECT".to_string(),
                        making_amount: None,
                        taking_amount: None,
                        error_msg: Some(msg),
                        transaction_hashes: vec![],
                    });
                }
                return Err(anyhow!("create_market_order: {e}"));
            }
        };
        let post_opts = PostOrderOptions { order_type: PolyOrderType::FOK, ..PostOrderOptions::default() };
        let resp = self
            .inner
            .post_order(signed, Some(&post_opts))
            .await
            .map_err(|e| anyhow!("post_order: {e}"))?;
        Ok(response_from(&resp))
    }

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
        let signed = self
            .inner
            .create_order(&args, opts.as_ref())
            .await
            .map_err(|e| anyhow!("create_order: {e}"))?;
        let post_opts = PostOrderOptions { order_type: ot, ..PostOrderOptions::default() };
        let resp = self
            .inner
            .post_order(signed, Some(&post_opts))
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
