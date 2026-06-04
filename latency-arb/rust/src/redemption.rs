
use anyhow::{anyhow, bail, Context, Result};
use base64::{
    engine::general_purpose::{URL_SAFE, URL_SAFE_NO_PAD},
    Engine,
};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::time::{SystemTime, UNIX_EPOCH};

use alloy_primitives::{address, hex, keccak256, Address, B256, U256};
use alloy_signer::SignerSync;
use alloy_signer_local::PrivateKeySigner;
use alloy_sol_types::{sol, SolCall};

const CTF_ADDRESS: Address = address!("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045");
const USDC_ADDRESS: Address = address!("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174");
const PROXY_FACTORY: Address = address!("0xaB45c5A4B0c941a2F231C04C3f49182e1A254052");
const RELAY_HUB: Address = address!("0xD216153c06E857cD7f72665E0aF1d7D82172F494");

const RELAYER_URL: &str = "https://relayer-v2.polymarket.com";

const PROXY_INIT_CODE_HASH: [u8; 32] = hex!(
    "d21df8dc65880a8606f09fe0ce3df9b8869287ab0b058be05aa9e8af6330a00b"
);

const DEFAULT_RELAY_GAS_LIMIT: u64 = 500_000;

sol! {
    function redeemPositions(
        address collateralToken,
        bytes32 parentCollectionId,
        bytes32 conditionId,
        uint256[] indexSets
    );

    struct ProxyArg {
        uint8 kind;
        address to;
        uint256 value;
        bytes data;
    }
    function proxy(ProxyArg[] txns);
}

fn hmac_sign(
    secret_b64: &str,
    timestamp: &str,
    method: &str,
    path: &str,
    body: Option<&str>,
) -> Result<String> {
    let trimmed = secret_b64.trim_end_matches('=');
    let secret = URL_SAFE_NO_PAD
        .decode(trimmed)
        .context("invalid builder secret base64")?;
    let mut msg = format!("{timestamp}{method}{path}");
    if let Some(b) = body {
        msg.push_str(b);
    }
    let mut mac = Hmac::<Sha256>::new_from_slice(&secret)
        .context("hmac key init")?;
    mac.update(msg.as_bytes());
    let digest = mac.finalize().into_bytes();
    Ok(URL_SAFE.encode(digest))
}

fn now_ts() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_string())
}

fn header_set(
    req: reqwest::RequestBuilder,
    key: &str,
    passphrase: &str,
    sig: &str,
    ts: &str,
) -> reqwest::RequestBuilder {
    req.header("POLY_BUILDER_API_KEY", key)
        .header("POLY_BUILDER_PASSPHRASE", passphrase)
        .header("POLY_BUILDER_SIGNATURE", sig)
        .header("POLY_BUILDER_TIMESTAMP", ts)
}

fn derive_proxy_wallet(eoa: Address) -> Address {
    let salt: B256 = keccak256(eoa.as_slice());

    let mut create2_input = Vec::with_capacity(1 + 20 + 32 + 32);
    create2_input.push(0xff);
    create2_input.extend_from_slice(PROXY_FACTORY.as_slice());
    create2_input.extend_from_slice(salt.as_slice());
    create2_input.extend_from_slice(&PROXY_INIT_CODE_HASH);

    let hash = keccak256(&create2_input);
    let mut addr = [0u8; 20];
    addr.copy_from_slice(&hash[12..]);
    Address::from(addr)
}

fn struct_hash(
    signer_addr: Address,
    proxy_data: &[u8],
    gas_price: u64,
    relayer_fee: u64,
    gas_limit: u64,
    nonce: u64,
    relay_addr: Address,
) -> B256 {
    let mut buf = Vec::with_capacity(4 + 20 + 20 + proxy_data.len() + 32 * 4 + 20 + 20);
    buf.extend_from_slice(b"rlx:");
    buf.extend_from_slice(signer_addr.as_slice());
    buf.extend_from_slice(PROXY_FACTORY.as_slice());
    buf.extend_from_slice(proxy_data);
    buf.extend_from_slice(&U256::from(gas_price).to_be_bytes::<32>());
    buf.extend_from_slice(&U256::from(relayer_fee).to_be_bytes::<32>());
    buf.extend_from_slice(&U256::from(gas_limit).to_be_bytes::<32>());
    buf.extend_from_slice(&U256::from(nonce).to_be_bytes::<32>());
    buf.extend_from_slice(RELAY_HUB.as_slice());
    buf.extend_from_slice(relay_addr.as_slice());
    keccak256(&buf)
}

fn encode_redeem_calldata(condition_id: &str) -> Result<Vec<u8>> {
    let cid_hex = condition_id.trim_start_matches("0x");
    if cid_hex.len() != 64 {
        bail!("conditionId must be 32 hex bytes, got {}", cid_hex.len());
    }
    let cid_bytes = hex::decode(cid_hex).context("decode conditionId")?;
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&cid_bytes);

    let call = redeemPositionsCall {
        collateralToken: USDC_ADDRESS,
        parentCollectionId: B256::ZERO,
        conditionId: arr.into(),
        indexSets: vec![U256::from(1u64), U256::from(2u64)],
    };
    Ok(call.abi_encode())
}

fn encode_proxy_calldata(redeem_datas: Vec<Vec<u8>>) -> Vec<u8> {
    let txns: Vec<ProxyArg> = redeem_datas
        .into_iter()
        .map(|data| ProxyArg {
            kind: 1, // DelegateCall semantic per Polymarket proxy
            to: CTF_ADDRESS,
            value: U256::ZERO,
            data: data.into(),
        })
        .collect();
    let call = proxyCall { txns };
    call.abi_encode()
}

#[allow(dead_code)]
pub async fn redeem_position(http: &reqwest::Client, condition_id: &str) -> Result<String> {
    redeem_positions(http, std::slice::from_ref(&condition_id.to_string())).await
}

pub async fn redeem_positions(
    http: &reqwest::Client,
    condition_ids: &[String],
) -> Result<String> {
    if condition_ids.is_empty() {
        return Err(anyhow!("empty batch"));
    }
    if condition_ids.len() > 15 {
        return Err(anyhow!("max 15 per batch, got {}", condition_ids.len()));
    }
    let private_key = std::env::var("PRIVATE_KEY")
        .context("PRIVATE_KEY not set")?;
    let builder_key = std::env::var("POLY_BUILDER_API_KEY")
        .context("POLY_BUILDER_API_KEY not set")?;
    let builder_secret = std::env::var("POLY_BUILDER_SECRET")
        .context("POLY_BUILDER_SECRET not set")?;
    let builder_pass = std::env::var("POLY_BUILDER_PASSPHRASE")
        .context("POLY_BUILDER_PASSPHRASE not set")?;

    let pk_clean = private_key.trim_start_matches("0x");
    let signer: PrivateKeySigner = pk_clean.parse()
        .context("parse PRIVATE_KEY")?;
    let signer_addr = signer.address();
    let proxy_wallet = derive_proxy_wallet(signer_addr);

    let mut redeem_datas = Vec::with_capacity(condition_ids.len());
    for cid in condition_ids {
        redeem_datas.push(encode_redeem_calldata(cid)?);
    }
    let proxy_data = encode_proxy_calldata(redeem_datas);
    let proxy_data_hex = format!("0x{}", hex::encode(&proxy_data));

    // 3. GET /relay-payload to get nonce + relay address
    let payload_path = format!("/relay-payload?address={signer_addr}&type=PROXY");
    let ts = now_ts();
    let sig = hmac_sign(&builder_secret, &ts, "GET", &payload_path, None)?;
    let resp = header_set(
        http.get(format!("{RELAYER_URL}{payload_path}")),
        &builder_key,
        &builder_pass,
        &sig,
        &ts,
    )
    .send()
    .await
    .context("relay-payload request")?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        bail!("relay-payload HTTP {status}: {body}");
    }
    let payload: serde_json::Value = resp.json().await.context("parse relay-payload")?;
    let relay_address_str = payload
        .get("address")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("relay-payload missing 'address'"))?
        .to_string();
    let relay_address: Address = relay_address_str
        .parse()
        .context("parse relay address")?;
    let nonce: u64 = payload
        .get("nonce")
        .and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
        .ok_or_else(|| anyhow!("relay-payload missing 'nonce'"))?;

    // 4. Sign the struct hash with EIP-191 personal_sign
    let hash = struct_hash(
        signer_addr,
        &proxy_data,
        0,
        0,
        DEFAULT_RELAY_GAS_LIMIT,
        nonce,
        relay_address,
    );
    let sig_raw = signer
        .sign_hash_sync(&hash)
        .context("sign struct hash")?;
    let mut eip191 = Vec::with_capacity(32 + 28);
    eip191.extend_from_slice(b"\x19Ethereum Signed Message:\n32");
    eip191.extend_from_slice(hash.as_slice());
    let eip191_hash = keccak256(&eip191);
    let signature = signer
        .sign_hash_sync(&eip191_hash)
        .context("sign eip191 hash")?;
    let sig_bytes = signature.as_bytes();
    let signature_hex = format!("0x{}", hex::encode(sig_bytes));
    let _ = sig_raw;

    let body_str = format!(
        concat!(
            r#"{{"type": "PROXY", "from": "{from}", "to": "{to}", "proxyWallet": "{proxy}", "#,
            r#""data": "{data}", "nonce": "{nonce}", "signature": "{sig}", "#,
            r#""signatureParams": {{"gasPrice": "0", "gasLimit": "{gas}", "relayerFee": "0", "#,
            r#""relayHub": "{hub}", "relay": "{relay}"}}, "metadata": "Redeem winnings"}}"#
        ),
        from = signer_addr,
        to = PROXY_FACTORY,
        proxy = proxy_wallet,
        data = proxy_data_hex,
        nonce = nonce,
        sig = signature_hex,
        gas = DEFAULT_RELAY_GAS_LIMIT,
        hub = RELAY_HUB,
        relay = relay_address_str, // raw casing from Polymarket
    );

    let ts2 = now_ts();
    let sig2 = hmac_sign(&builder_secret, &ts2, "POST", "/submit", Some(&body_str))?;

    let submit = header_set(
        http.post(format!("{RELAYER_URL}/submit")),
        &builder_key,
        &builder_pass,
        &sig2,
        &ts2,
    )
    .header("Content-Type", "application/json")
    .body(body_str)
    .send()
    .await
    .context("submit request")?;

    let submit_status = submit.status();
    if !submit_status.is_success() {
        let text = submit.text().await.unwrap_or_default();
        bail!("submit HTTP {submit_status}: {text}");
    }
    let submit_body: serde_json::Value = submit.json().await.context("parse submit")?;
    let tx_id = submit_body
        .get("transactionID")
        .or_else(|| submit_body.get("transactionId"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let mut tx_hash = submit_body
        .get("transactionHash")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if tx_id.is_empty() {
        return Ok(tx_hash); // no tx id to poll; caller will verify on-chain later
    }

    // 6. Poll /transaction until state is terminal (~up to 2 min)
    for _ in 0..60 {
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        let poll_path = format!("/transaction?id={tx_id}");
        let ts3 = now_ts();
        let sig3 = hmac_sign(&builder_secret, &ts3, "GET", &poll_path, None)?;
        let poll = header_set(
            http.get(format!("{RELAYER_URL}{poll_path}")),
            &builder_key,
            &builder_pass,
            &sig3,
            &ts3,
        )
        .send()
        .await;
        let Ok(poll_resp) = poll else { continue };
        if !poll_resp.status().is_success() {
            continue;
        }
        let Ok(tx_data) = poll_resp.json::<serde_json::Value>().await else { continue };
        let (state, hash_from_poll) = if let Some(first) = tx_data.as_array().and_then(|a| a.first()) {
            (
                first.get("state").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                first
                    .get("transactionHash")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&tx_hash)
                    .to_string(),
            )
        } else {
            (
                tx_data.get("state").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                tx_data
                    .get("transactionHash")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&tx_hash)
                    .to_string(),
            )
        };
        if !hash_from_poll.is_empty() {
            tx_hash = hash_from_poll;
        }
        match state.as_str() {
            "STATE_MINED" | "STATE_CONFIRMED" => return Ok(tx_hash),
            "STATE_FAILED" => bail!("relayer tx FAILED id={tx_id} hash={tx_hash}"),
            _ => continue,
        }
    }
    bail!("relayer tx not confirmed after 120s id={tx_id} hash={tx_hash}")
}
