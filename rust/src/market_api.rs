use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration, Timelike, Utc};
use serde_json::Value;

use crate::config::MarketConfig;

pub const GAMMA_API: &str = "https://gamma-api.polymarket.com";
pub const CLOB_HOST: &str = "https://clob.polymarket.com";
pub const CHAIN_ID: u64 = 137;

#[derive(Clone, Debug)]
pub struct Market {
    pub slug: String,
    pub up_token: String,
    pub down_token: String,
    pub end_time: DateTime<Utc>,
    pub start_time: DateTime<Utc>,
    pub condition_id: String,
}

fn ensure_list(v: &Value) -> Value {
    match v {
        Value::String(s) => serde_json::from_str(s).unwrap_or(Value::Null),
        other => other.clone(),
    }
}

fn parse_iso(s: &str) -> Result<DateTime<Utc>> {
    let s = s.replace('Z', "+00:00");
    Ok(DateTime::parse_from_rfc3339(&s)?.with_timezone(&Utc))
}

fn parse_market(event: &Value) -> Result<Market> {
    let markets = event
        .get("markets")
        .and_then(|v| v.as_array())
        .ok_or_else(|| anyhow!("no markets array"))?;
    let m = markets.first().ok_or_else(|| anyhow!("empty markets"))?;

    let slug = event.get("slug").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let end_time = parse_iso(m.get("endDate").and_then(|v| v.as_str()).unwrap_or_default())?;
    let start_time = parse_iso(
        m.get("eventStartTime")
            .and_then(|v| v.as_str())
            .or_else(|| m.get("startDate").and_then(|v| v.as_str()))
            .unwrap_or_default(),
    )
    .unwrap_or(end_time - Duration::seconds(900));
    let condition_id = m
        .get("conditionId")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let tokens = m.get("clobTokenIds").map(ensure_list).unwrap_or(Value::Null);
    let ids: Vec<String> = tokens
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    if ids.len() < 2 {
        return Err(anyhow!("need 2 clobTokenIds, got {}", ids.len()));
    }

    let outcomes = m.get("outcomes").map(ensure_list).unwrap_or(Value::Null);
    let outcome_strs: Vec<String> = outcomes
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let up_idx = outcome_strs
        .iter()
        .position(|o| o.eq_ignore_ascii_case("Up"))
        .unwrap_or(0);
    let down_idx = 1 - up_idx;

    Ok(Market {
        slug,
        up_token: ids[up_idx].clone(),
        down_token: ids[down_idx].clone(),
        end_time,
        start_time,
        condition_id,
    })
}

async fn try_slug(client: &reqwest::Client, slug: &str) -> Option<Value> {
    let url = format!("{GAMMA_API}/events");
    let resp = client.get(&url).query(&[("slug", slug)]).send().await.ok()?;
    let data: Value = resp.json().await.ok()?;
    let arr = data.as_array()?;
    arr.first().cloned()
}

pub async fn find_market(client: &reqwest::Client, config: &MarketConfig) -> Result<Market> {
    let now = Utc::now();
    let align = config.window_align_m;
    let minute = (now.minute() as i64 / align) * align;
    let window_start = now
        .with_minute(minute as u32)
        .unwrap()
        .with_second(0)
        .unwrap()
        .with_nanosecond(0)
        .unwrap();

    for offset in [0, -align, align, -2 * align, 2 * align] {
        let candidate = window_start + Duration::minutes(offset);
        let ts = candidate.timestamp();
        let slug = format!("{}-{}", config.slug_prefix, ts);
        if let Some(event) = try_slug(client, &slug).await {
            if let Ok(market) = parse_market(&event) {
                if now < market.end_time || market.start_time > now {
                    return Ok(market);
                }
            }
        }
    }

    // Fallback: search active up-or-down markets
    let url = format!("{GAMMA_API}/events");
    let resp = client
        .get(&url)
        .query(&[
            ("active", "true"),
            ("closed", "false"),
            ("tag_slug", "up-or-down"),
            ("limit", "100"),
        ])
        .send()
        .await?;
    let data: Value = resp.json().await?;
    if let Some(arr) = data.as_array() {
        for event in arr {
            let slug = event.get("slug").and_then(|v| v.as_str()).unwrap_or("");
            if !slug.contains(config.slug_prefix) {
                continue;
            }
            if let Ok(market) = parse_market(event) {
                if now < market.end_time {
                    return Ok(market);
                }
            }
        }
    }

    Err(anyhow!("no active market for {}", config.slug_prefix))
}

/// Poll Gamma until the market is fully resolved, returning 1 (UP won) or 0 (DOWN won).
///
/// Requires all three:
///   - closed = true
///   - umaResolutionStatus = "resolved"
///   - outcomePrices is exactly {"0","1"} (or {0.0, 1.0})
pub async fn poll_resolution(
    client: &reqwest::Client,
    slug: &str,
    max_attempts: u32,
    delay_s: f64,
) -> Option<u8> {
    for _ in 0..max_attempts {
        if let Ok(resp) = client
            .get(format!("{GAMMA_API}/events"))
            .query(&[("slug", slug)])
            .send()
            .await
        {
            if let Ok(data) = resp.json::<Value>().await {
                if let Some(event) = data.as_array().and_then(|a| a.first()) {
                    if let Some(market) = event
                        .get("markets")
                        .and_then(|v| v.as_array())
                        .and_then(|a| a.first())
                    {
                        let closed = market.get("closed").and_then(|v| v.as_bool()).unwrap_or(false);
                        let uma = market
                            .get("umaResolutionStatus")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        if closed && uma == "resolved" {
                            let prices = ensure_list(
                                market.get("outcomePrices").unwrap_or(&Value::Null),
                            );
                            let outcomes = ensure_list(
                                market.get("outcomes").unwrap_or(&Value::Null),
                            );
                            if let (Some(p_arr), Some(o_arr)) =
                                (prices.as_array(), outcomes.as_array())
                            {
                                let pfloats: Vec<f64> = p_arr
                                    .iter()
                                    .filter_map(|v| {
                                        v.as_str().and_then(|s| s.parse().ok()).or_else(|| v.as_f64())
                                    })
                                    .collect();
                                if pfloats.len() == 2 {
                                    let mut sorted = pfloats.clone();
                                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                    if sorted == [0.0, 1.0] {
                                        let up_idx = o_arr
                                            .iter()
                                            .position(|o| {
                                                o.as_str()
                                                    .map(|s| s.eq_ignore_ascii_case("Up"))
                                                    .unwrap_or(false)
                                            })
                                            .unwrap_or(0);
                                        return Some(if pfloats[up_idx] >= 0.5 { 1 } else { 0 });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_secs_f64(delay_s)).await;
    }
    None
}

