use anyhow::{anyhow, Result};
use rust_decimal::prelude::*;

pub fn f64_to_decimal(val: f64) -> Result<Decimal> {
    Decimal::from_f64(val).ok_or_else(|| anyhow!("cannot convert {val} to Decimal"))
}

pub fn now_s() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

pub fn now_ms() -> i64 {
    (now_s() * 1000.0) as i64
}
