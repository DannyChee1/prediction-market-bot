/// Underlying crypto asset. Feeds (Binance/Coinbase/Chainlink) are keyed by
/// asset — markets with the same asset share one instance of each feed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Asset {
    Btc,
    Eth,
}

impl Asset {
    pub const fn binance_symbol(self) -> &'static str {
        match self {
            Self::Btc => "btcusdt",
            Self::Eth => "ethusdt",
        }
    }
    pub const fn coinbase_product(self) -> &'static str {
        match self {
            Self::Btc => "BTC-USD",
            Self::Eth => "ETH-USD",
        }
    }
    pub const fn chainlink_symbol(self) -> &'static str {
        match self {
            Self::Btc => "btc/usd",
            Self::Eth => "eth/usd",
        }
    }
    pub const fn short_name(self) -> &'static str {
        match self {
            Self::Btc => "BTC",
            Self::Eth => "ETH",
        }
    }
}

#[derive(Clone, Debug)]
pub struct MarketConfig {
    pub asset: Asset,
    pub slug_prefix: &'static str,
    pub display_name: &'static str,
    pub window_align_m: i64,
    pub window_duration_s: i64,
}

pub const BTC_15M: MarketConfig = MarketConfig {
    asset: Asset::Btc,
    slug_prefix: "btc-updown-15m",
    display_name: "BTC 15m",
    window_align_m: 15,
    window_duration_s: 900,
};

pub const BTC_5M: MarketConfig = MarketConfig {
    asset: Asset::Btc,
    slug_prefix: "btc-updown-5m",
    display_name: "BTC 5m",
    window_align_m: 5,
    window_duration_s: 300,
};

pub const ETH_15M: MarketConfig = MarketConfig {
    asset: Asset::Eth,
    slug_prefix: "eth-updown-15m",
    display_name: "ETH 15m",
    window_align_m: 15,
    window_duration_s: 900,
};

pub const ETH_5M: MarketConfig = MarketConfig {
    asset: Asset::Eth,
    slug_prefix: "eth-updown-5m",
    display_name: "ETH 5m",
    window_align_m: 5,
    window_duration_s: 300,
};

/// Resolve a `--market` flag to one or more market configs.
/// `all` runs every supported market in one process against a single bankroll.
pub fn resolve(market: &str) -> Vec<MarketConfig> {
    match market {
        "btc" => vec![BTC_15M, BTC_5M],
        "btc_15m" => vec![BTC_15M],
        "btc_5m" => vec![BTC_5M],
        "eth" => vec![ETH_15M, ETH_5M],
        "eth_15m" => vec![ETH_15M],
        "eth_5m" => vec![ETH_5M],
        "all" => vec![BTC_15M, BTC_5M, ETH_15M, ETH_5M],
        _ => vec![],
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct ArbParams {
    /// Fixed threshold (used when adaptive is disabled). Also serves as
    /// the cross-venue consensus base: Coinbase must move >= crossvenue_min_ratio × delta_usd.
    pub delta_usd: f64,
    pub window_s: f64,
    pub cooldown_s: f64,
    pub book_stale_ms: f64,
    pub min_ask: f64,
    pub max_ask: f64,
    pub min_tau_s: f64,
    pub size_usd: f64,
    pub max_bankroll_frac: f64,
    pub min_order_shares: f64,
    pub trend_window_s: f64,
    pub trend_min_ratio: f64,
    pub crossvenue_min_ratio: f64,
    /// Adaptive threshold: delta_required = max(delta_floor, min(delta_cap, sigma_k × σ_2s)).
    /// If delta_floor == delta_cap, adaptive is effectively disabled.
    pub delta_floor: f64,
    pub delta_cap: f64,
    pub sigma_k: f64,
    /// Max open positions per market window slug.
    pub max_positions_per_window: usize,
    /// Per-slot ramp on the Binance-delta threshold. Each existing slot on
    /// the current window inflates the effective threshold by this factor:
    ///   `threshold_effective = base × (1 + ramp_per_slot × slots_used)`
    /// Set to 0 to disable (all slots use the same threshold).
    pub ramp_per_slot: f64,
    /// UTC hour gate (start_inclusive, end_exclusive). None = always allow.
    /// Set on ETH to [12, 18) — data showed ~100% WR in that window and
    /// heavy losses off-hours (thin liquidity + fat-tail adverse selection).
    /// Hours wrap around 00z if start > end (e.g. (22, 6) = 22:00-06:00).
    pub allowed_hours_utc: Option<(u8, u8)>,
    /// Reject fires whose Binance z-score exceeds this cap. None = no cap.
    /// Set on ETH because ETH z>8 fires are dominated by Binance spoofs /
    /// thin-book wicks that revert (opposite pattern vs BTC, where z8+ is
    /// the strongest signal).
    pub z_cap: Option<f64>,
}

impl Default for ArbParams {
    fn default() -> Self {
        Self {
            delta_usd: 25.0,
            window_s: 2.0,
            cooldown_s: 4.0,
            book_stale_ms: 600.0,
            min_ask: 0.15,
            max_ask: 0.85,
            min_tau_s: 30.0,
            size_usd: 10.0,
            max_bankroll_frac: 0.05,
            min_order_shares: 5.0,
            trend_window_s: 15.0,
            trend_min_ratio: 0.5,
            crossvenue_min_ratio: 0.5,
            delta_floor: 10.0,
            delta_cap: 25.0,
            sigma_k: 2.5,
            max_positions_per_window: 2,
            ramp_per_slot: 0.5,
            allowed_hours_utc: None,
            z_cap: None,
        }
    }
}
