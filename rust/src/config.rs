#[derive(Clone, Debug)]
pub struct MarketConfig {
    pub slug_prefix: &'static str,
    pub chainlink_symbol: &'static str,
    pub binance_symbol: &'static str,
    pub display_name: &'static str,
    pub window_align_m: i64,
    pub window_duration_s: i64,
}

pub const BTC_15M: MarketConfig = MarketConfig {
    slug_prefix: "btc-updown-15m",
    chainlink_symbol: "btc/usd",
    binance_symbol: "btcusdt",
    display_name: "BTC 15m",
    window_align_m: 15,
    window_duration_s: 900,
};

pub const BTC_5M: MarketConfig = MarketConfig {
    slug_prefix: "btc-updown-5m",
    chainlink_symbol: "btc/usd",
    binance_symbol: "btcusdt",
    display_name: "BTC 5m",
    window_align_m: 5,
    window_duration_s: 300,
};

pub fn resolve(market: &str) -> Vec<MarketConfig> {
    match market {
        "btc" => vec![BTC_15M, BTC_5M],
        "btc_15m" => vec![BTC_15M],
        "btc_5m" => vec![BTC_5M],
        _ => vec![],
    }
}

#[derive(Clone, Debug)]
pub struct ArbParams {
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
    /// Trend-confirmation window. Require that the Binance move over
    /// `trend_window_s` agrees in direction with the 2s trigger move,
    /// and is at least `trend_min_ratio` of its magnitude.
    pub trend_window_s: f64,
    pub trend_min_ratio: f64,
    /// Cross-venue consensus: require Coinbase to have moved at least
    /// `crossvenue_min_ratio * delta_usd` in the same direction as the
    /// Binance trigger over the same `window_s`.
    pub crossvenue_min_ratio: f64,
}

impl Default for ArbParams {
    fn default() -> Self {
        Self {
            delta_usd: 30.0,
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
        }
    }
}
