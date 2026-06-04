"""Shared market configuration for BTC/ETH/SOL/XRP Up/Down markets."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class MarketConfig:
    slug_prefix: str       # "btc-updown-15m" / "btc-updown-5m"
    chainlink_symbol: str  # "btc/usd" / "eth/usd"
    data_subdir: str       # "btc_15m" / "eth_15m" / "btc_5m" / "eth_5m"
    display_name: str      # "BTC 15m" / "BTC 5m"
    window_duration_s: float = 900.0   # 900 for 15m, 300 for 5m
    window_align_m: int = 15           # minute alignment for find_market
    max_sigma: float = 8e-05           # per-second sigma ceiling
    min_sigma: float = 1e-6            # per-second sigma floor
    max_z: float = 3.0
    min_trade_sigma: float = 0.0
    binance_symbol: str = ""           # e.g. "btcusdt" for Binance bookTicker
    tail_mode: str = "normal"          # "normal", "student_t", or "kou"
    tail_nu_default: float = 20.0      # Student-t degrees of freedom (ignored for normal)
    kou_lambda: float = 0.007          # Kou jump intensity per observation
    kou_p_up: float = 0.51             # Kou upward jump probability
    kou_eta1: float = 1100.0           # Kou upward jump rate (1/mean_size)
    kou_eta2: float = 1100.0           # Kou downward jump rate
    hawkes_params: tuple[float, float, float, float] | None = None
    min_entry_z: float = 0.5           # Minimum |z| to enter
    min_entry_price: float = 0.25      # Minimum contract price to enter
    edge_threshold: float = 0.06       # Minimum edge to enter
    market_blend: float = 0.0          # Blend p_model with contract mid (0=off)
    max_trades_per_window: int = 1
    same_direction_stacking_only: bool = True
    max_model_market_disagreement: float = 1.0
    sigma_estimator: str = "yz"
    # Stale-feature gates: each is a HARD SKIP (not a threshold widen).
    # All are live-only — backtest never populates the *_age_ms ctx fields.
    max_book_age_ms: float | None = None        # Skip if book WS older than this
    max_chainlink_age_ms: float | None = None   # Skip if chainlink price older
    max_binance_age_ms: float | None = None     # Skip if binance trade older
    max_trade_tape_age_ms: float | None = None  # Skip if trade tape older
    stale_quote_mode: bool = False
    stale_threshold: float = 0.03   # minimum fair-ask edge after fees

MARKET_CONFIGS: dict[str, MarketConfig] = {
    "btc": MarketConfig(
        slug_prefix="btc-updown-15m",
        chainlink_symbol="btc/usd",
        data_subdir="btc_15m",
        display_name="BTC 15m",
        window_duration_s=900.0,
        window_align_m=15,
        min_sigma=2e-05,
        max_sigma=4e-04,
        binance_symbol="btcusdt",
        tail_mode="kou",
        tail_nu_default=20.0,
        kou_lambda=0.0684,
        kou_p_up=0.5013,
        kou_eta1=4504.3,
        kou_eta2=4509.6,
        hawkes_params=(0.011611, 0.0300, 0.0500, 3.0),
        max_trades_per_window=1,
        min_entry_z=0.15,
        max_model_market_disagreement=0.30,
        # 2026-04-11 Test #3: calm-market σ floor. See dataclass comment.
        min_trade_sigma=2.5e-5,
        market_blend=0.5,
        max_chainlink_age_ms=60_000.0,   # chainlink heartbeat ~30s; 60s = 2 misses
        max_binance_age_ms=2_000.0,      # binance bookTicker is 100ms; 2s = severe lag
        max_trade_tape_age_ms=10_000.0,  # trade tape is bursty; 10s of silence is ok
    ),
    "eth": MarketConfig(
        slug_prefix="eth-updown-15m",
        chainlink_symbol="eth/usd",
        data_subdir="eth_15m",
        display_name="ETH 15m",
        window_duration_s=900.0,
        window_align_m=15,
        max_sigma=1.0e-04,
        binance_symbol="ethusdt",
        tail_mode="student_t",
        tail_nu_default=13.0,
    ),
    "btc_5m": MarketConfig(
        slug_prefix="btc-updown-5m",
        chainlink_symbol="btc/usd",
        data_subdir="btc_5m",
        display_name="BTC 5m",
        window_duration_s=300.0,
        window_align_m=5,
        min_sigma=2e-05,
        max_sigma=2e-04,
        binance_symbol="btcusdt",
        tail_mode="kou",
        tail_nu_default=20.0,
        kou_lambda=0.0758,
        kou_p_up=0.5014,
        kou_eta1=4884.8,
        kou_eta2=4867.7,
        hawkes_params=(0.023712, 0.0200, 0.0500, 3.0),
        min_entry_z=0.15,
        min_entry_price=0.20,       # avoid deep OTM tail (was 0.10)
        edge_threshold=0.06,
        market_blend=0.3,           # pull p_model toward market mid
        max_model_market_disagreement=0.30,
        # 2026-04-11 Test #3: calm-market σ floor. See dataclass comment.
        min_trade_sigma=2.5e-5,
        max_book_age_ms=5000.0,
        max_chainlink_age_ms=30_000.0,   # 30s = 1 missed chainlink heartbeat
        max_binance_age_ms=1_500.0,      # tighter than 15m — 5m bot reacts faster
        max_trade_tape_age_ms=8_000.0,
    ),
    "eth_5m": MarketConfig(
        slug_prefix="eth-updown-5m",
        chainlink_symbol="eth/usd",
        data_subdir="eth_5m",
        display_name="ETH 5m",
        window_duration_s=300.0,
        window_align_m=5,
        max_sigma=1.0e-04,
        binance_symbol="ethusdt",
        tail_mode="student_t",
        tail_nu_default=15.0,
    ),
    "sol": MarketConfig(
        slug_prefix="sol-updown-15m",
        chainlink_symbol="sol/usd",
        data_subdir="sol_15m",
        display_name="SOL 15m",
        window_duration_s=900.0,
        window_align_m=15,
        max_sigma=1.2e-04,
        binance_symbol="solusdt",
        # Live runs SOL with no max_z override → class default 1.0.
        # Pinned here so backtest matches live. Revisit if SOL is re-tuned.
        max_z=1.0,
    ),
    "sol_5m": MarketConfig(
        slug_prefix="sol-updown-5m",
        chainlink_symbol="sol/usd",
        data_subdir="sol_5m",
        display_name="SOL 5m",
        window_duration_s=300.0,
        window_align_m=5,
        max_sigma=1.2e-04,
        binance_symbol="solusdt",
        max_z=1.0,
    ),
    "xrp": MarketConfig(
        slug_prefix="xrp-updown-15m",
        chainlink_symbol="xrp/usd",
        data_subdir="xrp_15m",
        display_name="XRP 15m",
        window_duration_s=900.0,
        window_align_m=15,
        max_sigma=1.2e-04,
        binance_symbol="xrpusdt",
        max_z=1.0,
    ),
    "xrp_5m": MarketConfig(
        slug_prefix="xrp-updown-5m",
        chainlink_symbol="xrp/usd",
        data_subdir="xrp_5m",
        display_name="XRP 5m",
        window_duration_s=300.0,
        window_align_m=5,
        max_sigma=1.2e-04,
        binance_symbol="xrpusdt",
        max_z=1.0,
    ),
    "btc_1h": MarketConfig(
        slug_prefix="bitcoin-up-or-down",
        chainlink_symbol="btc/usd",
        data_subdir="btc_1h",
        display_name="BTC 1h",
        window_duration_s=3600.0,
        window_align_m=60,
        min_sigma=2e-05,
        max_z=1.0,
        max_sigma=1e-03,
        binance_symbol="btcusdt",
        tail_mode="kou",
        min_entry_z=0.10,
        # 2026-04-10: raised from default 0.25 to 0.40. Entries at <$0.45
        # had 35% WR — deep OTM contrarian bets that fail on 1h.
        min_entry_price=0.40,
        market_blend=0.5,
        max_model_market_disagreement=0.30,
        max_trades_per_window=20,
        edge_threshold=0.04,
        max_book_age_ms=10_000.0,
        max_chainlink_age_ms=60_000.0,
        max_binance_age_ms=5_000.0,
        max_trade_tape_age_ms=15_000.0,
        min_trade_sigma=2.5e-5,
    ),
}

DEFAULT_MARKET = "btc"

# Paired configs: base asset -> (15m_key, 5m_key)
_PAIRED = {
    "btc": ("btc", "btc_5m"),
    "eth": ("eth", "eth_5m"),
    "sol": ("sol", "sol_5m"),
    "xrp": ("xrp", "xrp_5m"),
    "btc_1h": ("btc_1h",),  # standalone, not paired with 5m/15m yet
}

def get_config(market: str) -> MarketConfig:
    """Look up a MarketConfig by key. Raises KeyError for unknown markets."""
    return MARKET_CONFIGS[market]

def get_paired_configs(market: str) -> list[tuple[str, MarketConfig]]:
    """Return list of (key, config) pairs for a market.

    'btc' -> [('btc', BTC 15m config), ('btc_5m', BTC 5m config)]
    'btc_5m' -> [('btc_5m', BTC 5m config)]  (single timeframe)
    """
    if market in _PAIRED:
        return [(k, MARKET_CONFIGS[k]) for k in _PAIRED[market]]
    return [(market, MARKET_CONFIGS[market])]
