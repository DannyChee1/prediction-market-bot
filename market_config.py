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
    binance_symbol: str = ""           # e.g. "btcusdt" for Binance bookTicker
    tail_mode: str = "normal"          # "normal", "student_t", or "kou"
    tail_nu_default: float = 20.0      # Student-t degrees of freedom (ignored for normal)
    kou_lambda: float = 0.007          # Kou jump intensity per observation
    kou_p_up: float = 0.51             # Kou upward jump probability
    kou_eta1: float = 1100.0           # Kou upward jump rate (1/mean_size)
    kou_eta2: float = 1100.0           # Kou downward jump rate


MARKET_CONFIGS: dict[str, MarketConfig] = {
    "btc": MarketConfig(
        slug_prefix="btc-updown-15m",
        chainlink_symbol="btc/usd",
        data_subdir="btc_15m",
        display_name="BTC 15m",
        window_duration_s=900.0,
        window_align_m=15,
        min_sigma=3e-05,
        binance_symbol="btcusdt",
        tail_mode="kou",
        tail_nu_default=20.0,
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
        min_sigma=7e-05,
        binance_symbol="btcusdt",
        tail_mode="kou",
        tail_nu_default=20.0,
        kou_lambda=0.005,
        kou_p_up=0.526,
        kou_eta1=1254.1,
        kou_eta2=1200.5,
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
    ),
}

DEFAULT_MARKET = "btc"

# Paired configs: base asset -> (15m_key, 5m_key)
_PAIRED = {
    "btc": ("btc", "btc_5m"),
    "eth": ("eth", "eth_5m"),
    "sol": ("sol", "sol_5m"),
    "xrp": ("xrp", "xrp_5m"),
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
