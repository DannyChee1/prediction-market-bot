"""Shared market configuration for BTC/ETH 15-minute Up/Down markets."""

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


MARKET_CONFIGS: dict[str, MarketConfig] = {
    "btc": MarketConfig(
        slug_prefix="btc-updown-15m",
        chainlink_symbol="btc/usd",
        data_subdir="btc_15m",
        display_name="BTC 15m",
        window_duration_s=900.0,
        window_align_m=15,
        binance_symbol="btcusdt",
    ),
    "eth": MarketConfig(
        slug_prefix="eth-updown-15m",
        chainlink_symbol="eth/usd",
        data_subdir="eth_15m",
        display_name="ETH 15m",
        window_duration_s=900.0,
        window_align_m=15,
        max_sigma=5e-05,
        binance_symbol="ethusdt",
    ),
    "btc_5m": MarketConfig(
        slug_prefix="btc-updown-5m",
        chainlink_symbol="btc/usd",
        data_subdir="btc_5m",
        display_name="BTC 5m",
        window_duration_s=300.0,
        window_align_m=5,
        binance_symbol="btcusdt",
    ),
    "eth_5m": MarketConfig(
        slug_prefix="eth-updown-5m",
        chainlink_symbol="eth/usd",
        data_subdir="eth_5m",
        display_name="ETH 5m",
        window_duration_s=300.0,
        window_align_m=5,
        max_sigma=5e-05,
        binance_symbol="ethusdt",
    ),
}

DEFAULT_MARKET = "btc"


def get_config(market: str) -> MarketConfig:
    """Look up a MarketConfig by key. Raises KeyError for unknown markets."""
    return MARKET_CONFIGS[market]
