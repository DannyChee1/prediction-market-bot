"""Shared market configuration for BTC/ETH 15-minute Up/Down markets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MarketConfig:
    slug_prefix: str       # "btc-updown-15m" / "eth-updown-15m"
    chainlink_symbol: str  # "btc/usd" / "eth/usd"
    data_subdir: str       # "btc" / "eth"
    display_name: str      # "BTC" / "ETH"


MARKET_CONFIGS: dict[str, MarketConfig] = {
    "btc": MarketConfig(
        slug_prefix="btc-updown-15m",
        chainlink_symbol="btc/usd",
        data_subdir="btc",
        display_name="BTC",
    ),
    "eth": MarketConfig(
        slug_prefix="eth-updown-15m",
        chainlink_symbol="eth/usd",
        data_subdir="eth",
        display_name="ETH",
    ),
}

DEFAULT_MARKET = "btc"


def get_config(market: str) -> MarketConfig:
    """Look up a MarketConfig by key. Raises KeyError for unknown markets."""
    return MARKET_CONFIGS[market]
