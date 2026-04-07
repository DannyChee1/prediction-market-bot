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
    min_entry_z: float = 0.5           # Minimum |z| to enter
    min_entry_price: float = 0.25      # Minimum contract price to enter
    edge_threshold: float = 0.06       # Minimum edge to enter
    market_blend: float = 0.0          # Blend p_model with contract mid (0=off)
    # σ estimator selector for _compute_vol. Options:
    #   "yz"   — Yang-Zhang on 5s OHLC bars (legacy default; suboptimal
    #            on continuously-traded feeds because the var_oc term
    #            assumes overnight gaps that don't exist).
    #   "rv"   — plain realized variance of normalised log returns
    #   "ewma" — RiskMetrics-style λ=0.94 EWMA of squared returns
    # A/B test (scripts/validate_sigma_estimators.py) on real data shows
    # EWMA and RV both beat YZ by 25-39% on 1-step forecast MSE for both
    # BTC 5m and BTC 15m. EWMA is the new default for BTC markets.
    sigma_estimator: str = "yz"
    # Stale-feature gates: each is a HARD SKIP (not a threshold widen).
    # All are live-only — backtest never populates the *_age_ms ctx fields.
    max_book_age_ms: float | None = None        # Skip if book WS older than this
    max_chainlink_age_ms: float | None = None   # Skip if chainlink price older
    max_binance_age_ms: float | None = None     # Skip if binance trade older
    max_trade_tape_age_ms: float | None = None  # Skip if trade tape older


MARKET_CONFIGS: dict[str, MarketConfig] = {
    "btc": MarketConfig(
        slug_prefix="btc-updown-15m",
        chainlink_symbol="btc/usd",
        data_subdir="btc_15m",
        display_name="BTC 15m",
        window_duration_s=900.0,
        window_align_m=15,
        # Bounds chosen from empirical 90s realized-σ distribution on
        # ~106k BTC 15m observations: p5≈9.2e-6, p50≈3.9e-5, p99≈4.1e-4.
        # Old [3e-5, 8e-5] band clamped 37% of obs UP and 19% DOWN — over
        # half the data was being capped. New bounds [1e-5, 4e-4] let
        # realized vol drive the model instead of being floored to a near
        # constant.
        min_sigma=1e-05,
        max_sigma=4e-04,
        binance_symbol="btcusdt",
        tail_mode="kou",
        tail_nu_default=20.0,
        # Market-blend: post-fix 50-day backfill (14k BTC 5m / 4.7k BTC 15m
        # REST + live windows) sweep showed BTC 15m Sharpe climbs from 0.61
        # @ blend=0.0 to 1.36 @ blend=0.5 (+123%), max drawdown drops from
        # 11.7% to 4.7%, with only a ~50% trade-count reduction (1294 → 620
        # test trades). Post-fix model still benefits from market-consensus
        # smoothing even though the Kou drift bug is gone — the 15m window
        # gives the model more time to drift away from market mid before
        # resolution, and pulling it back toward mid at entry nets a big
        # Sharpe win. See tasks/findings/post_fix_revalidation_2026-04-07.md.
        market_blend=0.5,
        # σ estimator: A/B test (validation_runs/sigma_estimators/btc.json)
        # showed EWMA has 26% lower 1-step forecast MSE than YZ. BUT a
        # second ablation (validation_runs/ablation_btc.json) showed that
        # swapping in EWMA HURTS BTC 15m PnL by ~$1.7k on the same test
        # set, because edge_threshold/kelly_fraction/filtration_threshold
        # were tuned to YZ. Set to "ewma" only AFTER re-tuning those.
        # sigma_estimator="ewma",
        # Stale-feature gates (live-only).
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
        # Bounds chosen from empirical 90s realized-σ distribution on
        # ~9.8k BTC 5m observations: p5≈5.8e-6, p50≈2.95e-5, p99≈1.5e-4.
        # The old [7e-5, 8e-5] band (a 14% range) was clamping 88.4% of
        # samples UP and 8.3% DOWN — only 3.3% of observations were ever
        # free to drive the prediction. The original tight band was a
        # workaround for the Kou risk-neutral drift bug, which has now
        # been fixed in `_model_cdf` (drift_z ∝ 1/σ blew up at the floor).
        # New bounds [1e-5, 2e-4] cover ~p10..p99.5.
        min_sigma=1e-05,
        max_sigma=2e-04,
        binance_symbol="btcusdt",
        tail_mode="kou",
        tail_nu_default=20.0,
        # Kou jump params — currently inert: see _model_cdf comment for why.
        # Kept in config for forward-compat in case `kou_cdf` is wired in.
        kou_lambda=0.007,
        kou_p_up=0.526,
        kou_eta1=1254.1,
        kou_eta2=1200.5,
        min_entry_z=0.0,            # blend filters disagreement (was 0.7)
        min_entry_price=0.20,       # avoid deep OTM tail (was 0.10)
        edge_threshold=0.06,
        market_blend=0.3,           # pull p_model toward market mid
        max_book_age_ms=1000.0,     # skip trades during book WS disconnects
        # σ estimator: see note on btc 15m above. EWMA wins on σ
        # forecasting but hurts PnL by $360 in ablation
        # (validation_runs/ablation_btc_5m.json). Opt in only after
        # re-tuning downstream hyperparameters.
        # sigma_estimator="ewma",
        # Stale-feature gates (live-only). 5m markets are more sensitive to
        # data freshness than 15m, so the chainlink window is tighter.
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
