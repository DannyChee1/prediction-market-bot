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
    # Self-exciting Hawkes intensity for jump clustering. When non-None,
    # DiffusionSignal maintains a per-window HawkesIntensity instance
    # fed by detected jumps and publishes _hawkes_intensity / _hawkes_n_events
    # to ctx for downstream consumers (filtration features, dashboards).
    # Tuple is (mu, alpha, beta, k_sigma) where alpha < beta is required
    # for stationarity. Fit via /tmp/fit_hawkes.py on parquet history.
    hawkes_params: tuple[float, float, float, float] | None = None
    min_entry_z: float = 0.5           # Minimum |z| to enter
    min_entry_price: float = 0.25      # Minimum contract price to enter
    edge_threshold: float = 0.06       # Minimum edge to enter
    market_blend: float = 0.0          # Blend p_model with contract mid (0=off)
    # Per-window trade caps. The 1-trade-per-window default keeps the
    # bot from chasing every signal flip. Setting this >1 enables
    # "averaging in" — additional trades in the same window after the
    # first fills.
    #
    # Default applies to ALL markets that don't override (eth, sol, xrp,
    # 5m variants). At max=1 the same_direction_stacking_only flag is
    # dormant since there's never a "second trade" decision to gate.
    # Only btc 15m currently sets max_trades_per_window=2.
    max_trades_per_window: int = 1
    # When max_trades_per_window > 1, restrict subsequent trades to
    # the SAME direction as the first. Prevents the bot from hedging
    # against itself when the signal whipsaws within a window.
    same_direction_stacking_only: bool = True
    # Model-vs-market disagreement gate. When |p_model_raw - mid_up|
    # exceeds this, skip the trade — the diffusion model is probably
    # overconfident from a sigma collapse, not detecting real edge.
    # 1.0 = disabled (default). btc 15m sets 0.30 because backtest
    # showed +0.46 Sharpe and -0.8pp DD at that threshold paired
    # with the min_sigma=2e-5 upstream defense. btc_5m leaves it off
    # because the marginal trades it filters are profitable.
    max_model_market_disagreement: float = 1.0
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
        # half the data was being capped. Post-Kou-fix bounds [1e-5, 4e-4]
        # let realized vol drive the model.
        #
        # 2026-04-09: BUMP min_sigma 1e-5 → 2e-5 after live overconfidence
        # cluster. Live evidence: 13 fills in one session at sigma ≤ 1.2e-5
        # produced p_model 0.84-0.9987 on 15m UP, all losers. The hard
        # floor 1e-5 sits at ~p5; sigma_baseline (used by adaptive floor
        # MIN_SIGMA_RATIO * baseline) is also tiny in quiet windows so
        # the relative defense never lifts above the absolute floor.
        # New 2e-5 sits at ~p25-30 of empirical distribution — clamps
        # the bottom quartile (the dangerous quiet-period tail) without
        # touching the median. At sigma=2e-5 with τ=200s, a $20 BTC move
        # gives p_model ≈ 0.84 instead of 0.98. Still high, but no longer
        # extreme. Tune downward to 1.5e-5 if this over-clamps after a
        # few sessions of evidence.
        min_sigma=2e-05,
        max_sigma=4e-04,
        binance_symbol="btcusdt",
        tail_mode="kou",
        tail_nu_default=20.0,
        # Kou jump params moment-matched from 4.8k post-fix BTC 15m
        # windows (2M+ returns) at k=3σ jump threshold. See comment on
        # btc_5m below for methodology. Inert under tail_mode="kou";
        # used only under tail_mode="kou_full".
        kou_lambda=0.0684,
        kou_p_up=0.5013,
        kou_eta1=4504.3,
        kou_eta2=4509.6,
        # Hawkes self-exciting jump intensity, fit on 1000 BTC 15m post-fix
        # parquets via /tmp/fit_hawkes.py at k_sigma=3.0. Branching ratio
        # alpha/beta = 0.6 (moderate clustering), half-life ln(2)/beta = 13.9s.
        # Steady-state intensity 0.029/s ≈ 1.74 jumps/min. Inert until a
        # downstream consumer reads _hawkes_intensity (e.g. retrained
        # filtration model with hawkes feature).
        hawkes_params=(0.011611, 0.0300, 0.0500, 3.0),
        # 2026-04-09: reverted from 2 → 1 after live evidence that
        # stacking AMPLIFIES sigma-collapse losses. In the failure mode
        # we care about (sigma at floor → p_model pinned at 0.84-0.99),
        # the signal "persists" trivially because the model is locked,
        # so the second trade is just doubling down on a broken output.
        # Live session 2026-04-08 had two simultaneous 15m UP fills in
        # the same window, both losers (-$4.99 + -$4.66). Original
        # rationale (averaging in when signal evolves over a long
        # window) is sound in theory but fails when the model itself
        # is the failure point. Re-enable only after sigma-floor fix
        # is verified to remove the overconfidence cluster.
        max_trades_per_window=1,
        # 2026-04-09: lowered from default 0.5 → 0.15 after live evidence
        # that the min_sigma=2e-5 floor (just shipped) reduces typical |z|
        # from "frequently above 0.5" to "almost always 0.03-0.31 range".
        # The 0.5 default was calibrated against the OLD broken sigma values
        # where z routinely hit the ±1.0 cap. Post-fix it filters basically
        # every legitimate setup. Empirical evidence: 49,774 evals on a
        # single 15m window post-fix produced 0 trades — the top FLAT
        # reasons were min_z gates at z=0.030, 0.042, 0.110, 0.111, 0.112,
        # 0.113, 0.309 — none above 0.5. The new 0.15 threshold still
        # filters true noise (z near 0) but lets meaningful directional
        # signals through. Defended by the new edge persistence gate
        # (10s for 15m) and max_model_market_disagreement=0.30 above.
        min_entry_z=0.15,
        # Model-vs-market disagreement gate. Backtest A/B at 0.30:
        # btc 15m Sharpe 1.65 → 2.11 (+0.46), DD 2.3% → 1.5% (-0.8pp),
        # win rate 57.7% → 62.0%. Catches the cases where σ collapses
        # below the min_sigma=2e-5 floor (e.g. via the sigma_baseline
        # adaptive path) and the model becomes confidently wrong vs
        # the contract mid. Backstop to the upstream min_sigma fix.
        max_model_market_disagreement=0.30,
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
        # 2026-04-09: raised 1e-5 → 2e-5 to match btc 15m. Live evidence:
        # 33 trades at PF=0.39 with the old floor, while 15m (with 2e-5
        # floor + disagreement gate) went 3/3 wins. Same BTC, same feeds,
        # same model — only difference was the defensive stack. Mirroring.
        min_sigma=2e-05,
        max_sigma=2e-04,
        binance_symbol="btcusdt",
        tail_mode="kou",
        tail_nu_default=20.0,
        # Kou jump params moment-matched from 14k post-fix BTC 5m windows
        # (2M+ returns). Old values (λ=0.007, p_up=0.526, η=1254/1200)
        # were fit against the buggy model — p_up asymmetry was an
        # artifact of the Kou drift bug's downward p_UP bias. New values
        # reflect the actually-observed jump distribution at k=3σ.
        # Used only when tail_mode="kou_full"; inert under tail_mode="kou".
        # See /tmp/fit_kou_params.py + tasks/findings/post_fix_revalidation_2026-04-07.md.
        kou_lambda=0.0758,
        kou_p_up=0.5014,
        kou_eta1=4884.8,
        kou_eta2=4867.7,
        # Hawkes self-exciting jump intensity, fit on 1000 BTC 5m post-fix
        # parquets via /tmp/fit_hawkes.py at k_sigma=3.0. Branching ratio
        # alpha/beta = 0.4 (moderate clustering), half-life ln(2)/beta = 13.9s.
        # Steady-state intensity 0.040/s ≈ 2.37 jumps/min. Inert until a
        # downstream consumer reads _hawkes_intensity (e.g. retrained
        # filtration model with hawkes feature).
        hawkes_params=(0.023712, 0.0200, 0.0500, 3.0),
        # 2026-04-09: raised 0.0 → 0.15 to match btc 15m. With min_sigma
        # now at 2e-5, typical |z| is in 0.03-0.30 range; 0.15 filters
        # the noise while letting meaningful setups through.
        min_entry_z=0.15,
        min_entry_price=0.20,       # avoid deep OTM tail (was 0.10)
        # NOTE on edge_threshold: user asked to halve trade frequency
        # for better Sharpe. Backtest A/B showed this is the WRONG lever:
        #   edge=0.06 (current): 3577 trades, $26,826 PnL, Sharpe 1.52
        #   edge=0.08:           1596 trades,  $9,382 PnL, Sharpe 1.18
        #   edge=0.10:            904 trades,  $5,279 PnL, Sharpe 1.16
        # The marginal trades being filtered are collectively profitable,
        # so over-filtering hurts more than it helps. Keeping at 0.06.
        # If the user still wants to reduce live activity, the better
        # levers are min_entry_z (selectivity by signal strength) or
        # max_spread (selectivity by liquidity), not edge_threshold.
        edge_threshold=0.06,
        market_blend=0.3,           # pull p_model toward market mid
        # 2026-04-09: mirroring 15m's disagreement gate. Live data:
        # 15m with this gate at 0.30 went 3/3 wins; 5m without it went
        # 10/33 wins (PF=0.39). When |p_model_raw - mid_up| > 0.30,
        # the model is probably overconfident from sigma collapse.
        max_model_market_disagreement=0.30,
        # 2026-04-09: bumped 1000 → 5000 after live evidence that calm
        # markets trigger false stale-book FLATs. The Polymarket book WS
        # only sends updates when the book CHANGES — during quiet periods
        # the bot was seeing book_age 4-9 SECONDS even though the WS was
        # alive (PINGs were going through). The 1s threshold was meant
        # to detect WS disconnects, but in practice it was firing on
        # calm-market silence. 5s still catches real disconnects (the
        # Rust feed reconnects after 30s of no data anyway) but stops
        # false-firing during normal calm periods. The Binance freshness
        # gate (max_binance_age_ms=1500) provides the actual fast-market
        # protection — Binance updates ~10ms even in calm markets.
        max_book_age_ms=5000.0,
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
    "btc_1h": MarketConfig(
        slug_prefix="bitcoin-up-or-down",
        chainlink_symbol="btc/usd",
        data_subdir="btc_1h",
        display_name="BTC 1h",
        window_duration_s=3600.0,
        window_align_m=60,
        min_sigma=2e-05,
        # 1h windows accumulate more vol samples → per-second sigma can
        # be higher than 15m. The 4e-4 ceiling was hitting on ~30% of
        # windows, clipping the signal. 1e-3 covers p99.9 of 1h sigma.
        max_sigma=1e-03,
        binance_symbol="btcusdt",
        tail_mode="kou",
        # 2026-04-10: raised from 0.10 to 0.30 after trade analysis showed
        # |z|<0.3 entries have 47% WR (losing money). |z|>0.3 = 65% WR.
        min_entry_z=0.30,
        # 2026-04-10: raised from default 0.25 to 0.40. Entries at <$0.45
        # had 35% WR — deep OTM contrarian bets that fail on 1h.
        min_entry_price=0.40,
        market_blend=0.5,
        max_model_market_disagreement=0.30,
        max_trades_per_window=2,    # 1h has room for 2 entries
        # Lower edge threshold for 1h — the early_edge_mult inflates
        # the dynamic threshold by sqrt(tau/3600), which at mid-window
        # pushes it to ~0.11 at edge_threshold=0.06. With 0.04 the
        # mid-window threshold is ~0.074, which lets legitimate setups
        # through. The 1h window gives more time for the edge to
        # materialize vs 5m/15m where a quick threshold is defensive.
        edge_threshold=0.04,
        max_book_age_ms=10_000.0,
        max_chainlink_age_ms=60_000.0,
        max_binance_age_ms=5_000.0,
        max_trade_tape_age_ms=15_000.0,
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
