"""
signal_diffusion — DiffusionSignal class (the model body).

Extracted from backtest.py during the P10.2 module split. The class body
is byte-for-byte identical to its old location; only the surrounding
imports changed. All helpers, dataclasses, calibration, and the Signal
ABC live in backtest_core.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from backtest_core import (
    MIN_FINAL_REMAINING_S,
    MAX_START_GAP_S,
    MIN_SIGMA_RATIO,
    poly_fee,
    norm_cdf,
    fast_t_cdf,
    kou_cdf,
    _betainc,
    _build_ohlc_bars,
    _yang_zhang_vol,
    _compute_vol_deduped,
    _time_prior_sigma,
    _load_priors_for_subdir,
    _HOURLY_VOL_MULT,
    _DOW_VOL_MULT,
    _GLOBAL_MEAN_SIGMA,
    _lookup_cross_asset_z,
    _CROSS_ASSET_TAU_CHECKPOINTS,
    CalibrationTable,
    Snapshot,
    Decision,
    Signal,
    compute_vamp,
)


class DiffusionSignal(Signal):
    """
    Models BTC as GBM. p_model = Phi(delta / (sigma * sqrt(tau))).
    Buys whichever side has edge > threshold, sized by half-Kelly.
    """
    name = "diffusion"

    def __init__(
        self,
        bankroll: float,
        vol_lookback_s: int = 90,
        min_sigma: float = 1e-6,
        edge_threshold: float = 0.04,
        early_edge_mult: float = 1.2,
        window_duration: float = 900.0,
        max_bet_fraction: float = 0.05,
        min_order_shares: float = 5.0,
        kelly_fraction: float = 0.25,
        slippage: float = 0.0,
        max_z: float = 1.0,
        momentum_lookback_s: int = 30,
        max_spread: float = 0.05,
        spread_edge_penalty: float = 1.0,
        vol_regime_lookback_s: int = 300,
        vol_regime_mult: float = 3.0,
        max_entry_time_s: float | None = None,
        reversion_discount: float = 0.0,
        momentum_majority: float = 1.0,
        maker_mode: bool = False,
        edge_threshold_step: float = 0.0,
        calibration_table: CalibrationTable | None = None,
        maker_warmup_s: float = 100.0,
        min_entry_price: float = 0.10,
        cal_prior_strength: float = 500.0,
        # 2026-04-09: disabled calibration fusion by default. With
        # Z_BIN_WIDTH=0.5, all post-fix z values (typically 0.03-0.31)
        # round to z_bin=0.0 and look up the same calibration cell,
        # which returns p≈0.50 by symmetry. The fusion then pulls every
        # p_model toward 0.5 — the same direction as market_blend —
        # producing a double-shrinkage that adds negative value. Setting
        # cal_max_weight=0.0 makes the fusion return pure p_gbm. The
        # calibration infrastructure stays for future use with a finer
        # bin grid or an empirical quantile map (see audit P3.2).
        cal_max_weight: float = 0.0,
        inventory_skew: float = 0.02,
        maker_withdraw_s: float = 60.0,
        sigma_ema_alpha: float = 0.30,
        max_sigma: float | None = None,
        vamp_mode: str = "none",
        vamp_filter_threshold: float = 0.07,
        max_entry_price: float = 1.0,
        # NO/DOWN bias: reduce edge threshold for DOWN by this fraction
        # of base threshold (optimism tax — YES/UP tends to be overpriced)
        down_edge_bonus: float = 0.0,
        # Microstructure toxicity filter
        toxicity_threshold: float = 0.75,
        toxicity_edge_mult: float = 1.5,
        # Volatility kill switch (absolute EMA ceiling)
        vol_kill_sigma: float | None = None,
        # Regime-scaled z: scale z by sigma_calibration / sigma_ema
        regime_z_scale: bool = False,
        sigma_calibration: float | None = None,
        # VPIN flow toxicity filter
        vpin_threshold: float = 0.95,
        vpin_edge_mult: float = 1.5,
        vpin_window: int = 20,
        vpin_bar_s: float = 60.0,
        # Oracle lag detection (Binance vs Chainlink discrepancy)
        oracle_lag_threshold: float = 0.002,
        oracle_lag_mult: float = 2.0,
        # F4: oracle lead-lag as a profit signal. The Polymarket RTDS
        # rebroadcasts Chainlink with ~1.2s of constant delay (verified
        # via scripts/measure_feed_latency.py). When Binance has moved
        # but Chainlink hasn't propagated yet, the next Chainlink update
        # will likely close the gap toward Binance — that's how the
        # rebroadcast pipeline works, not a model assumption. Bias
        # p_model in the direction of the gap so we trade as if we
        # already saw the next Chainlink update.
        #
        # Formula:
        #   gap = (binance_mid - chainlink_price) / chainlink_price
        #   bias = oracle_lead_bias * clip(gap / oracle_lag_threshold, -1, 1)
        #   p_model_biased = clip(p_model + bias, 0.01, 0.99)
        #
        # At gap=threshold (0.2%), bias = oracle_lead_bias (default
        # 0.05 = 5pp). The clip prevents huge biases from extreme gaps.
        # Set to 0.0 to disable.
        oracle_lead_bias: float = 0.05,
        # Model-vs-market disagreement gate. When the diffusion model's
        # raw output disagrees with the contract mid by more than this
        # many percentage points, the model is probably wrong (not the
        # market) — sustained low-vol periods make σ-based z-scores
        # confidently incorrect. Skip the trade entirely.
        #
        # Empirically observed: σ collapse 4e-5 → 1.4e-5 over 1 second
        # produces z=-1.00(cap), p_model_raw=0.16, while contract mid
        # is at 0.47. The 31pp disagreement is a model failure, not
        # an information edge. market_blend=0.3 only partially absorbs
        # this; the gate is a hard backstop. Set to 1.0 to disable.
        max_model_market_disagreement: float = 0.30,
        # Order book imbalance alpha: shift p_model by obi_weight * OBI
        obi_weight: float = 0.03,
        # Chainlink blend: seconds before expiry to start blending
        # effective price from Binance toward Chainlink for z-score.
        # At chainlink_blend_s remaining → 100% Binance;
        # at 0 remaining → 100% Chainlink (matches resolution source).
        chainlink_blend_s: float = 120.0,
        # Fat-tail CDF: "normal", "student_t", "kou", "market_adaptive"
        tail_mode: str = "normal",
        tail_nu_default: float = 20.0,
        # Kou jump-diffusion parameters (only used when tail_mode="kou")
        kou_lambda: float = 0.007,   # jump intensity per observation
        kou_p_up: float = 0.51,      # probability of upward jump
        kou_eta1: float = 1100.0,    # rate of upward jumps (1/mean_size)
        kou_eta2: float = 1100.0,    # rate of downward jumps
        # Market-adaptive parameters (only used when tail_mode="market_adaptive")
        market_blend_alpha: float = 0.30,  # weight on GBM vs market
        # Direct p_model blend with contract mid (anti-fade-the-market)
        # Applied AFTER _p_model() for any tail_mode. 0=off, 0.3=BTC 5m default.
        market_blend: float = 0.0,
        # Stale-book gate: skip trades when book WS data is older than this.
        # Live data shows trades during book WS disconnects (book_age >1s) win
        # 25% vs 58% for fresh-book trades. None=off, 1000=BTC 5m default.
        max_book_age_ms: float | None = None,
        # Additional stale-feature gates (live-only — backtest replays data
        # so the freshness fields are never set, and these gates are no-ops).
        # Each is a hard skip (not a threshold widen). Pattern matches the
        # max_book_age_ms gate above. Notebook 24 ("Trading with Violated
        # Model Assumptions") makes the case that *any* stale input
        # invalidates the model entirely; widening the threshold trades a
        # bigger position for a noisier signal, which is the wrong direction.
        max_chainlink_age_ms: float | None = None,
        max_binance_age_ms: float | None = None,
        max_trade_tape_age_ms: float | None = None,
        # σ estimator: "yz" (legacy default), "rv" (plain realized variance),
        # or "ewma" (RiskMetrics λ=0.94 EWMA of squared returns).
        # IMPORTANT: although EWMA has 26% lower 1-step forecast MSE than
        # YZ on real BTC data (see scripts/validate_sigma_estimators.py),
        # ablation shows that swapping in EWMA HURTS backtest PnL because
        # downstream hyperparameters (edge_threshold, kelly_fraction,
        # filtration_threshold) were tuned to the YZ-flavoured σ. EWMA is
        # left as opt-in until those parameters are re-tuned. See
        # validation_runs/ablation_btc_5m.json.
        sigma_estimator: str = "yz",
        # Optional HMM regime classifier (Quant Guild #51). When provided,
        # it classifies each window into a latent regime once enough ticks
        # have arrived to compute the feature vector, then scales
        # `kelly_fraction` by the per-state multiplier baked into the
        # classifier. None = no regime adjustment (legacy behavior).
        regime_classifier=None,
        regime_early_tau_s: float | None = None,
        # Kalman-smoothed order-book imbalance gate. When True, the
        # imbalance gate at decision time uses an AR(1) Kalman estimate
        # instead of the raw L2 OBI snapshot. Logically better (less
        # flicker), but ablation showed it lets in trades that the raw
        # gate would have filtered, hurting precision. Off by default
        # until the downstream filtration_threshold is re-tuned.
        use_kalman_obi: bool = False,
        # Restore train/inference parity for the `mid_momentum`
        # filtration feature (the inference path used to return 0
        # during the first ~60s of every window while training computed
        # the actual delta). Off by default because the existing
        # filtration_model.pkl was trained against the broken inference
        # behavior — flipping this on without retraining the filtration
        # model degrades calibration of the confidence threshold.
        # When you retrain filtration_model.pkl with the parity-restored
        # features, set this to True to deploy the consistent path.
        mid_momentum_parity: bool = False,
        # Hawkes-modulated jump intensity (Quant Guild #94). When non-None,
        # the signal maintains a HawkesIntensity instance per market and
        # publishes the current λ(t) to ctx as `_hawkes_intensity` so it
        # can be consumed by the dashboard, the filtration model, or a
        # future regime-aware sizing rule. Does NOT affect any current
        # gating or sizing — it is purely a published feature. Pass a
        # tuple (mu, alpha, beta, k_sigma) to enable. None = disabled.
        hawkes_params: tuple[float, float, float, float] | None = None,
        # Market data subdir (e.g. "btc_15m") used to look up the
        # per-market hourly_priors.json file when computing the cold-start
        # σ prior in `_smoothed_sigma`. None = fall back to the static
        # constants in `_HOURLY_VOL_MULT` / `_GLOBAL_MEAN_SIGMA`.
        data_subdir: str | None = None,
        # Avellaneda-Stoikov unified quoting mode
        as_mode: bool = False,
        gamma_inv: float = 0.15,       # risk aversion for inventory penalty
        gamma_spread: float = 0.75,    # risk aversion for base spread
        min_edge: float = 0.05,        # floor on required edge
        tox_spread: float = 0.05,      # additive spread from toxicity
        vpin_spread: float = 0.05,     # additive spread from VPIN
        lag_spread: float = 0.08,      # additive spread from oracle lag
        edge_step: float = 0.01,       # additive spread per fill
        contract_vol_lookback_s: int = 60,  # lookback for contract mid vol
        # Kalman filter for sigma estimation (replaces EMA when True)
        use_kalman_sigma: bool = True,
        kalman_q: float = 0.10,   # process noise ratio (vol-of-vol / vol)
        kalman_r: float = 0.075,  # observation noise ratio (≈ 1/sqrt(2*90))
        # XGBoost filtration model (optional confidence gate)
        filtration_model=None,         # FiltrationModel instance or None
        filtration_threshold: float = 0.55,
        filtration_asset_id: int = 0,  # see filtration_model.ASSET_IDS
        filtration_baseline_vol_s: int = 300,
        # Filtration mode: how to use the model's confidence score
        #   "gate"     — binary skip if confidence < filtration_threshold (legacy)
        #   "size_mult"— always pass, multiply Kelly by a confidence-derived
        #                multiplier so low-confidence trades shrink instead
        #                of being eliminated. Read at the Kelly step.
        # Default is size_mult — A/B test showed it strictly dominates gate
        # on both BTC markets and meaningfully wins on btc 15m
        # (Sharpe 1.28 → 1.36, PnL +$238, DD 6.3% → 5.8%).
        filtration_mode: str = "size_mult",
        # Confidence-to-multiplier mapping for "size_mult" mode (classification).
        # Linear from filtration_size_mult_floor (mult=0) to 1.0 (mult=1).
        # At conf=floor, the trade is fully suppressed; at conf=1.0, full
        # size; in between, scaled linearly. Default floor 0.45 means trades
        # the model thinks are worse than coin-flip get zero size.
        filtration_size_mult_floor: float = 0.45,
        # Multiplier ceiling for regression-mode filtration (target=regression).
        # Confidence is now predicted PnL per dollar (typically -1.0 to ~+1.0).
        # Linear from 0.0 (mult=0) to filtration_ev_full (mult=1). Above the
        # ceiling, mult stays at 1.0. Default 0.50 means a predicted 50% per-
        # dollar return saturates the multiplier.
        filtration_ev_full: float = 0.50,
        # Oracle hard cancel: return FLAT immediately when Binance-Chainlink
        # gap exceeds this fraction.  Set to 0.0 to disable (default).
        # 0.004 = cancel when gap > 0.4% (Chainlink triggers at 0.5%).
        oracle_cancel_threshold: float = 0.0,
        # Cross-asset disagreement gate: skip trade when a correlated asset
        # (e.g. ETH when trading BTC) has a z-score pointing the other way.
        # Requires cross_asset_z_lookup precomputed by build_cross_asset_lookup().
        cross_asset_z_lookup: dict | None = None,
        cross_asset_min_z: float = 0.3,   # both |z| must exceed this to veto
        # Minimum |z| to enter: filter out low-conviction setups
        min_entry_z: float = 0.0,
        # Edge persistence gate: require edge to be supported for at least
        # this many seconds before firing the trade. Defends against
        # spike-chasing where a fast Binance bookticker move briefly
        # crosses the edge threshold and then mean-reverts. The bot would
        # otherwise fire on the spike and lose when the price snaps back.
        # 0.0 = disabled (legacy behavior). Recommended ~5s for 5m
        # markets, ~10s for 15m markets. Tracks per-side first-edge
        # timestamps in ctx; resets when edge drops back below threshold.
        edge_persistence_s: float = 0.0,
        # Stale-quote sniper mode: opt-in taker strategy that buys when
        # the Polymarket CLOB ask is stale relative to Binance fair value.
        # See decide_stale_quote() for the logic.
        stale_quote_mode: bool = False,
        stale_threshold: float = 0.03,
    ):
        self.bankroll = bankroll
        self.vol_lookback_s = vol_lookback_s
        self.min_sigma = min_sigma
        self.edge_threshold = edge_threshold
        self.early_edge_mult = early_edge_mult
        self.window_duration = window_duration
        self.max_bet_fraction = max_bet_fraction
        self.min_order_shares = min_order_shares
        self.kelly_fraction = kelly_fraction
        self.slippage = slippage
        self.max_z = max_z
        self.momentum_lookback_s = momentum_lookback_s
        self.max_spread = max_spread
        self.spread_edge_penalty = spread_edge_penalty
        self.vol_regime_lookback_s = vol_regime_lookback_s
        self.vol_regime_mult = vol_regime_mult
        self.max_entry_time_s = max_entry_time_s
        self.reversion_discount = reversion_discount
        self.momentum_majority = momentum_majority
        self.maker_mode = maker_mode
        self.edge_threshold_step = edge_threshold_step
        self.calibration_table = calibration_table
        self.maker_warmup_s = maker_warmup_s
        self.min_entry_price = min_entry_price
        self.cal_prior_strength = cal_prior_strength
        self.cal_max_weight = cal_max_weight
        self.inventory_skew = inventory_skew
        self.maker_withdraw_s = maker_withdraw_s
        self.sigma_ema_alpha = sigma_ema_alpha
        self.max_sigma = max_sigma
        self.vamp_mode = vamp_mode                    # "none", "cost", "filter"
        self.vamp_filter_threshold = vamp_filter_threshold
        self.max_entry_price = max_entry_price
        self.down_edge_bonus = down_edge_bonus
        self.toxicity_threshold = toxicity_threshold
        self.toxicity_edge_mult = toxicity_edge_mult
        self.vol_kill_sigma = vol_kill_sigma
        self.regime_z_scale = regime_z_scale
        self.sigma_calibration = sigma_calibration
        self.vpin_threshold = vpin_threshold
        self.vpin_edge_mult = vpin_edge_mult
        self.vpin_window = vpin_window
        self.vpin_bar_s = vpin_bar_s
        self.oracle_lag_threshold = oracle_lag_threshold
        self.oracle_lag_mult = oracle_lag_mult
        self.oracle_lead_bias = float(oracle_lead_bias)
        self.max_model_market_disagreement = float(max_model_market_disagreement)
        self.obi_weight = obi_weight
        self.chainlink_blend_s = chainlink_blend_s
        self.tail_mode = tail_mode
        self.tail_nu_default = tail_nu_default
        self.kou_lambda = kou_lambda
        self.kou_p_up = kou_p_up
        self.kou_eta1 = kou_eta1
        self.kou_eta2 = kou_eta2
        self.market_blend_alpha = market_blend_alpha
        self.market_blend = market_blend
        self.max_book_age_ms = max_book_age_ms
        self.max_chainlink_age_ms = max_chainlink_age_ms
        self.max_binance_age_ms = max_binance_age_ms
        self.max_trade_tape_age_ms = max_trade_tape_age_ms
        if sigma_estimator not in ("yz", "rv", "ewma"):
            raise ValueError(
                f"sigma_estimator must be 'yz', 'rv', or 'ewma', "
                f"got {sigma_estimator!r}"
            )
        self.sigma_estimator = sigma_estimator
        self.regime_classifier = regime_classifier
        # If the caller didn't tell us when to compute regime features,
        # default to "after window is half-elapsed" — guarantees enough
        # history for the early-z feature.
        if regime_early_tau_s is None:
            regime_early_tau_s = window_duration / 2.0
        self.regime_early_tau_s = float(regime_early_tau_s)
        self.use_kalman_obi = bool(use_kalman_obi)
        self.mid_momentum_parity = bool(mid_momentum_parity)
        self.hawkes_params = hawkes_params  # (mu, alpha, beta, k_sigma) or None
        self.data_subdir = data_subdir
        self.as_mode = as_mode
        self.gamma_inv = gamma_inv
        self.gamma_spread = gamma_spread
        self.min_edge = min_edge
        self.tox_spread = tox_spread
        self.vpin_spread = vpin_spread
        self.lag_spread = lag_spread
        self.edge_step = edge_step
        self.contract_vol_lookback_s = contract_vol_lookback_s
        self.use_kalman_sigma = use_kalman_sigma
        self.kalman_q = kalman_q
        self.kalman_r = kalman_r
        self.filtration_model = filtration_model
        self.filtration_threshold = filtration_threshold
        self.filtration_asset_id = filtration_asset_id
        self.filtration_baseline_vol_s = filtration_baseline_vol_s
        if filtration_mode not in ("gate", "size_mult"):
            raise ValueError(
                f"filtration_mode must be 'gate' or 'size_mult', got {filtration_mode!r}"
            )
        self.filtration_mode = filtration_mode
        self.filtration_size_mult_floor = float(filtration_size_mult_floor)
        self.filtration_ev_full = float(filtration_ev_full)
        self.oracle_cancel_threshold = oracle_cancel_threshold
        self.cross_asset_z_lookup = cross_asset_z_lookup
        self.cross_asset_min_z = cross_asset_min_z
        self.min_entry_z = min_entry_z
        self.edge_persistence_s = edge_persistence_s
        self.stale_quote_mode = stale_quote_mode
        self.stale_threshold = stale_threshold

    def _compute_vol(
        self,
        prices: list[float],
        timestamps: list[int] | None = None,
    ) -> float:
        """Realized σ estimator dispatch.

        The estimator is selected by `self.sigma_estimator`:
          * "yz"   — Yang-Zhang on 5s OHLC bars (legacy, backward compatible
                     default — but loses 25-39% on 1-step forecast MSE per
                     scripts/validate_sigma_estimators.py because YZ's
                     var_oc term assumes overnight gaps that don't exist
                     on a continuously-traded feed).
          * "rv"   — plain stdev of normalised log returns
          * "ewma" — RiskMetrics-style EWMA with λ=0.94. Default for BTC
                     markets per A/B test.

        Falls back to YZ if anything unexpected happens, so this is
        safe to roll out incrementally.
        """
        if self.sigma_estimator == "yz":
            return _compute_vol_deduped(prices, timestamps)
        # Lazy import keeps backtest.py independent of scripts/ at import time
        try:
            from scripts.sigma_estimators import (
                ewma_sigma_per_s,
                realized_variance_per_s,
            )
        except ImportError:
            return _compute_vol_deduped(prices, timestamps)
        if timestamps is None:
            timestamps = [i * 1000 for i in range(len(prices))]
        if self.sigma_estimator == "ewma":
            return ewma_sigma_per_s(prices, timestamps, lambda_=0.94)
        if self.sigma_estimator == "rv":
            return realized_variance_per_s(prices, timestamps)
        # Unknown — fall back to YZ rather than crash
        return _compute_vol_deduped(prices, timestamps)

    def _record_book_state(self, snapshot: "Snapshot", ctx: dict) -> None:
        """Persist lightweight book state for diagnostics/filtration features."""
        ctx["_last_ts_ms"] = snapshot.ts_ms
        if snapshot.best_bid_up is None or snapshot.best_ask_up is None:
            return
        mid_up = (snapshot.best_bid_up + snapshot.best_ask_up) / 2.0
        mid_hist = ctx.setdefault("_mid_up_history", [])
        mid_hist.append(float(mid_up))
        if len(mid_hist) > 600:
            del mid_hist[:-600]

    def _maybe_update_tail_nu(self, hist: list[float], ctx: dict) -> None:
        """Estimate Student-t degrees of freedom from recent realized kurtosis."""
        if self.tail_mode != "student_t" or len(hist) < 30:
            return

        k_n = min(90, len(hist))
        k_prices = hist[-k_n:]
        k_rets = [
            math.log(k_prices[i] / k_prices[i - 1])
            for i in range(1, len(k_prices))
            if k_prices[i - 1] > 0 and k_prices[i] > 0
        ]
        if len(k_rets) < 20:
            return

        mean = sum(k_rets) / len(k_rets)
        var = sum((r - mean) ** 2 for r in k_rets) / len(k_rets)
        if var <= 0:
            return

        m4 = sum((r - mean) ** 4 for r in k_rets) / len(k_rets)
        excess_kurt = (m4 / (var * var)) - 3.0
        if excess_kurt > 0.2:
            ctx["_tail_nu"] = max(
                self.tail_nu_default,
                min(30.0, 4.0 + 6.0 / excess_kurt),
            )
        else:
            ctx["_tail_nu"] = 30.0  # near-Gaussian

    def _apply_regime_z_scale(
        self,
        z_raw: float,
        sigma_per_s: float,
        ctx: dict,
    ) -> tuple[float, float]:
        """Adjust z by current-vs-calibration vol regime.

        When live vol is above the calibration regime, shrink z.
        When live vol is below the calibration regime, amplify z.

        WARNING (2026-04-09): this is NOT the textbook calibration-space
        transformation. The correct transform to look up a calibration-table
        probability from live z would be ``z_live * sigma_per_s / sigma_cal``
        (multiply — not divide — by the ratio). The current factor of
        ``sigma_cal / sigma_per_s`` is a heuristic that double-shrinks in
        high-vol regimes (because z_raw already has sigma_per_s in its
        denominator) and double-amplifies in quiet regimes — up to 4x at
        the [0.5, 2.0] clamp bounds. This path is gated off by default
        (``regime_z_scale=False``); revisit before ever enabling.
        """
        factor = 1.0
        if (
            self.regime_z_scale
            and self.sigma_calibration
            and self.sigma_calibration > 0
            and sigma_per_s > 0
        ):
            factor = self.sigma_calibration / sigma_per_s
            factor = max(0.5, min(2.0, factor))
            z_raw *= factor
        ctx["_regime_z_factor"] = factor
        return z_raw, factor

    def _smoothed_sigma(self, raw_sigma: float, ctx: dict) -> float:
        """Kalman filter for sigma estimation (falls back to EMA if disabled).

        Kalman filter advantages over EMA:
          - Adaptive gain: reacts fast to vol spikes, smooth during calm periods
          - Principled noise model: Q (process noise) and R (observation noise)
          - Tracks uncertainty explicitly via posterior variance P

        State:   x  = true (latent) volatility
        Process: x_t = x_{t-1} + w,  w ~ N(0, Q)
        Obs:     y_t = x_t + v,       v ~ N(0, R)

        Q = (x * kalman_q)^2  — vol-of-vol proportional to current vol
        R = (x * kalman_r)^2  — measurement noise (Yang-Zhang ~90 samples
                                 has std ≈ sigma/sqrt(2N), so r ≈ 1/sqrt(180))
        """
        if raw_sigma == 0.0:
            # No measurable vol in lookback. Use best available prior:
            # 1. Kalman state from recent ticks (most accurate)
            # 2. Time-of-day / day-of-week historical prior (covers cold start)
            x = ctx.get("_kalman_x")
            if x is not None and x > 0:
                return max(min(x, self.max_sigma) if self.max_sigma else x,
                           self.min_sigma)
            # Cold start: use time-based prior so we don't go FLAT
            # on the very first window (especially weekends).
            ts_hist = ctx.get("ts_history", [])
            ts_ms = ctx.get("_last_ts_ms", 0) or (ts_hist[-1] if ts_hist else 0)
            if ts_ms > 0:
                prior = _time_prior_sigma(ts_ms,
                                          data_subdir=self.data_subdir)
                return max(min(prior, self.max_sigma) if self.max_sigma else prior,
                           self.min_sigma)
            return 0.0

        if not self.use_kalman_sigma:
            # Legacy EMA path
            ema = ctx.get("_sigma_ema")
            ema = raw_sigma if ema is None else (
                self.sigma_ema_alpha * raw_sigma + (1 - self.sigma_ema_alpha) * ema
            )
            ctx["_sigma_ema"] = ema
            if self.max_sigma is not None:
                ema = min(ema, self.max_sigma)
            return max(ema, self.min_sigma)

        # Kalman filter path
        x = ctx.get("_kalman_x")
        P = ctx.get("_kalman_P")

        if x is None:
            # Initialise: start at first observation, high uncertainty
            x = raw_sigma
            P = (raw_sigma * self.kalman_r) ** 2

        # Process and observation noise (scale with current estimate)
        Q = (x * self.kalman_q) ** 2
        R = (x * self.kalman_r) ** 2
        if R < 1e-20:
            R = 1e-20

        # Predict
        P_pred = P + Q

        # Update (Kalman gain)
        K = P_pred / (P_pred + R)
        x_new = x + K * (raw_sigma - x)
        P_new = (1.0 - K) * P_pred

        ctx["_kalman_x"] = x_new
        ctx["_kalman_P"] = P_new
        ctx["_kalman_gain"] = K  # expose for diagnostics

        if self.max_sigma is not None:
            x_new = min(x_new, self.max_sigma)
        return max(x_new, self.min_sigma)

    def _maybe_publish_hawkes(self, hist: list[float], ts_hist: list[int],
                              sigma_per_s: float, ctx: dict) -> None:
        """Maintain a per-window Hawkes intensity tracker and publish
        the current λ(t) to ctx as `_hawkes_intensity`. No-op when
        `self.hawkes_params is None`.

        We rebuild the tracker on the first tick of each window
        (detected by absence of the cached state), then incrementally
        feed new ticks as they arrive. The "jump" definition is
        |z| > k_sigma using the current realized sigma.
        """
        if self.hawkes_params is None or sigma_per_s <= 0 or len(hist) < 2:
            return
        try:
            from scripts.hawkes import HawkesIntensity
        except ImportError:
            return
        mu, alpha, beta, k_sigma = self.hawkes_params
        h = ctx.get("_hawkes_state")
        if h is None:
            h = HawkesIntensity(mu=mu, alpha=alpha, beta=beta)
            ctx["_hawkes_state"] = h
            # Seed: scan the existing history for jumps
            from scripts.hawkes import detect_jumps
            jump_times = detect_jumps(hist, ts_hist, sigma_per_s, k_sigma)
            for t in jump_times:
                h.add_event(t)
        else:
            # Incremental: only check the most recent tick (i.e. compare
            # the last two prices). This is a tiny per-tick cost and
            # avoids re-scanning the whole history.
            if len(hist) >= 2 and hist[-1] > 0 and hist[-2] > 0:
                dt = (ts_hist[-1] - ts_hist[-2]) / 1000.0
                if dt > 0:
                    r = math.log(hist[-1] / hist[-2])
                    z = r / (sigma_per_s * math.sqrt(dt))
                    if abs(z) > k_sigma:
                        h.add_event(ts_hist[-1] / 1000.0)
        ctx["_hawkes_intensity"] = h.intensity_at(ts_hist[-1] / 1000.0)
        ctx["_hawkes_n_events"] = h.n_events

    def _maybe_compute_regime(self, ctx: dict, tau: float,
                              hist: list[float],
                              ts_hist: list[int]) -> float:
        """Classify the current window's regime once enough data exists.

        Returns the kelly multiplier to apply (1.0 if no classifier or
        not enough data yet). Caches the regime label and multiplier in
        ctx for the rest of the window so we don't redo the work on
        every tick.

        The classifier is only invoked when (a) there is no cached
        result for this window, AND (b) the current tau is past the
        early-tau threshold (we have enough ticks for stable features).
        """
        if self.regime_classifier is None:
            return 1.0
        cached = ctx.get("_regime_kelly_mult")
        if cached is not None:
            return float(cached)
        # Wait until we are past the early-tau threshold before classifying.
        # Note: tau decreases over the window, so "past" means tau < threshold.
        if tau > self.regime_early_tau_s:
            return 1.0
        if not hist or not ts_hist or len(hist) < 30:
            return 1.0
        try:
            from regime_classifier import compute_window_regime_features
            features = compute_window_regime_features(
                hist, ts_hist,
                early_tau_target=self.regime_early_tau_s,
            )
            if features is None:
                return 1.0
            state_idx, label, kelly_mult = self.regime_classifier.classify_window(
                features
            )
            ctx["_regime_state"] = state_idx
            ctx["_regime_label"] = label
            ctx["_regime_kelly_mult"] = float(kelly_mult)
            return float(kelly_mult)
        except Exception as exc:
            # Never let a regime-classifier failure block trading. Log
            # via ctx so the dashboard can surface it; fall through to 1.0.
            ctx["_regime_error"] = f"{type(exc).__name__}: {exc}"
            ctx["_regime_kelly_mult"] = 1.0
            return 1.0

    def _kalman_obi_update(
        self,
        state_key: str,
        raw_obi: float,
        ctx: dict,
    ) -> float:
        """One-dimensional AR(1) Kalman smoother on order-book imbalance.

        State model:    x_t = β · x_{t-1} + w_t,   w ~ N(0, Q)
        Observation:    y_t = x_t + v_t,           v ~ N(0, R)

        Per the mean-reversion notebook (#95), this is the simplest
        Kalman form that catches book-flicker noise without lagging
        real regime shifts. β = 0.95 means the latent imbalance has
        a 20-tick half-life (≈20 seconds), Q is small (state changes
        slowly), R is larger (raw L2 OBI is noisy).

        Two independent state slots are kept in ctx (one per side)
        keyed by `state_key`. Returns the posterior mean.
        """
        beta = 0.95
        Q = 0.001    # small state noise — true imbalance changes slowly
        R = 0.05     # observation noise — raw OBI flickers a lot

        state = ctx.get(state_key)
        if state is None:
            x = float(raw_obi)
            P = 1.0
        else:
            x, P = state
            # Predict
            x = beta * x
            P = beta * beta * P + Q
            # Update
            K = P / (P + R)
            x = x + K * (float(raw_obi) - x)
            P = (1.0 - K) * P
        ctx[state_key] = (x, P)
        return x

    def _check_filtration(
        self,
        z: float,
        sigma: float,
        tau: float,
        snapshot: "Snapshot",
        ctx: dict,
    ) -> bool:
        """Return True if filtration model approves this trade, False to filter out.

        Returns True unconditionally when no filtration model is loaded,
        or when |z| is too small to have meaningful signal direction.
        """
        if self.filtration_model is None:
            return True
        if abs(z) < 0.10:
            return True  # no directional signal to filter

        from filtration_model import extract_features

        # Baseline vol for regime ratio (use longer lookback from history)
        hist = ctx.get("price_history", [])
        ts_hist = ctx.get("ts_history", [])
        lo_base = max(0, len(hist) - self.filtration_baseline_vol_s)
        sigma_base = _compute_vol_deduped(hist[lo_base:], ts_hist[lo_base:])
        vol_regime_ratio = sigma / sigma_base if sigma_base > 0 else 1.0

        # Buy pressure from recent trade side history
        trade_sides = ctx.get("_trade_side_history", [])
        if trade_sides:
            recent = trade_sides[-60:]
            buy_pressure = sum(1 for s in recent if s == "BUY") / len(recent)
        else:
            buy_pressure = 0.5

        # Mid momentum: two paths.
        #
        # Legacy (default, mid_momentum_parity=False): returns 0 for any
        # window with < 62 mids. The existing filtration_model.pkl was
        # trained against this behavior, so flipping the parity flag on
        # without retraining the model degrades calibration.
        #
        # Parity-restored (opt-in, mid_momentum_parity=True): mirrors the
        # training-time computation in
        # train_filtration.py:compute_mid_momentum:
        #     mids = df["mid_up"].iloc[lo:idx+1].dropna()
        #     return mids.iloc[-1] - mids.iloc[0]
        # i.e. use the earliest available index up to 60 ticks back.
        # Use this path AFTER you retrain the filtration model.
        mid_history = ctx.get("_mid_up_history", [])
        if self.mid_momentum_parity:
            if len(mid_history) >= 2:
                lookback_idx = max(0, len(mid_history) - 61)
                mid_momentum = float(mid_history[-1]) - float(mid_history[lookback_idx])
            else:
                mid_momentum = 0.0
        else:
            mid_momentum = (float(mid_history[-1]) - float(mid_history[-61]))  \
                if len(mid_history) >= 62 else 0.0

        ts_ms = ctx.get("_last_ts_ms", 0)
        dt = pd.Timestamp(ts_ms, unit="ms", tz="UTC") if ts_ms else None
        hour_of_day = dt.hour if dt else 12
        is_weekend  = int(dt.weekday() >= 5) if dt else 0

        spread_up   = (snapshot.best_ask_up or 0.5) - (snapshot.best_bid_up or 0.5)
        spread_down = (snapshot.best_ask_down or 0.5) - (snapshot.best_bid_down or 0.5)
        imb_up   = sum(sz for _, sz in snapshot.bid_levels_up) - \
                   sum(sz for _, sz in snapshot.ask_levels_up)
        total_up = sum(sz for _, sz in snapshot.bid_levels_up) + \
                   sum(sz for _, sz in snapshot.ask_levels_up)
        imb_up = imb_up / total_up if total_up > 0 else 0.0

        imb_dn   = sum(sz for _, sz in snapshot.bid_levels_down) - \
                   sum(sz for _, sz in snapshot.ask_levels_down)
        total_dn = sum(sz for _, sz in snapshot.bid_levels_down) + \
                   sum(sz for _, sz in snapshot.ask_levels_down)
        imb_dn = imb_dn / total_dn if total_dn > 0 else 0.0

        # Hawkes features (default 0.0/0 when hawkes is disabled — the
        # filtration model trained without hawkes_params will treat these
        # as constant features and effectively ignore them).
        hawkes_intensity = float(ctx.get("_hawkes_intensity", 0.0) or 0.0)
        hawkes_n_events = int(ctx.get("_hawkes_n_events", 0) or 0)

        features = extract_features(
            z=z,
            sigma=sigma,
            tau=tau,
            spread_up=max(0.0, spread_up),
            spread_down=max(0.0, spread_down),
            imbalance5_up=imb_up,
            imbalance5_down=imb_dn,
            buy_pressure=buy_pressure,
            vol_regime_ratio=vol_regime_ratio,
            mid_up_momentum=mid_momentum,
            hour_of_day=hour_of_day,
            is_weekend=is_weekend,
            asset_id=self.filtration_asset_id,
            hawkes_intensity=hawkes_intensity,
            hawkes_n_events=hawkes_n_events,
        )

        confidence = self.filtration_model.predict_proba(features)
        ctx["_filtration_confidence"] = confidence
        # In size_mult mode, never gate on confidence — the multiplier is
        # applied at the Kelly step instead. Confidence is still stored
        # in ctx so the Kelly step can read it.
        if self.filtration_mode == "size_mult":
            return True
        return confidence >= self.filtration_threshold

    def _filtration_size_multiplier(self, ctx: dict) -> float:
        """Return Kelly multiplier from filtration confidence in size_mult mode.

        Two cases depending on the filtration model's target_type:

        Classification model (legacy):
          confidence is P(direction correct) ∈ [0, 1].
          Linear interpolation from filtration_size_mult_floor (mult=0)
          to 1.0 (mult=1). Default floor 0.45.

        Regression model (filtration_model_pnl.pkl):
          confidence is predicted_pnl_per_dollar (typically -1.0 to ~+1.0).
          Linear interpolation from 0.0 (mult=0) to filtration_ev_full (mult=1).
          Below 0 → zero size (predicted negative EV → don't trade).
          Above filtration_ev_full → full size.
          Default ceiling 0.50 (i.e. 50% per-dollar EV → max sizing).

        Returns 1.0 when:
          - filtration_mode != "size_mult" (no-op)
          - no filtration model loaded
          - confidence not in ctx (early-exit fired in _check_filtration)

        That last case is important: when |z| < 0.10 the filtration check
        early-exits and never populates _filtration_confidence — those
        trades shouldn't be downsized. Returning 1.0 keeps them at the
        baseline Kelly size.
        """
        if self.filtration_mode != "size_mult" or self.filtration_model is None:
            return 1.0
        conf = ctx.get("_filtration_confidence")
        if conf is None:
            return 1.0
        target_type = getattr(self.filtration_model, "target_type", "classification")
        if target_type == "regression":
            # conf is predicted PnL per dollar; map to [0, 1] mult
            if conf <= 0.0:
                return 0.0
            ceiling = self.filtration_ev_full
            if conf >= ceiling:
                return 1.0
            return float(conf / ceiling)
        # classification
        floor = self.filtration_size_mult_floor
        if conf <= floor:
            return 0.0
        # Linear from (floor, 0) to (1.0, 1.0)
        return float((conf - floor) / (1.0 - floor))

    def _smoothed_sigma_p(self, raw: float, ctx: dict) -> float:
        """EMA smoothing for contract mid vol (separate state from BTC sigma)."""
        if raw == 0.0:
            return 0.0
        ema = ctx.get("_sigma_p_ema")
        if ema is None:
            ema = raw
        else:
            ema = self.sigma_ema_alpha * raw + (1 - self.sigma_ema_alpha) * ema
        ctx["_sigma_p_ema"] = ema
        return ema

    def _contract_sigma_p(self, ctx: dict) -> float:
        """Realized vol of contract mid price (sigma_p per second).

        Uses the same Yang-Zhang estimator as BTC vol but on the UP
        contract mid = (best_bid + best_ask) / 2 over a rolling window.
        """
        mids = ctx.get("_contract_mids", [])
        ts = ctx.get("_contract_mid_ts", [])
        if len(mids) < 10:
            return 0.0  # insufficient data — min_edge floor covers warmup
        cutoff = ts[-1] - self.contract_vol_lookback_s * 1000
        start = next((i for i, t in enumerate(ts) if t >= cutoff), 0)
        raw = _compute_vol_deduped(mids[start:], ts[start:])
        return self._smoothed_sigma_p(raw, ctx)

    @staticmethod
    def _compute_toxicity(snapshot: "Snapshot", max_spread: float) -> float:
        """Composite toxicity score in [0, 1] from microstructure signals.

        Components (each normalized to [0, 1]):
          1. Spread width:   avg(spread_up, spread_down) / max_spread
          2. Book imbalance: abs(total_bid_depth - total_ask_depth) / total_depth
          3. Parity gap:     abs(up_mid + down_mid - 1.0)

        Final score = weighted average (40% spread, 30% imbalance, 30% gap).
        """
        bid_up = snapshot.best_bid_up or 0.0
        ask_up = snapshot.best_ask_up or 1.0
        bid_down = snapshot.best_bid_down or 0.0
        ask_down = snapshot.best_ask_down or 1.0

        # 1. Normalized spread
        spread_up = ask_up - bid_up
        spread_down = ask_down - bid_down
        avg_spread = (spread_up + spread_down) / 2.0
        spread_score = min(1.0, avg_spread / max_spread) if max_spread > 0 else 0.0

        # 2. Book imbalance across all levels
        bid_depth_up = sum(sz for _, sz in snapshot.bid_levels_up)
        ask_depth_up = sum(sz for _, sz in snapshot.ask_levels_up)
        bid_depth_down = sum(sz for _, sz in snapshot.bid_levels_down)
        ask_depth_down = sum(sz for _, sz in snapshot.ask_levels_down)
        total_bid = bid_depth_up + bid_depth_down
        total_ask = ask_depth_up + ask_depth_down
        total_depth = total_bid + total_ask
        imbalance_score = abs(total_bid - total_ask) / total_depth if total_depth > 0 else 0.0

        # 3. Mid-parity deviation
        # Book mid = average of up-mid and (1 - down-mid) — they should agree
        mid_up = (bid_up + ask_up) / 2.0
        mid_down = (bid_down + ask_down) / 2.0
        # Implied probability from up and down mids should sum to ~1.0
        # Deviation from this parity signals mispricing / toxic flow
        parity_gap = abs((mid_up + mid_down) - 1.0)
        gap_score = min(1.0, parity_gap / 0.10)  # normalize: 0.10 gap = score 1.0

        return 0.40 * spread_score + 0.30 * imbalance_score + 0.30 * gap_score

    @staticmethod
    def _compute_vpin(bars, window: int) -> float:
        """Volume-weighted VPIN: sum(|sell - buy|) / sum(total) over last W bars.

        Returns 0.0 when insufficient bars (graceful warmup).
        """
        if len(bars) < window:
            return 0.0
        recent = list(bars)[-window:]
        imbalance_sum = 0.0
        volume_sum = 0.0
        for buy_vol, sell_vol in recent:
            imbalance_sum += abs(sell_vol - buy_vol)
            volume_sum += buy_vol + sell_vol
        if volume_sum <= 0:
            return 0.0
        return imbalance_sum / volume_sum

    @staticmethod
    def _compute_oracle_lag(chainlink_price: float, binance_mid) -> float:
        """Price discrepancy between Binance mid and Chainlink oracle.

        Returns abs(binance_mid - chainlink) / chainlink.
        Returns 0.0 if binance_mid is None (graceful degradation).
        """
        if binance_mid is None or chainlink_price <= 0:
            return 0.0
        return abs(binance_mid - chainlink_price) / chainlink_price

    def _oracle_lead_bias(self, chainlink_price: float, binance_mid) -> float:
        """Signed bias on p_model from Binance vs Chainlink lead-lag.

        F4: when binance_mid > chainlink_price, the next chainlink
        update will likely move UP (rebroadcast tax = ~1.2s, so
        Binance is reading the future Chainlink). Bias p_model
        toward UP. Inverse for negative gap.

        Returns a value in [-oracle_lead_bias, +oracle_lead_bias],
        scaled by gap / oracle_lag_threshold and clipped to ±1.
        Returns 0.0 when:
          - oracle_lead_bias parameter is 0 (disabled)
          - binance_mid is None (graceful degradation)
          - chainlink_price <= 0
        """
        if (self.oracle_lead_bias == 0.0
                or binance_mid is None
                or chainlink_price <= 0):
            return 0.0
        gap = (binance_mid - chainlink_price) / chainlink_price
        if self.oracle_lag_threshold <= 0:
            return 0.0
        scaled = gap / self.oracle_lag_threshold
        scaled = max(-1.0, min(1.0, scaled))
        return self.oracle_lead_bias * scaled

    def _check_stale_features(self, ctx: dict) -> str | None:
        """Return a reason string if any input feed is stale, else None.

        Live trader populates the *_age_ms fields in ctx every tick. The
        backtest engine does not, so all gates are no-ops in backtest by
        construction (`age is None` short-circuits to "fresh"). The same
        is true of `max_book_age_ms` — see decide_both_sides for the
        existing pattern this mirrors.

        The thresholds are configured per-market in market_config.py and
        passed through DiffusionSignal.__init__.
        """
        if self.max_chainlink_age_ms is not None:
            age = ctx.get("_chainlink_age_ms")
            if age is not None and age > self.max_chainlink_age_ms:
                return (f"stale chainlink ({age:.0f}ms > "
                        f"{self.max_chainlink_age_ms:.0f}ms)")
        if self.max_binance_age_ms is not None:
            age = ctx.get("_binance_age_ms")
            if age is not None and age > self.max_binance_age_ms:
                return (f"stale binance ({age:.0f}ms > "
                        f"{self.max_binance_age_ms:.0f}ms)")
        if self.max_trade_tape_age_ms is not None:
            age = ctx.get("_trade_tape_age_ms")
            if age is not None and age > self.max_trade_tape_age_ms:
                return (f"stale trade tape ({age:.0f}ms > "
                        f"{self.max_trade_tape_age_ms:.0f}ms)")
        return None

    def _model_cdf(self, z: float, ctx: dict) -> float:
        """CDF dispatch: normal, student_t, kou, or market_adaptive."""
        if self.tail_mode == "normal":
            return norm_cdf(z)
        if self.tail_mode == "kou":
            # Our sigma_per_s is estimated from realized log returns and
            # already absorbs whatever jump variance was present in the
            # lookback window — i.e. it is a *total* per-second vol
            # estimate, not a continuous-component estimate. Under that
            # interpretation the right physical-measure CDF for binary
            # prediction is the plain Gaussian with mu=0:
            #     P(S_T > S_0) ≈ Phi(delta / (sigma * sqrt(tau))) = Phi(z)
            #
            # Earlier versions added a `drift_z = -lambda*zeta*sqrt(tau)/sigma`
            # term where zeta is the Kou *risk-neutral* martingale correction.
            # That correction belongs in option pricing under Q-measure, not
            # in physical-measure binary prediction. Worse, drift_z scales
            # as 1/sigma, so when sigma hits min_sigma (e.g. weekends) it
            # produced a -1.7% to -7% systematic downward bias on p_UP that
            # asymmetrically favored BUY_DOWN trades. Removed.
            #
            # For the proper Kou with jump-variance fattening without
            # double-counting, use tail_mode="kou_full" (below), which
            # uses bipower variation for the continuous σ component.
            return norm_cdf(z)
        if self.tail_mode == "kou_full":
            # Proper Kou jump-diffusion under physical measure.
            #
            # σ input to kou_cdf must be the CONTINUOUS-component σ
            # (bipower variation), so the function's internal
            # `lam_tau*ej2` jump-variance addition doesn't double-count
            # what our total-σ already absorbs.
            #
            # `mu_override=0.0` = physical-measure drift (short-horizon
            # crypto has ~zero expected log-return per second). The
            # kou_cdf default is the Q-measure martingale correction
            # which would reintroduce the -1.7% to -7% p_UP bias we
            # fought to remove from the plain "kou" path.
            sigma_cont = ctx.get("_sigma_continuous_per_s", 0.0)
            tau = ctx.get("_tau", 300.0)
            if sigma_cont <= 0 or tau <= 0:
                return norm_cdf(z)  # graceful fallback
            delta_log = ctx.get("_delta_log", 0.0)
            # P(UP) = P(log(S_T/S_0) > 0 | observed log(S_t/S_0) = delta_log)
            #       = P(log(S_T/S_t) > -delta_log)   (future increment)
            #       = 1 - kou_cdf(-delta_log, σ_cont, λ, p_up, η1, η2, τ)
            p_down = kou_cdf(
                -delta_log, sigma_cont,
                self.kou_lambda, self.kou_p_up,
                self.kou_eta1, self.kou_eta2, tau,
                mu_override=0.0,
            )
            return 1.0 - p_down
        if self.tail_mode == "market_adaptive":
            p_gbm = norm_cdf(z)
            market_mid = ctx.get("_market_mid", 0.5)
            choppiness = ctx.get("_choppiness", 1.0)
            elapsed_frac = ctx.get("_elapsed_frac", 0.5)

            # 1. Blend GBM with market price
            alpha = self.market_blend_alpha
            p_blend = alpha * p_gbm + (1.0 - alpha) * market_mid

            # 2. Choppiness discount: pull toward 0.5 when choppy
            # choppiness = actual_crossovers / expected_crossovers
            # >1 means choppier than GBM predicts → less confident
            chop_factor = min(1.0, 1.0 / max(choppiness, 0.3))
            p_chop = chop_factor * p_blend + (1.0 - chop_factor) * 0.5

            # 3. Time-confidence: S-curve, low at 0-25%, rises through 50%+
            # Logistic: f(t) = 1 / (1 + exp(-k*(t - t0)))
            t_conf = 1.0 / (1.0 + math.exp(-8.0 * (elapsed_frac - 0.35)))
            p_final = t_conf * p_chop + (1.0 - t_conf) * 0.5
            return p_final
        # student_t fallback
        nu = ctx.get("_tail_nu", self.tail_nu_default)
        return fast_t_cdf(z, nu)

    def _p_model(self, z_capped: float, tau: float, ctx: dict | None = None) -> float:
        """Model probability of UP via Bayesian fusion of GBM + calibration.

        p = w * p_calibrated + (1 - w) * p_gbm
        w = n / (n + n0)

        With few observations, leans on GBM prior.
        With many observations, leans on calibration.
        """
        p_gbm = self._model_cdf(z_capped, ctx) if ctx is not None else norm_cdf(z_capped)
        if self.calibration_table is not None:
            p_cal, n = self.calibration_table.lookup_with_count(z_capped, tau)
            if n > 0:
                w = min(n / (n + self.cal_prior_strength), self.cal_max_weight)
                return w * p_cal + (1 - w) * p_gbm
        return p_gbm

    def decide(self, snapshot: Snapshot, ctx: dict) -> Decision:
        effective_price = ctx.get("_binance_mid") or snapshot.chainlink_price

        # Build price + timestamp history
        hist = ctx.setdefault("price_history", [])
        ts_hist = ctx.setdefault("ts_history", [])
        if ctx.pop("_live_history_appended", False):
            pass  # signal_ticker already appended this tick
        else:
            hist.append(effective_price)
            ts_hist.append(snapshot.ts_ms)

        if len(hist) < 2:
            return Decision("FLAT", 0.0, 0.0,
                            f"need history ({len(hist)}s collected)")

        ask_up = snapshot.best_ask_up
        ask_down = snapshot.best_ask_down
        bid_up = snapshot.best_bid_up
        bid_down = snapshot.best_bid_down
        if ask_up is None or ask_down is None or bid_up is None or bid_down is None:
            return Decision("FLAT", 0.0, 0.0, "missing book")
        if ask_up <= 0 or ask_up >= 1 or ask_down <= 0 or ask_down >= 1:
            return Decision("FLAT", 0.0, 0.0, "invalid asks")

        # Stale-feature gates (parity with decide_both_sides). The
        # single-side path was previously missing the book-age gate
        # entirely, so any `max_book_age_ms` setting in market_config
        # was silently inactive in FOK mode. The four checks below now
        # also fire in single-side mode, matching the dual-side path.
        # All four are no-ops in backtest because BacktestEngine never
        # populates the *_age_ms ctx fields.
        if self.max_book_age_ms is not None:
            book_age = ctx.get("_book_age_ms")
            if book_age is not None and book_age > self.max_book_age_ms:
                return Decision("FLAT", 0.0, 0.0,
                                f"stale book ({book_age:.0f}ms > "
                                f"{self.max_book_age_ms}ms)")
        stale_reason = self._check_stale_features(ctx)
        if stale_reason is not None:
            return Decision("FLAT", 0.0, 0.0, stale_reason)

        self._record_book_state(snapshot, ctx)

        # Spread gate: wide spreads signal uncertain/illiquid pricing
        spread_up = ask_up - bid_up
        spread_down = ask_down - bid_down
        if spread_up > self.max_spread or spread_down > self.max_spread:
            return Decision("FLAT", 0.0, 0.0,
                f"spread too wide (up={spread_up:.3f} down={spread_down:.3f} "
                f"max={self.max_spread})")

        # Microstructure toxicity
        toxicity = self._compute_toxicity(snapshot, self.max_spread)
        ctx["_toxicity"] = toxicity

        tau = snapshot.time_remaining_s
        if tau <= 0:
            return Decision("FLAT", 0.0, 0.0, "window expired")

        # Late-entry gate: only trade when time remaining <= max_entry_time_s
        if self.max_entry_time_s is not None and tau > self.max_entry_time_s:
            return Decision("FLAT", 0.0, 0.0,
                f"too early ({tau:.0f}s left > {self.max_entry_time_s:.0f}s gate)")

        # Dynamic edge threshold: higher early, decays with sqrt(tau)
        dyn_threshold = self.edge_threshold * (
            1.0 + self.early_edge_mult * math.sqrt(tau / self.window_duration)
        )

        # Toxicity penalty: widen threshold in adverse microstructure
        if toxicity > self.toxicity_threshold:
            excess = (toxicity - self.toxicity_threshold) / (1.0 - self.toxicity_threshold)
            dyn_threshold *= 1.0 + self.toxicity_edge_mult * excess

        # VPIN flow toxicity penalty
        trade_bars = ctx.get("_trade_bars", [])
        vpin = self._compute_vpin(trade_bars, self.vpin_window)
        ctx["_vpin"] = vpin
        if vpin > self.vpin_threshold:
            vpin_excess = (vpin - self.vpin_threshold) / (1.0 - self.vpin_threshold)
            dyn_threshold *= 1.0 + self.vpin_edge_mult * vpin_excess

        # Oracle lag penalty: widen threshold when Binance mid diverges from Chainlink
        binance_mid = ctx.get("_binance_mid")
        oracle_lag = self._compute_oracle_lag(snapshot.chainlink_price, binance_mid)
        ctx["_oracle_lag"] = oracle_lag
        # Hard cancel: skip this tick entirely when gap exceeds cancel threshold
        if self.oracle_cancel_threshold > 0 and oracle_lag > self.oracle_cancel_threshold:
            return Decision("FLAT", 0.0, 0.0,
                            f"oracle hard cancel (lag={oracle_lag:.4f} > {self.oracle_cancel_threshold:.4f})")
        if oracle_lag > self.oracle_lag_threshold:
            lag_excess = min((oracle_lag - self.oracle_lag_threshold) / self.oracle_lag_threshold, 1.0)
            dyn_threshold *= 1.0 + self.oracle_lag_mult * lag_excess

        # Realized vol (short window for model)
        raw_sigma = self._compute_vol(hist[-self.vol_lookback_s:], ts_hist[-self.vol_lookback_s:])

        # Vol regime filter: compare recent vol to longer baseline
        sigma_baseline = 0.0
        if len(hist) >= self.vol_regime_lookback_s:
            sigma_baseline = self._compute_vol(hist[-self.vol_regime_lookback_s:], ts_hist[-self.vol_regime_lookback_s:])
            if sigma_baseline > 0 and raw_sigma > self.vol_regime_mult * sigma_baseline:
                return Decision("FLAT", 0.0, 0.0,
                    f"vol spike ({raw_sigma:.2e} > "
                    f"{self.vol_regime_mult}x baseline {sigma_baseline:.2e})")

        # Adaptive sigma floor: prevent the short-window σ from dropping
        # much below the long-window baseline. This stops the "sigma
        # collapse" pattern where 90s realized variance briefly bottoms
        # out during a quiet sub-period of an otherwise volatile session,
        # which would otherwise blow up z = delta / (sigma · √τ) and
        # spike p_model into the cap for one tick (causing rogue trades
        # that disappear next tick when sigma recovers). Floor is
        # MIN_SIGMA_RATIO * sigma_baseline.
        if sigma_baseline > 0:
            adaptive_floor = MIN_SIGMA_RATIO * sigma_baseline
            if raw_sigma < adaptive_floor:
                raw_sigma = adaptive_floor

        # EMA smoothing + asset cap
        sigma_per_s = self._smoothed_sigma(raw_sigma, ctx)
        if sigma_per_s == 0.0:
            return Decision("FLAT", 0.0, 0.0, "zero vol")

        # Hawkes intensity (opt-in feature; published to ctx, no gating)
        self._maybe_publish_hawkes(hist, ts_hist, sigma_per_s, ctx)

        # Vol kill switch
        if self.vol_kill_sigma is not None and sigma_per_s > self.vol_kill_sigma:
            return Decision("FLAT", 0.0, 0.0,
                f"vol kill switch (sigma={sigma_per_s:.2e} "
                f"> {self.vol_kill_sigma:.2e})")

        self._maybe_update_tail_nu(hist, ctx)

        # Near-expiry Chainlink blend (same as decide_both_sides)
        price = effective_price
        if (self.chainlink_blend_s > 0
                and tau < self.chainlink_blend_s
                and snapshot.chainlink_price
                and snapshot.chainlink_price > 0):
            blend_w = 1.0 - (tau / self.chainlink_blend_s)
            price = (1.0 - blend_w) * effective_price + blend_w * snapshot.chainlink_price

        # Model probability (z capped to prevent overconfidence)
        delta = (price - snapshot.window_start_price) / snapshot.window_start_price
        z_raw = delta / (sigma_per_s * math.sqrt(tau))

        # Regime-scaled z (same as decide_both_sides)
        z_raw, _ = self._apply_regime_z_scale(z_raw, sigma_per_s, ctx)

        z = max(-self.max_z, min(self.max_z, z_raw))
        ctx["_z_raw"] = z_raw
        ctx["_z"] = z
        # Populate context for kou / market_adaptive / kou_full CDFs
        ctx["_sigma_per_s"] = sigma_per_s
        ctx["_tau"] = tau
        # Uncapped log-delta for kou_full (kou_cdf handles its own
        # distribution spread so no reason to cap delta like we do z)
        if price > 0 and snapshot.window_start_price > 0:
            ctx["_delta_log"] = math.log(price / snapshot.window_start_price)
        else:
            ctx["_delta_log"] = 0.0
        # Continuous-component σ via bipower variation — only when the
        # model tail mode needs it. BV is cheap (one mean over a product)
        # but still worth skipping on the default path.
        if self.tail_mode == "kou_full":
            from scripts.sigma_estimators import bipower_variation_per_s
            ctx["_sigma_continuous_per_s"] = bipower_variation_per_s(
                hist[-self.vol_lookback_s:], ts_hist[-self.vol_lookback_s:]
            )
        # Window duration — use self.window_duration (config-wired), not
        # snapshot.time_remaining_s, to avoid baking the "late-join" value
        # when the bot wakes mid-window. See decide_both_sides for rationale.
        dur_s = self.window_duration if self.window_duration > 0 else (tau + 1.0)
        ctx["_window_duration_s"] = dur_s
        ctx["_elapsed_frac"] = max(0.0, min(1.0, 1.0 - tau / dur_s)) if dur_s > 0 else 0.5
        ctx["_market_mid"] = getattr(snapshot, "mid_up", 0.5) or 0.5
        # choppiness: track zero-crossings of delta
        _prev_sign = ctx.get("_prev_delta_sign", 0)
        _cur_sign = 1 if delta > 0 else (-1 if delta < 0 else 0)
        if _prev_sign != 0 and _cur_sign != 0 and _cur_sign != _prev_sign:
            ctx["_crossovers"] = ctx.get("_crossovers", 0) + 1
        ctx["_prev_delta_sign"] = _cur_sign
        # expected crossovers under GBM ≈ sqrt(2 * elapsed / pi) * sigma_factor
        elapsed_s = dur_s - tau
        expected_xo = max(1.0, math.sqrt(2.0 * max(elapsed_s, 1.0) / math.pi))
        ctx["_choppiness"] = ctx.get("_crossovers", 0) / expected_xo

        # Minimum z gate: filter low-conviction setups
        if self.min_entry_z > 0 and abs(z) < self.min_entry_z:
            return Decision("FLAT", 0.0, 0.0,
                            f"min_z gate (|z|={abs(z):.3f} < {self.min_entry_z:.3f})")

        # Cross-asset disagreement veto
        if self.cross_asset_z_lookup is not None:
            wstart = ctx.get("_window_start_ms")
            z2 = _lookup_cross_asset_z(self.cross_asset_z_lookup, wstart, tau)
            if (z2 is not None
                    and abs(z) >= self.cross_asset_min_z
                    and abs(z2) >= self.cross_asset_min_z
                    and z * z2 < 0):
                return Decision("FLAT", 0.0, 0.0,
                                f"cross-asset veto (z={z:.3f}, z2={z2:.3f})")

        p_model = self._p_model(z, tau, ctx)
        ctx["_p_model_raw"] = p_model
        ctx["_p_display"] = norm_cdf(z)
        ctx["_sigma_per_s"] = sigma_per_s

        # Model-vs-market disagreement gate. When the diffusion model
        # disagrees with the contract mid by more than X percentage
        # points, the model is probably overconfident due to a σ
        # collapse (sustained low-vol windows produce confidently
        # incorrect z-scores). Skip the trade — trust the market.
        if self.max_model_market_disagreement < 1.0:
            mid_up_obs = (bid_up + ask_up) / 2.0
            disagreement = abs(p_model - mid_up_obs)
            if disagreement > self.max_model_market_disagreement:
                return Decision("FLAT", 0.0, 0.0,
                    f"model-market disagreement (|p_model={p_model:.2f} - "
                    f"mid={mid_up_obs:.2f}|={disagreement:.2f} > "
                    f"{self.max_model_market_disagreement:.2f})")

        # F4: oracle lead-lag bias. Apply BEFORE filtration so the
        # filtration model's z/p features see the bias-adjusted view.
        # No-op when oracle_lead_bias=0 or binance_mid is unavailable.
        lead_bias = self._oracle_lead_bias(snapshot.chainlink_price, binance_mid)
        if lead_bias != 0.0:
            p_model = max(0.01, min(0.99, p_model + lead_bias))
            ctx["_oracle_lead_bias"] = lead_bias

        # Filtration gate: XGBoost confidence check
        if not self._check_filtration(z, sigma_per_s, tau, snapshot, ctx):
            conf = ctx.get("_filtration_confidence", 0.0)
            return Decision("FLAT", 0.0, 0.0,
                            f"filtration ({conf:.3f} < {self.filtration_threshold})")

        # Mean-reversion discount: pull p_model toward 0.5
        if self.reversion_discount > 0:
            p_model = p_model * (1 - self.reversion_discount) + 0.5 * self.reversion_discount

        # Market blend: pull p_model toward contract mid (anti-fade-the-market)
        if self.market_blend > 0:
            mid_up = (bid_up + ask_up) / 2.0
            p_model = (1.0 - self.market_blend) * p_model + self.market_blend * mid_up
            p_model = max(0.01, min(0.99, p_model))

        ctx["_p_model_trade"] = p_model

        # Effective costs (taker mode)
        p_up_cost = ask_up + poly_fee(ask_up) + self.slippage
        p_down_cost = ask_down + poly_fee(ask_down) + self.slippage

        # Edges (penalized by spread — wider spread = less trustworthy pricing)
        edge_up = p_model - p_up_cost - self.spread_edge_penalty * spread_up
        edge_down = (1.0 - p_model) - p_down_cost - self.spread_edge_penalty * spread_down
        ctx["_edge_up"] = edge_up
        ctx["_edge_down"] = edge_down
        ctx["_dyn_threshold_up"] = dyn_threshold
        ctx["_dyn_threshold_down"] = dyn_threshold

        if edge_up >= edge_down and edge_up > dyn_threshold:
            side, edge, eff_price, p_side = "BUY_UP", edge_up, p_up_cost, p_model
        elif edge_down > edge_up and edge_down > dyn_threshold:
            side, edge, eff_price, p_side = "BUY_DOWN", edge_down, p_down_cost, 1.0 - p_model
        else:
            return Decision("FLAT", 0.0, 0.0,
                            f"no edge (up={edge_up:.4f} down={edge_down:.4f} "
                            f"thresh={dyn_threshold:.4f})")

        if eff_price > self.max_entry_price:
            return Decision("FLAT", 0.0, 0.0,
                f"entry {eff_price:.3f} > max {self.max_entry_price:.2f}")

        # Momentum confirmation: majority of recent prices must be on
        # the same side of start price — prevents whipsawing when price
        # oscillates near the start price.
        start_px = snapshot.window_start_price
        mom_n = min(self.momentum_lookback_s, len(hist))
        if mom_n >= 2:
            mom_prices = hist[-mom_n:]
            if side == "BUY_UP":
                frac_ok = sum(1 for p in mom_prices if p >= start_px) / len(mom_prices)
                if frac_ok < self.momentum_majority:
                    return Decision("FLAT", 0.0, 0.0,
                        f"momentum fail: only {frac_ok:.0%} above start in last {mom_n}s")
            else:
                frac_ok = sum(1 for p in mom_prices if p <= start_px) / len(mom_prices)
                if frac_ok < self.momentum_majority:
                    return Decision("FLAT", 0.0, 0.0,
                        f"momentum fail: only {frac_ok:.0%} below start in last {mom_n}s")

        # Order book imbalance: require buy pressure on the chosen side.
        # imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
        # When `use_kalman_obi=True` (opt-in), the gate consumes a
        # Kalman-smoothed OBI from notebook #95 instead of the raw L2
        # snapshot. The raw value is always recorded in ctx for diagnostics.
        if side == "BUY_UP":
            bid_d = sum(sz for _, sz in snapshot.bid_levels_up)
            ask_d = sum(sz for _, sz in snapshot.ask_levels_up)
            kalman_key = "_obi_kalman_up"
        else:
            bid_d = sum(sz for _, sz in snapshot.bid_levels_down)
            ask_d = sum(sz for _, sz in snapshot.ask_levels_down)
            kalman_key = "_obi_kalman_down"
        total_d = bid_d + ask_d
        imbalance_raw = (bid_d - ask_d) / total_d if total_d > 0 else 0.0
        ctx["_obi_raw"] = imbalance_raw
        if self.use_kalman_obi:
            imbalance_gate = self._kalman_obi_update(
                kalman_key, imbalance_raw, ctx
            )
            ctx["_obi_smooth"] = imbalance_gate
        else:
            imbalance_gate = imbalance_raw
        if imbalance_gate < 0:
            return Decision("FLAT", 0.0, 0.0,
                f"imbalance disagrees ({imbalance_gate:+.3f} for {side})")

        # Delta velocity: require price moving in the direction of the bet.
        # OLS slope over last 30s of BTC prices.
        vel_n = min(30, len(hist))
        if vel_n >= 5:
            recent = hist[-vel_n:]
            x_mean = (vel_n - 1) / 2.0
            y_mean = sum(recent) / vel_n
            num = sum((i - x_mean) * (p - y_mean) for i, p in enumerate(recent))
            den = sum((i - x_mean) ** 2 for i in range(vel_n))
            slope = num / den if den > 0 else 0.0
            if side == "BUY_UP" and slope < 0:
                return Decision("FLAT", 0.0, 0.0,
                    f"delta velocity disagrees (slope={slope:+.2f} $/s for BUY_UP)")
            elif side == "BUY_DOWN" and slope > 0:
                return Decision("FLAT", 0.0, 0.0,
                    f"delta velocity disagrees (slope={slope:+.2f} $/s for BUY_DOWN)")

        if eff_price >= 1.0:
            return Decision("FLAT", 0.0, 0.0, "eff price >= 1")

        # Half-Kelly with filtration-based sizing modifier.
        # Note: the regime classifier (_maybe_compute_regime) is
        # intentionally NOT called here — it always returns 1.0 on real
        # data (high_acc state), and decide_both_sides (the live path)
        # never called it either. Removing it normalizes the two code
        # paths. If the HMM is retrained with aggressive multipliers
        # in the future, wire it into _size_decision (shared by both
        # paths) instead of calling it in decide() alone.
        kelly_f = max(0.0, (p_side - eff_price) / (1.0 - eff_price))
        filt_mult = self._filtration_size_multiplier(ctx)
        kelly_fraction_adj = self.kelly_fraction * filt_mult
        frac = min(kelly_fraction_adj * kelly_f, self.max_bet_fraction)
        if frac <= 0:
            return Decision("FLAT", 0.0, 0.0,
                            "kelly <= 0" + (
                                f" (filt_mult={filt_mult:.2f})"
                                if filt_mult <= 0.0 else ""
                            ))

        size_usd = self.bankroll * frac

        # Floor to exchange minimum (5 shares), skip if bankroll can't cover
        min_usd = self.min_order_shares * eff_price
        if size_usd < min_usd:
            if self.bankroll >= min_usd:
                size_usd = min_usd
            else:
                return Decision("FLAT", 0.0, 0.0,
                    f"bankroll ${self.bankroll:.2f} < min order ${min_usd:.2f} (5 shares)")

        # Store expected price range in ctx for the engine to attach to Fill
        price = snapshot.chainlink_price
        move_1sig = sigma_per_s * math.sqrt(tau) * price
        ctx["_expected_range"] = {
            "btc_at_fill": price,
            "start_price": snapshot.window_start_price,
            "expected_low": snapshot.window_start_price - move_1sig,
            "expected_high": snapshot.window_start_price + move_1sig,
        }

        sprd = spread_up if side == "BUY_UP" else spread_down
        reason = (
            f"p={p_model:.4f} sig={sigma_per_s:.2e} z={z:.2f}"
            f"{'(cap)' if abs(z_raw) > self.max_z else ''} "
            f"tau={tau:.0f}s edge={edge:.4f} thresh={dyn_threshold:.4f} "
            f"spread={sprd:.3f} kelly={kelly_f:.4f} ${size_usd:.0f}"
        )
        return Decision(side, edge, size_usd, reason)

    def _size_decision(
        self, side: str, edge: float, eff_price: float,
        p_side: float, snapshot: Snapshot, sigma_per_s: float,
        tau: float, z: float, z_raw: float, p_model: float,
        dyn_threshold: float, spread: float,
        ctx: dict | None = None,
    ) -> Decision:
        """Shared sizing logic for a single side. Returns a Decision.

        `ctx` is optional for backward compat — when provided, the
        filtration size multiplier (size_mult mode) is applied.
        """
        if eff_price >= 1.0:
            return Decision("FLAT", 0.0, 0.0, "eff price >= 1")
        if eff_price > self.max_entry_price:
            return Decision("FLAT", 0.0, 0.0,
                f"entry {eff_price:.3f} > max {self.max_entry_price:.2f}")
        if eff_price < self.min_entry_price:
            return Decision("FLAT", 0.0, 0.0,
                f"entry {eff_price:.2f} < min {self.min_entry_price:.2f}")

        kelly_f = max(0.0, (p_side - eff_price) / (1.0 - eff_price))
        # Filtration size multiplier (no-op in gate mode or no model)
        filt_mult = self._filtration_size_multiplier(ctx) if ctx is not None else 1.0
        frac = min(self.kelly_fraction * filt_mult * kelly_f, self.max_bet_fraction)
        if frac <= 0:
            return Decision("FLAT", 0.0, 0.0,
                            "kelly <= 0" + (
                                f" (filt_mult={filt_mult:.2f})"
                                if filt_mult <= 0.0 else ""
                            ))

        size_usd = self.bankroll * frac
        min_usd = self.min_order_shares * eff_price
        if size_usd < min_usd:
            # When filtration downsized the trade (filt_mult < 1), don't
            # silently bump back up to the floor — that defeats the
            # graduated moderation. Return FLAT instead. This means
            # filtration only fires trades at full or near-full sizing on
            # small bankrolls, which is the correct behavior: "not confident
            # enough to trade at the minimum size" = skip.
            if filt_mult < 1.0:
                return Decision("FLAT", 0.0, 0.0,
                    f"filtration size below min order "
                    f"(${size_usd:.2f} < ${min_usd:.2f}, filt_mult={filt_mult:.2f})")
            if self.bankroll >= min_usd:
                size_usd = min_usd
            else:
                return Decision("FLAT", 0.0, 0.0,
                    f"bankroll ${self.bankroll:.2f} < min order ${min_usd:.2f}")

        # Store expected range
        price = snapshot.chainlink_price
        move_1sig = sigma_per_s * math.sqrt(tau) * price

        reason = (
            f"p={p_model:.4f} sig={sigma_per_s:.2e} z={z:.2f}"
            f"{'(cap)' if abs(z_raw) > self.max_z else ''} "
            f"tau={tau:.0f}s edge={edge:.4f} thresh={dyn_threshold:.4f} "
            f"spread={spread:.3f} kelly={kelly_f:.4f} ${size_usd:.0f}"
        )
        return Decision(side, edge, size_usd, reason)

    def decide_both_sides(
        self, snapshot: Snapshot, ctx: dict,
    ) -> tuple[Decision, Decision]:
        """Evaluate Up and Down independently for maker mode.

        Returns (up_decision, down_decision) — both can be non-FLAT.
        Skips momentum, imbalance, and velocity filters.
        Uses mid-price with 0% fee for cost calculation.
        """
        flat = Decision("FLAT", 0.0, 0.0, "")

        effective_price = ctx.get("_binance_mid") or snapshot.chainlink_price

        # Build price + timestamp history (always, even during warmup, so vol is ready)
        # In live mode, signal_ticker already appended and sets a flag.
        hist = ctx.setdefault("price_history", [])
        ts_hist = ctx.setdefault("ts_history", [])
        if ctx.pop("_live_history_appended", False):
            pass  # signal_ticker already appended this tick
        else:
            hist.append(effective_price)
            ts_hist.append(snapshot.ts_ms)

        # Early p_model for display: compute whenever possible so the
        # dashboard shows live probabilities even during warmup / gates.
        # Uses raw (unsmoothed) sigma and pure GBM norm_cdf (not
        # calibration table) so the display updates continuously —
        # calibration bins are wide and would cause the value to appear
        # stuck.  The full computation later overwrites _p_model_raw
        # with the refined calibrated value for trading decisions.
        # Require enough history for a reliable vol estimate — too few
        # samples give a tiny sigma that sends z to the ±max_z cap,
        # making every market show the same p_up/p_down.
        #
        # 2026-04-09 fix: explicitly clear stale display values at the
        # top so a vol-collapse / warmup tick doesn't leave the OLD
        # _p_display value in ctx for the dashboard to read. Previously
        # the user reported "p_up stays frozen during stale book" — root
        # cause was this block silently NOT updating when _raw == 0
        # (which happens when hist contains many identical prices, e.g.
        # both feeds briefly stuck), and the dashboard kept reading the
        # last successful value indefinitely. Now we always set
        # _p_display (None when we can't compute) and _p_display_ts so
        # the dashboard can render "---" or "[stale]" when appropriate.
        ctx.pop("_p_display", None)
        ctx.pop("_p_model_raw", None)
        ctx["_p_display_ts"] = snapshot.ts_ms

        _min_hist_display = max(10, self.vol_lookback_s // 3)
        if len(hist) >= _min_hist_display and snapshot.time_remaining_s > 0 and snapshot.window_start_price:
            _raw = self._compute_vol(
                hist[-self.vol_lookback_s:], ts_hist[-self.vol_lookback_s:]
            )
            if _raw > 0:
                _tau = snapshot.time_remaining_s
                _delta = (effective_price - snapshot.window_start_price) / snapshot.window_start_price
                _z = _delta / (_raw * math.sqrt(_tau))
                _z = max(-self.max_z, min(self.max_z, _z))
                ctx["_p_display"] = norm_cdf(_z)
                ctx["_p_model_raw"] = self._p_model(_z, _tau, ctx)
                ctx["_sigma_per_s"] = _raw
                ctx["_p_display_fresh"] = True
            else:
                # Vol collapsed — flag the display as unfresh so the
                # dashboard knows the model couldn't compute this tick.
                ctx["_p_display_fresh"] = False
        else:
            ctx["_p_display_fresh"] = False

        # NOTE: maker warmup and end-of-window withdrawal are handled by
        # _evaluate_maker() in tracker.py.  We must NOT early-return here
        # because that blocks the full model computation (vol, z, p_model,
        # edges) which the dashboard and fill logs need every tick.

        if len(hist) < 2:
            reason = f"need history ({len(hist)}s collected)"
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))

        ask_up = snapshot.best_ask_up
        ask_down = snapshot.best_ask_down
        bid_up = snapshot.best_bid_up
        bid_down = snapshot.best_bid_down
        if ask_up is None or ask_down is None or bid_up is None or bid_down is None:
            reason = "missing book"
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))
        if ask_up <= 0 or ask_up >= 1 or ask_down <= 0 or ask_down >= 1:
            reason = "invalid asks"
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))

        # Stale-book gate: don't trade against a book older than max_book_age_ms.
        # The book WS occasionally drops and reconnects; during the gap the bot
        # is making decisions on minutes-old prices. Live data shows these
        # stale-book trades win 25% (vs 58% fresh) and lose ~$ for $.
        if self.max_book_age_ms is not None:
            book_age = ctx.get("_book_age_ms")
            if book_age is not None and book_age > self.max_book_age_ms:
                reason = f"stale book ({book_age:.0f}ms > {self.max_book_age_ms}ms)"
                return (Decision("FLAT", 0.0, 0.0, reason),
                        Decision("FLAT", 0.0, 0.0, reason))

        # Other stale-feature gates (chainlink / binance / trade tape).
        # Same pattern as the book-age gate above. Backtest never sets the
        # *_age_ms fields, so this is a no-op in backtest.
        stale_reason = self._check_stale_features(ctx)
        if stale_reason is not None:
            return (Decision("FLAT", 0.0, 0.0, stale_reason),
                    Decision("FLAT", 0.0, 0.0, stale_reason))

        self._record_book_state(snapshot, ctx)

        # Spread gate
        spread_up = ask_up - bid_up
        spread_down = ask_down - bid_down
        if spread_up > self.max_spread or spread_down > self.max_spread:
            reason = (f"spread too wide (up={spread_up:.3f} down={spread_down:.3f} "
                      f"max={self.max_spread})")
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))

        # Track contract mid prices for A-S realized vol (sigma_p)
        if self.as_mode:
            mid_up = (bid_up + ask_up) / 2.0
            contract_mids = ctx.setdefault("_contract_mids", [])
            contract_mid_ts = ctx.setdefault("_contract_mid_ts", [])
            contract_mids.append(mid_up)
            contract_mid_ts.append(snapshot.ts_ms)

        # VAMP computation
        vamp_up = compute_vamp(snapshot.bid_levels_up, snapshot.ask_levels_up)
        vamp_down = compute_vamp(snapshot.bid_levels_down, snapshot.ask_levels_down)

        # VAMP filter: skip when book is too gappy (large gap between VAMP and best bid)
        if self.vamp_mode == "filter":
            if (vamp_up is not None and
                    abs(vamp_up - bid_up) > self.vamp_filter_threshold):
                reason = (f"vamp filter (up gap={abs(vamp_up - bid_up):.3f} "
                          f"> {self.vamp_filter_threshold})")
                return (Decision("FLAT", 0.0, 0.0, reason),
                        Decision("FLAT", 0.0, 0.0, reason))
            if (vamp_down is not None and
                    abs(vamp_down - bid_down) > self.vamp_filter_threshold):
                reason = (f"vamp filter (down gap={abs(vamp_down - bid_down):.3f} "
                          f"> {self.vamp_filter_threshold})")
                return (Decision("FLAT", 0.0, 0.0, reason),
                        Decision("FLAT", 0.0, 0.0, reason))

        # Microstructure toxicity filter: composite score from spread,
        # book imbalance, and mid-parity gap.  Rather than a hard cutoff,
        # toxicity widens the edge threshold proportionally so we still
        # trade in mildly toxic regimes but at a higher bar.
        toxicity = self._compute_toxicity(snapshot, self.max_spread)
        ctx["_toxicity"] = toxicity

        tau = snapshot.time_remaining_s
        if tau <= 0:
            reason = "window expired"
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))

        # Vol — reuse the display vol from the early computation above
        # (line ~1625) if it was computed on the same history. The display
        # path calls _compute_vol with identical arguments (same lookback,
        # same hist slice); recomputing is pure waste (~100-200us saved).
        _cached_display_sigma = ctx.get("_sigma_per_s")
        if _cached_display_sigma is not None and _cached_display_sigma > 0:
            raw_sigma = _cached_display_sigma
        else:
            raw_sigma = self._compute_vol(hist[-self.vol_lookback_s:], ts_hist[-self.vol_lookback_s:])

        # Vol regime filter
        sigma_baseline = 0.0
        if len(hist) >= self.vol_regime_lookback_s:
            sigma_baseline = self._compute_vol(hist[-self.vol_regime_lookback_s:], ts_hist[-self.vol_regime_lookback_s:])
            if sigma_baseline > 0 and raw_sigma > self.vol_regime_mult * sigma_baseline:
                reason = (f"vol spike ({raw_sigma:.2e} > "
                          f"{self.vol_regime_mult}x baseline {sigma_baseline:.2e})")
                return (Decision("FLAT", 0.0, 0.0, reason),
                        Decision("FLAT", 0.0, 0.0, reason))

        # Adaptive sigma floor (see decide() for explanation).
        if sigma_baseline > 0:
            adaptive_floor = MIN_SIGMA_RATIO * sigma_baseline
            if raw_sigma < adaptive_floor:
                raw_sigma = adaptive_floor

        # EMA smoothing + asset cap
        sigma_per_s = self._smoothed_sigma(raw_sigma, ctx)
        if sigma_per_s == 0.0:
            reason = "zero vol"
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))

        # Hawkes intensity (opt-in feature; published to ctx, no gating).
        # Same call as decide() — needed in both paths so the filtration
        # model sees populated _hawkes_intensity / _hawkes_n_events.
        self._maybe_publish_hawkes(hist, ts_hist, sigma_per_s, ctx)

        # Volatility kill switch: hard pause when EMA sigma exceeds
        # an absolute ceiling.  Distinct from the relative regime filter
        # above — this catches sustained high-vol episodes where the
        # baseline itself has drifted up.
        if self.vol_kill_sigma is not None and sigma_per_s > self.vol_kill_sigma:
            reason = (f"vol kill switch (sigma={sigma_per_s:.2e} "
                      f"> {self.vol_kill_sigma:.2e})")
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))

        self._maybe_update_tail_nu(hist, ctx)

        # Microstructure metrics: compute excess values for threshold/spread
        trade_bars = ctx.get("_trade_bars", [])
        vpin = self._compute_vpin(trade_bars, self.vpin_window)
        ctx["_vpin"] = vpin
        tox_excess = (max(0.0, (toxicity - self.toxicity_threshold)
                         / (1.0 - self.toxicity_threshold))
                      if toxicity > self.toxicity_threshold else 0.0)
        vpin_excess = (max(0.0, (vpin - self.vpin_threshold)
                          / (1.0 - self.vpin_threshold))
                       if vpin > self.vpin_threshold else 0.0)

        binance_mid = ctx.get("_binance_mid")
        oracle_lag = self._compute_oracle_lag(snapshot.chainlink_price, binance_mid)
        ctx["_oracle_lag"] = oracle_lag
        # Hard cancel: skip both sides when gap exceeds cancel threshold
        if self.oracle_cancel_threshold > 0 and oracle_lag > self.oracle_cancel_threshold:
            reason = f"oracle hard cancel (lag={oracle_lag:.4f} > {self.oracle_cancel_threshold:.4f})"
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))
        lag_excess = (min((oracle_lag - self.oracle_lag_threshold)
                         / self.oracle_lag_threshold, 1.0)
                      if oracle_lag > self.oracle_lag_threshold else 0.0)

        # Legacy multiplicative threshold (used when as_mode=False)
        if not self.as_mode:
            window_trades = ctx.get("window_trade_count", 0)
            base_threshold = self.edge_threshold + self.edge_threshold_step * window_trades
            dyn_threshold = base_threshold * (
                1.0 + self.early_edge_mult * math.sqrt(tau / self.window_duration)
            )
            if tox_excess > 0:
                dyn_threshold *= 1.0 + self.toxicity_edge_mult * tox_excess
            if vpin_excess > 0:
                dyn_threshold *= 1.0 + self.vpin_edge_mult * vpin_excess
            if lag_excess > 0:
                dyn_threshold *= 1.0 + self.oracle_lag_mult * lag_excess
        else:
            dyn_threshold = 0.0  # placeholder, A-S computes opt_spread later

        # Near-expiry Chainlink blend: as tau → 0, Chainlink determines
        # resolution so we blend the pricing price from Binance toward
        # Chainlink to avoid betting on a Binance-Chainlink divergence.
        # Vol estimation keeps using Binance (more accurate tick data).
        price = effective_price
        chainlink_blend_w = 0.0
        if (self.chainlink_blend_s > 0
                and tau < self.chainlink_blend_s
                and snapshot.chainlink_price
                and snapshot.chainlink_price > 0):
            chainlink_blend_w = 1.0 - (tau / self.chainlink_blend_s)
            price = ((1.0 - chainlink_blend_w) * effective_price
                     + chainlink_blend_w * snapshot.chainlink_price)
        ctx["_chainlink_blend_w"] = chainlink_blend_w

        # z normalization: fractional delta / (sigma * sqrt(tau))
        delta = (price - snapshot.window_start_price) / snapshot.window_start_price
        z_raw = delta / (sigma_per_s * math.sqrt(tau))

        # Regime-scaled z: adjust z for current vol regime relative to
        # the calibration-period vol.  High-vol regimes shrink z (wider
        # distribution → less confident directional signal), low-vol
        # regimes amplify it.  Clamped to [0.5, 2.0] to avoid extreme
        # suppression or overconfidence from a single bar.
        z_raw, _ = self._apply_regime_z_scale(z_raw, sigma_per_s, ctx)

        z = max(-self.max_z, min(self.max_z, z_raw))
        ctx["_z_raw"] = z_raw
        ctx["_z"] = z
        # Populate context for kou / market_adaptive / kou_full CDFs
        ctx["_sigma_per_s"] = sigma_per_s
        ctx["_tau"] = tau
        # Uncapped log-delta for kou_full (kou_cdf handles its own spread)
        if price > 0 and snapshot.window_start_price > 0:
            ctx["_delta_log"] = math.log(price / snapshot.window_start_price)
        else:
            ctx["_delta_log"] = 0.0
        # Continuous-component σ via bipower variation — only computed
        # when the model tail mode needs it.
        if self.tail_mode == "kou_full":
            from scripts.sigma_estimators import bipower_variation_per_s
            ctx["_sigma_continuous_per_s"] = bipower_variation_per_s(
                hist[-self.vol_lookback_s:], ts_hist[-self.vol_lookback_s:]
            )
        # Window duration — always use the config value (self.window_duration)
        # as the stable reference. The previous implementation used
        # `snapshot.time_remaining_s` on the first tick, which baked a wrong
        # "duration" into ctx if the bot joined mid-window (e.g. 400s into a
        # 900s window stored dur_s=400 forever, and elapsed_frac went negative).
        dur_s = self.window_duration if self.window_duration > 0 else (tau + 1.0)
        ctx["_window_duration_s"] = dur_s
        ctx["_elapsed_frac"] = max(0.0, min(1.0, 1.0 - tau / dur_s)) if dur_s > 0 else 0.5
        ctx["_market_mid"] = getattr(snapshot, "mid_up", 0.5) or 0.5
        _prev_sign = ctx.get("_prev_delta_sign", 0)
        _cur_sign = 1 if delta > 0 else (-1 if delta < 0 else 0)
        if _prev_sign != 0 and _cur_sign != 0 and _cur_sign != _prev_sign:
            ctx["_crossovers"] = ctx.get("_crossovers", 0) + 1
        ctx["_prev_delta_sign"] = _cur_sign
        elapsed_s = dur_s - tau
        expected_xo = max(1.0, math.sqrt(2.0 * max(elapsed_s, 1.0) / math.pi))
        ctx["_choppiness"] = ctx.get("_crossovers", 0) / expected_xo

        # Minimum z gate: filter low-conviction setups
        if self.min_entry_z > 0 and abs(z) < self.min_entry_z:
            reason = f"min_z gate (|z|={abs(z):.3f} < {self.min_entry_z:.3f})"
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))

        # Cross-asset disagreement veto
        if self.cross_asset_z_lookup is not None:
            wstart = ctx.get("_window_start_ms")
            z2 = _lookup_cross_asset_z(self.cross_asset_z_lookup, wstart, tau)
            if (z2 is not None
                    and abs(z) >= self.cross_asset_min_z
                    and abs(z2) >= self.cross_asset_min_z
                    and z * z2 < 0):
                reason = f"cross-asset veto (z={z:.3f}, z2={z2:.3f})"
                return (Decision("FLAT", 0.0, 0.0, reason),
                        Decision("FLAT", 0.0, 0.0, reason))

        p_model = self._p_model(z, tau, ctx)

        # Model-vs-market disagreement gate (mirror of decide()).
        # When raw p_model disagrees with mid by > threshold pp, the
        # σ-based z-score is probably overconfident due to a low-vol
        # collapse. Trust the market and skip both sides.
        if self.max_model_market_disagreement < 1.0:
            mid_up_obs = (bid_up + ask_up) / 2.0
            disagreement = abs(p_model - mid_up_obs)
            if disagreement > self.max_model_market_disagreement:
                reason = (f"model-market disagreement (|p_model={p_model:.2f} - "
                          f"mid={mid_up_obs:.2f}|={disagreement:.2f} > "
                          f"{self.max_model_market_disagreement:.2f})")
                return (Decision("FLAT", 0.0, 0.0, reason),
                        Decision("FLAT", 0.0, 0.0, reason))

        # F4: oracle lead-lag bias. Apply BEFORE filtration so the
        # filtration model's z/p features see the bias-adjusted view.
        # No-op when oracle_lead_bias=0 or binance_mid is unavailable.
        lead_bias = self._oracle_lead_bias(snapshot.chainlink_price, binance_mid)
        if lead_bias != 0.0:
            p_model = max(0.01, min(0.99, p_model + lead_bias))
            ctx["_oracle_lead_bias"] = lead_bias

        # Filtration gate: XGBoost confidence check
        if not self._check_filtration(z, sigma_per_s, tau, snapshot, ctx):
            conf = ctx.get("_filtration_confidence", 0.0)
            reason = f"filtration ({conf:.3f} < {self.filtration_threshold})"
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))

        # Expose model state so OrderMixin can snapshot it at fill time
        ctx["_p_model_raw"] = p_model
        ctx["_p_display"] = norm_cdf(z)  # smooth GBM for dashboard (no binning)
        ctx["_sigma_per_s"] = sigma_per_s
        ctx["_dyn_threshold_up"] = dyn_threshold

        if self.reversion_discount > 0:
            p_model = p_model * (1 - self.reversion_discount) + 0.5 * self.reversion_discount

        # Market blend: pull p_model toward contract mid (anti-fade-the-market).
        # When the model says BUY at $0.10 but the market trades at $0.10, the
        # market is usually right.  Backtested on 1159 BTC 5m windows: blend=0.3
        # turns -10% recent ROI into +10%.
        if self.market_blend > 0:
            mid_up = (bid_up + ask_up) / 2.0
            p_model = (1.0 - self.market_blend) * p_model + self.market_blend * mid_up
            p_model = max(0.01, min(0.99, p_model))

        # Order book imbalance → continuous p_model shift
        # OBI > 0 means more bid depth (buying pressure) → nudge p_up higher
        bid_depth_up = sum(sz for _, sz in snapshot.bid_levels_up)
        ask_depth_up = sum(sz for _, sz in snapshot.ask_levels_up)
        bid_depth_down = sum(sz for _, sz in snapshot.bid_levels_down)
        ask_depth_down = sum(sz for _, sz in snapshot.ask_levels_down)
        total_depth_up = bid_depth_up + ask_depth_up
        total_depth_down = bid_depth_down + ask_depth_down
        obi_up = (bid_depth_up - ask_depth_up) / total_depth_up if total_depth_up > 0 else 0.0
        obi_down = (bid_depth_down - ask_depth_down) / total_depth_down if total_depth_down > 0 else 0.0
        ctx["_obi_up"] = obi_up
        ctx["_obi_down"] = obi_down

        if self.obi_weight > 0:
            # Positive OBI on UP side → buy pressure → p_up higher
            # Positive OBI on DOWN side → buy pressure → p_down higher (p_up lower)
            p_model += self.obi_weight * (obi_up - obi_down)
            p_model = max(0.01, min(0.99, p_model))
        ctx["_p_model_trade"] = p_model

        # Cost basis: VAMP (volume-weighted mid) or best bid
        if self.vamp_mode == "cost":
            cost_up = vamp_up if vamp_up is not None else bid_up
            cost_down = vamp_down if vamp_down is not None else bid_down
        else:
            cost_up = bid_up
            cost_down = bid_down

        # Expose cost basis so OrderMixin can include it in model snapshot
        ctx["_cost_up"] = cost_up
        ctx["_cost_down"] = cost_down

        # Store expected range in ctx
        move_1sig = sigma_per_s * math.sqrt(tau) * price
        ctx["_expected_range"] = {
            "btc_at_fill": price,
            "start_price": snapshot.window_start_price,
            "expected_low": snapshot.window_start_price - move_1sig,
            "expected_high": snapshot.window_start_price + move_1sig,
        }

        # DOWN bonus: check book balance for down_edge_bonus
        down_bonus_active = False
        down_share = 0.5
        if self.down_edge_bonus > 0:
            total_d = total_depth_up + total_depth_down
            down_d = total_depth_down
            down_share = down_d / total_d if total_d > 0 else 0.5
            if 0.3 <= down_share <= 0.7:
                down_bonus_active = True
        ctx["_down_bonus_active"] = down_bonus_active
        ctx["_down_share"] = down_share

        # ── A-S unified quoting ───────────────────────────────────────
        if self.as_mode:
            # Realized contract vol (model-free, from Yang-Zhang on contract mids)
            sigma_p = self._contract_sigma_p(ctx)
            sigma_p_sq = sigma_p ** 2
            # A-S uses total remaining variance = sigma² * tau (NOT tau/T)
            total_var = sigma_p_sq * tau
            ctx["_sigma_p"] = sigma_p

            # Reservation price: shift belief by inventory risk
            # Higher total_var → larger penalty (more uncertain contract)
            n_up = ctx.get("inventory_up", 0)
            n_down = ctx.get("inventory_down", 0)
            q_up = n_up - n_down  # net long UP exposure
            inv_pen = self.gamma_inv * total_var
            r_up = p_model - inv_pen * q_up
            r_down = (1.0 - p_model) + inv_pen * q_up
            ctx["_inventory_skew"] = inv_pen

            # Edge from reservation value
            edge_up = r_up - cost_up - self.spread_edge_penalty * spread_up
            edge_down = r_down - cost_down - self.spread_edge_penalty * spread_down
            ctx["_edge_up"] = edge_up
            ctx["_edge_down"] = edge_down

            # Optimal spread: additive adverse selection components
            window_trades = ctx.get("window_trade_count", 0)
            base_spread = max(self.gamma_spread * total_var / 2.0,
                              self.min_edge)
            opt_spread = (base_spread
                          + self.tox_spread * tox_excess
                          + self.vpin_spread * vpin_excess
                          + self.lag_spread * lag_excess
                          + self.edge_step * window_trades)
            opt_spread_down = (opt_spread * (1.0 - self.down_edge_bonus)
                               if down_bonus_active else opt_spread)

            dyn_threshold = opt_spread
            dyn_threshold_down = opt_spread_down
            ctx["_dyn_threshold_up"] = opt_spread
            ctx["_dyn_threshold_down"] = opt_spread_down

            # Evaluate each side
            up_dec = flat
            down_dec = flat

            if edge_up > opt_spread:
                up_dec = self._size_decision(
                    "BUY_UP", edge_up, cost_up, r_up,
                    snapshot, sigma_per_s, tau, z, z_raw, p_model,
                    opt_spread, spread_up, ctx=ctx,
                )
            if edge_down > opt_spread_down:
                down_dec = self._size_decision(
                    "BUY_DOWN", edge_down, cost_down, r_down,
                    snapshot, sigma_per_s, tau, z, z_raw, p_model,
                    opt_spread_down, spread_down, ctx=ctx,
                )

        # ── Legacy multiplicative quoting ─────────────────────────────
        else:
            edge_up = p_model - cost_up - self.spread_edge_penalty * spread_up
            edge_down = (1.0 - p_model) - cost_down - self.spread_edge_penalty * spread_down
            ctx["_edge_up"] = edge_up
            ctx["_edge_down"] = edge_down

            # Inventory skew (legacy)
            if self.inventory_skew > 0:
                n_up = ctx.get("inventory_up", 0)
                n_down = ctx.get("inventory_down", 0)
                skew = self.inventory_skew * (tau / self.window_duration)
                ctx["_inventory_skew"] = skew
                edge_up -= skew * n_up
                edge_up += skew * n_down
                edge_down -= skew * n_down
                edge_down += skew * n_up

            dyn_threshold_down = dyn_threshold
            if down_bonus_active:
                dyn_threshold_down = dyn_threshold * (1.0 - self.down_edge_bonus)
            ctx["_dyn_threshold_down"] = dyn_threshold_down

            # Edge persistence gate: defend against spike-chasing.
            # Track when each side first crossed the edge threshold; if it
            # hasn't been above threshold for at least edge_persistence_s,
            # block the trade for that side this tick. Reset on no-edge.
            up_persist_block = False
            down_persist_block = False
            if self.edge_persistence_s > 0:
                now_ms = snapshot.ts_ms
                persist_ms = self.edge_persistence_s * 1000.0
                up_has_edge = (edge_up > dyn_threshold)
                down_has_edge = (edge_down > dyn_threshold_down)
                if up_has_edge:
                    if "_edge_up_first_ms" not in ctx:
                        ctx["_edge_up_first_ms"] = now_ms
                    if now_ms - ctx["_edge_up_first_ms"] < persist_ms:
                        up_persist_block = True
                else:
                    ctx.pop("_edge_up_first_ms", None)
                if down_has_edge:
                    if "_edge_down_first_ms" not in ctx:
                        ctx["_edge_down_first_ms"] = now_ms
                    if now_ms - ctx["_edge_down_first_ms"] < persist_ms:
                        down_persist_block = True
                else:
                    ctx.pop("_edge_down_first_ms", None)

            up_dec = flat
            down_dec = flat

            if edge_up > dyn_threshold and not up_persist_block:
                up_dec = self._size_decision(
                    "BUY_UP", edge_up, cost_up, p_model,
                    snapshot, sigma_per_s, tau, z, z_raw, p_model,
                    dyn_threshold, spread_up, ctx=ctx,
                )
            elif up_persist_block:
                elapsed = (snapshot.ts_ms - ctx["_edge_up_first_ms"]) / 1000.0
                up_dec = Decision("FLAT", 0.0, 0.0,
                    f"edge persistence (UP {elapsed:.1f}s < {self.edge_persistence_s}s)")
            if edge_down > dyn_threshold_down and not down_persist_block:
                down_dec = self._size_decision(
                    "BUY_DOWN", edge_down, cost_down, 1.0 - p_model,
                    snapshot, sigma_per_s, tau, z, z_raw, p_model,
                    dyn_threshold_down, spread_down, ctx=ctx,
                )
            elif down_persist_block:
                elapsed = (snapshot.ts_ms - ctx["_edge_down_first_ms"]) / 1000.0
                down_dec = Decision("FLAT", 0.0, 0.0,
                    f"edge persistence (DOWN {elapsed:.1f}s < {self.edge_persistence_s}s)")

        # If neither has edge, report combined reason
        if up_dec.action == "FLAT" and down_dec.action == "FLAT":
            if edge_up <= dyn_threshold and edge_down <= dyn_threshold_down:
                reason = (f"no edge (up={edge_up:.4f} down={edge_down:.4f} "
                          f"thresh={dyn_threshold:.4f}"
                          f"{f' down_thresh={dyn_threshold_down:.4f}' if dyn_threshold_down != dyn_threshold else ''})")
                up_dec = Decision("FLAT", 0.0, 0.0, reason)
                down_dec = Decision("FLAT", 0.0, 0.0, reason)

        # Min entry price gate: override any non-FLAT decision if bid is too cheap
        if self.min_entry_price > 0:
            if up_dec.action != "FLAT" and cost_up < self.min_entry_price:
                up_dec = Decision("FLAT", 0.0, 0.0,
                    f"entry {cost_up:.3f} < min {self.min_entry_price:.2f}")
            if down_dec.action != "FLAT" and cost_down < self.min_entry_price:
                down_dec = Decision("FLAT", 0.0, 0.0,
                    f"entry {cost_down:.3f} < min {self.min_entry_price:.2f}")

        return (up_dec, down_dec)

    # ── Stale-quote sniper ─────────────────────────────────────────────
    def decide_stale_quote(
        self, snapshot: Snapshot, ctx: dict,
    ) -> tuple[Decision, Decision]:
        """Detect when the Polymarket CLOB is stale relative to Binance.

        Instead of predicting direction with the diffusion model, this
        computes Binance-derived fair value and checks whether the best
        ask is cheaper than fair value minus taker fees.  If the edge
        exceeds ``stale_threshold`` (default 3 cents), fire a BUY as
        taker.

        Key differences from ``decide_both_sides``:
          - Compares fair value to best ASK (taker), not best BID (maker)
          - Includes the 7.2% Polymarket taker fee in the edge calc
          - No market_blend — the whole point is trusting Binance
          - No min_entry_z gate — the z-score is implicit in fair value
          - No filtration, A-S, OBI, or edge persistence
          - DOES apply stale-feature gates and max_trades_per_window

        Returns (up_decision, down_decision) — at most one is non-FLAT.
        """
        flat = Decision("FLAT", 0.0, 0.0, "")

        # ── Price history (always, for vol warmup) ────────────────────
        effective_price = ctx.get("_binance_mid") or snapshot.chainlink_price
        hist = ctx.setdefault("price_history", [])
        ts_hist = ctx.setdefault("ts_history", [])
        if ctx.pop("_live_history_appended", False):
            pass  # signal_ticker already appended this tick
        else:
            hist.append(effective_price)
            ts_hist.append(snapshot.ts_ms)

        if len(hist) < 2:
            reason = f"need history ({len(hist)}s collected)"
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))

        # ── Book validation ───────────────────────────────────────────
        ask_up = snapshot.best_ask_up
        ask_down = snapshot.best_ask_down
        if ask_up is None or ask_down is None:
            reason = "missing book"
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))
        if ask_up <= 0 or ask_up >= 1 or ask_down <= 0 or ask_down >= 1:
            reason = "invalid asks"
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))

        # ── Stale-feature gates (parity with decide_both_sides) ───────
        if self.max_book_age_ms is not None:
            book_age = ctx.get("_book_age_ms")
            if book_age is not None and book_age > self.max_book_age_ms:
                reason = f"stale book ({book_age:.0f}ms > {self.max_book_age_ms}ms)"
                return (Decision("FLAT", 0.0, 0.0, reason),
                        Decision("FLAT", 0.0, 0.0, reason))
        stale_reason = self._check_stale_features(ctx)
        if stale_reason is not None:
            return (Decision("FLAT", 0.0, 0.0, stale_reason),
                    Decision("FLAT", 0.0, 0.0, stale_reason))

        self._record_book_state(snapshot, ctx)

        # ── Tau ───────────────────────────────────────────────────────
        tau = snapshot.time_remaining_s
        if tau <= 0:
            reason = "window expired"
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))

        # ── Volatility ───────────────────────────────────────────────
        raw_sigma = self._compute_vol(
            hist[-self.vol_lookback_s:], ts_hist[-self.vol_lookback_s:]
        )
        sigma_per_s = self._smoothed_sigma(raw_sigma, ctx)
        if sigma_per_s == 0.0:
            reason = "zero vol"
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))
        ctx["_sigma_per_s"] = sigma_per_s

        # ── Fair value from Binance ───────────────────────────────────
        # Use raw Binance mid — no Chainlink blend, no market blend.
        binance_mid = ctx.get("_binance_mid")
        if binance_mid is None or binance_mid <= 0:
            reason = "no binance_mid"
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))

        wsp = snapshot.window_start_price
        if wsp is None or wsp <= 0:
            reason = "no window_start_price"
            return (Decision("FLAT", 0.0, 0.0, reason),
                    Decision("FLAT", 0.0, 0.0, reason))

        delta = (binance_mid - wsp) / wsp
        z_raw = delta / (sigma_per_s * math.sqrt(tau))
        z = max(-self.max_z, min(self.max_z, z_raw))
        # Use the configured tail model (kou, student_t, normal, etc.)
        # instead of hardcoded norm_cdf. Kou jump-diffusion accounts for
        # fat tails in crypto returns which makes the fair value more
        # accurate at extreme z-scores.
        ctx["_sigma_per_s"] = sigma_per_s
        fair_up = self._model_cdf(z, ctx)
        fair_down = 1.0 - fair_up

        ctx["_z_raw"] = z_raw
        ctx["_z"] = z
        ctx["_tau"] = tau
        ctx["_p_model_raw"] = fair_up
        ctx["_p_display"] = fair_up
        ctx["_stale_fair_up"] = fair_up
        ctx["_stale_fair_down"] = fair_down
        ctx["_p_display_fresh"] = True

        # ── Edge calculation (taker: buy at ask, pay fee) ─────────────
        fee_up = poly_fee(ask_up)       # taker fee on UP contract
        fee_down = poly_fee(ask_down)   # taker fee on DOWN contract
        edge_up = fair_up - ask_up - fee_up
        edge_down = fair_down - ask_down - fee_down

        ctx["_edge_up"] = edge_up
        ctx["_edge_down"] = edge_down
        ctx["_dyn_threshold_up"] = self.stale_threshold
        ctx["_dyn_threshold_down"] = self.stale_threshold

        # ── Fire on the side with better edge (if above threshold) ────
        up_dec = flat
        down_dec = flat

        # Pick the side with the larger edge; only one side fires per tick
        if edge_up > self.stale_threshold and edge_up >= edge_down:
            up_dec = self._stale_size_decision(
                "BUY_UP", edge_up, ask_up, fair_up,
                sigma_per_s, tau, z, z_raw,
            )
        elif edge_down > self.stale_threshold:
            down_dec = self._stale_size_decision(
                "BUY_DOWN", edge_down, ask_down, fair_down,
                sigma_per_s, tau, z, z_raw,
            )

        # No-edge reason
        if up_dec.action == "FLAT" and down_dec.action == "FLAT":
            if edge_up <= self.stale_threshold and edge_down <= self.stale_threshold:
                reason = (f"no stale edge (up={edge_up:.4f} down={edge_down:.4f} "
                          f"thresh={self.stale_threshold:.4f})")
                up_dec = Decision("FLAT", 0.0, 0.0, reason)
                down_dec = Decision("FLAT", 0.0, 0.0, reason)

        return (up_dec, down_dec)

    def _stale_size_decision(
        self, side: str, edge: float, ask_price: float,
        p_side: float, sigma_per_s: float, tau: float,
        z: float, z_raw: float,
    ) -> Decision:
        """Kelly sizing for stale-quote taker fills.

        Simpler than ``_size_decision`` — no filtration multiplier,
        no inventory skew, no A-S spread. The cost basis is the ask
        price (since we're crossing the spread as taker) plus fees.
        """
        fee = poly_fee(ask_price)
        eff_cost = ask_price + fee
        if eff_cost <= 0 or eff_cost >= 1.0:
            return Decision("FLAT", 0.0, 0.0, "eff cost out of range")

        # Min entry price gate
        if self.min_entry_price > 0 and ask_price < self.min_entry_price:
            return Decision("FLAT", 0.0, 0.0,
                f"entry {ask_price:.3f} < min {self.min_entry_price:.2f}")

        # Half-Kelly on the edge
        kelly_f = max(0.0, (p_side - eff_cost) / (1.0 - eff_cost))
        frac = min(self.kelly_fraction * kelly_f, self.max_bet_fraction)
        if frac <= 0:
            return Decision("FLAT", 0.0, 0.0, "kelly <= 0")

        size_usd = self.bankroll * frac
        min_usd = self.min_order_shares * eff_cost
        if size_usd < min_usd:
            if self.bankroll >= min_usd:
                size_usd = min_usd
            else:
                return Decision("FLAT", 0.0, 0.0,
                    f"bankroll ${self.bankroll:.2f} < min order ${min_usd:.2f}")

        reason = (
            f"STALE_QUOTE p={p_side:.4f} sig={sigma_per_s:.2e} z={z:.2f}"
            f" tau={tau:.0f}s edge={edge:.4f} thresh={self.stale_threshold:.4f}"
            f" ask={ask_price:.4f} fee={fee:.4f} kelly={kelly_f:.4f}"
            f" ${size_usd:.0f}"
        )
        return Decision(side, edge, size_usd, reason)
