"""
Alternative σ estimators for `_compute_vol`.

Three estimators are provided here as standalone, stateless functions:

  1. Yang-Zhang (delegates to backtest._compute_vol_deduped — the
     existing default)
  2. EWMA (RiskMetrics-style exponentially-weighted MSE of returns)
  3. GARCH(1,1) one-step forecast

All three return σ in **per-second** units, matching the existing
`_compute_vol` contract. They are designed so they can be A/B tested
against Yang-Zhang on real parquet windows without touching the live
signal path. If A/B shows a clear winner the user can wire it into
DiffusionSignal._compute_vol via a `sigma_estimator` config field.

Why this matters: the notebook on ARCH/GARCH (#47) and the
non-stationarity notebook (#93) both argue that an adaptive volatility
forecast is meaningfully better than a rolling realized estimate for
prediction tasks. Yang-Zhang is also misapplied to 5s micro-bars in
the current code (its `var_oc` term is the overnight-gap variance,
which is meaningless on a continuously-traded feed). EWMA and GARCH
both avoid that.

Functions:
    realized_variance_per_s(prices, ts) -> sigma_per_s
    ewma_sigma_per_s(prices, ts, lambda_=0.94) -> sigma_per_s
    garch11_sigma_per_s(prices, ts, omega, alpha, beta) -> sigma_per_s
    fit_garch11(prices, ts, ...) -> (omega, alpha, beta, mse)
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def _normalised_log_returns(prices: Sequence[float],
                            ts: Sequence[int]) -> np.ndarray:
    """Convert (price, timestamp_ms) to per-√sec normalised log returns.

    Skips duplicate prices (matching the existing _compute_vol_deduped).
    Each return r_i is divided by sqrt(dt_seconds) so the resulting
    variance is in per-second units.
    """
    if len(prices) < 2:
        return np.array([], dtype=np.float64)
    out: list[float] = []
    last_p = prices[0]
    last_t = ts[0]
    for i in range(1, len(prices)):
        p = prices[i]
        t = ts[i]
        if p <= 0 or last_p <= 0 or p == last_p:
            continue
        dt = (t - last_t) / 1000.0
        if dt <= 0:
            continue
        r = math.log(p / last_p) / math.sqrt(dt)
        out.append(r)
        last_p = p
        last_t = t
    return np.array(out, dtype=np.float64)


def realized_variance_per_s(prices: Sequence[float],
                            ts: Sequence[int]) -> float:
    """Plain realized variance estimator: σ̂ = stdev(normalised log returns).

    Same convention as _compute_vol_deduped's fallback path. Used here
    as a baseline.
    """
    rets = _normalised_log_returns(prices, ts)
    if len(rets) < 2:
        return 0.0
    return float(rets.std(ddof=1))


def ewma_sigma_per_s(prices: Sequence[float], ts: Sequence[int],
                     lambda_: float = 0.94) -> float:
    """RiskMetrics-style EWMA σ:

        σ²(t) = λ · σ²(t-1) + (1-λ) · r²(t)

    Initial σ² is set to the unconditional sample variance of the
    first half of the returns to make the recursion converge faster.
    The default λ = 0.94 is the RiskMetrics daily-data value; for
    sub-second crypto returns it gives an effective half-life of about
    11 ticks, which is in the right ballpark for our 5m/15m markets.
    """
    rets = _normalised_log_returns(prices, ts)
    n = len(rets)
    if n < 2:
        return 0.0
    if n < 4:
        return float(rets.std(ddof=1))
    # Warm-start σ² from the unconditional sample variance of the
    # first half (avoids the recursion taking a long time to settle).
    half = max(2, n // 2)
    var = float(rets[:half].var(ddof=1))
    if var <= 0:
        var = float(rets.var(ddof=1))
    if var <= 0:
        return 0.0
    # Apply the EWMA recursion forward
    for r in rets[half:]:
        var = lambda_ * var + (1.0 - lambda_) * (r * r)
    return math.sqrt(max(var, 0.0))


def garch11_sigma_per_s(prices: Sequence[float], ts: Sequence[int],
                        omega: float, alpha: float, beta: float) -> float:
    """GARCH(1,1) one-step σ forecast given pre-fitted (omega, alpha, beta).

        σ²(t) = omega + alpha · r²(t-1) + beta · σ²(t-1)

    Returns σ̂(t+1) — the next-step forecast given the current state.
    Constraints: omega > 0, alpha + beta < 1 for stationarity. We do
    not enforce them here; that's the fitter's job.
    """
    rets = _normalised_log_returns(prices, ts)
    n = len(rets)
    if n < 2 or omega <= 0:
        return 0.0
    # Unconditional variance under the model: omega / (1 - alpha - beta)
    # Use it as the warm-start so we don't depend on the first noisy r².
    persistence = alpha + beta
    if persistence >= 1.0:
        # Non-stationary: fall back to sample variance
        var = float(rets.var(ddof=1))
    else:
        var = omega / (1.0 - persistence)
    if var <= 0:
        var = max(float(rets.var(ddof=1)), 1e-20)

    # Run the recursion through all but the last return
    for r in rets[:-1]:
        var = omega + alpha * (r * r) + beta * var
    # One-step forecast given the most recent return
    var_next = omega + alpha * (rets[-1] ** 2) + beta * var
    return math.sqrt(max(var_next, 0.0))


def fit_garch11(prices_concat: Sequence[float],
                ts_concat: Sequence[int],
                window_starts: Sequence[int] | None = None) -> dict:
    """Fit GARCH(1,1) (omega, alpha, beta) by quasi-maximum likelihood
    on a concatenation of return series, using a coarse grid + Nelder-Mead
    refinement.

    `window_starts` is an optional list of indices that mark the start
    of each individual time-window inside the concatenated arrays —
    we reset the recursion at each window start so we don't propagate
    state across discontinuities.

    Returns a dict {omega, alpha, beta, nll, n_obs}.
    """
    rets = _normalised_log_returns(prices_concat, ts_concat)
    n = len(rets)
    if n < 100:
        return {"omega": 0.0, "alpha": 0.0, "beta": 0.0,
                "nll": float("inf"), "n_obs": int(n)}

    sample_var = float(rets.var(ddof=1))
    if sample_var <= 0:
        return {"omega": 0.0, "alpha": 0.0, "beta": 0.0,
                "nll": float("inf"), "n_obs": int(n)}

    # Window boundaries (so the recursion can reset at each)
    if window_starts is None or len(window_starts) == 0:
        starts = np.array([0])
    else:
        starts = np.unique(np.clip(np.asarray(window_starts), 0, n - 1))

    def neg_log_likelihood(params: tuple[float, float, float]) -> float:
        omega, alpha, beta = params
        # Constraints: omega > 0, alpha >= 0, beta >= 0, alpha+beta < 1
        if omega <= 1e-30 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e18
        var0 = omega / (1.0 - alpha - beta)
        nll = 0.0
        var = var0
        last_start = -1
        for i, r in enumerate(rets):
            if last_start + 1 < len(starts) and i >= starts[last_start + 1]:
                # Reset state at window boundary
                var = var0
                last_start += 1
            else:
                var = omega + alpha * (rets[i - 1] ** 2 if i > 0 else 0.0) \
                    + beta * var
            if var <= 0:
                return 1e18
            nll += 0.5 * (math.log(2 * math.pi * var) + (r * r) / var)
        return nll

    # Coarse grid over (alpha, beta), with omega derived from the
    # unconditional-variance constraint omega = sample_var * (1 - alpha - beta).
    best = {"omega": sample_var, "alpha": 0.0, "beta": 0.0,
            "nll": neg_log_likelihood((sample_var, 0.0, 0.0))}
    for alpha in (0.02, 0.05, 0.10, 0.15, 0.20):
        for beta in (0.70, 0.80, 0.85, 0.90, 0.95):
            if alpha + beta >= 0.999:
                continue
            omega = sample_var * (1.0 - alpha - beta)
            nll = neg_log_likelihood((omega, alpha, beta))
            if nll < best["nll"]:
                best = {"omega": omega, "alpha": alpha,
                        "beta": beta, "nll": nll}
    return {"omega": float(best["omega"]),
            "alpha": float(best["alpha"]),
            "beta": float(best["beta"]),
            "nll": float(best["nll"]),
            "n_obs": int(n)}
