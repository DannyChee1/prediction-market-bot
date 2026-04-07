"""
Self-exciting Hawkes process for jump intensity (Quant Guild #94).

A Hawkes process models the conditional intensity λ(t) of an event
stream so that recent events boost λ for a while, then decay back to
the baseline. For crypto returns this captures the empirical clustering
of large moves: a 0.5%+ move makes the next 0.5%+ move more likely for
~60-120 seconds before the intensity decays.

Mathematical form (univariate, exponential kernel):

    λ(t) = μ + α · Σ exp(-β · (t - t_i))   for all event times t_i < t

Parameters:
    μ  — baseline rate (events per second)
    α  — branching ratio (excitation per event)
    β  — decay rate (1/β ≈ excitation half-life in seconds × ln(2))

Stationarity: α/β < 1.

This module exposes:

    HawkesIntensity      — online stateful tracker; called per tick
    fit_hawkes_mle       — offline parameter fit on a sequence of
                           inter-arrival times (constant baseline)
    detect_jumps         — given a price series + a "jump" threshold
                           (e.g. |return| > k·σ), return event timestamps

The intent is opt-in: DiffusionSignal can hold a HawkesIntensity instance
and surface λ(t) in ctx for the dashboard, the filtration features, or
a future regime-aware sizing rule. The default DiffusionSignal does not
construct one.

Usage:
    h = HawkesIntensity(mu=0.005, alpha=0.4, beta=0.025)
    for ts_ms, price in stream:
        if abs(log_return) > k * sigma:
            h.add_event(ts_ms / 1000)
        intensity = h.intensity_at(ts_ms / 1000)
        # use `intensity` as a feature
"""
from __future__ import annotations

import math
from collections import deque
from typing import Iterable


class HawkesIntensity:
    """Online univariate Hawkes intensity with exponential kernel.

    State is the running sum  S(t) = Σ exp(-β·(t - t_i))  for events
    that have happened. We update it incrementally so we don't iterate
    over all past events every tick:

        At time t, given the previous update time t_prev:
            S(t) = S(t_prev) · exp(-β · (t - t_prev))

        When a new event lands at t_new:
            S(t_new) = S(t_new) + 1

    The instantaneous intensity is then  λ(t) = μ + α · S(t).
    """

    __slots__ = ("mu", "alpha", "beta", "_S", "_t_last", "_n_events",
                 "_recent_event_times", "_max_recent")

    def __init__(self, mu: float, alpha: float, beta: float,
                 max_recent: int = 1024):
        if mu < 0 or alpha < 0 or beta <= 0:
            raise ValueError(f"need mu>=0, alpha>=0, beta>0; "
                             f"got mu={mu} alpha={alpha} beta={beta}")
        if alpha >= beta:
            # alpha >= beta means non-stationary (intensity grows). Allow
            # it but warn the caller via a marker on the instance.
            pass
        self.mu = float(mu)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._S = 0.0
        self._t_last = None  # last update wall time (seconds)
        self._n_events = 0
        # Keep a small ring buffer of recent event times for diagnostics.
        self._recent_event_times: deque[float] = deque(maxlen=max_recent)
        self._max_recent = max_recent

    def _decay_to(self, t: float) -> None:
        if self._t_last is None:
            self._t_last = t
            return
        dt = t - self._t_last
        if dt <= 0:
            return
        self._S *= math.exp(-self.beta * dt)
        self._t_last = t

    def add_event(self, t: float) -> None:
        """Register an event at time t (seconds, monotone)."""
        self._decay_to(t)
        self._S += 1.0
        self._n_events += 1
        self._recent_event_times.append(t)

    def intensity_at(self, t: float) -> float:
        """Return the current intensity λ(t)."""
        self._decay_to(t)
        return self.mu + self.alpha * self._S

    @property
    def n_events(self) -> int:
        return self._n_events

    @property
    def is_stationary(self) -> bool:
        return self.alpha < self.beta

    def __repr__(self) -> str:
        return (f"HawkesIntensity(mu={self.mu:.4f}, alpha={self.alpha:.4f}, "
                f"beta={self.beta:.4f}, n_events={self._n_events})")


# ── Offline fitting ─────────────────────────────────────────────────────────


def fit_hawkes_mle(event_times: Iterable[float],
                   T_total: float,
                   mu_grid: Iterable[float] | None = None,
                   alpha_grid: Iterable[float] | None = None,
                   beta_grid: Iterable[float] | None = None) -> dict:
    """Coarse-grid maximum likelihood for univariate exponential Hawkes.

    Closed-form ML for Hawkes is non-trivial; we use a coarse grid
    search over (mu, alpha, beta) and return the best by log-likelihood.

    Log-likelihood for a Hawkes process observed on [0, T] with events
    {t_i}:

        ℓ = Σ_i log λ(t_i) - ∫_0^T λ(s) ds

    For the exponential kernel, the integral is:

        ∫_0^T λ(s) ds = μ·T + (α/β) · Σ_i (1 - exp(-β·(T - t_i)))

    Parameters
    ----------
    event_times : iterable of float
        Sorted, monotone event times in seconds.
    T_total : float
        Length of the observation window.
    mu_grid, alpha_grid, beta_grid : iterables of float
        Optional override grids. Sensible defaults are used otherwise.

    Returns
    -------
    dict with keys mu, alpha, beta, log_likelihood, n_events
    """
    events = list(event_times)
    n = len(events)
    if n < 3 or T_total <= 0:
        return {"mu": 0.0, "alpha": 0.0, "beta": 1.0,
                "log_likelihood": float("-inf"), "n_events": int(n)}

    # Sensible default grids if caller didn't supply
    base_rate = n / T_total
    if mu_grid is None:
        mu_grid = [base_rate * f for f in (0.1, 0.25, 0.5, 0.75, 1.0)]
    if alpha_grid is None:
        alpha_grid = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    if beta_grid is None:
        beta_grid = [0.01, 0.025, 0.05, 0.1, 0.2, 0.5]

    def loglik(mu, alpha, beta) -> float:
        if mu <= 0 or alpha < 0 or beta <= 0:
            return float("-inf")
        # Compute λ at each event time using the running-sum recursion
        S = 0.0
        ll_sum = 0.0
        last_t = None
        for t in events:
            if last_t is not None:
                S = S * math.exp(-beta * (t - last_t))
            lam = mu + alpha * S
            if lam <= 0:
                return float("-inf")
            ll_sum += math.log(lam)
            S += 1.0  # event lands AFTER computing intensity
            last_t = t
        # Compensator: integral of intensity over [0, T_total]
        compensator = mu * T_total + (alpha / beta) * sum(
            1.0 - math.exp(-beta * (T_total - t)) for t in events
        )
        return ll_sum - compensator

    best = {"mu": 0.0, "alpha": 0.0, "beta": 1.0,
            "log_likelihood": float("-inf"), "n_events": int(n)}
    for mu in mu_grid:
        for alpha in alpha_grid:
            for beta in beta_grid:
                ll = loglik(mu, alpha, beta)
                if ll > best["log_likelihood"]:
                    best = {"mu": float(mu), "alpha": float(alpha),
                            "beta": float(beta), "log_likelihood": float(ll),
                            "n_events": int(n)}
    return best


def detect_jumps(prices: list[float], timestamps: list[int],
                 sigma_per_s: float, k_sigma: float = 2.5) -> list[float]:
    """Find timestamps (seconds) of price moves with |z|>k_sigma.

    z is the per-tick standardised return: log(p[i]/p[i-1]) / (σ·√dt).
    Returns event times in SECONDS so they can feed HawkesIntensity.
    """
    out: list[float] = []
    if len(prices) != len(timestamps) or len(prices) < 2 or sigma_per_s <= 0:
        return out
    for i in range(1, len(prices)):
        if prices[i] <= 0 or prices[i - 1] <= 0:
            continue
        dt = (timestamps[i] - timestamps[i - 1]) / 1000.0
        if dt <= 0:
            continue
        r = math.log(prices[i] / prices[i - 1])
        z = r / (sigma_per_s * math.sqrt(dt))
        if abs(z) > k_sigma:
            out.append(timestamps[i] / 1000.0)
    return out
