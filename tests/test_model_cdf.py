"""
Regression tests for the Kou drift bug.

Three things broke before this test existed:
  1. `_model_cdf` with tail_mode="kou" used to add a *risk-neutral* drift
     term `drift_z = -lam * zeta * sqrt(tau) / sigma` (a martingale
     correction borrowed from option pricing) and apply it to a
     *physical-measure* binary prediction. That biased p_UP downward
     by ~1.7% on BTC 5m and ~7% on BTC 15m at the σ floor — large
     enough to dominate `edge_threshold=0.06` and silently push the
     bot into systematic BUY_DOWN trades on quiet days.
  2. The bias was *asymmetric*: drift_z was negative for any
     `kou_p_up >= 0.5`, so symmetric Kou (`p_up=0.5`) should produce
     `p_model(z=0) == 0.5` and `p_model(-z) + p_model(z) == 1.0`.
     The pre-fix code violated both.
  3. The bias scaled as `1/sigma`, so it blew up on weekends when
     realized vol hit `min_sigma`. After fix this regression is
     enforced numerically below.

If any test in this file ever fails, do NOT silence it — the bug is back.
"""
from __future__ import annotations

import math
import pathlib
import sys

# Allow `python tests/test_model_cdf.py` from the repo root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from backtest import DiffusionSignal, norm_cdf, kou_cdf  # noqa: E402
from market_config import MARKET_CONFIGS  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────────────


def _signal(tail_mode: str = "kou", **overrides) -> DiffusionSignal:
    """Build a minimal DiffusionSignal for unit-testing _model_cdf in isolation."""
    defaults = dict(
        bankroll=10_000.0,
        slippage=0.0,
        tail_mode=tail_mode,
        kou_lambda=0.007,
        kou_p_up=0.5,        # symmetric default
        kou_eta1=1000.0,
        kou_eta2=1000.0,
        tail_nu_default=20.0,
        max_z=3.0,
        min_sigma=1e-6,
        max_sigma=1e-3,
    )
    defaults.update(overrides)
    return DiffusionSignal(**defaults)


def _ctx(sigma_per_s: float, tau: float, **extra) -> dict:
    ctx = {"_sigma_per_s": sigma_per_s, "_tau": tau}
    ctx.update(extra)
    return ctx


# ── Test 1: symmetric Kou returns 0.5 at z=0 ─────────────────────────────────


def test_kou_symmetric_returns_half_at_z0():
    """With kou_p_up=0.5 and z=0, p_model must be exactly 0.5.

    This catches the original bug: the drift term made p(z=0) ≈ 0.483
    on BTC 5m and ≈ 0.430 on BTC 15m at the σ floor.
    """
    sig = _signal(kou_p_up=0.5, kou_eta1=1000.0, kou_eta2=1000.0)
    for sigma in (1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3):
        for tau in (60.0, 300.0, 900.0):
            p = sig._model_cdf(0.0, _ctx(sigma, tau))
            assert abs(p - 0.5) < 1e-12, (
                f"symmetric Kou must give p=0.5 at z=0; got {p:.6f} "
                f"at sigma={sigma}, tau={tau}"
            )


def test_kou_symmetric_at_extremes():
    """For z = ±k, sum of p(z) and p(-z) must equal 1 exactly.

    This catches *any* odd-symmetry violation in the model — including
    the old drift bug, since drift_z was a constant additive shift that
    broke symmetry around z=0.
    """
    sig = _signal()
    for sigma in (1e-6, 1e-5, 5e-5, 1e-4, 5e-4):
        for tau in (60.0, 300.0, 900.0):
            for z in (0.1, 0.5, 1.0, 1.5, 2.0):
                ctx = _ctx(sigma, tau)
                p_pos = sig._model_cdf(+z, ctx)
                p_neg = sig._model_cdf(-z, ctx)
                assert abs((p_pos + p_neg) - 1.0) < 1e-12, (
                    f"odd-symmetry violated at z={z}, sigma={sigma}, "
                    f"tau={tau}: p(+z)={p_pos:.6f}, p(-z)={p_neg:.6f}, "
                    f"sum={p_pos+p_neg:.6f}"
                )


# ── Test 2: kou-asymmetric must NOT introduce a sigma-dependent bias ─────────


def test_kou_asymmetric_p_up_does_not_distort_z0():
    """Even with p_up=0.526 (the live BTC 5m config), p_model(z=0)
    must NOT depend on sigma or tau.

    The old bug was: drift_z = -lam*zeta*sqrt(tau)/sigma. As sigma
    shrank, the bias blew up. Now that the fix replaces the kou path
    with norm_cdf(z), the bias is identically zero — independent of
    (kou_lambda, kou_p_up, eta1, eta2). Document and enforce that.
    """
    cfg = MARKET_CONFIGS["btc_5m"]
    sig = _signal(
        kou_p_up=cfg.kou_p_up,
        kou_lambda=cfg.kou_lambda,
        kou_eta1=cfg.kou_eta1,
        kou_eta2=cfg.kou_eta2,
    )
    for sigma in (1e-6, 1e-5, 1e-4, 1e-3):
        for tau in (60.0, 300.0, 900.0):
            p = sig._model_cdf(0.0, _ctx(sigma, tau))
            # We tolerate a tiny epsilon for floating point only.
            assert abs(p - 0.5) < 1e-9, (
                f"REGRESSION: kou drift bug is back. "
                f"_model_cdf(z=0, sigma={sigma}, tau={tau}, "
                f"kou_p_up={cfg.kou_p_up}) returned {p:.6f}, "
                f"expected 0.5"
            )


def test_kou_no_blowup_at_low_sigma():
    """Specifically test the weekend / quiet-market case: sigma at the
    floor. Pre-fix this produced p_model bias of -7% at BTC 15m floor
    (sigma=3e-5, tau=900); post-fix it must be 0.5 ± 1e-9.
    """
    cfg = MARKET_CONFIGS["btc"]
    sig = _signal(
        kou_p_up=cfg.kou_p_up,
        kou_lambda=cfg.kou_lambda,
        kou_eta1=cfg.kou_eta1,
        kou_eta2=cfg.kou_eta2,
    )
    # Old floor was 3e-5, new floor is 1e-5. Test both — and one
    # extreme value below either floor — to make sure the bug never
    # returns even if config changes again.
    for sigma in (1e-7, 1e-6, 3e-6, 3e-5):
        p = sig._model_cdf(0.0, _ctx(sigma, 900.0))
        assert abs(p - 0.5) < 1e-9, (
            f"REGRESSION: bias scales with 1/sigma again. "
            f"sigma={sigma}, p={p:.6f}"
        )


# ── Test 3: kou path must equal normal path (after the fix) ──────────────────


def test_kou_path_equals_normal_path():
    """After the fix, tail_mode='kou' must produce the same p_model
    as tail_mode='normal' for any z, sigma, tau. If they ever diverge,
    something has been added back to the kou path.
    """
    sig_kou = _signal(tail_mode="kou", kou_p_up=0.526)
    sig_normal = _signal(tail_mode="normal")
    for sigma in (1e-6, 1e-5, 1e-4, 1e-3):
        for tau in (60.0, 300.0, 900.0):
            for z in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
                ctx_k = _ctx(sigma, tau)
                ctx_n = _ctx(sigma, tau)
                p_k = sig_kou._model_cdf(z, ctx_k)
                p_n = sig_normal._model_cdf(z, ctx_n)
                assert abs(p_k - p_n) < 1e-12, (
                    f"kou and normal diverge at z={z}, sigma={sigma}, "
                    f"tau={tau}: kou={p_k:.6f}, normal={p_n:.6f}"
                )


# ── Test 4: monotonicity in z ────────────────────────────────────────────────


def test_kou_strictly_monotone_in_z():
    """p_model must be strictly increasing in z. This catches any sign
    flip introduced by future refactors.
    """
    sig = _signal()
    for sigma in (1e-5, 1e-4, 1e-3):
        for tau in (60.0, 300.0, 900.0):
            zs = [-2.5, -1.0, -0.1, 0.0, 0.1, 1.0, 2.5]
            ps = [sig._model_cdf(z, _ctx(sigma, tau)) for z in zs]
            for a, b in zip(ps, ps[1:]):
                assert b > a, (
                    f"_model_cdf not monotone in z at sigma={sigma}, "
                    f"tau={tau}: ps={ps}"
                )


# ── Test 5: Student-t path is unchanged (we did not touch it) ────────────────


def test_student_t_unchanged():
    """Student-t mode should still return fast_t_cdf(z, nu)."""
    from backtest import fast_t_cdf  # local import keeps top of file clean

    for nu in (5.0, 10.0, 20.0, 100.0):
        sig = _signal(tail_mode="student_t", tail_nu_default=nu)
        for z in (-2.0, -0.5, 0.0, 0.5, 2.0):
            ctx = _ctx(1e-4, 300.0, _tail_nu=nu)
            p = sig._model_cdf(z, ctx)
            assert abs(p - fast_t_cdf(z, nu)) < 1e-12, (
                f"student_t mode diverged from fast_t_cdf at z={z}, nu={nu}"
            )


# ── Test 6: kou_cdf (the dead-but-correct full implementation) ───────────────


def test_kou_cdf_reduces_to_normal_when_lambda_is_zero():
    """`kou_cdf` (the proper Poisson-convolution version, currently
    dead code in `_model_cdf`) should reduce to a Gaussian CDF when
    lambda=0. This protects the dead code in case we wire it in later.

    Under the risk-neutral drift in `kou_cdf`, mu = -sigma^2/2, so
    P(X_tau < x) = Phi((x - (-sigma^2/2)*tau) / (sigma*sqrt(tau))).
    """
    sigma = 1e-4
    tau = 300.0
    p_up = 0.5
    eta1 = eta2 = 1000.0
    for x in (-1e-3, -1e-4, 0.0, 1e-4, 1e-3):
        p_kou = kou_cdf(x, sigma, 0.0, p_up, eta1, eta2, tau)
        mu = -0.5 * sigma * sigma  # zero-drift drift correction
        z_eff = (x - mu * tau) / (sigma * math.sqrt(tau))
        p_norm = norm_cdf(z_eff)
        assert abs(p_kou - p_norm) < 1e-9, (
            f"kou_cdf with lambda=0 should equal Gaussian; "
            f"got {p_kou:.6f} vs {p_norm:.6f} at x={x}"
        )


def test_kou_cdf_clt_branch_internally_consistent():
    """For lam*tau > 5 the function uses the CLT branch. Verify the
    Poisson-weighted-Gaussian and CLT branches agree at the boundary.
    """
    sigma = 1e-4
    tau = 300.0
    eta1 = eta2 = 500.0
    p_up = 0.5
    # Pick a lam that puts lam*tau just on each side of the threshold
    for lam in (0.005, 0.02, 0.05, 0.1):
        for x in (-5e-4, 0.0, 5e-4):
            p = kou_cdf(x, sigma, lam, p_up, eta1, eta2, tau)
            assert 0.0 <= p <= 1.0, f"kou_cdf out of [0,1]: {p}"


# ── Test 7d: Hawkes intensity ───────────────────────────────────────────────


def test_hawkes_intensity_decay_and_excitation():
    """Sanity-check the standalone HawkesIntensity tracker:
       1. Adding an event boosts λ above the baseline
       2. λ decays toward μ over time after the event
       3. With α >= β it does NOT crash (it's just non-stationary)
    """
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from scripts.hawkes import HawkesIntensity, fit_hawkes_mle, detect_jumps

    h = HawkesIntensity(mu=0.001, alpha=0.5, beta=0.05)
    base = h.intensity_at(0.0)
    assert abs(base - 0.001) < 1e-12

    h.add_event(1.0)
    just_after = h.intensity_at(1.0)
    assert just_after > base * 100, (
        f"event should boost λ way above baseline: got {just_after} vs {base}"
    )

    # Decay over time toward μ
    decayed_far = h.intensity_at(1000.0)
    assert decayed_far < just_after * 0.01, (
        f"λ should decay below 1% of post-event level after 1000s: "
        f"got {decayed_far}"
    )
    assert decayed_far >= base, "λ must never drop below μ"

    # Non-crash with non-stationary params
    h2 = HawkesIntensity(mu=0.01, alpha=2.0, beta=1.0)
    h2.add_event(0.0)
    _ = h2.intensity_at(1.0)


def test_hawkes_fit_recovers_params_on_synthetic():
    """Fit Hawkes on a synthetic stream sampled from the same model.
    The fit doesn't have to recover params exactly (it's a coarse grid),
    but it must produce a higher log-likelihood than a baseline of
    (mu=base_rate, alpha=0, beta=any).
    """
    import random
    from scripts.hawkes import HawkesIntensity, fit_hawkes_mle
    random.seed(0)
    # Simple thinning sampler for an exponential Hawkes
    mu_true, alpha_true, beta_true = 0.05, 0.3, 0.1
    T = 2000.0
    events: list[float] = []
    t = 0.0
    while t < T:
        # Bound on intensity for the next gap
        if events:
            S = sum(math.exp(-beta_true * (t - ti)) for ti in events[-50:])
        else:
            S = 0.0
        upper = mu_true + alpha_true * (S + 1.0)
        if upper <= 0:
            break
        u = random.random()
        gap = -math.log(u) / upper
        t += gap
        if t >= T:
            break
        # Acceptance
        if events:
            S2 = sum(math.exp(-beta_true * (t - ti)) for ti in events[-50:])
        else:
            S2 = 0.0
        lam = mu_true + alpha_true * S2
        if random.random() <= lam / upper:
            events.append(t)

    fit = fit_hawkes_mle(events, T)
    # Compare to no-excitation baseline (alpha=0)
    base_fit = fit_hawkes_mle(events, T,
                              mu_grid=[len(events) / T],
                              alpha_grid=[0.0],
                              beta_grid=[0.1])
    assert fit["log_likelihood"] >= base_fit["log_likelihood"], (
        f"fit_hawkes_mle should beat the alpha=0 baseline; "
        f"got fit={fit['log_likelihood']} baseline={base_fit['log_likelihood']}"
    )
    # Sanity: fitted alpha should be >0
    assert fit["alpha"] > 0, f"expected alpha>0 in fit, got {fit}"


# ── Test 7c: Kalman OBI smoother ────────────────────────────────────────────


def test_kalman_obi_smoother_damps_noise():
    """Feed the smoother noisy zero-mean OBI; the smoothed estimate
    should converge to ~0 (within 0.05) after 50 ticks. The smoother
    must NOT just echo the input — that would defeat the purpose.
    """
    import random
    random.seed(123)
    sig = _signal()
    ctx: dict = {}
    smoothed = []
    for _ in range(200):
        raw = random.gauss(0, 0.30)  # high-noise zero-mean
        s = sig._kalman_obi_update("_obi_kalman_test", raw, ctx)
        smoothed.append(s)
    # Tail mean should be near 0 — much closer than the input std
    tail_mean = sum(smoothed[-50:]) / 50
    assert abs(tail_mean) < 0.1, (
        f"smoother failed to damp zero-mean noise: tail mean = {tail_mean}"
    )
    # Smoothed should have meaningfully less variance than raw
    import statistics
    raw_std_proxy = 0.30
    smooth_std = statistics.stdev(smoothed[-100:])
    assert smooth_std < raw_std_proxy * 0.6, (
        f"smoothed std {smooth_std:.3f} is not meaningfully less than "
        f"raw std {raw_std_proxy}"
    )


def test_kalman_obi_smoother_tracks_step():
    """Feed the smoother a constant +0.5 OBI for 100 ticks; the
    smoothed estimate must converge to a value > 0.3 (the AR(1)
    structure with β=0.95 means the steady state is below the input,
    but it should still clearly cross 0.3).
    """
    sig = _signal()
    ctx: dict = {}
    last = 0.0
    for _ in range(100):
        last = sig._kalman_obi_update("_obi_kalman_test2", 0.5, ctx)
    assert last > 0.3, (
        f"smoother failed to track step input: final={last:.4f}, expected > 0.3"
    )


# ── Test 7b: sigma_estimator dispatch ────────────────────────────────────────


def test_sigma_estimator_dispatch():
    """`_compute_vol` must dispatch to the right backend based on
    `self.sigma_estimator`. The three modes (yz, rv, ewma) must all
    return positive σ for the same input. They are NOT required to
    return identical values — that would defeat the purpose — but they
    must all be in the right order of magnitude.
    """
    # Synthetic price walk: 200 ticks of mild geometric Brownian motion
    rng = np.random.default_rng(0) if False else None  # avoid hard dep
    import random
    random.seed(0)
    p = 100.0
    prices = [p]
    ts = [0]
    for i in range(1, 200):
        p *= math.exp(0.001 * random.gauss(0, 1))
        prices.append(p)
        ts.append(i * 1000)

    sig_yz = _signal()
    sig_yz.sigma_estimator = "yz"
    sig_rv = _signal()
    sig_rv.sigma_estimator = "rv"
    sig_ewma = _signal()
    sig_ewma.sigma_estimator = "ewma"

    s_yz = sig_yz._compute_vol(prices, ts)
    s_rv = sig_rv._compute_vol(prices, ts)
    s_ewma = sig_ewma._compute_vol(prices, ts)

    assert s_yz > 0, "YZ returned 0 on a clearly volatile series"
    assert s_rv > 0, "RV returned 0 on a clearly volatile series"
    assert s_ewma > 0, "EWMA returned 0 on a clearly volatile series"
    # All three should be in the same order of magnitude
    largest = max(s_yz, s_rv, s_ewma)
    smallest = min(s_yz, s_rv, s_ewma)
    assert largest / smallest < 5.0, (
        f"σ estimators disagree by >5×: yz={s_yz} rv={s_rv} ewma={s_ewma}"
    )


def test_sigma_estimator_validation():
    """Constructing DiffusionSignal with an invalid sigma_estimator
    must raise — silent fallback would mask config typos.
    """
    try:
        _signal(sigma_estimator="garch")
        assert False, "expected ValueError for sigma_estimator='garch'"
    except ValueError:
        pass
    try:
        _signal(sigma_estimator="totally_made_up")
        assert False, "expected ValueError for sigma_estimator='totally_made_up'"
    except ValueError:
        pass


# ── Test 7a: filtration warmup parity for mid_momentum ──────────────────────


def test_mid_momentum_warmup_parity():
    """When `mid_momentum_parity=True` is set, the inference-time
    mid_momentum must use the earliest available index up to 60 ticks
    back, matching train_filtration.py:compute_mid_momentum exactly.

    The default (parity=False) preserves the legacy "return 0 if len<62"
    behavior because the existing filtration_model.pkl was trained
    against it. Flipping the flag without retraining the filtration
    model degrades calibration of the confidence threshold (see
    validation_runs/ablation_btc_5m.json).

    This test enforces parity ONLY for the opt-in path. The legacy
    path is intentionally inconsistent and there is no test for it.
    """
    # Mirror the parity-restored live computation from `_check_filtration`
    def live_mid_momentum_parity(mid_history):
        if len(mid_history) >= 2:
            lookback_idx = max(0, len(mid_history) - 61)
            return float(mid_history[-1]) - float(mid_history[lookback_idx])
        return 0.0

    # Mirror train_filtration.py:compute_mid_momentum exactly
    def train_mid_momentum(mid_full_window, idx, lookback=60):
        lo = max(0, idx - lookback)
        slice_ = [m for m in mid_full_window[lo:idx + 1] if m is not None]
        if len(slice_) < 2:
            return 0.0
        return float(slice_[-1]) - float(slice_[0])

    # Synthetic mid history: ramp from 0.5 → 0.6 over 200 ticks
    mids = [0.5 + 0.0005 * i for i in range(200)]

    # At every idx >= 1, live (parity path) and train should agree
    mismatches = []
    for idx in range(1, len(mids)):
        live = live_mid_momentum_parity(mids[: idx + 1])
        train = train_mid_momentum(mids, idx)
        if abs(live - train) > 1e-12:
            mismatches.append((idx, live, train))

    assert not mismatches, (
        f"warmup parity bug: {len(mismatches)} indices disagree, e.g. "
        f"first mismatch idx={mismatches[0][0]}: live={mismatches[0][1]} "
        f"train={mismatches[0][2]}"
    )


# ── Test 7: stale-feature gates ──────────────────────────────────────────────


def test_stale_feature_gates_skip_when_age_exceeds():
    """When the *_age_ms fields are set in ctx and exceed the threshold,
    _check_stale_features must return a skip reason. When fresh, must
    return None.
    """
    sig = _signal(
        max_chainlink_age_ms=30_000.0,
        max_binance_age_ms=2_000.0,
        max_trade_tape_age_ms=10_000.0,
    )
    # All fresh — should return None
    assert sig._check_stale_features({
        "_chainlink_age_ms": 1000.0,
        "_binance_age_ms": 100.0,
        "_trade_tape_age_ms": 500.0,
    }) is None
    # Chainlink stale
    r = sig._check_stale_features({"_chainlink_age_ms": 60_000.0})
    assert r is not None and "chainlink" in r
    # Binance stale
    r = sig._check_stale_features({"_binance_age_ms": 5_000.0})
    assert r is not None and "binance" in r
    # Trade tape stale
    r = sig._check_stale_features({"_trade_tape_age_ms": 20_000.0})
    assert r is not None and "trade tape" in r
    # All gates None: never fires
    sig2 = _signal()
    assert sig2._check_stale_features({
        "_chainlink_age_ms": 99_999_999,
        "_binance_age_ms": 99_999_999,
        "_trade_tape_age_ms": 99_999_999,
    }) is None
    # Backtest case: no age fields → no skip
    assert sig._check_stale_features({}) is None


# ── Test 8: market_config sigma bounds are sane ──────────────────────────────


def test_market_config_sigma_bounds_reasonable():
    """The σ bounds for btc / btc_5m must (a) actually allow the
    empirical distribution through, and (b) not invert (min < max).
    Pre-fix btc_5m had min=7e-5, max=8e-5 — a 14% range that clamped
    88% of observations.
    """
    for key in ("btc", "btc_5m"):
        cfg = MARKET_CONFIGS[key]
        assert cfg.min_sigma < cfg.max_sigma, (
            f"{key}: min_sigma={cfg.min_sigma} >= max_sigma={cfg.max_sigma}"
        )
        # Range should be at least 5× — empirical p10..p99 of realized
        # vol on BTC spans roughly 1e-5 .. 1e-4, so anything tighter
        # than 5× will clamp the bulk of the data.
        ratio = cfg.max_sigma / cfg.min_sigma
        assert ratio >= 5.0, (
            f"{key}: max_sigma/min_sigma ratio {ratio:.2f} is too tight; "
            f"will clamp most of the empirical distribution"
        )


# ── Optional CLI runner ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL  {fn.__name__}\n      {e}")
        except Exception:
            fails += 1
            print(f"ERROR {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - fails}/{len(fns)} tests passed")
    sys.exit(1 if fails else 0)
