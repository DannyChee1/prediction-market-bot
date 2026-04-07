#!/usr/bin/env python3
"""
3a — Permutation test for sequence structure (Quant Guild #46 — Luck or Skill).

Sharpe = mean/std is INVARIANT under permutation of trade order, so a
naive "shuffle and recompute Sharpe" test is meaningless. The path-
dependent quantities that actually probe sequence structure are:

  1. Maximum drawdown — path-dependent. A strategy whose losses *cluster*
     in time will have a much worse MDD than the same multiset of trades
     in random order. If actual MDD >> median(shuffled MDD), the strategy
     is exposed to *regime concentration risk* even though Sharpe looks fine.

  2. Lag-1 autocorrelation of PnL — direct measure of "winning streaks".
     If lag-1 autocorr is significantly positive, wins follow wins (you
     ride regime persistence). If significantly negative, wins follow
     losses (mean-reverting). If ≈ 0, trades are i.i.d.

  3. Runs test (signs of PnL) — non-parametric test for clustering of
     wins vs losses. Z > 2 → too few runs (clustering). Z < -2 → too many
     runs (alternation). |Z| < 2 → no detectable structure.

For each of (1) and (2) we report a permutation p-value: how often the
shuffled trade order produced a value at least as extreme as the actual.

Reads:     validation_runs/<tag>/trades_test.parquet  (and bankroll_test.json)
Writes:    validation_runs/<tag>/permutation_sequence.json
Prints:    a summary table

Usage:
    python scripts/validate_permutation_sharpe.py validation_runs/postfix_btc_5m
    python scripts/validate_permutation_sharpe.py --n-perm 5000 validation_runs/postfix_btc
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


# ── Path-dependent metrics ───────────────────────────────────────────────────


def max_drawdown(pnls: np.ndarray, initial_bankroll: float = 10_000.0) -> float:
    """Return max drawdown as a positive fraction of peak bankroll."""
    bk = initial_bankroll + np.cumsum(pnls)
    bk = np.concatenate([[initial_bankroll], bk])
    peak = np.maximum.accumulate(bk)
    dd = (peak - bk) / peak
    return float(dd.max())


def lag1_autocorr(x: np.ndarray) -> float:
    """Lag-1 sample autocorrelation. Returns 0 if undefined."""
    if len(x) < 3:
        return 0.0
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom == 0:
        return 0.0
    num = float((x[:-1] * x[1:]).sum())
    return num / denom


def runs_test_z(signs: np.ndarray) -> float:
    """Wald-Wolfowitz runs test on a binary sequence (+1/-1).

    Returns the standardised Z. |Z| > 2 → reject the null of randomness.
    Z > 0 means *too few* runs (clustering); Z < 0 means too many
    (alternation).
    """
    s = np.where(signs > 0, 1, 0)
    n = len(s)
    if n < 2:
        return 0.0
    n1 = int(s.sum())
    n2 = n - n1
    if n1 == 0 or n2 == 0:
        return 0.0
    runs = 1 + int((s[1:] != s[:-1]).sum())
    expected_runs = (2 * n1 * n2) / n + 1
    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n * n * (n - 1))
    if var_runs <= 0:
        return 0.0
    # Note: positive Z = fewer runs than expected = clustering
    return (expected_runs - runs) / math.sqrt(var_runs)


# ── Permutation test driver ─────────────────────────────────────────────────


def permutation_test(
    pnls: np.ndarray,
    n_perm: int,
    initial_bankroll: float,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)

    actual_mdd = max_drawdown(pnls, initial_bankroll)
    actual_ac1 = lag1_autocorr(pnls)
    actual_runs_z = runs_test_z(pnls)

    perm_mdds = np.empty(n_perm, dtype=np.float64)
    perm_ac1s = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        perm = rng.permutation(pnls)
        perm_mdds[i] = max_drawdown(perm, initial_bankroll)
        perm_ac1s[i] = lag1_autocorr(perm)

    # MDD: actual is "bad" if it is LARGER than the shuffles. Two-tailed
    # check: how often does shuffling produce a *worse* (larger) MDD?
    n_geq_mdd = int((perm_mdds >= actual_mdd).sum())
    p_mdd = (n_geq_mdd + 1) / (n_perm + 1)

    # Lag-1 autocorr: shuffling should give zero-mean i.i.d. distribution.
    # Actual significantly different (either tail) means trades are not i.i.d.
    n_extreme_ac1 = int((np.abs(perm_ac1s) >= abs(actual_ac1)).sum())
    p_ac1 = (n_extreme_ac1 + 1) / (n_perm + 1)

    return {
        "n_trades": int(len(pnls)),
        "n_perm": int(n_perm),
        "initial_bankroll": float(initial_bankroll),

        "max_drawdown_actual": float(actual_mdd),
        "max_drawdown_perm_mean": float(perm_mdds.mean()),
        "max_drawdown_perm_p05": float(np.percentile(perm_mdds, 5)),
        "max_drawdown_perm_p50": float(np.percentile(perm_mdds, 50)),
        "max_drawdown_perm_p95": float(np.percentile(perm_mdds, 95)),
        "max_drawdown_p_value": float(p_mdd),

        "lag1_autocorr_actual": float(actual_ac1),
        "lag1_autocorr_perm_p05": float(np.percentile(perm_ac1s, 5)),
        "lag1_autocorr_perm_p95": float(np.percentile(perm_ac1s, 95)),
        "lag1_autocorr_p_value_two_sided": float(p_ac1),

        "runs_test_z": float(actual_runs_z),
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Permutation tests for trade-sequence structure")
    ap.add_argument("tag_dir", type=Path,
                    help="Path to a validation_runs/<tag>/ directory")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tag_dir = args.tag_dir.resolve()
    trades_path = tag_dir / "trades_test.parquet"
    if not trades_path.exists():
        print(f"ERROR: {trades_path} not found", file=sys.stderr)
        sys.exit(1)

    trades = pd.read_parquet(trades_path)
    if trades.empty or "pnl" not in trades.columns:
        print(f"ERROR: empty or missing pnl column in {trades_path}",
              file=sys.stderr)
        sys.exit(1)

    pnls = trades["pnl"].to_numpy(dtype=np.float64)

    metrics_path = tag_dir / "metrics.json"
    initial_bk = 10_000.0
    if metrics_path.exists():
        with open(metrics_path) as f:
            initial_bk = float(json.load(f)["args"]["bankroll"])

    print(f"Loaded {len(pnls)} trades from {trades_path.name}")
    print(f"  PnL: mean={pnls.mean():+.2f}  std={pnls.std(ddof=1):.2f}  "
          f"sum={pnls.sum():+.2f}")
    print(f"  Initial bankroll: ${initial_bk:,.0f}")
    print(f"  Running {args.n_perm} permutations...")

    result = permutation_test(pnls, args.n_perm, initial_bk, args.seed)

    print()
    print("=" * 64)
    print("  PERMUTATION TEST FOR SEQUENCE STRUCTURE")
    print("=" * 64)

    # Max drawdown
    print()
    print("  --- Max Drawdown (path-dependent) ---")
    print(f"  Actual MDD:                {result['max_drawdown_actual']:.4f}")
    print(f"  Shuffled MDD median:       {result['max_drawdown_perm_p50']:.4f}")
    print(f"  Shuffled MDD [5%, 95%]:    "
          f"[{result['max_drawdown_perm_p05']:.4f}, "
          f"{result['max_drawdown_perm_p95']:.4f}]")
    print(f"  P(shuffled MDD ≥ actual):  {result['max_drawdown_p_value']:.4f}")
    if result["max_drawdown_p_value"] < 0.05:
        print("  → Actual MDD is WORSE than 95% of shuffles. Losses CLUSTER "
              "in time —")
        print("    your real drawdown is regime/concentration risk that the "
              "Sharpe doesn't see.")
    elif result["max_drawdown_p_value"] > 0.95:
        print("  → Actual MDD is BETTER than 95% of shuffles. Losses are "
              "spread out —")
        print("    the strategy avoids loss-clustering. Real edge if win "
              "rate is also positive.")
    else:
        print("  → Actual MDD is consistent with i.i.d. trade ordering.")

    # Lag-1 autocorrelation
    print()
    print("  --- Lag-1 Autocorrelation of PnL ---")
    print(f"  Actual ρ₁:                 {result['lag1_autocorr_actual']:+.4f}")
    print(f"  Shuffled ρ₁ [5%, 95%]:     "
          f"[{result['lag1_autocorr_perm_p05']:+.4f}, "
          f"{result['lag1_autocorr_perm_p95']:+.4f}]")
    print(f"  P(|shuffled ρ₁| ≥ |actual|): "
          f"{result['lag1_autocorr_p_value_two_sided']:.4f}")
    if result["lag1_autocorr_p_value_two_sided"] < 0.05:
        sign = "POSITIVE" if result['lag1_autocorr_actual'] > 0 else "NEGATIVE"
        print(f"  → Significant {sign} autocorrelation. Trades are NOT i.i.d.")
    else:
        print("  → No significant autocorrelation. Trades look i.i.d.")

    # Runs test
    print()
    print("  --- Wald-Wolfowitz Runs Test on PnL signs ---")
    print(f"  Z = {result['runs_test_z']:+.3f}  "
          f"(positive = clustering of wins/losses)")
    if abs(result["runs_test_z"]) > 2:
        word = "fewer" if result["runs_test_z"] > 0 else "more"
        print(f"  → |Z| > 2: significantly {word} runs than i.i.d. would "
              "produce.")
    else:
        print("  → |Z| < 2: ordering is consistent with random.")

    print()

    out_path = tag_dir / "permutation_sequence.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  → wrote {out_path}")


if __name__ == "__main__":
    main()
