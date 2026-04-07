#!/usr/bin/env python3
"""
3c — Ergodicity Monte-Carlo simulation (Quant Guild #81 — Ergodicity).

Question:  In a binary market your wealth evolves multiplicatively
(every trade scales the bankroll by some factor). For multiplicative
processes, the *ensemble average* (mean across many simulated traders)
diverges from the *time average* (what a single trader actually
experiences over time). The backtest reports total_pnl, which is
closer to the ensemble average. The number that matters for the user
is the time average, which equals the median of bootstrapped wealth
paths.

If ensemble_mean > median by a wide margin, the strategy is
*non-ergodic* — a few lucky paths inflate the mean while the typical
trader gets a much worse outcome. This is exactly the "I'll never
make this much in real life" trap that flat-Kelly systems fall into
on noisy edges.

Method:    Bootstrap-sample the post-fix trades to build N synthetic
trader paths of length T. For each path, compute terminal wealth using
the SAME sizing rule the live bot uses (proportional to trade size).
Report:

  * Mean terminal wealth (ensemble average)
  * Median terminal wealth (time average ≈ what a typical trader sees)
  * Fraction of paths ending below initial bankroll (loss-rate)
  * Fraction of paths ending below 50% of initial (ruin proxy)
  * Time-average growth rate g = (1/T) Σ log(1 + r_t)

Reads:     validation_runs/<tag>/trades_test.parquet
Writes:    validation_runs/<tag>/ergodicity.json
           validation_runs/<tag>/ergodicity.png
Prints:    a summary table

Usage:
    python scripts/validate_ergodicity.py validation_runs/postfix_btc_5m
    python scripts/validate_ergodicity.py --n-paths 5000 --path-length 1000 \
        validation_runs/postfix_btc
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def simulate_paths(
    relative_pnls: np.ndarray,
    n_paths: int,
    path_length: int,
    initial_bankroll: float,
    seed: int,
) -> np.ndarray:
    """Bootstrap n_paths wealth trajectories of length path_length.

    `relative_pnls[i]` is `trade_i_pnl / bankroll_at_time_of_trade_i` —
    i.e. the actual fraction of the current bankroll won/lost on each
    historical trade. Sampling from this distribution preserves BOTH
    the per-trade outcome variance AND the strategy's actual sizing
    scheme (Kelly-driven, bigger stakes on higher-edge trades). It is
    NOT the same as `pnl_pct` which is per-stake return ignoring stake
    size.

    Returns: array shape (n_paths, path_length+1) of bankroll values
    starting at initial_bankroll.
    """
    rng = np.random.default_rng(seed)
    n = len(relative_pnls)
    if n == 0:
        return np.full((n_paths, path_length + 1), initial_bankroll)

    paths = np.empty((n_paths, path_length + 1), dtype=np.float64)
    paths[:, 0] = initial_bankroll

    # Sample all trade indices at once
    idx = rng.integers(0, n, size=(n_paths, path_length))
    rels = relative_pnls[idx]  # shape (n_paths, path_length)

    # Multiplicative wealth update: bk *= 1 + relative_pnl
    multipliers = 1.0 + rels
    multipliers = np.maximum(multipliers, 1e-12)  # cap at near-zero

    # Cumulative product across the time axis
    cum = np.cumprod(multipliers, axis=1)
    paths[:, 1:] = initial_bankroll * cum
    return paths


def compute_stats(paths: np.ndarray, initial_bankroll: float) -> dict:
    terminal = paths[:, -1]
    n_paths = len(terminal)

    # Ensemble vs time average
    ensemble_mean = float(terminal.mean())
    median_terminal = float(np.median(terminal))

    # Per-path log-return time average
    log_returns = np.log(np.clip(terminal / initial_bankroll, 1e-12, None))
    time_avg_growth_per_path = log_returns / max(paths.shape[1] - 1, 1)
    time_avg_growth_mean = float(time_avg_growth_per_path.mean())

    # Loss / ruin rates
    pct_below_initial = float((terminal < initial_bankroll).mean())
    pct_below_50pct = float((terminal < 0.5 * initial_bankroll).mean())
    pct_below_10pct = float((terminal < 0.1 * initial_bankroll).mean())
    pct_2x = float((terminal > 2.0 * initial_bankroll).mean())
    pct_10x = float((terminal > 10.0 * initial_bankroll).mean())

    return {
        "n_paths": int(n_paths),
        "path_length": int(paths.shape[1] - 1),
        "initial_bankroll": float(initial_bankroll),
        "ensemble_mean_terminal": ensemble_mean,
        "median_terminal": median_terminal,
        "ensemble_minus_median": ensemble_mean - median_terminal,
        "ensemble_over_median_ratio": (
            ensemble_mean / median_terminal if median_terminal > 0 else float("inf")
        ),
        "time_avg_growth_per_trade": time_avg_growth_mean,
        "pct_paths_below_initial": pct_below_initial,
        "pct_paths_below_50pct": pct_below_50pct,
        "pct_paths_below_10pct_proxy_ruin": pct_below_10pct,
        "pct_paths_above_2x": pct_2x,
        "pct_paths_above_10x": pct_10x,
        "terminal_p05": float(np.percentile(terminal, 5)),
        "terminal_p25": float(np.percentile(terminal, 25)),
        "terminal_p75": float(np.percentile(terminal, 75)),
        "terminal_p95": float(np.percentile(terminal, 95)),
    }


def plot_paths(paths: np.ndarray, initial_bankroll: float, out_path: Path,
               title: str):
    terminal = paths[:, -1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Path bundle (sample of 100 paths to keep it readable)
    sample = paths[: min(100, len(paths))]
    for p in sample:
        axes[0].plot(p, alpha=0.18, color="steelblue", linewidth=0.7)
    median_path = np.median(paths, axis=0)
    mean_path = paths.mean(axis=0)
    axes[0].plot(median_path, color="green", linewidth=2.0,
                 label=f"Median (final ${median_path[-1]:,.0f})")
    axes[0].plot(mean_path, color="red", linewidth=2.0, linestyle="--",
                 label=f"Ensemble mean (final ${mean_path[-1]:,.0f})")
    axes[0].axhline(initial_bankroll, color="k", linestyle=":", alpha=0.5)
    axes[0].set_xlabel("Trade #")
    axes[0].set_ylabel("Bankroll ($)")
    axes[0].set_yscale("log")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].set_title("100 sample wealth paths (log scale)")
    axes[0].grid(True, alpha=0.3)

    # Terminal wealth histogram
    bins = np.logspace(np.log10(max(terminal.min(), 1)),
                       np.log10(max(terminal.max(), 2)), 60)
    axes[1].hist(terminal, bins=bins, color="steelblue", alpha=0.7,
                 edgecolor="black", linewidth=0.3)
    axes[1].axvline(initial_bankroll, color="black", linestyle=":",
                    label=f"Initial ${initial_bankroll:,.0f}")
    axes[1].axvline(np.median(terminal), color="green", linewidth=2,
                    label=f"Median ${np.median(terminal):,.0f}")
    axes[1].axvline(terminal.mean(), color="red", linewidth=2, linestyle="--",
                    label=f"Mean ${terminal.mean():,.0f}")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Terminal bankroll ($)")
    axes[1].set_ylabel("Number of simulated traders")
    axes[1].legend(fontsize=9)
    axes[1].set_title("Distribution of terminal wealth across simulated traders")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _build_relative_pnls(trades: pd.DataFrame, bk_history: list[float],
                         initial_bankroll: float) -> np.ndarray:
    """Compute trade.pnl / bankroll_at_trade for each trade.

    bankroll_test.json is a list of N+1 values: bk[0] = initial,
    bk[i+1] = bk[i] + trade_i.pnl. So the bankroll JUST BEFORE trade i
    is bk[i].
    """
    if bk_history is None or len(bk_history) < len(trades) + 1:
        # Fall back: derive from cumsum
        bk = [float(initial_bankroll)]
        for p in trades["pnl"]:
            bk.append(bk[-1] + float(p))
        bk_history = bk
    bk_at_trade = np.asarray(bk_history[:-1], dtype=np.float64)
    pnls = trades["pnl"].to_numpy(dtype=np.float64)
    rel = pnls / np.maximum(bk_at_trade, 1.0)
    return rel


def main():
    ap = argparse.ArgumentParser(
        description="Ergodicity Monte-Carlo simulation")
    ap.add_argument("tag_dir", type=Path)
    ap.add_argument("--n-paths", type=int, default=2000,
                    help="Number of bootstrapped wealth paths (default 2000)")
    ap.add_argument("--path-length", type=int, default=500,
                    help="Trades per path (default 500)")
    ap.add_argument("--initial-bankroll", type=float, default=10_000.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tag_dir = args.tag_dir.resolve()
    trades_path = tag_dir / "trades_test.parquet"
    if not trades_path.exists():
        print(f"ERROR: {trades_path} not found", file=sys.stderr)
        sys.exit(1)

    trades = pd.read_parquet(trades_path)
    if trades.empty:
        print(f"ERROR: empty trade dump", file=sys.stderr)
        sys.exit(1)

    bk_path = tag_dir / "bankroll_test.json"
    bk_history = None
    if bk_path.exists():
        with open(bk_path) as f:
            bk_history = json.load(f)

    relative_pnls = _build_relative_pnls(
        trades, bk_history, args.initial_bankroll
    )

    print(f"Loaded {len(relative_pnls)} trades from {trades_path.name}")
    print(f"  Empirical p_win:    {(relative_pnls > 0).mean():.4f}")
    print(f"  Mean rel PnL:       {relative_pnls.mean():+.6f} per trade "
          f"(of bankroll-at-time)")
    print(f"  Std rel PnL:        {relative_pnls.std(ddof=1):+.6f}")
    print(f"  Simulating {args.n_paths} paths × {args.path_length} trades...")

    paths = simulate_paths(
        relative_pnls=relative_pnls,
        n_paths=args.n_paths,
        path_length=args.path_length,
        initial_bankroll=args.initial_bankroll,
        seed=args.seed,
    )
    stats = compute_stats(paths, args.initial_bankroll)
    stats["sizing_scheme"] = (
        "bootstrapped from actual per-trade pnl/bankroll_at_time "
        "(preserves Kelly-driven sizing)"
    )

    print()
    print("=" * 64)
    print("  ERGODICITY MONTE-CARLO RESULT")
    print("=" * 64)
    print(f"  N paths × length:           {stats['n_paths']} × "
          f"{stats['path_length']}")
    print(f"  Initial bankroll:           ${stats['initial_bankroll']:,.0f}")
    print()
    print(f"  Ensemble mean terminal:     "
          f"${stats['ensemble_mean_terminal']:>12,.2f}")
    print(f"  Median terminal:            "
          f"${stats['median_terminal']:>12,.2f}")
    print(f"  Mean / median ratio:        "
          f"{stats['ensemble_over_median_ratio']:>12.2f}×")
    print()
    print(f"  Time-avg growth per trade:  "
          f"{stats['time_avg_growth_per_trade']:>12.6f}")
    print(f"    (positive = typical trader grows; "
          f"negative = typical trader bleeds)")
    print()
    print(f"  Terminal wealth percentiles:")
    print(f"     5%:   ${stats['terminal_p05']:>12,.0f}")
    print(f"    25%:   ${stats['terminal_p25']:>12,.0f}")
    print(f"    50%:   ${stats['median_terminal']:>12,.0f}")
    print(f"    75%:   ${stats['terminal_p75']:>12,.0f}")
    print(f"    95%:   ${stats['terminal_p95']:>12,.0f}")
    print()
    print(f"  P(below initial):     {stats['pct_paths_below_initial']:.2%}")
    print(f"  P(below 50% initial): {stats['pct_paths_below_50pct']:.2%}")
    print(f"  P(ruin proxy <10%):   {stats['pct_paths_below_10pct_proxy_ruin']:.2%}")
    print(f"  P(above 2x):          {stats['pct_paths_above_2x']:.2%}")
    print(f"  P(above 10x):         {stats['pct_paths_above_10x']:.2%}")
    print()

    # Verdict
    if stats["time_avg_growth_per_trade"] < 0:
        print("  ✗ Time-average growth is NEGATIVE: the typical trader bleeds.")
        print("    The +PnL backtest result is concentrated in a small set of")
        print("    lucky paths. NOT deployable at this sizing.")
    elif stats["pct_paths_below_initial"] > 0.50:
        print("  ⚠ Over half of simulated traders end below their starting")
        print("    bankroll despite positive ensemble mean. Sizing is too")
        print("    aggressive for the noise level — reduce kelly_fraction.")
    elif stats["ensemble_over_median_ratio"] > 2.0:
        print(f"  ⚠ Ensemble mean / median = "
              f"{stats['ensemble_over_median_ratio']:.2f}×. The strategy is")
        print("    NON-ERGODIC: backtest PnL overstates what a typical trader")
        print("    will see. Consider lower fraction_per_trade.")
    else:
        print("  ✓ Time-average growth positive AND ensemble≈median.")
        print("    Strategy is ergodic at this sizing — backtest PnL is a")
        print("    realistic expectation for a typical trader.")
    print()

    # Persist
    out_json = tag_dir / "ergodicity.json"
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  → wrote {out_json}")

    out_png = tag_dir / "ergodicity.png"
    plot_paths(paths, args.initial_bankroll, out_png,
               title=f"{tag_dir.name} | bootstrap from actual sizing "
                     f"| {len(relative_pnls)} sample trades")
    print(f"  → wrote {out_png}")


if __name__ == "__main__":
    main()
