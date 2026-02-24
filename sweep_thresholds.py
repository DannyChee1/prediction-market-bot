#!/usr/bin/env python3
"""
Parameter sweep for tiered edge thresholds.

Sweeps (base, step) grid for BTC and ETH markets with max_trades_per_window=4.
Includes split-half overfitting mitigation.

Usage:
    py -3 sweep_thresholds.py              # sweep both markets
    py -3 sweep_thresholds.py --market btc # BTC only
    py -3 sweep_thresholds.py --market eth # ETH only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from backtest import DiffusionSignal, BacktestEngine, DATA_DIR
from market_config import MARKET_CONFIGS, get_config

# ── Sweep grid ──────────────────────────────────────────────────────────────

BASE_GRID = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
STEP_GRID = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]
EARLY_MULT_GRID = [0.0, 0.5, 1.0, 2.0, 4.0]

MAX_TRADES_PER_WINDOW = 4
MAX_DD_PCT_GUARDRAIL = 0.30

# Per-market configs
MARKET_PARAMS = {
    "btc": dict(
        bankroll=70.0,
        kelly_fraction=0.25,
        max_bet_fraction=0.05,
        signal_overrides={},
    ),
    "eth": dict(
        bankroll=30.0,
        kelly_fraction=0.25,
        max_bet_fraction=0.10,
        signal_overrides=dict(
            reversion_discount=0.15,
            momentum_lookback_s=15,
            momentum_majority=0.7,
            spread_edge_penalty=0.2,
        ),
    ),
}


def run_sweep(
    market: str,
    data_dir: Path,
    bankroll: float,
    kelly_fraction: float,
    max_bet_fraction: float,
    signal_overrides: dict,
    slugs: list[str] | None = None,
    quiet: bool = False,
) -> list[dict]:
    """Run full grid sweep, returns list of result dicts."""
    import io, contextlib

    rows = []
    total = len(BASE_GRID) * len(STEP_GRID) * len(EARLY_MULT_GRID)
    i = 0

    for base in BASE_GRID:
        for step in STEP_GRID:
            for emult in EARLY_MULT_GRID:
                i += 1
                if not quiet:
                    print(f"  [{market.upper()}] {i}/{total}: base={base:.2f} step={step:.2f} emult={emult:.1f}", end="")

                signal = DiffusionSignal(
                    bankroll=bankroll,
                    edge_threshold=base,
                    edge_threshold_step=step,
                    early_edge_mult=emult,
                    kelly_fraction=kelly_fraction,
                    max_bet_fraction=max_bet_fraction,
                    **signal_overrides,
                )
                engine = BacktestEngine(
                    signal=signal,
                    data_dir=data_dir,
                    initial_bankroll=bankroll,
                    max_trades_per_window=MAX_TRADES_PER_WINDOW,
                )

                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    _, metrics, _ = engine.run()

                row = {
                    "market": market,
                    "base": base,
                    "step": step,
                    "emult": emult,
                    **metrics,
                }
                rows.append(row)

                if not quiet:
                    pnl = metrics.get("total_pnl", 0)
                    n = metrics.get("n_trades", 0)
                    wr = metrics.get("win_rate", 0)
                    dd = metrics.get("max_dd_pct", 0)
                    print(f"  -> PnL=${pnl:+.2f}  trades={n}  WR={wr:.0%}  DD={dd:.1%}")

    return rows


def split_half_validation(
    market: str,
    data_dir: Path,
    bankroll: float,
    kelly_fraction: float,
    max_bet_fraction: float,
    signal_overrides: dict,
    best_base: float,
    best_step: float,
    best_emult: float = 4.0,
) -> dict:
    """Split data in half and verify the optimal config is profitable on both."""
    import io, contextlib, tempfile

    # Load all parquet files
    frames = []
    for f in sorted(data_dir.glob("*.parquet")):
        part = pd.read_parquet(f)
        if "chainlink_btc" in part.columns and "chainlink_price" not in part.columns:
            part.rename(columns={"chainlink_btc": "chainlink_price"}, inplace=True)
        frames.append(part)
    if not frames:
        return {"first_half_pnl": 0, "second_half_pnl": 0, "both_profitable": False}

    df = pd.concat(frames, ignore_index=True)
    df.sort_values("ts_ms", inplace=True, ignore_index=True)

    slugs = df["market_slug"].unique()
    mid = len(slugs) // 2
    first_slugs = set(slugs[:mid])
    second_slugs = set(slugs[mid:])

    results = {}
    for label, slug_set in [("first_half", first_slugs), ("second_half", second_slugs)]:
        subset = df[df["market_slug"].isin(slug_set)]
        if subset.empty:
            results[f"{label}_pnl"] = 0.0
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            subset.to_parquet(tmp_path / "data.parquet")

            signal = DiffusionSignal(
                bankroll=bankroll,
                edge_threshold=best_base,
                edge_threshold_step=best_step,
                early_edge_mult=best_emult,
                kelly_fraction=kelly_fraction,
                max_bet_fraction=max_bet_fraction,
                **signal_overrides,
            )
            engine = BacktestEngine(
                signal=signal,
                data_dir=tmp_path,
                initial_bankroll=bankroll,
                max_trades_per_window=MAX_TRADES_PER_WINDOW,
            )

            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                _, metrics, _ = engine.run()

            results[f"{label}_pnl"] = metrics.get("total_pnl", 0)
            results[f"{label}_trades"] = metrics.get("n_trades", 0)
            results[f"{label}_wr"] = metrics.get("win_rate", 0)

    results["both_profitable"] = (
        results.get("first_half_pnl", 0) > 0
        and results.get("second_half_pnl", 0) > 0
    )
    return results


def sweep_market(market: str):
    """Full sweep + validation for one market."""
    config = get_config(market)
    data_dir = DATA_DIR / config.data_subdir
    params = MARKET_PARAMS[market]

    print(f"\n{'='*62}")
    print(f"  SWEEP: {config.display_name}  |  bankroll=${params['bankroll']}")
    print(f"  kelly={params['kelly_fraction']}  max_bet={params['max_bet_fraction']}")
    total_combos = len(BASE_GRID) * len(STEP_GRID) * len(EARLY_MULT_GRID)
    print(f"  grid: {len(BASE_GRID)} base x {len(STEP_GRID)} step x {len(EARLY_MULT_GRID)} emult = {total_combos} combos")
    print(f"{'='*62}\n")

    rows = run_sweep(
        market=market,
        data_dir=data_dir,
        bankroll=params["bankroll"],
        kelly_fraction=params["kelly_fraction"],
        max_bet_fraction=params["max_bet_fraction"],
        signal_overrides=params["signal_overrides"],
    )

    df = pd.DataFrame(rows)

    # Filter by max drawdown guardrail
    valid = df[df["max_dd_pct"] < MAX_DD_PCT_GUARDRAIL].copy()
    if valid.empty:
        print(f"\n  WARNING: All configs exceed {MAX_DD_PCT_GUARDRAIL:.0%} max drawdown!")
        valid = df.copy()

    # Sort by total PnL
    valid.sort_values("total_pnl", ascending=False, inplace=True)

    print(f"\n{'='*62}")
    print(f"  TOP 5 CONFIGS: {config.display_name}")
    print(f"{'='*62}")
    cols = ["base", "step", "emult", "n_trades", "total_pnl", "win_rate", "max_dd_pct", "sharpe"]
    top5 = valid.head(5)
    print(top5[cols].to_string(index=False))

    # Check parameter clustering
    if len(top5) >= 3:
        base_range = top5["base"].max() - top5["base"].min()
        step_range = top5["step"].max() - top5["step"].min()
        print(f"\n  Parameter spread in top 5: base range={base_range:.2f}, step range={step_range:.2f}")
        if base_range <= 0.06 and step_range <= 0.06:
            print("  -> Clustered (good sign, less likely overfit)")
        else:
            print("  -> Dispersed (caution: may be noisy)")

    # Best config
    best = valid.iloc[0]
    best_base = best["base"]
    best_step = best["step"]
    best_emult = best["emult"]

    print(f"\n  BEST: base={best_base:.2f}  step={best_step:.2f}  emult={best_emult:.1f}  "
          f"PnL=${best['total_pnl']:+.2f}  trades={int(best['n_trades'])}  "
          f"WR={best['win_rate']:.0%}  DD={best['max_dd_pct']:.1%}")

    # Split-half validation
    print(f"\n  Running split-half validation...")
    split = split_half_validation(
        market=market,
        data_dir=data_dir,
        bankroll=params["bankroll"],
        kelly_fraction=params["kelly_fraction"],
        max_bet_fraction=params["max_bet_fraction"],
        signal_overrides=params["signal_overrides"],
        best_base=best_base,
        best_step=best_step,
        best_emult=best_emult,
    )
    print(f"  First half:  PnL=${split.get('first_half_pnl', 0):+.2f}  "
          f"trades={split.get('first_half_trades', 0)}  "
          f"WR={split.get('first_half_wr', 0):.0%}")
    print(f"  Second half: PnL=${split.get('second_half_pnl', 0):+.2f}  "
          f"trades={split.get('second_half_trades', 0)}  "
          f"WR={split.get('second_half_wr', 0):.0%}")
    if split["both_profitable"]:
        print("  -> PASS: profitable on both halves")
    else:
        print("  -> FAIL: not profitable on both halves (possible overfit)")

    return {
        "market": market,
        "best_base": best_base,
        "best_step": best_step,
        "best_emult": best_emult,
        "best_pnl": best["total_pnl"],
        "all_results": df,
        "split_validation": split,
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep tiered edge thresholds")
    parser.add_argument(
        "--market", default=None, choices=list(MARKET_CONFIGS),
        help="Market to sweep (default: both)",
    )
    parser.add_argument(
        "--maker", action="store_true",
        help="Sweep with maker mode (0%% fee, mid-price entry)",
    )
    args = parser.parse_args()

    if args.maker:
        for m in MARKET_PARAMS:
            MARKET_PARAMS[m]["signal_overrides"]["maker_mode"] = True

    markets = [args.market] if args.market else list(MARKET_PARAMS.keys())

    all_results = {}
    for m in markets:
        result = sweep_market(m)
        all_results[m] = result

    # Final summary
    print(f"\n{'='*62}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*62}")
    for m, r in all_results.items():
        split = r["split_validation"]
        tag = "PASS" if split["both_profitable"] else "FAIL"
        print(f"  {m.upper()}: base={r['best_base']:.2f}  step={r['best_step']:.2f}  "
              f"emult={r['best_emult']:.1f}  PnL=${r['best_pnl']:+.2f}  split={tag}")

    print(f"\n  Update live_trader.py with these values:")
    for m, r in all_results.items():
        print(f"    {m.upper()}: edge_threshold={r['best_base']:.2f}, "
              f"edge_threshold_step={r['best_step']:.2f}, "
              f"early_edge_mult={r['best_emult']:.1f}")


if __name__ == "__main__":
    main()
