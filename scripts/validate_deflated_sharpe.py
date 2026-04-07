#!/usr/bin/env python3
"""
3f — Deflated Sharpe with realistic N (Quant Guild #101 — Sharpe Ratio).

Background: The Sharpe ratio is monotonically increasing in the number
of strategies/parameters you try. With enough tries, you can find a
"profitable" strategy on pure noise. The deflated Sharpe formula
(Bailey & López de Prado, 2014) corrects for this by subtracting an
upper-bound on the Sharpe achievable by random selection across N
trials:

    SR_deflated = SR - sqrt( (2 * ln N) / T )  × annualisation_factor

where N = number of trials (parameter configs tried) and T = number
of trades observed.

backtest.py:_compute_metrics already implements this, but `n_trials`
is set per-engine call (defaults to 1, no penalty). The honest n_trials
is the *cumulative* number of parameter combinations tried during the
project: every kou_lambda sweep, every edge_threshold tweak, every
filtration threshold, etc. A defensible upper bound is N=100 to N=1000.

This script takes a trade dump and reports SR, deflated SR at
N ∈ {1, 10, 100, 1000, 10000}. The "deploy" decision should be made on
the deflated number, not the raw Sharpe.

Reads:     validation_runs/<tag>/trades_test.parquet  (and config.json)
Writes:    validation_runs/<tag>/deflated_sharpe.json
Prints:    a table of (N, SR_deflated, p_hat)

Usage:
    python scripts/validate_deflated_sharpe.py validation_runs/postfix_btc_5m
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


# NOTE: backtest.py:_compute_metrics annualises Sharpe with sqrt(96)
# regardless of market. We mirror that exactly so the raw Sharpe
# reported here matches what `backtest.py` prints. Both backtest.py
# and this script therefore report Sharpe in "96-period" units, not
# in true annualised units. If/when backtest.py is corrected to use
# per-market annualisation (sqrt(96) for 15m, sqrt(288) for 5m), this
# script should be updated in lockstep.
ANNUALISATION_FACTOR = 96


def sharpe(pnls: np.ndarray) -> float:
    if len(pnls) < 2:
        return 0.0
    mean = float(pnls.mean())
    std = float(pnls.std(ddof=1))
    if std <= 0:
        return 0.0
    return mean / std * math.sqrt(ANNUALISATION_FACTOR)


def deflated_sharpe(sr: float, n_trials: int, n_trades: int) -> float:
    """Deflated SR matching backtest.py:_compute_metrics exactly.

    haircut = sqrt(2 * ln(N) / T) * sqrt(96)
    SR_deflated = SR - haircut

    This is the "haircut" form of Bailey & López de Prado's deflated
    Sharpe (the full DSR also corrects for skew/kurtosis). Form is
    intentionally identical to backtest.py:2770 so per-N values from
    this script can be cross-checked against backtest.py output.
    """
    if n_trials <= 1 or n_trades < 2:
        return sr
    haircut = math.sqrt(2.0 * math.log(n_trials) / n_trades) \
        * math.sqrt(ANNUALISATION_FACTOR)
    return sr - haircut


def main():
    ap = argparse.ArgumentParser(
        description="Deflated Sharpe sweep across N (multiple-testing penalty)")
    ap.add_argument("tag_dir", type=Path)
    ap.add_argument("--n-list", type=str, default="1,10,100,1000,10000",
                    help="Comma-separated values of N (number of strategies "
                         "tried) to sweep. Default 1,10,100,1000,10000.")
    args = ap.parse_args()

    tag_dir = args.tag_dir.resolve()
    trades_path = tag_dir / "trades_test.parquet"
    if not trades_path.exists():
        print(f"ERROR: {trades_path} not found", file=sys.stderr)
        sys.exit(1)

    trades = pd.read_parquet(trades_path)
    pnls = trades["pnl"].to_numpy(dtype=np.float64)
    n_trades = len(pnls)
    if n_trades < 2:
        print("ERROR: need at least 2 trades", file=sys.stderr)
        sys.exit(1)

    sr_raw = sharpe(pnls)

    print(f"Loaded {n_trades} trades from {trades_path.name}")
    print(f"  Annualisation factor (matches backtest.py): "
          f"sqrt({ANNUALISATION_FACTOR})")
    print(f"  Raw Sharpe (matches backtest.py output):    {sr_raw:+.4f}")
    print()

    n_list = [int(s.strip()) for s in args.n_list.split(",") if s.strip()]
    rows = []
    for n in n_list:
        sr_def = deflated_sharpe(sr_raw, n, n_trades)
        rows.append({"n_trials": n, "sharpe_deflated": sr_def,
                     "haircut": sr_raw - sr_def})

    print("=" * 60)
    print("  DEFLATED SHARPE SWEEP")
    print("=" * 60)
    print(f"  {'N (trials)':>12}  {'SR_deflated':>14}  {'haircut':>10}")
    for r in rows:
        marker = ""
        if r["sharpe_deflated"] <= 0:
            marker = "  ← negative under this N"
        print(f"  {r['n_trials']:>12}  {r['sharpe_deflated']:>+14.4f}  "
              f"{r['haircut']:>+10.4f}{marker}")
    print()
    print("  Interpretation:")
    print("  - N=1   : honest if you only ever ran one configuration")
    print("  - N=10  : honest for a small parameter tune (~10 configs)")
    print("  - N=100 : honest if you've done a full sweep over 1-2 hyperparams")
    print("  - N=1000: honest for a research project with multiple sweeps")
    print("  - N=10k : extreme upper bound; only use if you sweep continuously")
    print()
    print("  Recommendation: don't deploy on a strategy whose deflated Sharpe")
    print("  goes negative at the N you actually have tried. Bailey/LdP show")
    print("  N=1000 is reasonable for an ML/quant project of any depth.")

    out_path = tag_dir / "deflated_sharpe.json"
    with open(out_path, "w") as f:
        json.dump({
            "n_trades": int(n_trades),
            "annualisation_factor": ANNUALISATION_FACTOR,
            "sharpe_raw": float(sr_raw),
            "sweep": rows,
        }, f, indent=2)
    print()
    print(f"  → wrote {out_path}")


if __name__ == "__main__":
    main()
