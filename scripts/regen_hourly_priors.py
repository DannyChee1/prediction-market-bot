#!/usr/bin/env python3
"""
Regenerate the time-of-day / day-of-week σ priors used by
backtest._time_prior_sigma (Quant Guild #93 — non-stationarity).

The priors live in backtest.py as hardcoded constants:

    _HOURLY_VOL_MULT  : dict[int, float]   # 0..23 → multiplier
    _DOW_VOL_MULT     : dict[int, float]   # 0..6  → multiplier
    _GLOBAL_MEAN_SIGMA : float             # per-second σ baseline

These were computed once from "1844 BTC windows + 1000 hours of
Binance klines" and have not been refreshed. Quant Guild #93 warns
that non-stationary processes need their priors refreshed regularly:
if any hour bucket drifts > 20% from the frozen value, the prior is
stale.

This script:
  1. Walks all parquet windows under data/<market>/.
  2. Computes the per-second realized σ for every 1-second tick using
     a 90-tick rolling window (matching backtest._compute_vol).
  3. Buckets ticks by UTC hour and weekday.
  4. Reports each bucket's mean σ and the multiplier vs the global
     mean.
  5. Writes the result to data/<market>/hourly_priors.json so
     `backtest.py` can load it at import time (with fallback to the
     existing constants).

Usage:
    python scripts/regen_hourly_priors.py --market btc_15m
    python scripts/regen_hourly_priors.py --market btc_5m

Output (one file per market):
    data/<market_subdir>/hourly_priors.json
        {
          "global_mean_sigma": float,
          "hourly_mult": {0: float, 1: float, ..., 23: float},
          "dow_mult":    {0: float, 1: float, ..., 6: float},
          "n_obs":       int,
          "computed_utc": "ISO 8601 timestamp",
          "data_subdir": "btc_15m",
        }
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backtest import _compute_vol_deduped  # noqa: E402
from market_config import MARKET_CONFIGS  # noqa: E402

DATA_DIR = REPO_ROOT / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True,
                    choices=list(MARKET_CONFIGS.keys()))
    ap.add_argument("--vol-lookback-s", type=int, default=90)
    ap.add_argument("--sample-every", type=int, default=15)
    args = ap.parse_args()

    cfg = MARKET_CONFIGS[args.market]
    subdir = DATA_DIR / cfg.data_subdir
    if not subdir.exists():
        print(f"ERROR: data dir not found: {subdir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(subdir.glob("*.parquet"))
    print(f"Scanning {len(files)} parquet files in {subdir} ...")

    sigma_by_hour: dict[int, list[float]] = defaultdict(list)
    sigma_by_dow:  dict[int, list[float]] = defaultdict(list)
    all_sigmas: list[float] = []

    for f in files:
        df = pd.read_parquet(f)
        if df.empty:
            continue
        if "chainlink_btc" in df.columns and "chainlink_price" not in df.columns:
            df = df.rename(columns={"chainlink_btc": "chainlink_price"})
        if "chainlink_price" not in df.columns:
            continue
        prices = df["chainlink_price"].tolist()
        ts_list = df["ts_ms"].tolist()
        n = len(prices)
        if n < args.vol_lookback_s + 2:
            continue
        for idx in range(args.vol_lookback_s, n, args.sample_every):
            lo = max(0, idx - args.vol_lookback_s)
            sigma = _compute_vol_deduped(prices[lo:idx + 1],
                                         ts_list[lo:idx + 1])
            if sigma <= 0:
                continue
            t = _dt.datetime.fromtimestamp(
                ts_list[idx] / 1000, tz=_dt.timezone.utc
            )
            sigma_by_hour[t.hour].append(sigma)
            sigma_by_dow[t.weekday()].append(sigma)
            all_sigmas.append(sigma)

    if not all_sigmas:
        print("No valid samples", file=sys.stderr)
        sys.exit(1)

    global_mean = float(np.mean(all_sigmas))
    print(f"  {len(all_sigmas)} valid sigma samples")
    print(f"  Global mean σ: {global_mean:.3e}")

    hourly_mult: dict[int, float] = {}
    print("\n  Hour-of-day multipliers (vs global mean):")
    print(f"  {'hour':>4}  {'n':>8}  {'σ_mean':>10}  {'mult':>6}")
    for h in range(24):
        s = sigma_by_hour.get(h, [])
        if s:
            mean_h = float(np.mean(s))
            mult = mean_h / global_mean if global_mean > 0 else 1.0
        else:
            mean_h = float("nan")
            mult = 1.0
        hourly_mult[h] = round(mult, 3)
        print(f"  {h:>4}  {len(s):>8}  {mean_h:>10.3e}  {mult:>6.3f}")

    dow_mult: dict[int, float] = {}
    DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print("\n  Day-of-week multipliers (vs global mean):")
    print(f"  {'dow':>4}  {'name':>4}  {'n':>8}  {'σ_mean':>10}  {'mult':>6}")
    for d in range(7):
        s = sigma_by_dow.get(d, [])
        if s:
            mean_d = float(np.mean(s))
            mult = mean_d / global_mean if global_mean > 0 else 1.0
        else:
            mean_d = float("nan")
            mult = 1.0
        dow_mult[d] = round(mult, 3)
        print(f"  {d:>4}  {DOW_NAMES[d]:>4}  {len(s):>8}  {mean_d:>10.3e}  "
              f"{mult:>6.3f}")

    out = {
        "global_mean_sigma": global_mean,
        "hourly_mult": hourly_mult,
        "dow_mult": dow_mult,
        "n_obs": len(all_sigmas),
        "computed_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "data_subdir": cfg.data_subdir,
    }
    out_path = subdir / "hourly_priors.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  → wrote {out_path}")
    print(f"  backtest.py will pick this up automatically on next import.")


if __name__ == "__main__":
    main()
