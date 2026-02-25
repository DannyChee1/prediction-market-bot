#!/usr/bin/env python3
"""Run backtests for all markets simultaneously."""

import subprocess
import sys

from market_config import MARKET_CONFIGS

procs = []
for market in MARKET_CONFIGS:
    args = [sys.executable, "backtest.py", "--market", market] + sys.argv[1:]
    procs.append(subprocess.Popen(args))
    print(f"  Started backtest --market {market}  (pid {procs[-1].pid})")

try:
    for p in procs:
        p.wait()
except KeyboardInterrupt:
    print("\n  Stopping...")
    for p in procs:
        p.terminate()
    for p in procs:
        p.wait()
