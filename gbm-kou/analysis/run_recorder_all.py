#!/usr/bin/env python3
"""Run recorders for all markets simultaneously."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from market_config import MARKET_CONFIGS

procs = []
for market in MARKET_CONFIGS:
    args = [sys.executable, "recorder.py", "--market", market] + sys.argv[1:]
    procs.append(subprocess.Popen(args))
    print(f"  Started recorder --market {market}  (pid {procs[-1].pid})")

try:
    for p in procs:
        p.wait()
except KeyboardInterrupt:
    print("\n  Stopping all recorders...")
    for p in procs:
        p.terminate()
    for p in procs:
        p.wait()
    print("  Done.")
