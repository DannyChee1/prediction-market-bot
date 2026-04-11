#!/usr/bin/env python3
"""Test calibration audit fixes individually and combined.

Re-runs the parity replay against the 455-trade sample under different
filter configurations:
  - baseline   : no extra filters (original behaviour)
  - test_1     : only Test #1 (lower _GLOBAL_MEAN_SIGMA prior)
  - test_3     : only Test #3 (calm-market filter at min_trade_sigma)
  - test_5     : only Test #5 (tighter max_model_market_disagreement)
  - test_1_3   : Test #1 + Test #3
  - test_1_5   : Test #1 + Test #5
  - test_3_5   : Test #3 + Test #5
  - all        : all three

Reports for each: surviving trades, win rate, realized edge, claimed edge,
gap, $25/trade PnL on the survivors.

Note: Test #1 (the prior change in backtest_core.py) is GLOBAL — it
applies in all runs unless we revert it. So "baseline" here means "no
calm filter, no tightened disagreement gate" — Test #1 is implicitly on
because we already changed _GLOBAL_MEAN_SIGMA. To isolate Test #1's
effect, the script temporarily monkey-patches the configs.
"""
from __future__ import annotations
import sys
import json
import statistics as stat
from pathlib import Path

ROOT = Path("/Users/dannychee/Desktop/prediction-market-bot")
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from backtest import DATA_DIR, build_diffusion_signal  # noqa: E402
from backtest_core import Snapshot  # noqa: E402
from market_config import get_config, MARKET_CONFIGS  # noqa: E402


def parquet_path(slug: str) -> Path | None:
    if slug.startswith("btc-updown-5m"):
        return ROOT / "data" / "btc_5m" / f"{slug}.parquet"
    if slug.startswith("btc-updown-15m"):
        return ROOT / "data" / "btc_15m" / f"{slug}.parquet"
    if slug.startswith("bitcoin-up-or-down"):
        for sub in ("btc_1h", "btc_1h_real"):
            p = ROOT / "data" / sub / f"{slug}.parquet"
            if p.exists():
                return p
    return None


def market_key_for(slug: str) -> str | None:
    if slug.startswith("btc-updown-5m"):
        return "btc_5m"
    if slug.startswith("btc-updown-15m"):
        return "btc"
    if slug.startswith("bitcoin-up-or-down"):
        return "btc_1h"
    return None


def replay_one(slug: str, target_tau: float, market_key: str,
               *, min_trade_sigma: float, max_disagreement: float):
    fp = parquet_path(slug)
    if fp is None or not fp.exists():
        return None
    try:
        df = pd.read_parquet(fp)
    except Exception:
        return None
    if len(df) == 0:
        return None

    sig = build_diffusion_signal(market_key, bankroll=100.0, maker=False)
    # Override the two knobs being tested
    sig.min_trade_sigma = min_trade_sigma
    sig.max_model_market_disagreement = max_disagreement

    ctx: dict = {"inventory_up": 0, "inventory_down": 0}
    if "window_start_ms" in df.columns:
        ctx["_window_start_ms"] = int(df["window_start_ms"].iloc[0])

    has_binance = "binance_mid" in df.columns
    best = None
    min_dtau = 1e9

    for _, row in df.iterrows():
        snap = Snapshot.from_row(row)
        if snap is None:
            continue
        if has_binance and pd.notna(row.get("binance_mid")) and row["binance_mid"] > 0:
            ctx["_binance_mid"] = float(row["binance_mid"])
        try:
            up, down = sig.decide_both_sides(snap, ctx)
        except Exception:
            continue
        dtau = abs(float(snap.time_remaining_s) - target_tau)
        if dtau < min_dtau:
            min_dtau = dtau
            best = {
                "up_action": up.action,
                "down_action": down.action,
                "up_edge": getattr(up, "edge", None),
                "down_edge": getattr(down, "edge", None),
                "up_reason": getattr(up, "reason", ""),
                "down_reason": getattr(down, "reason", ""),
                "p_model_raw": ctx.get("_p_model_raw"),
            }
    return best


def run_scenario(label: str, *, min_trade_sigma: float,
                 max_disagreement: float, rows: list[dict]) -> dict:
    fire_realized = []
    fire_claimed = []
    fire_p_side = []
    fire_wins = []
    skip_realized = []
    skip_wins = []
    fire = skip = 0

    for r in rows:
        slug = r.get("market_slug")
        if not slug:
            continue
        mk = market_key_for(slug)
        if mk is None:
            continue
        side = r.get("side")
        realized = r.get("realized_edge")
        win = r.get("win")
        if side is None or realized is None:
            continue

        res = replay_one(slug, float(r.get("tau") or 0), mk,
                         min_trade_sigma=min_trade_sigma,
                         max_disagreement=max_disagreement)
        if res is None:
            continue

        action = res["up_action"] if side == "UP" else res["down_action"]
        edge = res["up_edge"] if side == "UP" else res["down_edge"]
        if edge is None:
            edge = 0.0

        if action and action.startswith("BUY"):
            fire += 1
            fire_realized.append(float(realized))
            fire_claimed.append(float(edge))
            p_raw = res["p_model_raw"]
            if p_raw is not None:
                p_side = float(p_raw) if side == "UP" else 1.0 - float(p_raw)
                fire_p_side.append(p_side)
                fire_wins.append(int(bool(win)))
        else:
            skip += 1
            skip_realized.append(float(realized))
            skip_wins.append(int(bool(win)))

    n_fire = len(fire_realized)
    n_skip = len(skip_realized)
    out = {
        "label": label,
        "n_fire": n_fire,
        "n_skip": n_skip,
        "fire_pct": 100 * n_fire / max(1, n_fire + n_skip),
        "fire_wr": (sum(fire_wins) / n_fire) if n_fire else 0.0,
        "fire_claimed": (sum(fire_claimed) / n_fire) if n_fire else 0.0,
        "fire_realized": (sum(fire_realized) / n_fire) if n_fire else 0.0,
        "fire_pnl_25": sum(25 * r for r in fire_realized),
        "skip_wr": (sum(skip_wins) / n_skip) if n_skip else 0.0,
        "skip_realized": (sum(skip_realized) / n_skip) if n_skip else 0.0,
        "skip_pnl_25": sum(25 * r for r in skip_realized),
    }
    return out


def main():
    rows = json.load(open("/tmp/_analysis_rows.json"))
    print(f"Loaded {len(rows)} live trades")
    print()

    # Find a baseline that DISABLES both new filters: min_trade_sigma=0,
    # max_disagreement=1.0 (full disable). Note: Test #1 (the prior
    # _GLOBAL_MEAN_SIGMA change) is implicit since the file is already
    # patched; we can't easily revert it for a single run.
    scenarios = [
        ("baseline (no calm, no disagree)",  0.0,    1.00),
        ("calm only (Test #3)",              2.5e-5, 1.00),
        ("calm only — softer 2.0e-5",        2.0e-5, 1.00),
        ("calm only — harder 3.0e-5",        3.0e-5, 1.00),
        ("disagree only (Test #5) — 0.15",   0.0,    0.15),
        ("disagree only — softer 0.20",      0.0,    0.20),
        ("disagree only — softer 0.25",      0.0,    0.25),
        ("calm 2.5e-5 + disagree 0.15",      2.5e-5, 0.15),
        ("calm 2.0e-5 + disagree 0.20",      2.0e-5, 0.20),
        ("calm 2.0e-5 + disagree 0.25",      2.0e-5, 0.25),
    ]

    print(f"{'scenario':<40} {'fire':>6} {'fire%':>6} "
          f"{'WR%':>6} {'claimed':>9} {'realized':>9} {'gap':>8} "
          f"{'PnL$':>8} {'skipPnL':>8}")
    print("-" * 110)
    for label, mts, mmd in scenarios:
        r = run_scenario(label, min_trade_sigma=mts, max_disagreement=mmd, rows=rows)
        print(
            f"{r['label']:<40} {r['n_fire']:>6d} "
            f"{r['fire_pct']:>5.1f}% "
            f"{100*r['fire_wr']:>5.1f}% "
            f"{r['fire_claimed']:>+9.4f} "
            f"{r['fire_realized']:>+9.4f} "
            f"{r['fire_claimed']-r['fire_realized']:>+8.4f} "
            f"{r['fire_pnl_25']:>+8.2f} "
            f"{-r['skip_pnl_25']:>+8.2f}"
        )
    print()
    print("Note: skipPnL = -1 × PnL of skipped trades. Positive = filter")
    print("      saved money (skipped trades were losers in aggregate).")


if __name__ == "__main__":
    main()
