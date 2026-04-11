#!/usr/bin/env python3
"""Tick-by-tick signal debugger.

Replays a single recorded window through `decide_both_sides()` and dumps
every intermediate variable per tick so you can see exactly where sigma,
z, or p_model goes wrong. Flags rows where sigma is pinned at min_sigma
(the root of the 2026-04-09 and 2026-04-11 overconfidence bugs).

Usage
-----
    # Debug the most recent btc_5m window:
    python analysis/tick_debugger.py --market btc_5m --latest

    # Debug a specific window by slug:
    python analysis/tick_debugger.py --slug btc-updown-5m-1775936100

    # Show only the last N ticks (end-of-window):
    python analysis/tick_debugger.py --slug ... --tail 20

    # Zoom to a specific tau range (seconds remaining):
    python analysis/tick_debugger.py --slug ... --tau-min 250 --tau-max 290

    # Compare to live log for the same window (if available):
    python analysis/tick_debugger.py --slug ... --with-live

    # Toggle calibration table on/off (default: on, matching live):
    python analysis/tick_debugger.py --slug ... --no-cal

Output is a columnar table. Any row with `sigma == min_sigma` is flagged
with a red "*FLOOR*" marker — this is the signature of the sigma-collapse
bug that pins p_model at the norm_cdf cap.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path("/Users/dannychee/Desktop/prediction-market-bot")
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from backtest import DATA_DIR, build_diffusion_signal  # noqa: E402
from backtest_core import Snapshot, build_calibration_table, norm_cdf  # noqa: E402
from market_config import get_config  # noqa: E402

# ANSI colour codes (used when --color is set and stdout is a tty)
RED = "\033[91m"
YEL = "\033[93m"
GRN = "\033[92m"
DIM = "\033[2m"
OFF = "\033[0m"


def parquet_path_for_slug(slug: str) -> Path | None:
    """Resolve a slug to its parquet file by prefix convention."""
    if slug.startswith("btc-updown-5m"):
        return DATA_DIR / "btc_5m" / f"{slug}.parquet"
    if slug.startswith("btc-updown-15m"):
        return DATA_DIR / "btc_15m" / f"{slug}.parquet"
    if slug.startswith("bitcoin-up-or-down"):
        for sub in ("btc_1h", "btc_1h_real"):
            p = DATA_DIR / sub / f"{slug}.parquet"
            if p.exists():
                return p
    if slug.startswith("eth-updown-5m"):
        return DATA_DIR / "eth_5m" / f"{slug}.parquet"
    if slug.startswith("eth-updown-15m"):
        return DATA_DIR / "eth_15m" / f"{slug}.parquet"
    return None


def market_key_for_slug(slug: str) -> str | None:
    if slug.startswith("btc-updown-5m"):
        return "btc_5m"
    if slug.startswith("btc-updown-15m"):
        return "btc"
    if slug.startswith("bitcoin-up-or-down"):
        return "btc_1h"
    if slug.startswith("eth-updown-5m"):
        return "eth_5m"
    if slug.startswith("eth-updown-15m"):
        return "eth"
    return None


def latest_slug(market_key: str) -> str | None:
    """Return the most recent slug for a market_key based on mtime."""
    cfg = get_config(market_key)
    data_dir = DATA_DIR / cfg.data_subdir
    if not data_dir.exists():
        return None
    parquets = sorted(data_dir.glob("*.parquet"), key=lambda p: p.stat().st_mtime)
    if not parquets:
        return None
    return parquets[-1].stem


def build_live_snapshot_index(slug: str) -> dict[int, dict]:
    """If live logs contain diagnostic rows for this slug, index them by ts_ms.

    Returns a dict mapping approximate second-aligned timestamps to the live
    diagnostic record so the debugger can print side-by-side live_p vs
    replay_p.
    """
    index: dict[int, dict] = {}
    for name in ("live_trades_btc.jsonl", "live_trades_btc_1h.jsonl"):
        fp = ROOT / name
        if not fp.exists():
            continue
        try:
            with open(fp) as fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    if d.get("market_slug") != slug:
                        continue
                    if d.get("type") not in ("diagnostic", "limit_order", "taker_fill"):
                        continue
                    # Parse ts to an integer second
                    ts_iso = d.get("ts")
                    if not ts_iso:
                        continue
                    try:
                        # Best-effort conversion to epoch seconds
                        from datetime import datetime
                        ts_s = int(datetime.fromisoformat(ts_iso.replace("Z", "+00:00")).timestamp())
                    except Exception:
                        continue
                    index[ts_s] = d
        except Exception:
            pass
    return index


def fmt_sigma(s: float | None) -> str:
    if s is None:
        return "  ----  "
    return f"{s:.2e}"


def fmt_price(p: float | None) -> str:
    if p is None:
        return "  -----  "
    return f"{p:>9,.2f}"


def fmt_p(p: float | None) -> str:
    if p is None:
        return " --- "
    return f"{p:.3f}"


def fmt_delta(d: float | None) -> str:
    if d is None:
        return "    ----"
    return f"{d:+8.2f}"


def run_debugger(
    slug: str,
    *,
    market_key: str,
    tail: int | None = None,
    tau_min: float | None = None,
    tau_max: float | None = None,
    with_live: bool = False,
    use_calibration: bool = True,
    color: bool = True,
    show_book: bool = True,
) -> None:
    """Run the tick debugger for one window."""
    fp = parquet_path_for_slug(slug)
    if fp is None or not fp.exists():
        print(f"ERROR: parquet not found for slug {slug!r}", file=sys.stderr)
        print(f"  tried: {fp}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(fp)
    if len(df) == 0:
        print(f"ERROR: parquet is empty: {fp}", file=sys.stderr)
        sys.exit(1)

    # Build signal per-market, matching live's configuration exactly.
    sig = build_diffusion_signal(market_key, bankroll=100.0, maker=True)
    cfg = get_config(market_key)

    if use_calibration:
        try:
            cal_dir = DATA_DIR / cfg.data_subdir
            cal = build_calibration_table(cal_dir, vol_lookback_s=90)
            sig.calibration_table = cal
            sig.cal_prior_strength = 50.0
            sig.cal_max_weight = 0.70
        except Exception as exc:
            print(f"  [warn] calibration table failed: {exc}", file=sys.stderr)
            sig.calibration_table = None
    else:
        sig.calibration_table = None

    # Pull config numbers we'll reference
    min_sigma = cfg.min_sigma
    max_sigma = cfg.max_sigma
    max_z = cfg.max_z
    vol_lookback_s = getattr(sig, "vol_lookback_s", 90)
    market_blend = getattr(sig, "market_blend", 0.0) or 0.0

    has_binance = "binance_mid" in df.columns
    ctx: dict = {"inventory_up": 0, "inventory_down": 0}
    if "window_start_ms" in df.columns:
        ctx["_window_start_ms"] = int(df["window_start_ms"].iloc[0])

    live_index = build_live_snapshot_index(slug) if with_live else {}

    # ── Header ──────────────────────────────────────────────────────────────
    win_duration_s = cfg.window_duration_s
    window_start_ms = int(df["window_start_ms"].iloc[0]) if "window_start_ms" in df.columns else None
    print(f"\n{'=' * 110}")
    print(f"  TICK DEBUGGER  |  slug: {slug}")
    print(f"  market_key: {market_key}   window_duration: {win_duration_s:.0f}s   "
          f"rows: {len(df)}   parquet: {fp.name}")
    print(f"  cfg: min_sigma={min_sigma:.2e}  max_sigma={max_sigma:.2e}  "
          f"max_z={max_z:.1f}  vol_lookback={vol_lookback_s}s  "
          f"market_blend={market_blend}  calibration={'on' if use_calibration else 'off'}")
    print("=" * 110)
    hdr = (
        f"  {'tau':>4} | "
        f"{'cl_px':>9} {'bn_px':>9} | "
        f"{'Δcl':>7} {'Δbn':>7} | "
        f"{'hist':>4} | "
        f"{'raw_σ':>9} {'kalm_x':>9} {'σ_used':>9} | "
        f"{'z_raw':>6} {'z_cap':>6} | "
        f"{'p_gbm':>5} {'p_mdl':>5} | "
        f"{'edgeU':>6} {'edgeD':>6} | "
        f"{'action':<9}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    # ── Replay ──────────────────────────────────────────────────────────────
    rows = list(df.iterrows())
    if tau_min is not None or tau_max is not None:
        rows = [
            (i, r) for (i, r) in rows
            if (tau_min is None or r["time_remaining_s"] >= tau_min)
            and (tau_max is None or r["time_remaining_s"] <= tau_max)
        ]
    if tail is not None:
        rows = rows[-tail:]

    floor_count = 0
    total_ticks = 0

    for _, row in rows:
        snap = Snapshot.from_row(row)
        if snap is None:
            continue
        if has_binance and pd.notna(row.get("binance_mid")) and row["binance_mid"] > 0:
            ctx["_binance_mid"] = float(row["binance_mid"])

        try:
            up_dec, down_dec = sig.decide_both_sides(snap, ctx)
        except Exception as exc:
            print(f"  [ERROR at ts_ms={snap.ts_ms}] {exc}")
            continue

        total_ticks += 1

        # Pull intermediate state from ctx (populated by decide_both_sides)
        tau = snap.time_remaining_s
        cl_px = snap.chainlink_price
        bn_px = ctx.get("_binance_mid")
        start_px = snap.window_start_price
        delta_cl = cl_px - start_px if cl_px and start_px else None
        delta_bn = bn_px - start_px if bn_px and start_px else None
        hist_len = len(ctx.get("price_history", []))

        # Sigma pipeline snapshots
        # The signal stores _sigma_per_s AFTER all floor/cap application.
        sigma_used = ctx.get("_sigma_per_s")
        kalman_x = ctx.get("_kalman_x")
        # raw_sigma isn't directly stored, so recompute from history for visibility
        hist = ctx.get("price_history", [])
        ts_hist = ctx.get("ts_history", [])
        try:
            raw_sigma = sig._compute_vol(
                hist[-vol_lookback_s:], ts_hist[-vol_lookback_s:]
            ) if len(hist) >= 2 else 0.0
        except Exception:
            raw_sigma = None

        # z + p_model
        p_raw = ctx.get("_p_model_raw")
        # Reconstruct z from delta_bn (effective price used by signal)
        eff_px = bn_px if bn_px else cl_px
        z_raw = z_cap = None
        p_gbm = None
        if (eff_px is not None and start_px and sigma_used
                and sigma_used > 0 and tau > 0):
            try:
                d_log = math.log(eff_px / start_px) if eff_px > 0 else 0.0
                z_raw = d_log / (sigma_used * math.sqrt(tau))
                z_cap = max(-max_z, min(max_z, z_raw))
                p_gbm = norm_cdf(z_cap)
            except Exception:
                pass

        edge_up = up_dec.edge if up_dec else None
        edge_down = down_dec.edge if down_dec else None
        action_up = up_dec.action if up_dec else "-"
        action_down = down_dec.action if down_dec else "-"
        action = (
            f"{'U' if action_up == 'BUY_UP' else ' '}{'D' if action_down == 'BUY_DOWN' else ' '} "
            f"{up_dec.reason[:6] if up_dec and up_dec.reason else '':6s}"
        )

        # Floor flag
        flag = ""
        is_floored = (
            sigma_used is not None
            and abs(sigma_used - min_sigma) / min_sigma < 0.01
        )
        if is_floored:
            floor_count += 1
            if color:
                flag = f" {RED}*FLOOR*{OFF}"
            else:
                flag = " *FLOOR*"

        # Format row
        row_str = (
            f"  {tau:>4.0f} | "
            f"{fmt_price(cl_px)} {fmt_price(bn_px)} | "
            f"{fmt_delta(delta_cl)} {fmt_delta(delta_bn)} | "
            f"{hist_len:>4} | "
            f"{fmt_sigma(raw_sigma)} {fmt_sigma(kalman_x)} {fmt_sigma(sigma_used)} | "
            f"{(f'{z_raw:+6.2f}' if z_raw is not None else ' ---- '):>6} "
            f"{(f'{z_cap:+6.2f}' if z_cap is not None else ' ---- '):>6} | "
            f"{fmt_p(p_gbm):>5} {fmt_p(p_raw):>5} | "
            f"{(f'{edge_up:+6.3f}' if edge_up is not None else '  ---- '):>6} "
            f"{(f'{edge_down:+6.3f}' if edge_down is not None else '  ---- '):>6} | "
            f"{action:<9}"
        )
        print(row_str + flag)

        if show_book and (action_up == "BUY_UP" or action_down == "BUY_DOWN"):
            # Show book state on decision ticks
            print(f"       book: UP {snap.best_bid_up}/{snap.best_ask_up}  "
                  f"DOWN {snap.best_bid_down}/{snap.best_ask_down}")

    # ── Footer summary ──────────────────────────────────────────────────────
    print("  " + "-" * (len(hdr) - 2))
    if total_ticks > 0:
        pct = 100 * floor_count / total_ticks
        flag_color = RED if pct > 10 else (YEL if pct > 0 else GRN)
        reset = OFF if color else ""
        pct_str = f"{flag_color}{floor_count}/{total_ticks} ({pct:.1f}%){reset}"
        print(f"  sigma at min_sigma floor: {pct_str}")
    print("=" * 110)

    if live_index:
        print("\n  Live log cross-reference (if any):")
        for ts_s in sorted(live_index):
            d = live_index[ts_s]
            t = d.get("type")
            p_live = d.get("p_model") or d.get("p_side")
            s_live = d.get("sigma_per_s")
            print(f"    ts={ts_s}  type={t}  p_model={p_live}  sigma={s_live}")


def main():
    ap = argparse.ArgumentParser(
        description="Tick-by-tick signal debugger (dumps every intermediate variable)"
    )
    ap.add_argument("--slug", help="Window slug, e.g. btc-updown-5m-1775936100")
    ap.add_argument("--market", default="btc_5m",
                    help="Market key (btc, btc_5m, eth, etc.). Used with --latest.")
    ap.add_argument("--latest", action="store_true",
                    help="Debug the most recent parquet for --market.")
    ap.add_argument("--tail", type=int, default=None,
                    help="Only show the last N ticks")
    ap.add_argument("--tau-min", type=float, default=None,
                    help="Only show ticks with tau >= this value")
    ap.add_argument("--tau-max", type=float, default=None,
                    help="Only show ticks with tau <= this value")
    ap.add_argument("--with-live", action="store_true",
                    help="Cross-reference against live log entries for this slug")
    ap.add_argument("--no-cal", action="store_true",
                    help="Disable calibration table (default: on, matching live)")
    ap.add_argument("--no-color", action="store_true",
                    help="Disable ANSI colour output")
    args = ap.parse_args()

    if args.latest:
        slug = latest_slug(args.market)
        if slug is None:
            print(f"ERROR: no parquets found for {args.market}", file=sys.stderr)
            sys.exit(1)
        market_key = args.market
    elif args.slug:
        slug = args.slug
        market_key = market_key_for_slug(slug)
        if market_key is None:
            print(f"ERROR: cannot infer market_key from slug {slug!r}", file=sys.stderr)
            sys.exit(1)
    else:
        ap.error("must pass --slug or --latest")

    run_debugger(
        slug,
        market_key=market_key,
        tail=args.tail,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        with_live=args.with_live,
        use_calibration=not args.no_cal,
        color=not args.no_color,
    )


if __name__ == "__main__":
    main()
