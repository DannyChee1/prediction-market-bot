#!/usr/bin/env python3
"""
3e — Stratified calibration analysis (Quant Guild #93 — non-stationarity).

Question:  The global calibration analysis (analysis/calibration_analysis.py)
collapses every observation across hours, days, and vol regimes into
a single reliability diagram. That hides regime-conditional miscalibration.
Specifically, the pre-fix model had a -7% downward bias on weekends (low
σ regime) that was invisible in the global ECE because weekend trades
were diluted by weekday data.

This script reproduces the global ECE per stratum and reports:
  * is_weekend: weekday vs weekend
  * vol_tercile: low / mid / high realised σ
  * hour_bucket: 0-5, 6-11, 12-17, 18-23 UTC

For each (market × stratification), it prints the per-bucket Brier,
ECE, and bin counts. Materially different ECE across buckets means the
single-table calibration is averaging over heterogeneous regimes.

Reads:     parquet windows in data/<market_subdir>/
Writes:    validation_runs/stratified_calibration/<market>.json
           validation_runs/stratified_calibration/<market>.png
Prints:    a stratified ECE table per market

Usage:
    python scripts/validate_stratified_calibration.py
    python scripts/validate_stratified_calibration.py --markets btc btc_5m
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backtest import (  # noqa: E402
    _compute_vol_deduped,
    norm_cdf,
    fast_t_cdf,
    MIN_FINAL_REMAINING_S,
    MAX_START_GAP_S,
)
from market_config import MARKET_CONFIGS  # noqa: E402

OUT_DIR = REPO_ROOT / "validation_runs" / "stratified_calibration"
DATA_DIR = REPO_ROOT / "data"


# ── CDF dispatch (mirrors live model post-fix) ───────────────────────────────


def model_cdf(z: float, cfg) -> float:
    """Mirror DiffusionSignal._model_cdf POST-FIX.

    Critically, this uses `norm_cdf(z)` for tail_mode="kou" — the bug
    fix removed the buggy drift correction. Pre-fix versions of this
    function would have shown a 7% bias on the weekend stratum.
    """
    if cfg.tail_mode == "normal":
        return norm_cdf(z)
    if cfg.tail_mode == "kou":
        return norm_cdf(z)  # post-fix: no drift
    if cfg.tail_mode == "student_t":
        return fast_t_cdf(z, cfg.tail_nu_default)
    return norm_cdf(z)


# ── Stratification helpers ───────────────────────────────────────────────────


def _hour_bucket(ts_ms: int) -> str:
    import datetime as _dt
    h = _dt.datetime.fromtimestamp(ts_ms / 1000, tz=_dt.timezone.utc).hour
    if h < 6:
        return "00-05_UTC"
    elif h < 12:
        return "06-11_UTC"
    elif h < 18:
        return "12-17_UTC"
    else:
        return "18-23_UTC"


def _is_weekend(ts_ms: int) -> str:
    import datetime as _dt
    wd = _dt.datetime.fromtimestamp(ts_ms / 1000, tz=_dt.timezone.utc).weekday()
    return "weekend" if wd >= 5 else "weekday"


# ── Window iteration ─────────────────────────────────────────────────────────


def collect_observations(market_key: str, vol_lookback_s: int = 90,
                         max_z: float = 1.0, sample_every: int = 30):
    """Yield (z, p_model, outcome, sigma, tau, hour_bucket, is_weekend) records."""
    cfg = MARKET_CONFIGS[market_key]
    data_dir = DATA_DIR / cfg.data_subdir
    if not data_dir.exists():
        return [], []

    sigmas_all: list[float] = []
    records: list[tuple] = []
    files = sorted(data_dir.glob("*.parquet"))

    for f in files:
        df = pd.read_parquet(f)
        if df.empty:
            continue
        if "chainlink_btc" in df.columns and "chainlink_price" not in df.columns:
            df.rename(columns={"chainlink_btc": "chainlink_price"}, inplace=True)
        if "chainlink_price" not in df.columns:
            continue
        if "window_end_ms" in df.columns:
            if df["ts_ms"].iloc[-1] < df["window_end_ms"].iloc[0]:
                continue
        else:
            if df["time_remaining_s"].iloc[-1] > MIN_FINAL_REMAINING_S:
                continue
        if ("window_start_ms" in df.columns and "window_end_ms" in df.columns
                and "time_remaining_s" in df.columns):
            window_dur_s = (df["window_end_ms"].iloc[0]
                            - df["window_start_ms"].iloc[0]) / 1000
            if df["time_remaining_s"].iloc[0] < window_dur_s - MAX_START_GAP_S:
                continue
        sp = df["window_start_price"].dropna()
        if sp.empty:
            continue
        start_px = float(sp.iloc[0])
        final_px = float(df["chainlink_price"].iloc[-1])
        if pd.isna(start_px) or pd.isna(final_px) or start_px == 0:
            continue
        outcome = 1 if final_px >= start_px else 0

        prices = df["chainlink_price"].tolist()
        ts_list = df["ts_ms"].tolist()

        for idx in range(vol_lookback_s, len(df), sample_every):
            row = df.iloc[idx]
            tau = float(row["time_remaining_s"])
            if tau <= 0:
                continue
            lo = max(0, idx - vol_lookback_s)
            sigma = _compute_vol_deduped(prices[lo:idx + 1],
                                         ts_list[lo:idx + 1])
            if sigma <= 0:
                continue
            cur = float(row["chainlink_price"])
            delta = (cur - start_px) / start_px
            z_raw = delta / (sigma * math.sqrt(tau))
            z_capped = max(-max_z, min(max_z, z_raw))
            p_model = model_cdf(z_capped, cfg)
            ts_ms = int(row["ts_ms"])
            sigmas_all.append(sigma)
            records.append((p_model, outcome, sigma,
                            _hour_bucket(ts_ms), _is_weekend(ts_ms)))

    return records, sigmas_all


# ── Calibration metrics ──────────────────────────────────────────────────────


def reliability_metrics(records: list[tuple], n_bins: int = 10) -> dict:
    if not records:
        return {"n": 0, "brier": float("nan"), "ece": float("nan"),
                "logloss": float("nan"), "p_up_actual": float("nan"),
                "bins": []}
    p = np.array([r[0] for r in records], dtype=np.float64)
    y = np.array([r[1] for r in records], dtype=np.int64)
    n = len(p)
    brier = float(((p - y) ** 2).mean())
    eps = 1e-9
    pc = np.clip(p, eps, 1 - eps)
    logloss = -float((y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean())
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins_out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (p >= lo) & (p < hi)
        else:
            mask = (p >= lo) & (p <= hi)
        cnt = int(mask.sum())
        if cnt > 0:
            ap = float(p[mask].mean())
            aa = float(y[mask].mean())
            ece += (cnt / n) * abs(aa - ap)
            bins_out.append({"lo": lo, "hi": hi, "n": cnt,
                             "avg_pred": ap, "avg_actual": aa})
        else:
            bins_out.append({"lo": lo, "hi": hi, "n": 0,
                             "avg_pred": None, "avg_actual": None})
    return {"n": n, "brier": brier, "ece": ece, "logloss": logloss,
            "p_up_actual": float(y.mean()), "bins": bins_out}


# ── Stratification driver ───────────────────────────────────────────────────


def stratify(records: list[tuple], sigmas: list[float]) -> dict:
    """Bin records by hour bucket, weekend flag, and σ tercile."""
    sigmas_arr = np.array(sigmas)
    if len(sigmas_arr) == 0:
        return {}
    t1 = float(np.percentile(sigmas_arr, 33.33))
    t2 = float(np.percentile(sigmas_arr, 66.67))

    def vol_bucket(s):
        if s <= t1:
            return "low_vol"
        if s <= t2:
            return "mid_vol"
        return "high_vol"

    out: dict[str, dict] = {
        "global": reliability_metrics(records),
        "vol_terciles": {},
        "weekend": {},
        "hour_bucket": {},
        "vol_thresholds": {"t33": t1, "t67": t2},
    }

    by_vol = defaultdict(list)
    by_wknd = defaultdict(list)
    by_hour = defaultdict(list)
    for rec in records:
        by_vol[vol_bucket(rec[2])].append(rec)
        by_wknd[rec[4]].append(rec)
        by_hour[rec[3]].append(rec)

    for k in ("low_vol", "mid_vol", "high_vol"):
        out["vol_terciles"][k] = reliability_metrics(by_vol.get(k, []))
    for k in ("weekday", "weekend"):
        out["weekend"][k] = reliability_metrics(by_wknd.get(k, []))
    for k in ("00-05_UTC", "06-11_UTC", "12-17_UTC", "18-23_UTC"):
        out["hour_bucket"][k] = reliability_metrics(by_hour.get(k, []))
    return out


# ── Plot ─────────────────────────────────────────────────────────────────────


def plot_stratification(strat: dict, market: str, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def plot_one(ax, name, series, color):
        bins = [b for b in series["bins"] if b["avg_pred"] is not None
                and b["avg_actual"] is not None and b["n"] >= 5]
        if not bins:
            return
        ax.plot([b["avg_pred"] for b in bins],
                [b["avg_actual"] for b in bins],
                "o-", color=color, label=f"{name} (ECE={series['ece']:.4f})",
                linewidth=2, markersize=5)

    # Weekend
    plot_one(axes[0], "weekday", strat["weekend"]["weekday"], "tab:blue")
    plot_one(axes[0], "weekend", strat["weekend"]["weekend"], "tab:orange")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Predicted P(UP)")
    axes[0].set_ylabel("Actual UP rate")
    axes[0].set_title("By weekday/weekend")
    axes[0].legend(fontsize=9, loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # Vol tercile
    for name, color in (("low_vol", "tab:green"),
                        ("mid_vol", "tab:blue"),
                        ("high_vol", "tab:red")):
        plot_one(axes[1], name, strat["vol_terciles"][name], color)
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Predicted P(UP)")
    axes[1].set_ylabel("Actual UP rate")
    axes[1].set_title(f"By σ tercile (t33={strat['vol_thresholds']['t33']:.1e}, "
                      f"t67={strat['vol_thresholds']['t67']:.1e})")
    axes[1].legend(fontsize=9, loc="upper left")
    axes[1].grid(True, alpha=0.3)

    # Hour
    for name, color in (("00-05_UTC", "tab:green"),
                        ("06-11_UTC", "tab:blue"),
                        ("12-17_UTC", "tab:orange"),
                        ("18-23_UTC", "tab:red")):
        plot_one(axes[2], name, strat["hour_bucket"][name], color)
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[2].set_xlim(0, 1); axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Predicted P(UP)")
    axes[2].set_ylabel("Actual UP rate")
    axes[2].set_title("By hour-of-day bucket (UTC)")
    axes[2].legend(fontsize=9, loc="upper left")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"{market} — stratified calibration (POST-FIX)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Print helper ─────────────────────────────────────────────────────────────


def print_stratum_table(name: str, strat: dict):
    print(f"\n  {name}:")
    print(f"  {'bucket':<14}  {'n':>8}  {'p_up%':>8}  "
          f"{'Brier':>8}  {'ECE':>8}  {'LogLoss':>8}")
    for k, v in strat.items():
        if v["n"] == 0:
            print(f"  {k:<14}  {v['n']:>8}  {'—':>8}  {'—':>8}  "
                  f"{'—':>8}  {'—':>8}")
        else:
            print(f"  {k:<14}  {v['n']:>8}  "
                  f"{v['p_up_actual']*100:>7.1f}%  "
                  f"{v['brier']:>8.4f}  {v['ece']:>8.4f}  "
                  f"{v['logloss']:>8.4f}")


def main():
    ap = argparse.ArgumentParser(
        description="Stratified calibration analysis (post-fix)")
    ap.add_argument("--markets", nargs="*", default=["btc", "btc_5m"],
                    help="Market keys to analyse (default: btc btc_5m). "
                         "Other markets are stratified the same way but "
                         "they don't use the kou tail mode so they're "
                         "less interesting for this regression check.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}
    for market in args.markets:
        if market not in MARKET_CONFIGS:
            print(f"  WARNING: unknown market {market}, skipping")
            continue
        cfg = MARKET_CONFIGS[market]
        print(f"\nProcessing {cfg.display_name} ({cfg.tail_mode})...")
        records, sigmas = collect_observations(market)
        print(f"  Loaded {len(records)} observations")
        if not records:
            print("  No observations, skipping")
            continue

        strat = stratify(records, sigmas)
        print_stratum_table("Weekday/Weekend", strat["weekend"])
        print_stratum_table("Vol terciles", strat["vol_terciles"])
        print_stratum_table("Hour bucket (UTC)", strat["hour_bucket"])

        # Persist
        out_json = OUT_DIR / f"{market}.json"
        with open(out_json, "w") as f:
            json.dump(strat, f, indent=2, default=str)
        print(f"\n  → wrote {out_json}")

        out_png = OUT_DIR / f"{market}.png"
        plot_stratification(strat, cfg.display_name, out_png)
        print(f"  → wrote {out_png}")

        summary[market] = {
            "global_ece": strat["global"]["ece"],
            "weekday_ece": strat["weekend"]["weekday"]["ece"],
            "weekend_ece": strat["weekend"]["weekend"]["ece"],
            "low_vol_ece": strat["vol_terciles"]["low_vol"]["ece"],
            "high_vol_ece": strat["vol_terciles"]["high_vol"]["ece"],
        }

    print("\n" + "=" * 64)
    print("  SUMMARY: ECE BY STRATUM")
    print("=" * 64)
    print(f"  {'market':<10}  {'global':>8}  {'weekday':>8}  "
          f"{'weekend':>8}  {'low σ':>8}  {'high σ':>8}")
    for m, s in summary.items():
        print(f"  {m:<10}  {s['global_ece']:>8.4f}  "
              f"{s['weekday_ece']:>8.4f}  {s['weekend_ece']:>8.4f}  "
              f"{s['low_vol_ece']:>8.4f}  {s['high_vol_ece']:>8.4f}")
    print()
    print("Lower ECE = better calibrated. If weekday and weekend are within")
    print("each other's bin-noise (~ ±0.005 for these sample sizes), the bug")
    print("fix has eliminated the regime-conditional bias the old kou drift")
    print("was injecting on weekends/quiet hours.")


if __name__ == "__main__":
    main()
