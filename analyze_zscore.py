#!/usr/bin/env python3
"""
Z-score predictive power analysis for a MARKET MAKER (zero fees, fills at bid).

As a maker:
  - Buying UP: you post at bid_up, get filled when someone sells to you
  - Edge = p(UP) - bid_up  (no fee)
  - Adverse selection risk: fills happen when counterparty has information

Questions answered:
  1. How closely do p_model and mid_up track?
  2. Brier scores: model vs market mid vs market bid
  3. Edge at bid price across tau checkpoints
  4. When model diverges from market, who is right?
  5. Adverse selection: is the fill-weighted outcome worse than unconditional?
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
VOL_LOOKBACK_S = 90
MAX_Z = 1.5
TAU_CHECKPOINTS = [750, 600, 450, 300, 150, 60]


def norm_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def compute_sigma(prices, timestamps):
    changes = []
    for i, p in enumerate(prices):
        if p > 0 and (not changes or p != changes[-1][1]):
            changes.append((timestamps[i], p))
    if len(changes) < 3:
        return 0.0
    log_rets = []
    for j in range(1, len(changes)):
        dt = (changes[j][0] - changes[j-1][0]) / 1000.0
        if dt > 0:
            lr = math.log(changes[j][1] / changes[j-1][1])
            log_rets.append(lr / math.sqrt(dt))
    if len(log_rets) < 2:
        return 0.0
    return float(np.std(log_rets, ddof=1))


def process_directory(subdir: Path) -> list[dict]:
    rows = []
    for f in sorted(subdir.glob("*.parquet")):
        df = pd.read_parquet(f)
        if df.empty:
            continue

        if "chainlink_price" in df.columns:
            price_col = "chainlink_price"
        elif "chainlink_btc" in df.columns:
            price_col = "chainlink_btc"
        else:
            continue

        start_px_series = df["window_start_price"].dropna()
        if start_px_series.empty:
            continue
        start_px = float(start_px_series.iloc[0])
        if start_px == 0:
            continue

        final_px = float(df[price_col].iloc[-1])
        outcome_up = 1 if final_px >= start_px else 0

        prices  = df[price_col].tolist()
        ts_list = df["ts_ms"].tolist()
        tau_all = df["time_remaining_s"].tolist()

        for target_tau in TAU_CHECKPOINTS:
            best_idx = min(range(len(tau_all)),
                           key=lambda i: abs(tau_all[i] - target_tau))
            actual_tau = tau_all[best_idx]
            if abs(actual_tau - target_tau) > 30:
                continue

            lo = max(0, best_idx - VOL_LOOKBACK_S)
            sigma = compute_sigma(prices[lo:best_idx+1], ts_list[lo:best_idx+1])
            if sigma <= 0:
                continue

            current_px = prices[best_idx]
            delta = (current_px - start_px) / start_px
            z_raw = delta / (sigma * math.sqrt(actual_tau))
            z = max(-MAX_Z, min(MAX_Z, z_raw))
            p_model = norm_cdf(z)

            row = df.iloc[best_idx]
            bid_up   = row.get("best_bid_up")
            ask_up   = row.get("best_ask_up")
            bid_down = row.get("best_bid_down")
            ask_down = row.get("best_ask_down")
            mid_up   = row.get("mid_up")

            for val in [bid_up, ask_up, bid_down, ask_down, mid_up]:
                if pd.isna(val):
                    break
            else:
                bid_up, ask_up = float(bid_up), float(ask_up)
                bid_down, ask_down = float(bid_down), float(ask_down)
                mid_up = float(mid_up)

                if not (0 < bid_up < ask_up < 1 and 0 < bid_down < ask_down < 1):
                    continue

                spread_up   = ask_up - bid_up
                spread_down = ask_down - bid_down

                rows.append({
                    "tau_target":  target_tau,
                    "z":           z,
                    "p_model":     p_model,
                    "bid_up":      bid_up,
                    "ask_up":      ask_up,
                    "mid_up":      mid_up,
                    "bid_down":    bid_down,
                    "ask_down":    ask_down,
                    "spread_up":   spread_up,
                    "spread_down": spread_down,
                    # Maker edge (zero fees): fill at bid
                    "edge_maker_up":   outcome_up - bid_up,
                    "edge_maker_down": (1 - outcome_up) - bid_down,
                    "divergence":  p_model - mid_up,
                    "outcome_up":  outcome_up,
                })
    return rows


def brier(probs, outcomes):
    return float(np.mean([(p - o)**2 for p, o in zip(probs, outcomes)]))


def main():
    DIRS = {
        "btc_15m": 900, "eth_15m": 900, "sol_15m": 900, "xrp_15m": 900,
        "btc_5m":  300, "eth_5m":  300, "sol_5m":  300, "xrp_5m":  300,
    }

    all_rows = []
    for name in DIRS:
        d = DATA_DIR / name
        if not d.exists():
            continue
        r = process_directory(d)
        for row in r:
            row["asset"] = name
        all_rows.extend(r)
        print(f"  {name}: {len(r)} observations")

    df = pd.DataFrame(all_rows)
    print(f"\nTotal: {len(df)} observations\n")

    # ── 1. Tracking ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("1. HOW CLOSELY DO p_model AND mid_up TRACK?")
    print("=" * 70)
    corr = df["p_model"].corr(df["mid_up"])
    mae  = (df["p_model"] - df["mid_up"]).abs().mean()
    print(f"  Correlation(p_model, mid_up): {corr:.4f}")
    print(f"  Mean absolute difference:     {mae:.4f}")
    print(f"  Mean divergence (model-mid):  {df['divergence'].mean():+.4f}")

    # ── 2. Brier scores ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("2. BRIER SCORES — ZERO FEES (lower = better, 0.25 = random)")
    print("=" * 70)
    bs_model  = brier(df["p_model"], df["outcome_up"])
    bs_mid    = brier(df["mid_up"],  df["outcome_up"])
    bs_bid    = brier(df["bid_up"],  df["outcome_up"])
    bs_naive  = brier([0.5]*len(df), df["outcome_up"])
    print(f"  p_model Brier:   {bs_model:.5f}")
    print(f"  mid_up  Brier:   {bs_mid:.5f}")
    print(f"  bid_up  Brier:   {bs_bid:.5f}  ← your actual fill price as maker")
    print(f"  naive   Brier:   {bs_naive:.5f}")
    print(f"  Model vs market: {bs_mid - bs_model:+.5f}  (+ = model better)")

    # ── 3. Maker edge by tau ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("3. MAKER EDGE BY TAU (fill at bid, zero fees)")
    print("=" * 70)
    print(f"  {'tau':>6}  {'n':>5}  {'avg bid_up':>10}  {'UP win%':>8}  "
          f"{'edge_UP':>8}  {'edge_DOWN':>10}  {'spread':>8}")
    for tau in TAU_CHECKPOINTS:
        g = df[df["tau_target"] == tau]
        if len(g) < 10:
            continue
        print(f"  {tau:>6}s  {len(g):>5}  {g['bid_up'].mean():>10.4f}  "
              f"{g['outcome_up'].mean():>8.3f}  "
              f"{g['edge_maker_up'].mean():>+8.4f}  "
              f"{g['edge_maker_down'].mean():>+10.4f}  "
              f"{g['spread_up'].mean():>8.4f}")

    # ── 4. Divergence analysis ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("4. WHEN MODEL DIVERGES FROM MARKET — WHO IS RIGHT?")
    print("   (divergence = p_model - mid_up)")
    print("=" * 70)
    df["div_bin"] = pd.qcut(df["divergence"], q=5, labels=False)
    labels = ["very negative", "negative", "neutral", "positive", "very positive"]
    print(f"  {'bin':20}  {'mean div':>9}  {'n':>5}  "
          f"{'actual UP%':>10}  {'p_model':>8}  {'mid_up':>8}  {'edge@bid':>9}")
    for b in range(5):
        g = df[df["div_bin"] == b]
        print(f"  {labels[b]:20}  {g['divergence'].mean():>+9.4f}  {len(g):>5}  "
              f"{g['outcome_up'].mean():>10.3f}  {g['p_model'].mean():>8.4f}  "
              f"{g['mid_up'].mean():>8.4f}  {g['edge_maker_up'].mean():>+9.4f}")

    # ── 5. Adverse selection ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("5. ADVERSE SELECTION — DOES THE MARKET MOVE AGAINST YOU AFTER FILL?")
    print("   (proxy: last_trade_side_up in the data)")
    print("=" * 70)

    # Check if trade side data exists
    sample_f = sorted((DATA_DIR / "btc_15m").glob("*.parquet"))[0]
    sample_df = pd.read_parquet(sample_f)
    if "last_trade_side_up" in sample_df.columns:
        # Trades hitting the bid = someone SELLING to you (maker buy)
        # If last_trade_side = 'sell' or equivalent, you'd have been filled
        print(f"  Trade side column present: last_trade_side_up")
        print(f"  Unique values: {sample_df['last_trade_side_up'].dropna().unique()[:5]}")
    else:
        print("  No trade side data in parquet — cannot measure adverse selection directly.")

    # ── 6. Spread capture potential ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("6. SPREAD CAPTURE POTENTIAL (quoting both sides)")
    print("=" * 70)
    df["half_spread"] = df["spread_up"] / 2
    # If you quote both sides and get filled on both: you earn the spread
    # minus the directional P&L risk
    print(f"  Avg spread_up:              {df['spread_up'].mean():.4f}")
    print(f"  Avg spread_down:            {df['spread_down'].mean():.4f}")
    print(f"  Half-spread (what you earn per side): {df['half_spread'].mean():.4f}")
    print(f"  Avg maker edge UP  (bid):   {df['edge_maker_up'].mean():+.4f}")
    print(f"  Avg maker edge DOWN (bid):  {df['edge_maker_down'].mean():+.4f}")
    print(f"  Combined both sides:        {(df['edge_maker_up'] + df['edge_maker_down']).mean():+.4f}")
    print(f"\n  Note: combined edge = spread - 1 + bid_up + bid_down")
    print(f"  (positive only if bid_up + bid_down < 1, i.e. you post inside the book)")


if __name__ == "__main__":
    main()
