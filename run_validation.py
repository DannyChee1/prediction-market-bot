#!/usr/bin/env python3
"""Model Validation & Out-of-Sample Testing Script."""

import math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("data")
MARKETS = {
    "btc_15m": {"subdir": "btc_15m", "window_s": 900, "label": "BTC 15m"},
    "btc_5m":  {"subdir": "btc_5m",  "window_s": 300, "label": "BTC 5m"},
    "eth_15m": {"subdir": "eth_15m", "window_s": 900, "label": "ETH 15m"},
    "eth_5m":  {"subdir": "eth_5m",  "window_s": 300, "label": "ETH 5m"},
}

def norm_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))

def poly_fee(p):
    return 0.25 * (p * (1.0 - p)) ** 2

def compute_vol_deduped(prices, timestamps=None):
    changes = []
    for i, p in enumerate(prices):
        ts = timestamps[i] if timestamps is not None else i * 1000
        if p > 0 and (not changes or p != changes[-1][1]):
            changes.append((i, p, ts))
    if len(changes) < 3:
        return 0.0
    log_rets = []
    for j in range(1, len(changes)):
        if timestamps is not None:
            dt = (changes[j][2] - changes[j-1][2]) / 1000.0
        else:
            dt = changes[j][0] - changes[j-1][0]
        if dt > 0:
            lr = math.log(changes[j][1] / changes[j-1][1])
            log_rets.append(lr / math.sqrt(dt))
    if len(log_rets) < 2:
        return 0.0
    return float(np.std(log_rets, ddof=1))

MIN_FINAL_REMAINING_S = 5.0
MAX_START_GAP_S = 30.0

def load_windows(market_key):
    cfg = MARKETS[market_key]
    data_dir = DATA_DIR / cfg["subdir"]
    files = sorted(data_dir.glob("*.parquet"))
    windows = []
    skipped = 0
    for f in files:
        df = pd.read_parquet(f)
        if df.empty:
            skipped += 1
            continue
        if "chainlink_btc" in df.columns and "chainlink_price" not in df.columns:
            df.rename(columns={"chainlink_btc": "chainlink_price"}, inplace=True)
        if "window_end_ms" in df.columns:
            end_ms = df["window_end_ms"].iloc[0]
            last_ts = df["ts_ms"].iloc[-1]
            if last_ts < end_ms:
                skipped += 1
                continue
        else:
            final_remaining = df["time_remaining_s"].iloc[-1]
            if final_remaining > MIN_FINAL_REMAINING_S:
                skipped += 1
                continue
        if ("window_start_ms" in df.columns and "window_end_ms" in df.columns
                and "time_remaining_s" in df.columns):
            window_dur_s = (df["window_end_ms"].iloc[0] - df["window_start_ms"].iloc[0]) / 1000
            first_remaining = df["time_remaining_s"].iloc[0]
            if first_remaining < window_dur_s - MAX_START_GAP_S:
                skipped += 1
                continue
        start_prices = df["window_start_price"].dropna()
        if start_prices.empty:
            skipped += 1
            continue
        start_px = float(start_prices.iloc[0])
        final_px = float(df["chainlink_price"].iloc[-1])
        if pd.isna(start_px) or pd.isna(final_px) or start_px == 0:
            skipped += 1
            continue
        outcome = 1 if final_px >= start_px else 0
        windows.append((df, outcome, start_px, final_px))
    windows.sort(key=lambda x: x[0]["ts_ms"].iloc[0])
    return windows, skipped


def extract_signals(windows, vol_lookback_s=90, max_z=1.5, sample_every=30):
    records = []
    for df, outcome, start_px, final_px in windows:
        prices = df["chainlink_price"].tolist()
        ts_list = df["ts_ms"].tolist()
        for idx in range(vol_lookback_s, len(df), sample_every):
            row = df.iloc[idx]
            tau = row["time_remaining_s"]
            if tau <= 0:
                continue
            lo = max(0, idx - vol_lookback_s)
            price_slice = prices[lo:idx+1]
            ts_slice = ts_list[lo:idx+1]
            has_gap = False
            for k in range(1, len(ts_slice)):
                if ts_slice[k] - ts_slice[k-1] > 5000:
                    has_gap = True
                    break
            if has_gap:
                continue
            sigma = compute_vol_deduped(price_slice, ts_slice)
            if sigma <= 0:
                continue
            current_px = row["chainlink_price"]
            delta = (current_px - start_px) / start_px
            z_raw = delta / (sigma * math.sqrt(tau))
            z_capped = max(-max_z, min(max_z, z_raw))
            p_model = norm_cdf(z_capped)
            records.append({
                "p_model": p_model, "z_raw": z_raw, "z_capped": z_capped,
                "sigma": sigma, "tau": tau, "delta": delta,
                "outcome_up": outcome, "start_px": start_px,
                "current_px": current_px, "final_px": final_px,
            })
    return pd.DataFrame(records)


def calibration_analysis(sdf, n_bins=10, label=""):
    if sdf.empty:
        print(f"  {label}: No data")
        return None
    sdf = sdf.copy()
    sdf["p_bin"] = pd.cut(sdf["p_model"], bins=n_bins, labels=False)
    rows = []
    for b in range(n_bins):
        mask = sdf["p_bin"] == b
        if mask.sum() == 0:
            continue
        subset = sdf[mask]
        pred_mean = subset["p_model"].mean()
        actual_mean = subset["outcome_up"].mean()
        n = len(subset)
        se = math.sqrt(actual_mean * (1 - actual_mean) / n) if n > 1 else 0
        rows.append({
            "bin": b, "pred_p": round(pred_mean, 3), "actual_p": round(actual_mean, 3),
            "gap": round(actual_mean - pred_mean, 3), "n_obs": n,
            "ci_lo": round(max(0, actual_mean - 1.96*se), 3),
            "ci_hi": round(min(1, actual_mean + 1.96*se), 3),
        })
    cal_df = pd.DataFrame(rows)
    brier = ((sdf["p_model"] - sdf["outcome_up"]) ** 2).mean()
    ece = sum(abs(r["gap"]) * r["n_obs"] for _, r in cal_df.iterrows()) / len(sdf)
    print(f"\n{'='*70}")
    print(f"  CALIBRATION: {label}")
    print(f"  Brier Score: {brier:.4f} (0.25=random, 0=perfect)")
    print(f"  ECE:         {ece:.4f} (0=perfect calibration)")
    print(f"{'='*70}")
    print(cal_df.to_string(index=False))
    return cal_df, brier, ece


def z_score_analysis(sdf, label=""):
    if sdf.empty:
        return
    z_edges = [-3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3]
    sdf = sdf.copy()
    sdf["z_bin"] = pd.cut(sdf["z_capped"], bins=z_edges)
    print(f"\n{'='*70}")
    print(f"  Z-SCORE ANALYSIS: {label}")
    print(f"{'='*70}")
    rows = []
    for zb in sdf["z_bin"].cat.categories:
        mask = sdf["z_bin"] == zb
        if mask.sum() == 0:
            continue
        sub = sdf[mask]
        z_mid = (zb.left + zb.right) / 2
        predicted = norm_cdf(z_mid)
        actual = sub["outcome_up"].mean()
        rows.append({
            "z_bin": str(zb), "n": len(sub),
            "pred_up%": f"{predicted:.1%}", "actual_up%": f"{actual:.1%}",
            "gap": f"{(actual-predicted):+.1%}",
            "avg_sigma": f"{sub['sigma'].mean():.2e}", "avg_tau": f"{sub['tau'].mean():.0f}s",
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n  z_raw stats: mean={sdf['z_raw'].mean():.3f}, std={sdf['z_raw'].std():.3f}, "
          f"skew={sdf['z_raw'].skew():.3f}, kurt={sdf['z_raw'].kurtosis():.3f}")
    print(f"  |z_raw| > 1.5: {(sdf['z_raw'].abs() > 1.5).mean():.1%}")
    print(f"  |z_raw| > 3.0: {(sdf['z_raw'].abs() > 3.0).mean():.1%}")
    print(f"  sigma stats: median={sdf['sigma'].median():.2e}, "
          f"p95={sdf['sigma'].quantile(0.95):.2e}, max={sdf['sigma'].max():.2e}")


def vol_regime_analysis(sdf, label=""):
    if sdf.empty:
        return
    sdf = sdf.copy()
    sdf["vol_regime"] = pd.qcut(sdf["sigma"], q=4, labels=["Low", "Med-Low", "Med-High", "High"])
    print(f"\n{'='*70}")
    print(f"  VOLATILITY REGIME ANALYSIS: {label}")
    print(f"{'='*70}")
    rows = []
    for regime in ["Low", "Med-Low", "Med-High", "High"]:
        mask = sdf["vol_regime"] == regime
        if mask.sum() == 0:
            continue
        sub = sdf[mask]
        brier = ((sub["p_model"] - sub["outcome_up"]) ** 2).mean()
        correct = ((sub["p_model"] > 0.5) & (sub["outcome_up"] == 1)) | \
                  ((sub["p_model"] < 0.5) & (sub["outcome_up"] == 0))
        accuracy = correct.mean()
        confidence = (sub["p_model"] - 0.5).abs().mean()
        rows.append({
            "regime": regime, "n_obs": len(sub),
            "sigma_range": f"{sub['sigma'].min():.2e}-{sub['sigma'].max():.2e}",
            "brier": f"{brier:.4f}", "accuracy": f"{accuracy:.1%}",
            "avg_conf": f"{confidence:.3f}", "actual_up%": f"{sub['outcome_up'].mean():.1%}",
        })
    print(pd.DataFrame(rows).to_string(index=False))
    p95 = sdf["sigma"].quantile(0.95)
    tail = sdf[sdf["sigma"] > p95]
    rest = sdf[sdf["sigma"] <= p95]
    if len(tail) > 10:
        brier_tail = ((tail["p_model"] - tail["outcome_up"]) ** 2).mean()
        brier_rest = ((rest["p_model"] - rest["outcome_up"]) ** 2).mean()
        print(f"\n  Tail vol (>p95, sigma>{p95:.2e}): Brier={brier_tail:.4f} ({len(tail)} obs)")
        print(f"  Normal vol (<=p95):                 Brier={brier_rest:.4f}")
        print(f"  -> High-vol Brier is {'WORSE' if brier_tail > brier_rest else 'BETTER'} by {abs(brier_tail-brier_rest):.4f}")


def tau_analysis(sdf, window_s, label=""):
    if sdf.empty:
        return
    sdf = sdf.copy()
    if window_s == 900:
        tau_edges = [0, 120, 300, 600, 900]
        tau_labels = ["0-2m", "2-5m", "5-10m", "10-15m"]
    else:
        tau_edges = [0, 60, 120, 200, 300]
        tau_labels = ["0-1m", "1-2m", "2-3.3m", "3.3-5m"]
    sdf["tau_bin"] = pd.cut(sdf["tau"], bins=tau_edges, labels=tau_labels)
    print(f"\n{'='*70}")
    print(f"  TAU (TIME REMAINING) ANALYSIS: {label}")
    print(f"{'='*70}")
    rows = []
    for tl in tau_labels:
        mask = sdf["tau_bin"] == tl
        if mask.sum() == 0:
            continue
        sub = sdf[mask]
        brier = ((sub["p_model"] - sub["outcome_up"]) ** 2).mean()
        correct = ((sub["p_model"] > 0.5) & (sub["outcome_up"] == 1)) | \
                  ((sub["p_model"] < 0.5) & (sub["outcome_up"] == 0))
        accuracy = correct.mean()
        confidence = (sub["p_model"] - 0.5).abs().mean()
        rows.append({
            "tau_bin": tl, "n_obs": len(sub), "brier": f"{brier:.4f}",
            "accuracy": f"{accuracy:.1%}", "avg_|p-0.5|": f"{confidence:.3f}",
            "actual_up%": f"{sub['outcome_up'].mean():.1%}",
        })
    print(pd.DataFrame(rows).to_string(index=False))


def walk_forward_oos(windows, vol_lookback_s=90, max_z=1.5, train_frac=0.6, label=""):
    n = len(windows)
    split_idx = int(n * train_frac)
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD OOS TEST: {label}")
    print(f"  Total: {n} | Train: {split_idx} | Test: {n - split_idx}")
    print(f"{'='*70}")

    train_obs = []
    for df, outcome, start_px, _ in windows[:split_idx]:
        prices = df["chainlink_price"].tolist()
        ts_list = df["ts_ms"].tolist()
        for idx in range(vol_lookback_s, len(df), 30):
            row = df.iloc[idx]
            tau = row["time_remaining_s"]
            if tau <= 0:
                continue
            lo = max(0, idx - vol_lookback_s)
            sigma = compute_vol_deduped(prices[lo:idx+1], ts_list[lo:idx+1])
            if sigma <= 0:
                continue
            delta = (row["chainlink_price"] - start_px) / start_px
            z_raw = delta / (sigma * math.sqrt(tau))
            z_capped = max(-max_z, min(max_z, z_raw))
            train_obs.append((z_capped, tau, outcome))

    Z_BIN_WIDTH = 0.5
    TAU_EDGES = [0, 120, 300, 600, 900]
    cell_outcomes = defaultdict(list)
    for z_c, tau, out in train_obs:
        z_bin = round(z_c / Z_BIN_WIDTH) * Z_BIN_WIDTH
        tau_idx = 0
        for i in range(len(TAU_EDGES) - 1):
            if TAU_EDGES[i] <= tau < TAU_EDGES[i+1]:
                tau_idx = i
                break
        cell_outcomes[(z_bin, tau_idx)].append(out)
    cal_table = {}
    cal_counts = {}
    for key, outcomes in cell_outcomes.items():
        cal_table[key] = sum(outcomes) / len(outcomes)
        cal_counts[key] = len(outcomes)

    test_records = []
    for df, outcome, start_px, final_px in windows[split_idx:]:
        prices = df["chainlink_price"].tolist()
        ts_list = df["ts_ms"].tolist()
        for idx in range(vol_lookback_s, len(df), 30):
            row = df.iloc[idx]
            tau = row["time_remaining_s"]
            if tau <= 0:
                continue
            lo = max(0, idx - vol_lookback_s)
            sigma = compute_vol_deduped(prices[lo:idx+1], ts_list[lo:idx+1])
            if sigma <= 0:
                continue
            delta = (row["chainlink_price"] - start_px) / start_px
            z_raw = delta / (sigma * math.sqrt(tau))
            z_capped = max(-max_z, min(max_z, z_raw))
            p_gbm = norm_cdf(z_capped)
            z_bin = round(z_capped / Z_BIN_WIDTH) * Z_BIN_WIDTH
            tau_idx_v = 0
            for i in range(len(TAU_EDGES) - 1):
                if TAU_EDGES[i] <= tau < TAU_EDGES[i+1]:
                    tau_idx_v = i
                    break
            key = (z_bin, tau_idx_v)
            if key in cal_table and cal_counts.get(key, 0) >= 20:
                n_cal = cal_counts[key]
                w = n_cal / (n_cal + 100.0)
                p_fused = w * cal_table[key] + (1 - w) * p_gbm
            else:
                p_fused = p_gbm
            test_records.append({
                "p_gbm": p_gbm, "p_fused": p_fused,
                "z_capped": z_capped, "sigma": sigma, "tau": tau,
                "outcome_up": outcome,
            })

    test_df = pd.DataFrame(test_records)
    if test_df.empty:
        print("  No test data.")
        return

    brier_gbm = ((test_df["p_gbm"] - test_df["outcome_up"]) ** 2).mean()
    brier_fused = ((test_df["p_fused"] - test_df["outcome_up"]) ** 2).mean()
    acc_gbm = (((test_df["p_gbm"] > 0.5) & (test_df["outcome_up"] == 1)) |
               ((test_df["p_gbm"] < 0.5) & (test_df["outcome_up"] == 0))).mean()
    acc_fused = (((test_df["p_fused"] > 0.5) & (test_df["outcome_up"] == 1)) |
                 ((test_df["p_fused"] < 0.5) & (test_df["outcome_up"] == 0))).mean()

    print(f"\n  Out-of-sample results ({len(test_df)} observations):")
    print(f"  {'Metric':<25s} {'Pure GBM':>12s} {'Fused (cal)':>12s}")
    print(f"  {'-'*49}")
    print(f"  {'Brier Score':<25s} {brier_gbm:>12.4f} {brier_fused:>12.4f}")
    print(f"  {'Direction Accuracy':<25s} {acc_gbm:>11.1%} {acc_fused:>11.1%}")
    if brier_fused < brier_gbm:
        print(f"\n  -> Calibration HELPS OOS: Brier improved by {brier_gbm - brier_fused:.4f}")
    else:
        print(f"\n  -> Calibration HURTS OOS: Brier worsened by {brier_fused - brier_gbm:.4f}")


def rolling_walk_forward_backtest(
    windows, window_s=900, vol_lookback_s=90, max_z=1.5,
    edge_threshold=0.04, early_edge_mult=0.4,
    kelly_fraction=0.25, max_bet_fraction=0.0125,
    initial_bankroll=10000.0, min_warmup_windows=30,
    maker_mode=True, label="", quiet=False,
):
    bankroll = initial_bankroll
    results = []
    past_obs = []
    TAU_EDGES_local = [0, 120, 300, 600, 900] if window_s == 900 else [0, 60, 120, 200, 300]

    for wi, (df, outcome, start_px, final_px) in enumerate(windows):
        prices = df["chainlink_price"].tolist()
        ts_list = df["ts_ms"].tolist()
        window_obs = []
        for idx in range(vol_lookback_s, len(df), 30):
            row = df.iloc[idx]
            tau = row["time_remaining_s"]
            if tau <= 0:
                continue
            lo = max(0, idx - vol_lookback_s)
            sigma = compute_vol_deduped(prices[lo:idx+1], ts_list[lo:idx+1])
            if sigma <= 0:
                continue
            delta = (row["chainlink_price"] - start_px) / start_px
            z_raw = delta / (sigma * math.sqrt(tau))
            z_capped = max(-max_z, min(max_z, z_raw))
            window_obs.append((z_capped, tau, outcome))

        if wi < min_warmup_windows:
            past_obs.extend(window_obs)
            continue

        # Build cal from past only
        cell_outcomes = defaultdict(list)
        for z_c, tau_v, out in past_obs:
            z_bin = round(z_c / 0.5) * 0.5
            tau_idx = len(TAU_EDGES_local) - 2
            for i in range(len(TAU_EDGES_local) - 1):
                if TAU_EDGES_local[i] <= tau_v < TAU_EDGES_local[i+1]:
                    tau_idx = i
                    break
            cell_outcomes[(z_bin, tau_idx)].append(out)
        cal = {k: sum(v)/len(v) for k, v in cell_outcomes.items()}
        cal_n = {k: len(v) for k, v in cell_outcomes.items()}

        target_tau = window_s * 0.5
        best_idx = None
        best_dist = float('inf')
        for idx in range(vol_lookback_s, len(df)):
            tau = df.iloc[idx]["time_remaining_s"]
            dist = abs(tau - target_tau)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is None:
            past_obs.extend(window_obs)
            continue
        row = df.iloc[best_idx]
        tau = row["time_remaining_s"]
        if tau <= 0:
            past_obs.extend(window_obs)
            continue
        lo = max(0, best_idx - vol_lookback_s)
        sigma = compute_vol_deduped(prices[lo:best_idx+1], ts_list[lo:best_idx+1])
        if sigma <= 0:
            past_obs.extend(window_obs)
            continue

        current_px = row["chainlink_price"]
        delta = (current_px - start_px) / start_px
        z_raw = delta / (sigma * math.sqrt(tau))
        z_capped = max(-max_z, min(max_z, z_raw))
        p_gbm = norm_cdf(z_capped)

        z_bin = round(z_capped / 0.5) * 0.5
        tau_idx_v = len(TAU_EDGES_local) - 2
        for i in range(len(TAU_EDGES_local) - 1):
            if TAU_EDGES_local[i] <= tau < TAU_EDGES_local[i+1]:
                tau_idx_v = i
                break
        key = (z_bin, tau_idx_v)
        if key in cal and cal_n.get(key, 0) >= 20:
            w = cal_n[key] / (cal_n[key] + 100.0)
            p_model = w * cal[key] + (1 - w) * p_gbm
        else:
            p_model = p_gbm

        bid_up = row.get("best_bid_up")
        ask_up = row.get("best_ask_up")
        bid_down = row.get("best_bid_down")
        ask_down = row.get("best_ask_down")
        if any(pd.isna(x) for x in [bid_up, ask_up, bid_down, ask_down]):
            past_obs.extend(window_obs)
            continue
        if float(bid_up) <= 0 or float(bid_down) <= 0:
            past_obs.extend(window_obs)
            continue

        dyn_threshold = edge_threshold * (1.0 + early_edge_mult * math.sqrt(tau / window_s))
        if maker_mode:
            cost_up = float(bid_up)
            cost_down = float(bid_down)
        else:
            cost_up = float(ask_up) + poly_fee(float(ask_up))
            cost_down = float(ask_down) + poly_fee(float(ask_down))

        edge_up = p_model - cost_up
        edge_down = (1.0 - p_model) - cost_down

        action = "FLAT"
        edge = 0.0
        entry_price = 0.0
        p_side = 0.5
        if edge_up > dyn_threshold and edge_up >= edge_down:
            action, edge, entry_price, p_side = "BUY_UP", edge_up, cost_up, p_model
        elif edge_down > dyn_threshold and edge_down > edge_up:
            action, edge, entry_price, p_side = "BUY_DOWN", edge_down, cost_down, 1 - p_model

        if action != "FLAT" and 0 < entry_price < 1:
            kelly_f = max(0, (p_side - entry_price) / (1 - entry_price))
            frac = min(kelly_fraction * kelly_f, max_bet_fraction)
            size_usd = bankroll * frac
            if size_usd < 5 * entry_price:
                size_usd = min(5 * entry_price, bankroll * 0.02)
            shares = size_usd / entry_price
            cost = shares * entry_price
            won = (action == "BUY_UP" and outcome == 1) or (action == "BUY_DOWN" and outcome == 0)
            payout = shares if won else 0
            pnl = payout - cost
            bankroll += pnl
            results.append({
                "window_idx": wi, "action": action, "edge": edge,
                "p_model": p_model, "p_gbm": p_gbm, "entry_price": entry_price,
                "shares": shares, "cost": cost, "payout": payout, "pnl": pnl,
                "won": won, "outcome_up": outcome, "sigma": sigma, "tau": tau,
                "z_capped": z_capped, "bankroll": bankroll, "dyn_threshold": dyn_threshold,
            })

        past_obs.extend(window_obs)

    if not results:
        if not quiet:
            print(f"  {label}: No trades in walk-forward.")
        return None

    rdf = pd.DataFrame(results)
    if not quiet:
        n_trades = len(rdf)
        n_wins = int(rdf["won"].sum())
        win_rate = n_wins / n_trades
        total_pnl = rdf["pnl"].sum()
        avg_edge = rdf["edge"].mean()
        periods_per_day = 96 if window_s == 900 else 288
        std_r = rdf["pnl"].std()
        sharpe = (rdf["pnl"].mean() / std_r) * math.sqrt(periods_per_day) if std_r > 0 else 0
        peak = initial_bankroll
        max_dd = 0
        for _, r in rdf.iterrows():
            if r["bankroll"] > peak: peak = r["bankroll"]
            dd = peak - r["bankroll"]
            if dd > max_dd: max_dd = dd

        print(f"\n{'='*70}")
        print(f"  ROLLING WALK-FORWARD BACKTEST: {label}")
        print(f"{'='*70}")
        print(f"  Trades:         {n_trades}")
        print(f"  Win rate:       {win_rate:.1%} ({n_wins}/{n_trades})")
        print(f"  Total PnL:      ${total_pnl:+.2f}")
        print(f"  Avg PnL/trade:  ${rdf['pnl'].mean():+.2f}")
        print(f"  Avg edge:       {avg_edge:.4f}")
        print(f"  Max drawdown:   ${max_dd:.2f}")
        print(f"  Sharpe (ann.):  {sharpe:.2f}")
        print(f"  Final bankroll: ${bankroll:,.2f}")
        for side in ["BUY_UP", "BUY_DOWN"]:
            sm = rdf[rdf["action"] == side]
            if len(sm) > 0:
                print(f"  {side:>10s}: {int(sm['won'].sum())}/{len(sm)} wins ({sm['won'].mean():.1%}), "
                      f"avg edge={sm['edge'].mean():.4f}, avg pnl=${sm['pnl'].mean():+.2f}")
    return rdf


def blind_prediction_test(windows, window_s=900, vol_lookback_s=90, max_z=1.5, label=""):
    decision_points = [
        ("Early (75% left)", 0.75),
        ("Mid (50% left)", 0.50),
        ("Late (25% left)", 0.25),
    ]
    print(f"\n{'='*70}")
    print(f"  BLIND PREDICTION (Pure GBM, no tuning): {label}")
    print(f"{'='*70}")
    for dp_name, dp_frac in decision_points:
        target_tau = window_s * dp_frac
        predictions = []
        for df, outcome, start_px, final_px in windows:
            prices = df["chainlink_price"].tolist()
            ts_list = df["ts_ms"].tolist()
            best_idx = None
            best_dist = float('inf')
            for idx in range(vol_lookback_s, len(df)):
                tau = df.iloc[idx]["time_remaining_s"]
                dist = abs(tau - target_tau)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            if best_idx is None or best_dist > 30:
                continue
            row = df.iloc[best_idx]
            tau = row["time_remaining_s"]
            if tau <= 0:
                continue
            lo = max(0, best_idx - vol_lookback_s)
            sigma = compute_vol_deduped(prices[lo:best_idx+1], ts_list[lo:best_idx+1])
            if sigma <= 0:
                continue
            current_px = row["chainlink_price"]
            delta = (current_px - start_px) / start_px
            z_raw = delta / (sigma * math.sqrt(tau))
            z_capped = max(-max_z, min(max_z, z_raw))
            p_up = norm_cdf(z_capped)
            predicted_up = p_up > 0.5
            correct = (predicted_up and outcome == 1) or (not predicted_up and outcome == 0)
            confidence = abs(p_up - 0.5)
            predictions.append({
                "p_up": p_up, "predicted_up": predicted_up, "outcome_up": outcome,
                "correct": correct, "confidence": confidence, "z": z_capped, "sigma": sigma,
            })
        pdf = pd.DataFrame(predictions)
        if pdf.empty:
            continue
        acc = pdf["correct"].mean()
        brier = ((pdf["p_up"] - pdf["outcome_up"]) ** 2).mean()
        high_conf = pdf[pdf["confidence"] > 0.15]
        med_conf = pdf[(pdf["confidence"] > 0.05) & (pdf["confidence"] <= 0.15)]
        low_conf = pdf[pdf["confidence"] <= 0.05]
        print(f"\n  {dp_name}: {len(pdf)} predictions")
        print(f"    Overall accuracy: {acc:.1%} | Brier: {brier:.4f}")
        if len(high_conf) > 0:
            print(f"    High conf (|p-0.5|>0.15): {high_conf['correct'].mean():.1%} ({len(high_conf)} trades)")
        if len(med_conf) > 0:
            print(f"    Med  conf (0.05-0.15):    {med_conf['correct'].mean():.1%} ({len(med_conf)} trades)")
        if len(low_conf) > 0:
            print(f"    Low  conf (<0.05):         {low_conf['correct'].mean():.1%} ({len(low_conf)} trades)")


def market_bias_analysis(windows, label=""):
    records = []
    for df, outcome, start_px, final_px in windows:
        mid_idx = len(df) // 2
        row = df.iloc[mid_idx]
        bid_up = row.get("best_bid_up")
        ask_up = row.get("best_ask_up")
        bid_down = row.get("best_bid_down")
        ask_down = row.get("best_ask_down")
        if any(pd.isna(x) for x in [bid_up, ask_up, bid_down, ask_down]):
            continue
        mid_up = (float(bid_up) + float(ask_up)) / 2
        mid_down = (float(bid_down) + float(ask_down)) / 2
        records.append({
            "mid_up": mid_up, "mid_down": mid_down,
            "implied_sum": mid_up + mid_down, "outcome_up": outcome,
            "overround": mid_up + mid_down - 1.0,
        })
    if not records:
        return
    mdf = pd.DataFrame(records)
    print(f"\n{'='*70}")
    print(f"  MARKET BIAS ANALYSIS: {label}")
    print(f"{'='*70}")
    actual_up_rate = mdf['outcome_up'].mean()
    print(f"  Base rate:       UP wins {actual_up_rate:.1%}")
    print(f"  Market mid-UP:   {mdf['mid_up'].mean():.3f}")
    print(f"  Market mid-DOWN: {mdf['mid_down'].mean():.3f}")
    print(f"  Overround (vig): {mdf['overround'].mean():.3f}")
    pricing_gap = mdf['mid_up'].mean() - actual_up_rate
    print(f"  UP pricing gap:  market={mdf['mid_up'].mean():.3f} vs actual={actual_up_rate:.3f} -> {'OVERPRICED' if pricing_gap > 0.01 else 'UNDERPRICED' if pricing_gap < -0.01 else 'FAIR'} by {abs(pricing_gap):.3f}")


def fat_tail_analysis(windows, window_s=900, vol_lookback_s=90, label=""):
    deltas = []
    sigmas_at_mid = []
    for df, outcome, start_px, final_px in windows:
        delta_pct = (final_px - start_px) / start_px
        deltas.append(delta_pct)
        mid_idx = len(df) // 2
        lo = max(0, mid_idx - vol_lookback_s)
        prices = df["chainlink_price"].tolist()
        ts_list = df["ts_ms"].tolist()
        sigma = compute_vol_deduped(prices[lo:mid_idx+1], ts_list[lo:mid_idx+1])
        if sigma > 0:
            sigmas_at_mid.append(sigma)
    deltas = np.array(deltas)
    sigmas_at_mid = np.array(sigmas_at_mid)
    print(f"\n{'='*70}")
    print(f"  FAT TAIL ANALYSIS: {label}")
    print(f"{'='*70}")
    print(f"  Window returns: N={len(deltas)}, mean={deltas.mean()*100:.4f}%, "
          f"std={deltas.std()*100:.4f}%")
    print(f"  Skew={pd.Series(deltas).skew():.3f}, "
          f"Kurt={pd.Series(deltas).kurtosis():.3f} (Normal=0)")
    if len(sigmas_at_mid) > 0:
        median_sigma = np.median(sigmas_at_mid)
        expected_sigma_window = median_sigma * math.sqrt(window_s)
        if expected_sigma_window > 0:
            normalized = deltas[:len(sigmas_at_mid)] / expected_sigma_window
            tail_2sig = (np.abs(normalized) > 2).mean()
            tail_3sig = (np.abs(normalized) > 3).mean()
            expected_2sig = 2 * (1 - norm_cdf(2))
            expected_3sig = 2 * (1 - norm_cdf(3))
            print(f"  >2-sigma events: actual={tail_2sig:.1%} vs GBM={expected_2sig:.1%} "
                  f"(ratio={tail_2sig/expected_2sig:.1f}x)" if expected_2sig > 0 else "")
            print(f"  >3-sigma events: actual={tail_3sig:.1%} vs GBM={expected_3sig:.2%} "
                  f"(ratio={tail_3sig/expected_3sig:.1f}x)" if expected_3sig > 0 else "")
        print(f"  Sigma: median={np.median(sigmas_at_mid):.2e}, "
              f"p95={np.percentile(sigmas_at_mid, 95):.2e}, "
              f"p99={np.percentile(sigmas_at_mid, 99):.2e}, "
              f"max={np.max(sigmas_at_mid):.2e}")
        print(f"  Vol-of-vol (CoV): {np.std(sigmas_at_mid)/np.mean(sigmas_at_mid):.2f}")


def parameter_sweep(windows, window_s=900, vol_lookback_s=90, max_z=1.5,
                     initial_bankroll=10000.0, min_warmup_windows=30, label=""):
    edge_thresholds = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
    max_z_values = [1.0, 1.5, 2.0, 2.5]
    early_mults = [0.0, 0.2, 0.4, 0.6, 0.8]

    print(f"\n{'='*70}")
    print(f"  PARAMETER SWEEP: {label}")
    print(f"{'='*70}")

    print(f"\n  --- Edge Threshold (early_mult=0.4, max_z=1.5) ---")
    rows = []
    for et in edge_thresholds:
        rdf = rolling_walk_forward_backtest(
            windows, window_s=window_s, vol_lookback_s=vol_lookback_s, max_z=1.5,
            edge_threshold=et, early_edge_mult=0.4,
            initial_bankroll=initial_bankroll, min_warmup_windows=min_warmup_windows,
            label="", quiet=True,
        )
        if rdf is not None and len(rdf) > 0:
            rows.append({
                "edge": et, "trades": len(rdf),
                "win%": f"{rdf['won'].mean():.1%}",
                "pnl": f"${rdf['pnl'].sum():+.2f}",
                "avg_edge": f"{rdf['edge'].mean():.4f}",
                "bank": f"${rdf['bankroll'].iloc[-1]:,.0f}",
            })
        else:
            rows.append({"edge": et, "trades": 0, "win%": "N/A", "pnl": "$0", "avg_edge": "N/A", "bank": f"${initial_bankroll:,.0f}"})
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n  --- Max Z (edge=0.04, early_mult=0.4) ---")
    rows = []
    for mz in max_z_values:
        rdf = rolling_walk_forward_backtest(
            windows, window_s=window_s, vol_lookback_s=vol_lookback_s, max_z=mz,
            edge_threshold=0.04, early_edge_mult=0.4,
            initial_bankroll=initial_bankroll, min_warmup_windows=min_warmup_windows,
            label="", quiet=True,
        )
        if rdf is not None and len(rdf) > 0:
            rows.append({
                "max_z": mz, "trades": len(rdf),
                "win%": f"{rdf['won'].mean():.1%}",
                "pnl": f"${rdf['pnl'].sum():+.2f}",
                "avg_edge": f"{rdf['edge'].mean():.4f}",
            })
        else:
            rows.append({"max_z": mz, "trades": 0, "win%": "N/A", "pnl": "$0", "avg_edge": "N/A"})
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n  --- Early Edge Mult (edge=0.04, max_z=1.5) ---")
    rows = []
    for em in early_mults:
        rdf = rolling_walk_forward_backtest(
            windows, window_s=window_s, vol_lookback_s=vol_lookback_s, max_z=1.5,
            edge_threshold=0.04, early_edge_mult=em,
            initial_bankroll=initial_bankroll, min_warmup_windows=min_warmup_windows,
            label="", quiet=True,
        )
        if rdf is not None and len(rdf) > 0:
            rows.append({
                "early_mult": em, "trades": len(rdf),
                "win%": f"{rdf['won'].mean():.1%}",
                "pnl": f"${rdf['pnl'].sum():+.2f}",
                "avg_edge": f"{rdf['edge'].mean():.4f}",
            })
        else:
            rows.append({"early_mult": em, "trades": 0, "win%": "N/A", "pnl": "$0", "avg_edge": "N/A"})
    print(pd.DataFrame(rows).to_string(index=False))


# ── MAIN ──
if __name__ == "__main__":
    print("Loading data...")
    all_windows = {}
    for mk in MARKETS:
        wins, skp = load_windows(mk)
        all_windows[mk] = wins
        up_count = sum(1 for _, o, _, _ in wins if o == 1)
        n = len(wins)
        print(f'{MARKETS[mk]["label"]:>8s}: {n:3d} windows ({skp} skip) | '
              f'UP: {up_count} ({up_count/n*100:.1f}%) | DOWN: {n-up_count} ({(n-up_count)/n*100:.1f}%)')

    print("\n\n" + "#"*70)
    print("#  SECTION 1: CALIBRATION ANALYSIS")
    print("#"*70)
    signal_dfs = {}
    for mk in MARKETS:
        sdf = extract_signals(all_windows[mk])
        signal_dfs[mk] = sdf
        print(f'{MARKETS[mk]["label"]:>8s}: {len(sdf):,d} signal observations')
    for mk in MARKETS:
        calibration_analysis(signal_dfs[mk], label=MARKETS[mk]["label"])

    print("\n\n" + "#"*70)
    print("#  SECTION 2: Z-SCORE ANALYSIS")
    print("#"*70)
    for mk in MARKETS:
        z_score_analysis(signal_dfs[mk], label=MARKETS[mk]["label"])

    print("\n\n" + "#"*70)
    print("#  SECTION 3: VOLATILITY REGIME ANALYSIS")
    print("#"*70)
    for mk in MARKETS:
        vol_regime_analysis(signal_dfs[mk], label=MARKETS[mk]["label"])

    print("\n\n" + "#"*70)
    print("#  SECTION 4: TAU (TIME REMAINING) ANALYSIS")
    print("#"*70)
    for mk in MARKETS:
        tau_analysis(signal_dfs[mk], MARKETS[mk]["window_s"], label=MARKETS[mk]["label"])

    print("\n\n" + "#"*70)
    print("#  SECTION 5: FAT TAIL ANALYSIS")
    print("#"*70)
    for mk in MARKETS:
        fat_tail_analysis(all_windows[mk], window_s=MARKETS[mk]["window_s"], label=MARKETS[mk]["label"])

    print("\n\n" + "#"*70)
    print("#  SECTION 6: MARKET BIAS ANALYSIS")
    print("#"*70)
    for mk in MARKETS:
        market_bias_analysis(all_windows[mk], label=MARKETS[mk]["label"])

    print("\n\n" + "#"*70)
    print("#  SECTION 7: BLIND PREDICTION TEST (Pure GBM)")
    print("#"*70)
    for mk in MARKETS:
        blind_prediction_test(all_windows[mk], window_s=MARKETS[mk]["window_s"], label=MARKETS[mk]["label"])

    print("\n\n" + "#"*70)
    print("#  SECTION 8: OUT-OF-SAMPLE WALK-FORWARD TEST")
    print("#"*70)
    for mk in MARKETS:
        walk_forward_oos(all_windows[mk], label=MARKETS[mk]["label"])

    print("\n\n" + "#"*70)
    print("#  SECTION 9: ROLLING WALK-FORWARD BACKTEST")
    print("#"*70)
    for mk in MARKETS:
        rolling_walk_forward_backtest(all_windows[mk], window_s=MARKETS[mk]["window_s"], label=MARKETS[mk]["label"])

    print("\n\n" + "#"*70)
    print("#  SECTION 10: PARAMETER SWEEP (OOS)")
    print("#"*70)
    for mk in MARKETS:
        parameter_sweep(all_windows[mk], window_s=MARKETS[mk]["window_s"], label=MARKETS[mk]["label"])
