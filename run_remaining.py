#!/usr/bin/env python3
"""Remaining sections: Blind prediction, OOS, Walk-forward, Parameter sweep (fast)."""
import math, numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = Path("data")
MARKETS = {
    "btc_15m": {"subdir": "btc_15m", "window_s": 900, "label": "BTC 15m"},
    "btc_5m":  {"subdir": "btc_5m",  "window_s": 300, "label": "BTC 5m"},
    "eth_15m": {"subdir": "eth_15m", "window_s": 900, "label": "ETH 15m"},
    "eth_5m":  {"subdir": "eth_5m",  "window_s": 300, "label": "ETH 5m"},
}

def norm_cdf(x): return 0.5 * math.erfc(-x / math.sqrt(2.0))
def poly_fee(p): return 0.25 * (p * (1.0 - p)) ** 2

def compute_vol_deduped(prices, timestamps=None):
    changes = []
    for i, p in enumerate(prices):
        ts = timestamps[i] if timestamps is not None else i * 1000
        if p > 0 and (not changes or p != changes[-1][1]):
            changes.append((i, p, ts))
    if len(changes) < 3: return 0.0
    log_rets = []
    for j in range(1, len(changes)):
        dt = (changes[j][2] - changes[j-1][2]) / 1000.0 if timestamps else changes[j][0] - changes[j-1][0]
        if dt > 0:
            log_rets.append(math.log(changes[j][1] / changes[j-1][1]) / math.sqrt(dt))
    return float(np.std(log_rets, ddof=1)) if len(log_rets) >= 2 else 0.0

MIN_FINAL_REMAINING_S = 5.0
MAX_START_GAP_S = 30.0

def load_windows(market_key):
    cfg = MARKETS[market_key]
    data_dir = DATA_DIR / cfg["subdir"]
    windows, skipped = [], 0
    for f in sorted(data_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        if df.empty: skipped += 1; continue
        if "chainlink_btc" in df.columns and "chainlink_price" not in df.columns:
            df.rename(columns={"chainlink_btc": "chainlink_price"}, inplace=True)
        if "window_end_ms" in df.columns:
            if df["ts_ms"].iloc[-1] < df["window_end_ms"].iloc[0]: skipped += 1; continue
        else:
            if df["time_remaining_s"].iloc[-1] > MIN_FINAL_REMAINING_S: skipped += 1; continue
        if "window_start_ms" in df.columns and "window_end_ms" in df.columns and "time_remaining_s" in df.columns:
            dur = (df["window_end_ms"].iloc[0] - df["window_start_ms"].iloc[0]) / 1000
            if df["time_remaining_s"].iloc[0] < dur - MAX_START_GAP_S: skipped += 1; continue
        sp = df["window_start_price"].dropna()
        if sp.empty: skipped += 1; continue
        start_px, final_px = float(sp.iloc[0]), float(df["chainlink_price"].iloc[-1])
        if pd.isna(start_px) or pd.isna(final_px) or start_px == 0: skipped += 1; continue
        windows.append((df, 1 if final_px >= start_px else 0, start_px, final_px))
    windows.sort(key=lambda x: x[0]["ts_ms"].iloc[0])
    return windows, skipped

print("Loading data...")
all_windows = {}
for mk in MARKETS:
    all_windows[mk], _ = load_windows(mk)

# ── SECTION 7: BLIND PREDICTION ──
print("\n" + "#"*70)
print("#  SECTION 7: BLIND PREDICTION TEST")
print("#"*70)

for mk in MARKETS:
    windows = all_windows[mk]
    ws = MARKETS[mk]["window_s"]
    label = MARKETS[mk]["label"]
    print(f"\n{'='*70}")
    print(f"  BLIND PREDICTION (Pure GBM): {label}")
    print(f"{'='*70}")
    for dp_name, dp_frac in [("Early (75%)", 0.75), ("Mid (50%)", 0.50), ("Late (25%)", 0.25)]:
        target_tau = ws * dp_frac
        preds = []
        for df, outcome, start_px, _ in windows:
            prices, ts_list = df["chainlink_price"].tolist(), df["ts_ms"].tolist()
            best_idx, best_dist = None, float('inf')
            for idx in range(90, len(df)):
                d = abs(df.iloc[idx]["time_remaining_s"] - target_tau)
                if d < best_dist: best_dist = d; best_idx = idx
            if best_idx is None or best_dist > 30: continue
            row = df.iloc[best_idx]
            tau = row["time_remaining_s"]
            if tau <= 0: continue
            lo = max(0, best_idx - 90)
            sigma = compute_vol_deduped(prices[lo:best_idx+1], ts_list[lo:best_idx+1])
            if sigma <= 0: continue
            delta = (row["chainlink_price"] - start_px) / start_px
            z = max(-1.5, min(1.5, delta / (sigma * math.sqrt(tau))))
            p_up = norm_cdf(z)
            correct = (p_up > 0.5 and outcome == 1) or (p_up < 0.5 and outcome == 0)
            preds.append({"p_up": p_up, "correct": correct, "confidence": abs(p_up - 0.5), "outcome": outcome})
        if not preds: continue
        pdf = pd.DataFrame(preds)
        brier = ((pdf["p_up"] - pdf["outcome"]) ** 2).mean()
        hi = pdf[pdf["confidence"] > 0.15]
        md = pdf[(pdf["confidence"] > 0.05) & (pdf["confidence"] <= 0.15)]
        lo = pdf[pdf["confidence"] <= 0.05]
        print(f"\n  {dp_name}: {len(pdf)} preds, accuracy={pdf['correct'].mean():.1%}, Brier={brier:.4f}")
        if len(hi) > 0: print(f"    High conf (>0.15): {hi['correct'].mean():.1%} ({len(hi)} trades)")
        if len(md) > 0: print(f"    Med  conf (0.05-0.15): {md['correct'].mean():.1%} ({len(md)} trades)")
        if len(lo) > 0: print(f"    Low  conf (<0.05):  {lo['correct'].mean():.1%} ({len(lo)} trades)")

# ── SECTION 8: OOS FIXED SPLIT ──
print("\n" + "#"*70)
print("#  SECTION 8: OOS FIXED SPLIT TEST")
print("#"*70)

for mk in MARKETS:
    windows = all_windows[mk]
    label = MARKETS[mk]["label"]
    n = len(windows)
    split = int(n * 0.6)
    train_obs = []
    for df, outcome, start_px, _ in windows[:split]:
        prices, ts_list = df["chainlink_price"].tolist(), df["ts_ms"].tolist()
        for idx in range(90, len(df), 30):
            tau = df.iloc[idx]["time_remaining_s"]
            if tau <= 0: continue
            lo = max(0, idx - 90)
            sigma = compute_vol_deduped(prices[lo:idx+1], ts_list[lo:idx+1])
            if sigma <= 0: continue
            delta = (df.iloc[idx]["chainlink_price"] - start_px) / start_px
            z = max(-1.5, min(1.5, delta / (sigma * math.sqrt(tau))))
            train_obs.append((z, tau, outcome))

    cell = defaultdict(list)
    TAU_E = [0, 120, 300, 600, 900]
    for z, tau, out in train_obs:
        zb = round(z / 0.5) * 0.5
        ti = next((i for i in range(len(TAU_E)-1) if TAU_E[i] <= tau < TAU_E[i+1]), len(TAU_E)-2)
        cell[(zb, ti)].append(out)
    cal = {k: sum(v)/len(v) for k, v in cell.items()}
    cal_n = {k: len(v) for k, v in cell.items()}

    test_recs = []
    for df, outcome, start_px, _ in windows[split:]:
        prices, ts_list = df["chainlink_price"].tolist(), df["ts_ms"].tolist()
        for idx in range(90, len(df), 30):
            tau = df.iloc[idx]["time_remaining_s"]
            if tau <= 0: continue
            lo = max(0, idx - 90)
            sigma = compute_vol_deduped(prices[lo:idx+1], ts_list[lo:idx+1])
            if sigma <= 0: continue
            delta = (df.iloc[idx]["chainlink_price"] - start_px) / start_px
            z = max(-1.5, min(1.5, delta / (sigma * math.sqrt(tau))))
            p_gbm = norm_cdf(z)
            zb = round(z / 0.5) * 0.5
            ti = next((i for i in range(len(TAU_E)-1) if TAU_E[i] <= tau < TAU_E[i+1]), len(TAU_E)-2)
            k = (zb, ti)
            if k in cal and cal_n.get(k, 0) >= 20:
                w = cal_n[k] / (cal_n[k] + 100)
                p_f = w * cal[k] + (1-w) * p_gbm
            else: p_f = p_gbm
            test_recs.append({"p_gbm": p_gbm, "p_fused": p_f, "outcome": outcome})

    tdf = pd.DataFrame(test_recs)
    b_g = ((tdf["p_gbm"] - tdf["outcome"]) ** 2).mean()
    b_f = ((tdf["p_fused"] - tdf["outcome"]) ** 2).mean()
    a_g = (((tdf["p_gbm"] > 0.5) & (tdf["outcome"] == 1)) | ((tdf["p_gbm"] < 0.5) & (tdf["outcome"] == 0))).mean()
    a_f = (((tdf["p_fused"] > 0.5) & (tdf["outcome"] == 1)) | ((tdf["p_fused"] < 0.5) & (tdf["outcome"] == 0))).mean()
    print(f"\n  {label}: Train={split}, Test={n-split}, {len(tdf)} obs")
    print(f"    {'':20s} {'GBM':>10s} {'Fused':>10s}")
    print(f"    {'Brier':20s} {b_g:>10.4f} {b_f:>10.4f}")
    print(f"    {'Accuracy':20s} {a_g:>9.1%} {a_f:>9.1%}")
    print(f"    -> Calibration {'HELPS' if b_f < b_g else 'HURTS'} by {abs(b_g - b_f):.4f}")

# ── SECTION 9: WALK-FORWARD BACKTEST ──
print("\n" + "#"*70)
print("#  SECTION 9: ROLLING WALK-FORWARD BACKTEST")
print("#"*70)

def fast_wf_bt(windows, ws, et=0.04, em=0.4, mz=1.5, maker=True, label="", quiet=False):
    bankroll = 10000.0
    results = []
    past_obs = []
    TAU_E = [0, 120, 300, 600, 900] if ws == 900 else [0, 60, 120, 200, 300]

    for wi, (df, outcome, start_px, _) in enumerate(windows):
        prices, ts_list = df["chainlink_price"].tolist(), df["ts_ms"].tolist()
        wobs = []
        for idx in range(90, len(df), 60):  # coarser sampling for speed
            tau = df.iloc[idx]["time_remaining_s"]
            if tau <= 0: continue
            lo = max(0, idx - 90)
            sigma = compute_vol_deduped(prices[lo:idx+1], ts_list[lo:idx+1])
            if sigma <= 0: continue
            delta = (df.iloc[idx]["chainlink_price"] - start_px) / start_px
            z = max(-mz, min(mz, delta / (sigma * math.sqrt(tau))))
            wobs.append((z, tau, outcome))

        if wi < 30:
            past_obs.extend(wobs)
            continue

        cell = defaultdict(list)
        for z, tau, out in past_obs:
            zb = round(z / 0.5) * 0.5
            ti = next((i for i in range(len(TAU_E)-1) if TAU_E[i] <= tau < TAU_E[i+1]), len(TAU_E)-2)
            cell[(zb, ti)].append(out)
        cal = {k: sum(v)/len(v) for k, v in cell.items()}
        cal_n = {k: len(v) for k, v in cell.items()}

        target = ws * 0.5
        best_idx = min(range(90, len(df)), key=lambda i: abs(df.iloc[i]["time_remaining_s"] - target), default=None)
        if best_idx is None: past_obs.extend(wobs); continue
        row = df.iloc[best_idx]
        tau = row["time_remaining_s"]
        if tau <= 0: past_obs.extend(wobs); continue
        lo = max(0, best_idx - 90)
        sigma = compute_vol_deduped(prices[lo:best_idx+1], ts_list[lo:best_idx+1])
        if sigma <= 0: past_obs.extend(wobs); continue

        delta = (row["chainlink_price"] - start_px) / start_px
        z = max(-mz, min(mz, delta / (sigma * math.sqrt(tau))))
        p_gbm = norm_cdf(z)
        zb = round(z / 0.5) * 0.5
        ti = next((i for i in range(len(TAU_E)-1) if TAU_E[i] <= tau < TAU_E[i+1]), len(TAU_E)-2)
        k = (zb, ti)
        if k in cal and cal_n.get(k, 0) >= 20:
            w = cal_n[k] / (cal_n[k] + 100)
            p_model = w * cal[k] + (1-w) * p_gbm
        else: p_model = p_gbm

        bu, au, bd, ad = row.get("best_bid_up"), row.get("best_ask_up"), row.get("best_bid_down"), row.get("best_ask_down")
        if any(pd.isna(x) for x in [bu, au, bd, ad]): past_obs.extend(wobs); continue
        bu, bd = float(bu), float(bd)
        if bu <= 0 or bd <= 0: past_obs.extend(wobs); continue

        dt = et * (1 + em * math.sqrt(tau / ws))
        cu, cd = (bu, bd) if maker else (float(au) + poly_fee(float(au)), float(ad) + poly_fee(float(ad)))
        eu, ed = p_model - cu, (1 - p_model) - cd

        action, edge, ep, ps = "FLAT", 0, 0, 0.5
        if eu > dt and eu >= ed: action, edge, ep, ps = "BUY_UP", eu, cu, p_model
        elif ed > dt: action, edge, ep, ps = "BUY_DOWN", ed, cd, 1 - p_model

        if action != "FLAT" and 0 < ep < 1:
            kf = max(0, (ps - ep) / (1 - ep))
            frac = min(0.25 * kf, 0.0125)
            size = max(bankroll * frac, min(5 * ep, bankroll * 0.02))
            shares = size / ep
            cost = shares * ep
            won = (action == "BUY_UP" and outcome == 1) or (action == "BUY_DOWN" and outcome == 0)
            pnl = (shares if won else 0) - cost
            bankroll += pnl
            results.append({"action": action, "won": won, "pnl": pnl, "edge": edge, "bankroll": bankroll})
        past_obs.extend(wobs)

    if not results: return None
    rdf = pd.DataFrame(results)
    if not quiet:
        nt = len(rdf)
        nw = int(rdf["won"].sum())
        print(f"\n  {label}: {nt} trades, {nw}/{nt} wins ({nw/nt:.1%}), "
              f"PnL=${rdf['pnl'].sum():+.2f}, final=${bankroll:,.0f}")
        for s in ["BUY_UP", "BUY_DOWN"]:
            sm = rdf[rdf["action"] == s]
            if len(sm) > 0:
                print(f"    {s}: {int(sm['won'].sum())}/{len(sm)} ({sm['won'].mean():.1%}), avg pnl=${sm['pnl'].mean():+.2f}")
    return rdf

for mk in MARKETS:
    fast_wf_bt(all_windows[mk], MARKETS[mk]["window_s"], label=MARKETS[mk]["label"])

# ── SECTION 10: PARAMETER SWEEP ──
print("\n" + "#"*70)
print("#  SECTION 10: PARAMETER SWEEP")
print("#"*70)

for mk in MARKETS:
    label = MARKETS[mk]["label"]
    ws = MARKETS[mk]["window_s"]
    windows = all_windows[mk]

    print(f"\n{'='*70}")
    print(f"  PARAMETER SWEEP: {label}")
    print(f"{'='*70}")

    print(f"\n  --- Edge Threshold (early_mult=0.4, max_z=1.5) ---")
    rows = []
    for et in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15]:
        r = fast_wf_bt(windows, ws, et=et, em=0.4, mz=1.5, label="", quiet=True)
        if r is not None and len(r) > 0:
            rows.append({"edge": et, "trades": len(r), "win%": f"{r['won'].mean():.1%}",
                         "pnl": f"${r['pnl'].sum():+.1f}", "bank": f"${r['bankroll'].iloc[-1]:,.0f}"})
        else:
            rows.append({"edge": et, "trades": 0, "win%": "N/A", "pnl": "$0", "bank": "$10,000"})
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n  --- Max Z (edge=0.04) ---")
    rows = []
    for mz in [0.8, 1.0, 1.5, 2.0, 2.5]:
        r = fast_wf_bt(windows, ws, et=0.04, mz=mz, label="", quiet=True)
        if r is not None and len(r) > 0:
            rows.append({"max_z": mz, "trades": len(r), "win%": f"{r['won'].mean():.1%}",
                         "pnl": f"${r['pnl'].sum():+.1f}"})
        else:
            rows.append({"max_z": mz, "trades": 0, "win%": "N/A", "pnl": "$0"})
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n  --- Early Edge Mult (edge=0.04) ---")
    rows = []
    for em in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        r = fast_wf_bt(windows, ws, et=0.04, em=em, label="", quiet=True)
        if r is not None and len(r) > 0:
            rows.append({"early_m": em, "trades": len(r), "win%": f"{r['won'].mean():.1%}",
                         "pnl": f"${r['pnl'].sum():+.1f}"})
        else:
            rows.append({"early_m": em, "trades": 0, "win%": "N/A", "pnl": "$0"})
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\n  --- Taker vs Maker ---")
    for mode, lbl in [(True, "Maker"), (False, "Taker")]:
        r = fast_wf_bt(windows, ws, et=0.04, maker=mode, label="", quiet=True)
        if r is not None and len(r) > 0:
            print(f"    {lbl}: {len(r)} trades, win={r['won'].mean():.1%}, pnl=${r['pnl'].sum():+.1f}")
        else:
            print(f"    {lbl}: 0 trades")
