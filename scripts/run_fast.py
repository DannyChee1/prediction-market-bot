#!/usr/bin/env python3
"""Fast remaining sections."""
import math, numpy as np, pandas as pd, sys
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

def cvd(prices, timestamps=None):
    changes = []
    for i, p in enumerate(prices):
        ts = timestamps[i] if timestamps is not None else i * 1000
        if p > 0 and (not changes or p != changes[-1][1]):
            changes.append((i, p, ts))
    if len(changes) < 3: return 0.0
    lr = []
    for j in range(1, len(changes)):
        dt = (changes[j][2] - changes[j-1][2]) / 1000.0 if timestamps else changes[j][0] - changes[j-1][0]
        if dt > 0: lr.append(math.log(changes[j][1] / changes[j-1][1]) / math.sqrt(dt))
    return float(np.std(lr, ddof=1)) if len(lr) >= 2 else 0.0

def load_windows(mk):
    cfg = MARKETS[mk]
    d = DATA_DIR / cfg["subdir"]
    wins = []
    for f in sorted(d.glob("*.parquet")):
        df = pd.read_parquet(f)
        if df.empty: continue
        if "chainlink_btc" in df.columns and "chainlink_price" not in df.columns:
            df.rename(columns={"chainlink_btc": "chainlink_price"}, inplace=True)
        if "window_end_ms" in df.columns:
            if df["ts_ms"].iloc[-1] < df["window_end_ms"].iloc[0]: continue
        else:
            if df["time_remaining_s"].iloc[-1] > 5: continue
        if "window_start_ms" in df.columns and "window_end_ms" in df.columns:
            dur = (df["window_end_ms"].iloc[0] - df["window_start_ms"].iloc[0]) / 1000
            if df["time_remaining_s"].iloc[0] < dur - 30: continue
        sp = df["window_start_price"].dropna()
        if sp.empty: continue
        s, fp = float(sp.iloc[0]), float(df["chainlink_price"].iloc[-1])
        if pd.isna(s) or pd.isna(fp) or s == 0: continue
        wins.append((df, 1 if fp >= s else 0, s, fp))
    wins.sort(key=lambda x: x[0]["ts_ms"].iloc[0])
    return wins

print("Loading...", flush=True)
AW = {mk: load_windows(mk) for mk in MARKETS}

# BLIND PREDICTION
print("\n=== BLIND PREDICTION (Pure GBM, no tuning) ===", flush=True)
for mk in MARKETS:
    ws = MARKETS[mk]["window_s"]
    lbl = MARKETS[mk]["label"]
    for name, frac in [("Mid(50%)", 0.50), ("Late(25%)", 0.25)]:
        tt = ws * frac
        preds = []
        for df, oc, sp, _ in AW[mk]:
            pr = df["chainlink_price"].tolist()
            ts = df["ts_ms"].tolist()
            best_i = None
            best_d = float("inf")
            for i in range(90, len(df)):
                d = abs(df.iloc[i]["time_remaining_s"] - tt)
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i is None: continue
            r = df.iloc[best_i]
            tau = r["time_remaining_s"]
            if tau <= 0: continue
            lo = max(0, best_i - 90)
            sig = cvd(pr[lo:best_i+1], ts[lo:best_i+1])
            if sig <= 0: continue
            z = max(-1.5, min(1.5, ((r["chainlink_price"] - sp) / sp) / (sig * math.sqrt(tau))))
            p = norm_cdf(z)
            preds.append(int((p > 0.5 and oc == 1) or (p < 0.5 and oc == 0)))
        if preds:
            print(f"  {lbl} {name}: {sum(preds)}/{len(preds)} = {sum(preds)/len(preds):.1%}", flush=True)

# OOS FIXED SPLIT
print("\n=== OOS FIXED SPLIT (60/40) ===", flush=True)
for mk in MARKETS:
    windows = AW[mk]
    n = len(windows)
    sp = int(n * 0.6)
    tobs = []
    for df, oc, sx, _ in windows[:sp]:
        pr = df["chainlink_price"].tolist()
        ts = df["ts_ms"].tolist()
        for idx in range(90, len(df), 60):
            tau = df.iloc[idx]["time_remaining_s"]
            if tau <= 0: continue
            sig = cvd(pr[max(0,idx-90):idx+1], ts[max(0,idx-90):idx+1])
            if sig <= 0: continue
            z = max(-1.5, min(1.5, ((df.iloc[idx]["chainlink_price"] - sx) / sx) / (sig * math.sqrt(tau))))
            tobs.append((z, tau, oc))
    TE = [0, 120, 300, 600, 900]
    cell = defaultdict(list)
    for z, tau, o in tobs:
        zb = round(z / 0.5) * 0.5
        ti = next((i for i in range(len(TE)-1) if TE[i] <= tau < TE[i+1]), len(TE)-2)
        cell[(zb, ti)].append(o)
    cal = {k: sum(v)/len(v) for k, v in cell.items()}
    cn = {k: len(v) for k, v in cell.items()}
    bg_list = []
    bf_list = []
    for df, oc, sx, _ in windows[sp:]:
        pr = df["chainlink_price"].tolist()
        ts = df["ts_ms"].tolist()
        for idx in range(90, len(df), 60):
            tau = df.iloc[idx]["time_remaining_s"]
            if tau <= 0: continue
            sig = cvd(pr[max(0,idx-90):idx+1], ts[max(0,idx-90):idx+1])
            if sig <= 0: continue
            z = max(-1.5, min(1.5, ((df.iloc[idx]["chainlink_price"] - sx) / sx) / (sig * math.sqrt(tau))))
            pg = norm_cdf(z)
            zb = round(z / 0.5) * 0.5
            ti = next((i for i in range(len(TE)-1) if TE[i] <= tau < TE[i+1]), len(TE)-2)
            k = (zb, ti)
            if k in cal and cn.get(k, 0) >= 20:
                w = cn[k] / (cn[k] + 100)
                pf = w * cal[k] + (1-w) * pg
            else:
                pf = pg
            bg_list.append((pg - oc) ** 2)
            bf_list.append((pf - oc) ** 2)
    lbl = MARKETS[mk]["label"]
    bg_m = np.mean(bg_list)
    bf_m = np.mean(bf_list)
    verdict = "HELPS" if bf_m < bg_m else "HURTS"
    print(f"  {lbl}: GBM Brier={bg_m:.4f} | Fused={bf_m:.4f} | Cal {verdict} by {abs(bg_m-bf_m):.4f}", flush=True)

# WALK-FORWARD BACKTEST
print("\n=== WALK-FORWARD BACKTEST ===", flush=True)

def fast_wf(windows, ws, et=0.04, em=0.4, mz=1.5, maker=True):
    bank = 10000.0
    results = []
    past = []
    TE = [0, 120, 300, 600, 900] if ws == 900 else [0, 60, 120, 200, 300]
    for wi, (df, oc, sx, _) in enumerate(windows):
        pr = df["chainlink_price"].tolist()
        ts = df["ts_ms"].tolist()
        wobs = []
        for idx in range(90, len(df), 90):
            tau = df.iloc[idx]["time_remaining_s"]
            if tau <= 0: continue
            sig = cvd(pr[max(0,idx-90):idx+1], ts[max(0,idx-90):idx+1])
            if sig <= 0: continue
            z = max(-mz, min(mz, ((df.iloc[idx]["chainlink_price"] - sx) / sx) / (sig * math.sqrt(tau))))
            wobs.append((z, tau, oc))
        if wi < 30:
            past.extend(wobs)
            continue
        cell = defaultdict(list)
        for z, tau, o in past:
            zb = round(z / 0.5) * 0.5
            ti = next((i for i in range(len(TE)-1) if TE[i] <= tau < TE[i+1]), len(TE)-2)
            cell[(zb, ti)].append(o)
        cal_t = {k: sum(v)/len(v) for k, v in cell.items()}
        cn_t = {k: len(v) for k, v in cell.items()}
        target = ws * 0.5
        best_i = min(range(90, len(df)), key=lambda i: abs(df.iloc[i]["time_remaining_s"] - target), default=None)
        if best_i is None:
            past.extend(wobs)
            continue
        r = df.iloc[best_i]
        tau = r["time_remaining_s"]
        if tau <= 0:
            past.extend(wobs)
            continue
        lo = max(0, best_i - 90)
        sig = cvd(pr[lo:best_i+1], ts[lo:best_i+1])
        if sig <= 0:
            past.extend(wobs)
            continue
        z = max(-mz, min(mz, ((r["chainlink_price"] - sx) / sx) / (sig * math.sqrt(tau))))
        pg = norm_cdf(z)
        zb = round(z / 0.5) * 0.5
        ti = next((i for i in range(len(TE)-1) if TE[i] <= tau < TE[i+1]), len(TE)-2)
        k = (zb, ti)
        if k in cal_t and cn_t.get(k, 0) >= 20:
            w = cn_t[k] / (cn_t[k] + 100)
            pm = w * cal_t[k] + (1-w) * pg
        else:
            pm = pg
        bu = r.get("best_bid_up")
        bd = r.get("best_bid_down")
        au = r.get("best_ask_up")
        ad = r.get("best_ask_down")
        if any(pd.isna(x) for x in [bu, au, bd, ad]):
            past.extend(wobs)
            continue
        bu_f = float(bu)
        bd_f = float(bd)
        if bu_f <= 0 or bd_f <= 0:
            past.extend(wobs)
            continue
        dt = et * (1 + em * math.sqrt(tau / ws))
        if maker:
            cu, cd = bu_f, bd_f
        else:
            cu = float(au) + poly_fee(float(au))
            cd = float(ad) + poly_fee(float(ad))
        eu = pm - cu
        ed = (1-pm) - cd
        act = "FLAT"
        edge = 0
        ep = 0
        ps = 0.5
        if eu > dt and eu >= ed:
            act, edge, ep, ps = "UP", eu, cu, pm
        elif ed > dt:
            act, edge, ep, ps = "DN", ed, cd, 1-pm
        if act != "FLAT" and 0 < ep < 1:
            kf = max(0, (ps - ep) / (1 - ep))
            frac = min(0.25 * kf, 0.0125)
            sz = max(bank * frac, min(5*ep, bank*0.02))
            sh = sz / ep
            cost = sh * ep
            won = (act == "UP" and oc == 1) or (act == "DN" and oc == 0)
            pnl = (sh if won else 0) - cost
            bank += pnl
            results.append({"act": act, "won": won, "pnl": pnl, "bank": bank, "edge": edge})
        past.extend(wobs)
    return pd.DataFrame(results) if results else None

for mk in MARKETS:
    r = fast_wf(AW[mk], MARKETS[mk]["window_s"])
    lbl = MARKETS[mk]["label"]
    if r is not None and len(r) > 0:
        nt = len(r)
        nw = int(r["won"].sum())
        print(f"  {lbl}: {nw}/{nt} = {nw/nt:.1%}, PnL=${r['pnl'].sum():+.1f}, bank=${r['bank'].iloc[-1]:,.0f}", flush=True)
        for s in ["UP", "DN"]:
            sm = r[r["act"]==s]
            if len(sm):
                print(f"    {s}: {int(sm['won'].sum())}/{len(sm)} ({sm['won'].mean():.1%}), avg=${sm['pnl'].mean():+.2f}", flush=True)
    else:
        print(f"  {lbl}: no trades", flush=True)

# PARAMETER SWEEP (2 key markets)
print("\n=== PARAMETER SWEEP ===", flush=True)
for mk in ["btc_15m", "eth_15m"]:
    lbl = MARKETS[mk]["label"]
    ws = MARKETS[mk]["window_s"]
    print(f"\n  {lbl} - Edge Threshold:", flush=True)
    for et in [0.02, 0.04, 0.06, 0.08, 0.10, 0.15]:
        r = fast_wf(AW[mk], ws, et=et)
        if r is not None and len(r) > 0:
            print(f"    et={et:.2f}: {len(r):3d} trades, win={r['won'].mean():.1%}, pnl=${r['pnl'].sum():+.1f}", flush=True)
        else:
            print(f"    et={et:.2f}: 0 trades", flush=True)

    print(f"\n  {lbl} - Max Z:", flush=True)
    for mz in [0.8, 1.0, 1.5, 2.0]:
        r = fast_wf(AW[mk], ws, mz=mz)
        if r is not None and len(r) > 0:
            print(f"    mz={mz:.1f}: {len(r):3d} trades, win={r['won'].mean():.1%}, pnl=${r['pnl'].sum():+.1f}", flush=True)
        else:
            print(f"    mz={mz:.1f}: 0 trades", flush=True)

    print(f"\n  {lbl} - Maker vs Taker:", flush=True)
    for mode, ml in [(True, "Maker"), (False, "Taker")]:
        r = fast_wf(AW[mk], ws, maker=mode)
        if r is not None and len(r) > 0:
            print(f"    {ml}: {len(r)} trades, win={r['won'].mean():.1%}, pnl=${r['pnl'].sum():+.1f}", flush=True)
        else:
            print(f"    {ml}: 0 trades", flush=True)

print("\nDONE", flush=True)
