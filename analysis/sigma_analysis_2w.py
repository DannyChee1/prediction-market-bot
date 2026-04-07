"""2-week realized per-second sigma comparison using Binance 1m klines.

Uses 1-minute close prices. Each log-return is normalized by sqrt(60)
to convert to per-second sigma (same logic as _compute_vol with real timestamps).

Rolling window = 10 ticks (10 minutes) — short enough to capture local vol
regimes, comparable to the bot's 60s lookback on ~1s Chainlink ticks
(both give ~10 unique price changes per window).
"""

import math
import time
import requests
import numpy as np
from datetime import datetime, timezone

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
DISPLAY = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "SOLUSDT": "SOL", "XRPUSDT": "XRP"}

INTERVAL = "1m"
LIMIT = 1000
TWO_WEEKS_MS = 14 * 24 * 60 * 60 * 1000
ROLLING_WINDOW = 10  # ticks (minutes)


def fetch_klines(symbol, interval, limit, start_time=None, end_time=None):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time is not None:
        params["startTime"] = start_time
    if end_time is not None:
        params["endTime"] = end_time
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_2w_klines(symbol):
    """Fetch 2 weeks of 1m klines. Returns [(ts_ms, close), ...]."""
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - TWO_WEEKS_MS
    all_data = []
    cursor = start_ms
    while cursor < now_ms:
        klines = fetch_klines(symbol, INTERVAL, LIMIT, start_time=cursor)
        if not klines:
            break
        for k in klines:
            all_data.append((k[0], float(k[4])))  # open_time, close
        cursor = klines[-1][0] + 60_000  # next minute
        time.sleep(0.05)
    return all_data


def compute_vol_window(prices, timestamps):
    """Per-second sigma from a window of prices with ms timestamps."""
    changes = []
    for i, p in enumerate(prices):
        ts = timestamps[i]
        if p > 0 and (not changes or p != changes[-1][1]):
            changes.append((i, p, ts))
    if len(changes) < 3:
        return 0.0
    log_rets = []
    for j in range(1, len(changes)):
        dt = (changes[j][2] - changes[j - 1][2]) / 1000.0
        if dt > 0:
            lr = math.log(changes[j][1] / changes[j - 1][1])
            log_rets.append(lr / math.sqrt(dt))
    if len(log_rets) < 2:
        return 0.0
    return float(np.std(log_rets, ddof=1))


def rolling_sigmas(data, window):
    """Rolling per-second sigma with given tick window."""
    sigmas = []
    for i in range(len(data) - window):
        chunk = data[i:i + window]
        prices = [c[1] for c in chunk]
        timestamps = [c[0] for c in chunk]
        s = compute_vol_window(prices, timestamps)
        if s > 0:
            sigmas.append(s)
    return sigmas


def main():
    current_configs = {
        "BTC": 8e-05,
        "ETH": 5e-05,
        "SOL": 1.2e-04,
        "XRP": 1.5e-04,
    }

    results = {}
    for sym in SYMBOLS:
        name = DISPLAY[sym]
        print(f"Fetching {name} ({sym}) — 2 weeks of 1m klines...")
        data = fetch_2w_klines(sym)
        t0 = datetime.fromtimestamp(data[0][0] / 1000, tz=timezone.utc)
        t1 = datetime.fromtimestamp(data[-1][0] / 1000, tz=timezone.utc)
        print(f"  {len(data)} candles: {t0:%Y-%m-%d %H:%M} → {t1:%Y-%m-%d %H:%M} UTC")

        sigmas = rolling_sigmas(data, ROLLING_WINDOW)
        results[name] = np.array(sigmas)
        print(f"  {len(sigmas)} sigma windows")

    print("\n" + "=" * 85)
    print(f"Per-second sigma from Binance 1m klines (2 weeks, {ROLLING_WINDOW}-tick rolling)")
    print("=" * 85)
    hdr = f"{'Asset':<6} {'Mean':>10} {'Median':>10} {'P90':>10} {'P95':>10} {'P99':>10} {'Max':>10} {'Cfg':>10} {'%Cap':>6}"
    print(hdr)
    print("-" * len(hdr))

    for name in ["BTC", "ETH", "SOL", "XRP"]:
        s = results[name]
        cfg = current_configs[name]
        pct_cap = (s > cfg).sum() / len(s) * 100
        print(f"{name:<6} {s.mean():>10.2e} {np.median(s):>10.2e} "
              f"{np.percentile(s, 90):>10.2e} {np.percentile(s, 95):>10.2e} "
              f"{np.percentile(s, 99):>10.2e} {s.max():>10.2e} "
              f"{cfg:>10.2e} {pct_cap:>5.1f}%")

    # Ratios
    print("\n" + "=" * 85)
    print("Ratios relative to BTC")
    print("=" * 85)
    btc = results["BTC"]
    hdr2 = f"{'Asset':<6} {'Mean/BTC':>10} {'Med/BTC':>10} {'P95/BTC':>10} {'P99/BTC':>10} {'Cfg/BTC':>10}"
    print(hdr2)
    print("-" * len(hdr2))
    for name in ["BTC", "ETH", "SOL", "XRP"]:
        s = results[name]
        cfg_ratio = current_configs[name] / current_configs["BTC"]
        print(f"{name:<6} {s.mean()/btc.mean():>10.2f} {np.median(s)/np.median(btc):>10.2f} "
              f"{np.percentile(s,95)/np.percentile(btc,95):>10.2f} "
              f"{np.percentile(s,99)/np.percentile(btc,99):>10.2f} "
              f"{cfg_ratio:>10.2f}")

    # Suggestion: match BTC's cap percentile for all assets
    btc_cap_pct = (btc > current_configs["BTC"]).sum() / len(btc) * 100
    print(f"\nBTC caps at P{100-btc_cap_pct:.0f}. Matching that percentile for others:")
    target_pct = 100 - btc_cap_pct
    for name in ["BTC", "ETH", "SOL", "XRP"]:
        s = results[name]
        suggested = np.percentile(s, target_pct)
        current = current_configs[name]
        diff_pct = (suggested - current) / current * 100
        flag = "" if abs(diff_pct) < 20 else (" ← raise" if diff_pct > 0 else " ← lower")
        print(f"  {name}: {suggested:.2e}  (current: {current:.2e}, {diff_pct:+.0f}%){flag}")


if __name__ == "__main__":
    main()
