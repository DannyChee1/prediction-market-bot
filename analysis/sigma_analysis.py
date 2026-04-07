"""Compare realized per-second sigma across BTC, ETH, SOL, XRP using Binance klines.

Pulls 1-second klines (or 1-minute if 1s unavailable) and computes
per-second sigma the same way DiffusionSignal._compute_vol does:
  - skip duplicate prices (stale ticks)
  - log-return / sqrt(dt)
  - std(log_rets, ddof=1)

Computes sigma in rolling windows matching the vol_lookback_s (60s default)
and reports distribution stats.
"""

import math
import time
import requests
import numpy as np

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
DISPLAY = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "SOLUSDT": "SOL", "XRPUSDT": "XRP"}

# Use 1s klines, fetch max 1000 at a time
INTERVAL = "1s"
LIMIT = 1000  # max per request
NUM_REQUESTS = 12  # 12 * 1000s ≈ 3.3 hours of 1s data per asset
VOL_WINDOW = 60  # seconds, matches vol_lookback_s default


def fetch_klines(symbol: str, interval: str, limit: int, end_time: int | None = None):
    """Fetch klines from Binance public API."""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time is not None:
        params["endTime"] = end_time
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_all_klines(symbol: str) -> list[tuple[int, float]]:
    """Fetch multiple pages of 1s klines, returns [(ts_ms, close), ...]."""
    all_data = []
    end_time = None
    for i in range(NUM_REQUESTS):
        klines = fetch_klines(symbol, INTERVAL, LIMIT, end_time)
        if not klines:
            break
        for k in klines:
            ts_ms = k[0]       # open time
            close = float(k[4])  # close price
            all_data.append((ts_ms, close))
        # Next page: go backwards
        end_time = klines[0][0] - 1
        time.sleep(0.1)  # rate limit
    all_data.sort(key=lambda x: x[0])
    return all_data


def compute_vol(prices: list[float], timestamps: list[int]) -> float:
    """Exact replica of DiffusionSignal._compute_vol."""
    changes: list[tuple[int, float, int]] = []
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


def rolling_sigmas(data: list[tuple[int, float]], window_s: int) -> list[float]:
    """Compute sigma in rolling windows of window_s seconds."""
    if not data:
        return []
    sigmas = []
    window_ms = window_s * 1000
    # Slide by half-window for overlap
    step_ms = window_ms // 2
    t_start = data[0][0]
    t_end = data[-1][0]

    t = t_start
    while t + window_ms <= t_end:
        # Extract window
        prices = []
        timestamps = []
        for ts, p in data:
            if t <= ts < t + window_ms:
                prices.append(p)
                timestamps.append(ts)
        if len(prices) >= 10:
            s = compute_vol(prices, timestamps)
            if s > 0:
                sigmas.append(s)
        t += step_ms
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
        print(f"Fetching {name} ({sym})...")
        data = fetch_all_klines(sym)
        print(f"  Got {len(data)} ticks spanning "
              f"{(data[-1][0] - data[0][0]) / 3600000:.1f} hours")

        sigmas = rolling_sigmas(data, VOL_WINDOW)
        results[name] = sigmas
        print(f"  {len(sigmas)} rolling windows computed")

    # Print results
    print("\n" + "=" * 80)
    print(f"Per-second sigma distribution ({VOL_WINDOW}s rolling windows)")
    print("=" * 80)
    header = f"{'Asset':<6} {'Mean':>10} {'Median':>10} {'P90':>10} {'P95':>10} {'P99':>10} {'Max':>10} {'Current':>10}"
    print(header)
    print("-" * len(header))

    for name in ["BTC", "ETH", "SOL", "XRP"]:
        s = np.array(results[name])
        cfg = current_configs[name]
        print(f"{name:<6} {s.mean():>10.2e} {np.median(s):>10.2e} "
              f"{np.percentile(s, 90):>10.2e} {np.percentile(s, 95):>10.2e} "
              f"{np.percentile(s, 99):>10.2e} {s.max():>10.2e} {cfg:>10.2e}")

    # Ratios relative to BTC
    print("\n" + "=" * 80)
    print("Ratios relative to BTC")
    print("=" * 80)
    btc_mean = np.mean(results["BTC"])
    btc_p95 = np.percentile(results["BTC"], 95)
    btc_p99 = np.percentile(results["BTC"], 99)
    header2 = f"{'Asset':<6} {'Mean ratio':>12} {'P95 ratio':>12} {'P99 ratio':>12} {'Config ratio':>14}"
    print(header2)
    print("-" * len(header2))
    for name in ["BTC", "ETH", "SOL", "XRP"]:
        s = np.array(results[name])
        cfg_ratio = current_configs[name] / current_configs["BTC"]
        print(f"{name:<6} {np.mean(s)/btc_mean:>12.2f} "
              f"{np.percentile(s, 95)/btc_p95:>12.2f} "
              f"{np.percentile(s, 99)/btc_p99:>12.2f} "
              f"{cfg_ratio:>14.2f}")

    # Suggested max_sigma (P99 with 20% headroom)
    print("\n" + "=" * 80)
    print("Suggested max_sigma (P99 + 20% headroom)")
    print("=" * 80)
    for name in ["BTC", "ETH", "SOL", "XRP"]:
        s = np.array(results[name])
        suggested = np.percentile(s, 99) * 1.2
        current = current_configs[name]
        diff = "OK" if abs(suggested - current) / current < 0.3 else (
            "TOO HIGH" if current > suggested * 1.3 else "TOO LOW"
        )
        print(f"  {name}: {suggested:.2e}  (current: {current:.2e})  {diff}")


if __name__ == "__main__":
    main()
