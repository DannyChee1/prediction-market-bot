"""API helpers: SSL context, CLOB client builder, market lookup, resolution polling."""

from __future__ import annotations

import json
import os
import ssl
import sys
import time as _time
from datetime import datetime, timezone, timedelta

import requests

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds, BalanceAllowanceParams
    _HAS_PY_CLOB = True
except ImportError:
    _HAS_PY_CLOB = False

from market_config import MarketConfig

# ── SSL ──────────────────────────────────────────────────────────────────────
try:
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    SSL_CTX.check_hostname = False
    SSL_CTX.verify_mode = ssl.CERT_NONE

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Endpoints ────────────────────────────────────────────────────────────────
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_HOST = "https://clob.polymarket.com"
CHAIN_ID = 137


# ── CLOB client builder ─────────────────────────────────────────────────────

def build_clob_client() -> ClobClient:
    """Build an authenticated ClobClient from .env credentials."""
    private_key = os.getenv("PRIVATE_KEY")
    if not private_key:
        print("  ERROR: PRIVATE_KEY not set in .env")
        sys.exit(1)

    funder = os.getenv("POLY_FUNDER", "")
    sig_type = int(os.getenv("SIGNATURE_TYPE", "1"))

    if not funder:
        print("  ERROR: POLY_FUNDER not set in .env")
        sys.exit(1)

    client = ClobClient(
        host=CLOB_HOST,
        key=private_key,
        chain_id=CHAIN_ID,
        signature_type=sig_type,
        funder=funder,
    )

    api_key = os.getenv("POLY_API_KEY", "")
    api_secret = os.getenv("POLY_API_SECRET", "")
    passphrase = os.getenv("POLY_PASSPHRASE", "")

    if api_key and api_secret and passphrase:
        client.set_api_creds(ApiCreds(
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=passphrase,
        ))
    else:
        print("  Deriving API credentials...")
        creds = client.create_or_derive_api_creds()
        client.set_api_creds(creds)
        print(f"  Save these in .env:")
        print(f"    POLY_API_KEY={creds.api_key}")
        print(f"    POLY_API_SECRET={creds.api_secret}")
        print(f"    POLY_PASSPHRASE={creds.api_passphrase}")

    return client


def query_usdc_balance(client: ClobClient, debug: bool = False) -> float | None:
    """Fetch current USDC (collateral) balance from the CLOB API."""
    try:
        resp = client.get_balance_allowance(
            BalanceAllowanceParams(asset_type="COLLATERAL")
        )
        raw = float(resp.get("balance", 0))
        if raw > 1_000_000:
            return raw / 1e6
        return raw
    except Exception as exc:
        if debug:
            print(f"  [BALANCE] error: {exc}")
        return None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ensure_list(val):
    return json.loads(val) if isinstance(val, str) else val


def _try_slug(slug: str):
    try:
        resp = requests.get(
            f"{GAMMA_API}/events", params={"slug": slug}, timeout=10
        )
        data = resp.json()
        if data:
            return data[0], data[0]["markets"][0]
    except Exception:
        pass
    return None


def _generate_hourly_slug(config: MarketConfig, target_utc: datetime) -> str:
    """Generate a human-readable slug for 1-hour markets.

    Hourly markets use ET-based slugs like:
      bitcoin-up-or-down-april-9-2026-4pm-et

    The slug_prefix for 1h configs is the asset name part
    (e.g. "bitcoin-up-or-down"), and we append the date + hour in ET.
    """
    from zoneinfo import ZoneInfo
    et = target_utc.astimezone(ZoneInfo("America/New_York"))
    month = et.strftime("%B").lower()  # "april"
    day = et.day
    year = et.year
    hour = et.hour
    if hour == 0:
        time_str = "12am"
    elif hour < 12:
        time_str = f"{hour}am"
    elif hour == 12:
        time_str = "12pm"
    else:
        time_str = f"{hour - 12}pm"
    return f"{config.slug_prefix}-{month}-{day}-{year}-{time_str}-et"


def find_market(config: MarketConfig):
    now = datetime.now(timezone.utc)
    align = config.window_align_m

    # Hourly markets use human-readable ET-based slugs instead of
    # timestamp-based slugs. Detect by window duration.
    is_hourly = config.window_duration_s >= 3600

    if is_hourly:
        # Try current hour, then next, then previous (in case we're
        # near a boundary and the current window just closed).
        hour_start = now.replace(minute=0, second=0, microsecond=0)
        for offset_h in [0, 1, -1, 2]:
            candidate = hour_start + timedelta(hours=offset_h)
            slug = _generate_hourly_slug(config, candidate)
            result = _try_slug(slug)
            if result:
                event, market = result
                end = datetime.fromisoformat(
                    market["endDate"].replace("Z", "+00:00")
                )
                if now < end:
                    return event, market
    else:
        # Original timestamp-based slug search for 5m/15m markets
        minute = (now.minute // align) * align
        window_start = now.replace(minute=minute, second=0, microsecond=0)

        for offset in [0, -align, align, -2 * align, 2 * align]:
            candidate = window_start + timedelta(minutes=offset)
            ts = int(candidate.timestamp())
            result = _try_slug(f"{config.slug_prefix}-{ts}")
            if result:
                event, market = result
                end = datetime.fromisoformat(
                    market["endDate"].replace("Z", "+00:00")
                )
                start = datetime.fromisoformat(
                    market["eventStartTime"].replace("Z", "+00:00")
                )
                if now < end or start > now:
                    return event, market

    # Broad fallback: search active up-or-down markets
    try:
        resp = requests.get(
            f"{GAMMA_API}/events",
            params={
                "active": "true", "closed": "false",
                "tag_slug": "up-or-down", "limit": 100,
            },
            timeout=15,
        )
        for e in resp.json():
            if config.slug_prefix not in e.get("slug", ""):
                continue
            m = e["markets"][0]
            end = datetime.fromisoformat(m["endDate"].replace("Z", "+00:00"))
            if now < end:
                return e, m
    except Exception:
        pass

    return None, None


def poll_market_resolution(slug: str, max_attempts: int = 12,
                           delay: float = 5.0,
                           debug: bool = False) -> int | None:
    """Poll Gamma API until the market is FULLY RESOLVED (not just closed).

    2026-04-09 fix: previously this checked only `closed=True` and read
    `outcomePrices` as a winner indicator (`up_price > 0.5`). But the API
    sets `closed=True` BEFORE the on-chain resolution finalizes, and during
    that gap `outcomePrices` shows the LAST TRADE PRICES (not the
    settlement prices). A market that traded UP at 0.51 just before
    closing would be reported as "UP wins" by the old code, even when the
    actual Chainlink resolution was DOWN. Real loss documented:
    btc-updown-5m-1775709300 (window 04:35-04:40 UTC, 2026-04-09) — bot
    bet DOWN, recorded as UP loss, actual API state now shows DOWN won
    (outcomePrices=["0","1"], umaResolutionStatus="resolved"). Cost: $9.30
    in unredeemed winnings.

    The fix requires THREE conditions before trusting the outcome:
    1. `closed == True`
    2. `umaResolutionStatus == "resolved"` (UMA optimistic oracle confirmed)
    3. `outcomePrices` is exactly one "1" and one "0" (settlement prices)

    If any condition fails, keep polling (up to max_attempts).
    """
    for attempt in range(max_attempts):
        try:
            resp = requests.get(
                f"{GAMMA_API}/events", params={"slug": slug}, timeout=10
            )
            data = resp.json()
            if not data:
                _time.sleep(delay)
                continue
            market = data[0]["markets"][0]
            if not market.get("closed"):
                if debug:
                    print(f"  [RESOLVE] attempt {attempt + 1}: not closed yet")
                _time.sleep(delay)
                continue

            # NEW: require UMA resolution before trusting outcomePrices.
            uma_status = market.get("umaResolutionStatus", "")
            if uma_status != "resolved":
                if debug:
                    print(f"  [RESOLVE] attempt {attempt + 1}: closed but uma_status={uma_status!r} (waiting for finalization)")
                _time.sleep(delay)
                continue

            outcomes = _ensure_list(market["outcomes"])
            outcome_prices = _ensure_list(market["outcomePrices"])
            if len(outcomes) != 2 or len(outcome_prices) != 2:
                if debug:
                    print(f"  [RESOLVE] attempt {attempt + 1}: malformed outcomes/prices")
                _time.sleep(delay)
                continue

            # NEW: require settlement prices to be exactly 0/1.
            # If the market is showing intermediate prices like 0.51/0.49,
            # the on-chain resolution hasn't been applied yet — keep polling.
            try:
                price_strs = [str(p).strip() for p in outcome_prices]
                price_floats = [float(p) for p in outcome_prices]
            except (ValueError, TypeError):
                if debug:
                    print(f"  [RESOLVE] attempt {attempt + 1}: price parse fail: {outcome_prices}")
                _time.sleep(delay)
                continue

            settled = (
                {price_strs[0], price_strs[1]} == {"0", "1"}
                or sorted(price_floats) == [0.0, 1.0]
            )
            if not settled:
                if debug:
                    print(f"  [RESOLVE] attempt {attempt + 1}: prices not settled yet: {outcome_prices}")
                _time.sleep(delay)
                continue

            up_idx = outcomes.index("Up")
            up_price = float(outcome_prices[up_idx])
            return 1 if up_price >= 1.0 else 0
        except Exception as exc:
            if debug:
                print(f"  [RESOLVE] attempt {attempt + 1} error: {exc}")
            _time.sleep(delay)
    return None
