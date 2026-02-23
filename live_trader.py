#!/usr/bin/env python3
"""
Live Trading Bot for Polymarket 15-Min Up/Down Markets

Places real FOK market orders via the Polymarket CLOB API using the
DiffusionSignal from backtest.py. Tracks balance via API and enforces
circuit breakers.

Usage:
    py -3 live_trader.py                          # BTC, $10k bankroll
    py -3 live_trader.py --market eth              # ETH market
    py -3 live_trader.py --bankroll 500            # smaller bankroll
    py -3 live_trader.py --max-loss-pct 5          # stop at 5% loss
    py -3 live_trader.py --dry-run                 # signal + log, no real orders
    py -3 live_trader.py --debug
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import ssl
import sys
import time as _time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import requests
import websockets
from dotenv import load_dotenv

from web3 import Web3
from web3.exceptions import ContractLogicError

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    MarketOrderArgs, OrderArgs, OrderType, ApiCreds, BalanceAllowanceParams,
    OpenOrderParams,
)
from py_clob_client.order_builder.constants import BUY

from recorder import OrderBook
from backtest import (
    Snapshot, Decision, Fill, TradeResult,
    DiffusionSignal, walk_book, poly_fee, BacktestEngine,
    norm_cdf,
)
from market_config import MarketConfig, MARKET_CONFIGS, DEFAULT_MARKET, get_config

load_dotenv()

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

# ── Endpoints & paths ────────────────────────────────────────────────────────
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_HOST = "https://clob.polymarket.com"
CLOB_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
RTDS_WS = "wss://ws-live-data.polymarket.com"
CHAIN_ID = 137

TRADES_LOG = Path("live_trades.jsonl")   # overridden per-market in main()
STATE_FILE = Path("live_state.json")     # overridden per-market in main()

DEBUG = False
DRY_RUN = False
MULTI_MODE = False

# Shared display state for multi-market mode.
# Keys are market names ("btc", "eth"), values are dicts with tracker/price/book state.
MARKET_DISPLAY_STATES: dict[str, dict] = {}

# ── On-chain redemption constants ────────────────────────────────────────────
POLYGON_RPC = os.getenv("POLYGON_RPC", "https://polygon-rpc.com")

CTF_ADDRESS = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
USDC_ADDRESS = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")

CTF_ABI = [
    {
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "indexSets", "type": "uint256[]"},
        ],
        "name": "redeemPositions",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"name": "conditionId", "type": "bytes32"}],
        "name": "payoutDenominator",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]

PROXY_ABI = [
    {
        "inputs": [
            {
                "components": [
                    {"name": "to", "type": "address"},
                    {"name": "value", "type": "uint256"},
                    {"name": "data", "type": "bytes"},
                ],
                "name": "_calls",
                "type": "tuple[]",
            }
        ],
        "name": "proxy",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    }
]

PARENT_COLLECTION_ID = bytes(32)  # 0x0...0


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


def query_usdc_balance(client: ClobClient) -> float | None:
    """Fetch current USDC (collateral) balance from the CLOB API.

    The API returns balance in micro-USDC (6 decimal places),
    so we divide by 1e6 to get USD.
    """
    try:
        resp = client.get_balance_allowance(
            BalanceAllowanceParams(asset_type="COLLATERAL")
        )
        raw = float(resp.get("balance", 0))
        # API returns micro-USDC (6 decimals); convert to USD
        if raw > 1_000_000:  # clearly micro-USDC, not USD
            return raw / 1e6
        return raw
    except Exception as exc:
        if DEBUG:
            print(f"  [BALANCE] error: {exc}")
        return None


# ── Helpers ──────────────────────────────────────────────────────────────────

def fetch_chainlink_price_rest(config: MarketConfig) -> float | None:
    """Fetch current Chainlink price from Polymarket's REST API (fallback for RTDS)."""
    try:
        resp = requests.get(
            "https://data-api.polymarket.com/prices",
            params={"symbols": config.chainlink_symbol},
            timeout=10,
        )
        data = resp.json()
        if isinstance(data, list) and data:
            return float(data[0].get("price", 0))
        if isinstance(data, dict):
            val = data.get(config.chainlink_symbol) or data.get("price")
            if val:
                return float(val)
    except Exception:
        pass
    # Fallback: try CoinGecko
    cg_ids = {"btc": "bitcoin", "eth": "ethereum"}
    cg_id = cg_ids.get(config.data_subdir)
    if cg_id:
        try:
            resp = requests.get(
                f"https://api.coingecko.com/api/v3/simple/price",
                params={"ids": cg_id, "vs_currencies": "usd"},
                timeout=10,
            )
            data = resp.json()
            return float(data[cg_id]["usd"])
        except Exception:
            pass
    return None


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


def find_market(config: MarketConfig):
    now = datetime.now(timezone.utc)
    minute = (now.minute // 15) * 15
    window_start = now.replace(minute=minute, second=0, microsecond=0)

    for offset in [0, -15, 15, -30, 30]:
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

    # Broad fallback
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
                           delay: float = 5.0) -> int | None:
    for attempt in range(max_attempts):
        try:
            resp = requests.get(
                f"{GAMMA_API}/events", params={"slug": slug}, timeout=10
            )
            data = resp.json()
            if not data:
                continue
            market = data[0]["markets"][0]
            if not market.get("closed"):
                if DEBUG:
                    print(f"  [RESOLVE] attempt {attempt + 1}: not closed yet")
                _time.sleep(delay)
                continue
            outcomes = _ensure_list(market["outcomes"])
            outcome_prices = _ensure_list(market["outcomePrices"])
            up_idx = outcomes.index("Up")
            up_price = float(outcome_prices[up_idx])
            return 1 if up_price > 0.5 else 0
        except Exception as exc:
            if DEBUG:
                print(f"  [RESOLVE] attempt {attempt + 1} error: {exc}")
            _time.sleep(delay)
    return None


# ── Live snapshot builder ────────────────────────────────────────────────────

def snapshot_from_live(
    book_up: OrderBook, book_down: OrderBook,
    price: float | None, window_start_price: float | None,
    window_end: datetime, market_slug: str,
) -> Snapshot | None:
    if price is None or window_start_price is None:
        return None

    now = datetime.now(timezone.utc)
    time_remaining_s = max(0.0, (window_end - now).total_seconds())

    ba_up = book_up.best_ask
    ba_down = book_down.best_ask
    if ba_up is None or ba_down is None:
        return None

    bb_up = book_up.best_bid
    bb_down = book_down.best_bid

    return Snapshot(
        ts_ms=int(_time.time() * 1000),
        market_slug=market_slug,
        time_remaining_s=time_remaining_s,
        chainlink_price=price,
        window_start_price=window_start_price,
        best_bid_up=bb_up,
        best_ask_up=ba_up,
        best_bid_down=bb_down,
        best_ask_down=ba_down,
        size_bid_up=book_up.bids.get(bb_up) if bb_up else None,
        size_ask_up=book_up.asks.get(ba_up) if ba_up else None,
        size_bid_down=book_down.bids.get(bb_down) if bb_down else None,
        size_ask_down=book_down.asks.get(ba_down) if ba_down else None,
        ask_levels_up=tuple(book_up.top_asks(5)),
        ask_levels_down=tuple(book_down.top_asks(5)),
        bid_levels_up=tuple(book_up.top_bids(5)),
        bid_levels_down=tuple(book_down.top_bids(5)),
    )


# ── Live Trade Tracker ───────────────────────────────────────────────────────

class LiveTradeTracker:
    """Manages live trading state, order placement, and failsafes."""

    def __init__(
        self,
        client: ClobClient,
        signal: DiffusionSignal,
        initial_bankroll: float,
        latency_ms: int = 0,
        slippage: float = 0.0,
        cooldown_ms: int = 30_000,
        max_loss_pct: float = 50.0,
        max_trades_per_window: int = 1,
        stale_price_timeout_s: float = 60.0,
        min_balance_usd: float = 5.0,
        maker_mode: bool = False,
        trades_log: Path | None = None,
        state_file: Path | None = None,
        market_key: str = "",
    ):
        self.client = client
        self.signal = signal
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.latency_ms = latency_ms
        self.slippage = slippage
        self.cooldown_ms = cooldown_ms
        self.maker_mode = maker_mode
        self.market_key = market_key  # "btc" / "eth"

        # Per-tracker file paths (avoids global dependency)
        self.trades_log: Path = trades_log or TRADES_LOG
        self.state_file: Path = state_file or STATE_FILE

        # Failsafes
        self.max_loss_pct = max_loss_pct
        self.max_trades_per_window = max_trades_per_window
        self.stale_price_timeout_s = stale_price_timeout_s
        self.min_balance_usd = min_balance_usd
        self.circuit_breaker_tripped = False
        self.circuit_breaker_reason = ""

        self.ctx: dict = {}
        self.pending_fills: list[dict] = []  # live fills awaiting resolution
        self.all_results: list[TradeResult] = []
        self.last_fill_ts_ms: int = 0
        self.last_decision: Decision = Decision("FLAT", 0.0, 0.0, "initializing")
        self.last_price_update_ts: float = 0.0
        self.window_trade_count: int = 0
        self.min_order_shares: float = 5.0  # Polymarket min_order_size

        # Session stats
        self.windows_seen: int = 0
        self.windows_traded: int = 0
        self.total_fees: float = 0.0
        self.peak_bankroll: float = initial_bankroll
        self.max_drawdown: float = 0.0
        self.max_dd_pct: float = 0.0

        # API balance tracking
        self.api_balance: float | None = None
        self.last_balance_check_ts: float = 0.0

        # On-chain redemption
        self.condition_id: str = ""

        # Signal diagnostics
        self.flat_reason_counts: dict[str, int] = {}
        self.signal_eval_count: int = 0
        self.last_diag_ts: float = 0.0

        # Maker mode: open limit orders awaiting fill
        self.open_orders: list[dict] = []  # {order_id, side, price, size, token_id, ts}

        # Arb PnL tracking (maker dual-side)
        self.arb_pairs_completed: int = 0
        self.arb_total_pnl: float = 0.0
        self.last_arb_result: str = ""  # human-readable last arb outcome

    def new_window(self, window_end: datetime):
        # Log previous window's FLAT reason summary before resetting
        if self.flat_reason_counts:
            self._log_flat_summary()
        # Cancel any leftover open limit orders from previous window
        self._cancel_open_orders()
        self.ctx = {}
        self.pending_fills = []
        self.last_fill_ts_ms = 0
        self.last_decision = Decision("FLAT", 0.0, 0.0, "new window")
        self.windows_seen += 1
        self.window_trade_count = 0
        self.open_orders = []
        self.flat_reason_counts: dict[str, int] = {}
        self.signal_eval_count = 0
        self.last_diag_ts: float = 0.0

    def _check_circuit_breakers(self) -> str | None:
        """Returns reason string if trading should stop, None if OK."""
        if self.circuit_breaker_tripped:
            return self.circuit_breaker_reason

        # Max session loss
        total_pnl = sum(r.pnl for r in self.all_results)
        loss_pct = abs(total_pnl) / self.initial_bankroll * 100 if total_pnl < 0 else 0
        if loss_pct >= self.max_loss_pct:
            self.circuit_breaker_tripped = True
            self.circuit_breaker_reason = (
                f"CIRCUIT BREAKER: session loss {loss_pct:.1f}% "
                f"(${total_pnl:+.2f}) exceeds {self.max_loss_pct}% limit"
            )
            return self.circuit_breaker_reason

        # Min balance
        if self.bankroll < self.min_balance_usd:
            self.circuit_breaker_tripped = True
            self.circuit_breaker_reason = (
                f"CIRCUIT BREAKER: bankroll ${self.bankroll:.2f} "
                f"below minimum ${self.min_balance_usd:.2f}"
            )
            return self.circuit_breaker_reason

        # Max trades per window
        if self.window_trade_count >= self.max_trades_per_window:
            return (
                f"window trade limit ({self.window_trade_count}"
                f"/{self.max_trades_per_window})"
            )

        return None

    def _check_stale_price(self) -> bool:
        """Returns True if price data is stale."""
        if self.last_price_update_ts == 0:
            return True
        age = _time.time() - self.last_price_update_ts
        # Use a longer timeout (60s) to tolerate RTDS reconnections.
        # The 10s default was too aggressive and caused entire windows to be skipped.
        return age > 60.0

    def evaluate(
        self,
        snapshot: Snapshot,
        up_token: str,
        down_token: str,
    ) -> Decision:
        """Called every 1s: run signal, place real orders if triggered."""
        # Circuit breakers
        cb_reason = self._check_circuit_breakers()
        if cb_reason:
            self.last_decision = Decision("FLAT", 0.0, 0.0, cb_reason)
            return self.last_decision

        # Stale price guard
        if self._check_stale_price():
            self.last_decision = Decision(
                "FLAT", 0.0, 0.0,
                f"stale price ({_time.time() - self.last_price_update_ts:.0f}s old)"
            )
            return self.last_decision

        # Maker mode: dual-side market making (own cooldown / polling)
        if self.maker_mode:
            return self._evaluate_dual_maker(snapshot, up_token, down_token)

        # Cooldown
        if (self.last_fill_ts_ms > 0
                and snapshot.ts_ms - self.last_fill_ts_ms < self.cooldown_ms):
            self.last_decision = Decision(
                "FLAT", 0.0, 0.0,
                f"cooldown ({(snapshot.ts_ms - self.last_fill_ts_ms) / 1000:.0f}s"
                f" / {self.cooldown_ms / 1000:.0f}s)")
            return self.last_decision

        # Run signal
        decision = self.signal.decide(snapshot, self.ctx)
        self.last_decision = decision
        self.signal_eval_count += 1

        if decision.action == "FLAT":
            # Bucket the reason for end-of-window summary
            reason = decision.reason
            # Normalize to category (strip variable numbers)
            if reason.startswith("need "):
                cat = "warmup"
            elif reason.startswith("no edge"):
                cat = "no_edge"
            elif "momentum fail" in reason:
                cat = "momentum_fail"
            elif "spread too wide" in reason:
                cat = "spread_wide"
            elif "imbalance" in reason:
                cat = "imbalance"
            elif "delta velocity" in reason:
                cat = "delta_velocity"
            elif "vol spike" in reason:
                cat = "vol_spike"
            elif "zero vol" in reason:
                cat = "zero_vol"
            elif "missing book" in reason:
                cat = "missing_book"
            elif "kelly" in reason:
                cat = "kelly_zero"
            else:
                cat = reason[:30]
            self.flat_reason_counts[cat] = self.flat_reason_counts.get(cat, 0) + 1

            # Periodic diagnostic snapshot (every 60s)
            now = _time.time()
            if now - self.last_diag_ts >= 60:
                self.last_diag_ts = now
                self._log_diagnostic(snapshot, decision)
        else:
            self._place_order(snapshot, decision, up_token, down_token)

        return decision

    # ── Dual-side maker strategy ────────────────────────────────────────

    def _evaluate_dual_maker(self, snapshot, up_token, down_token):
        """Dual-side market making: limit orders on BOTH Up and Down.

        With 0% maker fee and mid-price execution:
        - Arb component: guaranteed profit when mid_up + mid_down < 1.0
        - Directional lean: more shares on the model-favored side
        """
        # Build price history (same tracking as signal.decide)
        hist = self.ctx.setdefault("price_history", [])
        hist.append(snapshot.chainlink_price)
        self.signal_eval_count += 1

        if len(hist) < self.signal.vol_lookback_s:
            reason = f"need {self.signal.vol_lookback_s}s history ({len(hist)})"
            self.flat_reason_counts["warmup"] = self.flat_reason_counts.get("warmup", 0) + 1
            self.last_decision = Decision("FLAT", 0.0, 0.0, reason)
            return self.last_decision

        bid_up = snapshot.best_bid_up
        ask_up = snapshot.best_ask_up
        bid_down = snapshot.best_bid_down
        ask_down = snapshot.best_ask_down
        if any(x is None or x <= 0 for x in [bid_up, ask_up, bid_down, ask_down]):
            self.flat_reason_counts["missing_book"] = self.flat_reason_counts.get("missing_book", 0) + 1
            self.last_decision = Decision("FLAT", 0.0, 0.0, "missing book")
            return self.last_decision

        tau = snapshot.time_remaining_s
        if tau <= 0:
            self.last_decision = Decision("FLAT", 0.0, 0.0, "window expired")
            return self.last_decision

        # Poll existing open orders for fills
        if self.open_orders:
            self._poll_open_orders(snapshot)

        # If we already have orders on both sides, just wait for fills
        existing_sides = {o["side"] for o in self.open_orders}
        if "UP" in existing_sides and "DOWN" in existing_sides:
            mid_up = (bid_up + ask_up) / 2.0
            mid_down = (bid_down + ask_down) / 2.0
            arb = 1.0 - (mid_up + mid_down)
            self.last_decision = Decision("FLAT", 0.0, 0.0,
                f"dual orders active (arb={arb:+.3f})")
            now = _time.time()
            if now - self.last_diag_ts >= 60:
                self.last_diag_ts = now
                self._log_diagnostic(snapshot, self.last_decision)
            return self.last_decision

        # Trade limit check
        if self.window_trade_count >= self.max_trades_per_window:
            reason = f"window trade limit ({self.window_trade_count}/{self.max_trades_per_window})"
            self.last_decision = Decision("FLAT", 0.0, 0.0, reason)
            return self.last_decision

        # Cooldown between order pairs
        if (self.last_fill_ts_ms > 0
                and snapshot.ts_ms - self.last_fill_ts_ms < self.cooldown_ms):
            self.last_decision = Decision("FLAT", 0.0, 0.0,
                f"cooldown ({(snapshot.ts_ms - self.last_fill_ts_ms) / 1000:.0f}s"
                f" / {self.cooldown_ms / 1000:.0f}s)")
            return self.last_decision

        # Compute realized vol
        sigma = self.signal._compute_vol(hist[-self.signal.vol_lookback_s:])
        if sigma == 0:
            self.flat_reason_counts["zero_vol"] = self.flat_reason_counts.get("zero_vol", 0) + 1
            self.last_decision = Decision("FLAT", 0.0, 0.0, "zero vol")
            return self.last_decision

        # Vol regime filter
        if len(hist) >= self.signal.vol_regime_lookback_s:
            sigma_bl = self.signal._compute_vol(
                hist[-self.signal.vol_regime_lookback_s:])
            if sigma_bl > 0 and sigma > self.signal.vol_regime_mult * sigma_bl:
                self.flat_reason_counts["vol_spike"] = self.flat_reason_counts.get("vol_spike", 0) + 1
                self.last_decision = Decision("FLAT", 0.0, 0.0,
                    f"vol spike ({sigma:.2e} > "
                    f"{self.signal.vol_regime_mult}x {sigma_bl:.2e})")
                return self.last_decision

        # Model probability
        delta = snapshot.chainlink_price - snapshot.window_start_price
        z_raw = delta / (sigma * math.sqrt(tau) * snapshot.chainlink_price)
        z = max(-self.signal.max_z, min(self.signal.max_z, z_raw))
        p_model = norm_cdf(z)
        if self.signal.reversion_discount > 0:
            p_model = (p_model * (1 - self.signal.reversion_discount)
                       + 0.5 * self.signal.reversion_discount)

        # Mid prices for limit orders
        mid_up = round((bid_up + ask_up) / 2.0, 2)
        mid_down = round((bid_down + ask_down) / 2.0, 2)

        # Stay below asks (maker, not crossing the spread)
        if mid_up >= ask_up:
            mid_up = round(ask_up - 0.01, 2)
        if mid_down >= ask_down:
            mid_down = round(ask_down - 0.01, 2)
        if mid_up <= 0 or mid_down <= 0 or mid_up >= 1.0 or mid_down >= 1.0:
            self.last_decision = Decision("FLAT", 0.0, 0.0, "invalid mid prices")
            return self.last_decision

        # Spread gate
        spread_up = ask_up - bid_up
        spread_down = ask_down - bid_down
        if spread_up > self.signal.max_spread or spread_down > self.signal.max_spread:
            self.flat_reason_counts["spread_wide"] = self.flat_reason_counts.get("spread_wide", 0) + 1
            self.last_decision = Decision("FLAT", 0.0, 0.0,
                f"spread too wide (up={spread_up:.3f} down={spread_down:.3f})")
            return self.last_decision

        arb_spread = 1.0 - (mid_up + mid_down)

        # Only place orders if arb spread is positive (guaranteed profit)
        if arb_spread <= 0:
            self.flat_reason_counts["no_arb"] = self.flat_reason_counts.get("no_arb", 0) + 1
            self.last_decision = Decision("FLAT", 0.0, 0.0,
                f"no arb (mid_up+mid_down={mid_up + mid_down:.3f} >= 1.0)")
            return self.last_decision

        # Pure arb: equal shares on both sides for guaranteed profit
        alloc_per_side = self.bankroll * self.signal.max_bet_fraction
        max_mid = max(mid_up, mid_down)
        shares = round(alloc_per_side / max_mid, 1)

        min_sh = self.min_order_shares
        if shares < min_sh:
            shares = min_sh

        up_shares = shares
        down_shares = shares

        up_cost = up_shares * mid_up
        down_cost = down_shares * mid_down
        total_cost = up_cost + down_cost

        if total_cost > self.bankroll:
            scale = (self.bankroll - 0.02) / total_cost
            up_shares = round(up_shares * scale, 1)
            down_shares = round(down_shares * scale, 1)
            if up_shares < min_sh or down_shares < min_sh:
                self.last_decision = Decision("FLAT", 0.0, 0.0,
                    f"bankroll ${self.bankroll:.2f} < min dual order")
                return self.last_decision
            up_cost = up_shares * mid_up
            down_cost = down_shares * mid_down
            total_cost = up_cost + down_cost

        arb_profit = arb_spread * shares
        reason = (
            f"arb: spread={arb_spread:+.4f} "
            f"{shares:.0f}sh @{mid_up}/{mid_down} "
            f"=${arb_profit:.3f}"
        )

        # Place both limit orders (equal shares, pure arb)
        up_ok = self._place_one_limit(
            "UP", up_token, mid_up, shares, snapshot, reason)
        dn_ok = self._place_one_limit(
            "DOWN", down_token, mid_down, shares, snapshot, reason)

        if up_ok or dn_ok:
            self.last_fill_ts_ms = snapshot.ts_ms  # cooldown starts after pair

        self.last_decision = Decision(
            "BUY_UP",
            arb_spread, total_cost, reason,
        )
        return self.last_decision

    def _place_one_limit(self, side_label, token_id, price, shares,
                         snapshot, reason):
        """Place a single GTC limit order (used by dual-side maker)."""
        cost_est = round(shares * price, 2)
        now_iso = datetime.now(timezone.utc).isoformat()

        trade_record = {
            "type": "limit_order",
            "ts": now_iso,
            "market_slug": snapshot.market_slug,
            "side": side_label,
            "price": price,
            "shares": shares,
            "cost_est": cost_est,
            "chainlink_price": round(snapshot.chainlink_price, 2),
            "time_remaining_s": round(snapshot.time_remaining_s, 1),
            "signal_reason": reason,
            "bankroll_before": round(self.bankroll, 2),
            "dual_side": True,
        }

        if DRY_RUN:
            trade_record["dry_run"] = True
            trade_record["status"] = "dry_run"
            self._log(trade_record)
            print(
                f"\n  [DRY RUN] LIMIT BUY {side_label} "
                f"{shares:.1f}sh @ {price:.4f} (~${cost_est:.2f})"
            )
            self.open_orders.append({
                "order_id": f"dry_{side_label}_{snapshot.ts_ms}",
                "side": side_label, "price": price, "size": shares,
                "token_id": token_id, "ts": now_iso,
                "market_slug": snapshot.market_slug,
                "time_remaining_s": snapshot.time_remaining_s,
                "chainlink_price": snapshot.chainlink_price,
                "window_start_price": snapshot.window_start_price,
            })
            self.bankroll -= cost_est
            self.signal.bankroll = self.bankroll
            self.window_trade_count += 1
            self.ctx["window_trade_count"] = self.window_trade_count
            self.pending_fills.append({
                "market_slug": snapshot.market_slug,
                "side": side_label, "cost_usd": cost_est, "shares": shares,
                "order_id": f"dry_{side_label}",
                "entry_ts": now_iso,
                "time_remaining_s": snapshot.time_remaining_s,
                "chainlink_price": snapshot.chainlink_price,
                "window_start_price": snapshot.window_start_price,
            })
            return True

        try:
            order_args = OrderArgs(
                token_id=token_id, price=price, size=shares, side=BUY,
            )
            signed_order = self.client.create_order(order_args)
            resp = self.client.post_order(signed_order, OrderType.GTC)
        except Exception as exc:
            trade_record["status"] = "error"
            trade_record["error"] = str(exc)
            self._log(trade_record)
            print(f"\n  [LIMIT ERROR] {side_label}: {exc}")
            return False

        success = resp.get("success", False)
        order_id = resp.get("orderID") or resp.get("id", "")
        status = resp.get("status", "unknown")
        trade_record["order_id"] = order_id
        trade_record["status"] = status
        trade_record["success"] = success

        if not success:
            err_msg = resp.get("errorMsg", "")
            self._log(trade_record)
            print(
                f"\n  [LIMIT] {side_label} {shares:.1f}sh @ {price:.4f} "
                f"-> REJECTED ({err_msg})")
            return False

        self._log(trade_record)
        self.open_orders.append({
            "order_id": order_id,
            "side": side_label, "price": price, "size": shares,
            "token_id": token_id, "ts": now_iso,
            "market_slug": snapshot.market_slug,
            "time_remaining_s": snapshot.time_remaining_s,
            "chainlink_price": snapshot.chainlink_price,
            "window_start_price": snapshot.window_start_price,
        })
        self.bankroll -= cost_est
        self.signal.bankroll = self.bankroll
        self.window_trade_count += 1
        self.ctx["window_trade_count"] = self.window_trade_count

        print(
            f"\n  [LIMIT] BUY {side_label} {shares:.1f}sh @ {price:.4f} "
            f"(~${cost_est:.2f}) | order={order_id[:12]}... "
            f"| Bankroll: ${self.bankroll:,.2f}"
        )
        return True

    def _place_order(
        self,
        snapshot: Snapshot,
        decision: Decision,
        up_token: str,
        down_token: str,
    ):
        """Place a real FOK market order on the CLOB."""
        if decision.action == "BUY_UP":
            side_label = "UP"
            token_id = up_token
        elif decision.action == "BUY_DOWN":
            side_label = "DOWN"
            token_id = down_token
        else:
            return

        amount_usd = round(decision.size_usd, 2)
        if amount_usd <= 0:
            return

        # Ensure order meets Polymarket's minimum (5 shares)
        if decision.action == "BUY_UP":
            ask_px = snapshot.best_ask_up
        else:
            ask_px = snapshot.best_ask_down
        min_usd = 5.0 * ask_px if ask_px and ask_px > 0 else 1.0
        if amount_usd < min_usd:
            if self.bankroll >= min_usd:
                amount_usd = round(min_usd, 2)
            else:
                self._log({
                    "type": "skip",
                    "reason": f"bankroll ${self.bankroll:.2f} < min order ${min_usd:.2f} (5 shares)",
                })
                return

        # Don't exceed bankroll
        if amount_usd > self.bankroll:
            amount_usd = round(self.bankroll - 0.01, 2)
            if amount_usd <= 0:
                return

        now_iso = datetime.now(timezone.utc).isoformat()
        trade_record = {
            "type": "order",
            "ts": now_iso,
            "market_slug": snapshot.market_slug,
            "side": side_label,
            "amount_usd": amount_usd,
            "chainlink_price": round(snapshot.chainlink_price, 2),
            "window_start_price": round(snapshot.window_start_price, 2),
            "time_remaining_s": round(snapshot.time_remaining_s, 1),
            "signal_reason": decision.reason,
            "bankroll_before": round(self.bankroll, 2),
        }

        if DRY_RUN:
            trade_record["dry_run"] = True
            trade_record["status"] = "dry_run"
            self._log(trade_record)
            print(
                f"\n  [DRY RUN] Would place: BUY {side_label} "
                f"${amount_usd:.2f} on {snapshot.market_slug}"
            )
            return

        # Place FOK market order
        try:
            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=amount_usd,
                side=BUY,
            )
            signed_order = self.client.create_market_order(order_args)
            resp = self.client.post_order(signed_order, OrderType.FOK)
        except Exception as exc:
            trade_record["status"] = "error"
            trade_record["error"] = str(exc)
            self._log(trade_record)
            print(f"\n  [ORDER ERROR] {exc}")
            # Cooldown after error to prevent API spam
            self.last_fill_ts_ms = snapshot.ts_ms
            return

        success = resp.get("success", False)
        order_id = resp.get("orderID") or resp.get("id", "")
        status = resp.get("status", "unknown")

        trade_record["order_id"] = order_id
        trade_record["status"] = status
        trade_record["success"] = success
        trade_record["response"] = resp

        if not success or status not in ("matched", "live"):
            # FOK didn't fill — trigger cooldown to avoid retrying immediately
            trade_record["filled"] = False
            self._log(trade_record)
            err_msg = resp.get("errorMsg", "")
            print(
                f"\n  [ORDER] {side_label} ${amount_usd:.2f} -> "
                f"NOT FILLED (status={status}, err={err_msg})"
            )
            self.last_fill_ts_ms = snapshot.ts_ms
            return

        # Order filled
        taking = resp.get("takingAmount", "")
        making = resp.get("makingAmount", "")

        # For FOK: makingAmount = USDC spent, takingAmount = shares received
        cost_usd = float(making) if making else amount_usd
        shares = float(taking) if taking else 0.0

        trade_record["filled"] = True
        trade_record["cost_usd"] = round(cost_usd, 6)
        trade_record["shares"] = round(shares, 6)
        self._log(trade_record)

        # Update tracking
        self.bankroll -= cost_usd
        self.signal.bankroll = self.bankroll
        self.last_fill_ts_ms = snapshot.ts_ms
        self.window_trade_count += 1
        self.ctx["window_trade_count"] = self.window_trade_count

        # Estimate fee for display (poly_fee on best ask)
        best_ask = (snapshot.best_ask_up if side_label == "UP"
                    else snapshot.best_ask_down)
        fee_est = poly_fee(best_ask) * shares if best_ask else 0.0
        self.total_fees += fee_est

        self.pending_fills.append({
            "market_slug": snapshot.market_slug,
            "side": side_label,
            "cost_usd": cost_usd,
            "shares": shares,
            "order_id": order_id,
            "entry_ts": now_iso,
            "time_remaining_s": snapshot.time_remaining_s,
            "chainlink_price": snapshot.chainlink_price,
            "window_start_price": snapshot.window_start_price,
        })

        entry_price = cost_usd / shares if shares > 0 else 0.0
        print(
            f"\n  [FILLED] BUY {side_label} ${cost_usd:.2f} "
            f"-> {shares:.1f} shares @ {entry_price:.4f} "
            f"| Bankroll: ${self.bankroll:,.2f}"
        )

    def _place_limit_order(
        self,
        snapshot: Snapshot,
        decision: Decision,
        up_token: str,
        down_token: str,
    ):
        """Place a GTC limit order at mid-price (maker, 0% fee)."""
        if decision.action == "BUY_UP":
            side_label = "UP"
            token_id = up_token
            best_bid = snapshot.best_bid_up
            best_ask = snapshot.best_ask_up
        elif decision.action == "BUY_DOWN":
            side_label = "DOWN"
            token_id = down_token
            best_bid = snapshot.best_bid_down
            best_ask = snapshot.best_ask_down
        else:
            return

        if best_bid is None or best_ask is None or best_bid <= 0 or best_ask <= 0:
            return

        # Limit price = mid-price, rounded to tick (0.01)
        mid_price = round((best_bid + best_ask) / 2.0, 2)
        if mid_price <= 0 or mid_price >= 1.0:
            return
        # Ensure we stay at or below the ask (maker, not taker)
        if mid_price >= best_ask:
            mid_price = round(best_ask - 0.01, 2)
        if mid_price <= 0:
            return

        # Size in shares (not USD)
        shares = round(decision.size_usd / mid_price, 1)
        if shares < self.min_order_shares:
            if self.bankroll >= self.min_order_shares * mid_price:
                shares = self.min_order_shares
            else:
                self._log({
                    "type": "skip",
                    "reason": f"bankroll ${self.bankroll:.2f} < min order "
                              f"${self.min_order_shares * mid_price:.2f} (5 shares)",
                })
                return

        cost_est = shares * mid_price
        if cost_est > self.bankroll:
            shares = round((self.bankroll - 0.01) / mid_price, 1)
            if shares < self.min_order_shares:
                return
            cost_est = shares * mid_price

        now_iso = datetime.now(timezone.utc).isoformat()
        trade_record = {
            "type": "limit_order",
            "ts": now_iso,
            "market_slug": snapshot.market_slug,
            "side": side_label,
            "price": mid_price,
            "shares": shares,
            "cost_est": round(cost_est, 2),
            "bid": best_bid,
            "ask": best_ask,
            "chainlink_price": round(snapshot.chainlink_price, 2),
            "time_remaining_s": round(snapshot.time_remaining_s, 1),
            "signal_reason": decision.reason,
            "bankroll_before": round(self.bankroll, 2),
        }

        if DRY_RUN:
            trade_record["dry_run"] = True
            trade_record["status"] = "dry_run"
            self._log(trade_record)
            print(
                f"\n  [DRY RUN] Would place: LIMIT BUY {side_label} "
                f"{shares:.1f}sh @ {mid_price:.4f} (~${cost_est:.2f}) "
                f"on {snapshot.market_slug}"
            )
            # Simulate fill for dry run
            self.bankroll -= cost_est
            self.signal.bankroll = self.bankroll
            self.last_fill_ts_ms = snapshot.ts_ms
            self.window_trade_count += 1
            self.ctx["window_trade_count"] = self.window_trade_count
            self.pending_fills.append({
                "market_slug": snapshot.market_slug,
                "side": side_label,
                "cost_usd": cost_est,
                "shares": shares,
                "order_id": "dry_run",
                "entry_ts": now_iso,
                "time_remaining_s": snapshot.time_remaining_s,
                "chainlink_price": snapshot.chainlink_price,
                "window_start_price": snapshot.window_start_price,
            })
            return

        # Place GTC limit order
        try:
            order_args = OrderArgs(
                token_id=token_id,
                price=mid_price,
                size=shares,
                side=BUY,
            )
            signed_order = self.client.create_order(order_args)
            resp = self.client.post_order(signed_order, OrderType.GTC)
        except Exception as exc:
            trade_record["status"] = "error"
            trade_record["error"] = str(exc)
            self._log(trade_record)
            print(f"\n  [LIMIT ORDER ERROR] {exc}")
            self.last_fill_ts_ms = snapshot.ts_ms
            return

        success = resp.get("success", False)
        order_id = resp.get("orderID") or resp.get("id", "")
        status = resp.get("status", "unknown")

        trade_record["order_id"] = order_id
        trade_record["status"] = status
        trade_record["success"] = success
        trade_record["response"] = resp

        if not success:
            trade_record["filled"] = False
            self._log(trade_record)
            err_msg = resp.get("errorMsg", "")
            print(
                f"\n  [LIMIT] {side_label} {shares:.1f}sh @ {mid_price:.4f} -> "
                f"REJECTED (status={status}, err={err_msg})"
            )
            self.last_fill_ts_ms = snapshot.ts_ms
            return

        self._log(trade_record)

        # Track open order for polling
        self.open_orders.append({
            "order_id": order_id,
            "side": side_label,
            "price": mid_price,
            "size": shares,
            "token_id": token_id,
            "ts": now_iso,
            "market_slug": snapshot.market_slug,
            "time_remaining_s": snapshot.time_remaining_s,
            "chainlink_price": snapshot.chainlink_price,
            "window_start_price": snapshot.window_start_price,
        })

        # Reserve bankroll for this order
        self.bankroll -= cost_est
        self.signal.bankroll = self.bankroll
        self.last_fill_ts_ms = snapshot.ts_ms
        self.window_trade_count += 1
        self.ctx["window_trade_count"] = self.window_trade_count

        print(
            f"\n  [LIMIT] BUY {side_label} {shares:.1f}sh @ {mid_price:.4f} "
            f"(~${cost_est:.2f}) | order={order_id[:12]}... "
            f"| Bankroll: ${self.bankroll:,.2f}"
        )

    def _poll_open_orders(self, snapshot: Snapshot):
        """Check open limit orders for fills, move filled ones to pending_fills."""
        still_open = []
        for order in self.open_orders:
            try:
                order_data = self.client.get_order(order["order_id"])
            except Exception as exc:
                if DEBUG:
                    print(f"  [POLL] error checking {order['order_id'][:12]}...: {exc}")
                still_open.append(order)
                continue

            status = order_data.get("status", "").upper()
            size_matched = float(order_data.get("size_matched", 0))

            if status == "MATCHED" or size_matched >= order["size"] * 0.99:
                # Fully filled
                cost_usd = size_matched * order["price"]
                self.pending_fills.append({
                    "market_slug": order["market_slug"],
                    "side": order["side"],
                    "cost_usd": cost_usd,
                    "shares": size_matched,
                    "order_id": order["order_id"],
                    "entry_ts": order["ts"],
                    "time_remaining_s": order["time_remaining_s"],
                    "chainlink_price": order["chainlink_price"],
                    "window_start_price": order["window_start_price"],
                })
                # Adjust bankroll: we reserved cost_est but actual cost may differ
                cost_est = order["size"] * order["price"]
                actual_cost = cost_usd
                drift = cost_est - actual_cost
                if abs(drift) > 0.01:
                    self.bankroll += drift
                    self.signal.bankroll = self.bankroll

                self.total_fees += 0.0  # maker = 0 fee

                entry_price = cost_usd / size_matched if size_matched > 0 else 0
                print(
                    f"\n  [FILLED] LIMIT {order['side']} "
                    f"${cost_usd:.2f} -> {size_matched:.1f}sh @ {entry_price:.4f} "
                    f"| Bankroll: ${self.bankroll:,.2f}"
                )
                self._log({
                    "type": "limit_fill",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "order_id": order["order_id"],
                    "side": order["side"],
                    "price": order["price"],
                    "size_matched": size_matched,
                    "cost_usd": round(cost_usd, 2),
                })
            elif status in ("CANCELLED", "EXPIRED"):
                # Order cancelled/expired — refund reserved bankroll
                cost_est = order["size"] * order["price"]
                self.bankroll += cost_est
                self.signal.bankroll = self.bankroll
                print(
                    f"\n  [CANCELLED] {order['side']} {order['size']:.1f}sh "
                    f"@ {order['price']:.4f} | refunded ${cost_est:.2f}"
                )
            else:
                # Still live
                still_open.append(order)

        self.open_orders = still_open

    def _cancel_open_orders(self):
        """Cancel all open limit orders (called at window end / shutdown)."""
        if DRY_RUN or not self.open_orders:
            return
        for order in self.open_orders:
            try:
                self.client.cancel(order["order_id"])
                # Refund reserved bankroll
                cost_est = order["size"] * order["price"]
                self.bankroll += cost_est
                self.signal.bankroll = self.bankroll
                print(
                    f"  [CANCEL] {order['side']} {order['size']:.1f}sh "
                    f"@ {order['price']:.4f} | refunded ${cost_est:.2f}"
                )
            except Exception as exc:
                print(f"  [CANCEL] error: {exc}")
        self.open_orders = []

    def resolve_window(
        self, slug: str, final_price: float | None,
        window_start_price: float | None,
    ):
        """At window close: poll for resolution, compute PnL, verify balance."""
        if not self.pending_fills:
            return

        print("\n  Polling Gamma API for market resolution...")
        outcome_up = poll_market_resolution(slug)

        if outcome_up is None:
            if final_price is not None and window_start_price is not None:
                outcome_up = 1 if final_price >= window_start_price else 0
                print(
                    f"  WARNING: API resolution unavailable, using local "
                    f"prices (start=${window_start_price:,.2f} "
                    f"final=${final_price:,.2f})"
                )
            else:
                print("  ERROR: Cannot resolve window")
                for fill in self.pending_fills:
                    self.bankroll += fill["cost_usd"]
                self.pending_fills = []
                return

        self.windows_traded += 1
        window_pnl = 0.0
        outcome_str = "UP" if outcome_up else "DOWN"

        for fill in self.pending_fills:
            won = ((fill["side"] == "UP" and outcome_up == 1) or
                   (fill["side"] == "DOWN" and outcome_up == 0))
            payout = fill["shares"] if won else 0.0
            pnl = payout - fill["cost_usd"]
            pnl_pct = pnl / fill["cost_usd"] if fill["cost_usd"] > 0 else 0.0

            # Payout is auto-settled by Polymarket — USDC returns to wallet
            self.bankroll += payout
            window_pnl += pnl

            # Build a TradeResult for session stats
            fake_fill = Fill(
                market_slug=fill["market_slug"],
                side=fill["side"],
                entry_ts_ms=int(_time.time() * 1000),
                time_remaining_s=fill["time_remaining_s"],
                entry_price=fill["cost_usd"] / fill["shares"] if fill["shares"] > 0 else 0,
                fee_per_share=0.0,
                shares=fill["shares"],
                cost_usd=fill["cost_usd"],
                signal_name="diffusion",
                decision_reason="",
            )
            result = TradeResult(
                fill=fake_fill,
                outcome_up=outcome_up,
                final_btc=final_price or 0.0,
                payout=payout,
                pnl=pnl,
                pnl_pct=pnl_pct,
            )
            self.all_results.append(result)

            tag = "WON" if pnl > 0 else "LOST"
            print(
                f"    {fill['side']} ${fill['cost_usd']:.2f} "
                f"-> {fill['shares']:.1f}sh -> {tag} ${pnl:+.2f}"
            )

            self._log({
                "type": "resolution",
                "ts": datetime.now(timezone.utc).isoformat(),
                "market_slug": fill["market_slug"],
                "side": fill["side"],
                "outcome": outcome_str,
                "cost_usd": round(fill["cost_usd"], 2),
                "shares": round(fill["shares"], 2),
                "payout": round(payout, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 4),
                "bankroll_after": round(self.bankroll, 2),
            })

        # Redeem winning CTF positions on-chain (background thread so we
        # don't block trading — on-chain resolution can lag API by 20+ min)
        has_winning = any(
            (f["side"] == "UP" and outcome_up == 1) or
            (f["side"] == "DOWN" and outcome_up == 0)
            for f in self.pending_fills
        )
        if has_winning and self.condition_id:
            import threading
            cond_id = self.condition_id
            print(f"  [REDEEM] Starting background redemption "
                  f"(conditionId: {cond_id[:10]}...)")
            t = threading.Thread(
                target=self.redeem_positions, args=(cond_id,),
                daemon=True)
            t.start()
        elif has_winning and not self.condition_id:
            print("  WARNING: Won but no conditionId — cannot auto-redeem. "
                  "Claim manually on Polymarket.")

        # Update drawdown
        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll
        dd = self.peak_bankroll - self.bankroll
        dd_pct = dd / self.peak_bankroll if self.peak_bankroll > 0 else 0
        if dd > self.max_drawdown:
            self.max_drawdown = dd
            self.max_dd_pct = dd_pct

        self.signal.bankroll = self.bankroll

        # Arb pair summary: detect matched UP+DOWN pairs
        up_fills = [f for f in self.pending_fills if f["side"] == "UP"]
        down_fills = [f for f in self.pending_fills if f["side"] == "DOWN"]
        if up_fills and down_fills:
            total_cost = sum(f["cost_usd"] for f in self.pending_fills)
            total_payout = sum(
                f["shares"] for f in self.pending_fills
                if (f["side"] == "UP" and outcome_up == 1)
                or (f["side"] == "DOWN" and outcome_up == 0)
            )
            arb_net = total_payout - total_cost
            n_pairs = min(len(up_fills), len(down_fills))
            self.arb_pairs_completed += n_pairs
            self.arb_total_pnl += arb_net
            self.last_arb_result = (
                f"{outcome_str} | {n_pairs} pair(s) | "
                f"cost=${total_cost:.2f} payout=${total_payout:.2f} "
                f"net=${arb_net:+.4f}"
            )
            self._log({
                "type": "arb_summary",
                "ts": datetime.now(timezone.utc).isoformat(),
                "market_slug": slug,
                "outcome": outcome_str,
                "n_pairs": n_pairs,
                "total_cost": round(total_cost, 4),
                "winning_payout": round(total_payout, 4),
                "arb_net_pnl": round(arb_net, 4),
                "arb_cumulative_pnl": round(self.arb_total_pnl, 4),
                "arb_pairs_total": self.arb_pairs_completed,
                "bankroll_after": round(self.bankroll, 2),
            })
            print(
                f"  Arb: {n_pairs} pair(s) | Cost: ${total_cost:.2f} | "
                f"Payout: ${total_payout:.2f} | Net: ${arb_net:+.4f} | "
                f"Cumulative: ${self.arb_total_pnl:+.4f}"
            )
        else:
            # Single-side fill (no arb guarantee)
            self.last_arb_result = (
                f"{outcome_str} | single-side only | PnL=${window_pnl:+.2f}"
            )

        print(
            f"  Window resolved: {outcome_str} | "
            f"Window PnL: ${window_pnl:+.2f} | "
            f"Bankroll: ${self.bankroll:,.2f}"
        )

        self.pending_fills = []

    def check_api_balance(self):
        """Periodically verify our balance matches the API."""
        now = _time.time()
        if now - self.last_balance_check_ts < 30:
            return
        self.last_balance_check_ts = now

        api_bal = query_usdc_balance(self.client)
        if api_bal is not None:
            self.api_balance = api_bal
            drift = abs(api_bal - self.bankroll)
            if drift > 1.0 and len(self.pending_fills) == 0:
                # Only warn when no pending fills (fills cause temp drift)
                print(
                    f"\n  [BALANCE DRIFT] API=${api_bal:,.2f} vs "
                    f"tracked=${self.bankroll:,.2f} (drift=${drift:.2f})"
                )

    def cancel_all_orders(self):
        """Cancel all open orders — called on shutdown."""
        if DRY_RUN:
            return
        # Refund any tracked open limit orders
        self._cancel_open_orders()
        try:
            resp = self.client.cancel_all()
            print(f"  Cancelled all open orders: {resp}")
        except Exception as exc:
            print(f"  Warning: cancel_all failed: {exc}")

    def redeem_positions(self, condition_id: str,
                         max_retries: int = 3) -> str | None:
        """Call CTF.redeemPositions() on-chain to convert winning tokens to USDC.

        On-chain reportPayouts() can lag the API by 20+ minutes. We poll
        payoutDenominator(conditionId) (free eth_call) before sending a tx
        to avoid wasting gas on reverts.
        """
        if DRY_RUN:
            print("  [REDEEM] Skipped (dry run)")
            return None

        if not condition_id:
            print("  [REDEEM] Skipped (no conditionId)")
            return None

        # Poll on-chain resolution before spending gas (up to 30 min)
        print(f"  [REDEEM] Waiting for on-chain resolution...")
        condition_bytes = bytes.fromhex(condition_id.replace("0x", ""))
        resolved_on_chain = False
        for poll in range(60):  # 60 polls * 30s = 30 minutes max
            try:
                w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
                ctf = w3.eth.contract(address=CTF_ADDRESS, abi=CTF_ABI)
                denom = ctf.functions.payoutDenominator(condition_bytes).call()
                if denom > 0:
                    resolved_on_chain = True
                    print(f"  [REDEEM] On-chain resolution confirmed (poll {poll + 1})")
                    break
            except Exception as exc:
                if DEBUG:
                    print(f"  [REDEEM] Poll error: {exc}")
            if poll < 59:
                _time.sleep(30)

        if not resolved_on_chain:
            print(f"  [REDEEM] On-chain resolution not found after 30min. "
                  f"Claim manually on Polymarket.")
            self._log({
                "type": "redemption",
                "ts": datetime.now(timezone.utc).isoformat(),
                "condition_id": condition_id,
                "status": "timeout_no_onchain_resolution",
            })
            return None

        # Now attempt the actual redemption tx
        for attempt in range(max_retries):
            if attempt > 0:
                print(f"  [REDEEM] Retry {attempt + 1}/{max_retries} in 10s...")
                _time.sleep(10)

            result = self._try_redeem_once(condition_id)
            if result is not None:
                return result

        print(f"  [REDEEM] Failed after {max_retries} attempts. "
              f"Claim manually on Polymarket.")
        return None

    def _try_redeem_once(self, condition_id: str) -> str | None:
        """Single attempt at on-chain CTF redemption."""
        try:
            w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
            if not w3.is_connected():
                print("  [REDEEM] ERROR: Cannot connect to Polygon RPC")
                return None

            private_key = os.getenv("PRIVATE_KEY", "")
            if not private_key:
                print("  [REDEEM] ERROR: PRIVATE_KEY not set")
                return None

            signer = w3.eth.account.from_key(private_key)
            sig_type = int(os.getenv("SIGNATURE_TYPE", "1"))
            funder = os.getenv("POLY_FUNDER", "")

            # Check gas balance
            pol_balance = w3.eth.get_balance(signer.address)
            if pol_balance < w3.to_wei(0.001, "ether"):
                print(f"  [REDEEM] ERROR: Insufficient POL for gas "
                      f"({w3.from_wei(pol_balance, 'ether'):.6f} POL). "
                      f"Send POL to {signer.address}")
                return None

            # Encode CTF.redeemPositions(USDC, 0x0, conditionId, [1, 2])
            ctf = w3.eth.contract(address=CTF_ADDRESS, abi=CTF_ABI)
            condition_bytes = bytes.fromhex(condition_id.replace("0x", ""))
            redeem_data = ctf.encode_abi(
                "redeemPositions",
                args=[
                    USDC_ADDRESS,
                    PARENT_COLLECTION_ID,
                    condition_bytes,
                    [1, 2],  # indexSets: both outcomes
                ],
            )

            # Convert encode_abi output to raw bytes reliably.
            # web3.py may return hex string ("0x...") or HexBytes.
            if isinstance(redeem_data, str):
                redeem_bytes = bytes.fromhex(
                    redeem_data[2:] if redeem_data.startswith("0x")
                    else redeem_data
                )
            else:
                redeem_bytes = bytes(redeem_data)

            if sig_type == 0:
                # EOA: send tx directly to CTF contract
                tx = {
                    "to": CTF_ADDRESS,
                    "data": redeem_data,
                    "from": signer.address,
                    "nonce": w3.eth.get_transaction_count(signer.address),
                    "gas": 200_000,
                    "maxFeePerGas": w3.eth.gas_price * 2,
                    "maxPriorityFeePerGas": w3.to_wei(30, "gwei"),
                    "chainId": CHAIN_ID,
                }
            else:
                # Proxy wallet: wrap in proxy([ProxyCall])
                proxy_addr = Web3.to_checksum_address(funder)
                proxy_contract = w3.eth.contract(address=proxy_addr, abi=PROXY_ABI)
                proxy_call_data = proxy_contract.encode_abi(
                    "proxy",
                    args=[[(CTF_ADDRESS, 0, redeem_bytes)]],
                )
                tx = {
                    "to": proxy_addr,
                    "data": proxy_call_data,
                    "from": signer.address,
                    "nonce": w3.eth.get_transaction_count(signer.address),
                    "gas": 300_000,
                    "maxFeePerGas": w3.eth.gas_price * 2,
                    "maxPriorityFeePerGas": w3.to_wei(30, "gwei"),
                    "chainId": CHAIN_ID,
                }

            print(f"  [REDEEM] Sending tx (sig_type={sig_type}, "
                  f"conditionId={condition_id[:10]}...)...")
            signed_tx = w3.eth.account.sign_transaction(tx, private_key)
            tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            tx_hash_hex = tx_hash.hex()

            print(f"  [REDEEM] Tx sent: {tx_hash_hex}")
            print(f"  [REDEEM] Waiting for confirmation...")

            receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
            status = receipt.get("status", 0)

            if status == 1:
                gas_used = receipt.get("gasUsed", 0)
                print(f"  [REDEEM] Confirmed! Gas used: {gas_used}")
                self._log({
                    "type": "redemption",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "tx_hash": tx_hash_hex,
                    "condition_id": condition_id,
                    "gas_used": gas_used,
                    "status": "success",
                })
                return tx_hash_hex
            else:
                print(f"  [REDEEM] Tx reverted (hash: {tx_hash_hex}) — "
                      f"market may not be resolved on-chain yet")
                self._log({
                    "type": "redemption",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "tx_hash": tx_hash_hex,
                    "condition_id": condition_id,
                    "status": "reverted",
                })
                return None

        except ContractLogicError as exc:
            print(f"  [REDEEM] Contract reverted: {exc}")
            self._log({
                "type": "redemption",
                "ts": datetime.now(timezone.utc).isoformat(),
                "condition_id": condition_id,
                "status": "contract_error",
                "error": str(exc),
            })
            return None

        except Exception as exc:
            print(f"  [REDEEM] ERROR: {type(exc).__name__}: {exc}")
            self._log({
                "type": "redemption",
                "ts": datetime.now(timezone.utc).isoformat(),
                "condition_id": condition_id,
                "status": "error",
                "error": str(exc),
            })
            return None

    def _log_diagnostic(self, snapshot: Snapshot, decision: Decision):
        """Log a signal diagnostic snapshot every 60s for post-hoc analysis."""
        hist = self.ctx.get("price_history", [])
        sigma_per_s = self.signal._compute_vol(hist[-self.signal.vol_lookback_s:]) if len(hist) >= self.signal.vol_lookback_s else 0.0
        tau = snapshot.time_remaining_s
        window_trade_count = self.ctx.get("window_trade_count", 0)
        base_for_trade = self.signal.edge_threshold + self.signal.edge_threshold_step * window_trade_count
        dyn_threshold = base_for_trade * (
            1.0 + self.signal.early_edge_mult * math.sqrt(tau / self.signal.window_duration)
        ) if tau > 0 else base_for_trade

        # Compute edges (same as decide())
        ask_up = snapshot.best_ask_up
        ask_down = snapshot.best_ask_down
        bid_up = snapshot.best_bid_up
        bid_down = snapshot.best_bid_down
        edge_up = edge_down = p_model = 0.0
        spread_up = spread_down = 0.0

        if (ask_up and ask_down and bid_up and bid_down
                and 0 < ask_up < 1 and 0 < ask_down < 1
                and sigma_per_s > 0 and tau > 0):
            spread_up = ask_up - bid_up
            spread_down = ask_down - bid_down
            delta = snapshot.chainlink_price - snapshot.window_start_price
            z_raw = delta / (sigma_per_s * math.sqrt(tau) * snapshot.chainlink_price)
            from backtest import norm_cdf
            z = max(-self.signal.max_z, min(self.signal.max_z, z_raw))
            p_model = norm_cdf(z)
            if self.maker_mode:
                mid_up = (bid_up + ask_up) / 2.0
                mid_down = (bid_down + ask_down) / 2.0
                p_up_cost = mid_up
                p_down_cost = mid_down
                arb_spread = 1.0 - (mid_up + mid_down)
            else:
                p_up_cost = ask_up + poly_fee(ask_up) + self.signal.slippage
                p_down_cost = ask_down + poly_fee(ask_down) + self.signal.slippage
                arb_spread = None
            edge_up = p_model - p_up_cost - self.signal.spread_edge_penalty * spread_up
            edge_down = (1.0 - p_model) - p_down_cost - self.signal.spread_edge_penalty * spread_down

        diag = {
            "type": "diagnostic",
            "ts": datetime.now(timezone.utc).isoformat(),
            "market_slug": snapshot.market_slug,
            "tau": round(tau, 0),
            "chainlink_price": round(snapshot.chainlink_price, 2),
            "window_start_price": round(snapshot.window_start_price, 2),
            "delta": round(snapshot.chainlink_price - snapshot.window_start_price, 2),
            "sigma_per_s": f"{sigma_per_s:.2e}",
            "p_model": round(p_model, 4),
            "ask_up": ask_up,
            "ask_down": ask_down,
            "spread_up": round(spread_up, 4) if spread_up else None,
            "spread_down": round(spread_down, 4) if spread_down else None,
            "edge_up": round(edge_up, 4),
            "edge_down": round(edge_down, 4),
            "dyn_threshold": round(dyn_threshold, 4),
            "best_edge": round(max(edge_up, edge_down), 4),
            "edge_gap": round(max(edge_up, edge_down) - dyn_threshold, 4),
            "reason": decision.reason,
            "hist_len": len(hist),
            "evals": self.signal_eval_count,
        }
        if arb_spread is not None:
            diag["arb_spread"] = round(arb_spread, 4)
            diag["mode"] = "maker_dual"
        self._log(diag)

    def _log_flat_summary(self):
        """Log end-of-window summary of FLAT reason distribution."""
        total = sum(self.flat_reason_counts.values())
        if total == 0:
            return
        # Sort by count descending
        sorted_reasons = sorted(self.flat_reason_counts.items(),
                                key=lambda x: -x[1])
        summary = {k: v for k, v in sorted_reasons}
        # Print to console
        print(f"\n  -- Signal Diagnostic (window #{self.windows_seen}) --")
        print(f"  Total evaluations: {self.signal_eval_count}  |  FLAT: {total}  |  Trades: {self.window_trade_count}")
        for reason, count in sorted_reasons[:5]:
            pct = count / total * 100
            print(f"    {reason:25s}  {count:4d}  ({pct:5.1f}%)")

        self._log({
            "type": "flat_summary",
            "ts": datetime.now(timezone.utc).isoformat(),
            "window_num": self.windows_seen,
            "total_evals": self.signal_eval_count,
            "total_flat": total,
            "trades": self.window_trade_count,
            "reasons": summary,
        })

    def _log(self, record: dict):
        try:
            with open(self.trades_log, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def save_state(self):
        wins = [r for r in self.all_results if r.pnl > 0]
        data = {
            "bankroll": round(self.bankroll, 2),
            "initial_bankroll": round(self.initial_bankroll, 2),
            "windows_seen": self.windows_seen,
            "windows_traded": self.windows_traded,
            "total_trades": len(self.all_results),
            "wins": len(wins),
            "losses": len(self.all_results) - len(wins),
            "total_pnl": round(sum(r.pnl for r in self.all_results), 2),
            "total_fees": round(self.total_fees, 2),
            "peak_bankroll": round(self.peak_bankroll, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_dd_pct": round(self.max_dd_pct, 4),
            "api_balance": round(self.api_balance, 2) if self.api_balance else None,
            "circuit_breaker": self.circuit_breaker_tripped,
            "arb_pairs": self.arb_pairs_completed,
            "arb_pnl": round(self.arb_total_pnl, 4),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    @classmethod
    def load_state(cls, path: Path | None = None) -> dict | None:
        p = path or STATE_FILE
        if not p.exists():
            return None
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None


# ── WebSocket Handlers ───────────────────────────────────────────────────────

async def clob_ws(
    up_token: str, down_token: str,
    book_up: OrderBook, book_down: OrderBook,
    flat_state: dict, cancel: asyncio.Event,
):
    token_map = {up_token: "up", down_token: "down"}
    book_map = {"up": book_up, "down": book_down}
    backoff = 2

    while not cancel.is_set():
        try:
            async with websockets.connect(
                CLOB_WS, ssl=SSL_CTX, ping_interval=None
            ) as ws:
                backoff = 2
                await ws.send(json.dumps({
                    "assets_ids": [up_token, down_token],
                    "type": "market",
                    "custom_feature_enabled": True,
                }))

                async def heartbeat():
                    try:
                        while not cancel.is_set():
                            await asyncio.sleep(10)
                            await ws.send("PING")
                    except Exception:
                        pass

                hb = asyncio.create_task(heartbeat())
                try:
                    async for raw in ws:
                        if cancel.is_set():
                            break
                        if raw == "PONG" or not raw:
                            continue
                        if DEBUG:
                            t = datetime.now().strftime("%H:%M:%S")
                            print(f"\n  [CLOB {t}] {raw[:300]}")

                        try:
                            payload = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        msgs = payload if isinstance(payload, list) else [payload]
                        for msg in msgs:
                            if not isinstance(msg, dict):
                                continue
                            etype = msg.get("event_type")
                            asset_id = msg.get("asset_id")
                            side = token_map.get(asset_id)

                            if etype == "book" and side:
                                book_map[side].on_snapshot(
                                    msg.get("bids", []), msg.get("asks", []))
                                bb = book_map[side].best_bid
                                ba = book_map[side].best_ask
                                flat_state[f"{side}_best_bid"] = str(bb) if bb else None
                                flat_state[f"{side}_best_ask"] = str(ba) if ba else None

                            elif etype == "price_change":
                                for ch in msg.get("price_changes", []):
                                    s = token_map.get(ch.get("asset_id"))
                                    if s:
                                        book_map[s].on_price_change(
                                            ch["price"], ch["size"], ch["side"])
                                        bb = book_map[s].best_bid
                                        ba = book_map[s].best_ask
                                        flat_state[f"{s}_best_bid"] = str(bb) if bb else None
                                        flat_state[f"{s}_best_ask"] = str(ba) if ba else None

                finally:
                    hb.cancel()

        except Exception as exc:
            if cancel.is_set():
                return
            if DEBUG:
                print(f"\n  [CLOB] error: {type(exc).__name__}: {exc}")
            await asyncio.sleep(min(backoff, 30))
            backoff = min(backoff * 2, 60)


async def rtds_ws(price_state: dict, cancel: asyncio.Event,
                  config: MarketConfig, tracker: LiveTradeTracker):
    backoff = 2

    while not cancel.is_set():
        try:
            async with websockets.connect(
                RTDS_WS, ssl=SSL_CTX, ping_interval=None
            ) as ws:
                backoff = 2
                await ws.send(json.dumps({
                    "action": "subscribe",
                    "subscriptions": [{
                        "topic": "crypto_prices_chainlink",
                        "type": "*",
                    }],
                }))
                print(f"  [RTDS] connected, filtering for {config.chainlink_symbol}")

                async def heartbeat():
                    try:
                        while not cancel.is_set():
                            await asyncio.sleep(5)
                            await ws.send("PING")
                    except Exception:
                        pass

                hb = asyncio.create_task(heartbeat())
                try:
                    async for raw in ws:
                        if cancel.is_set():
                            break
                        if raw == "PONG" or not raw:
                            continue

                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        payload = msg.get("payload", {})
                        symbol = payload.get("symbol")
                        if symbol is None or symbol != config.chainlink_symbol:
                            continue

                        data_arr = payload.get("data")
                        if isinstance(data_arr, list) and data_arr:
                            p = data_arr[-1].get("value")
                            if p is not None:
                                price_state["price"] = float(p)
                                tracker.last_price_update_ts = _time.time()
                                if price_state["window_start_price"] is None:
                                    price_state["window_start_price"] = float(p)
                            continue

                        p = payload.get("value")
                        if p is not None:
                            price_state["price"] = float(p)
                            tracker.last_price_update_ts = _time.time()
                            if price_state["window_start_price"] is None:
                                price_state["window_start_price"] = float(p)

                finally:
                    hb.cancel()

        except Exception as exc:
            if cancel.is_set():
                return
            print(f"\n  [RTDS] reconnecting: {type(exc).__name__}: {exc}")
            await asyncio.sleep(min(backoff, 30))
            backoff = min(backoff * 2, 60)


# ── Signal Ticker ────────────────────────────────────────────────────────────

async def signal_ticker(
    tracker: LiveTradeTracker,
    book_up: OrderBook, book_down: OrderBook,
    price_state: dict, window_end: datetime,
    market_slug: str, up_token: str, down_token: str,
    cancel: asyncio.Event,
    skip_trading: bool = False,
):
    while not cancel.is_set():
        await asyncio.sleep(1)
        if cancel.is_set():
            break

        snap = snapshot_from_live(
            book_up, book_down,
            price_state.get("price"),
            price_state.get("window_start_price"),
            window_end, market_slug,
        )
        if snap is not None and not skip_trading:
            tracker.evaluate(snap, up_token, down_token)

        # Periodic balance check
        tracker.check_api_balance()


# ── Display ──────────────────────────────────────────────────────────────────

def render_display(
    tracker: LiveTradeTracker,
    price_state: dict,
    flat_state: dict,
    market_title: str,
    window_start: datetime,
    window_end: datetime,
    config: MarketConfig,
):
    lines = ["\033[2J\033[H"]
    mode = "DRY RUN" if DRY_RUN else "LIVE"
    lines.append("=" * 62)
    lines.append(f"  [{mode}] {config.display_name} Up/Down 15m: {market_title}")
    lines.append("=" * 62)
    lines.append("")

    now = datetime.now(timezone.utc)
    remaining = (window_end - now).total_seconds()

    if remaining > 0:
        m, s = int(remaining // 60), int(remaining % 60)
        bar_len = 30
        filled = max(0, min(bar_len, int((remaining / 900) * bar_len)))
        lines.append(
            f"  Time Remaining:  {m:02d}:{s:02d}  "
            f"[{'#' * filled}{'-' * (bar_len - filled)}]"
        )
    else:
        lines.append("  Time Remaining:  EXPIRED  (resolving...)")

    lines.append(
        f"  Window:          {window_start.strftime('%H:%M:%S UTC')}"
        f" -> {window_end.strftime('%H:%M:%S UTC')}"
    )
    lines.append("")

    price = price_state.get("price")
    start_px = price_state.get("window_start_price")
    price_label = f"Chainlink {config.chainlink_symbol.upper()}"
    if price is not None:
        age = _time.time() - tracker.last_price_update_ts
        stale_tag = f"  STALE {age:.0f}s" if age > tracker.stale_price_timeout_s else ""
        lines.append(f"  {price_label}:  ${price:>12,.2f}{stale_tag}")
        if start_px is not None:
            delta = price - start_px
            lines.append(
                f"  Start Price:{' ' * (len(price_label) - 11)}${start_px:>12,.2f}"
                f"  (delta: ${delta:+,.2f})"
            )
    else:
        lines.append(f"  {price_label}:     waiting...")
    lines.append("")

    # Book table
    def fp(key):
        v = flat_state.get(key)
        if v is None:
            return "   ---   "
        try:
            return f"  {float(v):.4f}  "
        except (ValueError, TypeError):
            return f"  {str(v):>7}  "

    lines.append("  +----------+-----------+-----------+")
    lines.append("  | Outcome  |  Best Bid |  Best Ask |")
    lines.append("  +----------+-----------+-----------+")
    lines.append(f"  |    Up    |{fp('up_best_bid')}|{fp('up_best_ask')}|")
    lines.append(f"  |   Down   |{fp('down_best_bid')}|{fp('down_best_ask')}|")
    lines.append("  +----------+-----------+-----------+")
    lines.append("")

    # Trading section
    lines.append(f"  -- {mode} Trading (DiffusionSignal) " + "-" * 20)

    dec = tracker.last_decision
    status = dec.action if dec.action != "FLAT" else "FLAT"

    # Balance line
    bal_str = f"${tracker.bankroll:,.2f}"
    if tracker.api_balance is not None:
        bal_str += f"  (API: ${tracker.api_balance:,.2f})"
    lines.append(f"  Bankroll: {bal_str}  |  Status: {status}")
    lines.append(f"  Reason:   {dec.reason[:60]}")

    # Show top FLAT reason distribution this window
    if tracker.flat_reason_counts and tracker.signal_eval_count > 0:
        top = sorted(tracker.flat_reason_counts.items(), key=lambda x: -x[1])
        top_str = "  |  ".join(f"{k}:{v}" for k, v in top[:3])
        lines.append(f"  FLAT dist: {top_str}")

    # Circuit breaker warning
    if tracker.circuit_breaker_tripped:
        lines.append(f"  *** {tracker.circuit_breaker_reason} ***")

    # Price history info
    hist = tracker.ctx.get("price_history", [])
    hist_len = len(hist)
    vol_str = ""
    if hist_len >= 20:
        recent = hist[-20:]
        log_ret = [
            math.log(recent[i] / recent[i - 1])
            for i in range(1, len(recent))
            if recent[i - 1] > 0 and recent[i] > 0
        ]
        if len(log_ret) >= 2:
            vol_str = f"  |  Vol(20s): {float(np.std(log_ret, ddof=1)):.2e}"
    lines.append(f"  History:  {hist_len}s{vol_str}")
    lines.append("")

    # Open limit orders (maker mode)
    if tracker.open_orders:
        for order in tracker.open_orders:
            lines.append(
                f"  Open Order:   {order['side']} "
                f"{order['size']:.1f}sh @ {order['price']:.4f} "
                f"(~${order['size'] * order['price']:.2f})"
            )

    # Current window fills
    if tracker.pending_fills:
        for fill in tracker.pending_fills:
            entry_px = fill["cost_usd"] / fill["shares"] if fill["shares"] > 0 else 0
            rem_m = int(fill["time_remaining_s"]) // 60
            rem_s = int(fill["time_remaining_s"]) % 60
            lines.append(
                f"  This Window:  {fill['side']} @ {entry_px:.4f} "
                f"x {fill['shares']:.1f}sh ${fill['cost_usd']:.2f} "
                f"[{rem_m}:{rem_s:02d} left]"
            )
    else:
        lines.append("  This Window:  no trades")

    # Last result
    if tracker.maker_mode and tracker.last_arb_result:
        lines.append(f"  Last Arb:     {tracker.last_arb_result}")
    elif tracker.all_results:
        r = tracker.all_results[-1]
        tag = "WON" if r.pnl > 0 else "LOST"
        lines.append(
            f"  Last Result:  {r.fill.side} "
            f"${r.fill.cost_usd:.2f} -> {tag} ${r.pnl:+.2f}"
        )
    lines.append("")

    # Session stats
    total_pnl = sum(r.pnl for r in tracker.all_results)
    if tracker.maker_mode:
        lines.append(
            f"  Session:  {tracker.windows_traded}/{tracker.windows_seen}"
            f" windows  |  Arb pairs: {tracker.arb_pairs_completed}"
        )
        lines.append(
            f"            Arb PnL: ${tracker.arb_total_pnl:+,.4f}"
            f"  |  Total PnL: ${total_pnl:+,.2f}"
            f"  |  DD: ${tracker.max_drawdown:.0f} ({tracker.max_dd_pct:.1%})"
        )
    else:
        wins = [r for r in tracker.all_results if r.pnl > 0]
        total = len(tracker.all_results)
        win_count = len(wins)
        win_str = f"{win_count}/{total} ({win_count / total:.0%})" if total > 0 else "---"
        lines.append(
            f"  Session:  {tracker.windows_traded}/{tracker.windows_seen}"
            f" windows traded  |  Win: {win_str}"
        )
        lines.append(
            f"            PnL: ${total_pnl:+,.2f}"
            f"  |  Fees: ~${tracker.total_fees:.2f}"
            f"  |  DD: ${tracker.max_drawdown:.0f} ({tracker.max_dd_pct:.1%})"
        )
    lines.append("")
    lines.append("  Ctrl+C to exit (cancels open orders)")

    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


async def display_ticker(
    tracker: LiveTradeTracker,
    price_state: dict,
    flat_state: dict,
    market_title: str,
    window_start: datetime,
    window_end: datetime,
    cancel: asyncio.Event,
    config: MarketConfig,
):
    while not cancel.is_set():
        render_display(
            tracker, price_state, flat_state,
            market_title, window_start, window_end, config,
        )
        await asyncio.sleep(1)


# ── Multi-Market Display ────────────────────────────────────────────────────

def render_multi_display():
    """Render a combined display for all active markets."""
    lines = ["\033[2J\033[H"]
    mode = "DRY RUN" if DRY_RUN else "LIVE"
    lines.append("=" * 62)
    lines.append(f"  [{mode}] Arb Bot  |  Maker (0% fee)")
    lines.append("=" * 62)

    total_arb_pnl = 0.0
    total_bankroll = 0.0
    total_pairs = 0

    for key in sorted(MARKET_DISPLAY_STATES):
        state = MARKET_DISPLAY_STATES[key]
        tracker = state["tracker"]
        price_state = state["price_state"]
        flat_state = state["flat_state"]
        config = state["config"]
        window_end = state["window_end"]

        now = datetime.now(timezone.utc)
        remaining = (window_end - now).total_seconds()

        price = price_state.get("price")
        price_str = f"${price:>12,.2f}" if price else "waiting..."

        if remaining > 0:
            m, s = int(remaining // 60), int(remaining % 60)
            time_str = f"{m:02d}:{s:02d}"
        else:
            time_str = "RESOLVING"

        lines.append("")
        lines.append(f"  -- {config.display_name} --  {price_str}  |  Time: {time_str}")

        # Book (compact)
        def fp(k):
            v = flat_state.get(k)
            return f"{float(v):.2f}" if v else "---"

        bid_up = fp("up_best_bid")
        ask_up = fp("up_best_ask")
        bid_down = fp("down_best_bid")
        ask_down = fp("down_best_ask")
        lines.append(f"  UP {bid_up}/{ask_up}  |  DOWN {bid_down}/{ask_down}")

        # Status
        dec = tracker.last_decision
        lines.append(f"  {dec.reason[:55]}")

        # Open orders
        if tracker.open_orders:
            parts = []
            for o in tracker.open_orders:
                parts.append(f"{o['side']} {o['size']:.0f}@{o['price']:.2f}")
            lines.append(f"  Orders: {' | '.join(parts)}")

        # Pending fills
        if tracker.pending_fills:
            parts = []
            for f in tracker.pending_fills:
                parts.append(f"{f['side']} {f['shares']:.0f}sh ${f['cost_usd']:.2f}")
            lines.append(f"  Fills:  {' | '.join(parts)}")

        # Per-market stats
        lines.append(
            f"  Pairs: {tracker.arb_pairs_completed}  |  "
            f"Arb PnL: ${tracker.arb_total_pnl:+.4f}  |  "
            f"Bankroll: ${tracker.bankroll:,.2f}"
        )

        total_arb_pnl += tracker.arb_total_pnl
        total_bankroll += tracker.bankroll
        total_pairs += tracker.arb_pairs_completed

    lines.append("")
    lines.append("-" * 62)
    lines.append(
        f"  TOTAL:  {total_pairs} arb pairs  |  "
        f"Arb PnL: ${total_arb_pnl:+.4f}  |  "
        f"Bankroll: ${total_bankroll:,.2f}"
    )
    lines.append("")
    lines.append("  Ctrl+C to exit (cancels open orders)")

    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


async def multi_display_ticker(cancel: asyncio.Event):
    """Single display ticker that renders all active markets."""
    while not cancel.is_set():
        if MARKET_DISPLAY_STATES:
            render_multi_display()
        await asyncio.sleep(1)


# ── Window Lifecycle ─────────────────────────────────────────────────────────

async def run_window(tracker: LiveTradeTracker, config: MarketConfig,
                     price_state: dict, show_display: bool = True):
    """Run a single 15-minute trading window.

    price_state is a shared dict continuously updated by the persistent RTDS
    websocket. At window start we snapshot the current live price as the
    window_start_price, which is much more accurate than carrying a stale
    price from after the previous window's RTDS disconnected.
    """
    print(f"  Searching for active {config.display_name} 15-minute market...")
    event, market = find_market(config)

    if not event or not market:
        print(f"  No active {config.display_name} 15-minute market found. Retrying in 30s...")
        await asyncio.sleep(30)
        return

    outcomes = _ensure_list(market["outcomes"])
    tokens = _ensure_list(market["clobTokenIds"])
    up_token = tokens[outcomes.index("Up")]
    down_token = tokens[outcomes.index("Down")]

    end = datetime.fromisoformat(market["endDate"].replace("Z", "+00:00"))
    start = datetime.fromisoformat(
        market["eventStartTime"].replace("Z", "+00:00")
    )
    slug = event["slug"]
    title = event["title"]

    # Capture conditionId for on-chain redemption
    tracker.condition_id = market.get("conditionId", "")
    if not tracker.condition_id:
        print("  WARNING: conditionId missing from market data — auto-redeem disabled for this window")

    # Note: CLOB API returns neg_risk=true for all tokens (routing flag),
    # but these binary Up/Down markets use standard CTF redemption.

    book_up = OrderBook()
    book_down = OrderBook()

    # Get the window start price. Priority:
    # 1. Live RTDS price (most accurate, ~1-2s old)
    # 2. REST API fallback (if RTDS hasn't connected yet)
    current_price = price_state.get("price")
    price_source = "RTDS"

    if current_price is None:
        # RTDS not connected yet — wait up to 10s for it
        print(f"  Waiting for RTDS price feed...")
        for _ in range(10):
            await asyncio.sleep(1)
            current_price = price_state.get("price")
            if current_price is not None:
                break

    if current_price is None:
        # Still no RTDS — fetch from REST API
        print(f"  RTDS unavailable, fetching price from REST API...")
        current_price = fetch_chainlink_price_rest(config)
        price_source = "REST API"
        if current_price is not None:
            price_state["price"] = current_price
            tracker.last_price_update_ts = _time.time()

    price_state["window_start_price"] = current_price

    flat_state = {
        "up_best_bid": None, "up_best_ask": None,
        "down_best_bid": None, "down_best_ask": None,
    }

    tracker.new_window(end)

    # Detect if we're joining mid-window on startup — start price will be wrong
    now_check = datetime.now(timezone.utc)
    elapsed_since_start = (now_check - start).total_seconds()
    skip_trading = (tracker.windows_seen == 1 and elapsed_since_start > 10)
    if skip_trading:
        tracker.last_decision = Decision(
            "FLAT", 0.0, 0.0,
            f"WARM-UP: joined {elapsed_since_start:.0f}s into window, feeds warming up"
        )
        print(f"  [WARM-UP] Joined {elapsed_since_start:.0f}s after window start — "
              f"skipping trading, warming up feeds for next window")

    mode = "DRY RUN" if DRY_RUN else "LIVE"
    print(f"  [{mode}] Market: {title}")
    print(f"  Window:   {start.strftime('%H:%M:%S')} -> {end.strftime('%H:%M:%S')} UTC")
    if current_price is not None:
        print(f"  Start price: ${current_price:,.2f} ({price_source})")
    else:
        print(f"  WARNING: No price available — trading disabled this window")
    print(f"  Bankroll: ${tracker.bankroll:,.2f}  |  Min order: {tracker.min_order_shares:.0f} shares")

    # Update shared display state (for multi-market combined display)
    MARKET_DISPLAY_STATES[config.data_subdir] = {
        "tracker": tracker,
        "price_state": price_state,
        "flat_state": flat_state,
        "config": config,
        "window_start": start,
        "window_end": end,
        "market_title": title,
    }

    cancel = asyncio.Event()
    tasks = [
        asyncio.create_task(
            clob_ws(up_token, down_token, book_up, book_down,
                    flat_state, cancel)
        ),
        # RTDS runs persistently in run() — NOT per-window
        asyncio.create_task(
            signal_ticker(tracker, book_up, book_down, price_state,
                          end, slug, up_token, down_token, cancel,
                          skip_trading=skip_trading)
        ),
    ]
    if show_display:
        tasks.append(asyncio.create_task(
            display_ticker(tracker, price_state, flat_state,
                           title, start, end, cancel, config)
        ))

    now = datetime.now(timezone.utc)
    await asyncio.sleep(max(0, (end - now).total_seconds()) + 5)

    cancel.set()
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # Log FLAT reason summary for this window before resolving
    if tracker.flat_reason_counts:
        tracker._log_flat_summary()
        tracker.flat_reason_counts = {}

    tracker.resolve_window(
        slug,
        price_state.get("price"),
        price_state.get("window_start_price"),
    )
    tracker.save_state()


async def run(tracker: LiveTradeTracker, config: MarketConfig):
    """Run a single market."""
    price_state: dict = {"price": None, "window_start_price": None}

    rtds_cancel = asyncio.Event()
    rtds_task = asyncio.create_task(
        rtds_ws(price_state, rtds_cancel, config, tracker)
    )

    try:
        while True:
            await run_window(tracker, config, price_state)
    finally:
        rtds_cancel.set()
        rtds_task.cancel()
        await asyncio.gather(rtds_task, return_exceptions=True)


async def run_multi(runners: list[tuple[LiveTradeTracker, MarketConfig]]):
    """Run multiple markets concurrently with a combined display."""
    display_cancel = asyncio.Event()

    async def market_loop(tracker: LiveTradeTracker, config: MarketConfig):
        price_state: dict = {"price": None, "window_start_price": None}
        rtds_cancel = asyncio.Event()
        rtds_task = asyncio.create_task(
            rtds_ws(price_state, rtds_cancel, config, tracker)
        )
        try:
            while True:
                await run_window(tracker, config, price_state,
                                 show_display=False)
        finally:
            rtds_cancel.set()
            rtds_task.cancel()
            await asyncio.gather(rtds_task, return_exceptions=True)

    tasks = [
        asyncio.create_task(market_loop(t, c)) for t, c in runners
    ]
    tasks.append(asyncio.create_task(multi_display_ticker(display_cancel)))

    try:
        await asyncio.gather(*tasks)
    finally:
        display_cancel.set()
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    global DEBUG, DRY_RUN, TRADES_LOG, STATE_FILE, MULTI_MODE

    parser = argparse.ArgumentParser(
        description="Live trading bot for Polymarket Up/Down 15-min markets"
    )
    parser.add_argument(
        "--market", default=DEFAULT_MARKET,
        choices=[*MARKET_CONFIGS, "all"],
        help="Market to trade, or 'all' for both (default: btc)",
    )
    parser.add_argument(
        "--bankroll", type=float, default=None,
        help="Override bankroll (default: use API USDC balance)",
    )
    parser.add_argument(
        "--latency", type=int, default=0,
        help="Simulated order latency in ms (default: 0)",
    )
    parser.add_argument(
        "--slippage", type=float, default=0.0,
        help="Slippage buffer for signal (default: 0.0)",
    )
    parser.add_argument(
        "--max-loss-pct", type=float, default=50.0,
        help="Circuit breaker: max session loss %% (default: 50)",
    )
    parser.add_argument(
        "--max-trades-per-window", type=int, default=4,
        help="Max trades per 15-min window (default: 4)",
    )
    parser.add_argument(
        "--edge-threshold-step", type=float, default=None,
        help="Additional edge required per subsequent trade in a window (default: per-market)",
    )
    parser.add_argument(
        "--maker", action="store_true",
        help="Use GTC limit orders (maker, 0%% fee) instead of FOK market orders",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run signal but don't place real orders",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from saved state file",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    DEBUG = args.debug
    DRY_RUN = args.dry_run

    multi = args.market == "all"
    MULTI_MODE = multi
    markets = list(MARKET_CONFIGS) if multi else [args.market]

    # Build authenticated client (shared across markets)
    print(f"\n  {'='*62}")
    mode = "DRY RUN" if DRY_RUN else "LIVE TRADING"
    maker_tag = " [MAKER]" if args.maker else ""
    market_label = "BTC + ETH" if multi else get_config(markets[0]).display_name
    print(f"  {mode}{maker_tag} -- {market_label} Up/Down 15m")
    print(f"  {'='*62}")

    client = build_clob_client()

    # Determine total bankroll from API
    api_balance = query_usdc_balance(client)
    if api_balance is not None:
        print(f"  API USDC balance: ${api_balance:,.2f}")

    total_bankroll = args.bankroll
    if total_bankroll is None:
        if api_balance is not None:
            total_bankroll = api_balance
        else:
            total_bankroll = 10_000.0
            print(f"  WARNING: Could not query balance, using default ${total_bankroll:,.0f}")

    # Gas balance check for on-chain redemption
    try:
        w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
        if w3.is_connected():
            private_key = os.getenv("PRIVATE_KEY", "")
            signer = w3.eth.account.from_key(private_key)
            pol_balance = w3.eth.get_balance(signer.address)
            pol_ether = w3.from_wei(pol_balance, "ether")
            print(f"  Signer POL:      {pol_ether:.4f} POL")
            if pol_ether < 0.01:
                print(f"  WARNING: Low POL balance ({pol_ether:.4f}) — "
                      f"need gas for CTF redemption txs. Send >= 0.01 POL "
                      f"to {signer.address}")
        else:
            print(f"  WARNING: Cannot connect to Polygon RPC ({POLYGON_RPC}) — "
                  f"auto-redemption may fail")
    except Exception as exc:
        print(f"  WARNING: Gas check failed: {exc}")

    # Per-market bankroll split (60/40 in multi mode)
    BANKROLL_SPLIT = {"btc": 0.60, "eth": 0.40}

    def build_signal_kw(market_key: str) -> dict:
        """Signal kwargs for a given market."""
        if market_key == "eth":
            kw = dict(
                edge_threshold=0.10,
                edge_threshold_step=0.0,
                max_bet_fraction=0.10,
                reversion_discount=0.15,
                momentum_lookback_s=15,
                momentum_majority=0.7,
                spread_edge_penalty=0.2,
            )
        else:
            kw = dict(
                edge_threshold=0.10,
                edge_threshold_step=0.0,
                max_bet_fraction=0.05,
            )
        if args.edge_threshold_step is not None:
            kw["edge_threshold_step"] = args.edge_threshold_step
        if args.maker:
            kw["maker_mode"] = True
            kw["max_bet_fraction"] = 0.03
        return kw

    def build_tracker(market_key: str, bankroll: float) -> LiveTradeTracker:
        """Create a tracker for a single market."""
        config = get_config(market_key)
        log_path = Path(f"live_trades_{config.data_subdir}.jsonl")
        state_path = Path(f"live_state_{config.data_subdir}.json")

        signal_kw = build_signal_kw(market_key)
        signal = DiffusionSignal(
            bankroll=bankroll, slippage=args.slippage, **signal_kw)

        saved = None
        if args.resume:
            saved = LiveTradeTracker.load_state(state_path)
            if saved:
                bankroll = saved["bankroll"]
                print(f"  [{config.display_name}] Resumed: bankroll=${bankroll:,.2f}, "
                      f"{saved.get('total_trades', 0)} trades, "
                      f"PnL=${saved.get('total_pnl', 0):+,.2f}")

        tracker = LiveTradeTracker(
            client=client,
            signal=signal,
            initial_bankroll=bankroll,
            latency_ms=args.latency,
            slippage=args.slippage,
            max_loss_pct=args.max_loss_pct,
            max_trades_per_window=args.max_trades_per_window,
            maker_mode=args.maker,
            trades_log=log_path,
            state_file=state_path,
            market_key=market_key,
        )
        tracker.api_balance = api_balance

        if saved:
            tracker.windows_seen = saved.get("windows_seen", 0)
            tracker.windows_traded = saved.get("windows_traded", 0)
            tracker.total_fees = saved.get("total_fees", 0.0)
            tracker.peak_bankroll = saved.get("peak_bankroll", bankroll)
            tracker.max_drawdown = saved.get("max_drawdown", 0.0)
            tracker.max_dd_pct = saved.get("max_dd_pct", 0.0)
            tracker.arb_pairs_completed = saved.get("arb_pairs", 0)
            tracker.arb_total_pnl = saved.get("arb_pnl", 0.0)

        return tracker, config, log_path, state_path

    # Build trackers for each market
    trackers = []
    for mkt in markets:
        if multi:
            bk = total_bankroll * BANKROLL_SPLIT.get(mkt, 0.5)
        else:
            bk = total_bankroll
        tracker, config, log_path, state_path = build_tracker(mkt, bk)
        trackers.append((tracker, config))

        order_type = "GTC LIMIT dual-side (maker, 0% fee)" if args.maker else "FOK MARKET (taker)"
        print(f"  [{config.display_name}] Bankroll: ${bk:,.2f}  |  "
              f"Log: {log_path}  |  State: {state_path}")

    print(f"  Order type:      {order_type}")
    print(f"  Max loss:        {args.max_loss_pct}%")
    print(f"  Max trades/win:  {args.max_trades_per_window}")
    print()

    try:
        if multi:
            asyncio.run(run_multi(trackers))
        else:
            asyncio.run(run(trackers[0][0], trackers[0][1]))
    except KeyboardInterrupt:
        print(f"\n  Shutting down...")
        for tracker, config in trackers:
            if tracker.flat_reason_counts:
                tracker._log_flat_summary()
            tracker.cancel_all_orders()
            tracker.save_state()
            total_pnl = sum(r.pnl for r in tracker.all_results)
            print(f"  [{config.display_name}] PnL: ${total_pnl:+,.2f} | "
                  f"Arb PnL: ${tracker.arb_total_pnl:+,.4f} | "
                  f"Bankroll: ${tracker.bankroll:,.2f}")
        print("  Exiting.")


if __name__ == "__main__":
    main()
