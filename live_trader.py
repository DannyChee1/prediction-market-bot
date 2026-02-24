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
import collections
import json
import math
import os
import ssl
import sys
import threading
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
from py_clob_client.order_builder.constants import BUY, SELL

from recorder import OrderBook
from backtest import (
    Snapshot, Decision, Fill, TradeResult,
    DiffusionSignal, walk_book, poly_fee, BacktestEngine,
    norm_cdf, CalibrationTable, build_calibration_table,
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
    align = config.window_align_m
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
        stale_price_timeout_s: float = 10.0,
        min_balance_usd: float = 5.0,
        maker_mode: bool = False,
        window_duration_s: float = 900.0,
        edge_cancel_threshold: float = 0.02,
        max_order_age_s: float = 120.0,
        requote_cooldown_s: float = 3.0,
        max_exposure_pct: float = 20.0,
        maker_warmup_s: float = 180.0,
        exit_enabled: bool = False,
        exit_threshold: float = 0.03,
        exit_min_hold_s: float = 30.0,
        exit_min_remaining_s: float = 60.0,
        max_positions: int = 4,
    ):
        self.client = client
        self.signal = signal
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.latency_ms = latency_ms
        self.slippage = slippage
        self.cooldown_ms = cooldown_ms

        # Failsafes
        self.max_loss_pct = max_loss_pct
        self.max_trades_per_window = max_trades_per_window
        self.stale_price_timeout_s = stale_price_timeout_s
        self.min_balance_usd = min_balance_usd
        self.circuit_breaker_tripped = False
        self.circuit_breaker_reason = ""

        # Maker mode
        self.maker_mode = maker_mode
        self.window_duration_s = window_duration_s
        self.max_exposure_pct = max_exposure_pct
        self.maker_warmup_s = maker_warmup_s
        self.open_orders: list[dict] = []  # resting limit orders
        self.edge_cancel_threshold = edge_cancel_threshold
        self.max_order_age_s = max_order_age_s
        self.requote_cooldown_s = requote_cooldown_s
        self.last_requote_ts: dict[str, float] = {"UP": 0.0, "DOWN": 0.0}

        self.ctx: dict = {}
        self.pending_fills: list[dict] = []  # live fills awaiting resolution
        self.all_results: list[TradeResult] = []
        self.last_fill_ts_ms: int = 0
        self.last_decision: Decision = Decision("FLAT", 0.0, 0.0, "initializing")
        self.last_up_decision: Decision = Decision("FLAT", 0.0, 0.0, "")
        self.last_down_decision: Decision = Decision("FLAT", 0.0, 0.0, "")
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

        # Early exit / position management
        self.exit_enabled = exit_enabled
        self.exit_threshold = exit_threshold
        self.exit_min_hold_s = exit_min_hold_s
        self.exit_min_remaining_s = exit_min_remaining_s
        self.max_positions = max_positions
        self.position_count: int = 0
        self.exited_sides: set[str] = set()

        # Signal diagnostics
        self.flat_reason_counts: dict[str, int] = {}
        self.signal_eval_count: int = 0
        self.last_diag_ts: float = 0.0

    def new_window(self, window_end: datetime):
        # Log previous window's FLAT reason summary before resetting
        if self.flat_reason_counts:
            self._log_flat_summary()
        # Cancel leftover limit orders from previous window
        if self.open_orders:
            self._cancel_open_orders()
        self.ctx = {}
        self.pending_fills = []
        self.open_orders = []
        self.last_fill_ts_ms = 0
        self.last_decision = Decision("FLAT", 0.0, 0.0, "new window")
        self.last_up_decision = Decision("FLAT", 0.0, 0.0, "")
        self.last_down_decision = Decision("FLAT", 0.0, 0.0, "")
        self.windows_seen += 1
        self.window_trade_count = 0
        self.position_count = 0
        self.exited_sides = set()
        self.flat_reason_counts: dict[str, int] = {}
        self.signal_eval_count = 0
        self.last_diag_ts: float = 0.0
        self.last_requote_ts = {"UP": 0.0, "DOWN": 0.0}

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
        return age > self.stale_price_timeout_s

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

        # ── Maker mode: dual-side limit orders ──
        if self.maker_mode:
            return self._evaluate_maker(snapshot, up_token, down_token)

        # ── FOK mode (original path) ──
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
            self._bucket_flat_reason(decision.reason)
            # Periodic diagnostic snapshot (every 60s)
            now = _time.time()
            if now - self.last_diag_ts >= 60:
                self.last_diag_ts = now
                self._log_diagnostic(snapshot, decision)
        else:
            self._place_order(snapshot, decision, up_token, down_token)

        return decision

    def _evaluate_maker(
        self,
        snapshot: Snapshot,
        up_token: str,
        down_token: str,
    ) -> Decision:
        """Maker mode: continuous quote management with cancel/replace.

        Every 1s tick:
        1. Time blackout checks (warmup 60s, end-of-window 30s)
        2. Poll open orders for fills
        3. ALWAYS run decide_both_sides() to get current edge at current bid
        4. For each side: manage existing orders or place new ones
        5. Bucket FLAT reasons, periodic diagnostics
        """
        tau = snapshot.time_remaining_s
        elapsed = self.window_duration_s - tau

        # Time blackout: no trading in first N seconds (default 180s)
        if elapsed < self.maker_warmup_s:
            reason = f"maker warmup ({elapsed:.0f}s < {self.maker_warmup_s:.0f}s)"
            self.last_decision = Decision("FLAT", 0.0, 0.0, reason)
            self._bucket_flat_reason(reason)
            return self.last_decision

        # Time blackout: cancel open orders and stop in last 30s
        if tau < 30:
            if self.open_orders:
                self._cancel_open_orders()
            reason = f"maker end-of-window ({tau:.0f}s < 30s)"
            self.last_decision = Decision("FLAT", 0.0, 0.0, reason)
            self._bucket_flat_reason(reason)
            return self.last_decision

        # Poll open orders for fills
        self._poll_open_orders(snapshot)

        # Evaluate early exits on filled positions
        if self.exit_enabled:
            self._evaluate_exits(snapshot, up_token, down_token)

        # ALWAYS run dual-side signal to get current edge at current bid
        self.ctx["window_trade_count"] = self.window_trade_count
        up_dec, down_dec = self.signal.decide_both_sides(snapshot, self.ctx)
        self.last_up_decision = up_dec
        self.last_down_decision = down_dec
        self.signal_eval_count += 1

        # Use the better-edged side for display
        if up_dec.action != "FLAT" or down_dec.action != "FLAT":
            self.last_decision = up_dec if up_dec.edge >= down_dec.edge else down_dec
        else:
            self.last_decision = up_dec  # both FLAT, use up's reason

        now = _time.time()

        # Current bids for comparison
        current_bids = {
            "UP": snapshot.best_bid_up,
            "DOWN": snapshot.best_bid_down,
        }

        # Manage each side independently
        for dec, token, side_label in [
            (up_dec, up_token, "UP"),
            (down_dec, down_token, "DOWN"),
        ]:
            # Don't re-enter a side that was early-exited this window
            if side_label in self.exited_sides:
                continue

            existing = self._get_open_order(side_label)

            if existing is not None:
                # Has existing order — check if it should be cancelled or requoted
                order_age = now - existing.get("placed_ts_unix", now)
                current_edge = dec.edge if dec.action != "FLAT" else 0.0
                current_bid = current_bids.get(side_label)

                # Edge decayed below cancel threshold?
                if current_edge < self.edge_cancel_threshold:
                    self._cancel_single_order(existing, f"edge_gone ({current_edge:.4f} < {self.edge_cancel_threshold})")
                    continue

                # Order too old?
                if order_age > self.max_order_age_s:
                    cancelled = self._cancel_single_order(
                        existing, f"age={order_age:.0f}s > {self.max_order_age_s:.0f}s")
                    # Optionally replace if still has edge
                    if cancelled and dec.action != "FLAT" and dec.size_usd > 0:
                        if self.position_count < self.max_positions:
                            self._place_limit_order(snapshot, dec, token, side_label)
                    continue

                # Bid improved? Cancel+replace (rate-limited)
                if (current_bid is not None
                        and current_bid > existing["price"]
                        and now - self.last_requote_ts.get(side_label, 0) >= self.requote_cooldown_s):
                    old_price = existing["price"]
                    cancelled = self._cancel_single_order(
                        existing, f"requote_{old_price:.2f}->{current_bid:.2f}")
                    if cancelled and dec.action != "FLAT" and dec.size_usd > 0:
                        if self.position_count < self.max_positions:
                            self._place_limit_order(snapshot, dec, token, side_label)
                            self.last_requote_ts[side_label] = now
                    continue

                # Otherwise: keep resting

            else:
                # No existing order — place if edge exceeds place threshold
                if (dec.action != "FLAT" and dec.size_usd > 0
                        and self.position_count < self.max_positions):
                    # Check total exposure before placing
                    total_committed = sum(o["cost_est"] for o in self.open_orders)
                    max_committed = self.initial_bankroll * (self.max_exposure_pct / 100.0)
                    if total_committed + dec.size_usd > max_committed:
                        self._bucket_flat_reason("exposure_limit")
                        continue
                    self._place_limit_order(snapshot, dec, token, side_label)

        # Bucket FLAT reasons for diagnostics
        if up_dec.action == "FLAT":
            self._bucket_flat_reason(up_dec.reason)
        if down_dec.action == "FLAT":
            self._bucket_flat_reason(down_dec.reason)

        # Periodic diagnostic
        if now - self.last_diag_ts >= 60:
            self.last_diag_ts = now
            self._log_diagnostic(snapshot, self.last_decision)

        return self.last_decision

    def _bucket_flat_reason(self, reason: str):
        """Categorize a FLAT reason for end-of-window summary."""
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
        elif "maker warmup" in reason:
            cat = "maker_warmup"
        elif "maker end-of-window" in reason:
            cat = "maker_eow"
        elif "cooldown" in reason:
            cat = "cooldown"
        else:
            cat = reason[:30]
        self.flat_reason_counts[cat] = self.flat_reason_counts.get(cat, 0) + 1

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
                f" | BTC: ${snapshot.chainlink_price:,.2f}"
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
        self.position_count += 1

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
            "fill_ts_unix": _time.time(),
            "time_remaining_s": snapshot.time_remaining_s,
            "chainlink_price": snapshot.chainlink_price,
            "window_start_price": snapshot.window_start_price,
        })

        entry_price = cost_usd / shares if shares > 0 else 0.0
        print(
            f"\n  [FILLED] BUY {side_label} ${cost_usd:.2f} "
            f"-> {shares:.1f} shares @ {entry_price:.4f} "
            f"| BTC: ${snapshot.chainlink_price:,.2f} "
            f"| Bankroll: ${self.bankroll:,.2f}"
        )

    # ── Maker mode: limit order methods ────────────────────────────────────

    def _get_open_order(self, side: str) -> dict | None:
        """Return the open order dict for a given side ("UP"/"DOWN"), or None."""
        for o in self.open_orders:
            if o.get("side") == side:
                return o
        return None

    def _cancel_single_order(self, order: dict, reason: str) -> bool:
        """Cancel one order, refund reserved bankroll, remove from open_orders.

        Returns False if the API cancel fails (order may have already filled).
        """
        order_id = order["order_id"]
        if not DRY_RUN:
            try:
                self.client.cancel(order_id)
            except Exception as exc:
                if DEBUG:
                    print(f"\n  [CANCEL] error cancelling {order_id[:12]}...: {exc}")
                return False

        # Refund reserved bankroll
        self.bankroll += order["cost_est"]
        self.signal.bankroll = self.bankroll

        # Remove from open_orders
        self.open_orders = [o for o in self.open_orders if o["order_id"] != order_id]

        self._log({
            "type": "limit_cancel",
            "ts": datetime.now(timezone.utc).isoformat(),
            "order_id": order_id,
            "side": order["side"],
            "status": "cancelled_by_bot",
            "reason": reason,
            "refund": round(order["cost_est"], 2),
        })
        print(
            f"\n  [CANCEL] {order['side']} {order['shares']:.1f}sh "
            f"@ {order['price']:.4f} — {reason}"
        )
        return True

    def _place_limit_order(
        self,
        snapshot: Snapshot,
        decision: Decision,
        token_id: str,
        side_label: str,
    ):
        """Place a GTC limit order at best bid for maker mode."""
        if decision.action == "BUY_UP":
            best_bid = snapshot.best_bid_up
        else:
            best_bid = snapshot.best_bid_down

        if best_bid is None or best_bid <= 0:
            return

        limit_price = best_bid
        if limit_price <= 0 or limit_price >= 1.0:
            return

        # Convert USD size to shares
        shares = round(decision.size_usd / limit_price, 1)
        if shares < self.min_order_shares:
            shares = self.min_order_shares

        cost_est = shares * limit_price
        if cost_est > self.bankroll:
            shares = round((self.bankroll - 0.01) / limit_price, 1)
            if shares < self.min_order_shares:
                return
            cost_est = shares * limit_price

        now_iso = datetime.now(timezone.utc).isoformat()
        now_unix = _time.time()
        trade_record = {
            "type": "limit_order",
            "ts": now_iso,
            "market_slug": snapshot.market_slug,
            "side": side_label,
            "price": limit_price,
            "shares": shares,
            "cost_est": round(cost_est, 2),
            "edge": round(decision.edge, 4),
            "signal_reason": decision.reason,
            "bankroll_before": round(self.bankroll, 2),
            "chainlink_price": round(snapshot.chainlink_price, 2),
        }

        if DRY_RUN:
            trade_record["dry_run"] = True
            trade_record["status"] = "dry_run"
            self._log(trade_record)
            print(
                f"\n  [DRY RUN] Would place limit: BUY {side_label} "
                f"{shares:.1f}sh @ {limit_price:.4f} (${cost_est:.2f})"
                f" | BTC: ${snapshot.chainlink_price:,.2f}"
            )
            # Simulate resting order in dry run (not immediate fill)
            self.bankroll -= cost_est
            self.signal.bankroll = self.bankroll
            self.open_orders.append({
                "order_id": f"dry_{side_label}_{int(now_unix)}",
                "side": side_label,
                "price": limit_price,
                "shares": shares,
                "cost_est": cost_est,
                "market_slug": snapshot.market_slug,
                "placed_ts": now_iso,
                "placed_ts_unix": now_unix,
                "time_remaining_s": snapshot.time_remaining_s,
                "chainlink_price": snapshot.chainlink_price,
                "window_start_price": snapshot.window_start_price,
            })
            return

        # Place GTC limit order
        try:
            order_args = OrderArgs(
                token_id=token_id,
                price=limit_price,
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
                f"\n  [LIMIT] {side_label} {shares:.1f}sh @ {limit_price:.4f} "
                f"-> REJECTED (status={status}, err={err_msg})"
            )
            return

        self._log(trade_record)

        # Check if the order was immediately filled (status=matched + takingAmount)
        taking = resp.get("takingAmount", "")
        making = resp.get("makingAmount", "")
        if status == "matched" and taking:
            # Immediately filled — go straight to pending_fills, no resting order
            filled_shares = float(taking)
            actual_cost = float(making) if making else filled_shares * limit_price
            self.bankroll -= actual_cost
            self.signal.bankroll = self.bankroll

            self.pending_fills.append({
                "market_slug": snapshot.market_slug,
                "side": side_label,
                "cost_usd": actual_cost,
                "shares": filled_shares,
                "order_id": order_id,
                "entry_ts": now_iso,
                "fill_ts_unix": _time.time(),
                "time_remaining_s": snapshot.time_remaining_s,
                "chainlink_price": snapshot.chainlink_price,
                "window_start_price": snapshot.window_start_price,
            })
            self.last_fill_ts_ms = snapshot.ts_ms
            self.window_trade_count += 1
            self.position_count += 1

            entry_px = actual_cost / filled_shares if filled_shares > 0 else 0
            print(
                f"\n  [LIMIT FILLED] BUY {side_label} {filled_shares:.1f}sh "
                f"@ {entry_px:.4f} (${actual_cost:.2f})"
                f" | BTC: ${snapshot.chainlink_price:,.2f}"
            )
            return

        # Reserve bankroll for resting order
        self.bankroll -= cost_est
        self.signal.bankroll = self.bankroll

        # Track the open order
        self.open_orders.append({
            "order_id": order_id,
            "side": side_label,
            "price": limit_price,
            "shares": shares,
            "cost_est": cost_est,
            "market_slug": snapshot.market_slug,
            "placed_ts": now_iso,
            "placed_ts_unix": now_unix,
            "time_remaining_s": snapshot.time_remaining_s,
            "chainlink_price": snapshot.chainlink_price,
            "window_start_price": snapshot.window_start_price,
        })

        print(
            f"\n  [LIMIT] BUY {side_label} {shares:.1f}sh @ {limit_price:.4f} "
            f"(${cost_est:.2f}) -> resting (id={order_id[:12]}...)"
        )

    def _poll_open_orders(self, snapshot: Snapshot):
        """Check open limit orders for fills or cancellations."""
        if not self.open_orders:
            return

        still_open = []
        for order in self.open_orders:
            order_id = order["order_id"]

            # Dry-run: simulate fill after 5s resting
            if DRY_RUN and order_id.startswith("dry_"):
                age = _time.time() - order.get("placed_ts_unix", _time.time())
                if age >= 5.0:
                    actual_cost = order["shares"] * order["price"]
                    drift = order["cost_est"] - actual_cost
                    self.bankroll += drift
                    self.signal.bankroll = self.bankroll
                    self.pending_fills.append({
                        "market_slug": order["market_slug"],
                        "side": order["side"],
                        "cost_usd": actual_cost,
                        "shares": order["shares"],
                        "order_id": order_id,
                        "entry_ts": order["placed_ts"],
                        "fill_ts_unix": _time.time(),
                        "time_remaining_s": order["time_remaining_s"],
                        "chainlink_price": order["chainlink_price"],
                        "window_start_price": order["window_start_price"],
                    })
                    self.last_fill_ts_ms = snapshot.ts_ms
                    self.window_trade_count += 1
                    self.position_count += 1
                    entry_px = actual_cost / order["shares"] if order["shares"] > 0 else 0
                    print(
                        f"\n  [DRY FILL] {order['side']} {order['shares']:.1f}sh "
                        f"@ {entry_px:.4f} (${actual_cost:.2f})"
                    )
                else:
                    still_open.append(order)
                continue

            try:
                resp = self.client.get_order(order_id)
            except Exception as exc:
                if DEBUG:
                    print(f"\n  [POLL] error checking {order_id[:12]}...: {exc}")
                still_open.append(order)
                continue

            status = resp.get("status", "unknown")
            size_matched = float(resp.get("size_matched", 0) or 0)
            original_size = float(resp.get("original_size", order["shares"]) or order["shares"])
            fill_pct = size_matched / original_size if original_size > 0 else 0

            if status == "MATCHED" or fill_pct >= 0.99:
                # Fully filled — move to pending_fills
                actual_cost = size_matched * order["price"]
                # Adjust bankroll: we reserved cost_est, actual may differ
                drift = order["cost_est"] - actual_cost
                self.bankroll += drift
                self.signal.bankroll = self.bankroll

                self.pending_fills.append({
                    "market_slug": order["market_slug"],
                    "side": order["side"],
                    "cost_usd": actual_cost,
                    "shares": size_matched,
                    "order_id": order_id,
                    "entry_ts": order["placed_ts"],
                    "fill_ts_unix": _time.time(),
                    "time_remaining_s": order["time_remaining_s"],
                    "chainlink_price": order["chainlink_price"],
                    "window_start_price": order["window_start_price"],
                })
                self.last_fill_ts_ms = snapshot.ts_ms
                self.window_trade_count += 1
                self.position_count += 1

                entry_px = actual_cost / size_matched if size_matched > 0 else 0
                print(
                    f"\n  [LIMIT FILLED] {order['side']} {size_matched:.1f}sh "
                    f"@ {entry_px:.4f} (${actual_cost:.2f})"
                )
                self._log({
                    "type": "limit_fill",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "order_id": order_id,
                    "side": order["side"],
                    "shares": round(size_matched, 2),
                    "price": order["price"],
                    "cost_usd": round(actual_cost, 2),
                })

            elif status in ("CANCELLED", "EXPIRED"):
                # Refund reserved bankroll
                self.bankroll += order["cost_est"]
                self.signal.bankroll = self.bankroll
                print(
                    f"\n  [LIMIT {status}] {order['side']} "
                    f"{order['shares']:.1f}sh @ {order['price']:.4f} "
                    f"— refunded ${order['cost_est']:.2f}"
                )
                self._log({
                    "type": "limit_cancel",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "order_id": order_id,
                    "side": order["side"],
                    "status": status,
                    "refund": round(order["cost_est"], 2),
                })
            else:
                # Still open
                still_open.append(order)

        self.open_orders = still_open

    def _cancel_open_orders(self):
        """Cancel all open limit orders and refund reserved bankroll."""
        for order in self.open_orders:
            order_id = order["order_id"]
            if not DRY_RUN:
                try:
                    self.client.cancel(order_id)
                except Exception as exc:
                    if DEBUG:
                        print(f"\n  [CANCEL] error: {exc}")
            # Refund reserved bankroll
            self.bankroll += order["cost_est"]
            self.signal.bankroll = self.bankroll
            self._log({
                "type": "limit_cancel",
                "ts": datetime.now(timezone.utc).isoformat(),
                "order_id": order_id,
                "side": order["side"],
                "status": "cancelled_by_bot",
                "refund": round(order["cost_est"], 2),
            })
        if self.open_orders:
            print(f"\n  [CANCEL] Cancelled {len(self.open_orders)} open limit order(s)")
        self.open_orders = []

    # ── Early exit methods ─────────────────────────────────────────────────

    def _evaluate_exits(self, snapshot: Snapshot, up_token: str, down_token: str):
        """Evaluate filled positions for early exit based on EV comparison."""
        if not self.pending_fills:
            return

        tau = snapshot.time_remaining_s
        now = _time.time()

        # Don't exit when too close to expiry — let resolution handle it
        if tau < self.exit_min_remaining_s:
            return

        # Compute current p_model from live price data
        hist = self.ctx.get("price_history", [])
        if len(hist) < max(2, self.signal.vol_lookback_s):
            return

        sigma_per_s = self.signal._compute_vol(hist[-self.signal.vol_lookback_s:])
        if sigma_per_s <= 0 or tau <= 0:
            return

        delta = (snapshot.chainlink_price - snapshot.window_start_price) / snapshot.window_start_price
        z_raw = delta / (sigma_per_s * math.sqrt(tau))
        z = max(-self.signal.max_z, min(self.signal.max_z, z_raw))
        p_model = self.signal._p_model(z, tau)

        token_map = {"UP": up_token, "DOWN": down_token}
        bid_map = {
            "UP": snapshot.best_bid_up,
            "DOWN": snapshot.best_bid_down,
        }

        exits_to_process = []
        for fill in self.pending_fills:
            side = fill["side"]

            # Skip if already exited this side
            if side in self.exited_sides:
                continue

            # Min hold time
            fill_age = now - fill.get("fill_ts_unix", now)
            if fill_age < self.exit_min_hold_s:
                continue

            best_bid = bid_map.get(side)
            if best_bid is None or best_bid <= 0:
                continue

            # EV calculations
            entry_price = fill["cost_usd"] / fill["shares"] if fill["shares"] > 0 else 0
            ev_hold = p_model if side == "UP" else (1.0 - p_model)
            ev_sell = best_bid  # 0% maker fee for selling at bid

            # Hard stop-loss: exit if bid < 50% of entry price
            hard_stop = best_bid < entry_price * 0.50

            # EV-based exit
            ev_edge = ev_sell - ev_hold
            should_exit = hard_stop or ev_edge > self.exit_threshold

            if should_exit:
                token_id = token_map.get(side, "")
                reason = "hard_stop" if hard_stop else "ev_edge"
                exits_to_process.append((fill, token_id, best_bid, ev_hold, ev_sell, reason))

        # Process exits (separate loop to avoid mutating pending_fills during iteration)
        for fill, token_id, sell_price, ev_hold, ev_sell, reason in exits_to_process:
            success = self._execute_exit(fill, snapshot, token_id, sell_price, ev_hold, ev_sell, reason)
            if success:
                self.exited_sides.add(fill["side"])
                self.pending_fills = [f for f in self.pending_fills if f is not fill]
                self.position_count = max(0, self.position_count - 1)

    def _execute_exit(
        self, fill: dict, snapshot: Snapshot, token_id: str,
        sell_price: float, ev_hold: float, ev_sell: float, reason: str,
    ) -> bool:
        """Execute a FOK market sell to exit a position. Returns True if filled."""
        shares = fill["shares"]
        entry_price = fill["cost_usd"] / shares if shares > 0 else 0
        now_iso = datetime.now(timezone.utc).isoformat()

        if DRY_RUN:
            proceeds = shares * sell_price
            exit_pnl = proceeds - fill["cost_usd"]
            self.bankroll += proceeds
            self.signal.bankroll = self.bankroll

            self._log({
                "type": "early_exit",
                "ts": now_iso,
                "market_slug": fill["market_slug"],
                "side": fill["side"],
                "shares": round(shares, 2),
                "entry_price": round(entry_price, 4),
                "sell_price": round(sell_price, 4),
                "proceeds": round(proceeds, 2),
                "pnl": round(exit_pnl, 2),
                "ev_hold": round(ev_hold, 4),
                "ev_sell": round(ev_sell, 4),
                "reason": reason,
                "dry_run": True,
            })
            print(
                f"\n  [DRY EXIT] {fill['side']} {shares:.1f}sh "
                f"@ {sell_price:.4f} (entry {entry_price:.4f}) "
                f"PnL=${exit_pnl:+.2f} | EV: hold={ev_hold:.3f} sell={ev_sell:.3f} "
                f"({reason})"
            )
            self._record_exit_result(fill, proceeds, exit_pnl)
            return True

        # Live: place FOK market sell
        try:
            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=shares,
                side=SELL,
            )
            signed_order = self.client.create_market_order(order_args)
            resp = self.client.post_order(signed_order, OrderType.FOK)
        except Exception as exc:
            print(f"\n  [EXIT ERROR] {fill['side']}: {exc}")
            self._log({
                "type": "early_exit",
                "ts": now_iso,
                "side": fill["side"],
                "status": "error",
                "error": str(exc),
            })
            return False

        success = resp.get("success", False)
        status = resp.get("status", "unknown")

        if not success or status not in ("matched", "live"):
            if DEBUG:
                print(f"\n  [EXIT] {fill['side']} FOK not filled (status={status})")
            return False

        # FOK filled: takingAmount = USDC received
        taking = resp.get("takingAmount", "")
        proceeds = float(taking) if taking else shares * sell_price
        exit_pnl = proceeds - fill["cost_usd"]

        self.bankroll += proceeds
        self.signal.bankroll = self.bankroll

        self._log({
            "type": "early_exit",
            "ts": now_iso,
            "market_slug": fill["market_slug"],
            "side": fill["side"],
            "shares": round(shares, 2),
            "entry_price": round(entry_price, 4),
            "sell_price": round(sell_price, 4),
            "proceeds": round(proceeds, 2),
            "pnl": round(exit_pnl, 2),
            "ev_hold": round(ev_hold, 4),
            "ev_sell": round(ev_sell, 4),
            "reason": reason,
            "bankroll_after": round(self.bankroll, 2),
        })
        print(
            f"\n  [EXIT] {fill['side']} {shares:.1f}sh "
            f"@ {sell_price:.4f} (entry {entry_price:.4f}) "
            f"PnL=${exit_pnl:+.2f} | EV: hold={ev_hold:.3f} sell={ev_sell:.3f}"
        )
        self._record_exit_result(fill, proceeds, exit_pnl)
        return True

    def _record_exit_result(self, fill: dict, proceeds: float, exit_pnl: float):
        """Record an early exit as a TradeResult for session stats."""
        entry_price = fill["cost_usd"] / fill["shares"] if fill["shares"] > 0 else 0
        fake_fill = Fill(
            market_slug=fill["market_slug"],
            side=fill["side"],
            entry_ts_ms=int(_time.time() * 1000),
            time_remaining_s=fill["time_remaining_s"],
            entry_price=entry_price,
            fee_per_share=0.0,
            shares=fill["shares"],
            cost_usd=fill["cost_usd"],
            signal_name="diffusion",
            decision_reason="early_exit",
        )
        pnl_pct = exit_pnl / fill["cost_usd"] if fill["cost_usd"] > 0 else 0.0
        result = TradeResult(
            fill=fake_fill,
            outcome_up=-1,  # sentinel: early exit, not resolution
            final_btc=0.0,
            payout=proceeds,
            pnl=exit_pnl,
            pnl_pct=pnl_pct,
        )
        self.all_results.append(result)

        # Update drawdown
        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll
        dd = self.peak_bankroll - self.bankroll
        dd_pct = dd / self.peak_bankroll if self.peak_bankroll > 0 else 0
        if dd > self.max_drawdown:
            self.max_drawdown = dd
            self.max_dd_pct = dd_pct

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

        # Redeem winning CTF positions on-chain (background thread)
        has_winning = any(
            (f["side"] == "UP" and outcome_up == 1) or
            (f["side"] == "DOWN" and outcome_up == 0)
            for f in self.pending_fills
        )
        if has_winning and self.condition_id:
            cond_id = self.condition_id
            t = threading.Thread(
                target=self.redeem_positions, args=(cond_id,),
                daemon=True,
            )
            t.start()
            print(f"  [REDEEM] Started background redemption thread "
                  f"(conditionId: {cond_id[:10]}...)")
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
        # Cancel tracked limit orders first (refund bankroll)
        if self.open_orders:
            self._cancel_open_orders()
        if DRY_RUN:
            return
        try:
            resp = self.client.cancel_all()
            print(f"  Cancelled all open orders: {resp}")
        except Exception as exc:
            print(f"  Warning: cancel_all failed: {exc}")

    def redeem_positions(self, condition_id: str,
                         max_retries: int = 3,
                         max_poll_attempts: int = 60,
                         poll_interval_s: float = 30.0) -> str | None:
        """Call CTF.redeemPositions() on-chain to convert winning tokens to USDC.

        First polls payoutDenominator on-chain (free eth_call) to wait for
        the condition to be resolved on-chain before spending gas. On-chain
        resolution can lag the Gamma API by 20+ minutes.

        Routes through the proxy wallet (sig type 1) or direct EOA (sig type 0).
        Returns the tx hash on success, None on failure.
        """
        if DRY_RUN:
            print("  [REDEEM] Skipped (dry run)")
            return None

        if not condition_id:
            print("  [REDEEM] Skipped (no conditionId)")
            return None

        # Poll payoutDenominator until on-chain resolution arrives
        condition_bytes = bytes.fromhex(condition_id.replace("0x", ""))
        resolved = False
        try:
            w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
            ctf = w3.eth.contract(address=CTF_ADDRESS, abi=CTF_ABI)
            for poll in range(max_poll_attempts):
                try:
                    denom = ctf.functions.payoutDenominator(condition_bytes).call()
                    if denom > 0:
                        resolved = True
                        print(f"  [REDEEM] On-chain resolution confirmed "
                              f"(payoutDenominator={denom}, poll #{poll + 1})")
                        break
                except Exception as exc:
                    if DEBUG:
                        print(f"  [REDEEM] payoutDenominator poll error: {exc}")
                if poll < max_poll_attempts - 1:
                    if poll == 0:
                        print(f"  [REDEEM] Waiting for on-chain resolution "
                              f"(polling every {poll_interval_s:.0f}s, "
                              f"up to {max_poll_attempts * poll_interval_s / 60:.0f}m)...")
                    _time.sleep(poll_interval_s)
        except Exception as exc:
            print(f"  [REDEEM] RPC connection error: {exc}")

        if not resolved:
            print(f"  [REDEEM] On-chain resolution not found after "
                  f"{max_poll_attempts * poll_interval_s / 60:.0f}m. "
                  f"Claim manually on Polymarket.")
            return None

        # On-chain resolution confirmed — now send the actual redeem tx
        for attempt in range(max_retries):
            if attempt > 0:
                delay = 20 * attempt
                print(f"  [REDEEM] Retry {attempt + 1}/{max_retries} in {delay}s...")
                _time.sleep(delay)

            result = self._try_redeem_once(condition_id)
            if result is not None:
                return result

        print(f"  [REDEEM] Failed after {max_retries} tx attempts. "
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
        dyn_threshold = self.signal.edge_threshold * (
            1.0 + self.signal.early_edge_mult * math.sqrt(tau / self.signal.window_duration)
        ) if tau > 0 else self.signal.edge_threshold

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
            delta = (snapshot.chainlink_price - snapshot.window_start_price) / snapshot.window_start_price
            z_raw = delta / (sigma_per_s * math.sqrt(tau))
            z = max(-self.signal.max_z, min(self.signal.max_z, z_raw))
            p_model = self.signal._p_model(z, tau)
            if self.maker_mode:
                # Maker mode: edge at bid, 0% fee
                edge_up = p_model - bid_up - self.signal.spread_edge_penalty * spread_up
                edge_down = (1.0 - p_model) - bid_down - self.signal.spread_edge_penalty * spread_down
            else:
                # Taker mode: edge at ask + fee
                p_up_cost = ask_up + poly_fee(ask_up) + self.signal.slippage
                p_down_cost = ask_down + poly_fee(ask_down) + self.signal.slippage
                edge_up = p_model - p_up_cost - self.signal.spread_edge_penalty * spread_up
                edge_down = (1.0 - p_model) - p_down_cost - self.signal.spread_edge_penalty * spread_down

        self._log({
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
        })

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
            with open(TRADES_LOG, "a") as f:
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
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    @classmethod
    def load_state(cls) -> dict | None:
        if not STATE_FILE.exists():
            return None
        try:
            with open(STATE_FILE) as f:
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
                            ts_ms = data_arr[-1].get("timestamp") or payload.get("timestamp")
                            if p is not None:
                                price_state["price"] = float(p)
                                tracker.last_price_update_ts = _time.time()
                                if ts_ms is not None:
                                    price_state["price_history"].append((int(ts_ms), float(p)))
                                if price_state["window_start_price"] is None:
                                    price_state["window_start_price"] = float(p)
                            continue

                        p = payload.get("value")
                        ts_ms = payload.get("timestamp")
                        if p is not None:
                            price_state["price"] = float(p)
                            tracker.last_price_update_ts = _time.time()
                            if ts_ms is not None:
                                price_state["price_history"].append((int(ts_ms), float(p)))
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
    mode_str = "DRY RUN" if DRY_RUN else "LIVE"
    order_mode = "MAKER" if tracker.maker_mode else "FOK"
    exit_tag = "+EXIT" if tracker.exit_enabled else ""
    lines.append("=" * 62)
    lines.append(f"  [{mode_str}] [{order_mode}{exit_tag}] {config.display_name} Up/Down: {market_title}")
    lines.append("=" * 62)
    lines.append("")

    now = datetime.now(timezone.utc)
    remaining = (window_end - now).total_seconds()
    win_dur = tracker.window_duration_s

    if remaining > 0:
        m, s = int(remaining // 60), int(remaining % 60)
        bar_len = 30
        filled = max(0, min(bar_len, int((remaining / win_dur) * bar_len)))
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
    lines.append(f"  -- {mode_str} Trading ({order_mode}) " + "-" * 24)

    dec = tracker.last_decision
    status = dec.action if dec.action != "FLAT" else "FLAT"

    # Balance line
    bal_str = f"${tracker.bankroll:,.2f}"
    if tracker.api_balance is not None:
        bal_str += f"  (API: ${tracker.api_balance:,.2f})"
    lines.append(f"  Bankroll: {bal_str}  |  Status: {status}")
    lines.append(f"  Reason:   {dec.reason[:60]}")

    # Maker mode: show dual-side status
    if tracker.maker_mode:
        up_d = tracker.last_up_decision
        dn_d = tracker.last_down_decision
        up_tag = up_d.action if up_d.action != "FLAT" else "FLAT"
        dn_tag = dn_d.action if dn_d.action != "FLAT" else "FLAT"
        lines.append(
            f"  Up: {up_tag} (edge={up_d.edge:.4f})"
            f"  |  Down: {dn_tag} (edge={dn_d.edge:.4f})"
        )

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
    if tracker.maker_mode and tracker.open_orders:
        lines.append("  -- Open Limit Orders " + "-" * 36)
        for o in tracker.open_orders:
            age_s = int(_time.time() - o.get("placed_ts_unix", _time.time()))
            lines.append(
                f"    {o['side']:>4s}  {o['shares']:.1f}sh @ {o['price']:.4f}"
                f"  (${o['cost_est']:.2f})  age={age_s}s"
                f"  id={o['order_id'][:12]}..."
            )
        lines.append("")

    # Positions display (early exit mode)
    if tracker.exit_enabled and tracker.pending_fills:
        lines.append(
            f"  -- Positions ({tracker.position_count}/{tracker.max_positions}) "
            + "-" * 36
        )
        bid_map = {
            "UP": flat_state.get("up_best_bid"),
            "DOWN": flat_state.get("down_best_bid"),
        }
        for fill in tracker.pending_fills:
            entry_px = fill["cost_usd"] / fill["shares"] if fill["shares"] > 0 else 0
            bid_val = bid_map.get(fill["side"])
            bid_f = float(bid_val) if bid_val else 0.0
            upnl = (bid_f - entry_px) * fill["shares"]
            hold_s = int(_time.time() - fill.get("fill_ts_unix", _time.time()))
            lines.append(
                f"    {fill['side']:>4s}  {fill['shares']:.1f}sh "
                f"@ {entry_px:.4f}  bid={bid_f:.2f}  "
                f"uPnL=${upnl:+.2f}  hold={hold_s}s"
            )
        if tracker.exited_sides:
            lines.append(f"    Exited: {', '.join(sorted(tracker.exited_sides))}")
        lines.append("")
    elif tracker.pending_fills:
        # Standard fill display
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
    if tracker.all_results:
        r = tracker.all_results[-1]
        tag = "WON" if r.pnl > 0 else "LOST"
        lines.append(
            f"  Last Result:  {r.fill.side} "
            f"${r.fill.cost_usd:.2f} -> {tag} ${r.pnl:+.2f}"
        )
    lines.append("")

    # Session stats
    wins = [r for r in tracker.all_results if r.pnl > 0]
    total = len(tracker.all_results)
    total_pnl = sum(r.pnl for r in tracker.all_results)
    win_count = len(wins)
    win_str = f"{win_count}/{total} ({win_count / total:.0%})" if total > 0 else "---"
    open_str = f"  |  Open: {len(tracker.open_orders)}" if tracker.maker_mode else ""
    lines.append(
        f"  Session:  {tracker.windows_traded}/{tracker.windows_seen}"
        f" windows traded  |  Win: {win_str}{open_str}"
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


# ── Window Lifecycle ─────────────────────────────────────────────────────────

async def run_window(tracker: LiveTradeTracker, config: MarketConfig,
                     price_state: dict):
    """Run a single trading window.

    price_state is a shared dict continuously updated by the persistent RTDS
    websocket. At window start we snapshot the current live price as the
    window_start_price, which is much more accurate than carrying a stale
    price from after the previous window's RTDS disconnected.
    """
    win_m = int(config.window_duration_s / 60)
    print(f"  Searching for active {config.display_name} market...")
    event, market = find_market(config)

    if not event or not market:
        print(f"  No active {config.display_name} market found. Retrying in 30s...")
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

    # Wait until eventStartTime + 5s so the RTDS buffer has the start price.
    now = datetime.now(timezone.utc)
    target = start + timedelta(seconds=5)
    wait_s = (target - now).total_seconds()
    if 0 < wait_s <= 120:  # cap at 2 min to avoid hanging
        print(f"  Waiting {wait_s:.0f}s for start price (until {target.strftime('%H:%M:%S')} UTC)...")
        await asyncio.sleep(wait_s)

    # Look up exact Chainlink price at eventStartTime from RTDS buffer.
    # Polymarket's "Price to Beat" = Chainlink price at this exact second.
    start_ts_ms = int(start.timestamp() * 1000)
    start_price_exact = None
    history = price_state.get("price_history", [])
    if history:
        # Find exact match or closest within 2s
        best_entry = None
        best_diff = float("inf")
        for ts_ms, px in history:
            diff = abs(ts_ms - start_ts_ms)
            if diff < best_diff:
                best_diff = diff
                best_entry = (ts_ms, px)
        if best_entry and best_diff <= 2000:  # within 2 seconds
            start_price_exact = best_entry[1]

    if start_price_exact is not None:
        price_state["window_start_price"] = start_price_exact
        offset_ms = best_entry[0] - start_ts_ms
        print(f"  Start price: ${start_price_exact:,.2f} (Chainlink @ eventStart{offset_ms:+d}ms)")
    else:
        current_price = price_state.get("price")
        price_state["window_start_price"] = current_price
        print(f"  Start price: ${current_price:,.2f} (RTDS fallback — no exact match in buffer)"
              if current_price else "  Start price: waiting for RTDS...")

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
    # Start price already printed above during lookup
    print(f"  Bankroll: ${tracker.bankroll:,.2f}  |  Min order: {tracker.min_order_shares:.0f} shares")

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
        asyncio.create_task(
            display_ticker(tracker, price_state, flat_state,
                           title, start, end, cancel, config)
        ),
    ]

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
    # Shared price state that persists across windows.
    # The RTDS websocket continuously updates 'price'; at each window
    # transition run_window() snapshots it as 'window_start_price'.
    # Keeping RTDS persistent eliminates the reconnection gap that caused
    # stale "Price to Beat" values ($20+ errors).
    # price_history buffers (chainlink_ts_ms, price) for exact Price to Beat lookup.
    price_state: dict = {
        "price": None,
        "window_start_price": None,
        "price_history": collections.deque(maxlen=600),  # ~10 min at 1/s
    }

    # Persistent RTDS connection — stays alive across window transitions
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


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    global DEBUG, DRY_RUN, TRADES_LOG, STATE_FILE

    parser = argparse.ArgumentParser(
        description="Live trading bot for Polymarket Up/Down markets"
    )
    parser.add_argument(
        "--market", default=DEFAULT_MARKET, choices=list(MARKET_CONFIGS),
        help="Market to trade (default: btc)",
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
        "--max-trades-per-window", type=int, default=None,
        help="Max trades per window (default: 1 for FOK, 4 for maker)",
    )
    parser.add_argument(
        "--maker", action="store_true",
        help="Enable maker mode: GTC limit orders, dual-side evaluation, 0%% fee",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run signal but don't place real orders",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from saved state file",
    )
    parser.add_argument(
        "--edge-cancel-threshold", type=float, default=0.02,
        help="Maker: cancel open orders when edge drops below this (default: 0.02)",
    )
    parser.add_argument(
        "--max-order-age", type=float, default=120.0,
        help="Maker: auto-cancel orders resting longer than this (seconds, default: 120)",
    )
    parser.add_argument(
        "--requote-cooldown", type=float, default=3.0,
        help="Maker: min seconds between cancel+replace on same side (default: 3)",
    )
    parser.add_argument("--max-exposure-pct", type=float, default=20.0,
                        help="Max %% of initial bankroll committed to open orders (default: 20)")
    parser.add_argument("--maker-warmup", type=float, default=180.0,
                        help="Maker: seconds to wait after window opens before trading (default: 180)")
    parser.add_argument("--calibrated", action="store_true",
                        help="Use empirically calibrated probabilities")
    parser.add_argument("--early-exit", action="store_true",
                        help="Enable early exit of positions based on EV")
    parser.add_argument("--exit-threshold", type=float, default=0.03,
                        help="Min EV edge to trigger exit (default: 0.03)")
    parser.add_argument("--exit-min-hold", type=float, default=30.0,
                        help="Min seconds after fill before evaluating exit (default: 30)")
    parser.add_argument("--exit-min-remaining", type=float, default=60.0,
                        help="Don't exit when < this many seconds remain (default: 60)")
    parser.add_argument("--max-positions", type=int, default=4,
                        help="Max simultaneous filled positions (default: 4)")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    DEBUG = args.debug
    DRY_RUN = args.dry_run

    config = get_config(args.market)
    TRADES_LOG = Path(f"live_trades_{config.data_subdir}.jsonl")
    STATE_FILE = Path(f"live_state_{config.data_subdir}.json")

    # Default max trades depends on mode
    if args.max_trades_per_window is not None:
        max_trades = args.max_trades_per_window
    else:
        max_trades = 4 if args.maker else 1

    # Build authenticated client
    print(f"\n  {'='*62}")
    mode = "DRY RUN" if DRY_RUN else "LIVE TRADING"
    order_mode = "MAKER" if args.maker else "FOK"
    if args.calibrated:
        order_mode += "+CAL"
    if args.early_exit:
        order_mode += "+EXIT"
    print(f"  {mode} [{order_mode}] -- {config.display_name} Up/Down")
    print(f"  {'='*62}")

    client = build_clob_client()

    # Determine bankroll
    api_balance = query_usdc_balance(client)
    if api_balance is not None:
        print(f"  API USDC balance: ${api_balance:,.2f}")

    bankroll = args.bankroll
    saved = None
    if args.resume:
        saved = LiveTradeTracker.load_state()
        if saved:
            bankroll = saved["bankroll"]
            print(f"  Resumed: bankroll=${bankroll:,.2f}, "
                  f"{saved.get('total_trades', 0)} trades, "
                  f"PnL=${saved.get('total_pnl', 0):+,.2f}")

    if bankroll is None:
        if api_balance is not None:
            bankroll = api_balance
        else:
            bankroll = 10_000.0
            print(f"  WARNING: Could not query balance, using default ${bankroll:,.0f}")

    # Per-market signal overrides (ETH needs tighter filters due to mean reversion)
    signal_kw: dict = {}
    base_market = args.market.replace("_5m", "")
    if base_market == "eth":
        signal_kw = dict(
            edge_threshold=0.15,        # higher bar (BTC default 0.10)
            reversion_discount=0.15,    # ETH mean-reverts ~33%, discount p toward 0.5
            momentum_lookback_s=15,     # shorter lookback (ETH oscillates more)
            momentum_majority=0.7,      # 70% majority instead of 100% (BTC default)
            spread_edge_penalty=0.2,    # reduced from 1.0 (avoids double-counting)
        )

    # Maker mode signal overrides
    signal_kw["window_duration"] = config.window_duration_s
    cooldown_ms = 30_000
    stale_timeout = 10.0

    if args.maker:
        signal_kw["maker_mode"] = True
        signal_kw["max_bet_fraction"] = 0.02
        signal_kw["edge_threshold"] = 0.06
        signal_kw["momentum_majority"] = 0.0
        signal_kw["spread_edge_penalty"] = 0.0  # bid pricing handles spread naturally
        cooldown_ms = 30_000  # no global cooldown in maker (kept for FOK compat)
        stale_timeout = 60.0

    # Calibration: build empirical lookup table and adjust thresholds
    cal_table = None
    if args.calibrated:
        from backtest import DATA_DIR
        cal_data_dir = DATA_DIR / config.data_subdir
        print(f"  Building calibration table from {cal_data_dir} ...")
        cal_table = build_calibration_table(cal_data_dir)
        n_cells = len(cal_table.table)
        n_obs = sum(cal_table.counts.values())
        print(f"  Calibration table: {n_cells} cells, {n_obs} observations")
        signal_kw["calibration_table"] = cal_table
        # Calibrated edges are smaller/honest — still require meaningful edge
        if args.maker:
            signal_kw["edge_threshold"] = 0.06
            signal_kw["early_edge_mult"] = 2.0
        else:
            signal_kw["edge_threshold"] = 0.06
            signal_kw["early_edge_mult"] = 2.0

    signal = DiffusionSignal(bankroll=bankroll, slippage=args.slippage, **signal_kw)
    tracker = LiveTradeTracker(
        client=client,
        signal=signal,
        initial_bankroll=bankroll,
        latency_ms=args.latency,
        slippage=args.slippage,
        cooldown_ms=cooldown_ms,
        max_loss_pct=args.max_loss_pct,
        max_trades_per_window=max_trades,
        stale_price_timeout_s=stale_timeout,
        maker_mode=args.maker,
        window_duration_s=config.window_duration_s,
        edge_cancel_threshold=args.edge_cancel_threshold,
        max_order_age_s=args.max_order_age,
        requote_cooldown_s=args.requote_cooldown,
        max_exposure_pct=args.max_exposure_pct,
        maker_warmup_s=args.maker_warmup,
        exit_enabled=args.early_exit,
        exit_threshold=args.exit_threshold,
        exit_min_hold_s=args.exit_min_hold,
        exit_min_remaining_s=args.exit_min_remaining,
        max_positions=args.max_positions,
    )
    tracker.api_balance = api_balance

    if saved:
        tracker.windows_seen = saved.get("windows_seen", 0)
        tracker.windows_traded = saved.get("windows_traded", 0)
        tracker.total_fees = saved.get("total_fees", 0.0)
        tracker.peak_bankroll = saved.get("peak_bankroll", bankroll)
        tracker.max_drawdown = saved.get("max_drawdown", 0.0)
        tracker.max_dd_pct = saved.get("max_dd_pct", 0.0)

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

    print(f"  Bankroll:        ${bankroll:,.2f}")
    print(f"  Mode:            {order_mode}")
    print(f"  Max loss:        {args.max_loss_pct}%")
    print(f"  Max trades/win:  {max_trades}")
    if args.maker:
        print(f"  Max exposure:    {args.max_exposure_pct}%")
        print(f"  Maker warmup:    {args.maker_warmup:.0f}s")
    if args.early_exit:
        print(f"  Early exit:      ON (threshold={args.exit_threshold}, "
              f"min_hold={args.exit_min_hold:.0f}s, "
              f"min_remaining={args.exit_min_remaining:.0f}s)")
        print(f"  Max positions:   {args.max_positions}")
    print(f"  Cooldown:        {cooldown_ms / 1000:.0f}s")
    print(f"  Stale timeout:   {stale_timeout:.0f}s")
    print(f"  Trades log:      {TRADES_LOG}")
    print(f"  State file:      {STATE_FILE}")
    print()

    try:
        asyncio.run(run(tracker, config))
    except KeyboardInterrupt:
        print(f"\n  Shutting down...")
        if tracker.flat_reason_counts:
            tracker._log_flat_summary()
        tracker.cancel_all_orders()
        tracker.save_state()
        total_pnl = sum(r.pnl for r in tracker.all_results)
        print(f"  Session PnL: ${total_pnl:+,.2f}")
        print(f"  Final bankroll: ${tracker.bankroll:,.2f}")
        print(f"  State saved to {STATE_FILE}")
        print("  Exiting.")


if __name__ == "__main__":
    main()
