"""LiveTradeTracker: trading state, signal evaluation, resolution, and logging."""

from __future__ import annotations

import collections
import json
import math
import time as _time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from backtest import (
    Snapshot, Decision, Fill, TradeResult,
    DiffusionSignal, poly_fee,
)
from market_api import poll_market_resolution
from orders import OrderMixin
from redemption import RedemptionMixin


# ── Redemption Queue ──────────────────────────────────────────────────────────

@dataclass
class RedemptionItem:
    condition_id: str
    market_slug: str
    enqueued_at: float      # time.time()
    attempts: int = 0
    last_attempt_ts: float = 0.0  # time.time() of last redemption attempt


class RedemptionQueue:
    """Persistent queue for retrying failed on-chain redemptions."""

    TTL_S = 14400           # discard items older than 4 hours
    MAX_ATTEMPTS = 10
    RETRY_INTERVAL_S = 90   # min seconds between retries of the same item
    MIN_DELAY_S = 120       # wait before first attempt (let resolution confirm)

    def __init__(self, queue_file: Path):
        self._file = queue_file
        self._items: list[RedemptionItem] = []
        self._last_process_ts: float = 0.0
        self._load()

    # ── public API ────────────────────────────────────────────────────

    def enqueue(self, condition_id: str, market_slug: str):
        """Add a new redemption to the back of the queue."""
        # Don't double-enqueue the same condition_id
        if any(it.condition_id == condition_id for it in self._items):
            return
        self._items.append(RedemptionItem(
            condition_id=condition_id,
            market_slug=market_slug,
            enqueued_at=_time.time(),
        ))
        self._save()

    def pop_next(self) -> RedemptionItem | None:
        """Return the front item if ready for retry, else None.

        Items are only eligible after RETRY_INTERVAL_S since enqueue/last
        attempt. Expired or max-attempt items are discarded silently here
        (caller should use process() which handles logging).
        """
        if not self._items:
            return None
        return self._items[0]

    def remove_front(self):
        """Remove the front item (after success or discard)."""
        if self._items:
            self._items.pop(0)
            self._save()

    def requeue_front(self):
        """Increment attempts on front item and move to back of queue."""
        if not self._items:
            return
        item = self._items.pop(0)
        item.attempts += 1
        self._items.append(item)
        self._save()

    def remove_by_id(self, condition_id: str):
        """Remove an item by condition_id."""
        self._items = [it for it in self._items if it.condition_id != condition_id]
        self._save()

    def requeue_by_id(self, condition_id: str):
        """Increment attempts on item and move to back of queue."""
        for i, it in enumerate(self._items):
            if it.condition_id == condition_id:
                item = self._items.pop(i)
                item.attempts += 1
                item.last_attempt_ts = _time.time()
                self._items.append(item)
                self._save()
                return

    @property
    def items(self) -> list:
        """Read-only snapshot of queue items."""
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def ready_items(self, now: float) -> list[RedemptionItem]:
        """Return items that are past MIN_DELAY_S, past retry interval,
        not expired, and not max attempts."""
        ready = []
        for item in self._items:
            age = now - item.enqueued_at
            if age > self.TTL_S:
                continue
            if item.attempts >= self.MAX_ATTEMPTS:
                continue
            # Must wait MIN_DELAY_S since enqueue before first attempt
            if age < self.MIN_DELAY_S:
                continue
            # Respect retry interval for subsequent attempts
            if item.attempts > 0:
                since_last = now - item.last_attempt_ts if item.last_attempt_ts else age
                if since_last < self.RETRY_INTERVAL_S:
                    continue
            ready.append(item)
        return ready

    # ── persistence ───────────────────────────────────────────────────

    def _save(self):
        # Atomic write: dump to temp file, fsync, rename. Prevents
        # partial-write corruption on crash mid-write. Logs the
        # exception loudly instead of swallowing — silent save failures
        # combined with hard-kill have lost whole sessions of state.
        try:
            data = [asdict(it) for it in self._items]
            tmp = self._file.with_suffix(self._file.suffix + ".tmp")
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                import os as _os
                _os.fsync(f.fileno())
            tmp.replace(self._file)
        except Exception as exc:
            print(f"  [REDEMPTION_QUEUE] save FAILED: "
                  f"{type(exc).__name__}: {exc}")

    def _load(self):
        if not self._file.exists():
            return
        try:
            with open(self._file) as f:
                data = json.load(f)
            self._items = [
                RedemptionItem(**d) for d in data
            ]
        except Exception:
            self._items = []


class LiveTradeTracker(OrderMixin, RedemptionMixin):
    """Manages live trading state, order placement, and failsafes."""

    def __init__(
        self,
        client,  # polybot_core.OrderClient or py_clob_client.ClobClient
        signal: DiffusionSignal,
        initial_bankroll: float,
        latency_ms: int = 0,
        slippage: float = 0.0,
        cooldown_ms: int = 30_000,
        max_loss_pct: float = 50.0,
        max_trades_per_window: int = 1,
        same_direction_stacking_only: bool = True,
        stale_price_timeout_s: float = 10.0,
        min_balance_usd: float = 5.0,
        window_duration_s: float = 900.0,
        edge_cancel_threshold: float = 0.02,
        max_order_age_s: float = 120.0,
        requote_cooldown_s: float = 3.0,
        min_requote_ticks: int = 2,
        tick_size: float = 0.01,
        max_exposure_pct: float = 20.0,
        maker_warmup_s: float = 100.0,
        exit_enabled: bool = False,
        exit_threshold: float = 0.03,
        exit_min_hold_s: float = 30.0,
        exit_min_remaining_s: float = 60.0,
        max_positions: int = 4,
        maker_withdraw_s: float = 60.0,
        exit_sell_buffer: float = 0.08,
        dual_side: bool = False,
        max_net_exposure: float = 0.0,    # max |UP_shares - DOWN_shares|, 0=disabled
        max_gross_exposure: float = 0.0,  # max total shares both sides, 0=disabled
        # Instance-level settings (formerly globals)
        debug: bool = False,
        dry_run: bool = False,
        trades_log: Path | None = None,
        state_file: Path | None = None,
    ):
        self.client = client
        self.signal = signal
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.latency_ms = latency_ms
        self.slippage = slippage
        self.cooldown_ms = cooldown_ms

        # Instance-level settings (formerly globals)
        self.debug = debug
        self.dry_run = dry_run
        self.trades_log = trades_log or Path("live_trades.jsonl")
        self.state_file = state_file or Path("live_state.json")

        # Failsafes
        self.max_loss_pct = max_loss_pct
        self.max_trades_per_window = max_trades_per_window
        self.same_direction_stacking_only = same_direction_stacking_only
        # Track first-trade side per window so subsequent trades can be
        # restricted to the same direction (averaging in, not hedging).
        # Reset on each new window. Set by _record_window_fill_side.
        self.window_first_side: str | None = None
        self.stale_price_timeout_s = stale_price_timeout_s
        self.min_balance_usd = min_balance_usd
        self.circuit_breaker_tripped = False
        self.circuit_breaker_reason = ""

        # Maker mode settings
        self.maker_withdraw_s = maker_withdraw_s
        self.window_duration_s = window_duration_s
        self.max_exposure_pct = max_exposure_pct
        self.maker_warmup_s = maker_warmup_s
        self.open_orders: list[dict] = []
        self.edge_cancel_threshold = edge_cancel_threshold
        self.max_order_age_s = max_order_age_s
        self.requote_cooldown_s = requote_cooldown_s
        self.min_requote_ticks = min_requote_ticks
        self.tick_size = tick_size
        self.last_requote_ts: dict[str, float] = {"UP": 0.0, "DOWN": 0.0}

        self.ctx: dict = {}
        self.cal_data_dir: Path | None = None
        self.pending_fills: list[dict] = []
        self.all_results: list[TradeResult] = []
        self.last_fill_ts_ms: int = 0
        self.last_decision: Decision = Decision("FLAT", 0.0, 0.0, "initializing")
        self.last_up_decision: Decision = Decision("FLAT", 0.0, 0.0, "")
        self.last_down_decision: Decision = Decision("FLAT", 0.0, 0.0, "")
        self.last_price_update_ts: float = 0.0
        self.window_trade_count: int = 0
        self.min_order_shares: float = 5.0
        # Recent events shown on display (thread-safe deque)
        self.event_log: collections.deque = collections.deque(maxlen=6)

        # Session stats
        self.windows_seen: int = 0
        self.windows_traded: int = 0
        self.total_fees: float = 0.0
        self.peak_bankroll: float = initial_bankroll
        self.max_drawdown: float = 0.0
        self.max_dd_pct: float = 0.0

        # Lifetime scalars — separate from all_results so the displayed
        # totals don't drift after the in-memory results list is truncated
        # at the 5000-trade cap during save_state. These are the source of
        # truth for lifetime PnL/win/loss counts; all_results is the
        # rolling history used by display widgets.
        self.lifetime_pnl: float = 0.0
        self.lifetime_wins: int = 0
        self.lifetime_losses: int = 0
        self.lifetime_trades: int = 0

        # On-chain redemption
        self.condition_id: str = ""
        queue_file = self.state_file.parent / self.state_file.name.replace(
            "live_state_", "live_redemption_queue_"
        )
        self.redemption_queue = RedemptionQueue(queue_file)

        # Early exit / position management
        self.exit_enabled = exit_enabled
        self.exit_threshold = exit_threshold
        self.exit_min_hold_s = exit_min_hold_s
        self.exit_min_remaining_s = exit_min_remaining_s
        self.max_positions = max_positions
        self.position_count: int = 0
        self.exited_sides: set[str] = set()
        self.dual_side = dual_side
        self.max_net_exposure = max_net_exposure
        self.max_gross_exposure = max_gross_exposure

        # Maker sell order state
        self.open_sell_orders: list[dict] = []
        self.exit_sell_buffer = exit_sell_buffer
        self.exit_sell_requote_cooldown_s: float = 3.0
        self.last_exit_requote_ts: dict[str, float] = {"UP": 0.0, "DOWN": 0.0}

        # UserFeed WebSocket (set externally by live_trader)
        self.user_feed = None
        self._last_rest_poll_ts: float = 0.0
        self._rest_poll_interval_s: float = 10.0  # REST reconciliation every 10s when WS active

        # Signal diagnostics
        self.flat_reason_counts: dict[str, int] = {}
        self.signal_eval_count: int = 0
        self.last_diag_ts: float = 0.0
        self.latency_samples: collections.deque = collections.deque(maxlen=512)

    def _record_window_fill_side(self, side: str) -> None:
        """Lock in the first-trade side for this window so subsequent
        trades can be filtered against it (same-direction stacking).
        Idempotent: only sets the first time per window. Called from
        every fill site in orders.py."""
        if self.window_first_side is None and side in ("UP", "DOWN"):
            self.window_first_side = side

    def _increment_lifetime_totals(self, result) -> None:
        """Update lifetime scalar counters when a trade resolves.

        These scalars (lifetime_pnl, lifetime_wins, lifetime_losses,
        lifetime_trades) are persisted to state separately from
        all_results, so the displayed lifetime PnL never drifts after
        the all_results list is truncated at 5000 entries during
        save_state. Without this, restart-then-save would silently
        drop the historical PnL of everything beyond window 5000.
        """
        self.lifetime_pnl += result.pnl
        self.lifetime_trades += 1
        if result.pnl > 0:
            self.lifetime_wins += 1
        else:
            self.lifetime_losses += 1

    def new_window(self, window_end: datetime):
        if self.flat_reason_counts:
            self._log_flat_summary()
        if self.open_orders:
            self._cancel_open_orders()
        if self.open_sell_orders:
            self._cancel_open_sell_orders()
        if self.pending_fills:
            for fill in self.pending_fills:
                print(
                    f"\n  [WARNING] Unresolved fill carried into new window: "
                    f"{fill['side']} {fill['shares']:.1f}sh ${fill['cost_usd']:.2f}"
                )
                self._log({
                    "type": "unresolved_fill",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "side": fill["side"],
                    "shares": round(fill["shares"], 2),
                    "cost_usd": round(fill["cost_usd"], 2),
                    "order_id": fill.get("order_id", ""),
                })
        self.ctx = {}
        self.pending_fills = []
        self.open_orders = []
        self.last_fill_ts_ms = 0
        self.last_decision = Decision("FLAT", 0.0, 0.0, "new window")
        self.last_up_decision = Decision("FLAT", 0.0, 0.0, "")
        self.last_down_decision = Decision("FLAT", 0.0, 0.0, "")
        self.windows_seen += 1
        self.window_trade_count = 0
        self.window_first_side = None
        self.position_count = 0
        self.exited_sides = set()
        self.open_sell_orders = []
        self.last_exit_requote_ts = {"UP": 0.0, "DOWN": 0.0}
        self.flat_reason_counts = {}
        self.signal_eval_count = 0
        self.last_diag_ts = 0.0
        self.last_requote_ts = {"UP": 0.0, "DOWN": 0.0}
        self.latency_samples = collections.deque(maxlen=512)

    def _check_circuit_breakers(self) -> str | None:
        """Returns reason string if trading should stop, None if OK."""
        if self.circuit_breaker_tripped:
            return self.circuit_breaker_reason

        total_pnl = sum(r.pnl for r in self.all_results)
        loss_pct = abs(total_pnl) / self.initial_bankroll * 100 if total_pnl < 0 else 0
        if loss_pct >= self.max_loss_pct:
            self.circuit_breaker_tripped = True
            self.circuit_breaker_reason = (
                f"CIRCUIT BREAKER: session loss {loss_pct:.1f}% "
                f"(${total_pnl:+.2f}) exceeds {self.max_loss_pct}% limit"
            )
            return self.circuit_breaker_reason

        if self.bankroll < self.min_balance_usd:
            self.circuit_breaker_tripped = True
            self.circuit_breaker_reason = (
                f"CIRCUIT BREAKER: bankroll ${self.bankroll:.2f} "
                f"below minimum ${self.min_balance_usd:.2f}"
            )
            return self.circuit_breaker_reason

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
        # ALWAYS run signal first — this accumulates price history for vol
        # warmup regardless of stale price, circuit breakers, or maker warmup.
        self.ctx["window_trade_count"] = self.window_trade_count
        self.ctx["inventory_up"] = sum(f["shares"] for f in self.pending_fills if f["side"] == "UP")
        self.ctx["inventory_down"] = sum(f["shares"] for f in self.pending_fills if f["side"] == "DOWN")
        up_dec, down_dec = self.signal.decide_both_sides(snapshot, self.ctx)
        self.last_up_decision = up_dec
        self.last_down_decision = down_dec

        cb_reason = self._check_circuit_breakers()
        if cb_reason:
            self.last_decision = Decision("FLAT", 0.0, 0.0, cb_reason)
            return self.last_decision

        if self._check_stale_price():
            self.last_decision = Decision(
                "FLAT", 0.0, 0.0,
                f"stale price ({_time.time() - self.last_price_update_ts:.0f}s old)"
            )
            return self.last_decision

        return self._evaluate_maker(snapshot, up_token, down_token, up_dec, down_dec)

    def _evaluate_maker(
        self,
        snapshot: Snapshot,
        up_token: str,
        down_token: str,
        up_dec: Decision,
        down_dec: Decision,
    ) -> Decision:
        """Maker mode: continuous quote management with cancel/replace."""
        tau = snapshot.time_remaining_s
        elapsed = self.window_duration_s - tau

        if elapsed < self.maker_warmup_s:
            reason = f"maker warmup ({elapsed:.0f}s < {self.maker_warmup_s:.0f}s)"
            self.last_decision = Decision("FLAT", 0.0, 0.0, reason)
            self._bucket_flat_reason(reason)
            return self.last_decision

        if tau < self.maker_withdraw_s:
            if self.open_orders:
                self._cancel_open_orders()
            if self.open_sell_orders:
                self._cancel_open_sell_orders()
            reason = f"maker end-of-window ({tau:.0f}s < {self.maker_withdraw_s:.0f}s)"
            self.last_decision = Decision("FLAT", 0.0, 0.0, reason)
            self._bucket_flat_reason(reason)
            return self.last_decision

        # Process UserFeed WS events first (instant fills/cancels)
        self._process_user_events(snapshot)

        # REST polling: every tick if no WS, every 10s as reconciliation if WS active
        now_poll = _time.time()
        if self.user_feed is None or (now_poll - self._last_rest_poll_ts >= self._rest_poll_interval_s):
            self._poll_open_orders(snapshot)
            self._last_rest_poll_ts = now_poll

        if self.exit_enabled:
            self._evaluate_exits(snapshot, up_token, down_token)
        self.signal_eval_count += 1

        if up_dec.action != "FLAT" or down_dec.action != "FLAT":
            self.last_decision = up_dec if up_dec.edge >= down_dec.edge else down_dec
        else:
            self.last_decision = up_dec

        now = _time.time()

        current_bids = {
            "UP": snapshot.best_bid_up,
            "DOWN": snapshot.best_bid_down,
        }

        # Exposure-aware position gating
        # Count both filled AND resting orders toward position limits.
        # Open orders MUST be included in net_exposure / gross_exposure
        # so the cap is airtight — without this, you can place an UP
        # then a DOWN limit while both pass the gate (because only
        # filled positions are summed), then end up dual-sided when
        # both fill. The same_direction_stacking_only guard depends on
        # this being correct.
        filled_sides = {f["side"] for f in self.pending_fills}
        open_sides = {o["side"] for o in self.open_orders}
        committed_sides = filled_sides | open_sides
        up_shares = sum(f["shares"] for f in self.pending_fills if f["side"] == "UP")
        down_shares = sum(f["shares"] for f in self.pending_fills if f["side"] == "DOWN")
        # Add open-order shares to both sides so net_exposure /
        # gross_exposure reflect the worst-case fill state.
        up_shares += sum(o.get("shares", 0) for o in self.open_orders if o.get("side") == "UP")
        down_shares += sum(o.get("shares", 0) for o in self.open_orders if o.get("side") == "DOWN")
        net_exposure = up_shares - down_shares   # +ve = long UP
        gross_exposure = up_shares + down_shares
        # Effective position count: filled + resting orders
        effective_positions = self.position_count + len(self.open_orders)

        for dec, token, side_label in [
            (up_dec, up_token, "UP"),
            (down_dec, down_token, "DOWN"),
        ]:
            if side_label in self.exited_sides:
                continue

            opposite = "DOWN" if side_label == "UP" else "UP"
            has_opposite = opposite in committed_sides

            if has_opposite:
                if not self.dual_side:
                    # Classic mode: block opposite side entirely
                    continue
                # Smart dual-side: only allow if it REDUCES net imbalance
                if side_label == "UP" and net_exposure > 0:
                    continue  # already long UP, buying more UP won't rebalance
                if side_label == "DOWN" and net_exposure < 0:
                    continue  # already long DOWN, buying more DOWN won't rebalance

            # Gross exposure cap
            if self.max_gross_exposure > 0 and gross_exposure >= self.max_gross_exposure:
                continue

            # Net exposure cap: block if this side would push imbalance past limit
            if self.max_net_exposure > 0:
                if side_label == "UP" and net_exposure >= self.max_net_exposure:
                    continue
                if side_label == "DOWN" and -net_exposure >= self.max_net_exposure:
                    continue

            if self.window_trade_count >= self.max_trades_per_window:
                break

            # Same-direction stacking guard: when max_trades_per_window > 1
            # AND a prior trade fired in this window, only allow subsequent
            # trades on the SAME side as the first. Prevents in-window
            # whipsaws from generating opposite-direction hedges that
            # fight each other.
            if (self.same_direction_stacking_only
                    and self.window_first_side is not None
                    and side_label != self.window_first_side):
                continue

            existing = self._get_open_order(side_label)

            if existing is not None:
                order_age = now - existing.get("placed_ts_unix", now)
                current_edge = dec.edge if dec.action != "FLAT" else 0.0
                current_bid = current_bids.get(side_label)

                # Skip cancel/requote on orders pending verify-retry. They
                # are waiting for _poll_open_orders to determine actual
                # status from Polymarket. Re-attempting cancel here would
                # spam the API and could trigger duplicate verify failures.
                if existing.get("_verify_pending"):
                    continue

                # Cancel-on-bid-drop: when the bid moves significantly below
                # our resting order price, our order is the highest bid in
                # the book and gets adversely picked off when sellers come
                # in. Cancel rather than wear the loss.
                # Threshold: 2 ticks below order price.
                if (current_bid is not None
                        and existing["price"] > current_bid + 2 * self.tick_size):
                    self._cancel_single_order(
                        existing,
                        f"bid_dropped ({existing['price']:.3f} -> {current_bid:.3f})"
                    )
                    continue

                if current_edge < self.edge_cancel_threshold:
                    self._cancel_single_order(existing, f"edge_gone ({current_edge:.4f} < {self.edge_cancel_threshold})")
                    continue

                if order_age > self.max_order_age_s:
                    cancelled = self._cancel_single_order(
                        existing, f"age={order_age:.0f}s > {self.max_order_age_s:.0f}s")
                    if cancelled and dec.action != "FLAT" and dec.size_usd > 0:
                        if effective_positions < self.max_positions:
                            self._place_limit_order(snapshot, dec, token, side_label)
                    continue

                # Queue priority: require minimum tick improvement AND edge increase
                min_improve = self.min_requote_ticks * self.tick_size
                edge_at_place = existing.get("edge_at_place", 0.0)
                if (current_bid is not None
                        and current_bid >= existing["price"] + min_improve
                        and current_edge > edge_at_place
                        and now - self.last_requote_ts.get(side_label, 0) >= self.requote_cooldown_s):
                    old_price = existing["price"]
                    cancelled = self._cancel_single_order(
                        existing, f"requote_{old_price:.2f}->{current_bid:.2f}")
                    if cancelled and dec.action != "FLAT" and dec.size_usd > 0:
                        if effective_positions < self.max_positions:
                            self._place_limit_order(snapshot, dec, token, side_label)
                            self.last_requote_ts[side_label] = now
                    continue

            else:
                if (dec.action != "FLAT" and dec.size_usd > 0
                        and effective_positions < self.max_positions
                        and self.window_trade_count < self.max_trades_per_window):
                    total_committed = sum(o["cost_est"] for o in self.open_orders)
                    max_committed = self.initial_bankroll * (self.max_exposure_pct / 100.0)
                    if total_committed + dec.size_usd > max_committed:
                        self._bucket_flat_reason("exposure_limit")
                        continue
                    self._place_limit_order(snapshot, dec, token, side_label)

        if up_dec.action == "FLAT":
            self._bucket_flat_reason(up_dec.reason)
        if down_dec.action == "FLAT":
            self._bucket_flat_reason(down_dec.reason)

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
        elif "vol kill" in reason:
            cat = "vol_kill"
        elif "toxicity" in reason:
            cat = "toxicity"
        else:
            cat = reason[:30]
        self.flat_reason_counts[cat] = self.flat_reason_counts.get(cat, 0) + 1

    # ── Early exit evaluation ──────────────────────────────────────────────

    def _evaluate_exits(self, snapshot: Snapshot, up_token: str, down_token: str):
        """Always-on maker limit sell orders for filled positions.

        After min hold time, always have a resting GTC limit sell on the book.
        Sell price = max(best_bid, p_model_side + buffer). Continuously requoted
        as the market moves. If sell_price >= 0.99, let it ride to $1.00.
        """
        if not self.pending_fills:
            return

        now = _time.time()

        # Poll existing sell orders for fills/cancellations
        self._poll_open_sell_orders(snapshot)

        # Reuse sigma already computed by decide_both_sides() this tick
        # to avoid double-updating the EMA.
        tau = snapshot.time_remaining_s
        sigma_per_s = self.ctx.get("_sigma_per_s", 0.0)
        if sigma_per_s <= 0 or tau <= 0:
            return

        delta = (snapshot.chainlink_price - snapshot.window_start_price) / snapshot.window_start_price
        z_raw = delta / (sigma_per_s * math.sqrt(tau))
        z = max(-self.signal.max_z, min(self.signal.max_z, z_raw))
        p_model = self.signal._p_model(z, tau, self.ctx)

        token_map = {"UP": up_token, "DOWN": down_token}
        bid_map = {
            "UP": snapshot.best_bid_up,
            "DOWN": snapshot.best_bid_down,
        }

        for fill in self.pending_fills:
            side = fill["side"]

            if side in self.exited_sides:
                continue

            # M5 fix: default to a LARGE age (not now), so fills restored
            # from a state file that didn't have fill_ts_unix don't
            # immediately become eligible for early exit. Restored fills
            # whose age is unknown are treated as "fully aged" (safe to
            # consider for exit), not "just filled" (which would block
            # exit indefinitely). Pick the larger of: actual age if
            # available, OR 2× the min hold (ensures eligibility).
            fill_ts = fill.get("fill_ts_unix")
            if fill_ts is None or fill_ts <= 0:
                fill_age = self.exit_min_hold_s * 2.0
            else:
                fill_age = now - fill_ts
            if fill_age < self.exit_min_hold_s:
                continue

            best_bid = bid_map.get(side)
            if best_bid is None or best_bid <= 0:
                continue

            # Sell price = max(best_bid, p_model_side + buffer)
            p_model_side = p_model if side == "UP" else (1.0 - p_model)
            floor_price = p_model_side + self.exit_sell_buffer
            sell_price = max(best_bid, floor_price)

            # Near certainty — let it ride to $1.00
            if sell_price >= 0.99:
                # Cancel existing sell if any — we're letting it ride
                existing = self._get_open_sell_order(side)
                if existing is not None:
                    self._cancel_single_sell_order(existing, "near_certainty_0.99")
                continue

            # Clamp to valid range
            sell_price = round(min(sell_price, 0.99), 2)
            if sell_price <= 0:
                continue

            token_id = token_map.get(side, "")
            existing = self._get_open_sell_order(side)

            if existing is None:
                # No existing sell → place new one
                self._place_limit_sell(fill, snapshot, token_id, side, sell_price)
            elif abs(existing["price"] - sell_price) >= 0.01:
                # Price changed → requote (with cooldown)
                if now - self.last_exit_requote_ts.get(side, 0) >= self.exit_sell_requote_cooldown_s:
                    old_price = existing["price"]
                    cancelled = self._cancel_single_sell_order(existing, f"requote_{old_price:.2f}->{sell_price:.2f}")
                    # If cancel returned False, order filled during cancel race — don't re-sell
                    if cancelled:
                        self._place_limit_sell(fill, snapshot, token_id, side, sell_price)
                    self.last_exit_requote_ts[side] = now

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
            outcome_up=-1,
            final_btc=0.0,
            payout=proceeds,
            pnl=exit_pnl,
            pnl_pct=pnl_pct,
        )
        self.all_results.append(result)
        self._increment_lifetime_totals(result)

        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll
        dd = self.peak_bankroll - self.bankroll
        dd_pct = dd / self.peak_bankroll if self.peak_bankroll > 0 else 0
        if dd > self.max_drawdown:
            self.max_drawdown = dd
            self.max_dd_pct = dd_pct

    # ── Resolution ─────────────────────────────────────────────────────────

    def resolve_window(
        self, slug: str, final_price: float | None,
        window_start_price: float | None,
        condition_id: str = "",
    ):
        """At window close: poll for resolution, compute PnL, redeem.

        condition_id is passed explicitly (captured at window end) to avoid
        race conditions when the next window overwrites self.condition_id.
        """
        # Use explicit param; fall back to self.condition_id for backwards compat
        cond_id = condition_id or self.condition_id

        if not self.pending_fills:
            print("  [RESOLVE] No pending fills — skipping resolution")
            return 0.0

        fill_summary = ", ".join(
            f"{f['side']} {f['shares']:.1f}sh ${f['cost_usd']:.2f}"
            for f in self.pending_fills
        )
        print(f"\n  Polling Gamma API for market resolution...")
        print(f"  [RESOLVE] {len(self.pending_fills)} fills: {fill_summary}")
        print(f"  [RESOLVE] conditionId: {cond_id[:16]}..." if cond_id else "  [RESOLVE] conditionId: MISSING")

        outcome_up = poll_market_resolution(slug, debug=self.debug)

        if outcome_up is not None:
            print(f"  [RESOLVE] API result: {'UP' if outcome_up else 'DOWN'}")
        else:
            print(f"  [RESOLVE] API returned None — using price fallback")
            if final_price is not None and window_start_price is not None:
                outcome_up = 1 if final_price >= window_start_price else 0
                print(
                    f"  WARNING: API resolution unavailable, using local "
                    f"prices (start=${window_start_price:,.2f} "
                    f"final=${final_price:,.2f}) -> {'UP' if outcome_up else 'DOWN'}"
                )
            else:
                print(f"  ERROR: Cannot resolve window (final_price={final_price}, start={window_start_price})")
                for fill in self.pending_fills:
                    self.bankroll += fill["cost_usd"]
                self.pending_fills = []
                return 0.0

        self.windows_traded += 1
        window_pnl = 0.0
        outcome_str = "UP" if outcome_up else "DOWN"

        for fill in self.pending_fills:
            won = ((fill["side"] == "UP" and outcome_up == 1) or
                   (fill["side"] == "DOWN" and outcome_up == 0))
            payout = fill["shares"] if won else 0.0
            pnl = payout - fill["cost_usd"]
            pnl_pct = pnl / fill["cost_usd"] if fill["cost_usd"] > 0 else 0.0

            self.bankroll += payout
            window_pnl += pnl

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
            self._increment_lifetime_totals(result)

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

        # Enqueue winning CTF positions for on-chain redemption
        # Skip in dry-run: no real positions to redeem.
        if not self.dry_run:
            has_winning = any(
                (f["side"] == "UP" and outcome_up == 1) or
                (f["side"] == "DOWN" and outcome_up == 0)
                for f in self.pending_fills
            )
            print(f"  [RESOLVE] has_winning={has_winning}, condition_id={'yes' if cond_id else 'MISSING'}")
            if has_winning and cond_id:
                self._log({
                    "type": "redemption_enqueued",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "condition_id": cond_id,
                    "market_slug": slug,
                })
                self.redemption_queue.enqueue(cond_id, slug)
                print(f"  [REDEEM] Enqueued for redemption "
                      f"(conditionId: {cond_id[:10]}..., "
                      f"queue size: {len(self.redemption_queue)})")
                # M6: process the queue inline immediately so a crash
                # between enqueue and the live_trader's background spawn
                # doesn't leave wins waiting >4h. enqueue() already
                # persisted to disk, so worst case is a duplicate
                # process attempt — process_redemption_queue handles
                # already-redeemed conditions idempotently.
                if not self.dry_run:
                    try:
                        self.process_redemption_queue(max_per_cycle=3)
                    except Exception as exc:
                        print(f"  [REDEEM] inline process error: "
                              f"{type(exc).__name__}: {exc}")
            elif has_winning and not cond_id:
                self._log({
                    "type": "redemption_skipped",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "reason": "no_condition_id",
                    "market_slug": slug,
                })
                print("  WARNING: Won but no conditionId — cannot auto-redeem. "
                      "Claim manually on Polymarket.")

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
        return window_pnl

    # ── Redemption Queue Processing ──────────────────────────────────────────

    def process_redemption_queue(self, max_per_cycle: int = 10):
        """Process ready items from the redemption queue (batch-aware).

        Called periodically by the background redemption loop and at startup.
        Batches multiple resolved conditions into a single relayer call to
        avoid 429 rate-limits on /submit.
        """
        if not self.redemption_queue:
            return

        # Skip entire cycle if relayer is rate-limiting us
        if self._is_rate_limited():
            return

        now = _time.time()

        # ── Pass 1: discard expired / abandoned items ────────────────
        for item in self.redemption_queue.items:       # snapshot
            age = now - item.enqueued_at

            if age > RedemptionQueue.TTL_S:
                print(f"  [REDEEM] Expired after {age / 60:.0f}m — discarding "
                      f"(conditionId: {item.condition_id[:10]}...)")
                self._log({
                    "type": "redemption_expired",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "condition_id": item.condition_id,
                    "market_slug": item.market_slug,
                    "age_s": round(age),
                    "attempts": item.attempts,
                })
                self.redemption_queue.remove_by_id(item.condition_id)
                continue

            if item.attempts >= RedemptionQueue.MAX_ATTEMPTS:
                print(f"  [REDEEM] Abandoned after {item.attempts} attempts "
                      f"(conditionId: {item.condition_id[:10]}...)")
                self._log({
                    "type": "redemption_abandoned",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "condition_id": item.condition_id,
                    "market_slug": item.market_slug,
                    "attempts": item.attempts,
                })
                self.redemption_queue.remove_by_id(item.condition_id)

        # ── Pass 2: collect ready items ────────────────────────────────
        ready = self.redemption_queue.ready_items(now)
        if not ready:
            return
        ready = ready[:max_per_cycle]

        # ── Pass 3: check on-chain resolution in one RPC pass ─────────
        candidate_ids = [it.condition_id for it in ready]
        resolved_ids = self.check_resolved_batch(candidate_ids)
        resolved_set = set(resolved_ids)

        # Requeue unresolved items (bumps attempt counter for retry interval)
        for item in ready:
            if item.condition_id not in resolved_set:
                self.redemption_queue.requeue_by_id(item.condition_id)

        if not resolved_ids:
            if candidate_ids:
                print(f"  [REDEEM] {len(candidate_ids)} item(s) checked — "
                      f"none resolved yet, will retry later")
            return

        print(f"  [REDEEM] {len(resolved_ids)}/{len(candidate_ids)} "
              f"condition(s) confirmed resolved on-chain")

        # ── Pass 4: redeem — single path or batch ─────────────────────
        try:
            if len(resolved_ids) == 1:
                # Single item: use existing single-condition path
                result = self.redeem_positions(
                    resolved_ids[0],
                    max_retries=1,
                    max_poll_attempts=3,     # already confirmed resolved
                    poll_interval_s=2.0,
                )
            else:
                # Batch: single relayer call for all resolved conditions
                result = self._try_redeem_batch(resolved_ids)
        except Exception as exc:
            result = None
            print(f"  [REDEEM] Queue processing error: "
                  f"{type(exc).__name__}: {exc}")
            self._log({
                "type": "redemption_error",
                "ts": datetime.now(timezone.utc).isoformat(),
                "condition_ids": resolved_ids,
                "error": f"{type(exc).__name__}: {exc}",
            })

        if result is not None:
            short_ids = [cid[:10] + "..." for cid in resolved_ids]
            print(f"  [REDEEM] SUCCESS — redeemed {len(resolved_ids)} condition(s): "
                  f"{short_ids}, tx: {result[:16]}...")
            for cid in resolved_ids:
                self.redemption_queue.remove_by_id(cid)
        else:
            for cid in resolved_ids:
                self.redemption_queue.requeue_by_id(cid)
            print(f"  [REDEEM] Will retry {len(resolved_ids)} item(s) later "
                  f"(queue size: {len(self.redemption_queue)})")

    # ── Startup: find unredeemed wins ──────────────────────────────────────

    def scan_unredeemed_wins(self):
        """Scan trade logs for winning positions that were never redeemed.

        Scans ALL log files in the same directory matching the base asset
        (e.g., live_trades_btc*.jsonl catches btc.jsonl, btc_15m.jsonl, btc_5m.jsonl).
        Adds unredeemed condition_ids back to the queue for the periodic
        processor to claim.
        """
        # Build list of log files to scan — current + any old per-timeframe logs
        log_dir = self.trades_log.parent
        base_name = self.trades_log.stem  # e.g., "live_trades_btc"
        log_files = sorted(log_dir.glob(f"{base_name}*.jsonl"))
        if not log_files:
            return

        enqueued: dict[str, str] = {}   # condition_id -> market_slug
        redeemed: set[str] = set()      # condition_ids successfully redeemed
        abandoned: set[str] = set()     # condition_ids abandoned after max attempts

        for log_file in log_files:
            try:
                with open(log_file) as f:
                    for line in f:
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue

                        typ = rec.get("type", "")

                        # Catch both old ("redemption_started") and new ("redemption_enqueued")
                        if typ in ("redemption_enqueued", "redemption_started"):
                            cid = rec.get("condition_id")
                            slug = rec.get("market_slug")
                            if cid and slug:
                                enqueued[cid] = slug

                        if typ == "redemption" and rec.get("status") == "success":
                            cid = rec.get("condition_id")
                            if cid:
                                redeemed.add(cid)

                        # Batch successes
                        if typ == "redemption_batch" and rec.get("status") == "success":
                            for cid in rec.get("condition_ids", []):
                                redeemed.add(cid)

                        if typ == "redemption_abandoned":
                            cid = rec.get("condition_id")
                            if cid:
                                abandoned.add(cid)
            except Exception as exc:
                print(f"  [STARTUP] Error scanning {log_file.name}: {exc}")

        # Enqueue any unredeemed wins (skip redeemed + abandoned)
        unredeemed = set(enqueued.keys()) - redeemed - abandoned
        # Also skip items already in the queue
        already_queued = {it.condition_id for it in self.redemption_queue.items}
        unredeemed -= already_queued

        for cid in unredeemed:
            slug = enqueued[cid]
            self.redemption_queue.enqueue(cid, slug)

        if unredeemed:
            print(f"  [STARTUP] Found {len(unredeemed)} unredeemed winning position(s) "
                  f"across {len(log_files)} log file(s) — added to queue")

    # ── Diagnostics & logging ──────────────────────────────────────────────

    def _log_diagnostic(self, snapshot: Snapshot, decision: Decision):
        """Log a signal diagnostic snapshot every 60s."""
        # None-safe rounder: dict.get(key, default) only substitutes the
        # default when the key is MISSING, not when the value is None.
        # Several ctx fields (_book_age_ms, _chainlink_age_ms, etc.) are
        # explicitly set to None by live_trader when no feed data is
        # available, which would crash round(None, 1).
        def _r(key, default, ndigits):
            v = self.ctx.get(key, default)
            if v is None:
                v = default
            return round(v, ndigits)

        hist = self.ctx.get("price_history", [])
        ts_hist = self.ctx.get("ts_history", [])
        raw_sigma = self.signal._compute_vol(
            hist[-self.signal.vol_lookback_s:],
            ts_hist[-self.signal.vol_lookback_s:] if ts_hist else None,
        ) if len(hist) >= self.signal.vol_lookback_s else 0.0
        sigma_per_s = self.signal._smoothed_sigma(raw_sigma, self.ctx) if raw_sigma > 0 else 0.0
        tau = snapshot.time_remaining_s
        dyn_threshold = self.signal.edge_threshold * (
            1.0 + self.signal.early_edge_mult * math.sqrt(tau / self.signal.window_duration)
        ) if tau > 0 else self.signal.edge_threshold

        ask_up = snapshot.best_ask_up
        ask_down = snapshot.best_ask_down
        bid_up = snapshot.best_bid_up
        bid_down = snapshot.best_bid_down
        edge_up = edge_down = p_model = 0.0
        spread_up = spread_down = 0.0

        # Snapshot fields can be None during warmup / missing book. Use
        # safe defaults so we can still emit a diagnostic row.
        cl_price = snapshot.chainlink_price if snapshot.chainlink_price is not None else 0.0
        ws_price = snapshot.window_start_price if snapshot.window_start_price is not None else 0.0

        if (ask_up and ask_down and bid_up and bid_down
                and 0 < ask_up < 1 and 0 < ask_down < 1
                and sigma_per_s > 0 and tau > 0
                and cl_price > 0 and ws_price > 0):
            spread_up = ask_up - bid_up
            spread_down = ask_down - bid_down
            delta = (cl_price - ws_price) / ws_price
            z_raw = delta / (sigma_per_s * math.sqrt(tau))
            z = max(-self.signal.max_z, min(self.signal.max_z, z_raw))
            p_model = self.signal._p_model(z, tau, self.ctx)
            # Maker mode: edge at bid, 0% fee
            edge_up = p_model - bid_up - self.signal.spread_edge_penalty * spread_up
            edge_down = (1.0 - p_model) - bid_down - self.signal.spread_edge_penalty * spread_down

        self._log({
            "type": "diagnostic",
            "ts": datetime.now(timezone.utc).isoformat(),
            "market_slug": snapshot.market_slug,
            "tau": round(tau, 0),
            "chainlink_price": round(cl_price, 2),
            "window_start_price": round(ws_price, 2),
            "delta": round(cl_price - ws_price, 2),
            "sigma_per_s": f"{sigma_per_s:.2e}",
            "p_model": round(p_model, 4),
            "ask_up": ask_up,
            "ask_down": ask_down,
            "spread_up": round(spread_up, 4) if spread_up else None,
            "spread_down": round(spread_down, 4) if spread_down else None,
            "edge_up": round(edge_up, 4),
            "edge_down": round(edge_down, 4),
            "dyn_threshold": round(dyn_threshold, 4),
            "dyn_threshold_down": _r("_dyn_threshold_down", dyn_threshold, 4),
            "best_edge": round(max(edge_up, edge_down), 4),
            "edge_gap": round(max(edge_up, edge_down) - dyn_threshold, 4),
            "toxicity": _r("_toxicity", 0.0, 4),
            "vpin": _r("_vpin", 0.0, 4),
            "oracle_lag": _r("_oracle_lag", 0.0, 6),
            "regime_z_factor": _r("_regime_z_factor", 1.0, 4),
            "signal_trigger_source": self.ctx.get("_signal_trigger_source"),
            "signal_trigger_age_ms": _r("_signal_trigger_age_ms", 0.0, 1),
            "signal_trigger_feed_age_ms": _r("_signal_trigger_feed_age_ms", 0.0, 1),
            "signal_eval_ms": _r("_signal_eval_ms", 0.0, 1),
            "decision_total_ms": _r("_decision_total_ms", 0.0, 1),
            "chainlink_age_ms": _r("_chainlink_age_ms", 0.0, 1),
            "binance_age_ms": _r("_binance_age_ms", 0.0, 1),
            "book_age_ms": _r("_book_age_ms", 0.0, 1),
            "down_bonus_active": self.ctx.get("_down_bonus_active", False),
            "down_share": _r("_down_share", 0.5, 4),
            "reason": decision.reason,
            "hist_len": len(hist),
            "evals": self.signal_eval_count,
        })

    def _log_flat_summary(self):
        """Log end-of-window summary of FLAT reason distribution."""
        total = sum(self.flat_reason_counts.values())
        if total == 0:
            return
        sorted_reasons = sorted(self.flat_reason_counts.items(),
                                key=lambda x: -x[1])
        summary = {k: v for k, v in sorted_reasons}
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
        # Serialize all_results so the display PnL persists across restarts.
        # Without this, restarting the bot wipes the in-memory list and the
        # display shows $0 PnL for the new session even though bankroll
        # reflects prior trades. Cap the persisted history at 5000 trades
        # to keep the state file small.
        results_serialized = [
            {
                "fill": {
                    "market_slug": r.fill.market_slug,
                    "side": r.fill.side,
                    "entry_ts_ms": r.fill.entry_ts_ms,
                    "time_remaining_s": r.fill.time_remaining_s,
                    "entry_price": r.fill.entry_price,
                    "fee_per_share": r.fill.fee_per_share,
                    "shares": r.fill.shares,
                    "cost_usd": r.fill.cost_usd,
                    "signal_name": r.fill.signal_name,
                    "decision_reason": r.fill.decision_reason,
                    "btc_at_fill": r.fill.btc_at_fill,
                    "start_price": r.fill.start_price,
                    "expected_low": r.fill.expected_low,
                    "expected_high": r.fill.expected_high,
                },
                "outcome_up": r.outcome_up,
                "final_btc": r.final_btc,
                "payout": r.payout,
                "pnl": r.pnl,
                "pnl_pct": r.pnl_pct,
            }
            for r in self.all_results[-5000:]
        ]
        data = {
            "bankroll": round(self.bankroll, 2),
            "initial_bankroll": round(self.initial_bankroll, 2),
            "windows_seen": self.windows_seen,
            "windows_traded": self.windows_traded,
            # Per-process display counters (derived from in-memory list).
            # These reset to the trimmed-list view after a restart, which
            # is fine for "session display" but not for lifetime totals —
            # see lifetime_* fields below.
            "total_trades": len(self.all_results),
            "wins": len(wins),
            "losses": len(self.all_results) - len(wins),
            "total_pnl": round(sum(r.pnl for r in self.all_results), 2),
            # Lifetime scalars — incremented on every fill resolution and
            # NOT reconstructed from all_results, so they survive the
            # 5000-trade truncation during save. These are the source of
            # truth for cumulative PnL across the bot's lifetime.
            "lifetime_pnl": round(self.lifetime_pnl, 2),
            "lifetime_wins": self.lifetime_wins,
            "lifetime_losses": self.lifetime_losses,
            "lifetime_trades": self.lifetime_trades,
            "total_fees": round(self.total_fees, 2),
            "peak_bankroll": round(self.peak_bankroll, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_dd_pct": round(self.max_dd_pct, 4),
            "circuit_breaker": self.circuit_breaker_tripped,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "all_results": results_serialized,
        }
        # Atomic write: dump to temp file, fsync, rename. Without
        # tmp+rename, a crash mid-write leaves a partial JSON file that
        # fails to parse on next load — silently losing all session
        # state. Log save failures loudly instead of swallowing.
        try:
            tmp = self.state_file.with_suffix(self.state_file.suffix + ".tmp")
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                import os as _os
                _os.fsync(f.fileno())
            tmp.replace(self.state_file)
        except Exception as exc:
            print(f"  [STATE] save_state FAILED for {self.state_file.name}: "
                  f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _restore_results(serialized: list) -> list[TradeResult]:
        """Reconstruct TradeResult objects from save_state's serialized form."""
        out = []
        for r in serialized:
            f = r.get("fill", {})
            try:
                fill = Fill(
                    market_slug=f.get("market_slug", ""),
                    side=f.get("side", ""),
                    entry_ts_ms=int(f.get("entry_ts_ms", 0)),
                    time_remaining_s=float(f.get("time_remaining_s", 0.0)),
                    entry_price=float(f.get("entry_price", 0.0)),
                    fee_per_share=float(f.get("fee_per_share", 0.0)),
                    shares=float(f.get("shares", 0.0)),
                    cost_usd=float(f.get("cost_usd", 0.0)),
                    signal_name=f.get("signal_name", ""),
                    decision_reason=f.get("decision_reason", ""),
                    btc_at_fill=float(f.get("btc_at_fill", 0.0)),
                    start_price=float(f.get("start_price", 0.0)),
                    expected_low=float(f.get("expected_low", 0.0)),
                    expected_high=float(f.get("expected_high", 0.0)),
                )
                out.append(TradeResult(
                    fill=fill,
                    outcome_up=int(r.get("outcome_up", 0)),
                    final_btc=float(r.get("final_btc", 0.0)),
                    payout=float(r.get("payout", 0.0)),
                    pnl=float(r.get("pnl", 0.0)),
                    pnl_pct=float(r.get("pnl_pct", 0.0)),
                ))
            except (TypeError, ValueError):
                continue
        return out

    @classmethod
    def load_state(cls, state_file: Path) -> dict | None:
        if not state_file.exists():
            return None
        try:
            with open(state_file) as f:
                return json.load(f)
        except Exception:
            return None
