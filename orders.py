"""OrderMixin: limit order placement, cancellation, polling, and FOK exit execution."""

from __future__ import annotations

import json
import time as _time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from backtest import Snapshot, Decision, Fill, TradeResult, poly_fee

if TYPE_CHECKING:
    from tracker import LiveTradeTracker


class OrderMixin:
    """Order management methods mixed into LiveTradeTracker."""

    # ── Helpers ────────────────────────────────────────────────────────────

    def _event(self: "LiveTradeTracker", msg: str):
        """Log an order event to the display buffer."""
        clean = msg.lstrip("\n").rstrip()
        self.event_log.append(clean)

    def _get_open_order(self: "LiveTradeTracker", side: str) -> dict | None:
        """Return the open order dict for a given side ("UP"/"DOWN"), or None."""
        for o in self.open_orders:
            if o.get("side") == side:
                return o
        return None

    @staticmethod
    def _fee_for_fill(price: float, shares: float) -> tuple[float, float]:
        """Compute (fee_total, fee_per_share) for a Polymarket fill.

        IMPORTANT: the bot only places GTC limit orders that rest on the
        bid (see _place_limit_order: limit_price = best_bid). When those
        get hit by a market taker, WE are the maker → 0% fee on
        Polymarket binary markets. The reviewer's P12.3 fix that applied
        2%·p·(1−p) was WRONG — it deducted fees from maker fills and
        caused visible bankroll drift (operator: "we're profiting but
        bankroll is going down").

        Returns (0.0, 0.0) for the maker case. The taker rate would only
        apply if our limit order somehow crossed the spread and immediately
        matched as a self-taker, which is extremely rare in practice and
        not worth tracking until the operator says otherwise.
        """
        return 0.0, 0.0

    @staticmethod
    def _model_log_fields(order: dict) -> dict:
        """Extract model_snapshot fields from an order dict for JSONL logging."""
        ms = order.get("model_snapshot")
        if not ms:
            return {}
        return {
            "p_model": ms["p_model"],
            "p_side": ms["p_side"],
            "cost_basis": ms["cost_basis"],
            "edge": ms["edge"],
            "sigma_per_s": ms["sigma_per_s"],
            "tau": ms["tau"],
            "expected_low": ms["expected_low"],
            "expected_high": ms["expected_high"],
            "dyn_threshold": ms["dyn_threshold"],
        }

    @staticmethod
    def _model_fill_line(order: dict) -> str:
        """Return a second-line string for model state at fill time, or ''."""
        ms = order.get("model_snapshot")
        if not ms:
            return ""
        side = order.get("side", "up").lower()
        return (
            f"\n    Model: p_{side}={ms['p_side']:.4f} "
            f"cost={ms['cost_basis']:.4f} edge={ms['edge']:.4f}"
            f" | Range: ${ms['expected_low']:,.0f}-${ms['expected_high']:,.0f}"
            f" | tau={ms['tau']:.0f}s"
        )

    def _latency_log_fields(self: "LiveTradeTracker") -> dict:
        # None-safe rounder: dict.get(key, default) only substitutes the
        # default when the key is MISSING, not when the value is None.
        # live_trader explicitly sets these *_age_ms fields to None when
        # the corresponding feed has no data, which would crash round(None, 1).
        def _r(key, ndigits):
            v = self.ctx.get(key, 0.0)
            if v is None:
                v = 0.0
            return round(v, ndigits)
        return {
            "signal_trigger_source": self.ctx.get("_signal_trigger_source"),
            "signal_trigger_age_ms": _r("_signal_trigger_age_ms", 1),
            "signal_trigger_feed_age_ms": _r("_signal_trigger_feed_age_ms", 1),
            "signal_eval_ms": _r("_signal_eval_ms", 1),
            "decision_total_ms": _r("_decision_total_ms", 1),
            "chainlink_age_ms": _r("_chainlink_age_ms", 1),
            "binance_age_ms": _r("_binance_age_ms", 1),
            "book_age_ms": _r("_book_age_ms", 1),
        }

    # ── Fill accounting helpers ─────────────────────────────────────────────
    #
    # Accounting invariant: at order placement we debit `cost_est` from
    # bankroll. Each fill (full or partial) records a pending_fill row
    # for the newly-filled shares and increments `_cumulative_filled_cost`
    # on the order. Partials do NOT refund bankroll — that would be a
    # double-count because the original cost_est has already been debited.
    # On terminal events (MATCHED or CANCELLED/EXPIRED), we refund
    # `cost_est - _cumulative_filled_cost` — the portion we reserved but
    # never actually spent.
    #
    # Prior bug: the 6 fill/cancel sites each recomputed the refund as
    # `cost_est - new_fill_cost`, which ignored prior partials and either
    # over-refunded (phantom bankroll credit) or under-refunded (leak).

    @staticmethod
    def _terminal_refund(order: dict) -> float:
        """Refund owed when an order reaches a terminal state.

        After a terminal event (MATCHED with full shares, or CANCELLED),
        the order's cumulative filled cost is final. The bankroll reservation
        (cost_est) minus cumulative filled cost is what we reserved but
        never spent, and should be returned to bankroll.
        """
        already = float(order.get("_cumulative_filled_cost", 0.0))
        return max(0.0, float(order["cost_est"]) - already)

    def _record_partial_fill(
        self: "LiveTradeTracker",
        order: dict,
        new_fill_shares: float,
        new_fill_cost: float,
        fee: float,
        size_matched_total: float,
    ):
        """Append a pending_fill row and update cumulative tracking.

        Does NOT touch bankroll. Does NOT increment window_trade_count or
        position_count for the second+ partial of the same order — those
        are per-order counters, not per-fill-event.
        """
        order["_cumulative_filled_cost"] = (
            float(order.get("_cumulative_filled_cost", 0.0)) + float(new_fill_cost)
        )
        order["_cumulative_filled_shares"] = (
            float(order.get("_cumulative_filled_shares", 0.0)) + float(new_fill_shares)
        )
        order["_last_matched"] = float(size_matched_total)
        self.total_fees += float(fee)

        self.pending_fills.append({
            "market_slug": order["market_slug"],
            "side": order["side"],
            "cost_usd": new_fill_cost,
            "shares": new_fill_shares,
            "fee": fee,
            "order_id": order["order_id"],
            "entry_ts": order["placed_ts"],
            "fill_ts_unix": _time.time(),
            "time_remaining_s": order["time_remaining_s"],
            "chainlink_price": order["chainlink_price"],
            "window_start_price": order["window_start_price"],
            "model_snapshot": order.get("model_snapshot"),
        })

        # Per-order (not per-event) counters: mark on first fill only.
        # A single 5-share order filled in 3 partials should count as ONE
        # trade for max_trades_per_window and ONE position.
        if not order.get("_counted", False):
            order["_counted"] = True
            self.window_trade_count += 1
            self.position_count += 1
            self._record_window_fill_side(order["side"])

    # ── Cancel ─────────────────────────────────────────────────────────────

    def _cancel_single_order(self: "LiveTradeTracker", order: dict, reason: str) -> bool:
        """Cancel one order, refund reserved bankroll, remove from open_orders.

        Returns True if actually cancelled.
        Returns False if cancel API failed OR order was already filled
        (filled orders are moved to pending_fills automatically).
        """
        order_id = order["order_id"]
        if not self.dry_run:
            try:
                self.client.cancel(order_id)
            except Exception as exc:
                if self.debug:
                    self._event(f"[CANCEL] error cancelling {order_id[:12]}...: {exc}")
                return False

            # Verify cancel — order may have filled in the race window
            try:
                resp = self.client.get_order(order_id)
                status = str(resp.get("status", "unknown")).upper()
                size_matched = float(resp.get("size_matched", 0) or 0)
                original_size = float(
                    resp.get("original_size", order["shares"]) or order["shares"]
                )
                fill_pct = (
                    size_matched / original_size if original_size > 0 else 0
                )

                # M3 fix: only treat as fully filled when the exchange
                # explicitly says MATCHED. The 0.99 fill_pct heuristic
                # used to mean "close enough" but the residual could
                # later trickle in as a surprise position the bot had
                # already removed from open_orders, going untracked.
                if status == "MATCHED":
                    # Order filled before cancel — treat as a terminal fill.
                    prev_matched = float(order.get("_last_matched", 0) or 0)
                    new_fill_shares = size_matched - prev_matched
                    if new_fill_shares >= 0.001:
                        new_fill_cost = new_fill_shares * order["price"]
                        fee, _ = self._fee_for_fill(order["price"], new_fill_shares)
                        new_fill_cost += fee
                        self._record_partial_fill(
                            order, new_fill_shares, new_fill_cost, fee, size_matched,
                        )
                        entry_px = order["price"]
                        self._event(
                            f"[CANCEL->FILL] {order['side']} "
                            f"{new_fill_shares:.1f}sh @ {entry_px:.4f} "
                            f"(${new_fill_cost:.2f}) — filled before cancel"
                            + self._model_fill_line(order)
                        )
                        self._log({
                            "type": "limit_fill",
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "order_id": order_id,
                            "side": order["side"],
                            "shares": round(new_fill_shares, 2),
                            "price": order["price"],
                            "cost_usd": round(new_fill_cost, 2),
                            "note": f"filled_before_cancel: {reason}",
                            **self._model_log_fields(order),
                        })
                    # Terminal: refund the unused portion of the reservation.
                    refund = self._terminal_refund(order)
                    self.bankroll += refund
                    self.signal.bankroll = self.bankroll
                    self.open_orders = [
                        o for o in self.open_orders
                        if o["order_id"] != order_id
                    ]
                    return False

                elif size_matched > 0:
                    # Partial fill before cancel — record the new partial,
                    # refund the unreserved portion, remove the order.
                    prev_matched = float(order.get("_last_matched", 0) or 0)
                    new_fill_shares = size_matched - prev_matched

                    if new_fill_shares >= 0.001:
                        new_fill_cost = new_fill_shares * order["price"]
                        fee, _ = self._fee_for_fill(order["price"], new_fill_shares)
                        new_fill_cost += fee
                        self._record_partial_fill(
                            order, new_fill_shares, new_fill_cost, fee, size_matched,
                        )
                        entry_px = order["price"]
                        self._event(
                            f"[CANCEL->PARTIAL] {order['side']} "
                            f"{new_fill_shares:.1f}/{original_size:.1f}sh "
                            f"@ {entry_px:.4f} (${new_fill_cost:.2f})"
                            + self._model_fill_line(order)
                        )
                        self._log({
                            "type": "partial_fill",
                            "ts": datetime.now(timezone.utc).isoformat(),
                            "order_id": order_id,
                            "side": order["side"],
                            "filled_shares": round(new_fill_shares, 2),
                            "price": order["price"],
                            "cost_usd": round(new_fill_cost, 2),
                            "note": f"partial_before_cancel: {reason}",
                            **self._model_log_fields(order),
                        })
                    # Terminal: refund the unused portion of the reservation.
                    refund = self._terminal_refund(order)
                    self.bankroll += refund
                    self.signal.bankroll = self.bankroll
                    self.open_orders = [
                        o for o in self.open_orders
                        if o["order_id"] != order_id
                    ]
                    self._log({
                        "type": "limit_cancel",
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "order_id": order_id,
                        "side": order["side"],
                        "status": "cancelled_partial_done",
                        "refund": round(refund, 2),
                    })
                    return False
            except Exception as verify_exc:
                # CRITICAL: do NOT refund or remove the order. The cancel
                # API call succeeded, but we don't know whether Polymarket
                # actually cancelled it OR matched it in the race window.
                # Falling through to refund here used to lose ~$2-3 per
                # event when the order had actually filled — the position
                # would still exist on Polymarket but the bot would treat
                # the funds as returned. Mark for re-verification on the
                # next decision cycle and leave it in open_orders.
                order["_verify_pending"] = True
                order["_verify_pending_since"] = _time.time()
                self._event(
                    f"[CANCEL WARNING] Could not verify "
                    f"{order_id[:12]}...: {verify_exc} — keeping order in "
                    f"open_orders for re-verification"
                )
                self._log({
                    "type": "cancel_verify_failed",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "order_id": order_id,
                    "side": order["side"],
                    "cost_est": round(order["cost_est"], 2),
                    "error": str(verify_exc),
                    "action": "deferred_pending_reverify",
                })
                return False

        # Refund only the UNFILLED portion of the reservation.
        already_filled = float(order.get("_cumulative_filled_cost", 0.0))
        unfilled_refund = self._terminal_refund(order)
        self.bankroll += unfilled_refund
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
            "refund": round(unfilled_refund, 2),
            "filled_before_cancel": round(already_filled, 2),
        })
        self._event(
            f"[CANCEL] {order['side']} {order['shares']:.1f}sh "
            f"@ {order['price']:.4f} — {reason}"
        )
        return True

    # ── Place limit order ──────────────────────────────────────────────────

    def _place_limit_order(
        self: "LiveTradeTracker",
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

        # Capture model state at order time for fill logging
        p_model = self.ctx.get("_p_model_raw", 0.0)
        expected_range = self.ctx.get("_expected_range", {})
        p_side = p_model if side_label == "UP" else 1.0 - p_model
        cost_basis = self.ctx.get(
            "_cost_down" if side_label == "DOWN" else "_cost_up", limit_price
        )
        model_snapshot = {
            "p_model": round(p_model, 4),
            "p_side": round(p_side, 4),
            "sigma_per_s": self.ctx.get("_sigma_per_s", 0.0),
            "tau": round(snapshot.time_remaining_s, 0),
            "dyn_threshold": round(self.ctx.get("_dyn_threshold_down" if side_label == "DOWN" else "_dyn_threshold_up", 0.0), 4),
            "expected_low": round(expected_range.get("expected_low", 0.0), 2),
            "expected_high": round(expected_range.get("expected_high", 0.0), 2),
            "cost_basis": round(cost_basis, 4),
            "fill_price": limit_price,
            "edge": round(decision.edge, 4),
        }

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
            "signal_trigger_ts_ms": int(self.ctx.get("_signal_trigger_ts_ms", 0) or 0),
            **model_snapshot,
            **self._latency_log_fields(),
        }

        if self.dry_run:
            trade_record["dry_run"] = True
            trade_record["status"] = "dry_run"
            self._log(trade_record)
            self._event(
                f"[DRY RUN] Would place limit: BUY {side_label} "
                f"{shares:.1f}sh @ {limit_price:.4f} (${cost_est:.2f})"
                f" | BTC: ${snapshot.chainlink_price:,.2f}"
            )
            # Simulate resting order in dry run
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
                "model_snapshot": model_snapshot,
                "edge_at_place": round(decision.edge, 4),
            })
            return

        # 2026-04-09: API 425 "service not ready" backoff. After a 425
        # response Polymarket needs ~5-10s before accepting more orders.
        # Without backoff the bot was firing 4+ identical orders in the
        # same second, all rejected. self._service_unavailable_until is
        # a unix timestamp; if now < that ts, skip this placement entirely.
        now_ts = _time.time()
        backoff_until = getattr(self, "_service_unavailable_until", 0.0)
        if now_ts < backoff_until:
            remaining = backoff_until - now_ts
            trade_record["status"] = "skipped_backoff"
            trade_record["error"] = f"service backoff ({remaining:.1f}s remaining)"
            self._log(trade_record)
            return

        # Place GTC limit order via Rust OrderClient (single call: sign + HTTP/2 POST)
        try:
            post_start_ms = int(_time.time() * 1000)
            resp = self.client.place_order(token_id, limit_price, shares, "BUY", "GTC", 1000)
            post_done_ms = int(_time.time() * 1000)
        except Exception as exc:
            trade_record["status"] = "error"
            trade_record["error"] = str(exc)
            self._log(trade_record)
            self._event(f"[LIMIT ORDER ERROR] {exc}")
            # Detect "service not ready" / 425 in error message and arm backoff
            err_str = str(exc).lower()
            if "service not ready" in err_str or "425" in err_str:
                self._service_unavailable_until = _time.time() + 10.0
                self._event(f"[LIMIT] Service not ready — backoff 10s")
            return

        success = resp.get("success", False)
        order_id = resp.get("orderID") or resp.get("id", "")
        status = str(resp.get("status", "unknown")).upper()

        trade_record["order_id"] = order_id
        trade_record["status"] = status
        trade_record["success"] = success
        trade_record["response"] = resp
        trade_record["order_post_ms"] = post_done_ms - post_start_ms
        trigger_ts_ms = int(self.ctx.get("_signal_trigger_ts_ms", 0) or 0)
        if trigger_ts_ms > 0:
            trade_record["signal_to_post_ms"] = max(0, post_start_ms - trigger_ts_ms)
            trade_record["signal_to_ack_ms"] = max(0, post_done_ms - trigger_ts_ms)

        if not success:
            trade_record["filled"] = False
            self._log(trade_record)
            err_msg = resp.get("errorMsg", "")
            self._event(
                f"[LIMIT] {side_label} {shares:.1f}sh @ {limit_price:.4f} "
                f"-> REJECTED (status={status}, err={err_msg})"
            )
            return

        self._log(trade_record)

        # Check if the order was immediately filled
        taking = resp.get("takingAmount", "")
        making = resp.get("makingAmount", "")
        filled_shares = float(taking) if taking else 0.0
        if status == "MATCHED" and filled_shares > 0:
            # makingAmount = gross cash leg (limit_price × shares). It is
            # NOT net of the Polymarket binary-market taker fee. Apply it
            # explicitly so bankroll matches the actual on-chain debit.
            gross_cost = float(making) if making else filled_shares * limit_price
            entry_price = gross_cost / filled_shares if filled_shares > 0 else limit_price
            taker_fee, _fps = self._fee_for_fill(entry_price, filled_shares)
            actual_cost = gross_cost + taker_fee
            self.bankroll -= actual_cost
            self.signal.bankroll = self.bankroll
            self.total_fees += taker_fee

            self.pending_fills.append({
                "market_slug": snapshot.market_slug,
                "side": side_label,
                "cost_usd": actual_cost,
                "shares": filled_shares,
                "fee": taker_fee,
                "order_id": order_id,
                "entry_ts": now_iso,
                "fill_ts_unix": _time.time(),
                "time_remaining_s": snapshot.time_remaining_s,
                "chainlink_price": snapshot.chainlink_price,
                "window_start_price": snapshot.window_start_price,
                "model_snapshot": model_snapshot,
                "signal_trigger_ts_ms": trigger_ts_ms,
                "order_posted_ts_ms": post_start_ms,
            })
            self.last_fill_ts_ms = snapshot.ts_ms
            self.window_trade_count += 1
            self.position_count += 1
            self._record_window_fill_side(side_label)

            entry_px = actual_cost / filled_shares if filled_shares > 0 else 0
            ms = model_snapshot
            self._event(
                f"[LIMIT FILLED] BUY {side_label} {filled_shares:.1f}sh "
                f"@ {entry_px:.4f} (${actual_cost:.2f})"
                f"\n    Model: p_{side_label.lower()}={ms['p_side']:.4f} "
                f"cost={ms['cost_basis']:.4f} edge={ms['edge']:.4f}"
                f" | Range: ${ms['expected_low']:,.0f}-${ms['expected_high']:,.0f}"
                f" | tau={ms['tau']:.0f}s"
            )
            return

        # Reserve bankroll for resting order
        self.bankroll -= cost_est
        self.signal.bankroll = self.bankroll

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
            "model_snapshot": model_snapshot,
            "edge_at_place": round(decision.edge, 4),
            "signal_trigger_ts_ms": trigger_ts_ms,
            "order_posted_ts_ms": post_start_ms,
        })

        self._event(
            f"[LIMIT] BUY {side_label} {shares:.1f}sh @ {limit_price:.4f} "
            f"(${cost_est:.2f}) -> resting (id={order_id[:12]}...)"
        )

    # ── Poll open orders ───────────────────────────────────────────────────

    def _poll_open_orders(self: "LiveTradeTracker", snapshot: Snapshot):
        """Check open limit orders for fills or cancellations."""
        if not self.open_orders:
            return

        still_open = []
        for order in self.open_orders:
            order_id = order["order_id"]

            # Dry-run: simulate fill after 5s resting
            if self.dry_run and order_id.startswith("dry_"):
                age = _time.time() - order.get("placed_ts_unix", _time.time())
                if age >= 5.0:
                    actual_cost = order["shares"] * order["price"]
                    fee, _ = self._fee_for_fill(order["price"], order["shares"])
                    actual_cost += fee
                    self.total_fees += fee
                    drift = order["cost_est"] - actual_cost
                    self.bankroll += drift
                    self.signal.bankroll = self.bankroll
                    self.pending_fills.append({
                        "market_slug": order["market_slug"],
                        "side": order["side"],
                        "cost_usd": actual_cost,
                        "shares": order["shares"],
                        "fee": fee,
                        "order_id": order_id,
                        "entry_ts": order["placed_ts"],
                        "fill_ts_unix": _time.time(),
                        "time_remaining_s": order["time_remaining_s"],
                        "chainlink_price": order["chainlink_price"],
                        "window_start_price": order["window_start_price"],
                        "model_snapshot": order.get("model_snapshot"),
                    })
                    self.last_fill_ts_ms = snapshot.ts_ms
                    self.window_trade_count += 1
                    self.position_count += 1
                    self._record_window_fill_side(order["side"])
                    entry_px = actual_cost / order["shares"] if order["shares"] > 0 else 0
                    self._event(
                        f"[DRY FILL] {order['side']} {order['shares']:.1f}sh "
                        f"@ {entry_px:.4f} (${actual_cost:.2f})"
                        + self._model_fill_line(order)
                    )
                    self._log({
                        "type": "limit_fill",
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "order_id": order_id,
                        "side": order["side"],
                        "shares": round(order["shares"], 2),
                        "price": order["price"],
                        "cost_usd": round(actual_cost, 2),
                        "note": "dry_run_fill",
                        **self._model_log_fields(order),
                    })
                else:
                    still_open.append(order)
                continue

            try:
                resp = self.client.get_order(order_id)
                # Successful poll clears any verify-pending flag from a
                # prior cancel-verify failure (P11.1) — we now have a
                # definitive status from the exchange.
                if order.get("_verify_pending"):
                    order.pop("_verify_pending", None)
                    order.pop("_verify_pending_since", None)
            except Exception as exc:
                if self.debug:
                    self._event(f"[POLL] error checking {order_id[:12]}...: {exc}")
                # P12.10: TTL on _verify_pending. If verification has
                # been failing for >60s, force a final cleanup: refund
                # the unfilled portion to bankroll, drop from open_orders,
                # and log loudly. The position (if any) becomes a
                # phantom that the operator must reconcile manually,
                # but at least the bot's bankroll stays usable instead
                # of leaking the reservation forever.
                pending_since = order.get("_verify_pending_since")
                if pending_since and (_time.time() - pending_since) > 60.0:
                    already_filled = float(order.get("_cumulative_filled_cost", 0.0))
                    refund = self._terminal_refund(order)
                    self.bankroll += refund
                    self.signal.bankroll = self.bankroll
                    self._event(
                        f"[POLL] TTL force-cleanup {order_id[:12]}... after "
                        f"{(_time.time() - pending_since):.0f}s of verify failures. "
                        f"Refunded ${refund:.2f}. Reconcile on-chain manually."
                    )
                    self._log({
                        "type": "verify_pending_ttl_cleanup",
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "order_id": order_id,
                        "side": order["side"],
                        "refund": round(refund, 2),
                        "filled_before_cleanup": round(already_filled, 2),
                        "warning": "phantom_position_possible",
                    })
                    continue  # don't append to still_open
                still_open.append(order)
                continue

            status = str(resp.get("status", "unknown")).upper()
            size_matched = float(resp.get("size_matched", 0) or 0)
            original_size = float(resp.get("original_size", order["shares"]) or order["shares"])
            fill_pct = size_matched / original_size if original_size > 0 else 0

            # M3 fix: only treat as fully filled when the exchange explicitly
            # says MATCHED. The 0.99 fill_pct heuristic missed the residual.
            if status == "MATCHED":
                # Terminal fill. Record any residual delta and refund the
                # unused reservation (cost_est - cumulative_filled_cost).
                prev_matched = float(order.get("_last_matched", 0) or 0)
                new_fill_shares = size_matched - prev_matched
                if new_fill_shares >= 0.001:
                    new_fill_cost = new_fill_shares * order["price"]
                    fee, _ = self._fee_for_fill(order["price"], new_fill_shares)
                    new_fill_cost += fee
                    self._record_partial_fill(
                        order, new_fill_shares, new_fill_cost, fee, size_matched,
                    )
                    self.last_fill_ts_ms = snapshot.ts_ms
                    entry_px = order["price"]
                    self._event(
                        f"[LIMIT FILLED] {order['side']} {new_fill_shares:.1f}sh "
                        f"@ {entry_px:.4f} (${new_fill_cost:.2f})"
                        + self._model_fill_line(order)
                    )
                    self._log({
                        "type": "limit_fill",
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "order_id": order_id,
                        "side": order["side"],
                        "shares": round(new_fill_shares, 2),
                        "price": order["price"],
                        "cost_usd": round(new_fill_cost, 2),
                        "fee": round(fee, 4),
                        **self._model_log_fields(order),
                    })
                # Terminal: refund whatever portion of cost_est wasn't spent.
                refund = self._terminal_refund(order)
                self.bankroll += refund
                self.signal.bankroll = self.bankroll
                # Not appended to still_open → removed from open_orders.

            elif status in ("CANCELLED", "EXPIRED"):
                refund = self._terminal_refund(order)
                self.bankroll += refund
                self.signal.bankroll = self.bankroll
                already_filled = float(order.get("_cumulative_filled_cost", 0.0))
                self._event(
                    f"[LIMIT {status}] {order['side']} "
                    f"{order['shares']:.1f}sh @ {order['price']:.4f} "
                    f"— refunded ${refund:.2f}"
                    + (f" (kept ${already_filled:.2f} of partials)" if already_filled > 0 else "")
                )
                self._log({
                    "type": "limit_cancel",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "order_id": order_id,
                    "side": order["side"],
                    "status": status,
                    "refund": round(refund, 2),
                    "filled_before_cancel": round(already_filled, 2),
                })
            elif size_matched > float(order.get("_last_matched", 0) or 0):
                prev_matched = float(order.get("_last_matched", 0) or 0)
                new_fill_shares = size_matched - prev_matched
                new_fill_cost = new_fill_shares * order["price"]
                fee, _ = self._fee_for_fill(order["price"], new_fill_shares)
                new_fill_cost += fee
                self._record_partial_fill(
                    order, new_fill_shares, new_fill_cost, fee, size_matched,
                )
                self.last_fill_ts_ms = snapshot.ts_ms

                # Remaining for display only.
                remaining_shares = max(0.0,
                    order["shares"] - order["_cumulative_filled_shares"])

                self._event(
                    f"[PARTIAL FILL] {order['side']} {new_fill_shares:.1f}sh "
                    f"@ {order['price']:.4f} (${new_fill_cost:.2f}) — "
                    f"{remaining_shares:.1f}sh still open"
                    + self._model_fill_line(order)
                )
                self._log({
                    "type": "partial_fill",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "order_id": order_id,
                    "side": order["side"],
                    "filled_shares": round(new_fill_shares, 2),
                    "remaining_shares": round(remaining_shares, 2),
                    "price": order["price"],
                    "cost_usd": round(new_fill_cost, 2),
                    **self._model_log_fields(order),
                })
                still_open.append(order)
            else:
                still_open.append(order)

        self.open_orders = still_open

    def _process_user_events(self: "LiveTradeTracker", snapshot: Snapshot):
        """Process buffered UserFeed WS events for instant fill/cancel handling.

        Called each tick BEFORE _poll_open_orders. Events from the Polymarket
        user WebSocket arrive as dicts with string keys/values.
        """
        if not hasattr(self, "user_feed") or self.user_feed is None:
            return
        if not self.open_orders:
            return

        try:
            events = self.user_feed.drain_events()
        except Exception:
            return

        if not events:
            return

        # M4: dedup by (event_id) across the entire process lifetime so
        # WS retransmits and WS+poll overlap can't double-record the
        # same event. The _last_matched cumulative tracker on each
        # order also dedupes by size, so this is belt+suspenders.
        if not hasattr(self, "_seen_user_event_ids"):
            self._seen_user_event_ids: set[str] = set()
            self._seen_event_ids_max: int = 4096

        # Index open orders by order_id for fast lookup
        order_map: dict[str, dict] = {}
        for order in self.open_orders:
            order_map[order["order_id"]] = order

        processed_ids: set[str] = set()

        for evt in events:
            # M4 dedup: per-event-id check (if exchange provides one).
            evt_id = evt.get("event_id") or evt.get("id") or ""
            if evt_id and evt_id in self._seen_user_event_ids:
                continue
            if evt_id:
                self._seen_user_event_ids.add(evt_id)
                # Bound the set size to avoid unbounded growth
                if len(self._seen_user_event_ids) > self._seen_event_ids_max:
                    # Drop the oldest half (set has no order — drop arbitrary)
                    self._seen_user_event_ids = set(
                        list(self._seen_user_event_ids)[self._seen_event_ids_max // 2:]
                    )

            order_id = evt.get("order_id", "")
            if not order_id or order_id not in order_map:
                continue

            order = order_map[order_id]
            status = evt.get("status", "").upper()
            event_type = evt.get("event_type", "").lower()

            # Fill event (full or partial).
            #
            # Polymarket emits a "trade" event_type for each match. The
            # prior version treated every trade event as a full fill and
            # added the order to processed_ids (removing it from
            # open_orders), which lost the remaining unfilled shares when
            # the match was actually partial. We now treat full vs
            # partial explicitly: only add to processed_ids when the
            # exchange explicitly says MATCHED OR size_matched >=
            # original shares (within a tolerance).
            if status == "MATCHED" or event_type == "trade":
                size_matched_str = evt.get("size_matched", "0")
                try:
                    size_matched = float(size_matched_str)
                except (ValueError, TypeError):
                    size_matched = 0.0

                # Treat a size_matched=0 "status-only" event as noise.
                # The prior fallback to order["shares"] silently promoted
                # ambiguous events into full fills, losing unfilled shares.
                if size_matched <= 0:
                    if status == "MATCHED":
                        # Fully filled per exchange but size missing —
                        # assume full original_size.
                        size_matched = float(order["shares"])
                    else:
                        # Not a fill, just a status update — skip.
                        continue

                prev_matched = float(order.get("_last_matched", 0) or 0)
                new_fill_shares = size_matched - prev_matched

                is_full = (
                    status == "MATCHED"
                    or size_matched >= float(order["shares"]) - 0.001
                )

                if new_fill_shares >= 0.001:
                    new_fill_cost = new_fill_shares * order["price"]
                    fee, _ = self._fee_for_fill(order["price"], new_fill_shares)
                    new_fill_cost += fee
                    self._record_partial_fill(
                        order, new_fill_shares, new_fill_cost, fee, size_matched,
                    )
                    self.last_fill_ts_ms = snapshot.ts_ms

                    entry_px = order["price"]
                    tag = "WS FILL" if is_full else "WS PARTIAL"
                    self._event(
                        f"[{tag}] {order['side']} {new_fill_shares:.1f}sh "
                        f"@ {entry_px:.4f} (${new_fill_cost:.2f})"
                        + self._model_fill_line(order)
                    )
                    self._log({
                        "type": "ws_fill" if is_full else "ws_partial_fill",
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "order_id": order_id,
                        "side": order["side"],
                        "shares": round(new_fill_shares, 2),
                        "price": order["price"],
                        "cost_usd": round(new_fill_cost, 2),
                        "fee": round(fee, 4),
                        **self._model_log_fields(order),
                    })

                if is_full:
                    # Terminal: refund unused reservation and mark for removal.
                    refund = self._terminal_refund(order)
                    self.bankroll += refund
                    self.signal.bankroll = self.bankroll
                    processed_ids.add(order_id)
                # Otherwise: order stays in open_orders for the next
                # partial/full event.

            elif status in ("CANCELLED", "EXPIRED"):
                refund = self._terminal_refund(order)
                self.bankroll += refund
                self.signal.bankroll = self.bankroll
                already_filled = float(order.get("_cumulative_filled_cost", 0.0))
                self._event(
                    f"[WS {status}] {order['side']} "
                    f"{order['shares']:.1f}sh @ {order['price']:.4f} "
                    f"— refunded ${refund:.2f}"
                    + (f" (kept ${already_filled:.2f} of partials)" if already_filled > 0 else "")
                )
                self._log({
                    "type": "ws_cancel",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "order_id": order_id,
                    "side": order["side"],
                    "status": status,
                    "refund": round(refund, 2),
                    "filled_before_cancel": round(already_filled, 2),
                })
                processed_ids.add(order_id)

        # Remove processed orders
        if processed_ids:
            self.open_orders = [
                o for o in self.open_orders if o["order_id"] not in processed_ids
            ]

    def _cancel_open_orders(self: "LiveTradeTracker"):
        """Cancel all open limit orders and refund reserved bankroll.

        Preserves verify-pending orders: ``_cancel_single_order`` sets
        ``_verify_pending=True`` on any order whose cancel API call
        succeeded but whose state we couldn't confirm via ``get_order``.
        Those orders must stay in ``open_orders`` so the next poll can
        reconcile them (CANCELLED → refund; MATCHED → add to
        pending_fills). Clobbering the list here silently leaked the
        bankroll reservation AND the on-chain position.
        """
        if not self.open_orders:
            return
        for order in list(self.open_orders):
            self._cancel_single_order(order, "end_of_window")
        # Keep verify-pending orders for later reconciliation; drop the rest.
        self.open_orders = [
            o for o in self.open_orders if o.get("_verify_pending")
        ]

    def cancel_all_orders(self: "LiveTradeTracker"):
        """Cancel all open orders — called on shutdown."""
        if self.open_orders:
            self._cancel_open_orders()
        if self.open_sell_orders:
            self._cancel_open_sell_orders()
        if self.dry_run:
            return
        try:
            resp = self.client.cancel_all()
            self._event(f"Cancelled all open orders: {resp}")
        except Exception as exc:
            self._event(f"Warning: cancel_all failed: {exc}")

    # ── Maker sell order lifecycle ────────────────────────────────────────

    def _get_open_sell_order(self: "LiveTradeTracker", side: str) -> dict | None:
        """Return the open sell order dict for a given side, or None."""
        for o in self.open_sell_orders:
            if o.get("side") == side:
                return o
        return None

    def _place_limit_sell(
        self: "LiveTradeTracker",
        fill: dict,
        snapshot: "Snapshot",
        token_id: str,
        side: str,
        sell_price: float,
    ):
        """Place a GTC limit SELL order for an existing position."""
        shares = fill["shares"]
        entry_price = fill["cost_usd"] / shares if shares > 0 else 0
        now_iso = datetime.now(timezone.utc).isoformat()
        now_unix = _time.time()

        if self.dry_run:
            order_id = f"drysell_{side}_{int(now_unix)}"
            sell_order = {
                "order_id": order_id,
                "side": side,
                "price": sell_price,
                "shares": shares,
                "market_slug": fill["market_slug"],
                "placed_ts": now_iso,
                "placed_ts_unix": now_unix,
                "source_fill": fill,
                "entry_price": entry_price,
            }
            self.open_sell_orders.append(sell_order)
            self._event(
                f"[SELL] {side} {shares:.1f}sh @ {sell_price:.4f} "
                f"(entry {entry_price:.4f}) -> resting [DRY]"
            )
            self._log({
                "type": "maker_sell_placed",
                "ts": now_iso,
                "side": side,
                "price": sell_price,
                "shares": round(shares, 2),
                "entry_price": round(entry_price, 4),
                "order_id": order_id,
                "dry_run": True,
            })
            return

        try:
            resp = self.client.place_order(
                token_id, sell_price, shares, "SELL", "GTC", 1000
            )
        except Exception as exc:
            self._event(f"[SELL ERROR] {side}: {exc}")
            self._log({
                "type": "maker_sell_placed",
                "ts": now_iso,
                "side": side,
                "price": sell_price,
                "shares": round(shares, 2),
                "status": "error",
                "error": str(exc),
            })
            return

        success = resp.get("success", False)
        order_id = resp.get("orderID") or resp.get("id", "")
        status = str(resp.get("status", "unknown")).upper()

        if not success:
            err_msg = resp.get("errorMsg", "")
            self._event(
                f"[SELL] {side} {shares:.1f}sh @ {sell_price:.4f} "
                f"-> REJECTED ({err_msg})"
            )
            self._log({
                "type": "maker_sell_placed",
                "ts": now_iso,
                "side": side,
                "price": sell_price,
                "shares": round(shares, 2),
                "status": "rejected",
                "error": err_msg,
            })
            return

        # Check if immediately filled
        taking = resp.get("takingAmount", "")
        proceeds = float(taking) if taking else 0.0
        if status == "MATCHED" and proceeds > 0:
            self._process_sell_fill(fill, proceeds, sell_price)
            self._log({
                "type": "maker_sell_placed",
                "ts": now_iso,
                "side": side,
                "price": sell_price,
                "shares": round(shares, 2),
                "order_id": order_id,
                "status": "immediate_fill",
                "proceeds": round(proceeds, 2),
            })
            return

        sell_order = {
            "order_id": order_id,
            "side": side,
            "price": sell_price,
            "shares": shares,
            "market_slug": fill["market_slug"],
            "placed_ts": now_iso,
            "placed_ts_unix": now_unix,
            "source_fill": fill,
            "entry_price": entry_price,
        }
        self.open_sell_orders.append(sell_order)
        self._event(
            f"[SELL] {side} {shares:.1f}sh @ {sell_price:.4f} "
            f"(entry {entry_price:.4f}) -> resting (id={order_id[:12]}...)"
        )
        self._log({
            "type": "maker_sell_placed",
            "ts": now_iso,
            "side": side,
            "price": sell_price,
            "shares": round(shares, 2),
            "entry_price": round(entry_price, 4),
            "order_id": order_id,
            "status": "resting",
        })

    def _process_sell_fill(
        self: "LiveTradeTracker",
        fill: dict,
        proceeds: float,
        sell_price: float,
    ):
        """Process a filled sell order: update bankroll, remove position, log."""
        entry_price = fill["cost_usd"] / fill["shares"] if fill["shares"] > 0 else 0
        exit_pnl = proceeds - fill["cost_usd"]

        self.bankroll += proceeds
        self.signal.bankroll = self.bankroll

        self.pending_fills = [f for f in self.pending_fills if f is not fill]
        self.exited_sides.add(fill["side"])
        self.position_count = max(0, self.position_count - 1)

        self._event(
            f"[SELL FILLED] {fill['side']} {fill['shares']:.1f}sh "
            f"@ {sell_price:.4f} (entry {entry_price:.4f}) "
            f"PnL=${exit_pnl:+.2f} (MAKER, 0% fee)"
        )
        self._log({
            "type": "maker_exit",
            "ts": datetime.now(timezone.utc).isoformat(),
            "market_slug": fill["market_slug"],
            "side": fill["side"],
            "shares": round(fill["shares"], 2),
            "entry_price": round(entry_price, 4),
            "sell_price": round(sell_price, 4),
            "proceeds": round(proceeds, 2),
            "pnl": round(exit_pnl, 2),
            "bankroll_after": round(self.bankroll, 2),
        })
        self._record_exit_result(fill, proceeds, exit_pnl)

    def _poll_open_sell_orders(self: "LiveTradeTracker", snapshot: "Snapshot"):
        """Check open sell orders for fills or cancellations."""
        if not self.open_sell_orders:
            return

        still_open = []
        for order in self.open_sell_orders:
            order_id = order["order_id"]

            # Dry-run: simulate sell fill after 10s resting
            if self.dry_run and order_id.startswith("drysell_"):
                age = _time.time() - order.get("placed_ts_unix", _time.time())
                if age >= 10.0:
                    proceeds = order["shares"] * order["price"]
                    self._process_sell_fill(order["source_fill"], proceeds, order["price"])
                    self._log({
                        "type": "maker_exit",
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "order_id": order_id,
                        "side": order["side"],
                        "note": "dry_run_sell_fill",
                    })
                else:
                    still_open.append(order)
                continue

            try:
                resp = self.client.get_order(order_id)
            except Exception as exc:
                if self.debug:
                    self._event(f"[SELL POLL] error checking {order_id[:12]}...: {exc}")
                still_open.append(order)
                continue

            status = str(resp.get("status", "unknown")).upper()
            size_matched = float(resp.get("size_matched", 0) or 0)
            original_size = float(
                resp.get("original_size", order["shares"]) or order["shares"]
            )
            fill_pct = size_matched / original_size if original_size > 0 else 0

            # M3 fix: strict MATCHED only — see comment in _poll_open_orders.
            if status == "MATCHED":
                proceeds = size_matched * order["price"]
                self._process_sell_fill(order["source_fill"], proceeds, order["price"])

            elif status in ("CANCELLED", "EXPIRED"):
                self._event(
                    f"[SELL {status}] {order['side']} "
                    f"{order['shares']:.1f}sh @ {order['price']:.4f}"
                )
                # No bankroll refund needed — sell orders don't reserve bankroll

            else:
                still_open.append(order)

        self.open_sell_orders = still_open

    def _cancel_single_sell_order(
        self: "LiveTradeTracker", order: dict, reason: str
    ) -> bool:
        """Cancel one sell order. Returns True if cancelled."""
        order_id = order["order_id"]

        if not self.dry_run:
            try:
                self.client.cancel(order_id)
            except Exception as exc:
                if self.debug:
                    self._event(f"[SELL CANCEL] error {order_id[:12]}...: {exc}")
                return False

            # Check for race-condition fill
            try:
                resp = self.client.get_order(order_id)
                status = str(resp.get("status", "unknown")).upper()
                size_matched = float(resp.get("size_matched", 0) or 0)
                original_size = float(
                    resp.get("original_size", order["shares"]) or order["shares"]
                )
                fill_pct = (
                    size_matched / original_size if original_size > 0 else 0
                )

                # M3 fix: only treat as fully filled when the exchange
                # explicitly says MATCHED. The 0.99 fill_pct heuristic
                # used to mean "close enough" but the residual could
                # later trickle in as a surprise position the bot had
                # already removed from open_orders, going untracked.
                if status == "MATCHED":
                    proceeds = size_matched * order["price"]
                    self.open_sell_orders = [
                        o for o in self.open_sell_orders
                        if o["order_id"] != order_id
                    ]
                    self._process_sell_fill(
                        order["source_fill"], proceeds, order["price"]
                    )
                    self._event(
                        f"[SELL CANCEL->FILL] {order['side']} "
                        f"{size_matched:.1f}sh @ {order['price']:.4f} "
                        f"— filled before cancel"
                    )
                    return False
            except Exception as verify_exc:
                if self.debug:
                    self._event(
                        f"[SELL CANCEL] verify failed "
                        f"{order_id[:12]}...: {verify_exc}"
                    )

        self.open_sell_orders = [
            o for o in self.open_sell_orders if o["order_id"] != order_id
        ]
        self._event(
            f"[SELL CANCEL] {order['side']} {order['shares']:.1f}sh "
            f"@ {order['price']:.4f} — {reason}"
        )
        self._log({
            "type": "maker_sell_cancel",
            "ts": datetime.now(timezone.utc).isoformat(),
            "order_id": order_id,
            "side": order["side"],
            "reason": reason,
        })
        return True

    def _cancel_open_sell_orders(self: "LiveTradeTracker"):
        """Cancel all open sell orders (end of window / new window)."""
        if not self.open_sell_orders:
            return
        for order in list(self.open_sell_orders):
            self._cancel_single_sell_order(order, "end_of_window")
        self.open_sell_orders = []

    # ── Early exit: FOK sell ───────────────────────────────────────────────

    def _execute_exit(
        self: "LiveTradeTracker",
        fill: dict, snapshot: Snapshot, token_id: str,
        sell_price: float, ev_hold: float, ev_sell: float, reason: str,
    ) -> bool:
        """Execute a FOK market sell to exit a position. Returns True if filled."""
        shares = fill["shares"]
        entry_price = fill["cost_usd"] / shares if shares > 0 else 0
        now_iso = datetime.now(timezone.utc).isoformat()

        if self.dry_run:
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
            self._event(
                f"[DRY EXIT] {fill['side']} {shares:.1f}sh "
                f"@ {sell_price:.4f} (entry {entry_price:.4f}) "
                f"PnL=${exit_pnl:+.2f} | EV: hold={ev_hold:.3f} sell={ev_sell:.3f} "
                f"({reason})"
            )
            self._record_exit_result(fill, proceeds, exit_pnl)
            return True

        # Live: place FOK market sell via Rust OrderClient
        try:
            resp = self.client.place_market_order(token_id, shares, "SELL", 1000)
        except Exception as exc:
            self._event(f"[EXIT ERROR] {fill['side']}: {exc}")
            self._log({
                "type": "early_exit",
                "ts": now_iso,
                "side": fill["side"],
                "status": "error",
                "error": str(exc),
            })
            return False

        success = resp.get("success", False)
        status = str(resp.get("status", "unknown")).upper()

        if not success or status not in ("MATCHED", "LIVE"):
            if self.debug:
                self._event(f"[EXIT] {fill['side']} FOK not filled (status={status})")
            return False

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
        self._event(
            f"[EXIT] {fill['side']} {shares:.1f}sh "
            f"@ {sell_price:.4f} (entry {entry_price:.4f}) "
            f"PnL=${exit_pnl:+.2f} | EV: hold={ev_hold:.3f} sell={ev_sell:.3f}"
        )
        self._record_exit_result(fill, proceeds, exit_pnl)
        return True
