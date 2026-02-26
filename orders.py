"""OrderMixin: limit order placement, cancellation, polling, and FOK exit execution."""

from __future__ import annotations

import json
import time as _time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from py_clob_client.clob_types import MarketOrderArgs, OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

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

                if status == "MATCHED" or fill_pct >= 0.99:
                    # Order filled before cancel — treat as a fill
                    actual_cost = size_matched * order["price"]
                    drift = order["cost_est"] - actual_cost
                    self.bankroll += drift
                    self.signal.bankroll = self.bankroll

                    self.open_orders = [
                        o for o in self.open_orders
                        if o["order_id"] != order_id
                    ]

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
                        "model_snapshot": order.get("model_snapshot"),
                    })
                    self.window_trade_count += 1
                    self.position_count += 1

                    entry_px = (
                        actual_cost / size_matched if size_matched > 0 else 0
                    )
                    self._event(
                        f"[CANCEL->FILL] {order['side']} "
                        f"{size_matched:.1f}sh @ {entry_px:.4f} "
                        f"(${actual_cost:.2f}) — filled before cancel"
                        + self._model_fill_line(order)
                    )
                    self._log({
                        "type": "limit_fill",
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "order_id": order_id,
                        "side": order["side"],
                        "shares": round(size_matched, 2),
                        "price": order["price"],
                        "cost_usd": round(actual_cost, 2),
                        "note": f"filled_before_cancel: {reason}",
                        **self._model_log_fields(order),
                    })
                    return False

                elif size_matched > 0:
                    # Partial fill before cancel — record filled, refund unfilled
                    filled_cost = size_matched * order["price"]
                    unfilled_cost_est = order["cost_est"] - filled_cost

                    self.bankroll += unfilled_cost_est
                    self.signal.bankroll = self.bankroll

                    self.open_orders = [
                        o for o in self.open_orders
                        if o["order_id"] != order_id
                    ]

                    self.pending_fills.append({
                        "market_slug": order["market_slug"],
                        "side": order["side"],
                        "cost_usd": filled_cost,
                        "shares": size_matched,
                        "order_id": order_id,
                        "entry_ts": order["placed_ts"],
                        "fill_ts_unix": _time.time(),
                        "time_remaining_s": order["time_remaining_s"],
                        "chainlink_price": order["chainlink_price"],
                        "window_start_price": order["window_start_price"],
                        "model_snapshot": order.get("model_snapshot"),
                    })
                    self.window_trade_count += 1
                    self.position_count += 1

                    entry_px = (
                        filled_cost / size_matched if size_matched > 0 else 0
                    )
                    self._event(
                        f"[CANCEL->PARTIAL] {order['side']} "
                        f"{size_matched:.1f}/{original_size:.1f}sh "
                        f"@ {entry_px:.4f} (${filled_cost:.2f}) — "
                        f"refunding ${unfilled_cost_est:.2f}"
                        + self._model_fill_line(order)
                    )
                    self._log({
                        "type": "partial_fill",
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "order_id": order_id,
                        "side": order["side"],
                        "filled_shares": round(size_matched, 2),
                        "price": order["price"],
                        "cost_usd": round(filled_cost, 2),
                        "refund": round(unfilled_cost_est, 2),
                        "note": f"partial_before_cancel: {reason}",
                        **self._model_log_fields(order),
                    })
                    return False
            except Exception as verify_exc:
                self._event(
                    f"[CANCEL WARNING] Could not verify "
                    f"{order_id[:12]}...: {verify_exc}"
                )
                self._log({
                    "type": "cancel_verify_failed",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "order_id": order_id,
                    "side": order["side"],
                    "cost_est": round(order["cost_est"], 2),
                    "error": str(verify_exc),
                })

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
            **model_snapshot,
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
            resp = self.client.post_order(signed_order, OrderType.GTC, post_only=True)
        except Exception as exc:
            trade_record["status"] = "error"
            trade_record["error"] = str(exc)
            self._log(trade_record)
            self._event(f"[LIMIT ORDER ERROR] {exc}")
            return

        success = resp.get("success", False)
        order_id = resp.get("orderID") or resp.get("id", "")
        status = str(resp.get("status", "unknown")).upper()

        trade_record["order_id"] = order_id
        trade_record["status"] = status
        trade_record["success"] = success
        trade_record["response"] = resp

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
        if status == "MATCHED" and taking:
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
                "model_snapshot": model_snapshot,
            })
            self.last_fill_ts_ms = snapshot.ts_ms
            self.window_trade_count += 1
            self.position_count += 1

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
                        "model_snapshot": order.get("model_snapshot"),
                    })
                    self.last_fill_ts_ms = snapshot.ts_ms
                    self.window_trade_count += 1
                    self.position_count += 1
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
            except Exception as exc:
                if self.debug:
                    self._event(f"[POLL] error checking {order_id[:12]}...: {exc}")
                still_open.append(order)
                continue

            status = str(resp.get("status", "unknown")).upper()
            size_matched = float(resp.get("size_matched", 0) or 0)
            original_size = float(resp.get("original_size", order["shares"]) or order["shares"])
            fill_pct = size_matched / original_size if original_size > 0 else 0

            if status == "MATCHED" or fill_pct >= 0.99:
                actual_cost = size_matched * order["price"]
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
                    "model_snapshot": order.get("model_snapshot"),
                })
                self.last_fill_ts_ms = snapshot.ts_ms
                self.window_trade_count += 1
                self.position_count += 1

                entry_px = actual_cost / size_matched if size_matched > 0 else 0
                self._event(
                    f"[LIMIT FILLED] {order['side']} {size_matched:.1f}sh "
                    f"@ {entry_px:.4f} (${actual_cost:.2f})"
                    + self._model_fill_line(order)
                )
                self._log({
                    "type": "limit_fill",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "order_id": order_id,
                    "side": order["side"],
                    "shares": round(size_matched, 2),
                    "price": order["price"],
                    "cost_usd": round(actual_cost, 2),
                    **self._model_log_fields(order),
                })

            elif status in ("CANCELLED", "EXPIRED"):
                self.bankroll += order["cost_est"]
                self.signal.bankroll = self.bankroll
                self._event(
                    f"[LIMIT {status}] {order['side']} "
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
            elif size_matched > order.get("_last_matched", 0):
                prev_matched = order.get("_last_matched", 0)
                new_fill_shares = size_matched - prev_matched
                new_fill_cost = new_fill_shares * order["price"]

                order["cost_est"] -= new_fill_cost
                order["shares"] -= new_fill_shares
                order["_last_matched"] = size_matched

                self.pending_fills.append({
                    "market_slug": order["market_slug"],
                    "side": order["side"],
                    "cost_usd": new_fill_cost,
                    "shares": new_fill_shares,
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

                self._event(
                    f"[PARTIAL FILL] {order['side']} {new_fill_shares:.1f}sh "
                    f"@ {order['price']:.4f} (${new_fill_cost:.2f}) — "
                    f"{order['shares']:.1f}sh still open"
                    + self._model_fill_line(order)
                )
                self._log({
                    "type": "partial_fill",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "order_id": order_id,
                    "side": order["side"],
                    "filled_shares": round(new_fill_shares, 2),
                    "remaining_shares": round(order["shares"], 2),
                    "price": order["price"],
                    "cost_usd": round(new_fill_cost, 2),
                    **self._model_log_fields(order),
                })
                still_open.append(order)
            else:
                still_open.append(order)

        self.open_orders = still_open

    def _cancel_open_orders(self: "LiveTradeTracker"):
        """Cancel all open limit orders and refund reserved bankroll."""
        if not self.open_orders:
            return
        for order in list(self.open_orders):
            self._cancel_single_order(order, "end_of_window")
        self.open_orders = []

    def cancel_all_orders(self: "LiveTradeTracker"):
        """Cancel all open orders — called on shutdown."""
        if self.open_orders:
            self._cancel_open_orders()
        if self.dry_run:
            return
        try:
            resp = self.client.cancel_all()
            self._event(f"Cancelled all open orders: {resp}")
        except Exception as exc:
            self._event(f"Warning: cancel_all failed: {exc}")

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
