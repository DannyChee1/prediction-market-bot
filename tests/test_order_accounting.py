"""
Regression tests for the execution-layer accounting fixes (Batch 2).

These cover the six historical bugs where the bot:
  1. Refunded `cost_est` instead of `cost_est - already_filled` on cancel
     after partial fills, producing phantom bankroll credits.
  2. Treated WebSocket partial-fill events as full fills, silently
     dropping the unfilled residual as a phantom resting order.
  3. Incremented `window_trade_count` / `position_count` per partial
     instead of per order, breaking the max_trades_per_window guarantee.
  4. Wiped verify-pending orders in `_cancel_open_orders`.

The core invariant is: for any order taken from placement to terminal
state (MATCHED or CANCELLED/EXPIRED), the net bankroll delta must
equal the total actually-paid cost across all fills. Partials do not
refund; only terminal events do.

If any test in this file ever fails, do NOT silence it — the bug is back.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from orders import OrderMixin  # noqa: E402


# ── Minimal fake tracker ─────────────────────────────────────────────────────


class _FakeSignal:
    def __init__(self, bankroll: float):
        self.bankroll = bankroll


class _FakeTracker(OrderMixin):
    """Just enough tracker state to exercise OrderMixin accounting."""

    def __init__(self, bankroll: float = 100.0):
        self.bankroll = bankroll
        self.signal = _FakeSignal(bankroll)
        self.pending_fills: list[dict] = []
        self.open_orders: list[dict] = []
        self.total_fees = 0.0
        self.window_trade_count = 0
        self.position_count = 0
        self.last_fill_ts_ms = 0
        self.window_first_side: str | None = None
        self.same_direction_stacking_only = False
        self.event_log: list[str] = []
        self.ctx: dict = {}
        self.dry_run = True
        self.debug = False

    def _log(self, _record):  # swallow
        pass

    def _record_window_fill_side(self, side: str):
        if self.window_first_side is None:
            self.window_first_side = side


def _make_order(
    cost_est: float,
    shares: float,
    price: float,
    order_id: str = "test-order-1",
    side: str = "UP",
) -> dict:
    return {
        "order_id": order_id,
        "market_slug": "test-slug",
        "side": side,
        "price": price,
        "shares": shares,
        "cost_est": cost_est,
        "placed_ts": 0,
        "time_remaining_s": 100.0,
        "chainlink_price": 50000.0,
        "window_start_price": 50000.0,
    }


# ── Tests ────────────────────────────────────────────────────────────────────


def test_single_full_fill_bankroll_delta_equals_cost():
    """Placement → single MATCHED event → bankroll delta should be -cost."""
    t = _FakeTracker(bankroll=100.0)
    order = _make_order(cost_est=1.50, shares=5.0, price=0.30)
    t.bankroll -= order["cost_est"]  # simulate placement debit
    t.open_orders.append(order)
    initial = 100.0

    # Simulate a full fill: 5 shares at $0.30.
    new_fill_shares = 5.0
    new_fill_cost = 5.0 * 0.30  # fee=0
    t._record_partial_fill(order, new_fill_shares, new_fill_cost, 0.0, 5.0)
    refund = t._terminal_refund(order)
    t.bankroll += refund

    actual_spent = initial - t.bankroll
    assert abs(actual_spent - 1.50) < 1e-9, (
        f"expected to spend $1.50, actually spent ${actual_spent:.4f}"
    )
    assert len(t.pending_fills) == 1
    assert t.window_trade_count == 1
    assert t.position_count == 1


def test_two_partials_then_match_no_double_refund():
    """The bug: partial→partial→MATCHED would over-refund.

    Historic trace: on a 5-share order at $0.30 (cost_est=1.50), the
    poll loop saw [partial 2 sh, partial 2 sh, MATCHED 5 sh]. The old
    `drift = cost_est - new_fill_cost` at MATCHED refunded 1.50 - 0.30
    = 1.20, leaving the bot thinking it had spent only $0.30 when it
    had actually spent $1.50. Phantom bankroll credit: $0.90.
    """
    t = _FakeTracker(bankroll=100.0)
    order = _make_order(cost_est=1.50, shares=5.0, price=0.30)
    t.bankroll -= order["cost_est"]
    t.open_orders.append(order)
    initial = 100.0

    # Partial 1: 2 shares at $0.30.
    t._record_partial_fill(order, 2.0, 2.0 * 0.30, 0.0, 2.0)
    # Partial 2: 2 more shares (delta).
    t._record_partial_fill(order, 2.0, 2.0 * 0.30, 0.0, 4.0)
    # Terminal MATCHED for the last share (delta=1).
    t._record_partial_fill(order, 1.0, 1.0 * 0.30, 0.0, 5.0)
    refund = t._terminal_refund(order)
    t.bankroll += refund

    actual_spent = initial - t.bankroll
    assert abs(actual_spent - 1.50) < 1e-9, (
        f"expected to spend $1.50, actually spent ${actual_spent:.4f}"
    )
    # All three partials were recorded as pending_fills, but only ONE
    # per-order counter increment.
    assert len(t.pending_fills) == 3
    assert t.window_trade_count == 1, (
        f"expected 1 window_trade_count after 3 partials of same order, "
        f"got {t.window_trade_count}"
    )
    assert t.position_count == 1


def test_partial_then_cancel_refunds_unfilled_only():
    """Partial fill, then the remainder is cancelled.

    5-share order at $0.30 (cost_est=1.50). First 3 shares fill as a
    partial ($0.90). Then we cancel and the remaining 2 shares are
    gone. Actual spend: $0.90. Refund on cancel: $0.60.
    """
    t = _FakeTracker(bankroll=100.0)
    order = _make_order(cost_est=1.50, shares=5.0, price=0.30)
    t.bankroll -= order["cost_est"]
    t.open_orders.append(order)
    initial = 100.0

    t._record_partial_fill(order, 3.0, 3.0 * 0.30, 0.0, 3.0)
    # Cancel the remaining 2 shares: terminal refund only.
    refund = t._terminal_refund(order)
    t.bankroll += refund

    actual_spent = initial - t.bankroll
    assert abs(actual_spent - 0.90) < 1e-9, (
        f"expected to spend $0.90 (3 sh of 5 filled), actually spent "
        f"${actual_spent:.4f}"
    )
    assert abs(refund - 0.60) < 1e-9
    assert len(t.pending_fills) == 1
    assert t.window_trade_count == 1


def test_cancel_with_no_fills_refunds_everything():
    t = _FakeTracker(bankroll=100.0)
    order = _make_order(cost_est=1.50, shares=5.0, price=0.30)
    t.bankroll -= order["cost_est"]
    t.open_orders.append(order)

    refund = t._terminal_refund(order)
    t.bankroll += refund

    assert abs(t.bankroll - 100.0) < 1e-9
    assert abs(refund - 1.50) < 1e-9
    assert t.window_trade_count == 0
    assert t.position_count == 0


def test_cancel_open_orders_preserves_verify_pending():
    """_cancel_open_orders used to do `self.open_orders = []` which
    wiped out verify-pending orders that the cancel path had
    deliberately kept for reconciliation on the next poll."""
    t = _FakeTracker(bankroll=100.0)
    normal = _make_order(cost_est=1.50, shares=5.0, price=0.30,
                         order_id="normal-1")
    pending = _make_order(cost_est=1.00, shares=10.0, price=0.10,
                          order_id="pending-1")
    pending["_verify_pending"] = True
    pending["_verify_pending_since"] = 0.0

    t.open_orders = [normal, pending]
    # Stub: simulate _cancel_single_order removing `normal` (successful
    # cancel) and leaving `pending` in place (verify-failed).
    def _stub_cancel(order, _reason):  # noqa: ARG001
        if order is normal:
            t.open_orders = [o for o in t.open_orders if o is not normal]
        return True
    t._cancel_single_order = _stub_cancel  # type: ignore[method-assign]

    t._cancel_open_orders()

    remaining_ids = [o["order_id"] for o in t.open_orders]
    assert remaining_ids == ["pending-1"], (
        f"verify-pending order should survive bulk cancel, but "
        f"open_orders = {remaining_ids}"
    )


def test_counted_flag_prevents_duplicate_increments():
    """If the same order's _record_partial_fill is called multiple times,
    window_trade_count should only go up by 1 total."""
    t = _FakeTracker(bankroll=100.0)
    order = _make_order(cost_est=1.50, shares=5.0, price=0.30)
    t.open_orders.append(order)

    for _ in range(5):
        t._record_partial_fill(order, 1.0, 0.30, 0.0, 0.0)

    assert t.window_trade_count == 1
    assert t.position_count == 1


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL  {fn.__name__}\n      {e}")
        except Exception:
            fails += 1
            print(f"ERROR {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - fails}/{len(fns)} tests passed")
    sys.exit(1 if fails else 0)
