from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from backtest_core import Fill, Snapshot, TradeResult
from display import _dedupe_results, _pending_fill_upnl
from signal_diffusion import DiffusionSignal
from tracker import LiveTradeTracker


class _ConstantEdgeModel:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, X):
        return np.full(len(X), self.value, dtype=np.float32)


def _make_result(
    market_slug: str = "btc-updown-5m-test",
    side: str = "DOWN",
    entry_ts_ms: int = 1_700_000_000_000,
    shares: float = 10.0,
    cost_usd: float = 4.2,
    outcome_up: int = 0,
    payout: float = 10.0,
    pnl: float = 5.8,
) -> TradeResult:
    fill = Fill(
        market_slug=market_slug,
        side=side,
        entry_ts_ms=entry_ts_ms,
        time_remaining_s=120.0,
        entry_price=cost_usd / shares,
        fee_per_share=0.0,
        shares=shares,
        cost_usd=cost_usd,
        signal_name="diffusion",
        decision_reason="test",
    )
    return TradeResult(
        fill=fill,
        outcome_up=outcome_up,
        final_btc=73_000.0,
        payout=payout,
        pnl=pnl,
        pnl_pct=(pnl / cost_usd) if cost_usd else 0.0,
    )


def _make_snapshot(ts_ms: int) -> Snapshot:
    return Snapshot(
        ts_ms=ts_ms,
        market_slug="btc-updown-5m-test",
        time_remaining_s=120.0,
        chainlink_price=73_000.0,
        window_start_price=73_000.0,
        best_bid_up=0.49,
        best_ask_up=0.50,
        best_bid_down=0.49,
        best_ask_down=0.50,
        size_bid_up=None,
        size_ask_up=None,
        size_bid_down=None,
        size_ask_down=None,
        ask_levels_up=(),
        ask_levels_down=(),
        bid_levels_up=(),
        bid_levels_down=(),
    )


def test_dedupe_results_collapses_shared_history_duplicates():
    a = _make_result()
    b = _make_result()  # same restored trade on the second tracker
    c = _make_result(market_slug="btc-updown-15m-test", entry_ts_ms=1_700_000_100_000)

    deduped = _dedupe_results([a, b, c])

    assert len(deduped) == 2
    assert round(sum(r.pnl for r in deduped), 2) == 11.60


def test_pending_fill_upnl_marks_position_to_current_bid():
    fill = {
        "side": "UP",
        "shares": 10.0,
        "cost_usd": 6.0,
    }

    upnl = _pending_fill_upnl(fill, {"up_best_bid": "0.50"})

    assert round(upnl, 2) == -1.00


def test_latency_arb_uses_latest_binance_tick_time_for_lookback():
    sig = DiffusionSignal(
        bankroll=100.0,
        latency_arb_mode=True,
        arb_delta_usd=15.0,
        arb_window_s=2.0,
        arb_book_stale_ms=0.0,
    )
    ctx = {
        # Latest Binance update is at 2500ms. The evaluation itself is
        # later (4300ms) because the wake came from another feed.
        "_binance_ring": [
            (1000, 73_000.0),
            (2500, 73_020.0),
        ],
        "_book_age_ms": 0.0,
    }

    up_dec, down_dec = sig.decide_latency_arb(_make_snapshot(4300), ctx)

    assert up_dec.action == "BUY_UP"
    assert down_dec.action == "FLAT"
    assert "LATENCY_ARB" in up_dec.reason


def test_experimental_taker_returns_one_sided_ev_decision():
    bundle = {
        "model": _ConstantEdgeModel(0.25),
        "feature_cols": ["trade_eff_cost", "trade_raw_edge_gbm"],
        "market": "btc_5m",
        "target": "edge_share",
        "taus": [120],
        "tau_tolerance_s": 30.0,
    }
    sig = DiffusionSignal(
        bankroll=100.0,
        window_duration=300.0,
        experimental_filtration_bundle=bundle,
        experimental_filtration_threshold=0.20,
        experimental_filtration_mode="taker",
        maker_warmup_s=0.0,
    )
    ctx = {
        "_binance_mid": 73_050.0,
        "price_history": [73_000.0, 73_010.0, 73_020.0, 73_035.0],
        "ts_history": [1_000, 2_000, 3_000, 4_000],
    }
    snap = _make_snapshot(5_000)

    up_dec, down_dec = sig.decide_both_sides(snap, ctx)

    assert up_dec.action == "BUY_UP"
    assert down_dec.action == "FLAT"
    assert up_dec.edge > 0.20
    assert "EXP_TAKER" in up_dec.reason
    assert round(float(ctx["_filtration_confidence"]), 2) == 0.25


def test_tracker_routes_experimental_taker_through_taker_dry_run():
    bundle = {
        "model": _ConstantEdgeModel(0.25),
        "feature_cols": ["trade_eff_cost", "trade_raw_edge_gbm"],
        "market": "btc_5m",
        "target": "edge_share",
        "taus": [120],
        "tau_tolerance_s": 30.0,
    }
    sig = DiffusionSignal(
        bankroll=100.0,
        window_duration=300.0,
        experimental_filtration_bundle=bundle,
        experimental_filtration_threshold=0.20,
        experimental_filtration_mode="taker",
        maker_warmup_s=0.0,
        min_order_shares=5.0,
    )
    tracker = LiveTradeTracker(
        client=None,
        signal=sig,
        initial_bankroll=100.0,
        dry_run=True,
    )
    tracker.ctx.update({
        "_binance_mid": 73_050.0,
        "price_history": [73_000.0, 73_010.0, 73_020.0, 73_035.0],
        "ts_history": [1_000, 2_000, 3_000, 4_000],
    })
    tracker.last_price_update_ts = time.time()
    snap = _make_snapshot(5_000)

    dec = tracker.evaluate(snap, "up-token", "down-token")

    assert dec.action == "BUY_UP"
    assert len(tracker.pending_fills) == 1
    assert tracker.pending_fills[0]["side"] == "UP"
    assert tracker.open_orders == []


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            fails += 1
            print(f"FAIL  {fn.__name__}\n      {exc}")
        except Exception:
            fails += 1
            print(f"ERROR {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - fails}/{len(fns)} tests passed")
    sys.exit(1 if fails else 0)
