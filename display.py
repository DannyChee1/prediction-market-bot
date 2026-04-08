"""Terminal display: live trading dashboard rendering (multi-timeframe)."""

from __future__ import annotations

import math
import sys
import time as _time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

from market_config import MarketConfig

if TYPE_CHECKING:
    from tracker import LiveTradeTracker


def _safe_list(obj):
    """Thread-safe snapshot of a list/deque."""
    try:
        return list(obj)
    except (RuntimeError, ValueError):
        return []


def _safe_dict(obj):
    try:
        return dict(obj)
    except RuntimeError:
        return {}


def _pct(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    return float(np.percentile(np.asarray(vals, dtype=float), q))


def _fp(flat_state: dict, key: str) -> str:
    v = flat_state.get(key)
    if v is None:
        return "---"
    try:
        return f"{float(v):.4f}"
    except (ValueError, TypeError):
        return str(v)[:7]


def _render_section(
    lines: list[str],
    tracker: "LiveTradeTracker",
    config: MarketConfig,
    flat_state: dict,
    window_start: datetime | None,
    window_end: datetime | None,
    market_title: str,
    status: str,
    price: float | None = None,
    window_start_price: float | None = None,
):
    """Render a single timeframe section (compact)."""
    label = config.display_name
    now = datetime.now(timezone.utc)

    # Section header
    lines.append(f"  -- {label}: {market_title[:42]} " + "-" * max(1, 58 - len(label) - len(market_title[:42])))

    # Between-windows state
    if status != "trading":
        status_map = {
            "resolving": "Resolving window...",
            "searching": "Searching for next window...",
            "waiting": "Waiting for window start...",
        }
        lines.append(f"  Status: {status_map.get(status, status)}")
        # Lifetime stats survive restarts (P12.4); fall back to session
        # if lifetime tracking is empty (pre-P12.4 state file).
        lifetime_pnl = getattr(tracker, "lifetime_pnl", 0.0)
        lifetime_trades = getattr(tracker, "lifetime_trades", 0)
        lifetime_wins = getattr(tracker, "lifetime_wins", 0)
        if lifetime_trades > 0:
            wr = lifetime_wins / lifetime_trades
            lines.append(
                f"  Bankroll: ${tracker.bankroll:,.2f}  |  "
                f"Lifetime: ${lifetime_pnl:+,.2f} "
                f"({lifetime_trades} trades, {wr:.0%})"
            )
        else:
            all_results = _safe_list(tracker.all_results)
            total = len(all_results)
            total_pnl = sum(r.pnl for r in all_results)
            wins = sum(1 for r in all_results if r.pnl > 0)
            win_str = f"{wins}/{total} ({wins / total:.0%})" if total > 0 else "---"
            lines.append(
                f"  Bankroll: ${tracker.bankroll:,.2f}  |  "
                f"Win: {win_str}  |  PnL: ${total_pnl:+,.2f}"
            )
        lines.append("")
        return

    # Active trading window
    if window_end is not None:
        remaining = (window_end - now).total_seconds()
        win_dur = tracker.window_duration_s
        if remaining > 0:
            m, s = int(remaining // 60), int(remaining % 60)
            bar_len = 24
            filled = max(0, min(bar_len, int((remaining / win_dur) * bar_len)))
            time_str = (
                f"{m:02d}:{s:02d}  "
                f"[{'#' * filled}{'-' * (bar_len - filled)}]"
            )
        else:
            time_str = "EXPIRED  (resolving...)"

        window_str = ""
        if window_start is not None:
            window_str = (
                f"  ({window_start.strftime('%H:%M')}"
                f"-{window_end.strftime('%H:%M')} UTC)"
            )
        lines.append(f"  Time: {time_str}{window_str}")

    # Per-window start price + delta
    if window_start_price is not None:
        delta_str = ""
        if price is not None:
            delta = price - window_start_price
            delta_str = f"  (delta: ${delta:+,.2f})"
        lines.append(f"  Start: ${window_start_price:>12,.2f}{delta_str}")

    # Book BBO (compact single line)
    up_bid = _fp(flat_state, "up_best_bid")
    up_ask = _fp(flat_state, "up_best_ask")
    dn_bid = _fp(flat_state, "down_best_bid")
    dn_ask = _fp(flat_state, "down_best_ask")
    lines.append(f"  Book:  Up {up_bid}/{up_ask}  |  Down {dn_bid}/{dn_ask}")

    # Signal status
    dec = tracker.last_decision
    p_model = tracker.ctx.get("_p_display") or tracker.ctx.get("_p_model_raw")
    edge_up = tracker.ctx.get("_edge_up")
    edge_dn = tracker.ctx.get("_edge_down")

    up_d = tracker.last_up_decision
    dn_d = tracker.last_down_decision
    up_tag = up_d.action if up_d.action != "FLAT" else "FLAT"
    dn_tag = dn_d.action if dn_d.action != "FLAT" else "FLAT"

    sig_parts = []
    if up_tag != "FLAT" or dn_tag != "FLAT":
        active = up_tag if up_tag != "FLAT" else dn_tag
        edge = edge_up if up_tag != "FLAT" else edge_dn
        if edge is not None:
            sig_parts.append(f"{active} (edge={edge:.4f})")
        else:
            sig_parts.append(active)
    else:
        reason = dec.reason[:30] if dec.reason else "no_edge"
        sig_parts.append(f"FLAT ({reason})")

    if p_model is not None:
        sig_parts.append(f"p_up={p_model:.4f}")
    lines.append(f"  Signal: {'  |  '.join(sig_parts)}")

    lat_samples = _safe_list(getattr(tracker, "latency_samples", []))
    total_vals = [
        float(s.get("decision_total_ms", 0.0))
        for s in lat_samples
        if s.get("decision_total_ms") is not None
    ]
    eval_vals = [
        float(s.get("signal_eval_ms", 0.0))
        for s in lat_samples
        if s.get("signal_eval_ms") is not None
    ]
    trigger_source = tracker.ctx.get("_signal_trigger_source") or "---"
    trigger_age = tracker.ctx.get("_signal_trigger_age_ms")
    eval_ms = tracker.ctx.get("_signal_eval_ms")
    chainlink_age = tracker.ctx.get("_chainlink_age_ms")
    binance_age = tracker.ctx.get("_binance_age_ms")
    book_age = tracker.ctx.get("_book_age_ms")
    lat_parts = [
        f"src={trigger_source}",
        f"trig={trigger_age:.1f}ms" if trigger_age is not None else "trig=---",
        f"eval={eval_ms:.1f}ms" if eval_ms is not None else "eval=---",
        f"CL={chainlink_age:.0f}ms" if chainlink_age is not None else "CL=---",
        f"BN={binance_age:.0f}ms" if binance_age is not None else "BN=---",
        f"Book={book_age:.0f}ms" if book_age is not None else "Book=---",
    ]
    if total_vals:
        p50_total = _pct(total_vals, 50)
        p95_total = _pct(total_vals, 95)
        lat_parts.append(
            f"e2e p50/p95={p50_total:.1f}/{p95_total:.1f}ms"
        )
    if eval_vals:
        p95_eval = _pct(eval_vals, 95)
        lat_parts.append(f"eval p95={p95_eval:.1f}ms")
    lines.append(f"  Lat: {'  |  '.join(lat_parts)}")

    # Bankroll + trades compact line. Show available cash AND the
    # cost locked up in resting orders + filled-but-unresolved positions
    # so the operator can see why bankroll != cumulative PnL when there
    # are open orders / pending fills.
    open_orders = _safe_list(tracker.open_orders)
    pending_fills = _safe_list(tracker.pending_fills)
    locked = sum(o.get("cost_est", 0) for o in open_orders) + \
             sum(f.get("cost_usd", 0) for f in pending_fills)
    avail = tracker.bankroll
    total_equity = avail + locked
    if locked > 0.01:
        lines.append(
            f"  Equity: ${total_equity:,.2f}  "
            f"(cash ${avail:,.2f} + locked ${locked:,.2f})  |  "
            f"Trades: {tracker.window_trade_count}/win  |  "
            f"Open: {len(open_orders)}  |  Pos: {len(pending_fills)}"
        )
    else:
        lines.append(
            f"  Bankroll: ${avail:,.2f}  |  "
            f"Trades: {tracker.window_trade_count}/win  |  "
            f"Open: {len(open_orders)}  |  Pos: {len(pending_fills)}"
        )

    # Toxicity / VPIN indicators (compact)
    toxicity = tracker.ctx.get("_toxicity", 0.0)
    vpin = tracker.ctx.get("_vpin", 0.0)
    if toxicity > 0 or vpin > 0:
        parts = []
        if toxicity > 0:
            parts.append(f"Tox={toxicity:.2f}")
        if vpin > 0:
            parts.append(f"VPIN={vpin:.2f}")
        lines.append(f"  {' | '.join(parts)}")

    # Signal error
    sig_err = tracker.ctx.get("_signal_error")
    if sig_err:
        lines.append(f"  !! SIGNAL ERROR: {sig_err[:70]} !!")

    # Circuit breaker
    if tracker.circuit_breaker_tripped:
        lines.append(f"  *** {tracker.circuit_breaker_reason[:70]} ***")

    # Open orders (compact)
    if open_orders:
        for o in open_orders:
            age_s = int(_time.time() - o.get("placed_ts_unix", _time.time()))
            lines.append(
                f"    ORDER {o['side']:>4s} {o['shares']:.1f}sh "
                f"@ {o['price']:.4f} (${o['cost_est']:.2f}) age={age_s}s"
            )

    # Positions (compact)
    if pending_fills:
        bid_map = {
            "UP": flat_state.get("up_best_bid"),
            "DOWN": flat_state.get("down_best_bid"),
        }
        for fill in pending_fills:
            entry_px = fill["cost_usd"] / fill["shares"] if fill["shares"] > 0 else 0
            bid_val = bid_map.get(fill["side"])
            bid_f = float(bid_val) if bid_val else 0.0
            upnl = (bid_f - entry_px) * fill["shares"]
            hold_s = int(_time.time() - fill.get("fill_ts_unix", _time.time()))
            lines.append(
                f"    POS  {fill['side']:>4s} {fill['shares']:.1f}sh "
                f"@ {entry_px:.4f} uPnL=${upnl:+.2f} hold={hold_s}s"
            )

    # Resting sell orders (compact)
    open_sell_orders = _safe_list(tracker.open_sell_orders)
    if open_sell_orders:
        for o in open_sell_orders:
            age_s = int(_time.time() - o.get("placed_ts_unix", _time.time()))
            lines.append(
                f"    SELL {o['side']:>4s} {o['shares']:.1f}sh "
                f"@ {o['price']:.4f} age={age_s}s"
            )

    lines.append("")


def render_display(
    price_state: dict,
    sections: list[dict],
    base_config: MarketConfig,
    dry_run: bool = False,
    exit_enabled: bool = False,
):
    """Render combined multi-timeframe display.

    sections: list of dicts with keys:
        tracker, config, flat_state, window_start, window_end,
        market_title, status
    """
    lines = ["\033[2J\033[H"]

    # ── Shared header ──
    mode_str = "DRY RUN" if dry_run else "LIVE"
    exit_tag = "+EXIT" if exit_enabled else ""
    asset = base_config.chainlink_symbol.split("/")[0].upper()
    lines.append("=" * 62)
    lines.append(f"  [{mode_str}] [MAKER{exit_tag}] {asset} Up/Down")
    lines.append("=" * 62)
    lines.append("")

    # ── Shared price info ──
    price = price_state.get("price")
    price_label = f"Chainlink {base_config.chainlink_symbol.upper()}"

    # Use first tracker's price update timestamp for staleness check
    first_tracker = sections[0]["tracker"] if sections else None
    if price is not None and first_tracker is not None:
        age = _time.time() - first_tracker.last_price_update_ts
        stale_tag = f"  STALE {age:.0f}s" if age > first_tracker.stale_price_timeout_s else ""
        lines.append(f"  {price_label}:  ${price:>12,.2f}{stale_tag}")
    else:
        lines.append(f"  {price_label}:     waiting...")

    # Price history vol from first tracker
    if first_tracker is not None:
        try:
            hist = list(first_tracker.ctx.get("price_history", []))
        except (RuntimeError, ValueError):
            hist = []
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
        lines.append(f"  History: {hist_len}s{vol_str}")
    lines.append("")

    # ── Per-timeframe sections ──
    for sec in sections:
        _render_section(
            lines,
            sec["tracker"],
            sec["config"],
            sec["flat_state"],
            sec.get("window_start"),
            sec.get("window_end"),
            sec.get("market_title", ""),
            sec.get("status", "searching"),
            price=price,
            window_start_price=sec.get("window_start_price"),
        )

    # ── Combined session stats ──
    all_results_combined = []
    total_fees = 0.0
    total_windows_traded = 0
    total_windows_seen = 0
    max_dd = 0.0
    max_dd_pct = 0.0
    event_logs = []

    # Aggregate lifetime + session totals across all timeframe trackers
    lifetime_pnl_total = 0.0
    lifetime_trades_total = 0
    lifetime_wins_total = 0
    for sec in sections:
        t = sec["tracker"]
        all_results_combined.extend(_safe_list(t.all_results))
        total_fees += t.total_fees
        total_windows_traded += t.windows_traded
        total_windows_seen += t.windows_seen
        if t.max_drawdown > max_dd:
            max_dd = t.max_drawdown
            max_dd_pct = t.max_dd_pct
        event_logs.extend(_safe_list(t.event_log))
        # Lifetime scalars (P12.4) — survive restarts via state file
        lifetime_pnl_total += getattr(t, "lifetime_pnl", 0.0)
        lifetime_trades_total += getattr(t, "lifetime_trades", 0)
        lifetime_wins_total += getattr(t, "lifetime_wins", 0)

    # Session = since last restart (in-memory all_results)
    session_total = len(all_results_combined)
    session_pnl = sum(r.pnl for r in all_results_combined)
    session_wins = sum(1 for r in all_results_combined if r.pnl > 0)

    lines.append("  -- Stats " + "-" * 50)
    if lifetime_trades_total > 0:
        # Show lifetime AND session when both are meaningful
        lifetime_wr = (lifetime_wins_total / lifetime_trades_total
                       if lifetime_trades_total > 0 else 0.0)
        lines.append(
            f"  Lifetime: ${lifetime_pnl_total:+,.2f}  "
            f"({lifetime_trades_total} trades, {lifetime_wr:.0%} win)  |  "
            f"DD: ${max_dd:.0f} ({max_dd_pct:.1%})"
        )
        if session_total > 0:
            session_wr = session_wins / session_total
            lines.append(
                f"  Session:  ${session_pnl:+,.2f}  "
                f"({session_total} trades, {session_wr:.0%} win)  |  "
                f"Windows: {total_windows_traded}/{total_windows_seen}"
            )
    else:
        # Pre-P12.4 path or first session: only show session
        win_str = (f"{session_wins}/{session_total} ({session_wins / session_total:.0%})"
                   if session_total > 0 else "---")
        lines.append(
            f"  PnL: ${session_pnl:+,.2f}  |  Win: {win_str}  |  "
            f"DD: ${max_dd:.0f} ({max_dd_pct:.1%})"
        )
        lines.append(
            f"  Windows: {total_windows_traded}/{total_windows_seen} traded  |  "
            f"Fees: ~${total_fees:.2f}"
        )

    # Last result
    if all_results_combined:
        r = all_results_combined[-1]
        tag = "WON" if r.pnl > 0 else "LOST"
        lines.append(
            f"  Last: {r.fill.side} ${r.fill.cost_usd:.2f} -> "
            f"{tag} ${r.pnl:+.2f}"
        )

    # Recent events (combined, deduplicated by recency)
    if event_logs:
        # Show most recent 4 events across all trackers
        lines.append("  -- Recent Events " + "-" * 40)
        for evt in event_logs[-4:]:
            lines.append(f"  {evt}")

    lines.append("")
    lines.append("  Ctrl+C to exit (cancels open orders)")

    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()
