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


def _result_key(result) -> tuple:
    fill = result.fill
    return (
        fill.market_slug,
        fill.side,
        int(fill.entry_ts_ms),
        round(float(fill.shares), 4),
        round(float(fill.cost_usd), 4),
        int(result.outcome_up),
        round(float(result.payout), 4),
        round(float(result.pnl), 4),
    )


def _dedupe_results(results):
    unique = []
    seen = set()
    for result in results:
        key = _result_key(result)
        if key in seen:
            continue
        seen.add(key)
        unique.append(result)
    return unique


def _fill_key(fill: dict) -> tuple:
    return (
        str(fill.get("order_id", "")),
        str(fill.get("market_slug", "")),
        str(fill.get("side", "")),
        round(float(fill.get("shares", 0.0) or 0.0), 4),
        round(float(fill.get("cost_usd", 0.0) or 0.0), 4),
        round(float(fill.get("fill_ts_unix", fill.get("entry_ts", 0.0)) or 0.0), 3),
    )


def _marked_fill_value(fill: dict, flat_state: dict) -> float:
    side = str(fill.get("side", "")).upper()
    bid_key = "up_best_bid" if side == "UP" else "down_best_bid"
    bid_val = flat_state.get(bid_key)
    try:
        bid = float(bid_val) if bid_val not in (None, "") else None
    except (TypeError, ValueError):
        bid = None
    shares = float(fill.get("shares", 0.0) or 0.0)
    cost = float(fill.get("cost_usd", 0.0) or 0.0)
    if bid is None or bid <= 0 or shares <= 0:
        return cost
    return bid * shares


def _pending_fill_upnl(fill: dict, flat_state: dict) -> float:
    return _marked_fill_value(fill, flat_state) - float(fill.get("cost_usd", 0.0) or 0.0)


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

    # Per-window start price + delta (show BOTH Chainlink and Binance)
    if window_start_price is not None:
        delta_str = ""
        if price is not None:
            delta = price - window_start_price
            delta_str = f"  (CL delta: ${delta:+,.2f}"
            # Show Binance delta too — this is what the model actually uses
            bn_mid = tracker.ctx.get("_binance_mid")
            if bn_mid and bn_mid > 0:
                bn_delta = bn_mid - window_start_price
                delta_str += f", BN delta: ${bn_delta:+,.2f}"
            delta_str += ")"
        lines.append(f"  Start: ${window_start_price:>12,.2f}{delta_str}")

    # Book BBO (compact single line) — tag with STALE indicator if the
    # book WS hasn't updated in >2s. The bot's max_book_age_ms gate may
    # be set higher (e.g. 5s for calm-market tolerance) but for the
    # operator's eye, anything >2s should be visually flagged.
    up_bid = _fp(flat_state, "up_best_bid")
    up_ask = _fp(flat_state, "up_best_ask")
    dn_bid = _fp(flat_state, "down_best_bid")
    dn_ask = _fp(flat_state, "down_best_ask")
    book_age_ms = tracker.ctx.get("_book_age_ms")
    book_stale_tag = ""
    if book_age_ms is not None and book_age_ms > 2000:
        book_stale_tag = f"  [BOOK STALE {book_age_ms/1000:.1f}s]"
    lines.append(f"  Book:  Up {up_bid}/{up_ask}  |  Down {dn_bid}/{dn_ask}{book_stale_tag}")

    # Signal status
    dec = tracker.last_decision
    # 2026-04-09: render p_up as "---" when the model couldn't compute
    # this tick (vol collapse, warmup, or window boundary). Previously
    # the dashboard kept showing the LAST successful p_model value
    # indefinitely during stale-book episodes, making it appear frozen.
    p_display_fresh = tracker.ctx.get("_p_display_fresh", False)
    p_model = tracker.ctx.get("_p_display") or tracker.ctx.get("_p_model_raw")
    if not p_display_fresh:
        p_model = None  # render as "---"
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
        # 2026-04-09: bumped 30 → 60 chars after WARM-UP message was being
        # truncated mid-word ("...into windo" missing "w"). Most reasons are
        # under 50 chars so 60 leaves comfortable headroom.
        reason = dec.reason[:60] if dec.reason else "no_edge"
        sig_parts.append(f"FLAT ({reason})")

    if p_model is not None:
        sig_parts.append(f"p_up={p_model:.4f}")
    else:
        # Model couldn't compute this tick (vol collapse, warmup, or
        # both feeds briefly stuck). Show "---" so the operator knows
        # the value isn't being updated, instead of seeing a frozen
        # last-known value.
        sig_parts.append("p_up=---")
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
    # so the operator can see why bankroll != realized PnL when there
    # are open orders / pending fills. Pending fills are marked to the
    # current bid so underwater positions show up in the headline stats
    # instead of looking like "cash vanished but PnL went up".
    open_orders = _safe_list(tracker.open_orders)
    pending_fills = _safe_list(tracker.pending_fills)
    reserved_cash = sum(o.get("cost_est", 0) for o in open_orders)
    marked_positions = sum(_marked_fill_value(f, flat_state) for f in pending_fills)
    open_upnl = sum(_pending_fill_upnl(f, flat_state) for f in pending_fills)
    locked = reserved_cash + marked_positions
    avail = tracker.bankroll
    total_equity = avail + locked
    if locked > 0.01:
        lines.append(
            f"  Equity: ${total_equity:,.2f}  "
            f"(cash ${avail:,.2f} + orders ${reserved_cash:,.2f} "
            f"+ pos ${marked_positions:,.2f})  |  "
            f"uPnL: ${open_upnl:+,.2f}  |  "
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
        for fill in pending_fills:
            entry_px = fill["cost_usd"] / fill["shares"] if fill["shares"] > 0 else 0
            upnl = _pending_fill_upnl(fill, flat_state)
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

    # Use first tracker's price update timestamp for staleness check.
    # Guard on `last_price_update_ts > 0` so a warmup state (timestamp 0)
    # doesn't render as "STALE 1700000000s".
    first_tracker = sections[0]["tracker"] if sections else None
    if (price is not None and first_tracker is not None
            and first_tracker.last_price_update_ts > 0):
        age = _time.time() - first_tracker.last_price_update_ts
        stale_tag = f"  STALE {age:.0f}s" if age > first_tracker.stale_price_timeout_s else ""
        lines.append(f"  {price_label}:  ${price:>12,.2f}{stale_tag}")
    elif price is not None:
        lines.append(f"  {price_label}:  ${price:>12,.2f}")
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
    open_upnl_total = 0.0
    seen_pending_keys: set[tuple] = set()
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
        for fill in _safe_list(t.pending_fills):
            key = _fill_key(fill)
            if key in seen_pending_keys:
                continue
            seen_pending_keys.add(key)
            open_upnl_total += _pending_fill_upnl(fill, sec["flat_state"])

    unique_results = _dedupe_results(all_results_combined)
    closed_total = len(unique_results)
    realized_pnl = sum(r.pnl for r in unique_results)
    closed_wins = sum(1 for r in unique_results if r.pnl > 0)
    net_pnl = realized_pnl + open_upnl_total

    lines.append("  -- Stats " + "-" * 50)
    if closed_total > 0:
        lines.append(
            f"  Realized: ${realized_pnl:+,.2f}  |  "
            f"Open uPnL: ${open_upnl_total:+,.2f}  |  "
            f"Net: ${net_pnl:+,.2f}"
        )
        closed_wr = closed_wins / closed_total
        lines.append(
            f"  Trades: {closed_wins}/{closed_total} ({closed_wr:.0%})  |  "
            f"Windows: {total_windows_traded}/{total_windows_seen}  |  "
            f"DD: ${max_dd:.0f} ({max_dd_pct:.1%})"
        )
    else:
        lines.append(
            f"  Realized: ${realized_pnl:+,.2f}  |  "
            f"Open uPnL: ${open_upnl_total:+,.2f}  |  "
            f"Net: ${net_pnl:+,.2f}  |  "
            f"DD: ${max_dd:.0f} ({max_dd_pct:.1%})"
        )
        lines.append(
            f"  Windows: {total_windows_traded}/{total_windows_seen} traded  |  "
            f"Fees: ~${total_fees:.2f}"
        )

    # Last result
    if unique_results:
        r = unique_results[-1]
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
