"""Terminal display: live trading dashboard rendering."""

from __future__ import annotations

import asyncio
import math
import sys
import time as _time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

from market_config import MarketConfig

if TYPE_CHECKING:
    from tracker import LiveTradeTracker


def render_display(
    tracker: "LiveTradeTracker",
    price_state: dict,
    flat_state: dict,
    market_title: str,
    window_start: datetime,
    window_end: datetime,
    config: MarketConfig,
):
    lines = ["\033[2J\033[H"]
    mode_str = "DRY RUN" if tracker.dry_run else "LIVE"
    exit_tag = "+EXIT" if tracker.exit_enabled else ""
    lines.append("=" * 62)
    lines.append(f"  [{mode_str}] [MAKER{exit_tag}] {config.display_name} Up/Down: {market_title}")
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
    lines.append(f"  -- {mode_str} Trading (MAKER) " + "-" * 24)

    dec = tracker.last_decision
    status = dec.action if dec.action != "FLAT" else "FLAT"

    # Balance line
    bal_str = f"${tracker.bankroll:,.2f}"
    if tracker.api_balance is not None:
        bal_str += f"  (API: ${tracker.api_balance:,.2f})"
    lines.append(f"  Bankroll: {bal_str}  |  Status: {status}")
    lines.append(f"  Reason:   {dec.reason[:60]}")

    # Dual-side status
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

    # Toxicity indicator
    toxicity = tracker.ctx.get("_toxicity", 0.0)
    if toxicity > 0:
        tox_bar = "#" * int(toxicity * 10)
        tox_label = "LOW" if toxicity < 0.3 else ("MED" if toxicity < 0.6 else "HIGH")
        lines.append(f"  Toxicity:  {toxicity:.2f} [{tox_bar:<10s}] {tox_label}")

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

    # Open limit orders
    if tracker.open_orders:
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
    open_str = f"  |  Open: {len(tracker.open_orders)}"
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
    tracker: "LiveTradeTracker",
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
