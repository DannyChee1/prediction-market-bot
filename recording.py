"""
Integrated recorder for live_trader.py.

Samples 1 row/second from the Rust BookFeed (read-lock, ~1µs) and flushes
to parquet every 60s.  Schema matches recorder.py output consumed by
backtest.py:Snapshot.from_row().
"""

from __future__ import annotations

import asyncio
import time as _time
from pathlib import Path

import pandas as pd

TOP_N = 5
DATA_DIR = Path("data")


def build_row(
    book_feed,
    up_token: str,
    down_token: str,
    meta: dict,
    chainlink_price: float | None,
    window_start_price: float | None,
    binance_mid: float | None = None,
) -> dict:
    """Sample current book state into a flat dict for one parquet row.

    ``meta`` must contain: market_slug, condition_id, token_id_up,
    token_id_down, window_start_ms, window_end_ms.
    """
    now_ms = int(_time.time() * 1000)
    remaining_s = max(0, (meta["window_end_ms"] - now_ms) / 1000)

    row: dict = {
        "ts_ms": now_ms,
        "market_slug": meta["market_slug"],
        "condition_id": meta["condition_id"],
        "token_id_up": meta["token_id_up"],
        "token_id_down": meta["token_id_down"],
        "window_start_ms": meta["window_start_ms"],
        "window_end_ms": meta["window_end_ms"],
        "time_remaining_s": round(remaining_s, 3),
        "chainlink_price": chainlink_price,
        "window_start_price": window_start_price,
        "binance_mid": binance_mid,
    }

    for label, token in [("up", up_token), ("down", down_token)]:
        snap = book_feed.snapshot(token)
        bb = snap.best_bid
        ba = snap.best_ask

        row[f"best_bid_{label}"] = bb
        row[f"best_ask_{label}"] = ba

        # Sizes at best
        row[f"size_bid_{label}"] = snap.bids[0][1] if snap.bids else None
        row[f"size_ask_{label}"] = snap.asks[0][1] if snap.asks else None

        # Mid / spread
        if bb is not None and ba is not None:
            row[f"mid_{label}"] = round((bb + ba) / 2, 6)
            row[f"spread_{label}"] = round(ba - bb, 6)
        else:
            row[f"mid_{label}"] = None
            row[f"spread_{label}"] = None

        # Top N price levels
        bids = snap.bids[:TOP_N]
        asks = snap.asks[:TOP_N]

        for i in range(TOP_N):
            if i < len(bids):
                row[f"bid_px_{label}_{i+1}"] = bids[i][0]
                row[f"bid_sz_{label}_{i+1}"] = bids[i][1]
            else:
                row[f"bid_px_{label}_{i+1}"] = None
                row[f"bid_sz_{label}_{i+1}"] = None

            if i < len(asks):
                row[f"ask_px_{label}_{i+1}"] = asks[i][0]
                row[f"ask_sz_{label}_{i+1}"] = asks[i][1]
            else:
                row[f"ask_px_{label}_{i+1}"] = None
                row[f"ask_sz_{label}_{i+1}"] = None

        # Depth and imbalance over top N
        bid_depth = sum(s for _, s in bids)
        ask_depth = sum(s for _, s in asks)
        total = bid_depth + ask_depth
        row[f"bid_depth{TOP_N}_{label}"] = round(bid_depth, 4)
        row[f"ask_depth{TOP_N}_{label}"] = round(ask_depth, 4)
        row[f"imbalance{TOP_N}_{label}"] = (
            round(bid_depth / total, 6) if total > 0 else None
        )

    # last_trade columns — set to None (drain_trades() is destructive and
    # already consumed by VPIN in _poll_book_feed; backtester doesn't use these)
    for label in ("up", "down"):
        row[f"last_trade_px_{label}"] = None
        row[f"last_trade_sz_{label}"] = None
        row[f"last_trade_side_{label}"] = None

    return row


def flush_parquet(rows: list[dict], slug: str, data_subdir: str) -> Path | None:
    """Flush rows to ``data/{data_subdir}/{slug}.parquet`` (append + atomic rename)."""
    if not rows:
        return None
    out_dir = DATA_DIR / data_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    path = out_dir / f"{slug}.parquet"

    if path.exists():
        try:
            existing = pd.read_parquet(path)
            # Normalize legacy column name so old and new rows share one column
            if "chainlink_btc" in existing.columns and "chainlink_price" not in existing.columns:
                existing = existing.rename(columns={"chainlink_btc": "chainlink_price"})
            df = pd.concat([existing, df], ignore_index=True)
        except Exception:
            pass

    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False, engine="pyarrow")
    tmp.replace(path)
    return path


async def record_sampler(
    book_feed,
    up_token: str,
    down_token: str,
    meta: dict,
    price_state: dict,
    window_start_price: float | None,
    slug: str,
    data_subdir: str,
    cancel: asyncio.Event,
    binance_state: dict | None = None,
):
    """Sample 1 row/s, auto-flush every 60s, final flush on cancel."""
    pending: list[dict] = []
    total_rows = 0
    last_flush = _time.monotonic()

    while not cancel.is_set():
        await asyncio.sleep(1)
        if cancel.is_set():
            break

        try:
            row = build_row(
                book_feed, up_token, down_token, meta,
                price_state.get("price"),
                window_start_price,
                binance_state.get("mid_price") if binance_state else None,
            )
            pending.append(row)
        except Exception:
            pass

        # Periodic flush every 60s — offloaded to a thread so the
        # parquet read+concat+write (~50-200ms) doesn't block the
        # event loop and delay signal evaluation.
        if _time.monotonic() - last_flush >= 60 and pending:
            flush_batch = list(pending)  # snapshot before clearing
            pending.clear()
            total_rows += len(flush_batch)
            last_flush = _time.monotonic()
            await asyncio.to_thread(flush_parquet, flush_batch, slug, data_subdir)

    # Final flush
    if pending:
        path = flush_parquet(pending, slug, data_subdir)
        total_rows += len(pending)
        if path:
            print(f"  [REC] Saved {total_rows} rows -> {path}")
