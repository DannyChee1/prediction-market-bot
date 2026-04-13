"""
Parquet classification utility.

94% of our btc_5m parquets are backfilled (no real L2 depth). Training
or analyzing on the mix produces garbage for microstructure features.
This module provides a single source of truth for:

    is_live(path_or_df) -> bool
    is_backfill(path_or_df) -> bool
    is_partial(path_or_df) -> bool
    classify(path_or_df) -> Literal["live", "backfill", "partial", "empty"]
    filter_live(paths) -> list[Path]

Detection signature (from 2026-04-11 audit of all 15k parquets):

    LIVE       : bid_depth5_up is populated (recorder writes real depth)
    BACKFILL   : bid_depth5_up.isna().all() (REST API can't expose depth;
                 `polymarket_rest_backfill.py:279-283` sets NaN explicitly)
    PARTIAL    : live but start_gap_s > 30 (recorder joined window mid-way)

Why not use size_bid_up: the old backfill script fills it with a dummy
constant 100.0, so it's NOT NaN on backfilled files. bid_depth5_up is
the clean flag.

Usage:
    from parquet_kind import classify, filter_live
    kind = classify("data/btc_5m/some.parquet")
    live_paths = filter_live(sorted(Path("data/btc_5m").glob("*.parquet")))
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Union

import pandas as pd

ParquetKind = Literal["live", "backfill", "partial", "empty", "error"]


def _read_classifier_columns(path: Path) -> pd.DataFrame | None:
    """Read only the columns needed for classification. Fast."""
    try:
        return pd.read_parquet(path, columns=[
            "bid_depth5_up",
            "time_remaining_s",
            "window_start_ms",
            "window_end_ms",
        ])
    except Exception:
        return None


def classify(
    src: Union[Path, str, pd.DataFrame],
    *,
    partial_threshold_s: float = 30.0,
) -> ParquetKind:
    """Classify a parquet as live / backfill / partial / empty / error.

    Args:
        src: path to parquet OR an already-loaded DataFrame with at
             least bid_depth5_up, time_remaining_s, window_start_ms,
             window_end_ms columns.
        partial_threshold_s: a live parquet that starts more than this
                             many seconds into its window is classified
                             as "partial". Default 30s = 10% of a 5m
                             window or 3.3% of a 15m window.

    Returns:
        One of "live", "backfill", "partial", "empty", "error".
    """
    if isinstance(src, (str, Path)):
        df = _read_classifier_columns(Path(src))
        if df is None:
            return "error"
    else:
        df = src
    if df is None or len(df) == 0:
        return "empty"

    # Backfill signature — the only reliable flag.
    if "bid_depth5_up" not in df.columns:
        return "backfill"
    if df["bid_depth5_up"].isna().all():
        return "backfill"

    # Partial: depth is populated (so it's from the live recorder) but
    # it started late. The recorder sometimes joins mid-window.
    try:
        wdur_s = (df["window_end_ms"].iloc[0] - df["window_start_ms"].iloc[0]) / 1000
        first_tau = df["time_remaining_s"].iloc[0]
        start_gap_s = wdur_s - first_tau
    except Exception:
        return "error"
    if start_gap_s > partial_threshold_s:
        return "partial"
    return "live"


def is_live(src: Union[Path, str, pd.DataFrame]) -> bool:
    return classify(src) == "live"


def is_backfill(src: Union[Path, str, pd.DataFrame]) -> bool:
    return classify(src) == "backfill"


def is_partial(src: Union[Path, str, pd.DataFrame]) -> bool:
    return classify(src) == "partial"


def filter_live(
    paths: list[Path],
    *,
    partial_threshold_s: float = 30.0,
    verbose: bool = False,
) -> list[Path]:
    """Return only the paths that classify as 'live'. Preserves order."""
    keep = []
    counts: dict[ParquetKind, int] = {
        "live": 0, "backfill": 0, "partial": 0, "empty": 0, "error": 0,
    }
    for p in paths:
        k = classify(p, partial_threshold_s=partial_threshold_s)
        counts[k] = counts.get(k, 0) + 1
        if k == "live":
            keep.append(p)
    if verbose:
        print(f"  filter_live: kept {len(keep)} / {len(paths)} ({100*len(keep)/max(1,len(paths)):.1f}%)")
        for k, v in counts.items():
            if v:
                print(f"    {k}: {v}")
    return keep
