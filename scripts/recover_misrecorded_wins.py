#!/usr/bin/env python3
"""Find and recover trades the bot misrecorded as losses but actually won.

Background: market_api.poll_market_resolution had a bug (fixed 2026-04-09)
where it would read last-trade prices instead of settlement prices when
the market was `closed=True` but the on-chain resolution hadn't finalized
yet. This caused some real wins to be recorded as losses, leaving the
winning CTF positions unredeemed on-chain.

This script:
  1. Scans live_trades_btc.jsonl for `resolution` records
  2. For each LOSS, queries Gamma API to verify the actual outcome
  3. Reports any trades where the bot's recorded outcome disagrees with reality
  4. With --enqueue, adds the misrecorded wins to the redemption queue
     so the bot's periodic redeem loop will claim them on next startup

Usage:
    uv run python scripts/recover_misrecorded_wins.py             # report only
    uv run python scripts/recover_misrecorded_wins.py --enqueue   # enqueue for redemption
    uv run python scripts/recover_misrecorded_wins.py --since 24h # last 24h only
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "live_trades_btc.jsonl"
QUEUE_FILE = ROOT / "live_redemption_queue_btc.json"
GAMMA_API = "https://gamma-api.polymarket.com"


def parse_since(s: str | None) -> datetime | None:
    if not s:
        return None
    m = re.fullmatch(r"(\d+)\s*([smhd])", s.strip().lower())
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        delta = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days"}[unit]
        return datetime.now(timezone.utc) - timedelta(**{delta: n})
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        sys.exit(f"ERROR: bad --since value '{s}'")


def get_actual_outcome(slug: str) -> tuple[str | None, str]:
    """Returns (actual_winner, status_str) where actual_winner is 'UP'|'DOWN'|None."""
    try:
        resp = requests.get(
            f"{GAMMA_API}/events", params={"slug": slug}, timeout=10
        )
        data = resp.json()
        if not data:
            return None, "no_data"
        m = data[0]["markets"][0]
        if not m.get("closed"):
            return None, "not_closed"
        if m.get("umaResolutionStatus") != "resolved":
            return None, f"uma={m.get('umaResolutionStatus','?')}"
        prices = m.get("outcomePrices", "?")
        if isinstance(prices, str):
            try:
                prices = json.loads(prices)
            except json.JSONDecodeError:
                return None, "bad_prices"
        if not isinstance(prices, list) or len(prices) != 2:
            return None, "malformed_prices"
        # Settled prices are exactly "0" and "1"
        try:
            price_floats = sorted(float(p) for p in prices)
        except (ValueError, TypeError):
            return None, "non_numeric_prices"
        if price_floats != [0.0, 1.0]:
            return None, f"unsettled prices={prices}"
        # outcomes is ["Up","Down"]
        outcomes = m.get("outcomes", [])
        if isinstance(outcomes, str):
            try:
                outcomes = json.loads(outcomes)
            except json.JSONDecodeError:
                return None, "bad_outcomes"
        up_idx = outcomes.index("Up")
        winner = "UP" if str(prices[up_idx]) in ("1", "1.0") else "DOWN"
        return winner, "resolved"
    except Exception as exc:
        return None, f"err: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enqueue", action="store_true",
                    help="Enqueue misrecorded wins for redemption (default: report only)")
    ap.add_argument("--since", help="Filter to trades since this point (e.g. '24h', '2d', ISO ts)")
    args = ap.parse_args()

    cutoff = parse_since(args.since)

    # 1. Collect all resolution records (and existing redemption_enqueued records)
    resolutions: list[dict] = []
    enqueued_slugs: set[str] = set()
    enqueued_cids: set[str] = set()
    with open(LOG) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = r.get("type", "")
            if t == "resolution":
                ts = r.get("ts", "")
                if cutoff and ts:
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        if dt < cutoff:
                            continue
                    except ValueError:
                        pass
                resolutions.append(r)
            elif t in ("redemption_enqueued", "redemption_started"):
                slug = r.get("market_slug")
                cid = r.get("condition_id")
                if slug:
                    enqueued_slugs.add(slug)
                if cid:
                    enqueued_cids.add(cid)

    # 2. Filter to LOSSES only — only those need verification
    losses = [r for r in resolutions if r.get("pnl", 0) <= 0]
    print(f"Found {len(resolutions)} resolutions, {len(losses)} recorded as losses")
    if cutoff:
        print(f"(filtered to since {cutoff.isoformat()})")
    print()

    # 3. Verify each loss against the API
    misrecorded: list[tuple[dict, str]] = []
    skipped = 0
    for i, r in enumerate(losses):
        slug = r.get("market_slug", "")
        our_side = r.get("side", "")
        bot_outcome = r.get("outcome", "")
        if not slug or not our_side:
            skipped += 1
            continue
        actual, status = get_actual_outcome(slug)
        if actual is None:
            print(f"  [{i+1}/{len(losses)}] {slug}  SKIP ({status})")
            skipped += 1
            continue
        if actual == our_side:
            # We actually won — bot misrecorded
            misrecorded.append((r, actual))
            print(f"  [{i+1}/{len(losses)}] {slug}  ⚠️  MISRECORDED  bot={bot_outcome}  actual={actual}  side={our_side}  shares={r.get('shares')}")
        else:
            print(f"  [{i+1}/{len(losses)}] {slug}  ok ({actual})")

    print()
    print(f"=== Summary ===")
    print(f"  Total losses checked:    {len(losses) - skipped}")
    print(f"  Skipped (unresolved):    {skipped}")
    print(f"  Misrecorded as losses:   {len(misrecorded)}")
    if misrecorded:
        total_owed = sum(r.get("shares", 0) for r, _ in misrecorded)
        print(f"  Total shares unredeemed: {total_owed:.1f}")
        print(f"  Approx $ unredeemed:     ${total_owed:.2f}  (1 share = $1 at resolution)")
    print()

    if not misrecorded:
        print("Nothing to recover. Bye.")
        return 0

    # 4. Look up condition_ids for the misrecorded slugs from the trade log
    # We need the condition_id to enqueue for redemption.
    # Find the most recent limit_fill / limit_order for each slug
    cid_by_slug: dict[str, str] = {}
    with open(LOG) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            slug = r.get("market_slug", "")
            cid = r.get("condition_id") or r.get("conditionId")
            if slug and cid and slug not in cid_by_slug:
                cid_by_slug[slug] = cid

    # If trade log doesn't have condition_id, query Gamma API
    needs_lookup = [slug for r, _ in misrecorded
                    for slug in [r.get("market_slug")]
                    if slug and slug not in cid_by_slug]
    for slug in set(needs_lookup):
        try:
            resp = requests.get(f"{GAMMA_API}/events", params={"slug": slug}, timeout=10)
            data = resp.json()
            if data:
                cid = data[0]["markets"][0].get("conditionId", "")
                if cid:
                    cid_by_slug[slug] = cid
        except Exception:
            pass

    print(f"Resolved condition_ids for {len(cid_by_slug)} slugs")
    print()

    if not args.enqueue:
        print("Run with --enqueue to add these to the redemption queue.")
        for r, actual in misrecorded:
            slug = r.get("market_slug", "")
            cid = cid_by_slug.get(slug, "MISSING")
            print(f"  {slug}  cid={cid[:16]}{'...' if len(cid) > 16 else ''}")
        return 0

    # 5. Enqueue for redemption — append to existing queue file
    # Use the same format as RedemptionQueue
    queue_data: list[dict] = []
    if QUEUE_FILE.exists():
        try:
            queue_data = json.loads(QUEUE_FILE.read_text())
            if not isinstance(queue_data, list):
                queue_data = []
        except json.JSONDecodeError:
            queue_data = []

    existing_cids = {item.get("condition_id") for item in queue_data}
    added = 0
    now_iso = datetime.now(timezone.utc).isoformat()
    for r, actual in misrecorded:
        slug = r.get("market_slug", "")
        cid = cid_by_slug.get(slug)
        if not cid:
            print(f"  SKIP {slug}: no condition_id available")
            continue
        if cid in existing_cids:
            print(f"  SKIP {slug}: already in queue")
            continue
        queue_data.append({
            "condition_id": cid,
            "market_slug": slug,
            "enqueued_at": now_iso,
            "attempts": 0,
            "source": "recover_misrecorded_wins",
        })
        existing_cids.add(cid)
        added += 1
        print(f"  ENQUEUED {slug} (cid={cid[:16]}...)")

    if added > 0:
        QUEUE_FILE.write_text(json.dumps(queue_data, indent=2))
        print()
        print(f"Wrote {added} item(s) to {QUEUE_FILE.name}")
        print("On next bot startup, the periodic redemption loop will pick these up.")
        print("Or restart the bot now to trigger immediate processing.")
    else:
        print("Nothing new to enqueue.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
