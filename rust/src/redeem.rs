//! Redemption queue with persistence, split-and-retry, and poison-pill
//! detection.
//!
//! Batching one CTF.redeemPositions call across N conditionIds is atomic —
//! if any single position in the batch is poison (already redeemed, not yet
//! settled on-chain, etc.), the whole tx reverts. The naive "requeue on fail"
//! loop then burns quota forever on the same doomed batch.
//!
//! This module handles three failure modes distinctly:
//!   1. Quota exhaustion (HTTP 429 / "quota exceeded"): re-queue unchanged,
//!      back off 30 min. Not the cid's fault, don't penalize it.
//!   2. On-chain revert ("relayer tx FAILED"): split the batch, retry each
//!      cid individually. Cids that fail get their attempt count bumped;
//!      dropped after MAX_ATTEMPTS with a loud log.
//!   3. Quota hit mid-split: stop burning quota, re-queue the rest, back off.
//!
//! The queue is persisted to disk so restarts don't strand pending wins in
//! CTF (unredeemed → no USDC).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;

/// After this many individual-retry failures, a cid is dropped with a warning.
/// Each individual retry happens within a flush cycle, so MAX_ATTEMPTS=6
/// with a 10-min FLUSH_INTERVAL gives a cid up to ~60 min to resolve its
/// transient issues before being dropped. UMA settlement occasionally lags
/// 30-45 min behind Gamma-reported resolution, so 3 strikes (original) was
/// too impatient and risked dropping legitimately-pending wins.
const MAX_ATTEMPTS: u8 = 6;

/// Max cids per batched redeemPositions tx. Polymarket's proxy encoding caps
/// at 15.
const BATCH_MAX: usize = 15;

/// Flush interval — how often we drain the queue and call the relayer.
/// Longer = more queueing, fewer quota units. 10 min keeps daily flushes
/// under Polymarket's relayer quota even in busy days.
const FLUSH_INTERVAL_S: u64 = 600;

/// Warmup delay before the FIRST flush. Short so a restart with a persisted
/// queue processes backlog promptly instead of sitting idle for 10 min.
const INITIAL_DELAY_S: u64 = 30;

/// Back-off after a 429/quota error. Polymarket's relayer uses a rolling
/// daily quota, so 30 min is enough to let some units reset.
const RATE_LIMIT_COOLDOWN_S: u64 = 1800;

/// Small gap between individual retries during split so we don't slam the
/// relayer back-to-back.
const INDIVIDUAL_RETRY_DELAY_MS: u64 = 150;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct QueuedRedemption {
    pub condition_id: String,
    /// Number of times THIS cid has failed an individual-retry attempt.
    /// Batch failures don't increment this (we don't know which cid was
    /// poison yet). Only bumped inside split_retry.
    #[serde(default)]
    pub attempts: u8,
}

impl QueuedRedemption {
    #[allow(dead_code)] // only used when self-redemption is re-enabled
    pub fn new(condition_id: String) -> Self {
        Self { condition_id, attempts: 0 }
    }
}

pub fn save_queue(path: &Path, queue: &[QueuedRedemption]) {
    match serde_json::to_string_pretty(queue) {
        Ok(json) => {
            if let Err(e) = std::fs::write(path, json) {
                eprintln!("  [REDEEM] persist failed: {e}");
            }
        }
        Err(e) => eprintln!("  [REDEEM] serialize failed: {e}"),
    }
}

pub fn load_queue(path: &Path) -> Vec<QueuedRedemption> {
    match std::fs::read_to_string(path) {
        Ok(data) => match serde_json::from_str::<Vec<QueuedRedemption>>(&data) {
            Ok(v) => {
                // De-dupe in case an older build persisted duplicates.
                // Preserves first-seen order and keeps the max attempts count
                // so we don't accidentally "heal" a cid that's been retrying.
                let mut seen = std::collections::HashMap::<String, u8>::new();
                let mut order: Vec<String> = Vec::new();
                for item in &v {
                    match seen.get_mut(&item.condition_id) {
                        Some(a) => *a = (*a).max(item.attempts),
                        None => {
                            seen.insert(item.condition_id.clone(), item.attempts);
                            order.push(item.condition_id.clone());
                        }
                    }
                }
                let deduped: Vec<QueuedRedemption> = order
                    .into_iter()
                    .map(|cid| QueuedRedemption {
                        attempts: seen[&cid],
                        condition_id: cid,
                    })
                    .collect();
                let dropped = v.len() - deduped.len();
                if !deduped.is_empty() {
                    if dropped > 0 {
                        eprintln!(
                            "  [REDEEM] restored {} pending redemptions from {} ({} duplicates removed)",
                            deduped.len(),
                            path.display(),
                            dropped,
                        );
                    } else {
                        eprintln!(
                            "  [REDEEM] restored {} pending redemptions from {}",
                            deduped.len(),
                            path.display()
                        );
                    }
                }
                deduped
            }
            Err(e) => {
                eprintln!("  [REDEEM] load failed ({e}), starting empty");
                Vec::new()
            }
        },
        Err(_) => Vec::new(),
    }
}

fn is_rate_limit(err: &str) -> bool {
    err.contains("429") || err.contains("quota")
}

fn short_cid(cid: &str) -> &str {
    cid.get(0..10).unwrap_or(cid)
}

/// Top-level redemption loop. Sleeps FLUSH_INTERVAL_S between flushes.
/// Always returns queue modifications to disk after each cycle, so a crash
/// between flushes loses at most 1 cycle of newly-queued wins (which are
/// also persisted by the parent's periodic save).
pub async fn run_redeem_loop(
    http: reqwest::Client,
    queue: Arc<Mutex<Vec<QueuedRedemption>>>,
    queue_path: PathBuf,
) {
    // Short warmup before first flush so other subsystems (feeds, order
    // client) stabilize. Then flush first, sleep second — this ensures a
    // restart with a persisted queue processes backlog within ~30s rather
    // than waiting a full 10-min cycle.
    tokio::time::sleep(Duration::from_secs(INITIAL_DELAY_S)).await;

    loop {
        let batch: Vec<QueuedRedemption> = {
            let mut q = queue.lock();
            let take = q.len().min(BATCH_MAX);
            q.drain(..take).collect()
        };

        if !batch.is_empty() {
            eprintln!("  [REDEEM] flushing batch of {} positions", batch.len());
            let cids: Vec<String> =
                batch.iter().map(|r| r.condition_id.clone()).collect();
            match crate::redemption::redeem_positions(&http, &cids).await {
                Ok(tx_hash) => {
                    eprintln!("  [REDEEM] OK batch={} tx={tx_hash}", cids.len());
                    persist_snapshot(&queue, &queue_path).await;
                }
                Err(e) => {
                    let err_str = e.to_string();
                    eprintln!("  [REDEEM] FAIL batch={}: {err_str}", cids.len());
                    if is_rate_limit(&err_str) {
                        // Quota problem — not the cids' fault. Re-queue unchanged.
                        requeue(&queue, batch).await;
                        persist_snapshot(&queue, &queue_path).await;
                        eprintln!("  [REDEEM] rate limited, extending cooldown");
                        tokio::time::sleep(Duration::from_secs(RATE_LIMIT_COOLDOWN_S))
                            .await;
                    } else {
                        // On-chain revert: isolate the poison via split-retry.
                        split_retry(&http, batch, &queue, &queue_path).await;
                    }
                }
            }
        }

        tokio::time::sleep(Duration::from_secs(FLUSH_INTERVAL_S)).await;
    }
}

async fn requeue(queue: &Arc<Mutex<Vec<QueuedRedemption>>>, batch: Vec<QueuedRedemption>) {
    let mut q = queue.lock();
    // Insert at front in reverse order so the original ordering is preserved.
    for r in batch.into_iter().rev() {
        q.insert(0, r);
    }
}

async fn persist_snapshot(queue: &Arc<Mutex<Vec<QueuedRedemption>>>, path: &Path) {
    let snap: Vec<QueuedRedemption> = queue.lock().clone();
    save_queue(path, &snap);
}

/// Retry each cid in the failed batch individually. Classify each outcome:
///   - OK: cid is removed (drained).
///   - Individual FAIL with attempts < MAX: re-queue, keep trying next cycle.
///   - Individual FAIL with attempts >= MAX: drop with loud log — this is
///     almost certainly a permanently poison cid (already redeemed, not
///     actually settled on-chain, etc.) and retrying wastes quota forever.
///   - Rate limited mid-loop: stop immediately, re-queue the rest unchanged,
///     extend cooldown. We don't want a single bad batch to burn all our
///     daily quota via split-retry amplification.
async fn split_retry(
    http: &reqwest::Client,
    batch: Vec<QueuedRedemption>,
    queue: &Arc<Mutex<Vec<QueuedRedemption>>>,
    queue_path: &Path,
) {
    eprintln!(
        "  [REDEEM] split-retry: trying {} cids individually",
        batch.len()
    );
    let mut quota_hit = false;
    let mut to_requeue: Vec<QueuedRedemption> = Vec::new();

    for mut item in batch {
        if quota_hit {
            to_requeue.push(item);
            continue;
        }
        tokio::time::sleep(Duration::from_millis(INDIVIDUAL_RETRY_DELAY_MS)).await;
        let single = vec![item.condition_id.clone()];
        match crate::redemption::redeem_positions(http, &single).await {
            Ok(tx) => {
                eprintln!(
                    "  [REDEEM] OK cid={} tx={tx}",
                    short_cid(&item.condition_id)
                );
            }
            Err(e) => {
                let err_str = e.to_string();
                if is_rate_limit(&err_str) {
                    quota_hit = true;
                    eprintln!("  [REDEEM] quota hit mid-split; stopping to conserve units");
                    to_requeue.push(item);
                    continue;
                }
                item.attempts = item.attempts.saturating_add(1);
                if item.attempts >= MAX_ATTEMPTS {
                    eprintln!(
                        "  [REDEEM] DROP cid={} after {} attempts: {err_str}",
                        short_cid(&item.condition_id),
                        item.attempts
                    );
                    // Intentionally not re-queued → dropped.
                } else {
                    eprintln!(
                        "  [REDEEM] individual FAIL cid={} attempts={}/{}: {err_str}",
                        short_cid(&item.condition_id),
                        item.attempts,
                        MAX_ATTEMPTS
                    );
                    to_requeue.push(item);
                }
            }
        }
    }

    if !to_requeue.is_empty() {
        requeue(queue, to_requeue).await;
    }
    persist_snapshot(queue, queue_path).await;

    if quota_hit {
        tokio::time::sleep(Duration::from_secs(RATE_LIMIT_COOLDOWN_S)).await;
    }
}
