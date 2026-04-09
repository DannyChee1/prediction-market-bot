 Full Audit Report — prediction-market-bot                                                                                      
                                                                                                                                 
  I read the code first-hand (signal_diffusion.py decision path, market_config.py, uncommitted diffs in                          
  orders.py/tracker.py/rust/src/client.rs, the Kelly/calibration layer, and every finding doc) and ran five parallel deep-dive   
  audits across the signal core, live-trading layer, state/execution layer, Rust feeds, and a research pass over strategy and    
  mechanism. What follows is the consolidated picture. Everything is cited file:line; everything is prioritized by blast radius ×
   ease.                                                                                                                         
                                                                                                                               
  ---                                                                                                                            
  TL;DR — the seven things that are actively bleeding you
                                                                                                                                 
  These are the findings where I'm most confident the bot is either losing money now or producing results you can't trust. Fix 
  these before anything else.                                                                                                    
                                                                                                                               
  1. The Rust BookFeed/UserFeed reconnect loop deadlocks on hb.await (rust/src/feed.rs:378 and :1089). The _cancel notifier is   
  never fired (the field is literally named with an underscore prefix because it's "unused"), so the heartbeat task's only exit
  condition never happens, and let _ = hb.await hangs forever on every reconnect. What you see as "reconnect" in the log probably
   never actually reconnects cleanly. Combined with #2, this is the single highest-impact bug in the repo and explains the "book
  WS disconnect" memory note.                                                                                                    
  2. BookFeed's last_update_ts is a global across all tokens (rust/src/feed.rs:253), not per-token. For a two-outcome YES/NO
  market this mostly holds together, but for any multi-token feed it means the staleness clock tells you "did anything update"   
  rather than "did this side update." Combined with the memory finding ("Polymarket only sends updates when the book changes"),
  this is the direct cause of the calm-market false-fires that led to the 1000ms→5000ms bump on btc_5m. The root fix is per-token
   timestamps inside SimpleBook + bumping the ts on PONG, not widening the gate.                                               
  3. pending_fills and open_orders still aren't persisted — confirmed by two independent audits. A crash between a fill and      
  resolve_window permanently loses the on-chain CTF position with no way to recover: scan_unredeemed_wins only finds positions 
  that wrote a redemption_enqueued log, which only happens at window end. Any mid-window crash forfeits that position's bankroll.
   The existing lifetime -$28.92 / peak $111 / trough $80 (from restart_marker_2026-04-09.md) is consistent with "we lost a few
  positions this way." Also: tracker.py:856-858 (the "cannot resolve window" fallback branch) silently refunds cost_usd to the   
  in-memory bankroll and clears pending_fills, producing a phantom bankroll credit for a position that is still alive on-chain.
  Same failure mode, different cause.                                                                                            
  4. The WS partial-fill handler in orders.py:876-939 treats partial fills as full fills. It adds the order to processed_ids on
  the first partial, falls back to size_matched = order["shares"] (the full order), and removes the order from open_orders. The
  remaining unfilled shares keep resting on Polymarket as a live order with no Python-side tracking. Every partial fill produces 
  a phantom resting order and a bankroll accounting mismatch. Multiple related follow-on bugs: partial-fill refund bugs at
  orders.py:212, :736, :941 refund the full cost_est instead of cost_est − already_filled; window_trade_count increments per     
  partial instead of per-order (breaks the max_trades_per_window=1 guarantee); position_count also inflates per partial.       
  5. The regime classifier is completely dead in live. _maybe_compute_regime is only called from decide()
  (signal_diffusion.py:1446), but tracker.py:429 always calls decide_both_sides(), which never invokes the HMM. Both .pkl files
  load, the kelly_mult passthrough is plumbed, and none of it runs. Every live trade uses kelly_mult=1.0. This isn't just a
  missed feature — the research agent and I both confirmed the classifier also returns high_acc→1.0 on virtually all features    
  when it does run, so fixing the wiring is half the battle. Remove the dead path or retrain with aggressive multipliers; don't
  leave it halfway.                                                                                                              
  6. build_diffusion_signal silently drops config.min_entry_z, config.edge_persistence_s, and config.window_duration_s for     
  non-15m backtests. This is the exact class of parity bug you already got bitten by on market_blend (memory:
  post_fix_revalidation_2026-04-07.md). Concrete consequences: (a) every btc_5m backtest runs with min_entry_z=0.0 even when the
  config says 0.15 or 0.5; (b) backtests never apply the edge-persistence gate; (c) btc_5m backtests run with                    
  window_duration=900, making dyn_threshold = edge_threshold * (1 + early_edge_mult * sqrt(tau/900)) — at tau=300s the early-edge
   multiplier is under-applied by ~42% throughout the 5m window, and inventory_skew * (tau/window_duration) is wrong in lockstep.
   Every parameter you've tuned on 5m backtests is tuned against a signal that doesn't match live. The doc comment at          
  signal_diffusion.py:852-861 calling this function a "single source of truth" is not accurate.
  7. Live max_trade_tape_age_ms is silently inoperative. live_trader.py:131-133 populates _chainlink_age_ms, _binance_age_ms,
  _book_age_ms but never _trade_tape_age_ms. signal_diffusion.py:1001-1005 short-circuits to "fresh" on missing keys, so the     
  fourth of four configured gates is a no-op in the live path. Ships in config, never fires.
                                                                                                                                 
  ---                                                                                                                          
  The rest, prioritized                     
                                                                  
  P1 — Correctness / accounting bugs that are silently wrong but not catastrophic alone
                                         
  P1.1 — GIL-held-during-lock hazard in Rust feeds. BookFeed::snapshot, drain_trades, UserFeed::drain_events all                 
  RUNTIME.block_on(async { self.books.read().await }) without py.allow_threads. The Python thread holds the GIL while one of two
  tokio worker threads is blocked on a tokio::sync::RwLock. If the write side is holding the lock inside the big parse block     
  (feed.rs:248-374), the Python thread deadlocks against a tokio worker that can't make progress because of GIL contention. This
  has been working by luck because snapshot() is called from the Python main thread which is external to the tokio runtime.
  OrderClient::get_balance at client.rs:261-287 is ALSO a regression of the 2026-04-09 GIL fix — it's the one hot-path method
  that still holds the GIL during the HTTP roundtrip. The fix is the same pattern the 2026-04-09 diff applied to place_order:
  wrap in py.allow_threads.                                                                                                      
                                                                                                                                 
  P1.2 — _cancel_open_orders clobbers verify-pending retention. orders.py:964-970 does self.open_orders = [] unconditionally     
  after the loop. This destroys the _verify_pending flag that _cancel_single_order (orders.py:290) deliberately set for orders   
  whose cancel couldn't be verified. Result: a cancel-verify-failed order silently loses both its bankroll reservation and its 
  on-chain position at the next window boundary. One-line fix: self.open_orders = [o for o in self.open_orders if 
  o.get("_verify_pending")].                
                                                                  
  P1.3 — recording.flush_parquet blocks the event loop with O(n²) I/O every 60s. recording.py:170-174 calls pd.read_parquet(path)
   + pd.concat + df.to_parquet(tmp) synchronously on the asyncio loop thread during every flush. The file grows monotonically
  during the window, so by the 14-minute mark of a 15m window the flush re-reads ~840 rows and rewrites ~900, on the same thread 
  that's supposed to be running signal_ticker. This can be hundreds of milliseconds of frozen decision loop 15 times per window.
  Fix: wrap with await asyncio.to_thread(...) and switch to pyarrow.parquet.ParquetWriter row-group append. recorder.py:745-749  
  has the same pattern.                                                                                                        
                                 
  P1.4 — signal_ticker appends to price_history on every 10ms idle wake. live_trader.py:144-158 falls through to the append block
   even when timed_out=True and the sequence hasn't changed. Result: during calm markets the history is filled with 100 duplicate
   prices per second. _compute_vol_deduped filters out consecutive identical prices before computing returns, but
  _build_ohlc_bars (the Yang-Zhang path) does not — it builds flat OHLC bars with zero range, which YZ reads as zero-vol         
  contributions, systematically biasing realized σ downward on calm markets. Since σ is the denominator of z, this is a direct 
  lever on the number of trades you fire on calm markets — and it's firing the wrong direction. Gate the append on "effective
  price actually changed".       

  P1.5 — Shared state file between 15m and 5m trackers. live_state_btc.json is shared by both trackers                           
  (live_trader.py:1229-1230), but each tracker maintains independent all_results and lifetime_* in memory. Whichever tracker
  saves last wins; on --resume, both trackers load the merged file into their own all_results, silently double-counting past     
  trades. Also missing from load_state: peak_bankroll, max_drawdown, max_dd_pct, total_fees, window_first_side. Drawdown       
  reporting is wrong until a new peak is seen.
                                                                  
  P1.6 — decide() vs decide_both_sides() divergence. The two decision paths have materially different gate sets:
  decide_both_sides is missing the regime classifier, the delta-velocity OLS filter, the hard-imbalance rejection, the momentum
  gate, and the edge-persistence gate is implemented only in the legacy (not as_mode) branch. Live always runs decide_both_sides;
   backtest only runs it under --maker. All FOK backtest results are not transferable to live. Either commit to one path or
  document the other as deprecated.                                                                                              
                                                                                                                               
  P1.7 — Min-order-size floor defeats filtration size_mult mode. signal_diffusion.py:1519-1526: when filt_mult * kelly_f shrinks
  size_usd below 5 * eff_price (~$0.50-$5), the floor snaps it back to the full 5-share minimum. The graduated downsizing between
   0 and 0.5 is silently overridden; the only filt_mult values that actually do anything are 0 (FLAT) and ≥ ~0.5. This is latent 
  until the next filtration model ships with softer output. On a $100 bankroll with 5-share minimums it's already biting.
                                                                                                                                 
  P1.8 — Partial-fill bankroll refund math. Three sites (orders.py:212, 736, 941) refund cost_est on a CANCELLED/EXPIRED order 
  without subtracting previously-filled portions. If an order had $10 of partials filled and $20 remaining unfilled, the refund
  is $30 instead of $20 — you get a +$10 phantom credit on every partial-then-cancelled order. Bankroll drifts positive relative
  to reality.                                                                                                                    
                                            
  P1.9 — window_trade_count increments on every partial. orders.py:528-531, 609-611, 713-715, 793-794, 917-919 all bump          
  window_trade_count += 1 per fill event, not per order. A single order filled in 3 partials trips max_trades_per_window=1 on the
   first partial and inflates position_count by 3. The same-direction-stacking guarantee is broken.
                                                                                                                                 
  P1.10 — Non-atomic JSONL logging + silent exception swallows. tracker.py:1287-1292 _log uses plain open("a") + write with no
  fsync and a bare except Exception: pass. The diagnostic cadence was just dropped to 5s, so you're doing 12× more of these per  
  second — and the critical events (limit_fill, ws_fill, resolution, redemption_enqueued) share the path. A crash mid-write of a
  redemption_enqueued record means scan_unredeemed_wins won't find the condition and the win is lost. Fix: fsync on the
  critical-event whitelist; always log OSError to stderr instead of swallowing.                                                  
                                                                  
  P1.11 — Calibration table bin width mismatch vs current z range. backtest_core.py:438 has Z_BIN_WIDTH = 0.5, so bins are at    
  -2.0, -1.5, …, +2.0 (9 bins). Post-2026-04-09 the 15m min_entry_z is 0.15 and the comment explicitly states "typical |z| is  
  0.03-0.31 range". Every trade that passes the gate rounds to z_bin = 0 — a single calibration cell that is symmetric around    
  p=0.5 and contains both z=+0.25 and z=-0.25 observations. The calibration fusion is therefore pulling p_model toward 0.5 for 
  virtually every trade that fires, and then market_blend=0.5 pulls it toward mid_up. You're double-shrinking. The effective     
  signal is roughly "trust the market, ignore z." Short-term fix: halve Z_BIN_WIDTH and rebuild the calibration table. Better  
  fix: move to an empirical quantile map (see P3.1 below).

  P1.12 — Filtration train/inference parity. Three independent issues, all in the same bug class:                                
  - imbalance5_up/imbalance5_down are read from a parquet column at train time (train_filtration.py:188-189) and recomputed from 
  L2 depth at inference (signal_diffusion.py:761-765). Whether the two match depends on whether the recorder wrote the column    
  with the same formula. Probably doesn't.                                                                                     
  - polymarket_rest_backfill.py:286, 300 writes NaN for these columns. Training then does float(row.get("imbalance5_up", 0.0) or 
  0.0) — in Python float('nan') or 0.0 returns NaN because NaN is truthy, so the or guard is ineffective. Every backfilled row   
  trains with NaN features, which XGBoost handles via a "default direction" that's whatever minimized loss on the NaN subset —
  not what you want at inference where imbalance is a real number.                                                            
  - mid_momentum is computed by tick count in training (training ~1 Hz, so 60 ticks ≈ 60 s) and at inference by tick count on a  
  multi-Hz live stream (60 ticks ≈ 0.5-5 s depending on market activity). The feature is silently scaled by the tick rate. Fix:
  store (ts_ms, mid) tuples in _mid_up_history and compute momentum by wall-clock time.                                          
                                                                                                                               
  Together these three are why every filtration retraining experiment (hawkes_filtration, filtration_regression, cost-sensitive) 
  has come back negative. It's not the model class; it's the features.                                                         
                                                                                                                                 
  P1.13 — Binance feed auto-restart bypasses SBE mode. live_trader.py:337 does binance_feed = BinanceFeed(symbol) directly on a
  60s stale detection, skipping _make_binance_feed which honors BINANCE_FEED_MODE=sbe and the API key. An operator who runs in   
  SBE mode will silently downgrade to JSON on every reconnect without any log.                                                 
                                                                                                                                 
  P1.14 — Edge-persistence gate resets on upstream gate flaps. signal_diffusion.py:2022-2046. When an upstream gate (stale
  feature, spread, disagreement) returns FLAT before the persistence block, neither _edge_up_first_ms nor the timer is touched.  
  On resumption, now_ms - _edge_up_first_ms keeps accumulating wall clock across the interruption, so "edge persisted for 10s"   
  can include 8 seconds where upstream gates were rejecting the whole tick. Probably not what you want the gate to measure.
                                                                                                                                 
  P1.15 — max_book_age_ms is measuring time-since-last-BBO-change, not WS liveness. Confirmed by two independent audits: the Rust
   side only bumps last_update_ts on real book/price_change messages; PING and PONG both continue before touching it. The        
  1000→5000 ms bump on btc_5m was a symptom; the real fix is bumping last_update_ts on PONG, so the gate becomes a true "WS    
  alive" check rather than a proxy for "book is moving."                                                                         
                                            
  P2 — Correctness concerns, latent bugs, rough edges                                                                            
                                                                                                                               
  - Rust C6/C7: no sequence/gap detection on price_change, no book reset on reconnect (old stale book lingers until the next full
   book snapshot), no crossed-book detection — negative spreads and in-cross mids propagate to Python silently. price_change with
   size < 0 is inserted as negative liquidity at feed.rs:342 because the only guard is size == 0 → remove.
  - Rust H13, H12: the staleness clock is wall-clock SystemTime::now() (can step backward under NTP), taken after JSON parse, and
   after acquiring the write lock. For sub-100ms latency research (oracle lag) this is too coarse; for the 5s gate it's fine.  
  - Rust M1: OrderedFloat::cmp via unwrap_or(Ordering::Equal) — NaN prices silently collapse into an arbitrary BTreeMap key,
  violating the Ord invariant (UB for BTreeMap). Filter NaN at parse time.                                                       
  - Rust M8: connect_async has no timeout — default TCP/TLS handshake can hang for 75+s on a black-holed IP, stalling the
  reconnect loop for over a minute per attempt.                                                                                  
  - Rust stderr spam: [BookFeed] no data for 30s, reconnecting and [PriceFeed] ... fire every 30s during calm markets because the
   feeds are event-driven and the server is idle. The task plan F12 is exactly this. The one-line fix is a rate-limit (log on    
  state transitions only).                                                                                                     
  - market_api.py:140 / recorder.py:163: if now < end or start > now: return is a logic bug. The or should be and; as written it 
  returns future-not-yet-started markets. Both files have the same bug.
  - _apply_regime_z_scale math is inverted (signal_diffusion.py:483). The factor is sigma_calibration / sigma_per_s and is       
  multiplied into z_raw — but z_raw already has sigma_per_s in its denominator, so multiplying by sigma_calibration/sigma_per_s
  is equivalent to dividing z twice by σ. The correct direction is sigma_per_s / sigma_calibration. Currently gated off, so      
  latent — but the moment anyone flips regime_z_scale=True, the signal quadruples in quiet regimes.                            
  - decide_both_sides's _window_duration_s bootstrap off-by-one (signal_diffusion.py:1806-1808). dur_s =                         
  ctx.get("_window_duration_s", tau + 1.0) then ctx["_window_duration_s"] = snapshot.time_remaining_s on the first tick. When the
   bot joins mid-window at tau=400, dur_s is stored as 400 forever, and elapsed_frac = 1 - tau/400 goes negative. Matters for    
  market_adaptive tail mode specifically.                                                                                        
  - Momentum gate in decide() compares binance-mid history against chainlink window_start_price (signal_diffusion.py:1380-1393).
  The bias is exactly the basis between the two feeds, which is nonzero on fast moves.                                           
  - Backtest walk-forward calibration reads z from decision.reason string (backtest.py:548-553) via rstrip("(cap)") — but      
  str.rstrip removes characters, not the literal suffix. "1.0a".rstrip("(cap)") becomes "1.". Use .removesuffix() or, better, add
   z_raw as a typed field on Fill.                                                                                             
  - Deflated Sharpe haircut is double-annualized (backtest.py:663). The haircut should be subtracted from the per-period SR      
  before annualization, not multiplied by ann_factor after.       
  - tracker.py:705-710 _evaluate_exits recomputes z/p_model using snapshot.chainlink_price instead of Binance mid. Exits are     
  evaluated on the lagged price — not hot today because exits are disabled, but a landmine.                                    
  - tracker.new_window clears pending_fills unconditionally (tracker.py:360) even though it also prints the unresolved-fill      
  warning. Combined with the restart gap, any carry-over at window boundary vanishes. The warning is printed; the on-chain CTF
  position is abandoned.                                                                                                         
  - Dashboard worker doesn't populate any *_age_ms fields (dashboard_signal_worker.py:203-221). Dashboard decisions silently   
  diverge from live for the same snapshot — dashboard says "would trade" when live would gate out on freshness.                  
  - Non-btc market configs (eth/sol/xrp/eth_5m/sol_5m/xrp_5m) have none of the post-Kou-fix defenses: no min_sigma override, no  
  market_blend, no max_model_market_disagreement, no stale-feature gates, no min_entry_z adjustment. If any of these are enabled
  in live, they're running the pre-fix unprotected signal. Audit before enabling.                                                
  - Several unused dead paths that confuse future readers: the Kou parameters on btc configs (inert under tail_mode="kou"), the
  HMM regime classifier files, feeds.py legacy WebSocket coroutines (only snapshot_from_book_feed is imported by live_trader),   
  as_mode branches in the signal that no market uses.                                                                          
                                                                                                                                 
  P2 inefficiencies worth noting                                                                                                 
                                                        
  - _compute_vol runs three times per tick when filtration is enabled: once for raw_sigma, once for sigma_baseline, once inside  
  _check_filtration. Each is a full Yang-Zhang rebuild on 90-300 ticks. Use an incremental estimator.                          
  - _contract_sigma_p linear-scans _contract_mids every tick → O(n²) per window. Use bisect_left.                                
  - _mid_up_history is a Python list trimmed by slicing (del hist[:-600]), which is O(n). Use collections.deque(maxlen=600).     
  - tracker.all_results is unbounded in memory; the display sum(r.pnl for r in all_results_combined) runs every second.
  - Rust hot-path: .cloned() of serde_json::Value arrays on every WebSocket batch (feed.rs:242), per-message SystemTime::now()   
  and .to_uppercase(), BTreeMap over a 20-level book where a sorted Vec would beat it.                                           
                                                            
  ---                                                                                                                            
  P3 — Strategic & model-level findings                                                                                        
                                                                                                                                 
  These are the parts where the underlying strategy itself is suspect, not just the implementation.                              
                                                                                                                                 
  P3.1 — Your measured "edge" is almost entirely market_blend shrinkage, and the backtest fill model is fantasy.                 
                                                                                                                                 
  Two compounding issues. First, post-Kou-fix the model is a driftless Gaussian Brownian bridge (signal_diffusion.py:1012-1032 — 
  the "kou" path just returns norm_cdf(z); the Kou lambda/eta params on btc are inert decoration). The comprehensive review      
  showed that when this model disagrees with the Polymarket mid, the market is right 70% of the time. Sharpe goes 0.61 → 1.36 as
  market_blend goes 0 → 0.5 on 15m. ROI goes 5% → 9% as blend goes 0 → 0.3 on 5m. Every increment toward the market improves
  results. The diffusion model is not adding edge; the shrinkage toward market is.
                                                                                                                                 
  Second, backtest.py:220-233 posts maker limit orders at best_bid, fills them instantly for free, and models no queue position, 
  no fill probability, and no conditional adverse-selection haircut. But live maker fills happen precisely when informed flow    
  walks through the queue to your level — Glosten-Milgrom adverse selection. The direct live measurement is already in the repo: 
  btc_5m_stale_book_gate_2026-04-06.md shows live maker trades at book_age < 100ms win 57.7% / +$6.21 on 71 trades while the old
  1000ms gate missed trades that win 25% / -$6.13 on 4 trades. Backtest Sharpe was 1.54 on the same sample. Backtest-to-live     
  Sharpe is gapped by 2-4× and everything in market_config.py was tuned against fantasy fills.                                 
                                 
  The single most impactful fix is to build an adverse-selection-aware fill model in the backtest (queue position + conditional
  mid drift + haircut), then re-sweep edge_threshold, kelly_fraction, and market_blend on top of it. Until this is done, every   
  A/B comparison is comparing two numbers that are both wrong in the same direction. This is prerequisite to trusting any other
  tuning.                                                                                                                        
                                                                                                                               
  P3.2 — Replace Φ(z) with an empirical (z_bin, τ_bin) → p_up quantile map.
                                                        
  distribution_fit_2026-04-05.md finding #3: the end-of-window z-score std ≈ 0.68-0.79 instead of 1.0, which means σ is
  systematically over-estimated by ~35% and every Φ(z) is compressed toward 0.5. The recent min_sigma and min_entry_z hotfixes   
  are downstream symptoms of the same input bias. An empirical CDF built per (z, τ) bucket on the post-fix REST-backfill (use
  narrower z bins than the current 0.5) auto-corrects the σ bias, the τ-dependence, the drift, and the non-Gaussianity in one    
  step. It also removes tail_mode/kou_lambda/tail_nu_default/max_z as tunables. Combine with Bayesian shrinkage toward         
  norm_cdf(z) via count-weighting, so low-count bins fall back to the Gaussian rather than producing noise. This subsumes P1.11
  (calibration bin width) and P2 _apply_regime_z_scale.                                                                          
  
  P3.3 — Add a real drift term. The "driftless Brownian bridge" is not the right null.                                           
                                                                                                                               
  external_signals_test_2026-04-05.md:63-71: BTC 5m has 57.1% UP during US hours with 1.25× larger moves. Non-trivial            
  physical-measure drift exists at the 5-15 min horizon. The current model assumes μ=0 identically. A 60s rolling              
  signed-volume-weighted return from the Binance trade tape (which you already subscribe to) is a cheap drift estimator; clip to
  ±1-2pp and feed into the z-score as z = (δ - μ·elapsed) / (σ·√τ). A 2pp drift correction is larger than the current 
  edge_threshold minus market_blend residual for most trades.
                                                                  
  P3.4 — The filtration model's failures are feature poverty, not architecture.                                                  
                                                        
  Three failed retrains (Hawkes features, PnL regression, cost-sensitive) all use the same feature set: mid/depth/σ/τ. The       
  microstructure signal that actually predicts 5-15m crypto returns is flow, not depth: VPIN/OFI/CVD on the Binance trade tape,
  and taker-hit signals on the Polymarket user feed (which you also already subscribe to).                                       
  crypto_microstructure_research.md:77-87 cites AUC 0.54-0.61 on these features. You're leaving them on the table. Before another
   filtration retrain, compute (a) Binance 30/60/120s CVD, (b) net taker flow, (c) trade-tape skew, (d) Polymarket taker hits on
  your book side (direct adverse-selection signal), and wire them into the feature extractor. This is also the prerequisite for
  proposal P3.3.                                                                                                                 
  
  P3.5 — OBI, market_blend, and max_model_market_disagreement are three levers pulling on the same signal.                       
                                                                                                                               
  Market_blend pulls p_model toward mid_up. OBI nudge pulls p_model further in the direction of book imbalance (which is the     
  microstructure input that determines the mid). max_model_market_disagreement rejects trades where p_model disagrees with mid_up
   by >30pp. All three are variants of "trust the mid." external_signals_test_2026-04-05.md:77-90 already showed OBI-to-outcome
  correlation is −0.092 (weak and probably wrong sign). Meanwhile you have a Kalman-smoothed OBI gate in the taker path, so it's
  three-and-a-half levers. The inconsistent Sharpe across regimes documented in btc_5m_phase2_negative_2026-04-06.md:37-52
  (OBI-off helps recent trades by +0.13 Sharpe and hurts older ones by −1.25) is almost certainly this multicollinearity.
                                                                                                                                 
  Pick one. If I had to pick: keep market_blend (cleaner mechanism), remove OBI nudge, convert max_model_market_disagreement from
   a hard gate into a variable shrinkage coefficient (P3.6).                                                                     
                                                                                                                               
  P3.6 — Liquidity-weighted Bayesian shrinkage instead of fixed market_blend.                                                    
                                                                                                                               
  Replace p_final = (1-blend)*p + blend*mid with p_final = (w_mid * mid + w_model * p_model) / (w_mid + w_model) where w_mid = 
  depth_at_mid / σ²_mid and w_model = 1 / σ²_model. On a thin Polymarket book, trust the model more; on a thick book, trust the
  mid. Zero knobs if you can get σ²_mid from recent mid variance over the trade tape. This generalizes the fixed market_blend
  kludge to a real shrinkage estimator.                                                                                          
                                            
  P3.7 — Microprice/VAMP everywhere, not just vamp_mode="cost".                                                                  
                                                                                                                                 
  Every mid_up = (bid + ask) / 2 in the pipeline should be a microprice (bid*ask_size + ask*bid_size) / (bid_size + ask_size) or
  a top-3-level VAMP (you already have compute_vamp). This includes the market_blend target, the max_model_market_disagreement   
  reference, and the toxicity-parity check. Microprice is ~20-30% more predictive of the next trade price than mid, especially on
   thin books. Low effort, straightforward.                                                                                      
                                                                                                                               
  P3.8 — Drawdown-aware Kelly.                                    
                                                                                                                                 
  f_effective = kelly_fraction * (1 - dd_pct / max_allowed_dd) * calibration_trust where calibration_trust is the p-value that
  the recent WR is within 1σ of the calibrated p_model. The bot's −$28.92 lifetime came from running the same size through a bad 
  run. A drawdown brake would have halved it after the first 10% DD.                                                           
                                                                                                                                 
  P3.9 — Cross-horizon consistency trade (5m vs 15m).                                                                          
                                                                  
  Inside every 15m window there are three nested 5m windows. The two markets must be consistent: p_up(15m) ≈ f(p_up(5m_1),       
  p_up(5m_2), p_up(5m_3)). When they disagree by more than the residual arbitrage cost, take the side your model predicts is
  correct. Backtestable with existing parquets. Free edge from a consistency condition the market is known to miss.              
                                                                                                                               
  P3.10 — Cross-asset lead-lag as confirmation, not veto.                                                                        
                                                                                                                               
  BTC→ETH same-direction rate is 78.5% (comprehensive_strategy_review_2026-04-05.md:68). The cross_asset_z_lookup infrastructure 
  already exists but only as a disagreement veto. When BTC is moving strongly 4 minutes into its window and ETH's window hasn't
  repriced, buy ETH. Convert from veto to additive bias.                                                                         
                                                                                                                               
  ---                                                                                                                            
  P4 — Stop doing                                                                                                              
                                                                                                                                 
  - Stop tuning the Kou parameters. They are inert under tail_mode="kou". Either delete them or commit to kou_full and do the
  bipower-variation work. The current state looks actionable but isn't — someone will waste a day here.                          
  - Stop treating classification AUC as evidence for filtration. Three filtration retrains have been decided by AUC/decile lift
  and all have come back PnL-negative. The only valid metric is walk-forward Sharpe on a properly held-out slice.                
  - Stop re-sweeping market_blend. It's converged: 0.3 on 5m, 0.5 on 15m. Further sweeps are noise.                            
  - Stop running Hawkes experiments without new motivation. The plan's posture ("if F7 also fails, delete the infra") is the
  right one. Commit to it.                                        
  - Stop treating the REST-backfilled sample as authoritative. It has synthetic L2 depth, proxy Chainlink = Binance, and no real
  spreads. It's good for blend sweeps and bad for anything that depends on feed latency, depth, or staleness.
  - Stop chasing "direct Chainlink" as a panacea. The 1.23s rebroadcast tax is real but f4_oracle_lead_lag_2026-04-08.md already 
  shipped negative — other bots see similar lag and the CLOB mid has absorbed most of it. Higher ROI elsewhere.
  - Stop adding more gates to the signal until you consolidate the existing ones. You have 20+ tunables per market and at least  
  four of them do overlapping work (P3.5). Every tuning session is 10 levers interacting.                                      
                                                                                                                                 
  ---                                                                                                                            
  Recommended order of operations                                                                                                
                                                                                                                                 
  If you do exactly this list in order, you will close out everything that's actively wrong and then be in a position to do    
  research from a trustworthy baseline.                                                                                          
                                                                                                                               
  Day 1 — stop the bleeding (P0)                                                                                                 
  1. rust/src/feed.rs: fix the hb.await deadlock (notify_waiters() before await; break heartbeat on send error). Move          
  last_update_ts into SimpleBook per-token. Bump last_update_ts on PONG so the 5000ms gate becomes a real liveness check, not a  
  BBO-change proxy. Wrap snapshot/drain_trades/drain_events/get_balance in py.allow_threads. This alone probably explains most of
   the "phantom freeze" and calm-market stale-gate episodes in the memory.                                                       
  2. tracker.py:856-858: fix the "cannot resolve window" branch to preserve pending_fills instead of refunding and clearing.
  3. tracker.py save_state + live_trader._build_tracker: persist pending_fills and open_orders with full per-fill cost and       
  _verify_pending flags. On load, reconcile against Polymarket REST.                                                      
  4. orders.py:970: one-line fix — self.open_orders = [o for o in self.open_orders if o.get("_verify_pending")].                 
  5. orders.py:876-939: fix the WS partial-fill handler to keep orders in open_orders until MATCHED. Also fix the three bankroll
  refund sites to subtract already_filled. Also fix window_trade_count / position_count to track per-order, not per-fill.        
  6. signal_diffusion.py: wire _maybe_compute_regime into decide_both_sides, OR rip out the HMM pkls and kelly_mult plumbing.  
  7. backtest.py:959-999: pass every MarketConfig field through to DiffusionSignal. Add a unit test assert signal.<attr> ==      
  config.<attr> for every field.                                                                                               
  8. live_trader.py:131-133: populate ctx["_trade_tape_age_ms"] from trade_state["current_bar"]["start_ts"].
                                                                                                                                 
  Day 2 — close the feedback loop                                                                                                
  9. recording.py:170-174: await asyncio.to_thread(flush_parquet, ...) + ParquetWriter row-group append.                         
  10. live_trader.py:144-158: gate price_history append on "eff_px actually changed."                                            
  11. live_trader.py:337: reconnect Binance via _make_binance_feed, not BinanceFeed().                                           
  12. Split live_state_{base}.json per-timeframe, or move to a shared state object.                                              
  13. tracker.py _log: fsync on critical event types; log OSError to stderr.                                                     
  14. market_api.py:140 + recorder.py:163: or → and in find_market.                                                              
  15. Halve backtest_core.py:438 Z_BIN_WIDTH to 0.25 and rebuild the calibration table.                                          
                                                                                                                                 
  Week 1 — close the backtest-to-live gap                                                                                        
  16. Build the adverse-selection-aware fill model in backtest.py (P3.1). Queue position, conditional mid drift between post and 
  fill, half-spread haircut. Re-tune edge_threshold, kelly_fraction, market_blend on top of it.                                  
  17. Replace Φ(z) with an empirical (z, τ) quantile map (P3.2). Keep norm_cdf(z) as a Bayesian fallback when the bin is         
  underpopulated.                                                                                                                
  18. Fix the filtration train/inference parity bugs (P1.12) and retrain with the same dataset you've been using before adding   
  new features. Lock down the baseline before any new-feature experiments.                                                       
                                                                                                                                 
  Week 2+ — research from a trustworthy baseline                                                                                 
  19. Add trade-tape features (CVD, net taker flow, SVWAP) as P3.4. Retrain filtration with them.                                
  20. Add a drift estimator from Binance trade tape (P3.3). Apply as an additive z-score correction with clipping.               
  21. Consolidate market_blend + OBI + disagreement gate into one liquidity-weighted shrinkage (P3.6).                         
  22. Audit non-BTC configs for post-fix defenses before enabling any of those markets live.                                     
                                                                                                                               
  ---                                                                                                                            
  Things I'm not touching because the operator has already vetoed them                                                         
                                                                                                                                 
  - Resolution-snipe / late-tau strategies. Memory feedback_no_end_window_snipe.md is clear and I'm respecting it.             
  - Weekend-skip rules. project_btc_5m_weekend_underperform.md has this as "don't ship."                                         
                                                                                                                                 
  ---                                                                                                                            
  One-line summary                                                                                                               
                                                                                                                                 
  The bot's actual edge mechanism is "market_blend shrinks a bad Gaussian model toward the mid, then the bot trades the tiny     
  residual." That residual would be slightly positive in backtest, but live adverse selection eats it, a broken set of           
  accounting/feed bugs occasionally loses whole positions on top, and a chain of parameter hot-fixes has been piling up against a
   model that was never re-validated after the Kou fix. Fix the correctness stack first (P0 + P1), close the backtest-to-live gap
   with an adverse-selection fill model and an empirical CDF (P3.1 + P3.2), and only then go looking for new features. Adding    
  more levers right now would just add more things to go wrong.                                                                  
                                                                                                                                 
⏺ All six tasks complete. The full audit is above.                                                                               
                                                           
  Quick summary of what I did:                                                                                                   
  - Read the key files first-hand (market_config.py, signal_diffusion.py decision path + Kelly + CDFs, the uncommitted diffs in
  orders.py/tracker.py/rust/src/client.rs, calibration table, restart marker, plans index)                                       
  - Launched 5 parallel deep-dive subagents (code auditors for signal/backtest, live/feeds, tracker/orders, Rust feeds; a      
  research agent for strategy/model critique) — that's where the bulk of the line-by-line coverage came from                     
  - Synthesized everything into the prioritized report above, cross-checking findings where two audits touched the same code     
                                                                                                                                 
  The seven P0 findings are the ones I'd act on first — especially the Rust reconnect deadlock (#1), per-token last_update_ts    
  (#2), pending_fills persistence + the silent-drop fallback (#3), the WS partial-fill handler (#4), the dead regime classifier  
  (#5), build_diffusion_signal dropping 5m window_duration (#6), and the inoperative max_trade_tape_age_ms gate (#7).            
                                                                                                                                 
  The biggest strategic finding is that the bot's measured "edge" is almost entirely market_blend shrinkage, not the Gaussian    
  diffusion model — and the backtest's idealized free-maker fills hide 2-4× of adverse selection that shows up in live. Until the
   backtest fill model is fixed, every parameter in market_config.py is tuned against fantasy, and every A/B comparison is       
  comparing two wrong numbers in the same direction. 