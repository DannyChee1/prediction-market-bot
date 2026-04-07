All fixes applied and validated. Here is the headline result.      
                                                                                                    
  Done — all bugs fixed and validated                                                                                            
                                                                                                                                 
  The numbers that matter                                                                                                        
                                                                                                                                 
  BTC 15m, identical seed and data, walk-forward out-of-sample:                                                                  
                                                            
  ┌─────────────────┬────────────┬─────────────────┐                                                                             
  │                 │ Before fix │    After fix    │        
  ├─────────────────┼────────────┼─────────────────┤                                                                             
  │ Total PnL       │    −$2,462 │         +$3,663 │        
  ├─────────────────┼────────────┼─────────────────┤
  │ Sharpe (annual) │      −0.85 │           +0.72 │                                                                             
  ├─────────────────┼────────────┼─────────────────┤                                                                             
  │ Win rate        │      54.4% │           63.7% │                                                                             
  ├─────────────────┼────────────┼─────────────────┤                                                                             
  │ Max drawdown    │      47.5% │           29.0% │        
  ├─────────────────┼────────────┼─────────────────┤                                                                             
  │ Verdict         │  ✗ NO EDGE │ ✓ EDGE DETECTED │        
  └─────────────────┴────────────┴─────────────────┘                                                                             
                                                            
  A $6,125 swing on a $10k bankroll from removing one wrong term in _model_cdf and loosening one σ band. Same seed, same data,   
  same code path, only the buggy lines changed.             
                                                                                                                                 
  BTC 5m, same conditions:                                                                                                       
  
  ┌──────────────┬────────────┬─────────────────┐                                                                                
  │              │ Before fix │    After fix    │           
  ├──────────────┼────────────┼─────────────────┤
  │ Total PnL    │      +$361 │    +$863 (2.4×) │
  ├──────────────┼────────────┼─────────────────┤
  │ Win rate     │      53.5% │           58.5% │                                                                                
  ├──────────────┼────────────┼─────────────────┤                                                                                
  │ Sharpe       │       0.05 │            0.08 │                                                                                
  ├──────────────┼────────────┼─────────────────┤                                                                                
  │ Max drawdown │      42.1% │           47.6% │           
  ├──────────────┼────────────┼─────────────────┤                                                                                
  │ Verdict      │   MARGINAL │ ✓ EDGE DETECTED │           
  └──────────────┴────────────┴─────────────────┘                                                                                
  
  (Drawdown went up because trade frequency went up 46% and σ-driven sizing is now non-flat. If the wider DD is uncomfortable,   
  dial kelly_fraction from 0.25 → 0.15 — but check that against the post-fix backtest, don't fudge the model.)
                                                                                                                                 
  Calibration ECE on the full multi-month dataset (kou markets only):                                                            
  
  ┌─────────────┬────────┬────────┬────────┐                                                                                     
  │             │ Before │ After  │   Δ    │                
  ├─────────────┼────────┼────────┼────────┤                                                                                     
  │ BTC 15m ECE │ 0.0423 │ 0.0273 │ −35.5% │                
  ├─────────────┼────────┼────────┼────────┤
  │ BTC 5m ECE  │ 0.0802 │ 0.0445 │ −44.5% │                                                                                     
  └─────────────┴────────┴────────┴────────┘                                                                                     
                                                                                                                                 
  ETH/SOL/XRP markets are byte-identical pre/post (they don't use tail_mode="kou"), confirming the fix is surgical.              
                                                            
  What was changed                                                                                                               
                                                            
  ┌──────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────┐  
  │               File               │                                         Change                                         │
  ├──────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ backtest.py:1374-1389            │ _model_cdf kou path now returns norm_cdf(z) — drift correction removed. Long comment   │
  │                                  │ in place explaining why so this never gets reverted.                                   │
  ├──────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤  
  │                                  │ Same fix in evaluate_params/get_bin_details; "current params" baseline now reads from  │
  │ analysis/optimize_kou_5m.py      │ MARKET_CONFIGS["btc_5m"] instead of being hard-coded; printout warns the lambda search │  
  │                                  │  is now inert.                                                                         │
  ├──────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤  
  │ analysis/calibration_analysis.py │ Same fix in model_cdf mirror.                                                          │
  ├──────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ market_config.py btc_5m          │ min_sigma 7e-5 → 1e-5, max_sigma 8e-5 → 2e-4. Old band clamped 88.4% of σ samples; new │
  │                                  │  band tracks the empirical p10..p99 of realized vol from 9,812 BTC 5m observations.    │  
  ├──────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ market_config.py btc             │ min_sigma 3e-5 → 1e-5, max_sigma → 4e-4. Same logic, p5..p99 of 106k observations.     │  
  ├──────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤  
  │ tests/test_model_cdf.py (new)    │ 10 regression tests, all passing. Covers symmetry, no-blowup-at-low-σ, kou≡normal      │
  │                                  │ post-fix, monotonicity, student_t untouched, and kou_cdf internal consistency.         │  
  ├──────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ validation_runs/ (new)           │ Full before/after backtest logs, calibration logs, calibration PNG, and RESULTS.md     │  
  │                                  │ with the side-by-side diff.                                                            │  
  └──────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────┘
                                                                                                                                 
  Test artifacts                                            

  $ python tests/test_model_cdf.py
  PASS  test_kou_asymmetric_p_up_does_not_distort_z0                                                                             
  PASS  test_kou_cdf_clt_branch_internally_consistent                                                                            
  PASS  test_kou_cdf_reduces_to_normal_when_lambda_is_zero                                                                       
  PASS  test_kou_no_blowup_at_low_sigma          ← the BTC 15m σ-floor regression                                                
  PASS  test_kou_path_equals_normal_path                                                                                         
  PASS  test_kou_strictly_monotone_in_z                                                                                          
  PASS  test_kou_symmetric_at_extremes                                                                                           
  PASS  test_kou_symmetric_returns_half_at_z0                                                                                    
  PASS  test_market_config_sigma_bounds_reasonable                                                                               
  PASS  test_student_t_unchanged                                                                                                 
  10/10 tests passed                                        
                                                                                                                                 
  What this means for your existing project memory          

  Two of your saved memories were workarounds for the bugs, not intrinsic findings:                                              
  
  - project_btc5m_market_blend.md (market_blend=0.3 fixes DOWN side): the blend was attenuating Bug A's downward bias by 30%.    
  With Bug A fixed, the blend may no longer be needed — re-run the same A/B with market_blend=0 and market_blend=0.3 on post-fix
  code to see which wins.                                                                                                        
  - project_btc_5m_weekend_underperform.md (Sat/Sun Sharpe ~3 vs weekday ~7-10): this was the 1/sigma blow-up of the drift bias
  hitting the σ floor on quiet days. Post-fix data should show weekend Sharpe converging with weekday. Re-test before adding any 
  weekend-skip rule.
                                                                                                                                 
  I haven't touched memory — that's your call. But both entries describe symptoms that the fix addresses at the root, so they're 
  stale-by-construction.
                                                                                                                                 
  What I deliberately did not touch                                                                                              
  
  - Yang-Zhang vol estimator on micro-bars (correct math, just suboptimal — it's a future improvement, not a bug)                
  - Half-Kelly sizing math (correct as-is)                  
  - Filtration model, microstructure gates, oracle lag, fee math                                                                 
  - The full proper kou_cdf function — left intact at backtest.py:164 as dead code in case you later want to wire it in to       
  actually fatten tails (would require revisiting whether sigma_per_s is total or continuous)                                    
  - Anything in the sibling /Users/dannychee/Desktop/prediction-market-bot/ working tree — all changes are in -research/. When   
  you're ready to ship, git diff between the two directories will give you a clean patch.                                        
                                                            
  To reproduce                                                                                                                   
                                                            
  python tests/test_model_cdf.py                                                # 10/10 pass
  python analysis/calibration_analysis.py                                       # writes ECE table + PNGs                        
  python backtest.py --market btc_5m --signal diffusion --train-frac 0.7 --seed 42                                               
  python backtest.py --market btc    --signal diffusion --train-frac 0.7 --seed 42                                               
                                                                                                                                 
  Full before/after logs and the side-by-side diff table are in validation_runs/RESULTS.md.                                      
                                                                                                                                 
✻ Brewed for 28m 15s                                                                                                             
                                                                                                    
⏺ Background command "Run baseline btc_5m backtest" completed (exit code 0)
                                                                                                                                 
⏺ Background command "Run baseline btc 15m backtest" completed (exit code 0)
                                                                                                                                 
⏺ Acknowledged — those are the two baseline backtests from earlier in the session. Both finished successfully and their output
  was already captured to validation_runs/baseline_btc_5m.txt and validation_runs/baseline_btc_15m.txt and used in the           
  side-by-side comparison in validation_runs/RESULTS.md. No further action needed.
                                                                                                                                 
❯   - The full proper kou_cdf function — left intact at backtest.py:164 as dead code in case you later want to wire it in to     
                                                                                                                                 
  actually fatten tails (would require revisiting whether sigma_per_s is total or continuous)                                    
                                                                                                                                 
                                                                                                                                 
What is it using right now, why aren't we using this? How long would it taek to test?                                            
                                                                                                                                 
⏺ What's running now                                                                                                             
                                                                                                                               
  After the fix, tail_mode="kou" is doing exactly this in _model_cdf:                                                            
  
  return norm_cdf(z)                                                                                                             
                                                            
  That's it. A plain standard-normal CDF on z = delta / (sigma * sqrt(tau)). The parameters kou_lambda, kou_p_up, kou_eta1,      
  kou_eta2 are still in MarketConfig but completely inert — nothing reads them at runtime. I left the kou label on the tail mode 
  because changing it would force config edits and a re-run of calibration tables, but functionally it's identical to            
  tail_mode="normal" now.                                   

  Why kou_cdf isn't wired in                                                                                                     
  
  Two real obstacles, neither of which is "code is missing":                                                                     
                                                            
  Obstacle 1 — variance double-counting                                                                                          
                                                            
  Your _compute_vol_deduped (backtest.py:314) estimates σ from realized log returns. Realized returns contain whatever jumps     
  actually happened in the lookback window. So sigma_per_s is a total vol estimate — continuous + jumps mashed together.
                                                                                                                                 
  kou_cdf (backtest.py:164) on the other hand decomposes total variance into:                                                    
  
  total_var = sigma**2 * tau   +   lam_tau * ej2                                                                                 
              ↑ continuous          ↑ jump component (added on top)                                                              
                                                                                                                                 
  If I plug your realized σ into the sigma argument and let the function add lam_tau * ej2 jump variance on top, I am counting   
  the jump variance twice. The model gets a too-fat predicted distribution → predictions cluster artificially toward 0.5 → you   
  stop trading edges that are actually real → ECE looks "good" because the model is timid about everything.                      
                                                            
  To use kou_cdf honestly you need σ to be the continuous-only component, not total. The textbook estimator for that is bipower  
  variation:
                                                                                                                                 
  BV = (π/2) · Σ |r_i| · |r_{i-1}|                          
                                                                                                                                 
  It's jump-robust by construction (the |r_i| · |r_{i-1}| product cancels isolated jumps because a jump only contaminates one of 
  the two terms). Then you have σ_continuous² ≈ BV and jump_variance ≈ realized_variance − BV.                                   
                                                                                                                                 
  Obstacle 2 — risk-neutral drift baked into the function                                                                        
   
  kou_cdf was written for option pricing. Look at line 185:                                                                      
                                                            
  mu = -0.5 * sigma**2 - lam * zeta                                                                                              
                                                            
  That mu is the Q-measure martingale-correcting drift. It belongs in option pricing where you need e^X · S_0 to be a martingale 
  under risk-neutral measure. For our physical-measure binary prediction P(S_T > S_0), we want the physical drift, which for
  short-horizon BTC is approximately zero.                                                                                       
                                                            
  So even if obstacle 1 didn't exist, I'd need to add a physical_measure=True flag or just pass mu_override=0.0 into kou_cdf.    
  Trivial code change, but a code change.
                                                                                                                                 
  These two obstacles are exactly why kou_cdf is dead code today rather than being the live path: wiring it in correctly is not a
   one-line fix, and wiring it in incorrectly would silently overstate variance and degrade the model in a way calibration
  metrics wouldn't even flag clearly.                                                                                            
                                                            
  Three test options, ranked by effort

  I'll be honest about what each one actually buys you.                                                                          
   
  Option A — quick-and-dirty test, 30–60 min                                                                                     
                                                            
  Wire kou_cdf in with mu=0.0 and your existing total-σ. Knowingly double-count jump variance, just to see what calibration looks
   like.                                                    
                                                                                                                                 
  What to do:                                                                                                                    
  1. Add a mu_override parameter to kou_cdf (~5 lines).
  2. Add tail_mode="kou_full" branch in _model_cdf that calls it (~10 lines).                                                    
  3. Run analysis/calibration_analysis.py and the two backtests from earlier.
                                                                                                                                 
  What you'll learn: essentially nothing useful. The model will be under-confident because of the double-count. ECE may move in  
  either direction. I do not recommend this unless you really just want to see the function execute end-to-end.                  
                                                                                                                                 
  Option B — bipower variation + proper Kou, 3–6 hours of focused work                                                           
                                                            
  The honest test. Build a jump-robust σ estimator alongside the existing one, use it as the continuous component, and let       
  kou_cdf add jump variance correctly.                      
                                                                                                                                 
  What to do:                                               
  1. Implement bipower variation in backtest.py next to _yang_zhang_vol (~30 lines). Estimator math is unambiguous; I can write
  it from memory.                                                                                                                
  2. Add a _continuous_sigma and _jump_variance accessor to the signal class so live trader and backtest both use it (~40 lines).
  3. Modify kou_cdf to take mu_override so we can pass 0 for physical measure (~5 lines).                                        
  4. Add tail_mode="kou_full" branch in _model_cdf that calls kou_cdf(-delta_log, σ_continuous, λ, p_up, η1, η2, τ,              
  mu_override=0.0) (~20 lines).                                                                                                  
  5. Re-estimate Kou params from observed jumps using max-likelihood, since the existing values were "optimized" against the     
  buggy model (~80 lines for an MLE script — could also just use moment matching as a first cut).                                
  6. A/B test: run calibration_analysis.py on kou_full vs current kou (which is just normal). Run backtests with --tail-mode     
  kou_full on btc_5m and btc_15m and compare to the post-fix-baseline numbers we just established.                           
  7. Add unit tests for bipower variation (it has a known closed-form result on a Brownian path with no jumps — BV → integrated  
  variance as N → ∞).                                                                                                           
                                                                                                                                 
  What you'll learn: whether jump variance actually helps prediction in 5m/15m horizons. My honest expectation: a small ECE 
  improvement (~10-20% on top of the post-fix ECE) and a small Sharpe improvement on BTC 15m, where jumps matter most. Not a     
  14%→24% style win — the bug fix already captured most of that.
                                                                                                                                 
  This is the test I would actually run if you want a real answer.                                                               
   
  Option C — full Hawkes-modulated time-varying λ, 1–2 days                                                                      
                                                            
  Same as B, but instead of a constant kou_lambda, fit a self-exciting Hawkes process to the inter-arrival times of jumps in your
   Binance trade tape. Each tick gets its own λ(t) = μ + α · Σ exp(-β·(t-t_i)).
                                                                                                                                 
  This is the right model for crypto because jumps cluster. But it's a meaningful project: Hawkes parameter estimation, online   
  state tracking, and the whole bipower variation pipeline from B as a prerequisite.
                                                                                                                                 
  What you'd learn: whether jump clustering matters on top of jump variance. Probably yes, but only on volatile days — and even  
  then, the marginal lift over Option B is uncertain.
                                                                                                                                 
  My recommendation                                                                                                              
   
  Skip A. Do B if you want a real answer. C is a separate project — don't tangle it up.                                          
                                                            
  The fact that the bug fix alone produced a $6,125 swing on BTC 15m means we already captured the majority of the available     
  improvement on the probability model. The question Option B answers is "is there an additional 0.05–0.20 Sharpe in proper jump
  modeling on top of the corrected baseline?" That's worth ~3-6 hours to find out.                                               
                                                            
  If you say go on B, the work decomposes cleanly:                                                                               
   
  1. Bipower variation function + unit test against synthetic Brownian (45 min)                                                  
  2. kou_cdf mu_override parameter + unit test against the existing Q-measure call (15 min)
  3. New tail_mode="kou_full" branch + plumbing through _p_model (30 min)                                                        
  4. Crude moment-match Kou param estimation script (45 min)                                                                     
  5. Run calibration + 2 backtests + write validation_runs/RESULTS_kou_full.md (1.5–2 hours, mostly waiting)                     
  6. Decision: ship kou_full or stay with the corrected normal (15 min)                                                          
                                                                                                                                 
  Want me to do it?                                                             