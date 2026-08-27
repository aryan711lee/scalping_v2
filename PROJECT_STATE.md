# PROJECT STATE

## Current Phase
Phase 7: Paper Trading — **Not Started**

## Status
Phases 1–6.5 and EXP_015/EXP_016 complete. EXP_016 (L6 1.4%/0.7% labels) confirmed Scenario C: L6 is not a learnable pattern on 3-min NSE data (best fold precision 6.3% vs 40.8% break-even). Phase 7 proceeds with EXP_015 configuration: L1 XGBoost model + L5 exits (2.00%/1.00%). See findings below.

## Architecture
- Broker: Fyers (primary)
- Database: SQLite (Phase 1)
- Timeframes: 1-min, 3-min, 5-min (research on 3-min)
- Universe: 15 liquid NSE stocks (NIFTY 100)
- Capital (eventual): Rs.5,000
- Mode: Paper trading first

## What Is Working

### Phase 1 — Data Infrastructure
- [x] Repo structure created
- [x] Fyers client (90-day chunking, retry logic)
- [x] Historical data downloaded (15 symbols × 3 timeframes, 2023-01-01 onward)
- [x] Cleaning pipeline
- [x] Resampling (1-min → 3-min, 5-min)
- [x] Database populated (SQLite, INSERT OR IGNORE upsert)
- [x] Validation passing (99.7–99.8% completeness, all symbols >= 95%)
- [x] NIFTY 50 index fetcher (data/ingestion/nifty.py)
- [x] 23 tests passing

### Phase 2 — Feature Engineering
- [x] Price action features (price_action.py)
- [x] Trend features — EMA(9/21/50), VWAP with daily reset (trend.py)
- [x] Momentum features — RSI with Wilder's smoothing, ROC (momentum.py)
- [x] Volatility features — ATR (Wilder's), Bollinger Bands, realized vol (volatility.py)
- [x] Volume features — volume ratio, OBV with daily reset (volume.py)
- [x] Market context features — NIFTY context, sine/cosine time encoding (market_context.py)
- [x] Regime features — trend regime, ATR percentile, volatility regime (regime.py)
- [x] Master pipeline (features/pipeline.py) and build script (scripts/build_features.py)
- [x] Feature files saved: 45 files (15 symbols × 3 timeframes), 64 features each
- [x] Feature validation passing (no NaN, no inf)
- [x] 42 new tests passing (65 total)

### Phase 3 — Baseline Strategy & Backtester
- [x] strategy/base.py — BaseStrategy abstract class
- [x] strategy/baseline.py — EMA/RSI/VWAP rule-based Model Zero
- [x] backtester/costs.py — full NSE intraday cost model (brokerage, STT, exchange fees, GST, stamp duty, SEBI charges, slippage)
- [x] backtester/portfolio.py — capital tracking, fixed_pct and full_capital sizing, Trade dataclass
- [x] backtester/engine.py — bar-by-bar simulation, signal@T fills@T+1 open (no look-ahead)
- [x] backtester/metrics.py — Sharpe, Calmar, profit factor, expectancy, max drawdown, regime/ToD breakdown
- [x] backtester/report.py — console summary + logs/ file output
- [x] scripts/run_backtest.py — CLI script with --strategy/--timeframe/--symbol/--from/--to args
- [x] Full backtest run: 15 symbols, 3min, 2023-01-02 to 2026-08-19
- [x] Report and trade CSV saved to logs/
- [x] EXP_001 added to EXPERIMENT_LOG.md
- [x] 63 new tests passing (128 total, 0 failures)

### Phase 5 — Walk-Forward Validation Framework
- [x] validation/splitter.py — WalkForwardSplitter with 3 folds + hardcoded test period
- [x] validation/dataset_splitter.py — split(), get_test_split(), get_feature_columns(), get_class_weights()
- [x] validation/evaluator.py — all metrics incl. combined_precision, signal_rate, aggregate_fold_results
- [x] validation/leakage_checker.py — check_fold_leakage(), check_test_isolation(), check_feature_leakage()
- [x] scripts/build_folds.py — end-to-end: loads data, splits, leakage checks, baseline eval, saves metadata
- [x] Fold summary table printed and saved to logs/fold_report.txt
- [x] fold_metadata.json saved with row counts and class distributions
- [x] All 7 fold index Parquets saved (fold_1/2/3 train+val + test_index)
- [x] Leakage checks: CLEAN on all 3 folds and test isolation
- [x] Baseline evaluated through fold evaluator — EXP_003 added to EXPERIMENT_LOG.md
- [x] 43 new tests passing (203 total, 0 failures)
- [x] PROJECT_STATE.md updated

### EXP_015 — L5 Labels (Between Phase 6.5 and Phase 7)
- [x] labels/variants.py — L5 variant added (target=2.00%, stop=1.00%, horizon=60)
- [x] 45 L5 label files built (15 symbols × 3 timeframes) — all flagged for low density
- [x] L5 distribution documented: 3min avg long~1.0%, short~0.7% — below 3% threshold
- [x] Scenario D: label density too low for ML training; backtest run with L1 model + L5 exits
- [x] EXP_015 backtest: 38.1% WR vs 38.5% break-even — near break-even, major improvement on EXP_014
- [x] EXP_015 Finding: L1 model signals DO identify 2% moves when held for 60 candles (180 min)
- [x] Phase 7 primary config revised: L1 model + L5 exits (2.00%/1.00%)

### EXP_016 — L6 Labels (Sweet-Spot Hypothesis Test)
- [x] labels/variants.py — L6 variant added (target=1.40%, stop=0.70%, horizon=40)
- [x] 45 L6 label files built (15 symbols × 3 timeframes)
- [x] L6 distribution documented: 3min avg long~2.1% — CAUTION zone (gate: ≥4% ideal)
- [x] XGBoost trained on L6: combined_precision=4.6%±1.5%, signal_rate=13.8%
- [x] Threshold sweep: best precision 6.3% at t=0.70 — 34pp below 40.8% break-even
- [x] Scenario C confirmed: L6 is not a learnable pattern on 3min NSE data
- [x] No backtest run (sweep conclusively ruled out viability)
- [x] Phase 7 config unchanged: L1 model + L5 exits (EXP_015 result)

### Phase 4 — Label Engineering & Research Environment
- [x] labels/variants.py — L1, L2, L3, L4 variant definitions
- [x] labels/constructor.py — bar-by-bar three-class labeling (all 5 critical rules enforced)
- [x] labels/validator.py — distribution, leakage, cross-variant checks
- [x] datasets/builder.py — feature+label join, session filters, multi-symbol builder
- [x] scripts/build_labels.py — end-to-end build script with validation reporting
- [x] 180 label files saved to data_storage/labels/ (15 symbols × 3 timeframes × 4 variants)
- [x] Validation report: no flags raised on any of the 180 combinations
- [x] research/notebooks/01_label_exploration.ipynb — label distribution analysis
- [x] research/notebooks/02_feature_correlation.ipynb — feature correlation & ranking
- [x] research/notebooks/03_baseline_review.ipynb — EXP_001 deep dive
- [x] EXP_002 added to EXPERIMENT_LOG.md with actual label distributions
- [x] 32 new tests passing (160 total, 0 failures before Phase 5)

## What Is Broken
None.

## Current Strategy
Baseline (Model Zero): EMA9/21/50 crossover + RSI 40-65 + VWAP above + volume ratio >= 1.2.

EXP_001 results (honest):
- 2,239 trades | Win rate: 16.5% | Net P&L: Rs.-170,125 (-340.3%)
- Gross P&L barely positive (Rs.8,811); costs paid Rs.178,936 (2031% of gross)
- Expectancy per trade: Rs.-75.98

Phase 6 ML (EXP_008b, selected final configuration):
- Best model: XGBoost L1, threshold=0.5, 55 features
- 1,906 trades | Win rate: 18.5% | Net P&L: Rs.-133,145 (-266.3%)
- Expectancy per trade: Rs.-69.86 (+Rs.6.12/trade vs baseline)
- Still loss-making: ML improves selectivity but cannot overcome structural cost problem
- Primary finding: model learned "when NOT to trade" (time-of-day dominates) rather than direction

EXP_002 results — label distribution (3min, avg 15 symbols):
- L1: +1=17.1% | -1=18.0% | 0=64.9% (recommended for Phase 6)
- L2: +1=11.7% | -1=12.0% | 0=76.2%
- L3: +1=25.6% | -1=26.8% | 0=47.6%
- L4: +1=14.1% | -1=14.7% | 0=71.2%

## Decisions Made
- Fyers chosen over Angel One for better data quality
- 3-min chosen as primary research timeframe
- SQLite for Phase 1 (upgrade to PostgreSQL in later phases if needed)
- Paper trading mode first
- WATCHLIST: replaced NSE:TATAMOTORS-EQ with NSE:KOTAKBANK-EQ (Fyers returns -300 invalid symbol for TATAMOTORS)
- VWAP reset: groupby(date).cumsum() — resets at 09:15 every trading day
- OBV reset: within-day diff() via groupby(date) — first candle of each day contributes 0
- RSI: Wilder's smoothing via ewm(alpha=1/14, adjust=False) — NOT simple rolling mean
- ATR: Wilder's smoothing via ewm(alpha=1/14, adjust=False)
- 64 features per symbol/timeframe (spec estimated ~54; actual count after implementation)
- Engine execution order: exits -> signals -> fill pending (spec pseudocode order)
  so a position entered at candle T+1 step 3 has its first exit check at candle T+2 step 1
- Engine uses to_dict("records") per day per symbol for fast row access (avoids iloc overhead)
- Unicode: Rs. used instead of Rs. symbol in console output (Windows CP1252 limitation)

## Decisions Rejected
- NIFTY/BANKNIFTY options: requires too much capital
- 1-min as primary timeframe: too noisy for initial research
- Rolling VWAP across days: corrupts all downstream VWAP features
- Simple rolling mean for RSI: gives systematically wrong values
- Checking exits on same candle as entry fill (would cause unrealistic immediate stops)

### Phase 6.5 — Signal Quality Improvement (Infrastructure)
- [x] features/momentum_quality.py — candle_consistency_3, volume_price_trend, ema9_slope
- [x] features/multi_timeframe.py — add_5min_context with direction='backward' + 5min timestamp shift
- [x] features/pipeline.py — updated: add_momentum_quality + optional df_5min → add_5min_context
- [x] scripts/build_features.py — updated: builds 5min→3min→1min order, caches 5min for 3min pass
- [x] scripts/train_models.py — updated: --variant (L1/L2/L3/L4) and --symbols args added
- [x] models/confidence_filter.py — ConfidenceFilter with top_pct/window_candles/min_confidence
- [x] scripts/analyse_per_symbol.py — EXP_010 per-symbol breakdown from trade log CSV
- [x] scripts/run_confidence_sweep.py — EXP_013 hyperparameter sweep
- [x] scripts/run_exp014_backtest.py — EXP_014 best combination backtest
- [x] tests/test_momentum_quality.py — 11 tests
- [x] tests/test_multi_timeframe.py — 9 tests (including look-ahead alignment + tz-aware datetime regression test)
- [x] tests/test_confidence_filter.py — 8 tests
- [x] models/confidence_filter.py — vectorized predict() via pandas rolling quantile (O(n) vs O(n×window))
- [x] 263 total tests passing (235 Phase 1-6 + 28 Phase 6.5), 0 failures

### Phase 6 — ML Model Training Pipeline
- [x] models/trainer.py — ModelTrainer with fold/final training, XGBoost label encoding, StandardScaler on train only
- [x] models/predictor.py — ModelPredictor with threshold-based signal selectivity, proba output [short/none/long]
- [x] models/selector.py — ModelSelector with composite score (precision - instability penalty - signal_rate penalty)
- [x] models/artifacts/ — xgboost_L1_final.joblib + best_model_meta.json saved
- [x] strategy/ml_strategy.py — MLStrategy wrapping predictor, afternoon filter, BaseStrategy compatible
- [x] scripts/train_models.py — EXP_004→007: LR, RF, XGBoost, threshold tuning; best model metadata saved
- [x] scripts/run_ml_backtest.py — EXP_008a (avoid_afternoon=True) and EXP_008b (avoid_afternoon=False) vs EXP_001
- [x] scripts/smoke_test_training.py — one symbol, one fold, 10k rows (pipeline validation)
- [x] tests/test_trainer.py, test_predictor.py, test_selector.py, test_ml_strategy.py — 32 tests, all passing
- [x] research/notebooks/04_model_results.ipynb — fold comparisons, threshold sweep, EXP_008 results, written conclusion
- [x] 235 total tests passing (203 Phase 1-5 + 32 Phase 6), 0 failures

## EXP_015 Finding (2026-08-23)

**Scenario D (label density) + Backtest run with L1 model + L5 exits.**

L5 labels: 0.5–1.6% signal density per direction on 3-min candles — below 3% threshold. L5 ML model not trained.

Backtest run using EXP_014 L1 model with L5 exit params (2%/1%) to quantify the cost-structure improvement empirically.

**EXP_015 backtest results (2026 holdout, L1 signals + L5 exits):**
- Win rate: **38.1%** (vs break-even ~38.5% — gap of only 0.4pp)
- Costs/gross: **200%** (vs 1712% with L1 exits — structural improvement confirmed)
- Net P&L: **Rs.-67,964** (vs -123,005 EXP_014, +Rs.55,041)
- Gross P&L: **Rs.67,777** (vs Rs.7,629 — 9× improvement)
- Trades: 1,297 | Max drawdown: -39.6% (vs -99.7%)

**Key finding:** The L1 signal selector captures 2% moves when given 60 candles instead of 20. Win rate jumps from 16.3% → 38.1% purely from changing exit parameters. The break-even gap is now 0.4pp, not 30pp.

**L5 label distribution (3min avg, 15 symbols):** long ~1.0%, short ~0.7%, none ~98.4%

**Revised Phase 7 decision:** Test L5 exit parameters (2%/1%) in paper trading with the L1 model, NOT L1 exits. The L5 exits nearly break even on the 2026 holdout; L1 exits lose at 1712% costs/gross. See EXPERIMENT_LOG.md for full analysis.

## EXP_016 Finding (2026-08-28)

**L6 (1.40%/0.70%, horizon=40)** failed the learnability test:
- Label density: avg 2.1% long (CAUTION zone) — 13/15 symbols individually below 2%
- Fold validation precision: 4.6% ± 1.5% (vs 40.8% break-even) — Scenario C
- Threshold sweep peak: 6.3% at t=0.70 — 34.5pp below break-even
- Conclusion: L6 does not represent a learnable pattern on 3min NSE data

**Root cause:** Large-cap NSE stocks on 3-minute bars do not produce enough 1.4% directional moves (horizon=40 candles) to provide reliable ML training signal. The density floor is approximately 17% (L1). Both L5 and L6 fall in the noise zone.

**Phase 7 configuration (unchanged from EXP_015):**
- **Primary:** L1 XGBoost + confidence filter + L5 exits (target=2.00%, stop=1.00%)
- **Reference:** EXP_014 L1 exits (target=0.40%, stop=0.20%)

**Future options (not started):**
- Option A: NIFTY Midcap 50 universe with L6 labels (higher vol, denser large moves)
- Option B: Daily candles with swing-trading labels (2–5 day holds)

## Next Task
Phase 7: Paper Trading

Phase 7 goal: run the EXP_015 configuration live on Fyers paper account and validate:
1. Execution fills match backtester T+1 open assumption
2. Real slippage vs assumed (0.05% per side)
3. Whether the 38.1% L5 win rate holds on live data (key question from EXP_015)

**Primary config:** XGBoost L1 + confidence filter + L5 exits (target=2.00%, stop=1.00%)
**Reference config:** EXP_014 — XGBoost L1 + confidence filter + L1 exits (target=0.40%, stop=0.20%)

Final model artifact: `models/artifacts/xgboost_L1_final.joblib`
Confidence filter config: `models/artifacts/confidence_filter_meta.json`

## Phase 6.5 Key Findings

### Experiment results (EXP_009–EXP_014, walk-forward 3 folds, L1 unless noted)
| Exp | Description | combined_precision | signal_rate | consistency | Verdict |
|-----|-------------|-------------------|-------------|-------------|---------|
| EXP_009 | L4 labels (wider target/stop) | 24.0% ± 3.1% | 7.3% | 12.8% | Worse than L1 — reject |
| EXP_010 | 5-symbol focused universe | 23.5% ± 5.1% | 11.0% | 21.9% | Worse on all dimensions — reject |
| EXP_011 | +3 momentum_quality features | 27.0% ± 3.7% | 4.1% | 13.5% | Neutral (within error of baseline) |
| EXP_012 | +5 multi-timeframe 5-min features | 27.4% ± 2.6% | 4.0% | 9.6% | **Best: consistency 13.3%→9.6%** |
| EXP_013 | Confidence filter sweep (45 combos) | 27.3% ± — | 3.2% | — | Best config: top_pct=0.20, wc=50, mc=0.35 |
| EXP_014 | Final backtest on 2026 holdout | 16.3% win rate | — | — | Rs.-123,005, costs 1712% of gross |

### EXP_014 vs predecessors (2026 holdout)
| Metric | EXP_001 (baseline) | EXP_008b (Phase 6) | EXP_014 (Phase 6.5) |
|--------|-------------------|-------------------|---------------------|
| Trades | 2,239 | 1,906 | 1,730 |
| Win rate | 16.5% | 18.5% | 16.3% |
| Net P&L | Rs.-170,125 | Rs.-133,145 | Rs.-123,005 |
| Expectancy | Rs.-75.98 | Rs.-69.86 | Rs.-71.10 |
| Costs/gross | 2031% | 1208% | 1712% |

### Root cause (unchanged from Phase 6)
NSE intraday cost structure is a structural barrier at Rs.50,000 capital. Break-even win rate ~37%; achieved ~16-18%. Phase 6.5 exhausted feature engineering and signal filtering levers. The model is technically sound; economics are unviable at this capital level.

### Critical bug fixed in Phase 6.5
`datetime64[ms] + pd.Timedelta(minutes=5)` upcasts to `datetime64[us]` in pandas 3.x, causing `merge_asof` "incompatible merge keys" error for 11/15 symbols during EXP_012 feature rebuild. Fix: save and restore the original dtype; add fallback dtype normalization between left/right DataFrames. Regression test added: `test_tz_aware_datetimes_compatible()`.

## Phase 6 Key Findings

### Model comparison (walk-forward, 3 folds, L1, 3min, 55 features)
| Model | combined_precision | signal_rate | consistency (std/mean) | viable |
|-------|-------------------|-------------|------------------------|--------|
| Logistic Regression | 34.6% ± 2.4% | 0.1% | 6.8% | No (no signals) |
| Random Forest | 34.7% ± 4.4% | 0.0% | 12.7% | No (zero signals) |
| XGBoost (t=0.50) | 27.3% ± 3.6% | 4.1% | 13.3% | Yes (selected) |

### Test holdout results (2026-01-01 → 2026-08-19, 15 symbols)
- **EXP_008a** (avoid_afternoon=True): 1,894 trades, 18.5% win rate, Rs.-133,932 net P&L, Rs.-70.71/trade
- **EXP_008b** (avoid_afternoon=False): 1,906 trades, 18.5% win rate, Rs.-133,145 net P&L, Rs.-69.86/trade
- **Selected**: EXP_008b (marginally better; afternoon filter not beneficial under ML)
- **vs EXP_001 baseline**: +Rs.36,980 net P&L improvement (+Rs.6.12/trade expectancy)

### Why ML could not reach 40% precision target
- Dominant features: session_minute (24%), is_closing_30min (22%) — time-of-day, not price/momentum
- Model learned "when NOT to trade" rather than directional signals
- Cost structure unchanged: costs = 1,208% of gross profit even with ML filtering
- Baseline strategy (EMA/RSI/VWAP) has poor directional power — ML filter helps, cannot cure

### Critical bugs fixed during Phase 6 implementation
- sklearn 1.9.0: removed `multi_class` parameter from LogisticRegression
- pandas 3+: `StringDtype` not `object` — fixed feature column filter with `is_numeric_dtype()`
- XGBoost early stopping: mlogloss plateau ≠ precision plateau (fold 1/2 showed 0% precision); removed `early_stopping_rounds` entirely
- `train_final` crash: retained `early_stopping_rounds` attribute without eval_set; fixed with explicit None reset

## Phase 5 Key Findings (bring to review conversation)

### Fold structure
- 1,476,595 total rows (15 symbols, 3min, L1, session-filtered)
- Fold 1: 603,870 train / 206,250 val | Fold 2: 810,120 / 202,950 | Fold 3: 1,013,070 / 206,565
- Test holdout: 256,960 rows (2026-01-01 → 2026-08-19) — completely isolated, never touched

### Baseline per-fold performance (EXP_001 rules through fold evaluator)
- Fold 1: combined_precision 18.1%, signal_rate 6.5%
- Fold 2: combined_precision 16.7%, signal_rate 6.6%
- Fold 3: combined_precision 12.1%, signal_rate 6.4%
- Aggregate: 15.6% ± 2.5% combined_precision | consistency 16.3% (ABOVE 10% threshold)

### Critical finding: fold consistency
- Fold 3 (2025-H2) shows significantly worse baseline precision (12.1% vs ~18% in earlier folds)
- L1 signal density also dropped in 2025-H2: only 13.2% per direction vs 18-19% in earlier folds
- This means the market was harder to trade in that period — rules degraded
- Phase 6 must produce fold std/mean < 10% to be considered stable

### Phase 6 targets
- combined_precision > 40% (Phase 6 primary hurdle; baseline is 15.6%)
- signal_rate ~ 5-10% (matching baseline frequency)
- Consistency std/mean < 10% across all 3 folds (baseline fails at 16.3%)

## Phase 4 Key Findings (bring to review conversation)

### Label distributions (3min primary timeframe)
- L1 recommended: 17.1% long / 18.0% short / 64.9% none — best balance for ML training
- 3min horizon=20 equals 60 min of real time — sufficient for 0.4% moves, explains higher density vs spec estimate
- 1min L1 matches spec expectation: ~8% per direction, 84% none

### Feature correlation (see notebook 02)
- Top predictors: vwap_distance, vwap_above, ema9_distance, ema21_distance, rsi_14, volume_ratio, trend_regime_enc
- Redundant candidates: raw EMA levels (ema_9, ema_21, ema_50), raw BB levels (bb_upper, bb_lower) — already captured by their distance variants
- Recommended drop before Phase 6 training: ~6 raw price-level features

### EXP_001 failure diagnosis (see notebook 03)
- Costs = 2031% of gross — trade frequency is the core problem
- Win rate 16.5% vs 33.3% break-even — EMA/RSI/VWAP rules have poor directional power
- ML must reduce entries from ~150/symbol/year to ~20-40 high-conviction signals
