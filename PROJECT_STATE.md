# PROJECT STATE

## Current Phase
Phase 6: ML Model Training Pipeline — Implementation Complete (models training pending)

## Status
Phase 6 code implemented. Ready to run training: `python scripts/train_models.py`

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
- This is the floor Phase 6 ML must beat

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

### Phase 6 — ML Model Training Pipeline
- [x] models/trainer.py — ModelTrainer with fold/final training, XGBoost label encoding, StandardScaler on train only
- [x] models/predictor.py — ModelPredictor with threshold-based signal selectivity, proba output [short/none/long]
- [x] models/selector.py — ModelSelector with composite score (precision - instability penalty - signal_rate penalty)
- [x] models/artifacts/ — artifact directory created
- [x] strategy/ml_strategy.py — MLStrategy wrapping predictor, afternoon filter, BaseStrategy compatible
- [x] scripts/train_models.py — EXP_004→007: LR, RF, XGBoost, threshold tuning; saves best model metadata
- [x] scripts/run_ml_backtest.py — EXP_008: loads test holdout ONLY here, vs EXP_001 comparison table
- [x] tests/test_trainer.py, test_predictor.py, test_selector.py, test_ml_strategy.py — 28 pass, 4 skip (xgb)
- [x] 231 total tests passing (203 Phase 1-5 + 28 Phase 6), 0 failures

## Next Task
Run model training: `python scripts/train_models.py`
Then backtest: `python scripts/run_ml_backtest.py`

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
