# EXPERIMENT LOG

## Format

Each experiment records:
- ID, Date, Hypothesis
- Dataset, Symbols, Timeframe, Date Range
- Features, Target, Model, Hyperparameters
- Training / Validation / Test periods
- Results: Sharpe, Profit Factor, Expectancy, Max Drawdown, Win Rate, Trades, Net Return
- Costs and Slippage assumptions
- Conclusion

---

## Experiments

---

## EXP_001 — Baseline Rule-Based Strategy (Model Zero)

**Date:** 2026-08-20
**Hypothesis:** A simple EMA crossover + RSI + VWAP filter rule set can generate
positive expectancy after realistic costs on 3-minute NSE large-cap candles.

**Dataset:**
- Symbols: 15 NSE large-cap stocks (NIFTY 100)
- Timeframe: 3-minute candles
- Date range: 2023-01-02 to 2026-08-19
- Features: 64 features from Phase 2

**Strategy:** Baseline (EMA9/21/50 crossover, RSI 40-65, VWAP filter, volume ratio >= 1.2)

**Capital config:**
- Starting capital: Rs.50,000 (paper)
- Leverage: 5x
- Sizing: fixed_pct 20%
- Target: +0.40% | Stop: -0.20% | Time exit: 15:15

**Results:**
```
Total trades:          2,239
Win rate:              16.5%
Net P&L:               Rs.-170,125
Net return:            -340.3%
Max drawdown:          -99.7%
Sharpe ratio:          -19.34
Profit factor:         0.07
Expectancy per trade:  Rs.-75.98
Calmar ratio:          -11.62
```

**Exit breakdown:**
```
Target hits:   682  (30.5%)
Stop losses:   1,279  (57.1%)
Time exits:    278  (12.4%)
```

**Direction breakdown:**
```
Long:   1,030 trades | win rate 13.4%
Short:  1,209 trades | win rate 19.1%
```

**Time-of-day breakdown:**
```
Morning   (09:15-10:59):  400 trades | win rate 19.5%
Midday    (11:00-13:29): 1,226 trades | win rate 19.3%
Afternoon (13:30-15:29):  613 trades | win rate 8.8%
```

**Cost impact:**
```
Gross P&L:          Rs.8,811
Total costs paid:   Rs.178,936
Costs as % of gross: 2031%
```

**Conclusion:**
The strategy does not beat zero. Win rate of 16.5% is far below the ~55% needed to
break even given the 2:1 loss-to-win ratio (stop=0.20%, target=0.40%). Gross P&L
is barely positive (Rs.8,811), meaning the entry conditions do generate some edge on
price direction, but transaction costs are catastrophic — 2031% of gross P&L. At this
capital level (Rs.50,000), each trade costs ~Rs.80-100 in brokerage + slippage while
targeting only Rs.150-200 in gross profit, leaving a razor-thin margin that the
16.5% win rate cannot sustain.

Afternoon session shows the worst win rate (8.8%) vs morning/midday (~19%). Short
trades marginally outperform longs (19.1% vs 13.4%).

**What this tells us:**
1. The entry signal logic (EMA + RSI + VWAP) has very poor directional predictive
   power at this 3-minute timeframe — 16.5% win rate is near-random at best.
2. Costs completely dominate at Rs.50,000 capital with 5x leverage and these trade
   sizes. ML must either (a) improve win rate significantly (to ~55%+), or (b) select
   only high-conviction entries to reduce trade frequency and improve gross-per-trade.
3. The 2:1 reward-to-risk ratio (0.40% target vs 0.20% stop) is insufficient to
   offset the 57% stop rate. Either targets need widening or stops need tightening,
   OR entries need to be far more selective.
4. This baseline must be beaten by Phase 6 ML on both expectancy per trade AND
   total net P&L after costs. The bar is Rs.-75.98 expectancy — any ML model
   producing positive expectancy would already be an improvement.

---

## EXP_002 -- Label Distribution Analysis

**Date:** 2026-08-20
**Hypothesis:** Three-class labels (L1-L4 variants) will show meaningful but minority
positive/negative classes, confirming the market is mostly non-tradeable (label=0) at
any given candle.

**Dataset:**
- Symbols: 15 NSE large-cap stocks
- Timeframe: 3-minute candles (primary), 1-min and 5-min also generated
- Date range: 2023-01-02 to 2026-08-19
- Total rows labeled: 1,678,057 (3min) | 5,035,960 (1min) | 1,006,473 (5min)

**Label distribution (3min, avg across 15 symbols):**
```
Variant  Long%   Short%   None%   Description
L1       17.1%   18.0%    64.9%   Standard 2:1 RR -- target=0.40%, stop=0.20%, horizon=20
L2       11.7%   12.0%    76.2%   Wider 2:1 RR   -- target=0.60%, stop=0.30%, horizon=30
L3       25.6%   26.8%    47.6%   Symmetric 1:1  -- target=0.30%, stop=0.30%, horizon=15
L4       14.1%   14.7%    71.2%   Intermediate   -- target=0.50%, stop=0.25%, horizon=25
```

**Label distribution (1min, all symbols combined):**
```
L1:  +1=8.2%   -1=8.1%   0=83.7%
L2:  +1=4.9%   -1=4.6%   0=90.6%
L3:  +1=12.3%  -1=12.5%  0=75.2%
L4:  +1=6.2%   -1=6.0%   0=87.8%
```

**Validation:** No flags raised across all 180 label files. Leakage spot-checks passed.
Cross-variant consistency confirmed (L2 distributions differ from L1 as expected).

**Key observations:**

1. 3min has higher signal density than expected (~17-18% per direction for L1 vs ~8% on 1min).
   horizon=20 on 3min = 60 minutes of forward time; enough for a 0.4% move in most NSE large-caps.
   The 1min data (horizon=20 = 20 minutes) shows the expected ~8% figure.

2. L3 (symmetric 1:1 RR) has the highest density but requires >50% win rate to profit.
   On 3min, none_pct=47.6% -- nearly half of candles have a tradeable label, suggesting many
   candles can move +/-0.3% within 45 minutes. Labels are real but the edge is harder to capture.

3. Long/Short balance is nearly symmetric across all variants and symbols (within ~1-2%).
   No structural directional bias. Labels are symmetric by construction.

4. Regime: Tradeable labels are more concentrated in bull/bear trend regimes than neutral.
   High-volatility regime generates more labels but also more conflicted ones.

5. Time of day: Signal density peaks in 10:00-12:00 and 13:30-14:30. Opening 30 min and
   closing 15 min (excluded by dataset builder) contribute negligible tradeable candles.

**Conclusion:**

Variant L1 recommended for Phase 6 primary training:
- Well-understood parameters (matches backtester from EXP_001)
- ~17% per direction gives ample minority-class samples (~285,000 long rows on 3min)
- 2:1 RR means model can be profitable at ~40% win rate
- 64.9% none_pct keeps class imbalance manageable

L4 is the secondary candidate (14% per direction, same 2:1 RR, larger targets reduce cost impact).
L2 and L3 deprioritised: L2 has too few signals; L3 requires unrealistically high win rate.

---

## EXP_003 -- Walk-Forward Fold Baseline Evaluation

**Date:** 2026-08-20
**Purpose:** Establish the walk-forward fold structure and run EXP_001 baseline rules
through the fold evaluator to generate per-fold benchmarks for Phase 6 ML.

**Dataset:**
- Symbols: 15 NSE large-cap stocks
- Timeframe: 3min (primary), Variant: L1
- Total rows loaded: 1,476,595 (after session filter: open +30min, close -15min)

**Fold structure (expanding window):**
```
Fold  Train Period              Train Rows   Val Period                Val Rows   Long%   Short%   None%
1     2023-01-02 -> 2024-06-30     603,870   2024-07-01 -> 2024-12-31   206,250   18.1%   19.3%   62.6%
2     2023-01-02 -> 2024-12-31     810,120   2025-01-01 -> 2025-06-30   202,950   19.0%   19.1%   61.9%
3     2023-01-02 -> 2025-06-30   1,013,070   2025-07-01 -> 2025-12-31   206,565   13.2%   13.2%   73.6%
TEST  [HOLDOUT]                         --   2026-01-01 -> 2026-08-19   256,960   19.1%   20.9%   60.0%
```

**Leakage checks:** CLEAN on all 3 folds and test isolation.

**Baseline (EXP_001 rules) per-fold evaluation on validation sets:**
```
                    Fold 1    Fold 2    Fold 3    Mean +/- Std
combined_precision  18.1%     16.7%     12.1%     15.6% +/- 2.5%
long_precision      17.7%     16.5%     13.1%     15.8% +/- 2.0%
short_precision     18.4%     16.9%     11.2%     15.5% +/- 3.1%
signal_rate          6.5%      6.6%      6.4%      6.5% +/- 0.1%
macro_f1             0.284     0.279     0.306     0.290 +/- 0.012
```

**Key observations:**

1. Baseline combined_precision (15.6%) is *lower* than EXP_001 aggregate win rate (16.5%).
   This is expected: EXP_001 measured trade win rate (which includes only filled orders),
   while combined_precision here measures raw signal accuracy on every validation candle.
   These are consistent but not identical metrics.

2. Fold 3 combined_precision (12.1%) is materially worse than Folds 1-2 (18.1%, 16.7%).
   Fold 3 validation covers 2025-H2, which shows lower L1 signal density (13.2% each direction
   vs 18-19% in earlier folds). The market regime in 2025-H2 generated fewer clean directional
   moves, and the EMA/RSI/VWAP rules deteriorated sharply.
   This is a warning: std/mean consistency = 16.3%, above the 10% acceptable threshold.
   Phase 6 models must be evaluated for stability across folds, not just mean performance.

3. Signal rate is stable across folds (6.5% +/- 0.1%). The rule-based filter is consistently
   selective. An ML model at 5-8% signal rate would be comparable in trade frequency.

4. The 2026 test holdout shows higher signal density (19.1% long, 20.9% short) than Fold 3
   validation, suggesting the market returned to a more trending regime in 2026.
   This is noted but cannot influence model selection — it is holdout data.

**What Phase 6 must beat (per fold):**
- combined_precision > 40% (vs baseline 15.6%) — this is the primary hurdle
- signal_rate ~ 5-10% — comparable frequency to baseline (lower is acceptable if precision rises)
- Consistency: std/mean < 10% across all 3 folds (baseline fails this at 16.3%)
- A model that achieves 40% combined_precision at 8% signal_rate will have ~7x better
  win rate than baseline with equivalent trade frequency

---

## EXP_004 — Logistic Regression
**Date:** 2026-08-21
**Model:** Logistic Regression
**Variant:** L1 | **Timeframe:** 3min

**Per-fold results:**
| Fold | combined_precision | signal_rate | train_rows |
|------|-------------------|-------------|------------|
| 1 | 32.0% | 0.1% | 603,870 |
| 2 | 37.7% | 0.2% | 810,120 |
| 3 | 34.0% | 0.0% | 1,013,070 |

**Aggregate:**
- combined_precision: 34.6% ± 2.4%
- signal_rate: 0.1% ± 0.1%
- consistency (std/mean): 6.8%

## EXP_005 — Random Forest
**Date:** 2026-08-21
**Model:** Random Forest
**Variant:** L1 | **Timeframe:** 3min

**Per-fold results:**
| Fold | combined_precision | signal_rate | train_rows |
|------|-------------------|-------------|------------|
| 1 | 40.8% | 0.0% | 603,870 |
| 2 | 30.6% | 0.0% | 810,120 |
| 3 | 32.8% | 0.0% | 1,013,070 |

**Aggregate:**
- combined_precision: 34.7% ± 4.4%
- signal_rate: 0.0% ± 0.0%
- consistency (std/mean): 12.7%
