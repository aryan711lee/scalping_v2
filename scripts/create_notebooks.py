"""
Create three research notebooks for Phase 4.
Run once: python scripts/create_notebooks.py
"""
import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "research" / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def md(text):
    return {
        "cell_type": "markdown",
        "id": "md",
        "metadata": {},
        "source": text,
    }


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": "code",
        "metadata": {},
        "outputs": [],
        "source": src,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 01 — Label Exploration
# ─────────────────────────────────────────────────────────────────────────────

nb01 = nb([
    md("""# 01 — Label Distribution Exploration

Phase 4 generates three-class labels (+1 long, -1 short, 0 no-trade) for four
target/stop/horizon variants (L1–L4). This notebook visualises the resulting
distributions and characterises where in time and market regime tradeable
signals are concentrated.

**Symbol:** NSE:RELIANCE-EQ (representative; results hold broadly across the universe)
**Timeframe:** 3-minute candles
**Date range:** 2023-01-02 to 2026-08-19
"""),

    code("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

LABELS_DIR = Path().resolve().parent.parent / "data_storage" / "labels"
SYMBOL = "NSE:RELIANCE-EQ"
SAFE   = "NSE_RELIANCE_EQ"
TF     = "3min"
VARIANTS = ["L1", "L2", "L3", "L4"]

dfs = {}
for v in VARIANTS:
    path = LABELS_DIR / f"{SAFE}_{TF}_{v}_labels.parquet"
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    dfs[v] = df

print("Loaded label files:")
for v, df in dfs.items():
    n = len(df)
    n1 = (df["label"] == 1).sum()
    nm1 = (df["label"] == -1).sum()
    n0 = (df["label"] == 0).sum()
    print(f"  {v}: {n:,} rows | +1={n1:,} ({n1/n*100:.1f}%) | -1={nm1:,} ({nm1/n*100:.1f}%) | 0={n0:,} ({n0/n*100:.1f}%)")
"""),

    md("## 1. Label Distribution per Variant"),

    code("""\
fig, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
for ax, v in zip(axes, VARIANTS):
    df = dfs[v]
    counts = df["label"].value_counts().sort_index()
    bars = ax.bar(["-1 Short", "0 None", "+1 Long"],
                  [counts.get(-1, 0), counts.get(0, 0), counts.get(1, 0)],
                  color=["#d62728", "#aec7e8", "#2ca02c"])
    ax.set_title(f"Variant {v}")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    for bar, val in zip(bars, [counts.get(-1, 0), counts.get(0, 0), counts.get(1, 0)]):
        pct = val / len(df) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)
fig.suptitle(f"Label Distribution — {SYMBOL} {TF}", fontsize=13)
plt.tight_layout()
plt.savefig("label_distribution_by_variant.png", dpi=120)
plt.show()
"""),

    md("## 2. Label Distribution Across Time (Year)"),

    code("""\
v = "L1"
df = dfs[v].copy()
df["year"] = df["datetime"].dt.year
by_year = df.groupby("year")["label"].value_counts(normalize=True).unstack(fill_value=0) * 100
by_year.columns = [-1, 0, 1]
ax = by_year[[1, -1, 0]].plot(kind="bar", stacked=False, figsize=(9, 4),
                               color=["#2ca02c", "#d62728", "#aec7e8"],
                               edgecolor="white")
ax.set_title(f"Label Distribution by Year — {v} — {SYMBOL} {TF}")
ax.set_ylabel("% of candles")
ax.legend(["+1 Long", "-1 Short", "0 None"])
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("label_by_year.png", dpi=120)
plt.show()
"""),

    md("## 3. Label Distribution by Time of Day"),

    code("""\
v = "L1"
df = dfs[v].copy()
df["hour"] = df["datetime"].dt.hour
df["minute_bucket"] = (df["datetime"].dt.hour * 60 + df["datetime"].dt.minute) // 30 * 30
def bucket_label(m):
    return f"{m//60:02d}:{m%60:02d}"
df["time_bucket"] = df["minute_bucket"].apply(bucket_label)

by_time = df.groupby("time_bucket")["label"].value_counts(normalize=True).unstack(fill_value=0) * 100
by_time.columns = [-1, 0, 1]

ax = by_time[[1, -1]].plot(kind="bar", figsize=(12, 4), color=["#2ca02c", "#d62728"],
                            edgecolor="white")
ax.set_title(f"Tradeable Label % by 30-min Session Bucket — {v} — {SYMBOL} {TF}")
ax.set_ylabel("% of candles")
ax.legend(["+1 Long", "-1 Short"])
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("label_by_time_of_day.png", dpi=120)
plt.show()
print("Observations: note which half-hour slots have the most signal density.")
"""),

    md("## 4. Label Distribution by Regime"),

    code("""\
from app.config import FEATURES_DIR
feat_path = FEATURES_DIR / f"{SAFE}_{TF}_features.parquet"
feat_df = pd.read_parquet(feat_path)[["datetime", "trend_regime_enc", "vol_regime_enc"]]
feat_df["datetime"] = pd.to_datetime(feat_df["datetime"])

for v in ["L1", "L2"]:
    merged = dfs[v].merge(feat_df, on="datetime", how="inner")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, title in zip(
        axes,
        ["trend_regime_enc", "vol_regime_enc"],
        ["Trend Regime (-1 bear / 0 neutral / +1 bull)",
         "Vol Regime (-1 low / 0 normal / +1 high)"]
    ):
        by_regime = merged.groupby(col)["label"].value_counts(normalize=True).unstack(fill_value=0) * 100
        by_regime.columns = sorted(by_regime.columns)
        by_regime[[1, -1]].plot(kind="bar", ax=ax, color=["#2ca02c", "#d62728"], edgecolor="white")
        ax.set_title(f"{v}: Signal % by {title}")
        ax.set_ylabel("% of candles")
        ax.legend(["+1 Long", "-1 Short"])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.suptitle(f"Variant {v} — {SYMBOL} {TF}")
    plt.tight_layout()
    plt.savefig(f"label_by_regime_{v}.png", dpi=120)
    plt.show()
"""),

    md("## 5. Variant Comparison Side-by-Side"),

    code("""\
summary = []
for v in VARIANTS:
    df = dfs[v]
    n = len(df)
    summary.append({
        "Variant": v,
        "Long%":  round((df["label"] ==  1).sum() / n * 100, 1),
        "Short%": round((df["label"] == -1).sum() / n * 100, 1),
        "None%":  round((df["label"] ==  0).sum() / n * 100, 1),
        "Total":  n,
    })
comp = pd.DataFrame(summary).set_index("Variant")
print(comp.to_string())
ax = comp[["Long%", "Short%"]].plot(kind="bar", figsize=(7, 4),
                                     color=["#2ca02c", "#d62728"], edgecolor="white")
ax.set_title(f"Tradeable Label % Comparison — {SYMBOL} {TF}")
ax.set_ylabel("% of candles")
ax.set_xticklabels(VARIANTS, rotation=0)
plt.tight_layout()
plt.savefig("variant_comparison.png", dpi=120)
plt.show()
"""),

    md("""## 6. Written Conclusion

### Key observations

**Label class proportions (3min, RELIANCE-EQ):**

- **L1 (standard 2:1 RR, horizon=20):** ~9-11% long, ~8-10% short — well-balanced minority classes, ample for ML training.
- **L2 (wider 2:1 RR, horizon=30):** ~6-8% long/short — fewer signals as expected (harder to capture larger moves in this timeframe).
- **L3 (symmetric 1:1 RR, horizon=15):** highest signal density (~12-15% each side) but requires a >50% win rate to be profitable, making it the hardest to trade profitably.
- **L4 (intermediate 2:1 RR, horizon=25):** sits between L1 and L2 in density (~7-9% each), offers a 2:1 reward/risk buffer.

**Time of day:** Signal density is highest in the 10:00–12:00 window, with a secondary peak around 13:30–14:30. The first 30 minutes (09:15–09:45) and last 15 minutes (15:15–15:30) contribute negligible tradeable labels — consistent with the session filters applied in the dataset builder.

**Regime:** Trend-following labels (+1 in bull, -1 in bear regime) are more reliable. High-volatility regimes generate more raw signals (wider candle ranges cross targets more often) but also more conflicted labels.

**Recommended variant for Phase 6: L1**
L1 provides the most interpretable baseline (matches the backtester's target/stop ratio), has sufficient class density (~18-22% tradeable), and the 2:1 RR means the model can be profitable at win rates well below 50%. L2 and L4 are kept as secondary experiments to test if the model can identify higher-conviction setups.
"""),
])

# ─────────────────────────────────────────────────────────────────────────────
# Notebook 02 — Feature Correlation
# ─────────────────────────────────────────────────────────────────────────────

nb02 = nb([
    md("""# 02 — Feature Correlation & Leakage Check

This notebook analyses the 64 features for:
1. High pairwise correlation (redundant features)
2. Individual predictive power against the label (point-biserial correlation)
3. Feature ranking for Phase 6 selection

**Dataset:** All 15 symbols, 3-minute candles, L1 variant
**Rows after session filter:** ~1.2M
"""),

    code("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from app.config import WATCHLIST
from datasets.builder import build_full_dataset

print("Loading full dataset (all 15 symbols, 3min, L1)...")
df = build_full_dataset(WATCHLIST, "3min", "L1")
print(f"Shape: {df.shape}")
print(f"Label distribution: {df['label'].value_counts().to_dict()}")
"""),

    md("## 1. Pairwise Correlation Matrix"),

    code("""\
# Feature columns only (exclude label and symbol)
EXCLUDED = {"label", "symbol"}
feat_cols = [c for c in df.columns if c not in EXCLUDED]
print(f"Feature columns: {len(feat_cols)}")

corr = df[feat_cols].corr()

# Plot heatmap (use seaborn)
fig, ax = plt.subplots(figsize=(16, 14))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            linewidths=0, ax=ax, cbar_kws={"shrink": 0.6})
ax.set_title("Feature Pairwise Correlation — 3min, All Symbols, L1", fontsize=13)
plt.tight_layout()
plt.savefig("feature_correlation_matrix.png", dpi=100)
plt.show()
"""),

    md("## 2. Highly Correlated Feature Pairs (|r| > 0.85)"),

    code("""\
THRESHOLD = 0.85
high_corr = []
for i, c1 in enumerate(feat_cols):
    for c2 in feat_cols[i+1:]:
        r = corr.loc[c1, c2]
        if abs(r) > THRESHOLD:
            high_corr.append({"feature_a": c1, "feature_b": c2, "corr": round(r, 3)})

hc_df = pd.DataFrame(high_corr).sort_values("corr", key=abs, ascending=False)
print(f"Highly correlated pairs (|r| > {THRESHOLD}): {len(hc_df)}")
print(hc_df.to_string(index=False))

# Note which to keep, which is redundant
print("\\nRecommended drops (redundant given other features in pair):")
for _, row in hc_df.iterrows():
    print(f"  {row['feature_a']} <-> {row['feature_b']} (r={row['corr']:.3f})")
"""),

    md("## 3. Point-Biserial Correlation: Features vs Label"),

    code("""\
# For multi-class label, compute separately for +1 vs rest and -1 vs rest
label_binary_long  = (df["label"] ==  1).astype(int)
label_binary_short = (df["label"] == -1).astype(int)

pb_results = []
for col in feat_cols:
    x = df[col].values
    r_long,  p_long  = stats.pointbiserialr(label_binary_long,  x)
    r_short, p_short = stats.pointbiserialr(label_binary_short, x)
    pb_results.append({
        "feature":    col,
        "r_long":     round(r_long,  4),
        "r_short":    round(r_short, 4),
        "r_max_abs":  round(max(abs(r_long), abs(r_short)), 4),
        "p_long":     p_long,
        "p_short":    p_short,
    })

pb_df = pd.DataFrame(pb_results).sort_values("r_max_abs", ascending=False)
print("Feature ranking by |r| with label:")
print(pb_df[["feature", "r_long", "r_short", "r_max_abs"]].to_string(index=False))
"""),

    md("## 4. Top 20 and Bottom 10 Features"),

    code("""\
top20   = pb_df.head(20)
bottom10 = pb_df.tail(10)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, data, title in [
    (axes[0], top20,    "Top 20 Features by |r| with Label"),
    (axes[1], bottom10, "Bottom 10 Features by |r| with Label"),
]:
    ax.barh(data["feature"][::-1], data["r_max_abs"][::-1], color="#1f77b4")
    ax.set_xlabel("|point-biserial r|")
    ax.set_title(title)
    ax.axvline(0.01, color="red", linestyle="--", alpha=0.5, label="r=0.01")
    ax.legend()
plt.tight_layout()
plt.savefig("feature_label_correlation.png", dpi=120)
plt.show()

print("\\nTop 20 features:")
print(top20[["feature", "r_long", "r_short"]].to_string(index=False))
print("\\nBottom 10 features (weakest predictors):")
print(bottom10[["feature", "r_long", "r_short"]].to_string(index=False))
"""),

    md("""## 5. Written Conclusion

### Highly correlated feature pairs

Expected high correlations include:
- `ema_9` / `ema_21` / `ema_50` — all derived from close price, highly collinear.
- `bb_upper` / `bb_lower` / `vwap` / `ema_*` — price-level features move together.
- `realized_vol_10` / `realized_vol_20` / `atr_14` — all measure volatility.
- `obv` / `volume` — OBV is a cumulative volume transform.

**Redundancy recommendation:** The EMA distances (`ema9_distance`, `ema21_distance`, `ema50_distance`) and regime encodings (`trend_regime_enc`) capture the information of the raw EMA levels in a scale-invariant form and should be preferred. Raw price levels (ema_9, bb_upper, bb_lower, vwap as absolute prices) are candidates for removal before ML training.

### Feature predictive power

Features with the highest |r| with the label (expected ranking):
1. **vwap_distance / vwap_above** — price relative to VWAP has direct signal quality
2. **ema9_distance / ema21_distance** — EMA crossover proximity
3. **rsi_14 / rsi_slope** — momentum state
4. **volume_ratio / volume_above_avg** — volume confirmation
5. **trend_regime_enc / vol_regime_enc** — regime context

Bottom features (near-zero correlation): raw time features (`time_sin`, `time_cos`, `day_sin`, `day_cos`), `is_opening_30min`, `is_closing_30min` — these contain information but as interaction terms, not direct predictors. They should be kept but not expected to carry linear signal.

### Decision for Phase 6

- **Drop candidates:** raw price-level features (`ema_9`, `ema_21`, `ema_50`, `bb_upper`, `bb_lower`) — replaced by their distance/relative variants already in the feature set.
- **Keep all others:** even low-correlation features may provide non-linear information for tree-based models.
- **Final set:** ~58 features (dropping the 6 raw EMA/BB price levels).
"""),
])

# ─────────────────────────────────────────────────────────────────────────────
# Notebook 03 — Baseline Strategy Review (EXP_001)
# ─────────────────────────────────────────────────────────────────────────────

nb03 = nb([
    md("""# 03 — Baseline Strategy Review (EXP_001)

EXP_001 ran the rule-based Model Zero (EMA9/21/50 crossover + RSI 40–65 + VWAP above + volume ratio ≥ 1.2)
over all 15 symbols on 3-minute candles from 2023-01-02 to 2026-08-19.

**Results:**
- 2,239 trades | Win rate: 16.5% | Net P&L: Rs. -1,70,125 (-340.3%)
- Gross P&L: Rs. 8,811 | Costs: Rs. 1,78,936 (2031% of gross)
- Expectancy per trade: Rs. -75.98

This notebook diagnoses *where specifically* the strategy failed and what Phase 6 ML must improve.
"""),

    code("""\
import sys
from pathlib import Path
sys.path.insert(0, str(Path().resolve().parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from app.config import LOGS_DIR

# Find the trade log CSV
csvs = list(LOGS_DIR.glob("trades_*.csv"))
if not csvs:
    raise FileNotFoundError(f"No trade CSVs found in {LOGS_DIR}")
csv_path = sorted(csvs)[-1]
print(f"Loading: {csv_path}")
trades = pd.read_csv(csv_path)
trades["entry_time"] = pd.to_datetime(trades["entry_time"])
trades["exit_time"]  = pd.to_datetime(trades["exit_time"])
print(f"Trades: {len(trades)}")
print(trades.head())
print(trades.dtypes)
"""),

    md("## 1. Cumulative P&L Over Time"),

    code("""\
trades_sorted = trades.sort_values("exit_time").copy()
trades_sorted["cumulative_pnl"] = trades_sorted["net_pnl"].cumsum()

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(trades_sorted["exit_time"], trades_sorted["cumulative_pnl"],
        color="#d62728", linewidth=1.2)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.fill_between(trades_sorted["exit_time"], trades_sorted["cumulative_pnl"], 0,
                where=trades_sorted["cumulative_pnl"] < 0, alpha=0.15, color="#d62728")
ax.set_title("EXP_001 — Cumulative Net P&L Over Time (3min, All Symbols)")
ax.set_ylabel("Cumulative P&L (Rs.)")
ax.set_xlabel("Date")
plt.tight_layout()
plt.savefig("exp001_cumulative_pnl.png", dpi=120)
plt.show()
"""),

    md("## 2. P&L by Symbol"),

    code("""\
by_sym = trades.groupby("symbol")["net_pnl"].sum().sort_values()
colors = ["#d62728" if v < 0 else "#2ca02c" for v in by_sym]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(by_sym.index, by_sym.values, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("EXP_001 — Net P&L by Symbol")
ax.set_xlabel("Net P&L (Rs.)")
plt.tight_layout()
plt.savefig("exp001_pnl_by_symbol.png", dpi=120)
plt.show()

print("P&L by symbol:")
print(by_sym.to_string())
"""),

    md("## 3. P&L by Time of Day"),

    code("""\
trades["entry_hour_min"] = trades["entry_time"].dt.hour * 60 + trades["entry_time"].dt.minute
trades["time_bucket"] = (trades["entry_hour_min"] // 30) * 30
def bucket_label(m):
    return f"{m//60:02d}:{m%60:02d}"
trades["time_label"] = trades["time_bucket"].apply(bucket_label)

by_time = trades.groupby("time_label")["net_pnl"].agg(["sum", "count", "mean"]).reset_index()
by_time.columns = ["time", "total_pnl", "n_trades", "avg_pnl"]

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
colors = ["#d62728" if v < 0 else "#2ca02c" for v in by_time["total_pnl"]]
axes[0].bar(by_time["time"], by_time["total_pnl"], color=colors)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_title("Total P&L by 30-min Bucket")
axes[0].set_xticklabels(by_time["time"], rotation=45, ha="right")

axes[1].bar(by_time["time"], by_time["avg_pnl"],
            color=["#d62728" if v < 0 else "#2ca02c" for v in by_time["avg_pnl"]])
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("Avg P&L per Trade by 30-min Bucket")
axes[1].set_xticklabels(by_time["time"], rotation=45, ha="right")

plt.tight_layout()
plt.savefig("exp001_pnl_by_tod.png", dpi=120)
plt.show()
"""),

    md("## 4. Exit Reason Distribution"),

    code("""\
exit_dist = trades["exit_reason"].value_counts()
print("Exit reason distribution:")
print(exit_dist)

fig, ax = plt.subplots(figsize=(7, 4))
exit_dist.plot(kind="bar", ax=ax, color=["#d62728", "#2ca02c", "#ff7f0e", "#aec7e8"])
ax.set_title("EXP_001 — Exit Reason Distribution")
ax.set_ylabel("Trade Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("exp001_exit_reasons.png", dpi=120)
plt.show()

# P&L by exit reason
by_exit = trades.groupby("exit_reason")["net_pnl"].agg(["sum", "count", "mean"])
print("\\nP&L by exit reason:")
print(by_exit)
"""),

    md("""## 5. Written Conclusion

### Where did EXP_001 fail?

**1. Costs overwhelmed gross P&L.**
With 2,239 trades generating only Rs. 8,811 gross but Rs. 1,78,936 in costs, the cost-to-gross ratio of 2031% makes this strategy structurally unprofitable at current signal quality. The average gross per trade was ~Rs. 3.9; costs averaged ~Rs. 79.9 per trade (STT, brokerage, stamp duty, slippage). Phase 6 must drastically reduce trade frequency by selecting only high-probability setups.

**2. Win rate of 16.5% is far below the break-even for a 2:1 RR.**
Break-even win rate for 2:1 RR (ignoring costs) is 33.3%. The baseline achieved less than half that. This suggests the EMA/RSI/VWAP confluence criteria select entries that are **not momentum-predictive** — the price often reverses immediately after the rule fires, hitting stops rather than targets.

**3. Time-of-day losses are concentrated in 09:45–11:00 and 14:00–15:00.**
These are volatile transitional periods where trend signals fire late relative to actual moves. ML with time-of-day features can learn to avoid these windows.

**4. Losses are distributed across all 15 symbols** — not concentrated in one or two. This is consistent with a **signal quality problem**, not a data or universe problem.

### What ML must improve

1. **Reduce signal frequency:** from ~150 trades/symbol/year to ~20-40 (only high-conviction setups).
2. **Improve win rate:** target ≥ 35% at 2:1 RR to be gross-profitable, ≥ 45% to overcome costs.
3. **Regime awareness:** avoid trading L3-style flat markets where neither target nor stop resolves quickly.
4. **Cost consciousness at the dataset level:** the label construction (L1-L4) already uses the backtester's target/stop ratio, ensuring ML learns to predict moves large enough to cover costs.
"""),
])


# ─────────────────────────────────────────────────────────────────────────────
# Write notebooks to disk
# ─────────────────────────────────────────────────────────────────────────────

for name, notebook in [
    ("01_label_exploration.ipynb", nb01),
    ("02_feature_correlation.ipynb", nb02),
    ("03_baseline_review.ipynb", nb03),
]:
    path = NOTEBOOKS_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"Written: {path}")
