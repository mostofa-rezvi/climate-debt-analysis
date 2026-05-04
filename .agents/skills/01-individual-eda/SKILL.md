---
name: 01-individual-eda
description: Stage 1 - Individual dataset analysis for rainfall, food prices, and external debt.
---

## Dataset A: Rainfall
**EDA and feature engineering.** Parse `date` to datetime and extract year, month, and decade (10-day period). Classify each observation's season: monsoon (June–October), pre-monsoon (March–May), post-monsoon (November), dry (December–February). Compute a binary `drought` flag (rfq < 75) and a `flood` flag (rfq > 150). Aggregate to monthly and annual totals per district.

**Supervised learning.** Target: `rfq` (anomaly quotient). Features: year, month, PCODE (label-encoded), 3-month lagged rfq, season. Use Linear Regression as the baseline, then a Random Forest to capture non-linearity. Evaluate with time-series cross-validation (`TimeSeriesSplit`) (train on pre-2018, test on 2019–2024 to avoid data leakage). Report R², MAE, and feature importances.

**Unsupervised learning.** Pivot to a feature matrix where each row is a PCODE and columns are average rfq per season. Apply K-Means (k=3–5, use elbow and silhouette) and make sure to include scaling with `StandardScaler`. Expected clusters: flood-prone (high monsoon rfq), drought-prone (low dry-season rfq), and climatically stable. Visualise on a Bangladesh map using district centroids.

## Dataset B: Food Prices
**EDA and feature engineering.** Parse dates, remove the erroneous 1900-08-15 outlier. Filter to categories relevant to food security: cereals/tubers, pulses/nuts, oil/fats. Compute a monthly national food price index (weighted average usdprice across staple commodities). Calculate month-over-month and year-over-year inflation rates. Merge with market locations for regional indexing.

**Supervised learning.** Target: `usdprice` for rice. Features: year, month, `admin1` (region), commodity category, lag-1 and lag-3 month price, season flag. Models: Linear Regression (baseline), Random Forest, and Gradient Boosting. Use `TimeSeriesSplit` cross-validation. Report coefficient signs for regression and feature importances for tree models.

**Unsupervised learning.** Two tasks: (1) cluster commodities by their price-over-time profile (flatten to annual average usdprice per year, apply K-Means with `StandardScaler`); (2) cluster markets by their price levels and volatility (standard deviation of usdprice vs mean) using K-Means with `StandardScaler`.

## Dataset C: External Debt
**EDA and feature engineering.** Pivot the long-format table into wide format: one row per year, each indicator as a column. Keep the ~8 most relevant indicators: total external debt, debt as % of GNI, imports, exports, current account balance, personal remittances, FDI inflows. Compute year-over-year debt growth rate and trade deficit (imports minus exports).

**Supervised learning.** Target: external debt stock (current US$). Features: year, imports, exports, current account balance, remittances, FDI. Use Linear Regression and Ridge. Add polynomial features or interaction terms (e.g., trade deficit × year). Evaluate with leave-one-year-out cross-validation given the small annual sample (or `TimeSeriesSplit`). Interpret which indicators have the largest regression coefficients (normalize features first).

**Unsupervised learning.** Cluster years (1995–2024) by their economic profile across all indicators. K-Means with k=3–4 and `StandardScaler`. Expected clusters: low-debt stable era (pre-2005), moderate growth era (2005–2015), high-debt stress era (2016+).
