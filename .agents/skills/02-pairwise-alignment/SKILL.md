---
name: 02-pairwise-alignment
description: Stage 2 - Pairwise (1-to-1) analysis for rainfall vs food price, food price vs debt, and rainfall vs debt.
---

## A ↔ B: Does rainfall anomaly predict food price?
**Merge logic.** Aggregate rainfall to monthly level per PCODE. Map PCODEs to admin1 regions using the PCODE prefix (BD10 = Barisal, BD20 = Chittagong, etc.) to match the food price `admin1` field. Join on (year, month, admin1). The overlap window is approximately 2000–2026.

**Feature engineering.** Add lag-1, lag-2, and lag-3 month rfq per region. Add a binary "monsoon shock" indicator (rfq > 150 in any of the three prior months). Add the regional drought indicator.

**Supervised learning.** Target: monthly usdprice of rice (or a composite cereal index). Features: lagged rfq, monsoon shock, month, region. Use Random Forest and interpret feature importances. Report R² improvement over a baseline model that uses only time and region. Use `TimeSeriesSplit` cross-validation.

**Unsupervised learning.** Cluster months jointly by rainfall anomaly and food price level. Use K-Means on normalised (rfq, usdprice_rice) using `StandardScaler` and visualise the cluster timeline.

## B ↔ C: Does food price level drive external debt?
**Merge logic.** Aggregate food prices to annual national averages per commodity category. Compute an annual food vulnerability index (average usdprice of cereals + pulses, normalised by year-2000 baseline). Join with the wide-format debt dataset on year. Overlap: 1998–2024.

**Supervised learning.** Target: annual debt growth rate (%) or total debt stock. Features: food vulnerability index, food price inflation (YoY), cereal price level, import value, trade deficit. Use Ridge Regression and Lasso. Use `TimeSeriesSplit` cross-validation.

**Unsupervised learning.** Cluster years by (food vulnerability index, debt growth rate). K-Means with `StandardScaler`.

## A ↔ C: Does rainfall anomaly directly predict debt trajectory?
**Merge logic.** Aggregate rainfall to annual national anomaly score (mean rfq across all PCODEs per year). Join with debt dataset on year. This is the most aggregated and smallest dataset (~30 rows).

**Supervised learning.** Simple Linear and Ridge Regression. Target: debt growth rate. Features: annual national rfq, monsoon season rfq, drought frequency, 1-year and 2-year lagged rfq. Use `TimeSeriesSplit` cross-validation. Apply Granger causality test to formally check if past rainfall predicts future debt.

**Unsupervised learning.** Association analysis — compute Pearson and Spearman correlation matrix between annual rfq, food price index, and debt indicators. Use a heatmap. Identify which debt indicators correlate most with rainfall anomalies.
