---

## Data understanding before anything else

**Dataset A — Rainfall** (`bgd-rainfall-subnat-full.csv` + `5ytd`): decadal (10-day) observations for 72 admin units (PCODEs) from 1981 to 2026. The key variables are `rfh` (actual rainfall), `rfh_avg` (long-term mean), and `rfq` (rainfall quotient — the anomaly ratio where >100 means above-average, <100 means below). The 5ytd file covers 2022–2026 and is the "recent window." Use the full file for long-run baselines.

**Dataset B — Food Prices** (`wfp_food_prices_bgd.csv` + `wfp_markets_bgd.csv`): monthly retail/wholesale prices across 117 markets and 8 categories (cereals, oil, pulses, etc.) in both BDT and USD from 1998 to 2026. The `usdprice` column is your best cross-time comparison unit. Join with `wfp_markets_bgd.csv` on `market_id` to get lat/lon for spatial analysis.

**Dataset C — External Debt** (`external-debt_bgd.csv`): 60 World Bank indicators for Bangladesh annually from 1971–2024. The key indicators are "External debt stocks, total (DOD, current US$)", "Current account balance", "Imports of goods, services…", and "Exports of goods, services…". Pivot to wide format (one row per year, indicators as columns) to use in ML.

---

## Stage 1 — Individual dataset analysis

### Dataset A: Rainfall

**EDA and feature engineering.** Parse `date` to datetime and extract year, month, and decade (10-day period). Classify each observation's season: monsoon (June–October), pre-monsoon (March–May), post-monsoon (November), dry (December–February). Compute a binary `drought` flag (rfq < 75) and a `flood` flag (rfq > 150). Aggregate to monthly and annual totals per district.

**Supervised learning.** Target: `rfq` (anomaly quotient). Features: year, month, PCODE (label-encoded), 3-month lagged rfq, season. Use Linear Regression as the baseline, then a Random Forest to capture non-linearity. Evaluate with time-series cross-validation (train on pre-2018, test on 2019–2024 to avoid data leakage). Report R², MAE, and feature importances. This answers: "which districts and months are most predictably anomalous?"

**Unsupervised learning.** Pivot to a feature matrix where each row is a PCODE and columns are average rfq per season. Apply K-Means (k=3–5, use elbow and silhouette). Expected clusters: flood-prone (high monsoon rfq), drought-prone (low dry-season rfq), and climatically stable. Visualise on a Bangladesh map using district centroids.

---

### Dataset B: Food Prices

**EDA and feature engineering.** Parse dates, remove the erroneous 1900-08-15 outlier. Filter to categories relevant to food security: cereals/tubers, pulses/nuts, oil/fats. Compute a monthly national food price index (weighted average usdprice across staple commodities). Calculate month-over-month and year-over-year inflation rates. Merge with market locations for regional indexing.

**Supervised learning.** Target: `usdprice` for rice (the most critical staple). Features: year, month, `admin1` (region), commodity category, lag-1 and lag-3 month price, season flag. Models: Linear Regression (baseline), Random Forest, and Gradient Boosting. Use time-split cross-validation. Report coefficient signs for regression (does time increase price? does season matter?) and feature importances for tree models.

**Unsupervised learning.** Two tasks: (1) cluster commodities by their price-over-time profile (flatten to annual average usdprice per year, apply K-Means — expect a "stable-price" cluster vs a "volatile/rising" cluster); (2) cluster markets by their price levels and volatility (standard deviation of usdprice vs mean). This identifies which markets are persistently expensive vs. which spike seasonally.

---

### Dataset C: External Debt

**EDA and feature engineering.** Pivot the long-format table into wide format: one row per year, each indicator as a column. Keep the ~8 most relevant indicators: total external debt, debt as % of GNI, imports, exports, current account balance, personal remittances, FDI inflows. Compute year-over-year debt growth rate and trade deficit (imports minus exports).

**Supervised learning.** Target: external debt stock (current US$). Features: year, imports, exports, current account balance, remittances, FDI. Use Linear Regression and Ridge (to handle multicollinearity between indicators). Add polynomial features or interaction terms (e.g., trade deficit × year). Evaluate with leave-one-year-out cross-validation given the small annual sample. Interpret which indicators have the largest regression coefficients (normalize features first).

**Unsupervised learning.** Cluster years (1995–2024) by their economic profile across all indicators. K-Means with k=3–4. Expected clusters: low-debt stable era (pre-2005), moderate growth era (2005–2015), high-debt stress era (2016+). This gives you a labelled timeline you can later correlate with rainfall and food price events.

---

## Stage 2 — Pairwise (1-to-1) analysis

The critical step here is **temporal alignment**. Rainfall is decadal, food prices are monthly, and debt is annual. You will create two shared time resolutions: monthly (for A↔B) and annual (for B↔C and A↔C).

### A ↔ B: Does rainfall anomaly predict food price?

**Merge logic.** Aggregate rainfall to monthly level per PCODE. Map PCODEs to admin1 regions using the PCODE prefix (BD10 = Barisal, BD20 = Chittagong, etc.) to match the food price `admin1` field. Join on (year, month, admin1). The overlap window is approximately 2000–2026.

**Feature engineering.** Add lag-1, lag-2, and lag-3 month rfq per region. Add a binary "monsoon shock" indicator (rfq > 150 in any of the three prior months). Add the regional drought indicator.

**Supervised learning.** Target: monthly usdprice of rice (or a composite cereal index). Features: lagged rfq, monsoon shock, month, region. Use Random Forest and interpret feature importances — the key question is whether lag-1 or lag-3 rfq has more predictive power (i.e., does the food price effect of a rainfall shock show up immediately or after a harvest delay of 2–3 months?). Report R² improvement over a baseline model that uses only time and region.

**Unsupervised learning.** Cluster months jointly by rainfall anomaly and food price level. A month with rfq < 75 and high rice price is your "drought-driven food crisis" cluster — this is the key event type your hypothesis is built on. Use K-Means on normalised (rfq, usdprice_rice) and visualise the cluster timeline.

---

### B ↔ C: Does food price level drive external debt?

**Merge logic.** Aggregate food prices to annual national averages per commodity category. Compute an annual food vulnerability index (average usdprice of cereals + pulses, normalised by year-2000 baseline). Join with the wide-format debt dataset on year. Overlap: 1998–2024.

**Supervised learning.** Target: annual debt growth rate (%) or total debt stock. Features: food vulnerability index, food price inflation (YoY), cereal price level, import value, trade deficit. Use Ridge Regression and Lasso (Lasso for automatic feature selection). The hypothesis is that high cereal prices force import substitution spending, widen the trade deficit, and increase external borrowing — so import value should be a strong mediator.

**Unsupervised learning.** Cluster years by (food vulnerability index, debt growth rate). Expect clusters like: "low food stress, low debt growth" (pre-2005), "moderate stress, moderate debt" (2005–2015), and "high food price + high debt growth" (post-2017 — especially 2022 global food crisis year).

---

### A ↔ C: Does rainfall anomaly directly predict debt trajectory?

**Merge logic.** Aggregate rainfall to annual national anomaly score (mean rfq across all PCODEs per year). Join with debt dataset on year. This is the most aggregated and smallest dataset (~30 rows) so use it for interpretability, not for large ML models.

**Supervised learning.** Simple Linear and Ridge Regression. Target: debt growth rate. Features: annual national rfq, monsoon season rfq, drought frequency (fraction of districts in drought per year), 1-year and 2-year lagged rfq. Granger causality test to formally check if past rainfall predicts future debt (this is a direct academic contribution for the project's novelty claim).

**Unsupervised learning.** Association analysis — compute Pearson and Spearman correlation matrix between annual rfq, food price index, and debt indicators. Use a heatmap. Identify which debt indicators correlate most with rainfall anomalies.

---

## Stage 3 — Full joint analysis (all three together)

**Master dataset construction.** Annual resolution, inner join on year. Columns: `annual_rfq_national`, `monsoon_rfq`, `drought_districts_pct`, `cereal_price_index`, `food_vulnerability_index`, `food_price_inflation_yoy`, `external_debt_stock`, `debt_growth_rate`, `imports`, `trade_deficit`, `current_account`. Expected shape: ~25–30 rows (2000–2024). This is small — use it for interpretability, not deep learning.

**Supervised learning — full causal chain.** Target: external debt stock or debt growth rate. Two model families:

First, a linear pipeline — normalize all features, fit Ridge/Lasso Regression. Interpret coefficients to see which of the three "layers" (rainfall, food price, or macroeconomic controls) explains the most variance. Try feature subsets: rain-only model, food-price-only model, combined model — compare R² across all three to quantify the marginal contribution of each layer. This directly tests your hypothesis.

Second, a Random Forest / Gradient Boosting model. Report SHAP values (or permutation importance) for all features. The expected finding: food price indicators will rank higher in importance than raw rainfall features, suggesting food price is the mediating mechanism (rainfall → food price → debt), not a direct link.

**Unsupervised learning — year risk profiling.** Normalize all features across the three domains and apply K-Means (k=3–4). Label each cluster descriptively: "low-risk stable year", "rainfall-shock + food-stress year", "high-debt-stress year", "crisis year (all three elevated)". Visualise as a timeline scatter plot (year on x-axis, coloured by cluster). This is your key deliverable — it shows which years Bangladesh was most vulnerable to the full causal chain simultaneously.

**Mediation test (the academic contribution).** Fit three regressions: (1) Rainfall → Debt (direct), (2) Rainfall → Food Price, (3) Rainfall + Food Price → Debt. If adding food price as a predictor reduces the rainfall coefficient in model (3) vs. model (1), food price is confirmed as a mediator. This is the formal statistical backing for your hypothesis and gives the project its novelty.

---

## Python implementation stack

| Task | Libraries |
|---|---|
| Data loading & merging | `pandas`, `numpy` |
| EDA & visualization | `matplotlib`, `seaborn`, `plotly` |
| Supervised learning | `scikit-learn` (LinearRegression, Ridge, Lasso, RandomForestRegressor, GradientBoostingRegressor) |
| Model evaluation | `sklearn.model_selection` (TimeSeriesSplit, cross_val_score) |
| Unsupervised learning | `sklearn.cluster` (KMeans), `sklearn.preprocessing` (StandardScaler) |
| Feature importance | `sklearn.inspection` (permutation_importance) or `shap` |
| Correlation / Granger | `scipy.stats`, `statsmodels` |
| Spatial mapping | `geopandas`, `folium` |

---

## Project report structure (aligned to the marking rubric)

The 15 technical points map directly to these stages: individual EDA (problem formulation and data description), Stage 1 ML (correct supervised + unsupervised with cross-validation and interpretation), Stage 2 pairwise (novelty — cross-domain linking), Stage 3 joint (originality — formal mediation test and risk profiling). The 5 presentation points map to clear figure labeling, explained tables, and a coherent narrative that carries the reader from "rainfall fell" all the way to "Bangladesh borrowed more." Click any box in the diagram above to get the Python code for that specific step.