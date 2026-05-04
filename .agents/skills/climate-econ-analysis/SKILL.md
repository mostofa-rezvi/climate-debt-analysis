---
name: climate-econ-analysis
description: Executes the 3-stage ML pipeline correlating rainfall, food prices, and external debt.
---

## Rules & Constraints (Strictly Follow)
- Always prioritize `pandas` for data manipulation and temporal alignment (decadal to monthly to annual).
- Use `scikit-learn` for all supervised and unsupervised learning.
- Strictly adhere to time-series cross-validation (`TimeSeriesSplit`) to prevent data leakage.
- Ensure all K-Means clustering implementations include scaling (`StandardScaler`).
- For Stage 3, apply formal statistical Mediation Testing using `statsmodels`.
- Do not invent data; strictly use the 5 provided CSV files: `bgd-rainfall-subnat-full.csv`, `5ytd.csv` (implied or combined), `wfp_food_prices_bgd.csv`, `wfp_markets_bgd.csv`, and `external-debt_bgd.csv`.

## Execution Steps

### Step 1: Stage 1 - Individual Analysis
1. Load datasets using `pandas`.
2. Run supervised learning (with `TimeSeriesSplit`): Predict `rfq` (Linear/RF), `usdprice` (RF/Gradient Boosting), and debt stock (Ridge) using `scikit-learn`.
3. Run unsupervised learning: K-Means clustering (with `StandardScaler`) for flood/drought regions, food price volatility, and debt era timelines.

### Step 2: Stage 2 - Pairwise Alignment
1. **A↔B**: Align decadal rainfall to monthly food prices. Extract lagged `rfq` features. Train RF to predict food prices based on rainfall shocks.
2. **B↔C**: Align monthly food to annual debt. Train Ridge Regression to predict debt growth via food vulnerability.
3. **A↔C**: Align rainfall to annual debt. Compute correlation matrices and run Granger Causality tests.
*Note: Use `TimeSeriesSplit` and `StandardScaler` where applicable.*

### Step 3: Stage 3 - Full Joint & Mediation (The Academic Contribution)
1. Merge all into an annual master dataset (~25-30 rows).
2. Train a global Random Forest to extract SHAP/Permutation feature importance.
3. Execute a 3-step Baron-Kenny Mediation Test using OLS regression from `statsmodels` to prove: Rainfall -> Food Price -> External Debt.
4. Output final visualizations to the `production_artifacts/` folder.