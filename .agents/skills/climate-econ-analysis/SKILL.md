---
name: climate-econ-analysis
description: Executes the 3-stage ML pipeline correlating rainfall, food prices, and external debt.
---

## Execution Steps

### Step 1: Stage 1 - Individual Analysis
1. Load `bgd-rainfall-subnat-full.csv`, `wfp_food_prices_bgd.csv`, and `external-debt_bgd.csv`.
2. Run supervised learning: Predict `rfq` (Linear/RF), `usdprice` (RF/Gradient Boosting), and debt stock (Ridge).
3. Run unsupervised learning: K-Means clustering for flood/drought regions, food price volatility, and debt era timelines.

### Step 2: Stage 2 - Pairwise Alignment
1. **A↔B**: Align decadal rainfall to monthly food prices. Extract lagged `rfq` features. Train RF to predict food prices based on rainfall shocks.
2. **B↔C**: Align monthly food to annual debt. Train Ridge Regression to predict debt growth via food vulnerability.
3. **A↔C**: Align rainfall to annual debt. Compute correlation matrices and run Granger Causality tests.

### Step 3: Stage 3 - Full Joint & Mediation (The Academic Contribution)
1. Merge all into an annual master dataset (~25-30 rows).
2. Train a global Random Forest to extract SHAP/Permutation feature importance.
3. Execute a 3-step Baron-Kenny Mediation Test using OLS regression to prove: Rainfall -> Food Price -> External Debt.
4. Output final visualizations to the `production_artifacts/` folder.