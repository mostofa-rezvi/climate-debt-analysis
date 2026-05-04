---
name: 03-joint-mediation
description: Stage 3 - Full joint analysis (all three together) with formal Mediation Testing.
---

## Master dataset construction
Annual resolution, inner join on year. Columns: `annual_rfq_national`, `monsoon_rfq`, `drought_districts_pct`, `cereal_price_index`, `food_vulnerability_index`, `food_price_inflation_yoy`, `external_debt_stock`, `debt_growth_rate`, `imports`, `trade_deficit`, `current_account`. Expected shape: ~25–30 rows (2000–2024).

## Supervised learning — full causal chain
Target: external debt stock or debt growth rate. Two model families:
1. Linear pipeline: normalize all features (`StandardScaler`), fit Ridge/Lasso Regression. Interpret coefficients to see which of the three "layers" (rainfall, food price, or macroeconomic controls) explains the most variance. Try feature subsets: rain-only model, food-price-only model, combined model — compare R² across all three. Use `TimeSeriesSplit` for cross-validation.
2. Tree model: Random Forest / Gradient Boosting model. Report SHAP values (or permutation importance) for all features. Expectation: food price indicators will rank higher in importance than raw rainfall features. Use `TimeSeriesSplit` for cross-validation.

## Unsupervised learning — year risk profiling
Normalize all features across the three domains (`StandardScaler`) and apply K-Means (k=3–4). Label each cluster descriptively: "low-risk stable year", "rainfall-shock + food-stress year", "high-debt-stress year", "crisis year (all three elevated)". Visualise as a timeline scatter plot (year on x-axis, coloured by cluster).

## Mediation test (the academic contribution)
Fit three regressions using `statsmodels` to execute a formal Baron-Kenny Mediation Test:
1. Rainfall → Debt (direct)
2. Rainfall → Food Price
3. Rainfall + Food Price → Debt

If adding food price as a predictor reduces the rainfall coefficient in model (3) vs. model (1), food price is confirmed as a mediator. This is the formal statistical backing for the hypothesis.
