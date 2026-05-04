import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import statsmodels.api as sm

class ClimateDebtModeler:
    def __init__(self):
        self.scaler = StandardScaler()

    # ==========================================
    # STAGE 1: Unsupervised & Supervised (Samples)
    # ==========================================
    def cluster_rainfall_regions(self, rainfall_df):
        """Unsupervised: K-Means clustering for drought/flood prone regions."""
        # Assuming rainfall_df is pivoted to PCODE rows and seasonal rfq columns
        features = ['monsoon_rfq', 'dry_rfq', 'pre_monsoon_rfq']
        X_scaled = self.scaler.fit_transform(rainfall_df[features])
        
        kmeans = KMeans(n_clusters=4, random_state=42)
        rainfall_df['climate_cluster'] = kmeans.fit_predict(X_scaled)
        return rainfall_df

    def predict_food_prices(self, price_df):
        """Supervised: Predict usdprice using RF and TimeSeries CV."""
        features = ['lag_1_price', 'lag_3_price', 'season_encoded', 'admin1_encoded']
        X = price_df[features].fillna(0)
        y = price_df['usdprice']

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Time-series cross validation
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
        model.fit(X, y)
        
        feature_importance = dict(zip(features, model.feature_importances_))
        return model, cv_scores, feature_importance

    # ==========================================
    # STAGE 2: Pairwise Learning 
    # ==========================================
    def pairwise_food_debt_ridge(self, merged_annual_df):
        """Supervised: Does food vulnerability drive debt? Handling multicollinearity."""
        features = ['food_vulnerability_idx', 'cereal_inflation_yoy', 'trade_deficit']
        X = self.scaler.fit_transform(merged_annual_df[features].fillna(0))
        y = merged_annual_df['debt_growth_rate']

        ridge = Ridge(alpha=1.0)
        ridge.fit(X, y)
        
        return dict(zip(features, ridge.coef_))

    # ==========================================
    # STAGE 3: Full Joint Analysis & Mediation
    # ==========================================
    def year_risk_profiling(self, master_df):
        """Unsupervised: Cluster years into risk eras (Stage 3 deliverable)."""
        features = ['annual_rfq_national', 'food_vulnerability_idx', 'debt_growth_rate']
        X_scaled = self.scaler.fit_transform(master_df[features])
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        master_df['risk_era_cluster'] = kmeans.fit_predict(X_scaled)
        # 0: Stable, 1: Moderate Stress, 2: Crisis Year
        return master_df

    def mediation_test(self, master_df):
        """
        The Academic Contribution: Formal Baron-Kenny Mediation Test
        Hypothesis: Rainfall (X) affects Debt (Y) strictly through Food Prices (M)
        """
        X = master_df['annual_rfq_national'] # Independent
        M = master_df['food_vulnerability_idx'] # Mediator
        Y = master_df['debt_growth_rate'] # Dependent
        
        # Step 1: Total effect of X on Y
        X_with_const = sm.add_constant(X)
        model_1 = sm.OLS(Y, X_with_const).fit()
        
        # Step 2: Effect of X on M
        model_2 = sm.OLS(M, X_with_const).fit()
        
        # Step 3: Effect of X and M on Y
        XM_with_const = sm.add_constant(pd.concat([X, M], axis=1))
        model_3 = sm.OLS(Y, XM_with_const).fit()
        
        print("=== MEDIATION TEST RESULTS ===")
        print(f"Path c (Rainfall -> Debt directly): P-val {model_1.pvalues['annual_rfq_national']:.4f}")
        print(f"Path a (Rainfall -> Food Price): P-val {model_2.pvalues['annual_rfq_national']:.4f}")
        print(f"Path b (Food Price -> Debt, controlling for Rain): P-val {model_3.pvalues['food_vulnerability_idx']:.4f}")
        print(f"Path c' (Rainfall -> Debt, controlling for Food): P-val {model_3.pvalues['annual_rfq_national']:.4f}")
        
        # Interpretation logic
        if model_3.pvalues['annual_rfq_national'] > 0.05 and model_1.pvalues['annual_rfq_national'] < 0.05:
            print("CONCLUSION: Full Mediation confirmed. Food price is the primary mechanism.")
            
        return model_1, model_2, model_3

if __name__ == "__main__":
    # The Antigravity agent will use pandas to load and align your CSVs here, 
    # then pass them into the ClimateDebtModeler class.
    print("Pipeline initialized. Ready for data ingestion.")