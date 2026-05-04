import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from statsmodels.tsa.stattools import grangercausalitytests
import os

# Create production_artifacts directory if it doesn't exist
os.makedirs('production_artifacts', exist_ok=True)

PCODE_MAP = {
    'BD10': 'Barisal',
    'BD20': 'Chittagong',
    'BD30': 'Dhaka',
    'BD40': 'Khulna',
    'BD50': 'Rajshahi',
    'BD55': 'Rangpur',
    'BD60': 'Sylhet',
    'BD45': 'Mymensingh'
}

class PairwiseAlignment:
    def __init__(self):
        self.scaler = StandardScaler()
        self.tscv = TimeSeriesSplit(n_splits=5)

    def load_data(self):
        self.rainfall = pd.read_csv('datasets/bgd-rainfall-subnat-full.csv')
        self.food = pd.read_csv('datasets/wfp_food_prices_bgd.csv')
        self.debt = pd.read_csv('datasets/external-debt_bgd.csv')
        
        # Preprocessing dates
        self.rainfall['date'] = pd.to_datetime(self.rainfall['date'])
        self.food['date'] = pd.to_datetime(self.food['date'])
        
        # Mapping PCODE to Admin Names
        self.rainfall['admin1'] = self.rainfall['PCODE'].map(PCODE_MAP)

    def align_A_B(self):
        """A (Rainfall) <-> B (Food Prices)"""
        print("\n--- Aligning A (Rainfall) <-> B (Food Prices) ---")
        
        # 1. Aggregate Rainfall to Monthly
        self.rainfall['year'] = self.rainfall['date'].dt.year
        self.rainfall['month'] = self.rainfall['date'].dt.month
        
        monthly_rf = self.rainfall.groupby(['year', 'month', 'admin1'])['rfq'].sum().reset_index()
        
        # 2. Extract Lagged Rainfall (Shocks)
        monthly_rf = monthly_rf.sort_values(['admin1', 'year', 'month'])
        monthly_rf['rfq_lag1'] = monthly_rf.groupby('admin1')['rfq'].shift(1)
        monthly_rf['rfq_lag2'] = monthly_rf.groupby('admin1')['rfq'].shift(2)
        monthly_rf['rfq_lag3'] = monthly_rf.groupby('admin1')['rfq'].shift(3)
        
        # 3. Aggregate Food Prices to Monthly Admin1
        # Use average usdprice across markets in the same admin1
        monthly_food = self.food.groupby(['date', 'admin1'])['usdprice'].mean().reset_index()
        monthly_food['year'] = monthly_food['date'].dt.year
        monthly_food['month'] = monthly_food['date'].dt.month
        
        # 4. Merge
        merged_ab = pd.merge(monthly_food, monthly_rf, on=['year', 'month', 'admin1'], how='inner')
        merged_ab = merged_ab.dropna()
        
        if merged_ab.empty:
            print("Warning: Merged A-B dataset is empty. Check admin name alignment.")
            return
            
        # 5. Train RF to predict food prices based on rainfall shocks
        features = ['rfq', 'rfq_lag1', 'rfq_lag2', 'rfq_lag3']
        X = merged_ab[features]
        y = merged_ab['usdprice']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=self.tscv, scoring='r2')
        print(f"A-B Alignment (Rain -> Food) R2: {cv_scores.mean():.4f}")
        
        model.fit(X, y)
        importances = dict(zip(features, model.feature_importances_))
        print("Feature Importances (Rainfall Shocks):", importances)

    def align_B_C(self):
        """B (Food Prices) <-> C (External Debt)"""
        print("\n--- Aligning B (Food Prices) <-> C (External Debt) ---")
        
        # 1. Aggregate Food to Annual
        self.food['year'] = self.food['date'].dt.year
        annual_food = self.food.groupby('year')['usdprice'].agg(['mean', 'std']).reset_index()
        annual_food.rename(columns={'mean': 'avg_food_price', 'std': 'food_volatility'}, inplace=True)
        
        # 2. Prepare Debt Data
        debt_stock = self.debt[self.debt['Indicator Name'].str.contains('External debt stocks, total', case=False)].copy()
        debt_stock['debt_growth'] = debt_stock['Value'].pct_change()
        debt_stock.rename(columns={'Year': 'year'}, inplace=True)
        
        # 3. Merge
        merged_bc = pd.merge(annual_food, debt_stock[['year', 'debt_growth', 'Value']], on='year', how='inner')
        merged_bc = merged_bc.dropna()
        
        if merged_bc.empty:
            print("Warning: Merged B-C dataset is empty.")
            return

        # 4. Train Ridge Regression
        features = ['avg_food_price', 'food_volatility']
        X = self.scaler.fit_transform(merged_bc[features])
        y = merged_bc['debt_growth']
        
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        coeffs = dict(zip(features, model.coef_))
        print("B-C Alignment (Food -> Debt Growth) Coefficients:", coeffs)

    def align_A_C(self):
        """A (Rainfall) <-> C (External Debt)"""
        print("\n--- Aligning A (Rainfall) <-> C (External Debt) ---")
        
        # 1. Aggregate Rainfall to Annual (National)
        annual_rf = self.rainfall.groupby('year')['rfq'].sum().reset_index()
        annual_rf.rename(columns={'rfq': 'annual_rfq'}, inplace=True)
        
        # 2. Merge with Debt
        debt_stock = self.debt[self.debt['Indicator Name'].str.contains('External debt stocks, total', case=False)].copy()
        debt_stock.rename(columns={'Year': 'year'}, inplace=True)
        merged_ac = pd.merge(annual_rf, debt_stock[['year', 'Value']], on='year', how='inner')
        merged_ac = merged_ac.dropna()
        
        if len(merged_ac) < 5:
            print("Insufficient data for A-C causality tests.")
            return
            
        # 3. Correlation
        corr = merged_ac[['annual_rfq', 'Value']].corr().iloc[0, 1]
        print(f"A-C National Correlation (Rainfall vs Debt Stock): {corr:.4f}")
        
        # 4. Granger Causality (Rainfall -> Debt)
        # Note: This is exploratory due to small N in annual data
        print("Granger Causality Test (Rainfall -> Debt):")
        try:
            results = grangercausalitytests(merged_ac[['Value', 'annual_rfq']], maxlag=2, verbose=False)
            for lag, data in results.items():
                print(f"Lag {lag}: p-value = {data[0]['ssr_ftest'][1]:.4f}")
        except Exception as e:
            print(f"Granger Causality failed: {e}")

if __name__ == "__main__":
    aligner = PairwiseAlignment()
    aligner.load_data()
    aligner.align_A_B()
    aligner.align_B_C()
    aligner.align_A_C()
