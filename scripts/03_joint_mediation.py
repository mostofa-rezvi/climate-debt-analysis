import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import os

# Create production_artifacts directory if it doesn't exist
os.makedirs('production_artifacts', exist_ok=True)

class JointMediationAnalysis:
    def __init__(self):
        self.scaler = StandardScaler()

    def prepare_master_dataset(self):
        print("\n--- Building Annual Master Dataset ---")
        
        # 1. Rainfall (X): Annual National Total
        rf_df = pd.read_csv('datasets/bgd-rainfall-subnat-full.csv')
        rf_df['date'] = pd.to_datetime(rf_df['date'])
        rf_df['year'] = rf_df['date'].dt.year
        annual_rf = rf_df.groupby('year')['rfq'].sum().reset_index()
        annual_rf.rename(columns={'rfq': 'annual_rfq'}, inplace=True)
        
        # 2. Food Vulnerability (M): Annual Price Volatility
        food_df = pd.read_csv('datasets/wfp_food_prices_bgd.csv')
        food_df['date'] = pd.to_datetime(food_df['date'])
        food_df['year'] = food_df['date'].dt.year
        annual_food = food_df.groupby('year')['usdprice'].agg(['mean', 'std']).reset_index()
        annual_food.rename(columns={'mean': 'avg_price', 'std': 'food_volatility'}, inplace=True)
        
        # 3. Debt (Y): Annual Growth Rate
        debt_df = pd.read_csv('datasets/external-debt_bgd.csv')
        debt_stock = debt_df[debt_df['Indicator Name'].str.contains('External debt stocks, total', case=False)].copy()
        debt_stock = debt_stock.sort_values('Year')
        debt_stock['debt_growth'] = debt_stock['Value'].pct_change()
        debt_stock.rename(columns={'Year': 'year'}, inplace=True)
        
        # 4. Merge All
        master_df = pd.merge(annual_rf, annual_food, on='year', how='inner')
        master_df = pd.merge(master_df, debt_stock[['year', 'debt_growth', 'Value']], on='year', how='inner')
        master_df = master_df.dropna()
        
        print(f"Master Dataset created with {len(master_df)} annual samples.")
        return master_df

    def run_feature_importance(self, df):
        print("\n--- Global Feature Importance (Random Forest) ---")
        features = ['annual_rfq', 'avg_price', 'food_volatility']
        X = df[features]
        y = df['debt_growth']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        print("Feature Importance Results:\n", importances)
        
        # Plot
        plt.figure(figsize=(10, 6))
        importances.plot(kind='bar', color='teal')
        plt.title('Feature Importance: Drivers of Debt Growth')
        plt.ylabel('Score')
        plt.savefig('production_artifacts/feature_importance.png')
        plt.close()

    def perform_mediation_test(self, df):
        """
        Baron-Kenny Mediation Test:
        Rainfall (X) -> Food Volatility (M) -> Debt Growth (Y)
        """
        print("\n--- Baron-Kenny Mediation Test ---")
        X = df['annual_rfq']
        M = df['food_volatility']
        Y = df['debt_growth']
        
        # Step 1: Total Effect (X -> Y)
        X_const = sm.add_constant(X)
        model1 = sm.OLS(Y, X_const).fit()
        c_total = model1.params[1]
        p_total = model1.pvalues[1]
        
        # Step 2: Path a (X -> M)
        model2 = sm.OLS(M, X_const).fit()
        a_path = model2.params[1]
        p_a = model2.pvalues[1]
        
        # Step 3: Path b and Direct Effect c' (X, M -> Y)
        XM_const = sm.add_constant(pd.concat([X, M], axis=1))
        model3 = sm.OLS(Y, XM_const).fit()
        b_path = model3.params['food_volatility']
        c_prime = model3.params['annual_rfq']
        p_b = model3.pvalues['food_volatility']
        p_c_prime = model3.pvalues['annual_rfq']
        
        print(f"1. Total Effect (c): Coeff={c_total:.4f}, p={p_total:.4f}")
        print(f"2. Path a (X -> M): Coeff={a_path:.4f}, p={p_a:.4f}")
        print(f"3. Path b (M -> Y): Coeff={b_path:.4f}, p={p_b:.4f}")
        print(f"4. Direct Effect (c'): Coeff={c_prime:.4f}, p={p_c_prime:.4f}")
        
        # Interpretation
        if p_a < 0.05 and p_b < 0.05:
            if p_c_prime > 0.05:
                print("RESULT: FULL MEDIATION. Climate impacts debt exclusively through food prices.")
            else:
                print("RESULT: PARTIAL MEDIATION. Both food and other climate channels drive debt.")
        else:
            print("RESULT: NO MEDIATION. One or more required paths are not statistically significant.")

    def cluster_risk_eras(self, df):
        print("\n--- Clustering Years into Risk Eras ---")
        features = ['annual_rfq', 'food_volatility', 'debt_growth']
        X_scaled = self.scaler.fit_transform(df[features])
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df['risk_era'] = kmeans.fit_predict(X_scaled)
        
        # Label Eras
        # Calculate mean debt growth per cluster to identify 'Crisis' years
        era_means = df.groupby('risk_era')['debt_growth'].mean().sort_values()
        labels = {era_means.index[0]: 'Stable', era_means.index[1]: 'Moderate Stress', era_means.index[2]: 'Crisis'}
        df['era_label'] = df['risk_era'].map(labels)
        
        print("Risk Era Summary:")
        print(df.groupby('era_label')['year'].agg(['count', 'min', 'max']))
        
        # Save results
        df.to_csv('production_artifacts/master_analysis_results.csv', index=False)

if __name__ == "__main__":
    analysis = JointMediationAnalysis()
    master_df = analysis.prepare_master_dataset()
    analysis.run_feature_importance(master_df)
    analysis.perform_mediation_test(master_df)
    analysis.cluster_risk_eras(master_df)
    print("\nStage 3 Complete. Results saved to production_artifacts/")
