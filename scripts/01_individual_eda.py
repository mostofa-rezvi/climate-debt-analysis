import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import os

# Create production_artifacts directory if it doesn't exist
os.makedirs('production_artifacts', exist_ok=True)

class IndividualEDA:
    def __init__(self):
        self.scaler = StandardScaler()
        self.tscv = TimeSeriesSplit(n_splits=5)

    def analyze_rainfall(self, df):
        print("\n--- Analyzing Rainfall Data ---")
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Feature Engineering: Lags and Month
        df['month'] = df['date'].dt.month
        df['lag_1'] = df.groupby('PCODE')['rfq'].shift(1)
        df['lag_3'] = df.groupby('PCODE')['rfq'].shift(3)
        df = df.dropna(subset=['month', 'lag_1', 'lag_3', 'rfq'])

        # Supervised: Predict rfq
        features = ['month', 'lag_1', 'lag_3']
        X = df[features]
        y = df['rfq']
        
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=self.tscv, scoring='r2')
        print(f"Rainfall (rfq) Prediction R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Unsupervised: Cluster regions
        region_stats = df.groupby('PCODE')['rfq'].agg(['mean', 'std']).reset_index()
        X_cluster = self.scaler.fit_transform(region_stats[['mean', 'std']])
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        region_stats['cluster'] = kmeans.fit_predict(X_cluster)
        
        print("Rainfall Clusters (Region-wise):")
        print(region_stats.groupby('cluster')['PCODE'].count())
        
        # Save plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=region_stats, x='mean', y='std', hue='cluster', palette='viridis')
        plt.title('Rainfall Clusters by Region (Mean vs Std)')
        plt.savefig('production_artifacts/rainfall_clusters.png')
        plt.close()

    def analyze_food_prices(self, df):
        print("\n--- Analyzing Food Price Data ---")
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter for a major commodity to make analysis focused
        rice_df = df[df['commodity'].str.contains('Rice', case=False)].copy()
        rice_df = rice_df.sort_values('date')
        
        # Feature Engineering
        rice_df['month'] = rice_df['date'].dt.month
        le = LabelEncoder()
        rice_df['market_enc'] = le.fit_transform(rice_df['market'])
        rice_df['lag_1'] = rice_df.groupby('market')['usdprice'].shift(1)
        rice_df = rice_df.dropna(subset=['month', 'market_enc', 'lag_1', 'usdprice'])

        # Supervised: Predict usdprice
        features = ['month', 'market_enc', 'lag_1']
        X = rice_df[features]
        y = rice_df['usdprice']
        
        model = GradientBoostingRegressor(random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=self.tscv, scoring='r2')
        print(f"Food Price (usdprice) Prediction R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Unsupervised: Cluster commodities by volatility
        df['price_volatility'] = df.groupby('commodity')['usdprice'].transform('std')
        df['mean_price'] = df.groupby('commodity')['usdprice'].transform('mean')
        comm_stats = df.groupby('commodity')[['mean_price', 'price_volatility']].mean().dropna()
        
        X_cluster = self.scaler.fit_transform(comm_stats)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        comm_stats['cluster'] = kmeans.fit_predict(X_cluster)
        
        print("Food Commodity Clusters (Volatility):")
        print(comm_stats['cluster'].value_counts())

    def analyze_debt(self, df):
        print("\n--- Analyzing External Debt Data ---")
        # Filter for External debt stocks
        debt_stock = df[df['Indicator Name'].str.contains('External debt stocks, total', case=False)].copy()
        debt_stock = debt_stock.sort_values('Year')
        
        # Feature Engineering: Lag
        debt_stock['lag_1'] = debt_stock['Value'].shift(1)
        debt_stock = debt_stock.dropna(subset=['Year', 'lag_1', 'Value'])

        # Supervised: Predict Debt Value
        X = debt_stock[['Year', 'lag_1']]
        y = debt_stock['Value']
        
        model = Ridge(alpha=1.0)
        cv_scores = cross_val_score(model, X, y, cv=self.tscv, scoring='r2')
        print(f"Debt Prediction (Ridge) R2: {cv_scores.mean():.4f}")

        # Unsupervised: Debt Eras
        X_cluster = self.scaler.fit_transform(debt_stock[['Value']])
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        debt_stock['era_cluster'] = kmeans.fit_predict(X_cluster)
        
        print("Debt Eras (Yearly Clusters):")
        for cluster in sorted(debt_stock['era_cluster'].unique()):
            years = debt_stock[debt_stock['era_cluster'] == cluster]['Year']
            print(f"Cluster {cluster}: {years.min()} - {years.max()}")

if __name__ == "__main__":
    eda = IndividualEDA()
    
    # Load data
    rainfall_df = pd.read_csv('datasets/bgd-rainfall-subnat-full.csv')
    food_df = pd.read_csv('datasets/wfp_food_prices_bgd.csv')
    debt_df = pd.read_csv('datasets/external-debt_bgd.csv')
    
    eda.analyze_rainfall(rainfall_df)
    eda.analyze_food_prices(food_df)
    eda.analyze_debt(debt_df)
