import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score

os.makedirs('production_artifacts/figures', exist_ok=True)

print("--- Dataset B: Food Prices EDA & ML ---")
df = pd.read_csv('datasets/wfp_food_prices_bgd.csv')

df['date'] = pd.to_datetime(df['date'])
df = df[df['date'].dt.year > 1900]
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

def get_season(month):
    if month in [6, 7, 8, 9, 10]: return 'monsoon'
    elif month in [3, 4, 5]: return 'pre-monsoon'
    elif month == 11: return 'post-monsoon'
    else: return 'dry'
df['season'] = df['month'].apply(get_season)

staples = df[df['category'].isin(['cereals and tubers', 'pulses and nuts', 'oil and fats'])]

# Supervised Learning
rice_df = df[df['commodity'].str.contains('Rice', case=False, na=False)].copy()
rice_df = rice_df.sort_values(by=['admin1', 'date'])
rice_df['lag_1_price'] = rice_df.groupby('admin1')['usdprice'].shift(1)
rice_df['lag_3_price'] = rice_df.groupby('admin1')['usdprice'].shift(3)
rice_df.dropna(subset=['lag_1_price', 'lag_3_price', 'usdprice'], inplace=True)

le_admin = LabelEncoder()
rice_df['admin1_encoded'] = le_admin.fit_transform(rice_df['admin1'])
le_season = LabelEncoder()
rice_df['season_encoded'] = le_season.fit_transform(rice_df['season'])

features = ['year', 'month', 'admin1_encoded', 'lag_1_price', 'lag_3_price', 'season_encoded']
X = rice_df[features]
y = rice_df['usdprice']

tscv = TimeSeriesSplit(n_splits=5)
gb = GradientBoostingRegressor(random_state=42)

for train_idx, test_idx in tscv.split(X):
    gb.fit(X.iloc[train_idx], y.iloc[train_idx])

preds = gb.predict(X.iloc[test_idx])
print(f"Gradient Boosting - Predict Rice Price - R2: {r2_score(y.iloc[test_idx], preds):.4f}")

# Plot Prediction vs Actual
plt.figure(figsize=(10, 5))
plt.plot(y.iloc[test_idx].values[:100], label='Actual Price', color='blue')
plt.plot(preds[:100], label='Predicted Price', color='red', linestyle='--')
plt.title('Rice Price: Actual vs Predicted (Test Set Sample)')
plt.legend()
plt.tight_layout()
plt.savefig('production_artifacts/figures/stage1_food_price_predictions.png')
plt.close()

# Unsupervised Learning
annual_comm = staples.groupby(['commodity', 'year'])['usdprice'].mean().unstack().fillna(0)
scaler = StandardScaler()
comm_scaled = scaler.fit_transform(annual_comm)
kmeans_comm = KMeans(n_clusters=3, random_state=42, n_init=10)
annual_comm['cluster'] = kmeans_comm.fit_predict(comm_scaled)
annual_comm.to_csv('production_artifacts/commodity_clusters.csv')

market_stats = staples.groupby('market')['usdprice'].agg(['mean', 'std']).fillna(0)
market_scaled = scaler.fit_transform(market_stats)
kmeans_market = KMeans(n_clusters=3, random_state=42, n_init=10)
market_stats['cluster'] = kmeans_market.fit_predict(market_scaled)
market_stats.to_csv('production_artifacts/market_clusters.csv')

# Plot Markets
plt.figure(figsize=(10, 6))
sns.scatterplot(data=market_stats, x='mean', y='std', hue='cluster', palette='Dark2', s=100)
plt.title('Market Clustering: Mean Price vs Volatility (Std Dev)')
plt.xlabel('Mean USD Price')
plt.ylabel('Price Volatility (Std Dev)')
plt.tight_layout()
plt.savefig('production_artifacts/figures/stage1_market_clusters.png')
plt.close()

print("Saved artifacts and figures for Dataset B.")
