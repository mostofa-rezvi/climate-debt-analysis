import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score

os.makedirs('production_artifacts/figures', exist_ok=True)

print("--- Pairwise A<->B: Rainfall & Food Prices ---")
df_rain = pd.read_csv('datasets/bgd-rainfall-subnat-full.csv')
df_food = pd.read_csv('datasets/wfp_food_prices_bgd.csv')

df_rain['date'] = pd.to_datetime(df_rain['date'])
df_rain['year'] = df_rain['date'].dt.year
df_rain['month'] = df_rain['date'].dt.month

pcode_map = {'BD10': 'Barisal', 'BD20': 'Chittagong', 'BD30': 'Dhaka', 'BD40': 'Khulna', 'BD50': 'Rajshahi', 'BD55': 'Rangpur', 'BD60': 'Sylhet', 'BD45': 'Mymensingh'}
df_rain['admin1'] = df_rain['PCODE'].astype(str).str[:4].map(pcode_map)
df_rain_monthly = df_rain.groupby(['year', 'month', 'admin1'])['rfq'].mean().reset_index()

df_rain_monthly = df_rain_monthly.sort_values(by=['admin1', 'year', 'month'])
for i in [1, 2, 3]:
    df_rain_monthly[f'lag_{i}_rfq'] = df_rain_monthly.groupby('admin1')['rfq'].shift(i)
df_rain_monthly['monsoon_shock'] = ((df_rain_monthly['lag_1_rfq'] > 150) | (df_rain_monthly['lag_2_rfq'] > 150) | (df_rain_monthly['lag_3_rfq'] > 150)).astype(int)
df_rain_monthly['drought'] = (df_rain_monthly['rfq'] < 75).astype(int)

df_food['date'] = pd.to_datetime(df_food['date'])
df_food = df_food[df_food['date'].dt.year > 1900]
df_food['year'] = df_food['date'].dt.year
df_food['month'] = df_food['date'].dt.month

rice_df = df_food[df_food['commodity'].str.contains('Rice', case=False, na=False)]
rice_monthly = rice_df.groupby(['year', 'month', 'admin1'])['usdprice'].mean().reset_index()

merged_ab = pd.merge(df_rain_monthly, rice_monthly, on=['year', 'month', 'admin1']).dropna()

# Supervised
le_admin = LabelEncoder()
merged_ab['admin1_encoded'] = le_admin.fit_transform(merged_ab['admin1'])
features = ['year', 'month', 'admin1_encoded', 'lag_1_rfq', 'lag_2_rfq', 'lag_3_rfq', 'monsoon_shock', 'drought']
X = merged_ab[features]
y = merged_ab['usdprice']

tscv = TimeSeriesSplit(n_splits=5)
rf = RandomForestRegressor(n_estimators=50, random_state=42)
for train_idx, test_idx in tscv.split(X):
    rf.fit(X.iloc[train_idx], y.iloc[train_idx])

print(f"RF - Predict Food Price from Rain - R2: {r2_score(y.iloc[test_idx], rf.predict(X.iloc[test_idx])):.4f}")

# Unsupervised
X_cluster = merged_ab[['rfq', 'usdprice']]
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
merged_ab['vulnerability_cluster'] = kmeans.fit_predict(X_cluster_scaled)

merged_ab.to_csv('production_artifacts/ab_vulnerability_clusters.csv', index=False)

# Plot timeline of vulnerability
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_ab, x='rfq', y='usdprice', hue='vulnerability_cluster', palette='magma', alpha=0.7)
plt.axvline(100, color='grey', linestyle='--')
plt.title('Rainfall Anomaly vs Food Price (Vulnerability Clusters)')
plt.xlabel('Rainfall Anomaly Quotient (RFQ)')
plt.ylabel('Rice Price (USD)')
plt.tight_layout()
plt.savefig('production_artifacts/figures/stage2_ab_vulnerability_scatter.png')
plt.close()
print("Saved artifacts and figures for A<->B.")
