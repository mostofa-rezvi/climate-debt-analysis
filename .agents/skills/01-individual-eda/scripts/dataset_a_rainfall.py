import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

os.makedirs('production_artifacts/figures', exist_ok=True)

print("--- Dataset A: Rainfall EDA & ML ---")
df = pd.read_csv('datasets/bgd-rainfall-subnat-full.csv')

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

def get_season(month):
    if month in [6, 7, 8, 9, 10]: return 'monsoon'
    elif month in [3, 4, 5]: return 'pre-monsoon'
    elif month == 11: return 'post-monsoon'
    else: return 'dry'

df['season'] = df['month'].apply(get_season)
df['drought'] = (df['rfq'] < 75).astype(int)
df['flood'] = (df['rfq'] > 150).astype(int)

df['period'] = df['date'].dt.to_period('M')
df_monthly = df.groupby(['period', 'PCODE', 'year', 'month', 'season']).agg(
    rfq=('rfq', 'mean'), rfh=('rfh', 'sum')
).reset_index()

df_monthly = df_monthly.sort_values(by=['PCODE', 'period'])
df_monthly['lag_3_rfq'] = df_monthly.groupby('PCODE')['rfq'].shift(3)
df_monthly.dropna(inplace=True)

# Supervised Learning
le_pcode = LabelEncoder()
df_monthly['PCODE_encoded'] = le_pcode.fit_transform(df_monthly['PCODE'])
le_season = LabelEncoder()
df_monthly['season_encoded'] = le_season.fit_transform(df_monthly['season'])

features = ['year', 'month', 'PCODE_encoded', 'lag_3_rfq', 'season_encoded']
X = df_monthly[features]
y = df_monthly['rfq']

tscv = TimeSeriesSplit(n_splits=5)
rf = RandomForestRegressor(n_estimators=50, random_state=42)

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print(f"Random Forest - Predict RFQ - R2: {r2_score(y_test, y_pred_rf):.4f}")

# Plot Feature Importances
plt.figure(figsize=(10, 6))
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette='viridis', legend=False)
plt.title('Rainfall RFQ Predictor: Feature Importances')
plt.tight_layout()
plt.savefig('production_artifacts/figures/stage1_rain_feature_importance.png')
plt.close()

# Unsupervised Learning
pivot_df = df_monthly.pivot_table(index='PCODE', columns='season', values='rfq', aggfunc='mean').fillna(100)
scaler = StandardScaler()
pivot_scaled = scaler.fit_transform(pivot_df)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
pivot_df['cluster'] = kmeans.fit_predict(pivot_scaled)
pivot_df.to_csv('production_artifacts/rainfall_clusters.csv')

# Visualisation of clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pivot_df, x='monsoon', y='dry', hue='cluster', palette='Set1', s=100)
plt.title('PCODE Rainfall Clusters (Monsoon vs Dry season anomalies)')
plt.axhline(100, color='grey', linestyle='--')
plt.axvline(100, color='grey', linestyle='--')
plt.tight_layout()
plt.savefig('production_artifacts/figures/stage1_rain_clusters.png')
plt.close()

# Note: Spatial Map visualization script logic here using folium/geopandas
# A real spatial map requires district shapefiles or centroid coordinates matching PCODE.
print("Saved artifacts and figures for Dataset A.")
