import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge, Lasso
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

os.makedirs('production_artifacts/figures', exist_ok=True)

print("--- Pairwise B<->C: Food Prices & External Debt ---")
df_food = pd.read_csv('datasets/wfp_food_prices_bgd.csv')
df_debt = pd.read_csv('datasets/external-debt_bgd.csv')

df_food['date'] = pd.to_datetime(df_food['date'])
df_food = df_food[df_food['date'].dt.year > 1900]
df_food['year'] = df_food['date'].dt.year

staples = df_food[df_food['category'].isin(['cereals and tubers', 'pulses and nuts'])]
annual_food = staples.groupby('year')['usdprice'].mean().reset_index()
baseline_2000 = annual_food[annual_food['year'] == 2000]['usdprice'].values[0] if 2000 in annual_food['year'].values else annual_food['usdprice'].mean()
annual_food['food_vulnerability_index'] = annual_food['usdprice'] / baseline_2000
annual_food['food_price_inflation_yoy'] = annual_food['usdprice'].pct_change()

relevant_indicators = {
    'External debt stocks, total (DOD, current US$)': 'total_external_debt',
    'Imports of goods, services and primary income (BoP, current US$)': 'imports',
    'Exports of goods, services and primary income (BoP, current US$)': 'exports'
}
df_debt_filtered = df_debt[df_debt['Indicator Name'].isin(relevant_indicators.keys())].copy()
df_debt_filtered['Indicator Name'] = df_debt_filtered['Indicator Name'].map(relevant_indicators)

# The data is already in long format (Year, Value)
wide_df = df_debt_filtered.pivot_table(index='Year', columns='Indicator Name', values='Value').reset_index()
wide_df.rename(columns={'Year': 'year'}, inplace=True)
wide_df['year'] = wide_df['year'].astype(int)
wide_df['debt_growth_rate'] = wide_df['total_external_debt'].pct_change()
wide_df['trade_deficit'] = wide_df['imports'] - wide_df['exports']

merged_bc = pd.merge(annual_food, wide_df, on='year').dropna()

features = ['food_vulnerability_index', 'food_price_inflation_yoy', 'usdprice', 'imports', 'trade_deficit']
target = 'debt_growth_rate'

scaler = StandardScaler()
X = merged_bc[features]
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features, index=X.index)
y = merged_bc[target]

tscv = TimeSeriesSplit(n_splits=5)
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)

for train_idx, test_idx in tscv.split(X_scaled):
    ridge.fit(X_scaled.iloc[train_idx], y.iloc[train_idx])
    lasso.fit(X_scaled.iloc[train_idx], y.iloc[train_idx])

preds = ridge.predict(X_scaled.iloc[test_idx])
print(f"Ridge - Predict Debt Growth - R2: {r2_score(y.iloc[test_idx], preds):.4f}")

X_cluster = merged_bc[['food_vulnerability_index', 'debt_growth_rate']]
X_cluster_scaled = scaler.fit_transform(X_cluster)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
merged_bc['bc_cluster'] = kmeans.fit_predict(X_cluster_scaled)

merged_bc.to_csv('production_artifacts/bc_clusters.csv', index=False)

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_bc, x='food_vulnerability_index', y='debt_growth_rate', hue='bc_cluster', palette='Set1', s=150)
plt.title('Food Vulnerability vs Debt Growth Clusters')
plt.xlabel('Food Vulnerability Index (Base 2000)')
plt.ylabel('Debt Growth Rate')
plt.axhline(0, color='grey', linestyle='--')
plt.tight_layout()
plt.savefig('production_artifacts/figures/stage2_bc_clusters.png')
plt.close()

print("Saved artifacts and figures for B<->C.")
