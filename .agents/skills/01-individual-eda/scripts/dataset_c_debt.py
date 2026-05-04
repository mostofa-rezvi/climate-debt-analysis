import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

os.makedirs('production_artifacts/figures', exist_ok=True)

print("--- Dataset C: External Debt EDA & ML ---")
df = pd.read_csv('datasets/external-debt_bgd.csv')

relevant_indicators = {
    'External debt stocks, total (DOD, current US$)': 'total_external_debt',
    'Imports of goods, services and primary income (BoP, current US$)': 'imports',
    'Exports of goods, services and primary income (BoP, current US$)': 'exports',
    'Current account balance (BoP, current US$)': 'current_account_balance',
    'Personal remittances, received (current US$)': 'personal_remittances',
    'Foreign direct investment, net inflows (BoP, current US$)': 'fdi_inflows'
}

df_filtered = df[df['Indicator Name'].isin(relevant_indicators.keys())].copy()
df_filtered['Indicator Name'] = df_filtered['Indicator Name'].map(relevant_indicators)

# The data is already in long format (Year, Value)
wide_df = df_filtered.pivot_table(index='Year', columns='Indicator Name', values='Value').reset_index()
wide_df.rename(columns={'Year': 'year'}, inplace=True)
wide_df['year'] = wide_df['year'].astype(int)
wide_df['debt_growth_rate'] = wide_df['total_external_debt'].pct_change()
wide_df['trade_deficit'] = wide_df['imports'] - wide_df['exports']
wide_df.dropna(inplace=True)

# Supervised Learning
features = ['year', 'imports', 'exports', 'current_account_balance', 'personal_remittances', 'fdi_inflows', 'trade_deficit']
target = 'total_external_debt'

scaler = StandardScaler()
X = wide_df[features]
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features, index=X.index)
y = wide_df[target]

tscv = TimeSeriesSplit(n_splits=5)
ridge = Ridge(alpha=1.0)

for train_idx, test_idx in tscv.split(X_scaled):
    ridge.fit(X_scaled.iloc[train_idx], y.iloc[train_idx])

preds = ridge.predict(X_scaled.iloc[test_idx])
print(f"Ridge Regression - Predict Debt Stock - R2: {r2_score(y.iloc[test_idx], preds):.4f}")

# Plot Regression Coeffs
plt.figure(figsize=(10, 6))
coeffs = pd.Series(ridge.coef_, index=features).sort_values()
sns.barplot(x=coeffs.values, y=coeffs.index, hue=coeffs.index, palette='coolwarm', legend=False)
plt.title('Ridge Regression Coefficients for Total External Debt')
plt.tight_layout()
plt.savefig('production_artifacts/figures/stage1_debt_coefficients.png')
plt.close()

# Unsupervised Learning
cluster_features = ['total_external_debt', 'imports', 'exports', 'current_account_balance', 'personal_remittances', 'fdi_inflows']
X_cluster = wide_df[cluster_features]
X_cluster_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
wide_df['economic_era_cluster'] = kmeans.fit_predict(X_cluster_scaled)
wide_df.to_csv('production_artifacts/economic_eras_clusters.csv', index=False)

# Plot Timeline of Eras
plt.figure(figsize=(12, 6))
sns.scatterplot(data=wide_df, x='year', y='total_external_debt', hue='economic_era_cluster', palette='Set2', s=150)
plt.plot(wide_df['year'], wide_df['total_external_debt'], color='grey', alpha=0.5, linestyle='--')
plt.title('Economic Eras: Total External Debt Trajectory (Clustered)')
plt.ylabel('Total External Debt (Current US$)')
plt.xlabel('Year')
plt.tight_layout()
plt.savefig('production_artifacts/figures/stage1_debt_era_timeline.png')
plt.close()

print("Saved artifacts and figures for Dataset C.")
