import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
import statsmodels.api as sm

os.makedirs('production_artifacts/figures', exist_ok=True)
print("--- Stage 3: Full Joint Analysis & Mediation Test ---")

df_rain = pd.read_csv('datasets/bgd-rainfall-subnat-full.csv')
df_food = pd.read_csv('datasets/wfp_food_prices_bgd.csv')
df_debt = pd.read_csv('datasets/external-debt_bgd.csv')

# Prep Data
df_rain['date'] = pd.to_datetime(df_rain['date'])
df_rain['year'] = df_rain['date'].dt.year
df_rain['month'] = df_rain['date'].dt.month
df_rain['season'] = df_rain['month'].apply(lambda m: 'monsoon' if m in [6,7,8,9,10] else 'other')
rain_annual = df_rain.groupby('year')['rfq'].mean().reset_index().rename(columns={'rfq': 'annual_rfq_national'})
rain_monsoon = df_rain[df_rain['season'] == 'monsoon'].groupby('year')['rfq'].mean().reset_index().rename(columns={'rfq': 'monsoon_rfq'})
drought_df = df_rain.groupby(['year', 'PCODE'])['rfq'].mean().reset_index()
drought_df['drought'] = (drought_df['rfq'] < 75).astype(int)
rain_drought = drought_df.groupby('year')['drought'].mean().reset_index().rename(columns={'drought': 'drought_districts_pct'})
master_rain = pd.merge(rain_annual, rain_monsoon, on='year').merge(rain_drought, on='year')

df_food['date'] = pd.to_datetime(df_food['date'])
df_food = df_food[df_food['date'].dt.year > 1900]
df_food['year'] = df_food['date'].dt.year
staples = df_food[df_food['category'].isin(['cereals and tubers', 'pulses and nuts'])]
food_annual = staples.groupby('year')['usdprice'].mean().reset_index().rename(columns={'usdprice': 'cereal_price_index'})
base_price = food_annual[food_annual['year'] == 2000]['cereal_price_index'].values[0] if 2000 in food_annual['year'].values else food_annual['cereal_price_index'].mean()
food_annual['food_vulnerability_index'] = food_annual['cereal_price_index'] / base_price
food_annual['food_price_inflation_yoy'] = food_annual['cereal_price_index'].pct_change()

relevant_indicators = {
    'External debt stocks, total (DOD, current US$)': 'external_debt_stock',
    'Imports of goods, services and primary income (BoP, current US$)': 'imports',
    'Exports of goods, services and primary income (BoP, current US$)': 'exports',
    'Current account balance (BoP, current US$)': 'current_account'
}
df_debt_filtered = df_debt[df_debt['Indicator Name'].isin(relevant_indicators.keys())].copy()
df_debt_filtered['Indicator Name'] = df_debt_filtered['Indicator Name'].map(relevant_indicators)
# The data is already in long format (Year, Value)
debt_annual = df_debt_filtered.pivot_table(index='Year', columns='Indicator Name', values='Value').reset_index()
debt_annual.rename(columns={'Year': 'year'}, inplace=True)
debt_annual['year'] = debt_annual['year'].astype(int)
debt_annual['debt_growth_rate'] = debt_annual['external_debt_stock'].pct_change()
debt_annual['trade_deficit'] = debt_annual['imports'] - debt_annual['exports']

master_df = pd.merge(master_rain, food_annual, on='year')
master_df = pd.merge(master_df, debt_annual, on='year').dropna()

features = ['annual_rfq_national', 'monsoon_rfq', 'drought_districts_pct', 
            'cereal_price_index', 'food_vulnerability_index', 'food_price_inflation_yoy',
            'imports', 'trade_deficit', 'current_account']
target = 'debt_growth_rate'

scaler = StandardScaler()
X = master_df[features]
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features, index=X.index)
y = master_df[target]

tscv = TimeSeriesSplit(n_splits=3)
ridge = Ridge(alpha=1.0)
rf = RandomForestRegressor(n_estimators=100, random_state=42)

for train_idx, test_idx in tscv.split(X_scaled):
    ridge.fit(X_scaled.iloc[train_idx], y.iloc[train_idx])
    rf.fit(X_scaled.iloc[train_idx], y.iloc[train_idx])

# SHAP/Permutation plot
r = permutation_importance(rf, X_scaled, y, n_repeats=10, random_state=42)
importances = pd.Series(r.importances_mean, index=features).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette='viridis', legend=False)
plt.title('Stage 3: Full Chain Permutation Importances (Random Forest)')
plt.tight_layout()
plt.savefig('production_artifacts/figures/stage3_full_chain_importance.png')
plt.close()

# Risk Profiling
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
master_df['risk_cluster'] = kmeans.fit_predict(X_scaled)
master_df.to_csv('production_artifacts/joint_master_clusters.csv', index=False)

cluster_labels = {0: 'Low-risk stable', 1: 'Rainfall shock', 2: 'Debt stress', 3: 'Crisis (all elevated)'}
master_df['cluster_label'] = master_df['risk_cluster'].map(cluster_labels)

plt.figure(figsize=(12, 6))
sns.scatterplot(data=master_df, x='year', y='debt_growth_rate', hue='cluster_label', palette='deep', s=150)
plt.title('Year Risk Profiling: Vulnerability Timeline')
plt.xlabel('Year')
plt.ylabel('Debt Growth Rate')
plt.tight_layout()
plt.savefig('production_artifacts/figures/stage3_year_risk_timeline.png')
plt.close()

# Baron-Kenny Mediation Test
print("\n--- Baron-Kenny Mediation Test ---")
X_bk1 = sm.add_constant(master_df['annual_rfq_national'])
model1 = sm.OLS(master_df['debt_growth_rate'], X_bk1).fit()
X_bk2 = sm.add_constant(master_df['annual_rfq_national'])
model2 = sm.OLS(master_df['cereal_price_index'], X_bk2).fit()
X_bk3 = sm.add_constant(master_df[['annual_rfq_national', 'cereal_price_index']])
model3 = sm.OLS(master_df['debt_growth_rate'], X_bk3).fit()

coef_rain_m1 = abs(model1.params['annual_rfq_national'])
coef_rain_m3 = abs(model3.params['annual_rfq_national'])
print(f"Direct Rain -> Debt coef: {coef_rain_m1:.6f}")
print(f"Mediation Rain -> Debt coef (with Food): {coef_rain_m3:.6f}")
if coef_rain_m3 < coef_rain_m1:
    print("CONCLUSION: Food price acts as a mediator (Rainfall coefficient reduced in Model 3).")
else:
    print("CONCLUSION: Mediation not strongly supported.")
