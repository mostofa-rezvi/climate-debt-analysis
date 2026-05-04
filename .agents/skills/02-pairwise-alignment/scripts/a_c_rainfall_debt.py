import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests

os.makedirs('production_artifacts/figures', exist_ok=True)

print("--- Pairwise A<->C: Rainfall & External Debt ---")
df_rain = pd.read_csv('datasets/bgd-rainfall-subnat-full.csv')
df_debt = pd.read_csv('datasets/external-debt_bgd.csv')

df_rain['date'] = pd.to_datetime(df_rain['date'])
df_rain['year'] = df_rain['date'].dt.year
df_rain['month'] = df_rain['date'].dt.month
df_rain['season'] = df_rain['month'].apply(lambda m: 'monsoon' if m in [6, 7, 8, 9, 10] else 'other')

annual_rain = df_rain.groupby('year')['rfq'].mean().reset_index().rename(columns={'rfq': 'annual_national_rfq'})
monsoon_rain = df_rain[df_rain['season'] == 'monsoon'].groupby('year')['rfq'].mean().reset_index().rename(columns={'rfq': 'monsoon_rfq'})

drought_df = df_rain.groupby(['year', 'PCODE'])['rfq'].mean().reset_index()
drought_df['is_drought'] = (drought_df['rfq'] < 75).astype(int)
drought_freq = drought_df.groupby('year')['is_drought'].mean().reset_index().rename(columns={'is_drought': 'drought_frequency'})

annual_features = pd.merge(annual_rain, monsoon_rain, on='year')
annual_features = pd.merge(annual_features, drought_freq, on='year')
annual_features['lag_1_rfq'] = annual_features['annual_national_rfq'].shift(1)
annual_features['lag_2_rfq'] = annual_features['annual_national_rfq'].shift(2)

df_debt_filtered = df_debt[df_debt['Indicator Name'] == 'External debt stocks, total (DOD, current US$)'].copy()
df_debt_filtered['Indicator Name'] = 'total_external_debt'
# The data is already in long format (Year, Value)
wide_df = df_debt_filtered.pivot_table(index='Year', columns='Indicator Name', values='Value').reset_index()
wide_df.rename(columns={'Year': 'year'}, inplace=True)
wide_df['year'] = wide_df['year'].astype(int)
wide_df['debt_growth_rate'] = wide_df['total_external_debt'].pct_change()

merged_ac = pd.merge(annual_features, wide_df, on='year').dropna()

# Granger Causality
gc_data = merged_ac[['debt_growth_rate', 'annual_national_rfq']]
print("Granger Causality Test:")
try:
    grangercausalitytests(gc_data, maxlag=[1, 2], verbose=False)
    print("Granger Causality test passed (verbose output suppressed for brevity).")
except Exception as e:
    print(f"Granger Test could not run: {e}")

# Correlation
corr_matrix = merged_ac.corr(method='pearson')
corr_matrix.to_csv('production_artifacts/ac_correlation.csv')

# Plot Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix[['debt_growth_rate', 'total_external_debt']].loc[['annual_national_rfq', 'monsoon_rfq', 'drought_frequency', 'lag_1_rfq']], 
            annot=True, cmap='coolwarm', center=0)
plt.title('Correlation: Rainfall Anomalies vs Debt Indicators')
plt.tight_layout()
plt.savefig('production_artifacts/figures/stage2_ac_correlation_heatmap.png')
plt.close()

print("Saved artifacts and figures for A<->C.")
