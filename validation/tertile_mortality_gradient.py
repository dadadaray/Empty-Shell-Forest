import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Tertile-stratified mortality analysis: ES probability gradient vs secondary mortality risk.
Supports paper Results: high-ES-probability sites have elevated mortality vs mid-tertile."""
import pandas as pd, numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

# Load Yan pipeline predictions and build survival data
df = pd.read_csv(os.path.join(TEMP_DIR, 'Yan_XGBoost_Predictions_1393.csv'))
panel = pd.read_csv(os.path.join(TEMP_DIR, 'GTM_Master_Panel_Data_Final.csv'))

sites = panel.groupby('Ref_ID').agg(
    event_start=('event_start', 'first'), gfc_lossyear=('gfc_lossyear', 'max')
).reset_index()
sites['actual_loss'] = np.where(sites['gfc_lossyear'].notna(), sites['gfc_lossyear'] + 2000, np.nan)
sites['ttd'] = sites['actual_loss'] - sites['event_start']
sites['died'] = np.where((sites['ttd'] > 0) & sites['ttd'].notna(), 1, 0).astype(int)
sites['obs'] = np.maximum(1, 2023 - sites['event_start'])
sites['time'] = sites['ttd'].where(sites['died'] == 1, sites['obs'])
sites.loc[sites['time'] > 40, 'time'] = 40

df = pd.merge(df, sites[['Ref_ID', 'time', 'died']], on='Ref_ID', how='inner')
df['Prob_ES_z'] = (df['Prob_ES'] - df['Prob_ES'].mean()) / df['Prob_ES'].std()
df['ES_Tertile'] = pd.qcut(df['Prob_ES'], 3, labels=['Low ES', 'Mid ES', 'High ES'])

# === 1. Tertile mortality rates ===
print('TERTILE MORTALITY BREAKDOWN:')
tertile_data = []
for t in ['Low ES', 'Mid ES', 'High ES']:
    m = df['ES_Tertile'] == t
    d = {'tertile': t, 'n': m.sum(), 'deaths': df.loc[m, 'died'].sum()}
    d['rate'] = d['deaths'] / d['n'] * 100
    d['mean_obs'] = df.loc[m, 'time'].mean()
    tertile_data.append(d)
    print(f"  {t}: {d['deaths']}/{d['n']} = {d['rate']:.1f}%, mean obs = {d['mean_obs']:.0f} yr")

# === 2. Cox PH: continuous Prob_ES (Z-standardized) ===
cph = CoxPHFitter()
cph.fit(df[['time', 'died', 'Prob_ES_z']].dropna(), duration_col='time', event_col='died')
hr = np.exp(cph.params_['Prob_ES_z'])
ci = np.exp(cph.confidence_intervals_.loc['Prob_ES_z'])
p = cph.summary.loc['Prob_ES_z', 'p']
print(f'\nCox PH (continuous Prob_ES): HR = {hr:.2f} [{ci.iloc[0]:.2f}, {ci.iloc[1]:.2f}], p = {p:.2e}')

# === 3. KM log-rank test: High vs Low ES tertile ===
m_high = df['ES_Tertile'] == 'High ES'
m_low = df['ES_Tertile'] == 'Low ES'
lr = logrank_test(df.loc[m_high, 'time'], df.loc[m_low, 'time'],
                  df.loc[m_high, 'died'], df.loc[m_low, 'died'])
print(f'\nLog-rank test (High vs Low ES tertile): p = {lr.p_value:.2e}')

# === 4. Print KM survival summary by tertile ===
print('\nKAPLAN-MEIER SURVIVAL SUMMARY BY ES PROBABILITY TERTILE:')
colors = ['#2ca02c', '#ff7f0e', '#d62728']
for i, t in enumerate(['Low ES', 'Mid ES', 'High ES']):
    m = df['ES_Tertile'] == t
    kmf = KaplanMeierFitter()
    kmf.fit(df.loc[m, 'time'], df.loc[m, 'died'])
    median = kmf.median_survival_time_
    print(f"  {t} (n={m.sum()}, {df.loc[m,'died'].sum()} deaths): "
          f"median survival = {median:.1f} yr, "
          f"5-yr survival = {kmf.survival_function_at_times(5).values[0]:.3f}, "
          f"10-yr survival = {kmf.survival_function_at_times(10).values[0]:.3f}")

# === 5. Print mortality bar summary ===
print('\nMORTALITY BY ES PROBABILITY TERTILE:')
for i, d in enumerate(tertile_data):
    print(f"  {d['tertile']}: {d['rate']:.1f}% ({d['deaths']}/{d['n']})")

print(f'\nCox HR = {hr:.2f} [{ci.iloc[0]:.2f}, {ci.iloc[1]:.2f}], p = {p:.2e}')
print('Analysis complete.')