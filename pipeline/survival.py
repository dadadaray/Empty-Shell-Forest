import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Kaplan-Meier survival analysis: ES vs Others, log-rank test. Paper Methods 5.4."""
import pandas as pd, numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# Load pre-computed Yan pipeline predictions
df = pd.read_csv(os.path.join(TEMP_DIR, 'Yan_XGBoost_Predictions_1393.csv'))
panel = pd.read_csv(os.path.join(TEMP_DIR, 'GTM_Master_Panel_Data_Final.csv'))

sites = panel.groupby('Ref_ID').agg(
    es=('event_start','first'), gl=('gfc_lossyear','max')
).reset_index()
sites['al'] = np.where(sites['gl'].notna(), sites['gl']+2000, np.nan)
sites['ttd'] = sites['al'] - sites['es']
sites['died'] = np.where((sites['ttd']>0)&sites['ttd'].notna(), 1, 0).astype(int)
sites['obs'] = np.maximum(1, 2023 - sites['es'])
sites['time'] = sites['ttd'].where(sites['died']==1, sites['obs'])
sites.loc[sites['time']>40, 'time'] = 40

df = pd.merge(df, sites[['Ref_ID','time','died']], on='Ref_ID', how='inner')
df['ES'] = (df['Prob_ES'] > 0.5).astype(int)

m_es = df['ES'] == 1; m_ot = df['ES'] == 0
kmf_es = KaplanMeierFitter(); kmf_ot = KaplanMeierFitter()
kmf_es.fit(df.loc[m_es,'time'], df.loc[m_es,'died'])
kmf_ot.fit(df.loc[m_ot,'time'], df.loc[m_ot,'died'])

lr = logrank_test(df.loc[m_es,'time'], df.loc[m_ot,'time'],
                  df.loc[m_es,'died'], df.loc[m_ot,'died'])

print(f"Empty Shell: n={m_es.sum()}, deaths={df.loc[m_es,'died'].sum()}")
print(f"Others:      n={m_ot.sum()}, deaths={df.loc[m_ot,'died'].sum()}")
print(f"Log-rank chi2 = {lr.test_statistic:.2f}, p = {lr.p_value:.2e}")
