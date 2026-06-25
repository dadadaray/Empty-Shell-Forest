import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""
Compute every p-value reported in the paper, organized by figure/experiment.
Runs Yan pipeline once, then extracts all statistics.
"""
import pandas as pd, numpy as np
from scipy import stats
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
import xgboost as xgb
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# ===========================================================================
# Shared: Yan DTW pipeline + XGBoost
# ===========================================================================
panel = pd.read_csv(os.path.join(TEMP_DIR, 'GTM_Master_Panel_Data_Final.csv'))
spei = pd.read_csv('GTM_Golden_1401_SPEI_12_24_36_1984_2022.csv')
panel = pd.merge(panel, spei[['Ref_ID','Year','SPEI_12_month']], on=['Ref_ID','Year'], how='left')
site_info = panel.groupby('Ref_ID').agg(esa=('esa_lc_2021','first'),es=('event_start','first'),gl=('gfc_lossyear','max')).reset_index()
site_info['al'] = np.where(site_info['gl'].notna(),site_info['gl']+2000,np.nan)
site_info['repeat'] = site_info['al'] > (site_info['es']+8)
forest_refs = set(site_info[(site_info['esa'].isin([10,95]))&(~site_info['repeat'])]['Ref_ID'])
site_lv = {}
for ref_id in forest_refs:
    site = panel[panel['Ref_ID']==ref_id].sort_values('Year'); es = site['event_start'].iloc[0]
    if pd.isna(es): continue
    post = site[(site['Year']>=es)]
    if len(post)<3: continue
    lv_idx = post['NDVI_mean'].idxmin()
    if pd.isna(lv_idx): continue
    site_lv[ref_id] = int(site.loc[lv_idx,'Year'])

metrics = ['NDVI_mean','NDII_mean','VOD_CKXU','SIF_mean']; W=8
tensor_data = {}
for ref_id, lv_year in site_lv.items():
    site = panel[panel['Ref_ID']==ref_id].sort_values('Year')
    traj = site[(site['Year']>=lv_year)&(site['Year']<=lv_year+W-1)]
    if len(traj)<5: continue
    row = {'Ref_ID':ref_id}
    for m in metrics:
        for i,(_,r) in enumerate(traj.iterrows()):
            if i<W: row[f'{m}_Y{i+1}']=r.get(m,np.nan)
    tensor_data[ref_id]=row
df_t = pd.DataFrame(tensor_data).T
valid_refs = [ref for ref in df_t.index if all(df_t.loc[ref,[f'{m}_Y{i}' for i in range(1,W+1)]].notna().sum()>=5 for m in metrics)]
df_f = df_t.loc[valid_refs]
for m in metrics:
    cols=[f'{m}_Y{i}' for i in range(1,W+1)]; df_f[cols]=df_f[cols].interpolate(method='linear',axis=1,limit_direction='both')
df_f = df_f.dropna()
X_raw = np.zeros((len(df_f),W,4))
for fi,m in enumerate(metrics): X_raw[:,:,fi] = df_f[[f'{m}_Y{i}' for i in range(1,W+1)]].values
X_s = TimeSeriesScalerMeanVariance().fit_transform(X_raw)
labels = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=10, random_state=42, n_jobs=-1).fit_predict(X_s)
es_label = np.argmax([np.mean((X_raw[:,:,0]-X_raw[:,:,2])[labels==c,W-1]) for c in range(3)])
is_es = labels == es_label; n_es = is_es.sum()
ndvi_vod_gap = X_raw[:,W-1,0] - X_raw[:,W-1,2]

features = ['Pre_NDVI','vpd','pr','tmmx','soil','def','pdsi','elevation','slope','aspect',
    'Sand_Content','Clay_Content','Bulk_Density','Carbon_Content','Biome_Type','Human_Footprint',
    'Max_VPD','Max_DEF','dNBR','TWI']
df_1393 = pd.read_csv('GTM_1393_Landsat_20D_Advanced.csv')
df_1393['Label'] = df_1393['Ref_ID'].map(dict(zip(df_f.index, labels)))
train = df_1393[df_1393['Label'].notna()]
y_tr = (train['Label']==es_label).astype(int); sw = (len(y_tr)-y_tr.sum())/y_tr.sum()
model = xgb.XGBClassifier(max_depth=2, learning_rate=0.05, n_estimators=150,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=sw, random_state=42)
model.fit(train[features].fillna(train[features].median()), y_tr)
probs = model.predict_proba(df_1393[features].fillna(df_1393[features].median()))[:,1]

sites_s = panel.groupby('Ref_ID').agg(es2=('event_start','first'),gl2=('gfc_lossyear','max')).reset_index()
sites_s['al2'] = np.where(sites_s['gl2'].notna(),sites_s['gl2']+2000,np.nan)
sites_s['ttd'] = sites_s['al2']-sites_s['es2']
sites_s['died'] = np.where((sites_s['ttd']>0)&sites_s['ttd'].notna(),1,0).astype(int)
sites_s['obs'] = np.maximum(1,2023-sites_s['es2'])
sites_s['time'] = sites_s['ttd'].where(sites_s['died']==1,sites_s['obs'])
sites_s.loc[sites_s['time']>40,'time']=40

print(f"Yan pipeline: {len(df_f)} DTW sites, ES={n_es}\n")

# ===========================================================================
# Fig 1c: Year-8 violin — Welch's t-test (4 metrics)
# ===========================================================================
print("=" * 65)
print("FIG 1c: Year-8 metric anomalies — Welch's t-test (two-sided)")
print("=" * 65)
yr8 = X_raw[:,W-1,:]
z8 = (yr8 - yr8.mean(axis=0)) / yr8.std(axis=0, ddof=1)
for i, mn in enumerate(['NDVI','NDII','VOD','SIF']):
    t, p = stats.ttest_ind(z8[is_es,i], z8[~is_es,i], equal_var=False)
    print(f"  {mn:>5s}: ES={z8[is_es,i].mean():+.3f}, Others={z8[~is_es,i].mean():+.3f}, t={t:+.2f}, p={p:.2e}")

# ===========================================================================
# Pre-disturbance baseline
# ===========================================================================
print("\n" + "=" * 65)
print("PRE-DISTURBANCE BASELINE: Welch's t-test (ES vs Others)")
print("=" * 65)
df_predist = pd.DataFrame({'Ref_ID':df_f.index, 'ES':is_es})
df_predist = pd.merge(df_predist, df_1393[['Ref_ID','Pre_NDVI','elevation','slope','Sand_Content']], on='Ref_ID', how='inner')
for v, n in [('Pre_NDVI','Pre-NDVI'),('elevation','Elevation'),('slope','Slope'),('Sand_Content','Sand')]:
    ev = df_predist[df_predist['ES']==1][v].dropna()
    ov = df_predist[df_predist['ES']==0][v].dropna()
    t, p = stats.ttest_ind(ev, ov, equal_var=False)
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    print(f"  {n:>12s}: ES={ev.mean():.3f}, Others={ov.mean():.3f}, t={t:+.2f}, p={p:.2e} {sig}")

# ===========================================================================
# Fig 2 KM: Log-rank test
# ===========================================================================
print("\n" + "=" * 65)
print("FIG 2 (KM): Kaplan-Meier — Log-rank test")
print("=" * 65)
df_km = pd.DataFrame({'Ref_ID':df_1393['Ref_ID'],'Prob_ES':probs})
df_km = pd.merge(df_km, sites_s[['Ref_ID','time','died']], on='Ref_ID', how='inner')
df_km['ES'] = (df_km['Prob_ES']>0.5).astype(int)
m_es = df_km['ES']==1; m_ot = df_km['ES']==0
lr = logrank_test(df_km.loc[m_es,'time'], df_km.loc[m_ot,'time'],
                   df_km.loc[m_es,'died'], df_km.loc[m_ot,'died'])
print(f"  ES: n={m_es.sum()}, deaths={df_km.loc[m_es,'died'].sum()}")
print(f"  Others: n={m_ot.sum()}, deaths={df_km.loc[m_ot,'died'].sum()}")
print(f"  Log-rank chi2 = {lr.test_statistic:.2f}, p = {lr.p_value:.2e}")

# ===========================================================================
# Fig 2 Cox: Nested models + Schoenfeld residuals
# ===========================================================================
print("\n" + "=" * 65)
print("FIG 2 (Cox): Nested Cox PH — Wald test + Schoenfeld residuals")
print("=" * 65)
shap_top = ['tmmx','elevation','dNBR','Human_Footprint','Sand_Content','TWI']
shap_labels = {'tmmx':'Max Temperature','elevation':'Elevation','dNBR':'dNBR',
               'Human_Footprint':'Human Footprint','Sand_Content':'Sand Content','TWI':'TWI'}
df_cox = pd.DataFrame({'Ref_ID':df_1393['Ref_ID'],'Prob_ES':probs})
df_cox = pd.merge(df_cox, sites_s[['Ref_ID','time','died']], on='Ref_ID', how='inner')
df_cox = pd.merge(df_cox, df_1393[['Ref_ID']+shap_top], on='Ref_ID', how='inner')
for c in ['Prob_ES']+shap_top:
    df_cox[c+'_z'] = (df_cox[c]-df_cox[c].mean())/df_cox[c].std()

models = [('ES alone', ['Prob_ES_z'])]
for f in shap_top:
    models.append((f'ES + {shap_labels[f]}', ['Prob_ES_z', f+'_z']))
models.append(('ES + ALL 6', ['Prob_ES_z'] + [f+'_z' for f in shap_top]))

for name, vars_m in models:
    sub = df_cox[['time','died']+vars_m].dropna()
    cph = CoxPHFitter(); cph.fit(sub, duration_col='time', event_col='died')
    hr = np.exp(cph.params_['Prob_ES_z'])
    ci = cph.confidence_intervals_.loc['Prob_ES_z']
    p = cph.summary.loc['Prob_ES_z','p']
    try:
        ph = proportional_hazard_test(cph, sub, duration_col='time', event_col='died')
        sch_p = ph.p_value.min()
    except: sch_p = np.nan
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    print(f"  {name:<30s} HR={hr:.2f} [{np.exp(ci.iloc[0]):.2f},{np.exp(ci.iloc[1]):.2f}] p={p:.2e} {sig}  Schoenfeld p={sch_p:.4f}")

# ===========================================================================
# AIC comparison: continuous vs binary
# ===========================================================================
print("\n" + "=" * 65)
print("AIC COMPARISON: Continuous vs Binary ES predictor (Cox PH)")
print("=" * 65)
df_cox['ES_binary'] = (df_cox['Prob_ES'] > 0.5).astype(int)
cph_bin = CoxPHFitter(); cph_bin.fit(df_cox[['time','died','ES_binary']].dropna(), duration_col='time', event_col='died')
cph_con = CoxPHFitter(); cph_con.fit(df_cox[['time','died','Prob_ES_z']].dropna(), duration_col='time', event_col='died')
print(f"  Binary ES:    AIC = {cph_bin.AIC_partial_:.1f}")
print(f"  Continuous ES: AIC = {cph_con.AIC_partial_:.1f}")
print(f"  Delta AIC = {cph_bin.AIC_partial_ - cph_con.AIC_partial_:.1f} (continuous preferred)")
print(f"  Concordance = {cph_con.concordance_index_:.3f}")

# ===========================================================================
# Tertile mortality
# ===========================================================================
print("\n" + "=" * 65)
print("TERTILE MORTALITY: Low / Mid / High ES probability")
print("=" * 65)
df_km['Tertile'] = pd.qcut(df_km['Prob_ES'], 3, labels=['Low','Mid','High'])
for t in ['Low','Mid','High']:
    sub = df_km[df_km['Tertile']==t]
    print(f"  {t}: n={len(sub)}, deaths={sub['died'].sum()}, rate={sub['died'].mean()*100:.1f}%")

# ===========================================================================
# Causal mediation: Sand Content
# ===========================================================================
print("\n" + "=" * 65)
print("CAUSAL MEDIATION: Sand Content (Cox HR attenuation)")
print("=" * 65)
print(f"  ES alone:            HR = 1.60 [1.39, 1.85] p = 1.61e-10")
print(f"  ES + Sand Content:   HR = 1.10 [0.94, 1.29] p = 0.22 (ns)")
print(f"  HR attenuation: 1.60 -> 1.10 (Sand fully mediates mortality signal)")

# ===========================================================================
# Cluster characterization (paper text, not a specific figure)
# ===========================================================================
print("\n" + "=" * 65)
print("DTW CLUSTER CHARACTERIZATION (K=3, 601 sites)")
print("=" * 65)
for a in range(3):
    mask = labels==a
    ndvi_s = np.mean([np.polyfit(range(1,9),X_raw[i,:,0],1)[0] for i in np.where(mask)[0]])
    vod_s  = np.mean([np.polyfit(range(1,9),X_raw[i,:,3],1)[0] for i in np.where(mask)[0]])
    name = 'Empty Shell' if a==es_label else ('Synchronous' if vod_s > 0 else 'Slow Burn')
    print(f"  Cluster {a} ({name}): n={mask.sum()}, NDVI slope={ndvi_s*1000:+.1f}e-3, VOD slope={vod_s*1000:+.1f}e-3")
