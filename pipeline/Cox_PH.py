import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Cox Proportional Hazards: nested models testing ES mortality signal with progressive confounder adjustment.
Paper Methods 5.4, Fig 2 Cox forest plot."""
import pandas as pd, numpy as np
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
import xgboost as xgb
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# --- Yan pipeline (DTW + XGBoost) ---
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

features = ['Pre_NDVI','vpd','pr','tmmx','soil','def','pdsi','elevation','slope','aspect',
    'Sand_Content','Clay_Content','Bulk_Density','Carbon_Content','Biome_Type','Human_Footprint',
    'Max_VPD','Max_DEF','dNBR','TWI']
df_1393 = pd.read_csv('GTM_1393_Landsat_20D_Advanced.csv')
df_1393['Label'] = df_1393['Ref_ID'].map(dict(zip(df_f.index, labels)))
train = df_1393[df_1393['Label'].notna()]
y_tr = (train['Label']==es_label).astype(int).values; sw = (len(y_tr)-y_tr.sum())/y_tr.sum()
model = xgb.XGBClassifier(max_depth=2, learning_rate=0.05, n_estimators=150,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=sw, random_state=42)
model.fit(train[features].fillna(train[features].median()), y_tr)
probs = model.predict_proba(df_1393[features].fillna(df_1393[features].median()))[:,1]

# --- Survival data ---
sites = panel.groupby('Ref_ID').agg(es=('event_start','first'),gl=('gfc_lossyear','max')).reset_index()
sites['al'] = np.where(sites['gl'].notna(),sites['gl']+2000,np.nan)
sites['ttd'] = sites['al']-sites['es']
sites['died'] = np.where((sites['ttd']>0)&sites['ttd'].notna(),1,0).astype(int)
sites['obs'] = np.maximum(1,2023-sites['es'])
sites['time'] = sites['ttd'].where(sites['died']==1,sites['obs'])
sites.loc[sites['time']>40,'time']=40

df_cox = pd.DataFrame({'Ref_ID':df_1393['Ref_ID'],'Prob_ES':probs})
df_cox = pd.merge(df_cox, sites[['Ref_ID','time','died']], on='Ref_ID', how='inner')

# --- Nested Cox models: ES alone + progressive confounder adjustment ---
shap_top = ['tmmx','elevation','dNBR','Human_Footprint','Sand_Content','TWI']
shap_labels = {'tmmx':'Max Temperature','elevation':'Elevation','dNBR':'dNBR',
               'Human_Footprint':'Human Footprint','Sand_Content':'Sand Content','TWI':'TWI'}

df_cox = pd.merge(df_cox, df_1393[['Ref_ID']+shap_top], on='Ref_ID', how='inner')
for c in ['Prob_ES']+shap_top:
    df_cox[c+'_z'] = (df_cox[c]-df_cox[c].mean())/df_cox[c].std()

models = [('ES alone', ['Prob_ES_z'])]
for f in shap_top:
    models.append((f'ES + {shap_labels[f]}', ['Prob_ES_z', f+'_z']))
models.append(('ES + ALL 6', ['Prob_ES_z'] + [f+'_z' for f in shap_top]))

print(f"{'Model':<30s} {'HR':>6s} {'95% CI':<16s} {'p-value':<12s} {'Schoenfeld p':>14s}")
print('-' * 85)
for name, vars_m in models:
    sub = df_cox[['time','died']+vars_m].dropna()
    cph = CoxPHFitter(); cph.fit(sub, duration_col='time', event_col='died')
    hr = np.exp(cph.params_['Prob_ES_z'])
    ci = cph.confidence_intervals_.loc['Prob_ES_z']
    p = cph.summary.loc['Prob_ES_z','p']
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'

    # Schoenfeld residual test (proportional hazards assumption)
    try:
        ph_test = proportional_hazard_test(cph, sub, duration_col='time', event_col='died')
        sch_p = ph_test.p_value.min()  # worst-case p across all covariates
    except:
        sch_p = np.nan

    print(f'{name:<30s} {hr:6.2f} [{np.exp(ci.iloc[0]):.2f}, {np.exp(ci.iloc[1]):.2f}]  p={p:.2e} {sig}   {sch_p:>12.4f}')

# --- AIC comparison: continuous vs binary ES predictor ---
print("\n--- AIC comparison: continuous vs binary ES predictor ---")
# Binary model
df_cox['ES_binary'] = (df_cox['Prob_ES'] > 0.5).astype(int)
cph_bin = CoxPHFitter(); cph_bin.fit(df_cox[['time','died','ES_binary']].dropna(), duration_col='time', event_col='died')
aic_bin = cph_bin.AIC_partial_
# Continuous model
cph_con = CoxPHFitter(); cph_con.fit(df_cox[['time','died','Prob_ES_z']].dropna(), duration_col='time', event_col='died')
aic_con = cph_con.AIC_partial_
delta_aic = aic_bin - aic_con
concordance = cph_con.concordance_index_
print(f"  Binary ES (Prob>0.5):    AIC = {aic_bin:.1f}")
print(f"  Continuous ES (Z-scored): AIC = {aic_con:.1f}")
print(f"  Delta AIC = {delta_aic:.1f} (negative = continuous preferred)")
print(f"  Concordance index = {concordance:.3f}")
