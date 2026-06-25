import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Fig 3c: Cox PH Forest Plot — 8 models: ES alone + 6 SHAP features + ALL."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
import xgboost as xgb
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],
    'font.size':14,'axes.labelsize':14,'axes.labelweight':'bold',
    'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':14,
    'axes.linewidth':1.2,'pdf.fonttype':42,'ps.fonttype':42
})

# === Yan pipeline ===
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

sites = panel.groupby('Ref_ID').agg(es=('event_start','first'),gl=('gfc_lossyear','max')).reset_index()
sites['al'] = np.where(sites['gl'].notna(),sites['gl']+2000,np.nan)
sites['ttd'] = sites['al']-sites['es']
sites['died'] = np.where((sites['ttd']>0)&sites['ttd'].notna(),1,0).astype(int)
sites['obs'] = np.maximum(1,2023-sites['es'])
sites['time'] = sites['ttd'].where(sites['died']==1,sites['obs'])
sites.loc[sites['time']>40,'time']=40

df_cox = pd.DataFrame({'Ref_ID':df_1393['Ref_ID'],'Prob_ES':probs})
df_cox = pd.merge(df_cox, sites[['Ref_ID','time','died']], on='Ref_ID', how='inner')

# SHAP top-6 physical features
shap_top = ['tmmx','elevation','dNBR','Human_Footprint','Sand_Content','TWI']
shap_labels = {'tmmx':'Max Temperature','elevation':'Elevation','dNBR':'dNBR',
               'Human_Footprint':'Human Footprint','Sand_Content':'Sand Content','TWI':'TWI'}

df_cox = pd.merge(df_cox, df_1393[['Ref_ID']+shap_top], on='Ref_ID', how='inner')
for c in ['Prob_ES']+shap_top:
    df_cox[c+'_z'] = (df_cox[c]-df_cox[c].mean())/df_cox[c].std()

# 8 models
models = [('ES alone', ['Prob_ES_z'])]
for f in shap_top:
    models.append((f'ES + {shap_labels[f]}', ['Prob_ES_z', f+'_z']))
models.append(('ES + ALL 6', ['Prob_ES_z'] + [f+'_z' for f in shap_top]))

hrs, ci_lows, ci_highs, pvals, labels_m = [], [], [], [], []
for name, vars_m in models:
    sub = df_cox[['time','died']+vars_m].dropna()
    cph = CoxPHFitter(); cph.fit(sub, duration_col='time', event_col='died')
    hr = np.exp(cph.params_['Prob_ES_z'])
    ci = cph.confidence_intervals_.loc['Prob_ES_z']
    p = cph.summary.loc['Prob_ES_z','p']
    hrs.append(hr); ci_lows.append(np.exp(ci.iloc[0])); ci_highs.append(np.exp(ci.iloc[1]))
    pvals.append(p); labels_m.append(name)

# === FOREST PLOT (9x7, matching Fig3b) ===
fig, ax = plt.subplots(figsize=(9, 7))
y_positions = list(range(len(models)-1, -1, -1))

for i in range(len(models)):
    hr, lo, hi, p = hrs[i], ci_lows[i], ci_highs[i], pvals[i]
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'

    # Color: red for ES alone, dark red for significant, gray for ns
    if i == 0:
        color, edgecolor = '#d73027', 'white'
    elif p < 0.05:
        color, edgecolor = '#fc8d59', 'none'
    else:
        color, edgecolor = '#B0BEC5', 'none'

    ax.errorbar(hr, y_positions[i], xerr=[[hr-lo],[hi-hr]],
                fmt='o', color=color, markersize=11, capsize=5, capthick=2, linewidth=2.5,
                markeredgecolor=edgecolor, markeredgewidth=1.5 if edgecolor!='none' else 0)
    ax.text(hr+0.06, y_positions[i]+0.08, f'{hr:.2f} {sig}',
            fontweight='bold', fontsize=12, color=color)
    ax.text(0.30, y_positions[i], labels_m[i], ha='right', fontsize=12, fontweight='bold',
            color='#333333')

ax.axvline(1.0, color='gray', ls='--', lw=1.5, alpha=0.5)
ax.set_xlabel('Hazard Ratio (95% CI)', fontweight='bold')
ax.set_yticks([])
ax.set_xlim(0.2, 2.8)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.grid(axis='x', linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.join(TEMP_DIR, 'Fig3c_Cox_Forest.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {TEMP_DIR}/Fig3c_Cox_Forest.pdf')
for name, hr, lo, hi, p in zip(labels_m, hrs, ci_lows, ci_highs, pvals):
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    print(f'  {name:<30s} HR={hr:.2f} [{lo:.2f}-{hi:.2f}] p={p:.4f} {sig}')
