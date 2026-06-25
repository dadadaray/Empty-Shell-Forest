import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Fig 3b: KM Survival — Yan-pipeline ES labels."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
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
decoupling = X_raw[:,:,0]-X_raw[:,:,2]
es_label = np.argmax([np.mean(decoupling[labels==c,W-1]) for c in range(3)])

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

df = pd.DataFrame({'Ref_ID':df_1393['Ref_ID'],'Prob_ES':probs})
df = pd.merge(df, sites[['Ref_ID','time','died']], on='Ref_ID', how='inner')
df['ES'] = (df['Prob_ES']>0.5).astype(int)

fig, ax = plt.subplots(figsize=(9, 7))
kmf_es = KaplanMeierFitter(); kmf_ot = KaplanMeierFitter()
m_es = df['ES']==1; m_ot = df['ES']==0
kmf_es.fit(df.loc[m_es,'time'], df.loc[m_es,'died'], label='Empty Shell (n=%d, %d deaths)'%(m_es.sum(), df.loc[m_es,'died'].sum()))
kmf_ot.fit(df.loc[m_ot,'time'], df.loc[m_ot,'died'], label='Others (n=%d, %d deaths)'%(m_ot.sum(), df.loc[m_ot,'died'].sum()))
kmf_es.plot_survival_function(ax=ax, color='#d73027', linewidth=3)
kmf_ot.plot_survival_function(ax=ax, color='#B0BEC5', linewidth=3)
lr = logrank_test(df.loc[m_es,'time'], df.loc[m_ot,'time'], df.loc[m_es,'died'], df.loc[m_ot,'died'])
ax.text(25, 0.82, 'Log-rank p = %.2e'%lr.p_value, fontsize=13, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
ax.set_xlabel('Years Since Disturbance', fontweight='bold')
ax.set_ylabel('Survival Probability', fontweight='bold')
ax.set_ylim(0.70, 1.02)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(fontsize=13, frameon=False); ax.grid(linestyle=':', alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(TEMP_DIR, 'Fig3b_KM_Survival.pdf'), dpi=300, bbox_inches='tight'); plt.close()
print(f'Saved: {TEMP_DIR}/Fig3b_KM_Survival.pdf (log-rank p=%.2e, ES_n=%d)'%(lr.p_value, m_es.sum()))
