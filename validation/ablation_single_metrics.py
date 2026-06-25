import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Metric ablation: drop one satellite metric at a time, recluster K=3, compare to full 4-metric DTW result.
Uses Yan pipeline 601-site tensor. Paper: Empty Shell identification is robust to single-metric removal."""
import pandas as pd, numpy as np
from sklearn.metrics import adjusted_rand_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# === Yan pipeline: build 601-site tensor ===
panel = pd.read_csv(os.path.join(TEMP_DIR, 'GTM_Master_Panel_Data_Final.csv'))
spei = pd.read_csv('GTM_Golden_1401_SPEI_12_24_36_1984_2022.csv')
panel = pd.merge(panel, spei[['Ref_ID','Year','SPEI_12_month']], on=['Ref_ID','Year'], how='left')

site_info = panel.groupby('Ref_ID').agg(
    esa=('esa_lc_2021','first'), es=('event_start','first'), gl=('gfc_lossyear','max')
).reset_index()
site_info['al'] = np.where(site_info['gl'].notna(), site_info['gl']+2000, np.nan)
site_info['repeat'] = site_info['al'] > (site_info['es']+8)
forest_refs = set(site_info[(site_info['esa'].isin([10,95])) & (~site_info['repeat'])]['Ref_ID'])

site_lv = {}
for ref_id in forest_refs:
    site = panel[panel['Ref_ID']==ref_id].sort_values('Year'); es = site['event_start'].iloc[0]
    if pd.isna(es): continue
    post = site[(site['Year']>=es)]
    if len(post)<3: continue
    lv_idx = post['NDVI_mean'].idxmin()
    if pd.isna(lv_idx): continue
    site_lv[ref_id] = int(site.loc[lv_idx,'Year'])

metrics_all = ['NDVI_mean','NDII_mean','VOD_CKXU','SIF_mean']; W=8
tensor_data = {}
for ref_id, lv_year in site_lv.items():
    site = panel[panel['Ref_ID']==ref_id].sort_values('Year')
    traj = site[(site['Year']>=lv_year)&(site['Year']<=lv_year+W-1)]
    if len(traj)<5: continue
    row = {'Ref_ID':ref_id}
    for m in metrics_all:
        for i,(_,r) in enumerate(traj.iterrows()):
            if i<W: row[f'{m}_Y{i+1}'] = r.get(m,np.nan)
    tensor_data[ref_id] = row

df_t = pd.DataFrame(tensor_data).T
valid_refs = [ref for ref in df_t.index if all(
    df_t.loc[ref,[f'{m}_Y{i}' for i in range(1,W+1)]].notna().sum()>=5 for m in metrics_all)]
df_f = df_t.loc[valid_refs]
for m in metrics_all:
    cols=[f'{m}_Y{i}' for i in range(1,W+1)]
    df_f[cols]=df_f[cols].interpolate(method='linear',axis=1,limit_direction='both')
df_f = df_f.dropna()
print(f"DTW sites: {len(df_f)}")

# Build full 4-metric tensor
N = len(df_f)
X_full = np.zeros((N, W, 4))
for fi, m in enumerate(metrics_all):
    X_full[:,:,fi] = df_f[[f'{m}_Y{i}' for i in range(1,W+1)]].values
X_full_s = TimeSeriesScalerMeanVariance().fit_transform(X_full)

# Baseline: K=3 with all 4 metrics
km = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=10, random_state=42, n_jobs=-1)
labels_full = km.fit_predict(X_full_s)

# Identify ES label
es_label = np.argmax([np.mean((X_full[:,:,0]-X_full[:,:,2])[labels_full==c,W-1]) for c in range(3)])
es_full = (labels_full == es_label).astype(int)
print(f"Full 4-metric: ES label={es_label}, n_ES={es_full.sum()}")

# Drop one metric at a time, recluster, compare
print(f"\n{'Metric dropped':<15s} {'ARI':>8s} {'ES agree':>10s} {'Impact':>12s}")
print('-' * 50)
for drop in metrics_all:
    keep = [m for m in metrics_all if m != drop]
    X_drop = np.zeros((N, W, 3))
    for fi, m in enumerate(keep):
        X_drop[:,:,fi] = df_f[[f'{m}_Y{i}' for i in range(1,W+1)]].values
    X_drop_s = TimeSeriesScalerMeanVariance().fit_transform(X_drop)
    labels_drop = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=10, random_state=42, n_jobs=-1).fit_predict(X_drop_s)

    es_label_drop = np.argmax([np.mean((X_drop[:,:,0]-X_drop[:,:,2])[labels_drop==c,W-1]) for c in range(3)])
    es_drop = (labels_drop == es_label_drop).astype(int)
    es_agree = (es_full == es_drop).mean()

    ari = adjusted_rand_score(labels_full, labels_drop)
    impact = 'critical' if ari < 0.70 else 'important' if ari < 0.85 else 'modest' if ari < 0.95 else 'negligible'
    print(f'{drop:<15s} {ari:8.3f} {es_agree:10.3f} {impact:>12s}')