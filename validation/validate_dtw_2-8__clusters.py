import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""
Direct cluster validation on the 601-site Yan pipeline dataset.
Loads GTM_Master_Panel_Data_Final.csv + SPEI, builds tensor with NDVI-minimum-based
recovery onset, runs K=3 DTW clustering, and validates with silhouette scores.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import cdist_dtw
import warnings
warnings.filterwarnings('ignore')

# ========================================
# 1. Load Yan pipeline data
# ========================================
print("Loading GTM_Master_Panel_Data_Final.csv + SPEI...")
panel = pd.read_csv(os.path.join(TEMP_DIR, 'GTM_Master_Panel_Data_Final.csv'))
spei = pd.read_csv('GTM_Golden_1401_SPEI_12_24_36_1984_2022.csv')
panel = pd.merge(panel, spei[['Ref_ID','Year','SPEI_12_month']], on=['Ref_ID','Year'], how='left')

# Site info and forest filter
site_info = panel.groupby('Ref_ID').agg(
    esa=('esa_lc_2021','first'),
    es=('event_start','first'),
    gl=('gfc_lossyear','max')
).reset_index()
site_info['al'] = np.where(site_info['gl'].notna(), site_info['gl']+2000, np.nan)
site_info['repeat'] = site_info['al'] > (site_info['es']+8)
forest_refs = set(site_info[(site_info['esa'].isin([10,95])) & (~site_info['repeat'])]['Ref_ID'])

# NDVI minimum year for recovery onset
site_lv = {}
for ref_id in forest_refs:
    site = panel[panel['Ref_ID']==ref_id].sort_values('Year')
    es = site['event_start'].iloc[0]
    if pd.isna(es):
        continue
    post = site[(site['Year']>=es)]
    if len(post)<3:
        continue
    lv_idx = post['NDVI_mean'].idxmin()
    if pd.isna(lv_idx):
        continue
    site_lv[ref_id] = int(site.loc[lv_idx, 'Year'])

print(f"  Forest sites with NDVI-minimum onset: {len(site_lv)}")

# ========================================
# 2. Build 4-metric tensor (W=8, aligned to NDVI minimum)
# ========================================
metrics = ['NDVI_mean', 'NDII_mean', 'SIF_mean', 'VOD_CKXU']
tensor_data = {}
for ref_id, lv_year in site_lv.items():
    site = panel[panel['Ref_ID']==ref_id].sort_values('Year')
    traj = site[(site['Year']>=lv_year) & (site['Year']<=lv_year+7)]
    if len(traj)<5:
        continue
    row = {'Ref_ID': ref_id}
    for m in metrics:
        for i, (_, r) in enumerate(traj.iterrows()):
            if i<8:
                row[f'{m}_Y{i+1}'] = r.get(m, np.nan)
    tensor_data[ref_id] = row

df_t = pd.DataFrame(tensor_data).T
valid_refs = [ref for ref in df_t.index if all(
    df_t.loc[ref, [f'{m}_Y{i}' for i in range(1,9)]].notna().sum()>=5 for m in metrics)]
df_f = df_t.loc[valid_refs]
for m in metrics:
    cols = [f'{m}_Y{i}' for i in range(1,9)]
    df_f[cols] = df_f[cols].interpolate(method='linear', axis=1, limit_direction='both')
df_f = df_f.dropna()

N = len(df_f)
X_3d = np.zeros((N, 8, 4))
for f_idx, m in enumerate(metrics):
    X_3d[:, :, f_idx] = df_f[[f'{m}_Y{i}' for i in range(1,9)]].values

print(f"  Tensor: {X_3d.shape} (samples x 8 years x 4 metrics)")

# Z-score normalize
X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X_3d)
N = len(X_scaled)

# ========================================
# 3. DTW distance matrix
# ========================================
print("Computing DTW distance matrix...")
D = cdist_dtw(X_scaled, n_jobs=-1)
D = np.abs(D)
D = (D + D.T) / 2
np.fill_diagonal(D, 0)
print(f"  Done: {D.shape}")

# ========================================
# 4. Silhouette sweep K=2..8
# ========================================
print("\nSilhouette sweep (DTW distance, K=2..8)...")
K_range = range(2, 9)
best_k = 2
best_sil = -1
all_sils = []

for k in K_range:
    dtw = TimeSeriesKMeans(n_clusters=k, metric="dtw", max_iter=10, random_state=42, n_jobs=-1)
    labels = dtw.fit_predict(X_scaled)
    sil = silhouette_score(D, labels, metric='precomputed')
    sizes = np.bincount(labels)
    all_sils.append(sil)
    print(f"  K={k} | Silhouette={sil:.4f} | Sizes={sizes}")
    if sil > best_sil:
        best_sil = sil
        best_k = k

print(f"\n  Best K: {best_k} (silhouette={best_sil:.4f})")

# ========================================
# 5. Run K=3 for detailed inspection
# ========================================
k3 = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=10, random_state=42, n_jobs=-1)
labels_3 = k3.fit_predict(X_scaled)
print(f"\n  K=3 sizes: {np.bincount(labels_3)}")

# ========================================
# 6. Print cluster summaries
# ========================================
print("\n" + "="*60)
print("CLUSTER CENTROID SUMMARY (K=3)")
print("="*60)
metric_names = ['NDVI (Greenness)', 'NDII (Canopy Water)', 'SIF (Photosynthesis)', 'VOD_C (Structure)']
for a in range(3):
    mask = labels_3 == a
    n = mask.sum()
    print(f"\n  Archetype {a+1} (n={n}):")
    for m_idx, m_name in enumerate(metric_names):
        traj = X_scaled[mask, :, m_idx].mean(axis=0)
        slope_val = np.polyfit(range(1,9), traj, 1)[0]
        final_val = traj[4:8].mean()
        print(f"    {m_name}: slope={slope_val*1000:+.1f}e-3/yr, final_mean={final_val:+.3f}")

# Phase diagram summary (final NDVI vs VOD)
print("\n" + "="*60)
print("RECOVERY STATE SPACE SUMMARY")
print("="*60)
final_ndvi = X_scaled[:, 4:8, 0].mean(axis=1)
final_vod = X_scaled[:, 4:8, 3].mean(axis=1)
for a in range(3):
    mask = labels_3 == a
    print(f"  Archetype {a+1}: NDVI_final={final_ndvi[mask].mean():+.3f} +/- {final_ndvi[mask].std():.3f}, "
          f"VOD_final={final_vod[mask].mean():+.3f} +/- {final_vod[mask].std():.3f}")

print(f"\nDone. Best K={best_k}, K=3 silhouette={all_sils[1]:.4f}")