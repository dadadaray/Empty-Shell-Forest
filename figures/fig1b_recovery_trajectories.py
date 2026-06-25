import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Fig1b: Single-panel group-mean recovery trajectories — ES vs Others, 4 metrics."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],
    'font.size':14,'axes.titlesize':14,'axes.titleweight':'bold',
    'axes.labelsize':14,'axes.labelweight':'bold','xtick.labelsize':14,
    'ytick.labelsize':14,'legend.fontsize':13,'axes.linewidth':1.0,
    'xtick.direction':'out','ytick.direction':'out',
    'pdf.fonttype':42,'ps.fonttype':42
})

# ============================================
# DATA PREP (Yan-aligned pipeline)
# ============================================
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
    row = {'Ref_ID':ref_id,'lv_year':lv_year}
    for m in metrics:
        for i,(_,r) in enumerate(traj.iterrows()):
            if i<W: row[f'{m}_Y{i+1}']=r.get(m,np.nan)
    tensor_data[ref_id]=row
df_t = pd.DataFrame(tensor_data).T
valid_refs = [ref for ref in df_t.index if all(df_t.loc[ref,[f'{m}_Y{i}' for i in range(1,W+1)]].notna().sum()>=5 for m in metrics)]
df_f = df_t.loc[valid_refs]
for m in metrics:
    cols=[f'{m}_Y{i}' for i in range(1,W+1)]
    df_f[cols]=df_f[cols].interpolate(method='linear',axis=1,limit_direction='both')
df_f = df_f.dropna()

X_raw = np.zeros((len(df_f),W,4))
for fi,m in enumerate(metrics):
    X_raw[:,:,fi] = df_f[[f'{m}_Y{i}' for i in range(1,W+1)]].values
X_s = TimeSeriesScalerMeanVariance().fit_transform(X_raw)
labels = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=10, random_state=42, n_jobs=-1).fit_predict(X_s)
decoupling = X_raw[:,:,0] - X_raw[:,:,2]
end_dec = [np.mean(decoupling[labels==c,W-1]) for c in range(3)]
es_label = np.argmax(end_dec); is_es = labels == es_label

n_es = sum(is_es); n_ot = sum(~is_es)

# ============================================
# GLOBAL Z-SCORE: standardize each metric across all 601×8 data
# ============================================
X_z = np.zeros_like(X_raw)
for i in range(4):
    gm = np.nanmean(X_raw[:,:,i]); gs = np.nanstd(X_raw[:,:,i], ddof=1)
    if gs == 0: gs = 1e-5
    X_z[:,:,i] = (X_raw[:,:,i] - gm) / gs

# ============================================
# FIGURE: single panel
# ============================================
mlabs = ['NDVI (Greenness)','NDII (Canopy Water)','VOD (Biomass)','SIF (Photosynthesis)']
mcols = ['#1a9850','#4575b4','#d73027','#fdae61']
mstyles = ['-',':','--','-.']
mmarkers = ['o','p','s','D']

fig, ax = plt.subplots(figsize=(9, 8))
x = np.arange(1, W+1)

for i in range(4):
    es_mean = np.nanmean(X_z[is_es, :, i], axis=0)
    ot_mean = np.nanmean(X_z[~is_es, :, i], axis=0)
    es_se = np.nanstd(X_z[is_es, :, i], axis=0, ddof=1) / np.sqrt(n_es)
    ot_se = np.nanstd(X_z[~is_es, :, i], axis=0, ddof=1) / np.sqrt(n_ot)

    # Others: gray tones, light band
    ax.fill_between(x, ot_mean-1.96*ot_se, ot_mean+1.96*ot_se, color='gray', alpha=0.10, edgecolor='none')
    ax.plot(x, ot_mean, ls=mstyles[i], marker=mmarkers[i], color='gray', lw=2.0, ms=5,
            markevery=1, label=f'{mlabs[i]} (Others)')

    # ES: colored, solid fill
    ax.fill_between(x, es_mean-1.96*es_se, es_mean+1.96*es_se, color=mcols[i], alpha=0.15, edgecolor='none')
    ax.plot(x, es_mean, ls=mstyles[i], marker=mmarkers[i], color=mcols[i], lw=2.5, ms=6,
            markevery=1, markerfacecolor='white', markeredgewidth=1.8,
            label=f'{mlabs[i]} (ES)')

ax.axhline(0, color='black', ls='-', lw=0.8, alpha=0.3, zorder=0)
ax.set_xlabel('Year from Recovery Onset', fontweight='bold')
ax.set_ylabel('Standardized Value (Global Z-score)', fontweight='bold')
ax.set_xticks(x)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle=':', alpha=0.3)

# Legend: 2 columns to save space
ax.legend(fontsize=10, frameon=False, ncol=3, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(TEMP_DIR, 'Fig1b_Group_Trajectories.pdf'), dpi=300, bbox_inches='tight')
plt.close()

print(f'Saved: {TEMP_DIR}/Fig1b_Group_Trajectories.pdf')
print(f'ES={n_es}, Others={n_ot}')
for i in range(4):
    es_end = np.nanmean(X_z[is_es, W-1, i])
    ot_end = np.nanmean(X_z[~is_es, W-1, i])
    print(f'{mlabs[i]}: ES end Z={es_end:.3f}, Others end Z={ot_end:.3f}')
