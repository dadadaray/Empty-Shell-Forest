import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Extended Data Fig. S1a: Empty Shell robustness across K=2,3,4,5."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],
    'font.size':14,'axes.labelsize':14,'xtick.labelsize':12,'ytick.labelsize':12,
    'legend.fontsize':14,'axes.linewidth':1.0,
    'pdf.fonttype':42,'ps.fonttype':42
})
COLOR_ES = '#d73027'
GREYS = ['#4d4d4d','#969696','#d9d9d9']

# ============================================
# DATA PREP (Yan-aligned, W=8)
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
    row = {'Ref_ID':ref_id};
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
for fi,m in enumerate(metrics): X_raw[:,:,fi] = df_f[[f'{m}_Y{i}' for i in range(1,W+1)]].values
X_s = TimeSeriesScalerMeanVariance().fit_transform(X_raw)

# Standardized decoupling
X_s_ndvi = (X_raw[:,:,0]-np.mean(X_raw[:,:,0],axis=1,keepdims=True))/(np.std(X_raw[:,:,0],axis=1,keepdims=True)+1e-5)
X_s_vod  = (X_raw[:,:,3]-np.mean(X_raw[:,:,3],axis=1,keepdims=True))/(np.std(X_raw[:,:,3],axis=1,keepdims=True)+1e-5)
decoupling_std = X_s_ndvi - X_s_vod
time_axis = np.arange(1,W+1)

fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 2, wspace=0.18, hspace=0.3)

for idx, k in enumerate([2,3,4,5]):
    ax = fig.add_subplot(gs[idx//2, idx%2])
    labels = TimeSeriesKMeans(n_clusters=k, metric="dtw", max_iter=10, random_state=42, n_jobs=-1).fit_predict(X_s)
    end_dec = [np.mean(decoupling_std[labels==c,W-1]) for c in range(k)]
    es_id = np.argmax(end_dec)

    # Plot non-ES clusters in greys, ES in red
    grey_idx = 0
    for c in range(k):
        c_traj = decoupling_std[labels==c]
        mu = np.mean(c_traj, axis=0); sd = np.std(c_traj, axis=0)
        if c == es_id:
            ax.plot(time_axis, mu, color=COLOR_ES, lw=3.0, zorder=5, label='ES (n=%d)'%len(c_traj))
            ax.fill_between(time_axis, mu-0.5*sd, mu+0.5*sd, color=COLOR_ES, alpha=0.2, zorder=4)
        else:
            ax.plot(time_axis, mu, color=GREYS[grey_idx % 3], lw=1.5, alpha=0.9, zorder=2)
            ax.fill_between(time_axis, mu-0.5*sd, mu+0.5*sd, color=GREYS[grey_idx % 3], alpha=0.10, zorder=1)
            grey_idx += 1

    ax.axhline(0, color='black', ls=':', alpha=0.5)
    ax.set_title('K = %d'%k, fontweight='bold', fontsize=15, loc='left')
    ax.set_xticks([1,4,8])
    ax.set_xlabel('Years from NDVI Minimum',fontweight='bold')
    if idx % 2 == 0: ax.set_ylabel('Standardized Decoupling (NDVI - VOD)', fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(fontsize=12, frameon=False, loc='upper left')

#plt.suptitle('Extended Data Fig. S1a: Empty Shell Anomaly across K=2-5',
             #fontweight='bold', fontsize=17, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(TEMP_DIR, 'FigS1a_K_Trend.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {TEMP_DIR}/FigS1a_K_Trend.pdf')
