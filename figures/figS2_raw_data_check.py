import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Fig S2b: Raw data reality check — NDVI, VOD, and SPEI-12 for a typical Empty Shell site."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],
    'font.size':14,'axes.labelsize':14,'axes.labelweight':'bold',
    'axes.titlesize':14,'axes.titleweight':'bold',
    'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12,
    'axes.linewidth':1.0,'xtick.direction':'out','ytick.direction':'out',
    'pdf.fonttype':42,'ps.fonttype':42
})

# Yan pipeline W=8
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

metrics = ['NDVI_mean','NDII_mean','VOD_CKXU','SIF_mean']
tensor_8 = {}
for ref_id, lv_year in site_lv.items():
    site = panel[panel['Ref_ID']==ref_id].sort_values('Year')
    traj = site[(site['Year']>=lv_year)&(site['Year']<=lv_year+7)]
    if len(traj)<5: continue
    row = {'Ref_ID':ref_id,'lv_year':lv_year}
    for m in metrics:
        for i,(_,r) in enumerate(traj.iterrows()):
            if i<8: row[f'{m}_Y{i+1}']=r.get(m,np.nan)
    tensor_8[ref_id]=row
df_8 = pd.DataFrame(tensor_8).T
v8 = [ref for ref in df_8.index if all(df_8.loc[ref,[f'{m}_Y{i}' for i in range(1,9)]].notna().sum()>=5 for m in metrics)]
df_8 = df_8.loc[v8]
for m in metrics:
    cols=[f'{m}_Y{i}' for i in range(1,9)]; df_8[cols]=df_8[cols].interpolate(method='linear',axis=1,limit_direction='both')
df_8 = df_8.dropna()
X8 = np.zeros((len(df_8),8,4))
for fi,m in enumerate(metrics): X8[:,:,fi] = df_8[[f'{m}_Y{i}' for i in range(1,9)]].values
X8_s = TimeSeriesScalerMeanVariance().fit_transform(X8)
lbls_8 = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=10, random_state=42, n_jobs=-1).fit_predict(X8_s)
dec_8 = X8[:,:,0]-X8[:,:,2]
es_label_8 = np.argmax([np.mean(dec_8[lbls_8==c,7]) for c in range(3)])

# Pick the clearest Empty Shell site (post-2004 for SIF, lag>=1, highest NDVI-VOD gap)
es_refs = df_8.index[lbls_8==es_label_8]
best = None; best_gap = -999
for r in es_refs:
    es_y_r = int(panel[panel['Ref_ID']==r]['event_start'].iloc[0])
    lv_y_r = int(df_8.loc[r,'lv_year'])
    if es_y_r>=2004 and lv_y_r>es_y_r:
        gap = X8[df_8.index.get_loc(r),7,0]-X8[df_8.index.get_loc(r),7,2]
        if gap>best_gap: best_gap=gap; best=r
if best is None: best = es_refs[0]
site = panel[panel['Ref_ID']==best].sort_values('Year')
es_y = int(site['event_start'].iloc[0]); lv_y = int(df_8.loc[best,'lv_year'])

# Plot NDVI, VOD, SPEI-12 timeline
all_yr = list(range(es_y-3, lv_y+8))
idx_dist = all_yr.index(es_y)

fig, ax = plt.subplots(figsize=(10, 6))
spei_vals = [site[site['Year']==yr]['SPEI_12_month'].values[0] if len(site[site['Year']==yr])>0 and pd.notna(site[site['Year']==yr]['SPEI_12_month'].values[0]) else np.nan for yr in all_yr]
ax.bar(range(len(all_yr)), spei_vals, color='#B0BEC5', alpha=0.35, width=0.6, zorder=0, label='SPEI-12')

ndvi_raw = [site[site['Year']==yr]['NDVI_mean'].values[0] if len(site[site['Year']==yr])>0 and pd.notna(site[site['Year']==yr]['NDVI_mean'].values[0]) else np.nan for yr in all_yr]
ax.plot(range(len(all_yr)), ndvi_raw, 'o-', color='#55A868', lw=2.5, markersize=7, label='NDVI (Greenness)')
ax.set_ylabel('NDVI', fontweight='bold', color='#55A868')
ax.tick_params(axis='y', labelcolor='#55A868')

ax_r = ax.twinx()
vod_raw = [site[site['Year']==yr]['VOD_CKXU'].values[0] if len(site[site['Year']==yr])>0 and pd.notna(site[site['Year']==yr]['VOD_CKXU'].values[0]) else np.nan for yr in all_yr]
ax_r.plot(range(len(all_yr)), vod_raw, 's--', color='#8C564B', lw=2.5, markersize=7, label='VOD (Biomass)')
ax_r.set_ylabel('VOD', fontweight='bold', color='#8C564B')
ax_r.tick_params(axis='y', labelcolor='#8C564B')

ax.axvline(idx_dist, color='#d73027', ls='--', lw=2, alpha=0.8)
ax.text(idx_dist+0.2, ax.get_ylim()[1]*0.85, 'Fire', fontsize=12, fontweight='bold', color='#d73027')

rel_yrs = [yr-es_y for yr in all_yr]
ax.set_xticks(range(len(all_yr))[::2])
ax.set_xticklabels([str(rel_yrs[i]) for i in range(0,len(all_yr),2)])
ax.set_xlabel('Years Relative to Disturbance', fontweight='bold')
ax.set_title(f'Fig S2b: Raw Data — Empty Shell Site {best}', fontweight='bold')
ax.spines['top'].set_visible(False); ax_r.spines['top'].set_visible(False)
ax.grid(axis='y', linestyle=':', alpha=0.3)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax_r.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, fontsize=11, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(TEMP_DIR, 'FigS2b_Raw_Data.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {TEMP_DIR}/FigS2b_Raw_Data.pdf (Site {best}, ES={es_y}, recovery onset={lv_y})')
