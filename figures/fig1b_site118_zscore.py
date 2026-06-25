import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Fig1b: Continuous Timeline with Recovery Onset (NDVI Minimum = Year 0)."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],
    'font.size':14,'axes.titlesize':14,'axes.titleweight':'bold',
    'axes.labelsize':14,'axes.labelweight':'bold','xtick.labelsize':14,
    'ytick.labelsize':14,'legend.fontsize':14,'axes.linewidth':1.0,
    'xtick.direction':'out','ytick.direction':'out',
    'pdf.fonttype':42,'ps.fonttype':42
})

# ============================================
# DATA PREP
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
decoupling = X_raw[:,:,0] - X_raw[:,:,2]  # 0=NDVI, 2=VOD (NOT 3=SIF!)
end_dec = [np.mean(decoupling[labels==c,W-1]) for c in range(3)]
es_label = np.argmax(end_dec); is_es = labels == es_label

def get_spei_baseline(site_df, es_year):
    """Return list of 3 baseline years (SPEI-filtered consecutive non-drought, valid NDVI & VOD)."""
    pre = site_df[(site_df['Year']<es_year)&(site_df['SPEI_12_month'].notna())].sort_values('Year')
    groups, cur = [], []
    for _, r in pre.iterrows():
        if r['SPEI_12_month'] > -1:
            cur.append(int(r['Year']))
        else:
            if len(cur) >= 3: groups.append(cur)
            cur = []
    if len(cur) >= 3: groups.append(cur)
    for g in reversed(groups):  # most recent first
        g3 = g[-3:]
        ok = True
        for y in g3:
            row = site_df[site_df['Year']==y]
            if len(row)==0 or pd.isna(row['NDVI_mean'].values[0]) or pd.isna(row['VOD_CKXU'].values[0]):
                ok = False; break
        if ok: return g3
    # Fallback
    non_dry = pre[pre['SPEI_12_month']>-1]
    return sorted(non_dry['Year'].tail(3).values) if len(non_dry)>=3 else sorted(pre['Year'].tail(3).values)

# Pick ES site: disturbed >= 2004 (SIF data), high decoupling, reasonable baseline variance
es_refs = df_f.index[is_es]
candidates = []
for r in es_refs:
    es_y_r = int(panel[panel['Ref_ID']==r]['event_start'].iloc[0])
    lv_y_r = int(df_f.loc[r,'lv_year'])
    lag = lv_y_r - es_y_r
    if es_y_r < 2004 or lag < 1: continue
    site_tmp = panel[panel['Ref_ID']==r].sort_values('Year')
    # Compute SPEI baseline and its NDVI/VOD SD
    bl = get_spei_baseline(site_tmp, es_y_r)
    bl_data = site_tmp[site_tmp['Year'].isin(bl)]
    bl_ndvi_sd = bl_data['NDVI_mean'].std(ddof=1) if len(bl_data)>=2 else 0
    bl_vod_sd = bl_data['VOD_CKXU'].std(ddof=1) if len(bl_data)>=2 else 0
    if pd.isna(bl_ndvi_sd) or pd.isna(bl_vod_sd): continue
    if bl_ndvi_sd < 0.015: continue
    if bl_vod_sd < 0.01: continue
    idx_loc = df_f.index.get_loc(r)
    gap = X_raw[idx_loc, W-1, 0] - X_raw[idx_loc, W-1, 2]  # NDVI - VOD decoupling
    # Also check max absolute Z-score across the timeline (want it in 3-8 range for visual clarity)
    all_yr_r = list(range(es_y_r-3, lv_y_r+8))
    max_z = 0
    for m in ['NDVI_mean','VOD_CKXU']:
        bl_vals = bl_data[m].dropna()
        mu_p = bl_vals.mean(); sd_p = bl_vals.std(ddof=1)
        if sd_p == 0: sd_p = 1e-5
        for yr in all_yr_r:
            v = site_tmp[site_tmp['Year']==yr][m].values
            if len(v)>0 and pd.notna(v[0]):
                max_z = max(max_z, abs((v[0]-mu_p)/sd_p))
    if not (3 <= max_z <= 12):
        continue
    # Verify Z-score pattern: NDVI recovers (end > 0) AND VOD stays low (end < 0)
    # Use year-8 data from X_raw tensor (recovery year W-1)
    ndvi_raw_end = X_raw[df_f.index.get_loc(r), W-1, 0]
    vod_raw_end = X_raw[df_f.index.get_loc(r), W-1, 2]
    ndvi_z_end = (ndvi_raw_end - bl_data['NDVI_mean'].mean()) / bl_ndvi_sd
    vod_z_end = (vod_raw_end - bl_data['VOD_CKXU'].mean()) / bl_vod_sd
    if ndvi_z_end > 0 and vod_z_end < 0:
        dec_z = ndvi_z_end - vod_z_end
        candidates.append((r, gap, lag, max_z, bl_ndvi_sd, bl_vod_sd, ndvi_z_end, vod_z_end, dec_z))

if candidates:
    candidates.sort(key=lambda x: x[-1], reverse=True)
    best = candidates[0][0]
else:
    best = max(es_refs,
        key=lambda r: X_raw[df_f.index.get_loc(r),W-1,0]-X_raw[df_f.index.get_loc(r),W-1,2])

site = panel[panel['Ref_ID']==best].sort_values('Year')
es_y=int(site['event_start'].iloc[0]); lv_y=int(df_f.loc[best,'lv_year'])

# ============================================
# CONTINUOUS TIMELINE
# ============================================
# Full timeline: pre-disturbance (es_y-3) → recovery onset (lv_y) → post (lv_y+8)
all_yr = list(range(es_y-3, lv_y+8))
idx_event = all_yr.index(es_y)    # disturbance
idx_recov = all_yr.index(lv_y)    # NDVI minimum = Year 0

all_m = ['NDVI_mean','SIF_mean','NDII_mean','VOD_CKXU']
mcols = {'NDVI_mean':'#1a9850','SIF_mean':'#fdae61','NDII_mean':'#4575b4','VOD_CKXU':'#d73027'}
mlabs = {'NDVI_mean':'NDVI (Greenness)','SIF_mean':'SIF (Photosynthesis)',
         'NDII_mean':'NDII (Canopy Water)','VOD_CKXU':'VOD (Biomass)'}
mstyles = {'NDVI_mean':'-','SIF_mean':'-.','NDII_mean':':','VOD_CKXU':'--'}

fig, ax = plt.subplots(figsize=(10, 7))

# Compute SPEI baseline for the selected site
bl_years = get_spei_baseline(site, es_y)
bl_data = site[site['Year'].isin(bl_years)]

for m in all_m:
    raw = [site[site['Year']==yr][m].values[0] if len(site[site['Year']==yr])>0 and pd.notna(site[site['Year']==yr][m].values[0]) else np.nan for yr in all_yr]
    # Z-score baseline = Yan SPEI-filtered 3 non-drought years
    bl_vals = bl_data[m].dropna()
    if len(bl_vals)<2: continue
    mu_p=bl_vals.mean(); sd_p=bl_vals.std(ddof=1)
    if sd_p==0: sd_p=1e-5
    z=[(v-mu_p)/sd_p if pd.notna(v) else np.nan for v in raw]
    ax.plot(range(len(all_yr)),z,color=mcols[m],lw=2.5,ms=6,marker='o',
            ls=mstyles[m],label=mlabs[m],markevery=1)

# Reference lines
ax.axhline(0,color='gray',ls='-',lw=1.2,alpha=0.5,zorder=0)
# Disturbance event (gray dashed)
ax.axvline(idx_event,color='#555555',lw=1.5,ls='--',alpha=0.7,zorder=5)
ax.text(idx_event-0.3,ax.get_ylim()[1]*0.65,'Disturbance\nEvent',fontsize=14,
        ha='right',color='#555555')
# Recovery onset = NDVI minimum (red dotted)
ax.axvline(idx_recov,color='#d73027',lw=1.5,ls=':',alpha=0.9,zorder=5)
ax.text(idx_recov+0.3,ax.get_ylim()[1]*0.65,'Recovery Onset\n(NDVI Minimum)',
        fontsize=14,ha='left',color='#d73027')

# X-axis: years relative to recovery onset (lv_y = 0)
rel_years = [yr-lv_y for yr in all_yr]
ax.set_xticks(range(len(all_yr)))
ax.set_xticklabels([str(r) for r in rel_years])
ax.set_xlabel('Years Relative to Recovery Onset (NDVI Minimum = 0)',fontweight='bold')

ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.set_ylabel('Standardized Anomaly (Z-score)',fontweight='bold')
#ax.set_title('b. Multi-Dimensional Biophysical Decoupling (Typical Empty Shell Site)',
             #fontweight='bold',fontsize=14,loc='left')
ax.legend(fontsize=14,frameon=False,ncol=2,loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(TEMP_DIR, 'Fig1b_site118_zscore.pdf'), dpi=300,bbox_inches='tight')
plt.close()
print(f'Saved: {TEMP_DIR}/Fig1b_site118_zscore.pdf')
print('Site %d: disturbed %d, NDVI min %d (lag=%dyr), timeline=%d yrs total'%
      (best,es_y,lv_y,lv_y-es_y,len(all_yr)))
