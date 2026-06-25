import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Supplementary Fig: Training set composition (601 DTW sites by biome x ES status)."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],
    'font.size':13,'axes.labelsize':13,'axes.labelweight':'bold',
    'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':11,
    'axes.linewidth':1.0,'pdf.fonttype':42,'ps.fonttype':42
})

# Yan pipeline
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

# Merge biome info
df_feat = pd.read_csv('GTM_1393_Landsat_20D_Advanced.csv')
df_plot = pd.DataFrame({'Ref_ID':df_f.index, 'ES':(labels==es_label).astype(int)})
df_plot = pd.merge(df_plot, df_feat[['Ref_ID','Biome_Type']], on='Ref_ID', how='inner')

biome_names = {1:'Trop.Moist',2:'Trop.Dry',3:'Trop.Conif',4:'Temp.Broad',5:'Temp.Conif',
               6:'Boreal',7:'Trop.Grass',8:'Temp.Grass',9:'Flood.Grass',10:'Mont.Grass',
               11:'Tundra',12:'Mediterr',13:'Desert',14:'Mangrove'}
df_plot['Biome'] = df_plot['Biome_Type'].map(biome_names).fillna('Other')

counts = df_plot.groupby(['Biome','ES']).size().unstack(fill_value=0)
counts['Total'] = counts.sum(axis=1)
counts = counts.sort_values('Total', ascending=False)
counts.index = [b + ' (n=' + str(int(counts.loc[b, 'Total'])) + ')' for b in counts.index]

fig, ax = plt.subplots(figsize=(12, 6))
biomes = counts.index
es_vals = counts.get(1, pd.Series(0, index=counts.index))
ot_vals = counts.get(0, pd.Series(0, index=counts.index))

ax.barh(range(len(biomes)), ot_vals, color='#4575b4', label='Others', edgecolor='white')
ax.barh(range(len(biomes)), es_vals, left=ot_vals, color='#d73027', label='Empty Shell', edgecolor='white')
ax.set_yticks(range(len(biomes))); ax.set_yticklabels(biomes)
ax.set_xlabel('Number of Sites'); ax.legend()
ax.set_title(f'DTW Training Set Composition (K=3, n={len(df_plot)}, ES={df_plot.ES.sum()})', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(TEMP_DIR, 'FigS2_Training_Distribution.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {TEMP_DIR}/FigS2_Training_Distribution.pdf')
