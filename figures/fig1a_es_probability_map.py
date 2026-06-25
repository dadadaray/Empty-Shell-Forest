import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Fig1a: Global ES Probability Map — Forest/grassland pixel base + ES site overlay."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import rasterio
import xgboost as xgb
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],
    'font.size':14,'axes.titlesize':14,'axes.titleweight':'bold',
    'axes.labelsize':14,'axes.labelweight':'bold','xtick.labelsize':14,
    'ytick.labelsize':14,'legend.fontsize':12,'axes.linewidth':1.0,
    'pdf.fonttype':42,'ps.fonttype':42
})

# ============================================
# Yan-aligned DTW + XGBoost
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
es_label = np.argmax(end_dec)

feats = ['Pre_NDVI','vpd','pr','tmmx','soil','def','pdsi','elevation','slope','aspect',
    'Sand_Content','Clay_Content','Bulk_Density','Carbon_Content','Biome_Type','Human_Footprint',
    'Max_VPD','Max_DEF','dNBR','TWI']
df_1393 = pd.read_csv('GTM_1393_Landsat_20D_Advanced.csv')
df_1393['Label'] = df_1393['Ref_ID'].map(dict(zip(df_f.index, labels)))
train = df_1393[df_1393['Label'].notna()]
y_tr = (train['Label']==es_label).astype(int).values
sw = (len(y_tr)-y_tr.sum())/y_tr.sum()
model = xgb.XGBClassifier(max_depth=2, learning_rate=0.05, n_estimators=150,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=sw, random_state=42)
model.fit(train[feats].fillna(train[feats].median()), y_tr)
probs_yan = model.predict_proba(df_1393[feats].fillna(df_1393[feats].median()))[:,1]

# ============================================
# GLOBAL FOREST / GRASSLAND PIXEL BASE (from merged TIFF)
# ============================================
STEP = 3  # 5km × 3 = 15km per pixel
with rasterio.open('ES_Global_20D_AGB_5km_Merged.tif') as src:
    biome = src.read(17)[::STEP, ::STEP]  # band 17 = Biome_Type
    tmmx  = src.read(1)[::STEP, ::STEP]   # band 1 = tmmx (NaN = ocean)
    H, W = biome.shape
    bounds = src.bounds

is_ocean = np.isnan(tmmx)
is_forest = np.isin(biome, [1,2,3,4,5,6,12]) & ~is_ocean
is_grass  = np.isin(biome, [7,8,9,10]) & ~is_ocean

# Build RGBA pixel layer: white background, green forest/grassland, transparent elsewhere
pixel_rgb = np.ones((H, W, 3), dtype=np.uint8) * 255  # white everywhere
pixel_rgb[is_forest] = [56, 108, 67]    # forest green
pixel_rgb[is_grass]  = [163, 194, 127]  # grassland light green

# ============================================
# ES SITE OVERLAY
# ============================================
coords = pd.read_csv('GTM_Master_Panel_Data_Spatial.csv')
pred_sites = pd.DataFrame({'Ref_ID': df_1393['Ref_ID'], 'Prob_ES': probs_yan})
pred_sites = pd.merge(pred_sites, coords[['Ref_ID','lat','long']].drop_duplicates(subset='Ref_ID'),
                      on='Ref_ID', how='inner')
probs = pred_sites['Prob_ES'].values

# ============================================
# FIGURE
# ============================================
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
ax.set_global()
ax.set_facecolor('white')

# Layer 1: forest/grassland pixel base
ax.imshow(pixel_rgb, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
          origin='upper', aspect='auto', interpolation='nearest', zorder=1)

# Layer 2: ES sites as colored circles — continuous red gradient
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar

# Build colormap: light gray at 0 → orange at 0.3 → deep red at 1.0
es_cmap = LinearSegmentedColormap.from_list('es_risk', [
    (0.00, '#e8e8e8'),   # very low risk = light gray
    (0.15, '#fdcc8a'),   # low
    (0.40, '#fc8d59'),   # moderate
    (0.65, '#e34a33'),   # high
    (1.00, '#67000d'),   # very high = dark red
])

# Sort so highest-probability sites plot on top
pred_sites = pred_sites.sort_values('Prob_ES')
sc = ax.scatter(pred_sites['long'], pred_sites['lat'],
                c=pred_sites['Prob_ES'], cmap=es_cmap, vmin=0, vmax=1,
                s=40, edgecolor='#333333', linewidth=0.3,
                transform=ccrs.PlateCarree(), zorder=3)

# Layer 3: coastlines
ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.6, color='#888888', zorder=7)

# Colorbar — inset at bottom-center over Antarctica
cbar_ax = fig.add_axes([0.3, 0.16, 0.4, 0.025])
cbar = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Empty Shell Probability', fontsize=12, fontweight='bold')
cbar.ax.tick_params(labelsize=10)

# Legend — bottom-left corner
from matplotlib.patches import Patch
legend_items = [
    Patch(facecolor='#386c43', label='Forest'),
    Patch(facecolor='#a3c27f', label='Grassland'),
]
ax.legend(handles=legend_items, loc='lower left', frameon=True, fontsize=11,
          title='Land Cover', title_fontsize=12)

plt.savefig(os.path.join(TEMP_DIR, 'Fig1a_Map.pdf'), dpi=250, bbox_inches='tight')
plt.close()

print(f'Saved: {TEMP_DIR}/Fig1a_Map.pdf')
print(f'Pixel grid: {H}x{W} (15km res), Forest: {is_forest.sum():,}, Grassland: {is_grass.sum():,}, Ocean: {is_ocean.sum():,}')
print(f'Sites: {len(pred_sites)}, ES (prob>0.5): {(probs>0.5).sum()}')
