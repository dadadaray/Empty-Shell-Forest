import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Global 5km pixel-level ES probability + phantom AGB carbon deficit.
Paper Methods 5.5, Fig 3. Delta_b computed dynamically from 601-site training data (Eq. 7).
dNBR fixed at 0.047 (standardized severe-disturbance scenario)."""
import rasterio, numpy as np, pandas as pd, xgboost as xgb, warnings
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
warnings.filterwarnings('ignore')

# ===========================================================================
# Step 1: Dynamic delta_b computation from training data (Eq. 7)
# ===========================================================================
print("Computing biome deficit factors from training data...")

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
    site = panel[panel['Ref_ID']==ref_id].sort_values('Year')
    es = site['event_start'].iloc[0]
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
            if i<W: row[f'{m}_Y{i+1}'] = r.get(m,np.nan)
    tensor_data[ref_id] = row

df_t = pd.DataFrame(tensor_data).T
valid_refs = [ref for ref in df_t.index if all(
    df_t.loc[ref,[f'{m}_Y{i}' for i in range(1,W+1)]].notna().sum()>=5 for m in metrics)]
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
es_label = np.argmax([np.mean((X_raw[:,:,0]-X_raw[:,:,2])[labels==c,W-1]) for c in range(3)])
print(f"DTW sites: {len(df_f)}, ES = Label {es_label} (n={(labels==es_label).sum()})")

# Merge biome info and compute delta_b per biome
df_dtw = pd.DataFrame({
    'Ref_ID': df_f.index, 'Label': labels,
    'dVOD': X_raw[:,W-1,2] - X_raw[:,0,2],
    'VOD_y1': X_raw[:,0,2], 'VOD_y2': X_raw[:,1,2]
})
df_feat = pd.read_csv('GTM_1393_Landsat_20D_Advanced.csv')
df_dtw = pd.merge(df_dtw, df_feat[['Ref_ID','Biome_Type']], on='Ref_ID', how='inner')

# Macro-climate groups (paper Table S2)
MOIST  = [1,2,3,4,14]
ARID   = [5,7,8,9,10,12,13]
BOREAL = [6,11]

print("\nBiome delta_b (Eq. 7):")
observed_deltas = {g: [] for g in ['moist','arid','boreal']}

for b in sorted(df_dtw['Biome_Type'].dropna().unique()):
    b_int = int(b)
    es_mask  = (df_dtw['Biome_Type']==b) & (df_dtw['Label']==es_label)
    ot_mask  = (df_dtw['Biome_Type']==b) & (df_dtw['Label']!=es_label)
    n_es, n_ot = es_mask.sum(), ot_mask.sum()

    if n_es >= 3 and n_ot >= 3:
        dVOD_es = df_dtw.loc[es_mask, 'dVOD'].mean()
        dVOD_ot = df_dtw.loc[ot_mask, 'dVOD'].mean()
        vod_base = df_dtw.loc[df_dtw['Biome_Type']==b, ['VOD_y1','VOD_y2']].mean(axis=1).mean()
        delta = max((dVOD_ot - dVOD_es) / vod_base, 0) if vod_base > 0 else 0
        group = 'moist' if b_int in MOIST else 'arid' if b_int in ARID else 'boreal'
        observed_deltas[group].append(delta)
        print(f"  Biome {b_int:2d}: n_ES={n_es:3d}, n_Other={n_ot:3d}, delta={delta:.4f}  [-> {group}]")
    else:
        group = 'moist' if b_int in MOIST else 'arid' if b_int in ARID else 'boreal'
        print(f"  Biome {b_int:2d}: n_ES={n_es:3d}, n_Other={n_ot:3d}  [insufficient, -> {group} default]")

# Three-group defaults: mean of observed deltas within each group
delta_moist  = np.mean(observed_deltas['moist'])  if observed_deltas['moist']  else 0.0047
delta_arid   = np.mean(observed_deltas['arid'])   if observed_deltas['arid']   else 0.0372
delta_boreal = 0.0

# Build biome_delta dict: ALL biomes get their group default
biome_delta = {}
for b in range(1, 15):
    if b in MOIST:  biome_delta[b] = delta_moist
    elif b in ARID:   biome_delta[b] = delta_arid
    else:             biome_delta[b] = delta_boreal

print(f"\nThree-group deltas (all biomes use group mean):")
print(f"  Moist  (biomes {MOIST}):  delta = {delta_moist:.4f}  (n_obs={len(observed_deltas['moist'])})")
print(f"  Arid   (biomes {ARID}):   delta = {delta_arid:.4f}  (n_obs={len(observed_deltas['arid'])})")
print(f"  Boreal (biomes {BOREAL}): delta = {delta_boreal:.4f}  (n_obs={len(observed_deltas['boreal'])})")

# ===========================================================================
# Step 2: Biexponential decay model for long-term carbon deficit
# Paper Methods 5.5: projects VOD gap beyond 8-year window (R^2=0.85)
# ===========================================================================
from scipy.optimize import curve_fit

# Cumulative VOD deficit trajectory (Others minus ES, accumulated over time)
vod_ot = np.array([X_raw[labels!=es_label, yr, 2].mean() for yr in range(W)])
vod_es = np.array([X_raw[labels==es_label, yr, 2].mean() for yr in range(W)])
annual_gap = vod_ot - vod_es
cum_gap = np.cumsum(annual_gap)

# Biexponential model: dual dynamics of rapid recovery + slow asymptotic saturation
# Paper Methods 5.5: "biexponential decay model (R^2=0.85) fitted to temporal VOD gap trajectory"
def biexp_gap(t, a, b, c, d):
    return a * (1 - np.exp(-b * t)) + c * (1 - np.exp(-d * t))

t_years = np.arange(1, W+1)
try:
    popt, _ = curve_fit(biexp_gap, t_years, cum_gap,
                        p0=[cum_gap[-1]/2, 0.3, cum_gap[-1]/2, 0.05], maxfev=10000)
    gap_pred = biexp_gap(t_years, *popt)
    ss_res = np.sum((cum_gap - gap_pred)**2)
    ss_tot = np.sum((cum_gap - cum_gap.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0

    t_long = np.arange(1, 41)
    gap_long = biexp_gap(t_long, *popt)
    asymp_deficit_ratio = gap_long[-1] / cum_gap[-1] if cum_gap[-1] > 0 else 1.0
    print(f"\nBiexponential VOD gap model:")
    print(f"  Annual VOD gap: {[f'{v:.4f}' for v in annual_gap]}")
    print(f"  Cumulative gap: {[f'{v:.4f}' for v in cum_gap]}")
    print(f"  R^2 = {r2:.3f}")
    print(f"  Asymptotic multiplier (yr40/yr8): {asymp_deficit_ratio:.2f}x")
except Exception as e:
    print(f"\nBiexponential fit failed: {e}")
    asymp_deficit_ratio = 1.0

# ===========================================================================
# Step 3: Global pixel-level prediction
# ===========================================================================
print("\nLoading XGBoost model...")
model = xgb.XGBClassifier()
model.load_model('XGBoost_Global_20F.json')

tiff_bands = [
    'tmmx','pr','vpd','soil','def','pdsi','Max_VPD','Max_DEF',
    'elevation','slope','aspect','TWI',
    'Sand_Content','Clay_Content','Bulk_Density','Carbon_Content',
    'Biome_Type','Human_Footprint','Pre_NDVI','dNBR'
]
training_order = [
    'Pre_NDVI','vpd','pr','tmmx','soil','def','pdsi','elevation','slope','aspect',
    'Sand_Content','Clay_Content','Bulk_Density','Carbon_Content','Biome_Type','Human_Footprint',
    'Max_VPD','Max_DEF','dNBR','TWI'
]

tiff_path = 'ES_Global_20D_AGB_5km_Merged.tif'
CHUNK_ROWS = 200

print(f"Opening {tiff_path}...")
with rasterio.open(tiff_path) as src:
    profile = src.profile; H, W = src.shape; res_deg = src.res[0]
    print(f"Raster: {H}x{W} = {H*W/1e6:.1f}M pixels")

    # Open output TIFFs for streaming write (chunk by chunk)
    profile.update(count=1, nodata=np.nan, compress='lzw')
    dst_prob = rasterio.open(os.path.join(TEMP_DIR, "Fig4a_Global_ES_Probability.tif"), 'w', **profile)
    dst_def  = rasterio.open(os.path.join(TEMP_DIR, "Fig4b_Global_Phantom_AGB_Deficit.tif"), 'w', **profile)

    total_C_Mg = 0.0
    n_es_total = 0
    n_val_total = 0

    for start_row in range(0, H, CHUNK_ROWS):
        end_row = min(start_row + CHUNK_ROWS, H)
        data = src.read(window=((start_row, end_row), (0, W)))
        chunk_H = data.shape[1]
        flat = data.transpose(1,2,0).reshape(-1, 21)

        X_chunk = np.zeros((flat.shape[0], 20), dtype=np.float32)
        for i, name in enumerate(training_order):
            X_chunk[:, i] = flat[:, tiff_bands.index(name)]
        agb = flat[:, 20]

        valid = (~np.isnan(X_chunk).any(axis=1)) & (~np.isnan(agb)) & (agb > 0)

        chunk_prob = np.full((chunk_H, W), np.nan, dtype=np.float32)
        chunk_def  = np.full((chunk_H, W), np.nan, dtype=np.float32)

        if valid.sum() > 0:
            Xv, agb_v = X_chunk[valid], agb[valid]
            Xv[:, training_order.index('dNBR')] = 0.047
            prob = model.predict_proba(Xv)[:, 1]

            biome = Xv[:, training_order.index('Biome_Type')].astype(int)
            delta = np.array([biome_delta.get(b, delta_moist) for b in biome])

            valid_idx = np.where(valid)[0]
            row_idx = start_row + valid_idx // W
            lats = np.linspace(src.bounds.top, src.bounds.bottom, H)
            pixel_area_ha = (res_deg * 111.32)**2 * np.cos(np.radians(np.abs(lats[row_idx]))) * 100
            deficit_Mg = prob * delta * agb_v * pixel_area_ha

            chunk_flat_p = np.full(chunk_H * W, np.nan, dtype=np.float32)
            chunk_flat_d = np.full(chunk_H * W, np.nan, dtype=np.float32)
            chunk_flat_p[valid] = prob.astype(np.float32)
            chunk_flat_d[valid] = deficit_Mg.astype(np.float32)
            chunk_prob = chunk_flat_p.reshape(chunk_H, W)
            chunk_def  = chunk_flat_d.reshape(chunk_H, W)

            total_C_Mg += np.nansum(chunk_def)
            n_es_total += (chunk_prob > 0.5).sum()
            n_val_total += (~np.isnan(chunk_prob)).sum()

        dst_prob.write(chunk_prob, 1, window=((start_row, end_row), (0, W)))
        dst_def.write(chunk_def, 1, window=((start_row, end_row), (0, W)))
        print(f"  Rows {start_row}-{end_row-1}: {valid.sum():,} valid", end='\r')

    dst_prob.close(); dst_def.close()

total_C_Pg = total_C_Mg / 1e9
asymp_C_Pg = total_C_Pg * asymp_deficit_ratio
print(f"\n8-year carbon deficit:    {total_C_Pg:.2f} Pg C")
print(f"Asymptotic projection:    {asymp_C_Pg:.2f} Pg C  (yr40/yr8 = {asymp_deficit_ratio:.2f}x)")
print(f"ES probability > 0.5: {n_es_total:,} / {n_val_total:,} pixels ({100*n_es_total/n_val_total:.1f}%)")
