import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Fig S1c: 1393 XGBoost-predicted ES distribution at K=2,3,4,5."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xgboost as xgb
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],
    'font.size':11,'axes.titlesize':13,'axes.titleweight':'bold',
    'pdf.fonttype':42,'ps.fonttype':42
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

features = ['Pre_NDVI','vpd','pr','tmmx','soil','def','pdsi','elevation','slope','aspect',
    'Sand_Content','Clay_Content','Bulk_Density','Carbon_Content','Biome_Type','Human_Footprint',
    'Max_VPD','Max_DEF','dNBR','TWI']
df_1393 = pd.read_csv('GTM_1393_Landsat_20D_Advanced.csv')
X_all = df_1393[features].fillna(df_1393[features].median())
coords = pd.read_csv('GTM_Master_Panel_Data_Spatial.csv')[['Ref_ID','lat','long']].drop_duplicates()

fig, axes = plt.subplots(2, 2, figsize=(18, 10), subplot_kw={'projection': ccrs.PlateCarree()})
axes = axes.flatten()

for idx, k in enumerate([2,3,4,5]):
    ax = axes[idx]
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor='#f4f4f4', alpha=0.9)
    ax.add_feature(cfeature.OCEAN, facecolor='#e0e6ed', alpha=0.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, alpha=0.4)

    # DTW clustering
    labels = TimeSeriesKMeans(n_clusters=k, metric="dtw", max_iter=10, random_state=42, n_jobs=-1).fit_predict(X_s)
    decoupling = X_raw[:,:,0] - X_raw[:,:,2]
    end_dec = [np.mean(decoupling[labels==c,W-1]) for c in range(k)]
    es_label = np.argmax(end_dec)

    # Train XGBoost on DTW-labeled sites -> predict on 1393
    label_map = dict(zip(df_f.index, labels))
    train = df_1393.copy(); train['Label'] = train['Ref_ID'].map(label_map)
    train = train.dropna(subset=['Label'])
    y_tr = (train['Label']==es_label).astype(int).values
    X_tr = train[features].fillna(train[features].median())
    sw = (len(y_tr)-y_tr.sum())/max(y_tr.sum(),1)
    model = xgb.XGBClassifier(max_depth=2, learning_rate=0.05, n_estimators=150,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=sw, random_state=42)
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_all)[:,1]

    # Classify: prob > 0.5 = ES
    pred_map = pd.DataFrame({'Ref_ID':df_1393['Ref_ID'], 'Prob_ES':probs})
    pred_map['ES'] = pred_map['Prob_ES'] > 0.5
    pred_map = pd.merge(pred_map, coords, on='Ref_ID', how='inner')

    es_pts = pred_map[pred_map['ES']]
    ot_pts = pred_map[~pred_map['ES']]

    ax.scatter(ot_pts['long'],ot_pts['lat'],color='#B0BEC5',s=8,alpha=0.35,edgecolor='none',
               zorder=1,label='Others (n=%d)'%len(ot_pts),transform=ccrs.PlateCarree())
    ax.scatter(es_pts['long'],es_pts['lat'],color='#d73027',s=25,alpha=0.85,edgecolor='white',
               linewidth=0.2,zorder=2,label='Empty Shell (n=%d)'%len(es_pts),transform=ccrs.PlateCarree())
    ax.legend(fontsize=9,loc='lower left',frameon=True,markerscale=1.2)
    ax.set_title('K=%d: ES Spatial Distribution (XGBoost-predicted, 1,393 sites)'%k,
                 fontweight='bold', fontsize=11, pad=-2)

plt.tight_layout()
plt.savefig(os.path.join(TEMP_DIR, 'FigS1c_K_Map.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {TEMP_DIR}/FigS1c_K_Map.pdf')
for k in [2,3,4,5]: print('K=%d: done'%k)
