import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Supplementary Fig. 2a: Regional Empty Shell Prevalence."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],
    'font.size':14,'axes.labelsize':14,'axes.labelweight':'bold',
    'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':12,
    'axes.linewidth':1.2,'pdf.fonttype':42,'ps.fonttype':42
})

# === Yan pipeline ===
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

features = ['Pre_NDVI','vpd','pr','tmmx','soil','def','pdsi','elevation','slope','aspect',
    'Sand_Content','Clay_Content','Bulk_Density','Carbon_Content','Biome_Type','Human_Footprint',
    'Max_VPD','Max_DEF','dNBR','TWI']
df1393 = pd.read_csv('GTM_1393_Landsat_20D_Advanced.csv')
df1393['Label'] = df1393['Ref_ID'].map(dict(zip(df_f.index, labels)))
train = df1393[df1393['Label'].notna()]
y_tr = (train['Label']==es_label).astype(int).values; sw = (len(y_tr)-y_tr.sum())/y_tr.sum()
model = xgb.XGBClassifier(max_depth=2, learning_rate=0.05, n_estimators=150,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=sw, random_state=42)
model.fit(train[features].fillna(train[features].median()), y_tr)
probs = model.predict_proba(df1393[features].fillna(df1393[features].median()))[:,1]

coords = pd.read_csv('GTM_Master_Panel_Data_Spatial.csv')[['Ref_ID','lat','long']].drop_duplicates(subset='Ref_ID')
df = pd.DataFrame({'Ref_ID':df1393['Ref_ID'],'Prob_ES':probs})
df = pd.merge(df, coords, on='Ref_ID', how='inner')
df['ES'] = df['Prob_ES'] > 0.5

def region(lat, lon):
    if 30 <= lat <= 45 and -10 <= lon <= 45: return 'Mediterranean'
    if 25 <= lat <= 55 and -130 <= lon <= -100: return 'W. North America'
    if -45 <= lat <= -10 and 110 <= lon <= 155: return 'Australia'
    if -35 <= lat <= 10 and -80 <= lon <= -35: return 'South America'
    if 45 <= lat <= 70: return 'Boreal / N. Europe'
    if -35 <= lat <= 35 and -20 <= lon <= 55: return 'Africa'
    return 'Other'

df['Region'] = df.apply(lambda r: region(r['lat'], r['long']), axis=1)

regions_order = ['Australia','Mediterranean','W. North America',
                 'South America','Africa','Boreal / N. Europe','Other']
region_data = []
for reg in regions_order:
    sub = df[df['Region']==reg]
    n = len(sub); n_es = sub['ES'].sum()
    region_data.append((reg, n_es/n*100, n_es, n))

# === FIGURE ===
fig, ax = plt.subplots(figsize=(8, 5))
labels = [r[0] for r in region_data]
prevs = [r[1] for r in region_data]
colors = ['#d73027','#fc8d59','#fdae61','#B0BEC5','#B0BEC5','#B0BEC5','#B0BEC5']

bars = ax.bar(range(len(labels)), prevs, color=colors, edgecolor='white', linewidth=0.8)
for i, (reg, pct, n_es, n) in enumerate(region_data):
    ax.text(i, pct+1.5, f'{pct:.1f}%\n({n_es}/{n})', ha='center', fontsize=11, fontweight='bold')

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30, ha='right')
ax.set_ylabel('Empty Shell Prevalence (%)', fontweight='bold')
ax.set_ylim(0, 100)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle=':', alpha=0.3)
ax.axhline(y=36.7, color='gray', ls='--', lw=1.2, alpha=0.5, label=f'Global mean (36.7%)')
ax.legend(fontsize=11, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(TEMP_DIR, 'FigS2a_Regional_Prevalence.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {TEMP_DIR}/FigS2a_Regional_Prevalence.pdf')
for reg, pct, n_es, n in region_data:
    print(f'  {reg:<20s}: {pct:.1f}% ({n_es}/{n})')
