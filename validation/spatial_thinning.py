import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Spatial thinning robustness (Methods 5.4).
Two ES definitions compared across grid resolutions:
  Pipeline: Yan DTW K=3 labels from full 601-site clustering.
  Criterion: VOD slope < 0 AND NDVI slope > population median — computed ONCE on full data.
Both thinned, trained, predicted + Cox PH on full 1393 sites."""
import pandas as pd, numpy as np, xgboost as xgb
from lifelines import CoxPHFitter
from scipy.stats import linregress

features = ['Pre_NDVI','vpd','pr','tmmx','soil','def','pdsi','elevation','slope','aspect',
    'Sand_Content','Clay_Content','Bulk_Density','Carbon_Content','Biome_Type','Human_Footprint',
    'Max_VPD','Max_DEF','dNBR','TWI']

df1393 = pd.read_csv('GTM_1393_Landsat_20D_Advanced.csv')
coords = pd.read_csv('GTM_Master_Panel_Data_Spatial.csv')[['Ref_ID','lat','long']].drop_duplicates(subset='Ref_ID')

# Survival data
panel = pd.read_csv(os.path.join(TEMP_DIR, 'GTM_Master_Panel_Data_Final.csv'))
sites_s = panel.groupby('Ref_ID').agg(es2=('event_start','first'),gl2=('gfc_lossyear','max')).reset_index()
sites_s['al2'] = np.where(sites_s['gl2'].notna(),sites_s['gl2']+2000,np.nan)
sites_s['ttd'] = sites_s['al2']-sites_s['es2']
sites_s['died'] = np.where((sites_s['ttd']>0)&sites_s['ttd'].notna(),1,0).astype(int)
sites_s['obs'] = np.maximum(1, 2023-sites_s['es2'])
sites_s['time'] = sites_s['ttd'].where(sites_s['died']==1, sites_s['obs'])
sites_s.loc[sites_s['time']>40, 'time'] = 40

# ===========================================================================
# Build criterion labels ONCE from full 601-site tensor
# ===========================================================================
panel2 = pd.read_csv(os.path.join(TEMP_DIR, 'GTM_Master_Panel_Data_Final.csv'))
spei = pd.read_csv('GTM_Golden_1401_SPEI_12_24_36_1984_2022.csv')
panel2 = pd.merge(panel2, spei[['Ref_ID','Year','SPEI_12_month']], on=['Ref_ID','Year'], how='left')
site_info = panel2.groupby('Ref_ID').agg(
    esa=('esa_lc_2021','first'),es=('event_start','first'),gl=('gfc_lossyear','max')).reset_index()
site_info['al'] = np.where(site_info['gl'].notna(),site_info['gl']+2000,np.nan)
site_info['repeat'] = site_info['al'] > (site_info['es']+8)
forest_refs = set(site_info[(site_info['esa'].isin([10,95]))&(~site_info['repeat'])]['Ref_ID'])
site_lv = {}
for ref_id in forest_refs:
    site = panel2[panel2['Ref_ID']==ref_id].sort_values('Year'); es = site['event_start'].iloc[0]
    if pd.isna(es): continue
    post = site[(site['Year']>=es)]
    if len(post)<3: continue
    lv_idx = post['NDVI_mean'].idxmin()
    if pd.isna(lv_idx): continue
    site_lv[ref_id] = int(site.loc[lv_idx,'Year'])

metrics = ['NDVI_mean','NDII_mean','VOD_CKXU','SIF_mean']; W=8
tensor_data = {}
for ref_id, lv_year in site_lv.items():
    site = panel2[panel2['Ref_ID']==ref_id].sort_values('Year')
    traj = site[(site['Year']>=lv_year)&(site['Year']<=lv_year+W-1)]
    if len(traj)<5: continue
    row = {'Ref_ID':ref_id}
    for m in metrics:
        for i,(_,r) in enumerate(traj.iterrows()):
            if i<W: row[f'{m}_Y{i+1}']=r.get(m,np.nan)
    tensor_data[ref_id]=row
df_t = pd.DataFrame(tensor_data).T
valid_refs = [ref for ref in df_t.index if all(
    df_t.loc[ref,[f'{m}_Y{i}' for i in range(1,W+1)]].notna().sum()>=5 for m in metrics)]
df_f = df_t.loc[valid_refs]
for m in metrics:
    cols=[f'{m}_Y{i}' for i in range(1,W+1)]; df_f[cols]=df_f[cols].interpolate(method='linear',axis=1,limit_direction='both')
df_f = df_f.dropna()

# Criterion: VOD declining + NDVI above population median (computed ONCE)
ndvi_slopes = []; vod_slopes = []; W=8
for ref in df_f.index:
    ndvi = df_f.loc[ref, [f'NDVI_mean_Y{i}' for i in range(1,W+1)]].values
    vod  = df_f.loc[ref, [f'VOD_CKXU_Y{i}' for i in range(1,W+1)]].values
    ndvi_slopes.append(linregress(range(W), ndvi).slope)
    vod_slopes.append(linregress(range(W), vod).slope)
ndvi_slopes = np.array(ndvi_slopes); vod_slopes = np.array(vod_slopes)
crit_es = (vod_slopes < 0) & (ndvi_slopes > np.median(ndvi_slopes))
criterion_map = dict(zip(df_f.index, crit_es.astype(int)))
print(f"Criterion ES (full 601): {crit_es.sum()}")

# Pipeline DTW labels
pipeline_map = dict(zip(pd.read_csv(os.path.join(TEMP_DIR, 'Yan_DTW_Labels_601.csv'))['Ref_ID'],
                        pd.read_csv(os.path.join(TEMP_DIR, 'Yan_DTW_Labels_601.csv'))['Empty_Shell']))

# ===========================================================================
# Thinning loop — both ES definitions
# ===========================================================================
print(f"\n{'Res':<12s} {'Thin_N':>6s} | {'Pipe_N':>6s} {'Pipe_ES':>7s} {'Pipe_HR':>7s} {'Pipe_p':>10s} | {'Crit_N':>6s} {'Crit_ES':>7s} {'Crit_HR':>7s} {'Crit_p':>10s}")
print('-' * 95)

def run_thinning(es_map, thin_refs):
    train_refs = [r for r in es_map if r in thin_refs]
    if len(train_refs) < 30: return None
    df_all = df1393.copy()
    df_all['ES'] = df_all['Ref_ID'].map(es_map)
    train = df_all[df_all['Ref_ID'].isin(train_refs) & df_all['ES'].notna()]
    y_tr = train['ES'].astype(int).values; n_es = y_tr.sum()
    if n_es < 5 or (len(y_tr)-n_es) < 5: return None
    sw = (len(y_tr)-n_es)/n_es
    model = xgb.XGBClassifier(max_depth=2, learning_rate=0.05, n_estimators=150,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=sw, random_state=42)
    model.fit(train[features].fillna(train[features].median()), y_tr)
    probs = model.predict_proba(df_all[features].fillna(train[features].median()))[:,1]
    df_cox = pd.DataFrame({'Ref_ID':df_all['Ref_ID'],'Prob_ES':probs})
    df_cox = pd.merge(df_cox, sites_s[['Ref_ID','time','died']], on='Ref_ID', how='inner')
    df_cox['P_z'] = (df_cox['Prob_ES']-df_cox['Prob_ES'].mean())/df_cox['Prob_ES'].std()
    cph = CoxPHFitter()
    cph.fit(df_cox[['time','died','P_z']].dropna(), duration_col='time', event_col='died')
    return {'N': len(train), 'ES': n_es, 'HR': np.exp(cph.params_['P_z']),
            'p': cph.summary.loc['P_z','p']}

for RES in [None, 0.25, 0.5, 1.0, 2.0]:
    if RES is None:
        thin_refs = set(df1393['Ref_ID']); label = 'None'
    else:
        td = coords.copy(); td['glat'] = (td['lat']/RES).round(); td['glon'] = (td['long']/RES).round()
        np.random.seed(42)
        thin_refs = set(td.groupby(['glat','glon'], group_keys=False).apply(
            lambda g: g.sample(1), include_groups=False)['Ref_ID'].values)
        label = f'{RES}'

    rp = run_thinning(pipeline_map, thin_refs)
    rc = run_thinning(criterion_map, thin_refs)

    def fmt(r): return (f'{r["N"]}', f'{r["ES"]}', f'{r["HR"]:.2f}', f'{r["p"]:.2e}') if r else ('N/A','N/A','N/A','N/A')
    pn, pe, ph, pp = fmt(rp)
    cn, ce, ch, cp = fmt(rc)
    print(f'{label:<12s} {len(thin_refs):>6d} | {pn:>6s} {pe:>7s} {ph:>7s} {pp:>10s} | {cn:>6s} {ce:>7s} {ch:>7s} {cp:>10s}')

print('\nPipeline: Yan DTW K=3 labels. Criterion: VOD slope<0 + NDVI slope>median (fixed once on full 601).')
print('Both: train on thinned labeled sites, predict + Cox on all 1393. Methods 5.4.')
