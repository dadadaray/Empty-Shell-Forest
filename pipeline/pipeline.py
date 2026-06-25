import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Yan-aligned DTW -> XGBoost -> 1393 prediction -> Cox PH."""
import pandas as pd, numpy as np
from lifelines import CoxPHFitter
import xgboost as xgb
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

panel = pd.read_csv(os.path.join(TEMP_DIR, 'GTM_Master_Panel_Data_Final.csv'))
spei = pd.read_csv('GTM_Golden_1401_SPEI_12_24_36_1984_2022.csv')
panel = pd.merge(panel, spei[['Ref_ID','Year','SPEI_12_month']], on=['Ref_ID','Year'], how='left')

site_info = panel.groupby('Ref_ID').agg(esa=('esa_lc_2021','first'),es=('event_start','first'),gl=('gfc_lossyear','max')).reset_index()
site_info['al'] = np.where(site_info['gl'].notna(),site_info['gl']+2000,np.nan)
site_info['repeat'] = site_info['al'] > (site_info['es']+8)
forest_refs = set(site_info[(site_info['esa'].isin([10,95]))&(~site_info['repeat'])]['Ref_ID'])

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

metrics = ['NDVI_mean','NDII_mean','VOD_CKXU','SIF_mean']
tensor_data = {}
for ref_id, lv_year in site_lv.items():
    site = panel[panel['Ref_ID']==ref_id].sort_values('Year')
    traj = site[(site['Year']>=lv_year)&(site['Year']<=lv_year+7)]
    if len(traj)<5: continue
    row = {'Ref_ID':ref_id}
    for m in metrics:
        for i,(_,r) in enumerate(traj.iterrows()):
            if i<8: row[f'{m}_Y{i+1}'] = r.get(m,np.nan)
    tensor_data[ref_id] = row

df_t = pd.DataFrame(tensor_data).T
valid_refs = [ref for ref in df_t.index if all(
    df_t.loc[ref,[f'{m}_Y{i}' for i in range(1,9)]].notna().sum()>=5 for m in metrics)]
df_f = df_t.loc[valid_refs]
for m in metrics:
    cols = [f'{m}_Y{i}' for i in range(1,9)]
    df_f[cols] = df_f[cols].interpolate(method='linear',axis=1,limit_direction='both')
df_f = df_f.dropna()
df_f.to_csv(os.path.join(TEMP_DIR, 'Yan_DTW_Tensor_601.csv'))
print(f'Saved: Yan_DTW_Tensor_601.csv ({len(df_f)} sites x 4 metrics x 8 years)')

# DTW K=3
X = np.zeros((len(df_f),8,4))
for fi,m in enumerate(metrics):
    X[:,:,fi] = df_f[[f'{m}_Y{i}' for i in range(1,9)]].values
X_raw = X.copy()
X_s = TimeSeriesScalerMeanVariance().fit_transform(X)
labels = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=10, random_state=42, n_jobs=-1).fit_predict(X_s)

# Find Empty Shell
es_label = None
for a in range(3):
    mask = labels==a
    ndvi_s = np.mean([np.polyfit(range(1,9),X_raw[i,:,0],1)[0] for i in np.where(mask)[0]])
    vod_s = np.mean([np.polyfit(range(1,9),X_raw[i,:,3],1)[0] for i in np.where(mask)[0]])
    if ndvi_s>0.002 and vod_s<0: es_label=a; break

if es_label is None:
    surv_t = panel.groupby('Ref_ID').agg(es2=('event_start','first'),gl2=('gfc_lossyear','max')).reset_index()
    surv_t['al2'] = np.where(surv_t['gl2'].notna(),surv_t['gl2']+2000,np.nan)
    surv_t['died'] = np.where((surv_t['al2']>surv_t['es2'])&surv_t['al2'].notna(),1,0).astype(int)
    df_tmp = pd.DataFrame({'Ref_ID':df_f.index,'Label':labels})
    df_tmp = pd.merge(df_tmp, surv_t[['Ref_ID','died']],on='Ref_ID',how='inner')
    mort = {a:df_tmp[df_tmp['Label']==a]['died'].mean() for a in range(3)}
    es_label = max(mort,key=mort.get)

print(f'DTW sites: {len(df_f)}, Empty Shell = Label {es_label}')

# Save DTW labels
df_labels = pd.DataFrame({'Ref_ID': df_f.index, 'DTW_Label': labels,
                          'Empty_Shell': (labels == es_label).astype(int)})
df_labels.to_csv(os.path.join(TEMP_DIR, 'Yan_DTW_Labels_601.csv'), index=False)
print(f'Saved: Yan_DTW_Labels_601.csv')

for a in range(3):
    mask = labels==a
    ndvi_s = np.mean([np.polyfit(range(1,9),X_raw[i,:,0],1)[0] for i in np.where(mask)[0]])
    vod_s = np.mean([np.polyfit(range(1,9),X_raw[i,:,3],1)[0] for i in np.where(mask)[0]])
    print(f'  L{a}: n={mask.sum():3d}, NDVI_s={ndvi_s*1000:+.1f}e-3, VOD_s={vod_s*1000:+.1f}e-3')

# Merge with 1393 GEE features for training
label_map = dict(zip(df_f.index, labels))
df_pred = pd.read_csv('GTM_1393_Landsat_20D_Advanced.csv')
df_pred['Label'] = df_pred['Ref_ID'].map(label_map)
df_pred['ES'] = (df_pred['Label'] == es_label).astype(int)
train = df_pred[df_pred['Label'].notna()]
n_train = len(train); n_es = train['ES'].sum()
print(f'\nTraining sites (Yan-labeled + GEE features): {n_train}, ES={n_es}')

features = ['Pre_NDVI','vpd','pr','tmmx','soil','def','pdsi','elevation','slope','aspect',
    'Sand_Content','Clay_Content','Bulk_Density','Carbon_Content','Biome_Type','Human_Footprint',
    'Max_VPD','Max_DEF','dNBR','TWI']

if n_es >= 5:
    y = train['ES']
    X_tr = train[features].fillna(train[features].median())
    sw = (len(y)-y.sum())/y.sum()
    model = xgb.XGBClassifier(max_depth=2, learning_rate=0.05, n_estimators=150,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=sw, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    yy = y.values
    aucs = []
    for tr,vl in cv.split(X_tr,y):
        model.fit(X_tr.iloc[tr], yy[tr])
        aucs.append(roc_auc_score(yy[vl], model.predict_proba(X_tr.iloc[vl])[:,1]))
    print(f'XGBoost AUC: {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}')

    model.fit(X_tr, y)
    probs = model.predict_proba(df_pred[features].fillna(df_pred[features].median()))[:,1]

    surv = panel.groupby('Ref_ID').agg(es2=('event_start','first'),gl2=('gfc_lossyear','max')).reset_index()
    surv['al2'] = np.where(surv['gl2'].notna(),surv['gl2']+2000,np.nan)
    surv['ttd'] = surv['al2']-surv['es2']
    surv['died'] = np.where((surv['ttd']>0)&surv['ttd'].notna(),1,0).astype(int)
    surv['obs'] = np.maximum(1,2023-surv['es2'])
    surv['time'] = surv['ttd'].where(surv['died']==1,surv['obs'])
    surv.loc[surv['time']>40,'time']=40

    df_cox = pd.DataFrame({'Ref_ID':df_pred['Ref_ID'],'Prob_ES':probs})
    df_cox = pd.merge(df_cox, surv[['Ref_ID','time','died']], on='Ref_ID', how='inner')
    df_cox['P_z'] = (df_cox['Prob_ES'] - df_cox['Prob_ES'].mean())/df_cox['Prob_ES'].std()
    nn = len(df_cox); dd = df_cox['died'].sum()

    df_cox['Tertile'] = pd.qcut(df_cox['Prob_ES'], 3, labels=['Low','Mid','High'])
    print(f'\nCox PH: {nn} sites, {dd} deaths')
    for t in ['Low','Mid','High']:
        sub = df_cox[df_cox['Tertile']==t]
        print(f'  {t}: n={len(sub)}, deaths={sub["died"].sum()}/{len(sub)}={sub["died"].mean()*100:.1f}%')

    cph = CoxPHFitter()
    cph.fit(df_cox[['time','died','P_z']].dropna(), duration_col='time', event_col='died')
    hr = np.exp(cph.params_['P_z']); p = cph.summary.loc['P_z','p']
    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    print(f'HR={hr:.2f}, p={p:.4f} {sig}')
    print(f'\nFINAL: pipeline -> HR={hr:.2f}')
else:
    print('Too few ES for XGBoast')
