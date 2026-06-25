import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Binary ES probability prediction: train on 601 Yan DTW sites, predict all 1,393.
Paper Methods 5.3. Output: Yan_XGBoost_Predictions_1393.csv"""
import pandas as pd, numpy as np, xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# === Yan DTW pipeline: 601-site training labels ===
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

# === Map labels to 1393 features ===
features = ['Pre_NDVI','vpd','pr','tmmx','soil','def','pdsi','elevation','slope','aspect',
    'Sand_Content','Clay_Content','Bulk_Density','Carbon_Content','Biome_Type','Human_Footprint',
    'Max_VPD','Max_DEF','dNBR','TWI']

df_pred = pd.read_csv('GTM_1393_Landsat_20D_Advanced.csv')
df_pred['Label'] = df_pred['Ref_ID'].map(dict(zip(df_f.index, labels)))
df_pred['ES'] = (df_pred['Label'] == es_label).astype(int)
train = df_pred[df_pred['Label'].notna()]
print(f"Training sites: {len(train)}, ES={train.ES.sum()}")

y = train['ES']
X_tr = train[features].fillna(train[features].median())
sw = (len(y)-y.sum())/y.sum()

model = xgb.XGBClassifier(max_depth=2, learning_rate=0.05, n_estimators=150,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=sw, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
yy = y.values; aucs, aps = [], []
for tr, vl in cv.split(X_tr, y):
    model.fit(X_tr.iloc[tr], yy[tr])
    proba = model.predict_proba(X_tr.iloc[vl])[:, 1]
    aucs.append(roc_auc_score(yy[vl], proba))
    aps.append(average_precision_score(yy[vl], proba))
print(f"AUC = {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
print(f"AP  = {np.mean(aps):.3f} +/- {np.std(aps):.3f}")

# === Train on full 601 and predict all 1,393 ===
model.fit(X_tr, y)
probs = model.predict_proba(df_pred[features].fillna(df_pred[features].median()))[:, 1]
df_pred['Prob_ES'] = probs
df_pred['ES_Predicted'] = (probs > 0.5).astype(int)

out = df_pred[['Ref_ID','Label','ES','Prob_ES','ES_Predicted']].copy()
out.to_csv(os.path.join(TEMP_DIR, 'Yan_XGBoost_Predictions_1393.csv'), index=False)
print(f"ES predicted (Prob>0.5): {out.ES_Predicted.sum()} / {len(out)}")
print("Saved: Yan_XGBoost_Predictions_1393.csv")
