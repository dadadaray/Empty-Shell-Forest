import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""
Yan-aligned window stability (3-10yr).
Trains XGBoost on DTW-labeled sites, predicts on ALL 1393, Cox PH on 1393.
The 1393 set is NOT filtered for repeat-death -> unbiased mortality signal.
"""
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
    site = panel[panel['Ref_ID']==ref_id].sort_values('Year'); es = site['event_start'].iloc[0]
    if pd.isna(es): continue
    post = site[(site['Year']>=es)]
    if len(post)<3: continue
    lv_idx = post['NDVI_mean'].idxmin()
    if pd.isna(lv_idx): continue
    site_lv[ref_id] = int(site.loc[lv_idx,'Year'])

metrics = ['NDVI_mean','NDII_mean','VOD_CKXU','SIF_mean']
surv = panel.groupby('Ref_ID').agg(es2=('event_start','first'),gl2=('gfc_lossyear','max')).reset_index()
surv['al2'] = np.where(surv['gl2'].notna(),surv['gl2']+2000,np.nan)
surv['ttd'] = surv['al2']-surv['es2']
surv['died'] = np.where((surv['ttd']>0)&surv['ttd'].notna(),1,0).astype(int)
surv['obs'] = np.maximum(1,2023-surv['es2'])
surv['time'] = surv['ttd'].where(surv['died']==1,surv['obs'])
surv.loc[surv['time']>40,'time']=40

features = ['Pre_NDVI','vpd','pr','tmmx','soil','def','pdsi','elevation','slope','aspect',
    'Sand_Content','Clay_Content','Bulk_Density','Carbon_Content','Biome_Type','Human_Footprint',
    'Max_VPD','Max_DEF','dNBR','TWI']
df_all = pd.read_csv('GTM_1393_Landsat_20D_Advanced.csv')
X_all = df_all[features].fillna(df_all[features].median())

# Header
print(f'{"W":>2s} {"Ntr":>4s} {"Label":>6s} {"Pattern":>16s} {"n":>5s} {"NDVI_s":>7s} {"VOD_s":>7s} '
      f'{"AUC":>6s} {"HR_1393":>7s} {"p":>7s} {"L_mort":>6s} {"M_mort":>6s} {"H_mort":>6s} {"Gap":>6s}')
print('-'*100)

for W in range(3, 11):
    # Build tensor
    tensor_data = {}
    for ref_id, lv_year in site_lv.items():
        site = panel[panel['Ref_ID']==ref_id].sort_values('Year')
        traj = site[(site['Year']>=lv_year)&(site['Year']<=lv_year+W-1)]
        if len(traj) < max(3, W-2): continue
        row = {'Ref_ID':ref_id}
        for m in metrics:
            for i,(_,r) in enumerate(traj.iterrows()):
                if i<W: row[f'{m}_Y{i+1}'] = r.get(m,np.nan)
        tensor_data[ref_id] = row

    df_t = pd.DataFrame(tensor_data).T
    # SAME filter as yan_full_pipeline_nothin.py: >=5 valid years
    min_yr = min(W, 5)  # require >=5 valid for W>=5, or all W years for W<5
    valid_refs = [ref for ref in df_t.index if all(
        df_t.loc[ref,[f'{m}_Y{i}' for i in range(1,W+1)]].notna().sum()>=min_yr for m in metrics)]
    if len(valid_refs) < 30:
        print(f'{W:2d} (too few sites: {len(valid_refs)})')
        continue
    df_f = df_t.loc[valid_refs]
    for m in metrics:
        cols = [f'{m}_Y{i}' for i in range(1,W+1)]
        df_f[cols] = df_f[cols].interpolate(method='linear',axis=1,limit_direction='both')
    df_f = df_f.dropna()
    n_dtw = len(df_f)

    # DTW K=3
    X_t = np.zeros((n_dtw,W,4))
    for fi,m in enumerate(metrics):
        X_t[:,:,fi] = df_f[[f'{m}_Y{i}' for i in range(1,W+1)]].values
    X_raw = X_t.copy()
    X_s = TimeSeriesScalerMeanVariance().fit_transform(X_t)
    labels = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=10, random_state=42, n_jobs=-1).fit_predict(X_s)

    # Map to 1393, train XGBoost, Cox PH on 1393
    label_map = dict(zip(df_f.index, labels))
    df_pred = df_all.copy()
    df_pred['Label'] = df_pred['Ref_ID'].map(label_map)
    train = df_pred[df_pred['Label'].notna()]
    X_tr = train[features].fillna(train[features].median())

    for a in range(3):
        mask = labels==a; n=mask.sum()
        ndvi_s = np.mean([np.polyfit(range(1,W+1),X_raw[i,:,0],1)[0] for i in np.where(mask)[0]])
        vod_s = np.mean([np.polyfit(range(1,W+1),X_raw[i,:,3],1)[0] for i in np.where(mask)[0]])

        if ndvi_s>0.002 and vod_s<-0.001:
            pat = 'EMPTY SHELL'
        elif ndvi_s>0.002 and vod_s>0.001:
            pat = 'SYNCHRONOUS'
        elif ndvi_s>0 and abs(vod_s)<=0.001:
            pat = 'NDVI-UP VOD-FLAT'
        else:
            pat = 'SLOW/OTHER'

        y_tr = (train['Label']==a).astype(int); yy = y_tr.values; n_pos = yy.sum()
        if n_pos < 5: continue

        sw = (len(yy)-n_pos)/n_pos
        model = xgb.XGBClassifier(max_depth=2, learning_rate=0.05, n_estimators=150,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=sw, random_state=42)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = []
        for tr,vl in cv.split(X_tr,y_tr):
            model.fit(X_tr.iloc[tr], yy[tr])
            aucs.append(roc_auc_score(yy[vl], model.predict_proba(X_tr.iloc[vl])[:,1]))

        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_all)[:,1]
        df_pred[f'P{a}'] = probs

        # Cox PH on 1393
        df_c = df_pred.merge(surv[['Ref_ID','time','died']], on='Ref_ID', how='inner')
        df_c[f'P{a}_z'] = (df_c[f'P{a}']-df_c[f'P{a}'].mean())/df_c[f'P{a}'].std()

        cph = CoxPHFitter()
        cph.fit(df_c[['time','died',f'P{a}_z']].dropna(), duration_col='time', event_col='died')
        hr = np.exp(cph.params_[f'P{a}_z']); p = cph.summary.loc[f'P{a}_z','p']

        # Tertile mortality on 1393
        df_c['Tert'] = pd.qcut(df_c[f'P{a}'], 3, labels=['L','M','H'])
        tm = {t: df_c[df_c['Tert']==t]['died'].mean()*100 for t in ['L','M','H']}
        gap = tm['H'] - tm['L']

        sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
        flag = ' <-- ES' if pat == 'EMPTY SHELL' else ''

        print(f'{W:2d} {len(train):4d} {"L"+str(a):>6s} {pat:16s} {n:5d} {ndvi_s*1000:+6.1f}e-3 {vod_s*1000:+6.1f}e-3 '
              f'{np.mean(aucs):5.3f} {hr:6.2f} {p:.4f} {sig} {tm["L"]:5.1f}% {tm["M"]:5.1f}% {tm["H"]:5.1f}% {gap:+5.1f}%{flag}')

    print()
