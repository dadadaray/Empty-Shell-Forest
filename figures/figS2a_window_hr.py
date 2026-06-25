import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Fig S2a: Window Robustness — ES HR across W=5-10 (K=3, Yan-aligned, 1393-site Cox PH)."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
import xgboost as xgb
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],
    'font.size':14,'axes.labelsize':14,'axes.labelweight':'bold',
    'xtick.labelsize':12,'ytick.labelsize':12,
    'axes.linewidth':1.2,'pdf.fonttype':42,'ps.fonttype':42
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

metrics = ['NDVI_mean','NDII_mean','VOD_CKXU','SIF_mean']
features = ['Pre_NDVI','vpd','pr','tmmx','soil','def','pdsi','elevation','slope','aspect',
    'Sand_Content','Clay_Content','Bulk_Density','Carbon_Content','Biome_Type','Human_Footprint',
    'Max_VPD','Max_DEF','dNBR','TWI']
df_1393 = pd.read_csv('GTM_1393_Landsat_20D_Advanced.csv')
X_all = df_1393[features].fillna(df_1393[features].median())

sites_s = panel.groupby('Ref_ID').agg(es2=('event_start','first'),gl2=('gfc_lossyear','max')).reset_index()
sites_s['al2'] = np.where(sites_s['gl2'].notna(),sites_s['gl2']+2000,np.nan)
sites_s['ttd'] = sites_s['al2']-sites_s['es2']
sites_s['died'] = np.where((sites_s['ttd']>0)&sites_s['ttd'].notna(),1,0).astype(int)
sites_s['obs'] = np.maximum(1,2023-sites_s['es2'])
sites_s['time'] = sites_s['ttd'].where(sites_s['died']==1,sites_s['obs'])
sites_s.loc[sites_s['time']>40,'time']=40

hrs = []; ci_lows = []; ci_highs = []; aucs_out = []; windows = list(range(3,11))

print('W   N_train  AUC    HR     CI_low  CI_high')
print('-'*45)
for W_test in windows:
    tensor_data = {}
    for ref_id, lv_year in site_lv.items():
        site = panel[panel['Ref_ID']==ref_id].sort_values('Year')
        traj = site[(site['Year']>=lv_year)&(site['Year']<=lv_year+W_test-1)]
        if len(traj)<max(3,W_test-2): continue
        row = {'Ref_ID':ref_id}
        for m in metrics:
            for i,(_,r) in enumerate(traj.iterrows()):
                if i<W_test: row[f'{m}_Y{i+1}']=r.get(m,np.nan)
        tensor_data[ref_id]=row
    df_t = pd.DataFrame(tensor_data).T
    min_v = min(W_test,5)
    valid_refs = [ref for ref in df_t.index if all(df_t.loc[ref,[f'{m}_Y{i}' for i in range(1,W_test+1)]].notna().sum()>=min_v for m in metrics)]
    if len(valid_refs)<30: continue
    df_f = df_t.loc[valid_refs]
    for m in metrics:
        cols=[f'{m}_Y{i}' for i in range(1,W_test+1)]; df_f[cols]=df_f[cols].interpolate(method='linear',axis=1,limit_direction='both')
    df_f = df_f.dropna()
    X_raw = np.zeros((len(df_f),W_test,4))
    for fi,m in enumerate(metrics): X_raw[:,:,fi] = df_f[[f'{m}_Y{i}' for i in range(1,W_test+1)]].values
    X_s = TimeSeriesScalerMeanVariance().fit_transform(X_raw)
    labels = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=10, random_state=42, n_jobs=-1).fit_predict(X_s)
    dec = X_raw[:,:,0]-X_raw[:,:,2]
    end_dec = [np.mean(dec[labels==c,W_test-1]) for c in range(3)]
    es_id = np.argmax(end_dec)
    label_map = dict(zip(df_f.index, labels))
    train = df_1393.copy(); train['Label'] = train['Ref_ID'].map(label_map)
    train = train.dropna(subset=['Label'])
    y_tr = (train['Label']==es_id).astype(int).values
    if y_tr.sum()<5: continue
    X_tr = train[features].fillna(train[features].median())
    sw = (len(y_tr)-y_tr.sum())/y_tr.sum()
    model = xgb.XGBClassifier(max_depth=2, learning_rate=0.05, n_estimators=150,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=sw, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr,vl in cv.split(X_tr, y_tr):
        model.fit(X_tr.iloc[tr], y_tr[tr]); aucs.append(roc_auc_score(y_tr[vl], model.predict_proba(X_tr.iloc[vl])[:,1]))
    model.fit(X_tr, y_tr); probs = model.predict_proba(X_all)[:,1]
    df_cox = pd.DataFrame({'Ref_ID':df_1393['Ref_ID'],'Prob_ES':probs})
    df_cox = pd.merge(df_cox, sites_s[['Ref_ID','time','died']], on='Ref_ID', how='inner')
    df_cox['P_z'] = (df_cox['Prob_ES']-df_cox['Prob_ES'].mean())/df_cox['Prob_ES'].std()
    cph = CoxPHFitter(); cph.fit(df_cox[['time','died','P_z']].dropna(), duration_col='time', event_col='died')
    hr = np.exp(cph.params_['P_z']); ci = cph.confidence_intervals_.loc['P_z']
    hr_lo = np.exp(ci.iloc[0]); hr_hi = np.exp(ci.iloc[1])
    hrs.append(hr); ci_lows.append(hr_lo); ci_highs.append(hr_hi); aucs_out.append(np.mean(aucs))
    print('%d   %4d    %.3f  %.2f  %.2f    %.2f'%(W_test,len(train),np.mean(aucs),hr,hr_lo,hr_hi))

# Plot
fig, ax = plt.subplots(figsize=(7,5))
ax.fill_between(windows, ci_lows, ci_highs, color='#d73027', alpha=0.15)
ax.errorbar(windows, hrs, yerr=[[h-l for h,l in zip(hrs,ci_lows)],[u-h for h,u in zip(hrs,ci_highs)]],
            fmt='o-', color='#d73027', lw=3.0, markersize=12, markerfacecolor='white',
            markeredgewidth=2.5, capsize=6, capthick=2)
ax.axhline(1.0, color='gray', ls='--', lw=1.5, alpha=0.5)
for w,hr in zip(windows,hrs):
    ax.text(w, hr+0.12, '%.2f'%hr, ha='center', fontweight='bold', fontsize=12, color='#d73027')
ax.set_xlabel('Window Length W (years)', fontweight='bold')
ax.set_ylabel('Hazard Ratio (1393-site Cox PH)', fontweight='bold')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle=':', alpha=0.4)
ax.set_xticks(windows)
plt.tight_layout(); plt.savefig(os.path.join(TEMP_DIR, 'FigS2a_Window_HR.pdf'), dpi=300, bbox_inches='tight'); plt.close()
print('\nSaved: FigS2a_Window_HR.pdf')
