import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Fig S2a/S2b: ROC + PR Curves."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

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
decoupling = X_raw[:,:,0]-X_raw[:,:,2]
es_label = np.argmax([np.mean(decoupling[labels==c,W-1]) for c in range(3)])

features = ['Pre_NDVI','vpd','pr','tmmx','soil','def','pdsi','elevation','slope','aspect',
    'Sand_Content','Clay_Content','Bulk_Density','Carbon_Content','Biome_Type','Human_Footprint',
    'Max_VPD','Max_DEF','dNBR','TWI']
df_1393 = pd.read_csv('GTM_1393_Landsat_20D_Advanced.csv')
df_1393['Label'] = df_1393['Ref_ID'].map(dict(zip(df_f.index, labels)))
train = df_1393[df_1393['Label'].notna()]
y_true = (train['Label']==es_label).astype(int).values
X_tr = train[features].fillna(train[features].median())
sw = (len(y_true)-y_true.sum())/y_true.sum()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aucs = []; aps = []
tprs = []; fprs = []; precs = []; recs = []
mean_fpr = np.linspace(0,1,100); mean_rec = np.linspace(0,1,100)
tprs_interp = []; prs_interp = []

for tr,vl in cv.split(X_tr, y_true):
    model = xgb.XGBClassifier(max_depth=2, learning_rate=0.05, n_estimators=150,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=sw, random_state=42)
    model.fit(X_tr.iloc[tr], y_true[tr])
    probs = model.predict_proba(X_tr.iloc[vl])[:,1]
    fpr,tpr,_ = roc_curve(y_true[vl], probs)
    tprs_interp.append(np.interp(mean_fpr, fpr, tpr)); tprs_interp[-1][0]=0.0
    aucs.append(auc(fpr,tpr)); fprs.append(fpr); tprs.append(tpr)
    prec,rec,_ = precision_recall_curve(y_true[vl], probs)
    prs_interp.append(np.interp(mean_rec, rec[::-1], prec[::-1])[::-1])
    aps.append(average_precision_score(y_true[vl], probs)); precs.append(prec); recs.append(rec)

# ROC
fig, ax = plt.subplots(figsize=(7,6))
for i in range(5): ax.plot(fprs[i], tprs[i], alpha=0.3, lw=1, color='#d73027')
ax.plot(mean_fpr, np.mean(tprs_interp,axis=0), color='#d73027', lw=3, label='Mean (AUC=%.3f +/- %.3f)'%(np.mean(aucs),np.std(aucs)))
ax.plot([0,1],[0,1],'k--',alpha=0.3)
ax.set_xlabel('False Positive Rate',fontweight='bold'); ax.set_ylabel('True Positive Rate',fontweight='bold')
ax.legend(fontsize=12,frameon=False); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(linestyle=':',alpha=0.4)
plt.tight_layout(); plt.savefig(os.path.join(TEMP_DIR, 'FigS2a_ROC.pdf'), dpi=300,bbox_inches='tight'); plt.close()
print(f'Saved: {TEMP_DIR}/FigS2a_ROC.pdf (AUC=%.3f +/- %.3f)'%(np.mean(aucs),np.std(aucs)))

# PR
fig, ax = plt.subplots(figsize=(7,6))
for i in range(5): ax.plot(recs[i], precs[i], alpha=0.3, lw=1, color='#1a9850')
ax.plot(mean_rec, np.mean(prs_interp,axis=0), color='#1a9850', lw=3, label='Mean (AP=%.3f +/- %.3f)'%(np.mean(aps),np.std(aps)))
baseline = y_true.sum()/len(y_true)
ax.axhline(baseline, color='gray', ls='--', lw=1.5, alpha=0.5, label='Baseline (%.3f)'%baseline)
ax.set_xlabel('Recall',fontweight='bold'); ax.set_ylabel('Precision',fontweight='bold')
ax.legend(fontsize=12,frameon=False); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(linestyle=':',alpha=0.4)
plt.tight_layout(); plt.savefig(os.path.join(TEMP_DIR, 'FigS2b_PR.pdf'), dpi=300,bbox_inches='tight'); plt.close()
print(f'Saved: {TEMP_DIR}/FigS2b_PR.pdf (AP=%.3f +/- %.3f)'%(np.mean(aps),np.std(aps)))
