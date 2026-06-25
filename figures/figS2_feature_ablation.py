import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Fig S2d: SHAP Ablation without Biome_Type. 8 output PDFs."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xgboost as xgb, shap, statsmodels.api as sm
from patsy import dmatrix
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from matplotlib.patches import Patch

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],
    'font.size':14,'axes.titlesize':14,'axes.titleweight':'bold',
    'axes.labelsize':14,'axes.labelweight':'bold','xtick.labelsize':12,
    'ytick.labelsize':12,'legend.fontsize':12,'axes.linewidth':1.0,
    'xtick.direction':'out','ytick.direction':'out',
    'pdf.fonttype':42,'ps.fonttype':42
})

# === DATA PREP (identical to fig2_shap) ===
panel = pd.read_csv(os.path.join(TEMP_DIR, 'GTM_Master_Panel_Data_Final.csv'))
spei = pd.read_csv('GTM_Golden_1401_SPEI_12_24_36_1984_2022.csv')
panel = pd.merge(panel, spei[['Ref_ID','Year','SPEI_12_month']], on=['Ref_ID','Year'], how='left')
site_info = panel.groupby('Ref_ID').agg(esa=('esa_lc_2021','first'),es=('event_start','first'),gl=('gfc_lossyear','max')).reset_index()
site_info['al'] = np.where(site_info['gl'].notna(),site_info['gl']+2000,np.nan)
site_info['repeat'] = site_info['al'] > (site_info['es']+8)
forest_refs = set(site_info[(site_info['esa'].isin([10,95]))&(~site_info['repeat'])]['Ref_ID'])

metrics = ['NDVI_mean','NDII_mean','VOD_CKXU','SIF_mean']; W=8
site_lv = {}
for ref_id in forest_refs:
    site = panel[panel['Ref_ID']==ref_id].sort_values('Year'); es = site['event_start'].iloc[0]
    if pd.isna(es): continue
    post = site[(site['Year']>=es)]
    if len(post)<3: continue
    lv_idx = post['NDVI_mean'].idxmin()
    if pd.isna(lv_idx): continue
    site_lv[ref_id] = int(site.loc[lv_idx,'Year'])

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

# Features WITHOUT Biome_Type
features = ['Pre_NDVI','vpd','pr','tmmx','soil','def','pdsi','elevation','slope','aspect',
    'Sand_Content','Clay_Content','Bulk_Density','Carbon_Content','Human_Footprint',
    'Max_VPD','Max_DEF','dNBR','TWI']

feature_categories = {
    'Climate': ['vpd','pr','tmmx','def','pdsi','Max_VPD','Max_DEF'],
    'Topography': ['elevation','slope','aspect','TWI'],
    'Soil': ['Sand_Content','Clay_Content','Bulk_Density','Carbon_Content','soil'],
    'Vegetation/Disturb.': ['Pre_NDVI','dNBR'],
    'Anthropogenic': ['Human_Footprint'],
}
cat_colors = {'Climate':'#4575b4','Topography':'#8c564b','Soil':'#fdae61',
              'Vegetation/Disturb.':'#1a9850','Anthropogenic':'#d73027'}
feat_to_cat = {}
for cat, feats in feature_categories.items():
    for f in feats: feat_to_cat[f] = cat

df_1393 = pd.read_csv('GTM_1393_Landsat_20D_Advanced.csv')
df_1393['Label'] = df_1393['Ref_ID'].map(dict(zip(df_f.index, labels)))
train = df_1393[df_1393['Label'].notna()]
y_tr = (train['Label']==es_label).astype(int); n_pos = y_tr.sum()
X_tr = train[features].fillna(train[features].median())
sw = (len(y_tr)-n_pos)/n_pos

model = xgb.XGBClassifier(max_depth=2, learning_rate=0.05, n_estimators=150,
    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=sw, random_state=42)
model.fit(X_tr, y_tr.values)

explainer = shap.TreeExplainer(model)
shap_vals = explainer(X_tr)
smat = shap_vals.values if hasattr(shap_vals,'values') else shap_vals
mean_abs = np.abs(smat).mean(axis=0)
idx_sorted = np.argsort(mean_abs)[::-1]
feat_names_sorted = [features[i] for i in idx_sorted]
feat_importance = mean_abs[idx_sorted]
feat_cats = [feat_to_cat.get(features[i],'Climate') for i in idx_sorted]
feat_colors_arr = [cat_colors[c] for c in feat_cats]

feature_labels_full = {
    'tmmx':'Maximum Temperature (C)','pr':'Precipitation (mm)',
    'vpd':'Vapor Pressure Deficit (hPa)','def':'Drought Severity Index',
    'pdsi':'Palmer Drought Severity Index','soil':'Soil Moisture (mm)',
    'Max_VPD':'Maximum VPD (hPa)','Max_DEF':'Maximum Drought Index',
    'elevation':'Elevation (m)','slope':'Slope (deg)','aspect':'Aspect (deg)',
    'TWI':'Topographic Wetness Index','Sand_Content':'Sand Content (%)',
    'Clay_Content':'Clay Content (%)','Bulk_Density':'Bulk Density (kg/m3)',
    'Carbon_Content':'Soil Carbon (g/kg)','Pre_NDVI':'Pre-disturbance NDVI',
    'dNBR':'Burn Severity (dNBR)','Human_Footprint':'Human Footprint Index',
}
feature_labels_short = {}
for k,v in feature_labels_full.items():
    feature_labels_short[k] = v.split(' (')[0] if ' (' in v else v

def clean_name(f, wu=True):
    return feature_labels_full.get(f,f) if wu else feature_labels_short.get(f,f)

clean_names_short = [clean_name(f, wu=False) for f in feat_names_sorted]
clean_names_full = [clean_name(f, wu=True) for f in feat_names_sorted]

# B-Spline helper
def plot_spline_ci(ax, x, y, df_val=5, sc='#4575b4'):
    idx_s = np.argsort(x)
    xs = x.values[idx_s] if hasattr(x,'values') else np.array(x)[idx_s]
    ys = y.values[idx_s] if hasattr(y,'values') else np.array(y)[idx_s]
    v = ~(np.isnan(xs)|np.isnan(ys)); xs=xs[v]; ys=ys[v]
    if len(xs)<20: return
    xspl = dmatrix("bs(x_train, df=%d, degree=3, include_intercept=False)"%df_val,
                   {"x_train":xs}, return_type='dataframe')
    xspl = sm.add_constant(xspl)
    m = sm.OLS(ys, xspl).fit()
    pred = m.get_prediction(xspl); ci = pred.summary_frame(alpha=0.05)
    ax.scatter(x, y, color=sc, s=18, alpha=0.5, edgecolor='none', zorder=1)
    ax.plot(xs, ci['mean'], color='#404040', lw=3.0, zorder=3)
    ax.fill_between(xs, ci['mean_ci_lower'], ci['mean_ci_upper'],
                    color='#7F7F7F', alpha=0.30, edgecolor='none', zorder=2)
    ax.axhline(0, linestyle='--', color='#B0BEC5', lw=1.5, zorder=0)

# === S2D_a: Importance Bar ===
fig, ax = plt.subplots(figsize=(10, 8))
top_n = 19; y_pos = range(top_n-1,-1,-1)
ax.barh(y_pos, feat_importance[:top_n], color=feat_colors_arr[:top_n],
        edgecolor='white', linewidth=0.5, height=0.7)
ax.set_yticks(list(y_pos)); ax.set_yticklabels(clean_names_short[:top_n])
ax.set_xlabel('Mean |SHAP| Value', fontweight='bold')
le = [Patch(facecolor=c, edgecolor='white', linewidth=0.5, label=cat) for cat,c in cat_colors.items()]
ax.legend(handles=le, fontsize=12, frameon=False, loc='lower right')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(axis='x', linestyle=':', alpha=0.4)
plt.tight_layout(); plt.savefig(os.path.join(TEMP_DIR, 'S2D_Ablation_Importance.pdf'), dpi=300, bbox_inches='tight'); plt.close()
print(f'Saved: {TEMP_DIR}/S2D_Ablation_Importance.pdf')

# === S2D_b: Beeswarm ===
fig = plt.figure(figsize=(12, 9))
top19_idx = idx_sorted[:19]
Xd = X_tr[[features[i] for i in top19_idx]].copy()
Xd.columns = [clean_name(f, wu=False) for f in [features[i] for i in top19_idx]]
shap.summary_plot(smat[:, top19_idx], Xd, max_display=19, show=False, plot_size=None)
plt.tight_layout(); plt.savefig(os.path.join(TEMP_DIR, 'S2D_Ablation_Beeswarm.pdf'), dpi=300, bbox_inches='tight'); plt.close()
print(f'Saved: {TEMP_DIR}/S2D_Ablation_Beeswarm.pdf')

# === S2D_c-h: Top 6 Dependence ===
for i in range(6):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    feat = feat_names_sorted[i]; fname_full = clean_names_full[i]
    f_idx = features.index(feat)
    xv = X_tr[feat]; yv = pd.Series(smat[:, f_idx])
    sc = cat_colors.get(feat_to_cat.get(feat,'Climate'),'#4575b4')
    plot_spline_ci(ax, xv, yv, df_val=5, sc=sc)
    ax.set_xlabel(fname_full, fontweight='bold')
    ax.set_ylabel('SHAP Value', fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    lbl = ['c','d','e','f','g','h'][i]
    plt.savefig(os.path.join(TEMP_DIR, 'S2D_Ablation_%s.pdf'%lbl), dpi=300, bbox_inches='tight'); plt.close()
    name = f'S2D_Ablation_{lbl}.pdf'; print(f'Saved: {TEMP_DIR}/{name}')

print('\nTop 6 SHAP (Biome removed):')
for i in range(6):
    print('  %d. %s: %.4f'%(i+1, feat_names_sorted[i], feat_importance[i]))
