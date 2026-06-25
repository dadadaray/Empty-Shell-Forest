import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Figure 2: SHAP dependence analysis with polynomial regression and 95% confidence intervals."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import statsmodels.api as sm
from patsy import dmatrix
import xgboost as xgb, shap
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],
    'font.size':14,'axes.titlesize':14,'axes.titleweight':'bold',
    'axes.labelsize':14,'axes.labelweight':'bold','xtick.labelsize':14,
    'ytick.labelsize':14,'legend.fontsize':12,'axes.linewidth':1.0,
    'xtick.direction':'out','ytick.direction':'out',
    'pdf.fonttype':42,'ps.fonttype':42
})

# ============================================
# DATA PREP (same as before)
# ============================================
panel = pd.read_csv(os.path.join(TEMP_DIR, 'GTM_Master_Panel_Data_Final.csv'))
spei = pd.read_csv('GTM_Golden_1401_SPEI_12_24_36_1984_2022.csv')
panel = pd.merge(panel, spei[['Ref_ID','Year','SPEI_12_month']], on=['Ref_ID','Year'], how='left')
site_info = panel.groupby('Ref_ID').agg(esa=('esa_lc_2021','first'),es=('event_start','first'),gl=('gfc_lossyear','max')).reset_index()
site_info['al'] = np.where(site_info['gl'].notna(),site_info['gl']+2000,np.nan)
site_info['repeat'] = site_info['al'] > (site_info['es']+8)
forest_refs = set(site_info[(site_info['esa'].isin([10,95]))&(~site_info['repeat'])]['Ref_ID'])

metrics = ['NDVI_mean','NDII_mean','VOD_CKXU','SIF_mean']; W=8
# Yan-aligned DTW pipeline (NDVI-minimum-based recovery onset)
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
    cols=[f'{m}_Y{i}' for i in range(1,W+1)]
    df_f[cols]=df_f[cols].interpolate(method='linear',axis=1,limit_direction='both')
df_f = df_f.dropna()

X_raw = np.zeros((len(df_f),W,4))
for fi,m in enumerate(metrics):
    X_raw[:,:,fi] = df_f[[f'{m}_Y{i}' for i in range(1,W+1)]].values
X_s = TimeSeriesScalerMeanVariance().fit_transform(X_raw)
labels = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=10, random_state=42, n_jobs=-1).fit_predict(X_s)
es_label = np.argmax([np.mean((X_raw[:,:,0]-X_raw[:,:,2])[labels==c,W-1]) for c in range(3)])

features = ['Pre_NDVI','vpd','pr','tmmx','soil','def','pdsi','elevation','slope','aspect',
    'Sand_Content','Clay_Content','Bulk_Density','Carbon_Content','Biome_Type','Human_Footprint',
    'Max_VPD','Max_DEF','dNBR','TWI']
feature_categories = {
    'Climate': ['vpd','pr','tmmx','def','pdsi','Max_VPD','Max_DEF'],
    'Topography': ['elevation','slope','aspect','TWI'],
    'Soil': ['Sand_Content','Clay_Content','Bulk_Density','Carbon_Content','soil'],
    'Vegetation/Disturb.': ['Pre_NDVI','dNBR','Biome_Type'],
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
feat_cats = [feat_to_cat[features[i]] for i in idx_sorted]
feat_colors_arr = [cat_colors[c] for c in feat_cats]

feature_labels_full = {
    'tmmx': 'Maximum Temperature (°C)',
    'pr': 'Precipitation (mm)',
    'vpd': 'Vapor Pressure Deficit (hPa)',
    'def': 'Drought Severity Index',
    'pdsi': 'Palmer Drought Severity Index',
    'soil': 'Soil Moisture (mm)',
    'Max_VPD': 'Maximum VPD (hPa)',
    'Max_DEF': 'Maximum Drought Index',
    'elevation': 'Elevation (m)',
    'slope': 'Slope (°)',
    'aspect': 'Aspect (°)',
    'TWI': 'Topographic Wetness Index',
    'Sand_Content': 'Sand Content (%)',
    'Clay_Content': 'Clay Content (%)',
    'Bulk_Density': 'Bulk Density (kg/m³)',
    'Carbon_Content': 'Soil Carbon (g/kg)',
    'Pre_NDVI': 'Pre-disturbance NDVI',
    'dNBR': 'Burn Severity (dNBR)',
    'Biome_Type': 'Biome Type',
    'Human_Footprint': 'Human Footprint Index',
}
feature_labels_short = {
    'tmmx': 'Maximum Temperature',
    'pr': 'Precipitation',
    'vpd': 'Vapor Pressure Deficit',
    'def': 'Drought Severity Index',
    'pdsi': 'Palmer Drought Severity Index',
    'soil': 'Soil Moisture',
    'Max_VPD': 'Maximum VPD',
    'Max_DEF': 'Maximum Drought Index',
    'elevation': 'Elevation',
    'slope': 'Slope',
    'aspect': 'Aspect',
    'TWI': 'Topographic Wetness Index',
    'Sand_Content': 'Sand Content',
    'Clay_Content': 'Clay Content',
    'Bulk_Density': 'Bulk Density',
    'Carbon_Content': 'Soil Carbon',
    'Pre_NDVI': 'Pre-disturbance NDVI',
    'dNBR': 'Burn Severity',
    'Biome_Type': 'Biome Type',
    'Human_Footprint': 'Human Footprint',
}
def clean_name(f, with_unit=True):
    return feature_labels_full.get(f, f) if with_unit else feature_labels_short.get(f, f)
clean_names_short = [clean_name(f, with_unit=False) for f in feat_names_sorted]
clean_names_full = [clean_name(f, with_unit=True) for f in feat_names_sorted]

# ============================================
# Helper: polynomial fit + 95% CI
# ============================================
def plot_spline_ci(ax, x, y, df=5, scatter_color='#4575b4'):
    """B-Spline fit with 95% CI. df controls flexibility (4=smooth, 7=local)."""
    import numpy as np
    # Sort
    sort_idx = np.argsort(x)
    x_s = x.values[sort_idx] if hasattr(x,'values') else np.array(x)[sort_idx]
    y_s = y.values[sort_idx] if hasattr(y,'values') else np.array(y)[sort_idx]
    valid = ~(np.isnan(x_s)|np.isnan(y_s))
    x_s = x_s[valid]; y_s = y_s[valid]
    if len(x_s) < 20: return
    # B-spline matrix
    x_spline = dmatrix("bs(x_train, df=%d, degree=3, include_intercept=False)"%df,
                       {"x_train":x_s}, return_type='dataframe')
    x_spline = sm.add_constant(x_spline)
    model = sm.OLS(y_s, x_spline).fit()
    preds = model.get_prediction(x_spline)
    ci = preds.summary_frame(alpha=0.05)
    # Plot
    ax.scatter(x, y, color=scatter_color, s=18, alpha=0.5, edgecolor='none', zorder=1)
    ax.plot(x_s, ci['mean'], color='#404040', lw=3.0, zorder=3)
    ax.fill_between(x_s, ci['mean_ci_lower'], ci['mean_ci_upper'],
                    color='#7F7F7F', alpha=0.30, edgecolor='none', zorder=2)
    ax.axhline(0, linestyle='--', color='#B0BEC5', lw=1.5, zorder=0)

# ============================================
# FIG 2a: Importance Bar Chart
# ============================================
fig, ax = plt.subplots(figsize=(10, 8))
top_n = 20; y_pos = range(top_n-1,-1,-1)
ax.barh(y_pos, feat_importance[:top_n], color=feat_colors_arr[:top_n],
        edgecolor='white', linewidth=0.5, height=0.7)
ax.set_yticks(list(y_pos)); ax.set_yticklabels(clean_names_short[:top_n])
ax.set_xlabel('Mean |SHAP| Value')
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, edgecolor='white', linewidth=0.5, label=cat)
                   for cat, c in cat_colors.items()]
ax.legend(handles=legend_elements, fontsize=14, frameon=False, loc='lower right')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(axis='x', linestyle=':', alpha=0.4)
plt.tight_layout(); plt.savefig(os.path.join(TEMP_DIR, 'Fig2a_Importance.pdf'), dpi=300, bbox_inches='tight')
plt.close(); print(f'Saved: {TEMP_DIR}/Fig2a_Importance.pdf')

# ============================================
# FIG 2b: Beeswarm
# ============================================
fig = plt.figure(figsize=(12, 9))
top20_idx = idx_sorted[:20]
X_display = X_tr[[features[i] for i in top20_idx]].copy()
X_display.columns = [clean_name(f, with_unit=False) for f in [features[i] for i in top20_idx]]
shap.summary_plot(smat[:, top20_idx], X_display, max_display=20, show=False, plot_size=None)
plt.tight_layout(); plt.savefig(os.path.join(TEMP_DIR, 'Fig2b_Beeswarm.pdf'), dpi=300, bbox_inches='tight')
plt.close(); print(f'Saved: {TEMP_DIR}/Fig2b_Beeswarm.pdf')

# ============================================
# FIG 2c-2h: Top 6 Dependence (polynomial+CI or categorical)
# ============================================
panel_labels = ['c','d','e','f','g','h']
# Derive scatter color from feature category (not hardcoded)

for i in range(6):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    feat = feat_names_sorted[i]; cname = clean_names_full[i]; feat_idx = features.index(feat)
    x_vals = X_tr[feat]; y_vals = pd.Series(smat[:, feat_idx])
    is_categorical = (feat == 'Biome_Type')
    scat_c = cat_colors[feat_to_cat[feat]]  # derive from feature category

    if is_categorical:
        import seaborn as sns
        df_p = pd.DataFrame({'Biome': x_vals, 'SHAP': y_vals}).dropna(subset=['Biome','SHAP'])
        df_p['Biome'] = df_p['Biome'].astype(int).astype(str)
        bio_order = sorted(df_p['Biome'].unique(), key=lambda x: int(x))
        bio_order = [b for b in bio_order if len(df_p[df_p['Biome']==b])>=5]
        # Stripplot with category color
        sns.stripplot(x='Biome', y='SHAP', data=df_p, order=bio_order,
                      color=scat_c, alpha=0.5, size=5, jitter=0.25, zorder=1, ax=ax)
        # I-bar error: #404040 matching spline line, lw=3 identical
        sns.pointplot(x='Biome', y='SHAP', data=df_p, order=bio_order,
                      color='#404040', errorbar=('ci',95), markers='_',
                      markersize=15, linewidth=3.0, capsize=0.25, linestyle='',
                      zorder=3, ax=ax)
        ax.axhline(0, linestyle='--', color='#B0BEC5', lw=1.5, zorder=0)
        ax.set_xlabel('Biome Type'); ax.set_ylabel('SHAP Value')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle=':', alpha=0.4)
        ax.tick_params(axis='x', rotation=45)
    else:
        plot_spline_ci(ax, x_vals, y_vals, df=5, scatter_color=scat_c)
        ax.set_xlabel(cname); ax.set_ylabel('SHAP Value')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    filename = os.path.join(TEMP_DIR, 'Fig2%s_%s.pdf' % (panel_labels[i], feat.replace('_','')))
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved: {filename}')

print('\nAll figures saved.')
for i in range(6):
    print('  %2d. %-20s |SHAP|=%.4f'%(i+1, feat_names_sorted[i], feat_importance[i]))
