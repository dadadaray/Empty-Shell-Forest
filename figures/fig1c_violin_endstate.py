import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Fig1c: Violin plot of year-8 standardized anomalies across four biophysical metrics."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],
    'font.size':14,'axes.titlesize':14,'axes.titleweight':'bold',
    'axes.labelsize':14,'axes.labelweight':'bold','xtick.labelsize':14,
    'ytick.labelsize':14,'legend.fontsize':14,'axes.linewidth':1.0,
    'xtick.major.width':1.0,'ytick.major.width':1.0,
    'xtick.direction':'out','ytick.direction':'out',
    'pdf.fonttype':42,'ps.fonttype':42
})

# Data prep (Yan-aligned, W=8)
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
    row = {'Ref_ID':ref_id,'lv_year':lv_year}
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
decoupling = X_raw[:,:,0] - X_raw[:,:,2]
end_dec = [np.mean(decoupling[labels==c,W-1]) for c in range(3)]
es_label = np.argmax(end_dec); is_es = labels == es_label

# Year-8 Z-scores
yr8_data = X_raw[:,W-1,:]
mean_all = np.mean(yr8_data, axis=0); std_all = np.std(yr8_data, axis=0, ddof=1)
std_all[std_all==0] = 1e-5
yr8_z = (yr8_data - mean_all) / std_all

mnames = ['NDVI','NDII','VOD','SIF']
mcolors = ['#000000','#000000','#000000','#000000']

# ============================================
# Violin plot — year-8 standardized anomalies across four metrics
# ============================================
fig, ax = plt.subplots(figsize=(9, 8))

es_data = [yr8_z[is_es,i] for i in range(4)]
ot_data = [yr8_z[~is_es,i] for i in range(4)]

x_es = np.arange(4) - 0.18
x_ot = np.arange(4) + 0.18

# Draw violins manually for full control
for i in range(4):
    # Others (gray, no edge)
    vp_ot = ax.violinplot(ot_data[i], positions=[x_ot[i]], widths=0.30,
                          showmeans=False, showmedians=False, showextrema=False)
    for body in vp_ot['bodies']:
        body.set_facecolor('#E5E7EB'); body.set_alpha(0.9); body.set_linewidth(0)

    # Empty Shell (red, slight transparency, no edge)
    vp_es = ax.violinplot(es_data[i], positions=[x_es[i]], widths=0.30,
                          showmeans=False, showmedians=False, showextrema=False)
    for body in vp_es['bodies']:
        body.set_facecolor('#d73027'); body.set_alpha(0.85); body.set_linewidth(0)

    # Custom inner lines: median (solid, thick) + quartiles (dashed)
    for data, pos, med_c, q_c, lw_m, lw_q in [
        (ot_data[i], x_ot[i], '#000000', '#555555', 2.5, 1.2),
        (es_data[i], x_es[i], '#FFFFFF', '#ffffff', 3.0, 1.5)]:
        q1, med, q3 = np.percentile(data, [25,50,75])
        # Median
        ax.hlines(med, pos-0.12, pos+0.12, colors=med_c, linewidth=lw_m, zorder=10)
        # Quartile dashed lines
        for q in [q1, q3]:
            ax.hlines(q, pos-0.08, pos+0.08, colors=q_c, linewidth=lw_q, ls='--', zorder=9)

# Reference line
ax.axhline(0, color='black', ls='-', lw=1.2, alpha=0.5, zorder=0)

# Colored X-axis tick labels
ax.set_xticks(np.arange(4))
ax.set_xticklabels(mnames)
for i, c in enumerate(mcolors):
    ax.get_xticklabels()[i].set_color(c)
    ax.get_xticklabels()[i].set_fontweight('bold')

# Significance stars
p_vals = [stats.ttest_ind(es_data[i], ot_data[i])[1] for i in range(4)]
for i in range(4):
    if p_vals[i] < 0.001: sig = '***'
    elif p_vals[i] < 0.01: sig = '**'
    elif p_vals[i] < 0.05: sig = '*'
    else: sig = 'n.s.'
    y_max = max(np.max(es_data[i]), np.max(ot_data[i]))
    ax.text(i, y_max+0.35, sig, ha='center', fontsize=14, fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d73027', alpha=0.85, label='Empty Shell (n=%d)'%sum(is_es)),
    Patch(facecolor='#E5E7EB', alpha=0.9, label='Others (n=%d)'%sum(~is_es))
]
ax.legend(handles=legend_elements, fontsize=14, frameon=False, loc='upper left')

ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.set_xlabel(''); ax.set_ylabel('Standardized Anomaly (Z-score) at Year 8')
ax.grid(axis='y', linestyle=':', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(TEMP_DIR, 'Fig1c_Violin.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {TEMP_DIR}/Fig1c_Violin.pdf')
for i in range(4):
    es_m=np.mean(es_data[i]); ot_m=np.mean(ot_data[i])
    print('  %s: ES=%.3f, Other=%.3f, p=%.4f'%(mnames[i],es_m,ot_m,p_vals[i]))
