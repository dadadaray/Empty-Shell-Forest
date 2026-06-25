import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Fig1b companion: Raw 4-metric data for site 118 (2-panel: canopy + structure)."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],
    'font.size':14,'axes.titlesize':14,'axes.titleweight':'bold',
    'axes.labelsize':14,'axes.labelweight':'bold','xtick.labelsize':14,
    'ytick.labelsize':13,'legend.fontsize':13,'axes.linewidth':1.0,
    'xtick.direction':'out','ytick.direction':'out',
    'pdf.fonttype':42,'ps.fonttype':42
})

panel = pd.read_csv(os.path.join(TEMP_DIR, 'GTM_Master_Panel_Data_Final.csv'))
spei = pd.read_csv('GTM_Golden_1401_SPEI_12_24_36_1984_2022.csv')
panel = pd.merge(panel, spei[['Ref_ID','Year','SPEI_12_month']], on=['Ref_ID','Year'], how='left')

site = panel[panel['Ref_ID']==118].sort_values('Year')
es_y = 2010; lv_y = 2011
all_yr = list(range(es_y-3, lv_y+8))
idx_dist = all_yr.index(es_y)
idx_recov = all_yr.index(lv_y)

# Extract all series
def get_series(col):
    return [site[site['Year']==yr][col].values[0] if len(site[site['Year']==yr])>0
            and pd.notna(site[site['Year']==yr][col].values[0]) else np.nan for yr in all_yr]

ndvi_raw = get_series('NDVI_mean')
sif_raw  = get_series('SIF_mean')
ndii_raw = get_series('NDII_mean')
vod_raw  = get_series('VOD_CKXU')
spei_vals = get_series('SPEI_12_month')

fig = plt.figure(figsize=(11, 9))
gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.12)

# ============================================================
# UPPER PANEL: NDVI + SIF (Canopy Greenness & Photosynthesis)
# ============================================================
ax1 = fig.add_subplot(gs[0])
ax1.set_ylabel('NDVI', fontweight='bold', color='#1a9850')
ax1.tick_params(axis='y', labelcolor='#1a9850')
ax1.plot(range(len(all_yr)), ndvi_raw, 'o-', color='#1a9850', lw=2.5, ms=7, label='NDVI (Greenness)')

ax1r = ax1.twinx()
ax1r.set_ylabel('SIF (W m$^{-2}$ $\mu$m$^{-1}$ sr$^{-1}$)', fontweight='bold', color='#fdae61')
ax1r.tick_params(axis='y', labelcolor='#fdae61')
ax1r.plot(range(len(all_yr)), sif_raw, 'D--', color='#fdae61', lw=2.5, ms=7, label='SIF (Photosynthesis)')

# Reference lines
ax1.axvline(idx_dist, color='#555555', ls='--', lw=1.5, alpha=0.7, zorder=5)
ax1.axvline(idx_recov, color='#d73027', ls=':', lw=1.8, alpha=0.9, zorder=5)
ax1.text(idx_dist-0.3, ax1.get_ylim()[1]*0.88, 'Disturbance\nEvent', fontsize=13,
         ha='right', color='#555555')
ax1.text(idx_recov+0.3, ax1.get_ylim()[1]*0.88, 'Recovery Onset\n(NDVI Minimum)', fontsize=13,
         ha='left', color='#d73027')

# Legend
lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax1r.get_legend_handles_labels()
ax1.legend(lines1+lines2, labs1+labs2, fontsize=13, frameon=False, loc='upper left')

ax1.spines['top'].set_visible(False); ax1r.spines['top'].set_visible(False)
ax1.set_xticklabels([])  # hide X labels on upper panel

# ============================================================
# LOWER PANEL: NDII + VOD (Canopy Water & Woody Biomass)
# ============================================================
ax2 = fig.add_subplot(gs[1])

# SPEI bars (background)
ax2.bar(range(len(all_yr)), spei_vals, color='#B0BEC5', alpha=0.35, width=0.6, zorder=0,
        label='SPEI-12 (right axis)')
ax2.axhline(-1, color='#B0BEC5', ls=':', lw=1.2, alpha=0.7)

ax2.set_ylabel('NDII', fontweight='bold', color='#4575b4')
ax2.tick_params(axis='y', labelcolor='#4575b4')
ax2.plot(range(len(all_yr)), ndii_raw, 'p-', color='#4575b4', lw=2.5, ms=7, label='NDII (Canopy Water)')

ax2r = ax2.twinx()
ax2r.set_ylabel('VOD', fontweight='bold', color='#d73027')
ax2r.tick_params(axis='y', labelcolor='#d73027')
ax2r.plot(range(len(all_yr)), vod_raw, 's--', color='#d73027', lw=2.5, ms=7, label='VOD (Biomass)')

# Reference lines
ax2.axvline(idx_dist, color='#555555', ls='--', lw=1.5, alpha=0.7, zorder=5)
ax2.axvline(idx_recov, color='#d73027', ls=':', lw=1.8, alpha=0.9, zorder=5)

# X-axis
rel_years = [yr-lv_y for yr in all_yr]
ax2.set_xticks(range(len(all_yr)))
ax2.set_xticklabels([str(r) for r in rel_years])
ax2.set_xlabel('Years Relative to Recovery Onset (NDVI Minimum = 0)', fontweight='bold')

# Legend (SPEI + NDII + VOD)
lines3, labs3 = ax2.get_legend_handles_labels()
lines4, labs4 = ax2r.get_legend_handles_labels()
ax2.legend(lines3+lines4, labs3+labs4, fontsize=13, frameon=False, loc='upper left')

ax2.spines['top'].set_visible(False); ax2r.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(TEMP_DIR, 'Fig1b_Raw_Data.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {TEMP_DIR}/Fig1b_Raw_Data.pdf')
print('Site 118 raw values:')
print(f'{"Year":>5s} {"Rel":>4s} {"NDVI":>8s} {"SIF":>8s} {"NDII":>8s} {"VOD":>8s} {"SPEI":>8s}')
for yr in all_yr:
    rel = yr - lv_y
    row = site[site['Year']==yr]
    vals = []
    for c in ['NDVI_mean','SIF_mean','NDII_mean','VOD_CKXU','SPEI_12_month']:
        a = row[c].values
        vals.append(a[0] if len(a)>0 and pd.notna(a[0]) else np.nan)
    print(f'{yr:5d} {rel:+4d} {vals[0]:8.4f} {vals[1]:8.4f} {vals[2]:8.4f} {vals[3]:8.4f} {vals[4]:8.2f}')
