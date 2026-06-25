import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Fig S1b: K vs AUC & HR trade-off."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],
    'font.size':14,'axes.labelsize':14,'axes.labelweight':'bold',
    'xtick.labelsize':12,'ytick.labelsize':12,
    'axes.linewidth':1.2,'pdf.fonttype':42,'ps.fonttype':42
})

K = [2, 3, 4, 5]
AUC = [0.850, 0.909, 0.888, 0.906]
HR = [1.64, 1.60, 1.91, 1.92]
HR_ci_low = [1.41, 1.39, 1.67, 1.69]
HR_ci_high = [1.91, 1.85, 2.18, 2.18]

fig, ax1 = plt.subplots(figsize=(7, 5.5))

# AUC bars
bar_colors = ['#bdbdbd','#333333','#bdbdbd','#bdbdbd']
bars = ax1.bar(K, AUC, width=0.3, color=bar_colors, edgecolor='white', linewidth=0.5, zorder=2)
ax1.set_ylabel('Cross-validated AUC', fontweight='bold', color='#333333')
ax1.set_ylim(0.82, 0.93)
ax1.tick_params(axis='y', labelcolor='#333333')
for bar, auc in zip(bars, AUC):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, '%.3f'%auc,
             ha='center', fontweight='bold', fontsize=12, color='#333333')

# HR line - separate y-axis range well above AUC
ax2 = ax1.twinx()
ax2.errorbar(K, HR, yerr=[[h-l for h,l in zip(HR,HR_ci_low)],[u-h for h,u in zip(HR,HR_ci_high)]],
             fmt='o-', color='#d73027', lw=3.0, markersize=12, capsize=6, capthick=2, zorder=5)
ax2.axhline(1.0, color='gray', ls='--', lw=1.5, alpha=0.4)
ax2.set_ylabel('Hazard Ratio (1393-site Cox PH)', fontweight='bold', color='#333333')
ax2.set_ylim(0.5, 2.8)
ax2.tick_params(axis='y', labelcolor='#333333')
for k_val, hr_val in zip(K, HR):
    ax2.text(k_val+0.15, hr_val+0.15, '%.2f'%hr_val, fontweight='bold', fontsize=13, color='#d73027')

# K=3 highlight
ax1.axvspan(2.5, 3.5, facecolor='#333333', alpha=0.06, zorder=0)
ax1.text(3, 0.828, 'K=3', ha='center', fontsize=10, fontweight='bold', color='#333333', alpha=0.6)

ax1.set_xticks(K)
ax1.set_xlabel('Number of Clusters (K)', fontweight='bold')
ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(TEMP_DIR, 'FigS1b_K_AUC_HR.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {TEMP_DIR}/FigS1b_K_AUC_HR.pdf')
