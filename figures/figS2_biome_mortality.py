import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

"""Fig S2c: Biome-specific repeat mortality: Empty Shell vs Others."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica'],
    'font.size':14,'axes.labelsize':14,'axes.labelweight':'bold',
    'xtick.labelsize':12,'ytick.labelsize':12,
    'axes.linewidth':1.2,'pdf.fonttype':42,'ps.fonttype':42
})

pred = pd.read_csv('GTM_1401_Sites_Probabilities_3class_TEMP.csv')
panel = pd.read_csv(os.path.join(TEMP_DIR, 'GTM_Master_Panel_Data_Final.csv'))
sites = panel.groupby('Ref_ID').agg(es=('event_start','first'),gl=('gfc_lossyear','max')).reset_index()
sites['al'] = np.where(sites['gl'].notna(),sites['gl']+2000,np.nan)
sites['died'] = np.where((sites['al']>sites['es'])&sites['al'].notna(),1,0).astype(int)
pred = pd.merge(pred, sites[['Ref_ID','died']], on='Ref_ID', how='inner')

# Biome grouping
bn = {1:'Trop.Moist',2:'Trop.Dry',4:'Temp.Broad',5:'Temp.Conif',6:'Boreal',
      7:'Trop.Grass',8:'Temp.Grass',12:'Mediterr',13:'Desert'}
pred['Biome'] = pred['Biome_Type'].map(bn).fillna('Other')
pred['ES'] = (pred['Predicted_Dominant_Archetype']=='Archetype_3').astype(int)

# Per biome: ES mortality vs Others mortality
biomes = ['Mediterr','Temp.Conif','Temp.Broad','Boreal','Trop.Moist','Trop.Grass','Temp.Grass']
data = []
for b in biomes:
    sub = pred[pred['Biome']==b]
    if len(sub)<10: continue
    es_m = sub[sub['ES']==1]['died'].mean()*100
    ot_m = sub[sub['ES']==0]['died'].mean()*100
    es_n = sum(sub['ES']==1); ot_n = sum(sub['ES']==0)
    data.append({'Biome':b,'ES_mort':es_m,'Other_mort':ot_m,'ES_n':es_n,'Other_n':ot_n,
                 'RR':es_m/ot_m if ot_m>0 else 0})

df = pd.DataFrame(data).sort_values('RR', ascending=True)

# Plot
fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(len(df)); w = 0.35
bars_es = ax.bar(x-w/2, df['ES_mort'], w, color='#d73027', edgecolor='white', linewidth=0.5, label='Empty Shell')
bars_ot = ax.bar(x+w/2, df['Other_mort'], w, color='#B0BEC5', edgecolor='white', linewidth=0.5, label='Others')

# RR annotations
for i, (_, row) in enumerate(df.iterrows()):
    rr = row['RR']
    ax.text(i, max(row['ES_mort'],row['Other_mort'])+1, '%.1fx'%rr, ha='center',
            fontweight='bold', fontsize=11, color='#d73027' if rr>1 else '#333333')

# Mortality rates on bars
for bar, val in zip(bars_es, df['ES_mort']):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, '%.1f%%'%val, ha='center', fontsize=9, color='#d73027', fontweight='bold')
for bar, val in zip(bars_ot, df['Other_mort']):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, '%.1f%%'%val, ha='center', fontsize=9, color='#757575')

ax.set_xticks(x); ax.set_xticklabels(df['Biome'], rotation=30, ha='right')
ax.set_ylabel('Repeat Mortality Rate (%)', fontweight='bold')
ax.legend(fontsize=12, frameon=False)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle=':', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(TEMP_DIR, 'FigS2c_Biome_Mortality.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print(f'Saved: {TEMP_DIR}/FigS2c_Biome_Mortality.pdf')
for _, row in df.iterrows():
    print('%s: ES=%.1f%%, Other=%.1f%%, RR=%.1fx'%(row['Biome'],row['ES_mort'],row['Other_mort'],row['RR']))
