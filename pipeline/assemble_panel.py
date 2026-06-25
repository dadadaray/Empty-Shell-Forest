import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

import pandas as pd

# Step 1: Align Landsat NDVI/NDII time series (1984-2002 + 2003-2020)
print("Aligning Landsat time series...")
df_part1 = pd.read_csv('GTM_Golden_1401_Landsat_NDVI&NDII_1984-2002.csv')
df_part2 = pd.read_csv('GTM_Golden_1401_Landsat_NDVI&NDII_2003-2020.csv')
df_landsat = pd.concat([df_part1, df_part2], ignore_index=True)
df_landsat.sort_values(by=['Ref_ID', 'Year'], inplace=True)
df_landsat['Recovery_Year'] = df_landsat['Year'] - df_landsat['event_start']

# Step 2: Merge with 5 other data layers via outer join on Ref_ID + Year
print("Merging SIF, VOD, AGB, canopy height...")
df_master = df_landsat.copy()
df_events = df_master[['Ref_ID', 'event_start']].drop_duplicates()

layers = [
    pd.read_csv('GTM_Golden_1401_GOSIF_2000_2024.csv'),
    pd.read_csv('GTM_Golden_1401_VODCA_CKXU_1987_2020.csv'),
    pd.read_csv('GTM_Golden_1401_VODCA_LBAND_2010_2020.csv').rename(columns={'VOD_L-Band': 'VOD_L'}),
    pd.read_csv('GTM_Golden_1401_AGB_Biomass_sat_io.csv').rename(columns={'AGB_tons_ha': 'AGB_tons_ha'}),
    pd.read_csv('GTM_Golden_1401_CanopyHeight_Official.csv').rename(columns={'Height_m': 'Height_m'}),
]

for layer in layers:
    cols = [c for c in layer.columns if c not in df_master.columns or c in ['Ref_ID', 'Year']]
    df_master = pd.merge(df_master, layer[cols], on=['Ref_ID', 'Year'], how='outer')

df_master = pd.merge(df_master.drop(columns=['event_start']), df_events, on='Ref_ID', how='left')
df_master['Recovery_Year'] = df_master['Year'] - df_master['event_start']
df_master = df_master.sort_values(by=['Ref_ID', 'Year']).reset_index(drop=True)

df_master.to_csv(os.path.join(TEMP_DIR, 'GTM_Master_Panel_Data_Final.csv'), index=False)
print(f"Panel assembled: {len(df_master)} rows, {df_master.Ref_ID.nunique()} sites")
