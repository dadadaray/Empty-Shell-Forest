import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TEMP_DIR
os.chdir(DATA_DIR)

import pandas as pd

df_orig = pd.read_csv('GTM_Full_1613.csv')
df_qa = pd.read_csv('GTM_1613_Ultimate_QA_Check.csv')

df = pd.merge(df_orig[['Ref_ID', 'long', 'lat']],
              df_qa[['Ref_ID', 'event_start', 'gfc_lossyear', 'driver_class', 'esa_lc_2021']],
              on='Ref_ID', how='inner')

# ESA filter: exclude cropland (40) and urban (50)
mask_esa = ~df['esa_lc_2021'].isin([40, 50])
# WRI filter: exclude permanent agriculture (1), commodity-driven deforestation (2), urbanization (6)
# NaN driver_class = unknown driver -> retained (isin returns False for NaN)
mask_wri = ~df['driver_class'].isin([1, 2, 6])
# Non-vegetated land cover filter: exclude water (80), snow/ice (70), no data (100)
mask_land = ~df['esa_lc_2021'].isin([70, 80, 100])

df_clean = df[mask_esa & mask_wri & mask_land].copy()
df_clean.to_csv(os.path.join(TEMP_DIR, 'GTM_Analysis_Sites_1393.csv'), index=False)

n1 = mask_esa.sum()
n2 = (mask_esa & mask_wri).sum()
n3 = (mask_esa & mask_wri & mask_land).sum()
print(f"Raw sites: 1613")
print(f"  - ESA (exclude cropland 40, urban 50): {n1}")
print(f"  - WRI (exclude permanent ag 1, commodity 2, urbanization 6): {n2}")
print(f"  - Land cover (exclude water 80, snow 70, nodata 100): {n3}")
print(f"Final analysis sites: {n3}")
