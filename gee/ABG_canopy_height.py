import ee
import geemap

"""Extract ESA CCI above-ground biomass (2020) and ETH Global Canopy Height. Output: GTM_Golden_1401_AGB_Biomass_sat_io.csv, GTM_Golden_1401_CanopyHeight_Official.csv"""

# 1. Initialize
ee.Authenticate()
ee.Initialize(project='testing-496011')

# 2. Load the 1,401 golden sites
sites = ee.FeatureCollection("projects/testing-496011/assets/GTM_Golden_1401")

# ==========================================
# Task A: Tier 5 - Above-ground biomass (latest community V6 path)
# ==========================================
# Load V6.0 biomass collection, select first band and rename to 'AGB_tons_ha'
cci_biomass = ee.ImageCollection("projects/sat-io/open-datasets/ESA/ESA_CCI_AGB").select([0], ['AGB_tons_ha'])

def extract_biomass(img):
    # Extract real year from system timestamp
    year = ee.Date(img.get('system:time_start')).get('year')
    return img.reduceRegions(
        collection=sites,
        reducer=ee.Reducer.first(),
        scale=100,  # ESA native resolution 100m
        tileScale=4
    ).map(lambda f: f.set('Year', year))

biomass_data = cci_biomass.map(extract_biomass).flatten()

task_agb = ee.batch.Export.table.toDrive(
    collection=biomass_data,
    description='GTM_Golden_1401_Tier5_AGB_Biomass_sat_io',
    folder='GEE_Forest_Recovery',
    fileFormat='CSV'
)
task_agb.start()
print("Tier 5 (ESA AGB biomass) extraction task submitted.")

# ==========================================
# Task B: Tier 6 - Canopy height (ETH 2020 official community version)
# ==========================================
# ETH canopy height path
canopy_height_2020 = ee.Image("projects/etc-data/assets/eth_global_canopy_height_2020_10m_v1").select([0], ['Height_m'])

height_data = canopy_height_2020.reduceRegions(
    collection=sites,
    reducer=ee.Reducer.first(),
    scale=10,  # 10m resolution
    tileScale=4
).map(lambda f: f.set('Year', 2020))

task_height = ee.batch.Export.table.toDrive(
    collection=height_data,
    description='GTM_Golden_1401_Tier6_CanopyHeight_Official',
    folder='GEE_Forest_Recovery',
    fileFormat='CSV'
)
task_height.start()
print("Tier 6 (ETH canopy height) extraction task submitted.")
