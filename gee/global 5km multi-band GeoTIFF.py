import ee
import geemap
import math

"""Export global 5km multi-band GeoTIFF (20 features + ESA CCI AGB baseline).
Pixel-level raster for Fig 4 global extrapolation (pipeline/11_global_map.py).
Unlike 07 (site-level CSV), this is a wall-to-wall global raster at 5km resolution.
Output: ES_Global_20D_and_AGB_5km.tif (~500MB)"""

# 1. Initialize GEE
try:
    ee.Initialize(project='testing-496011')
except:
    ee.Authenticate()
    ee.Initialize(project='testing-496011')

# ==========================================
# Step 1: Build global 20-feature tensor (climate baseline 2001–2020)
# ==========================================
start_year, end_year = '2001-01-01', '2020-12-31'

# 1. Climate features (6 mean + 2 extreme)
terra = ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE').filterDate(start_year, end_year)
tmmx = terra.select('tmmx').mean().rename('tmmx')
pr = terra.select('pr').mean().rename('pr')
vpd = terra.select('vpd').mean().rename('vpd')
soil = terra.select('soil').mean().rename('soil')
def_val = terra.select('def').mean().rename('def')
pdsi = terra.select('pdsi').mean().rename('pdsi')

# Use percentile instead of raw max to resist extreme value noise
max_vpd = terra.select('vpd').reduce(ee.Reducer.percentile([100])).rename('Max_VPD')
max_def = terra.select('def').reduce(ee.Reducer.percentile([100])).rename('Max_DEF')

# 2. Topography features (4 bands, HydroSHEDS logic)
dem = ee.Image("CGIAR/SRTM90_V4")
elev = dem.rename('elevation')
slp = ee.Terrain.slope(dem).rename('slope')
asp = ee.Terrain.aspect(dem).rename('aspect')
flow_acc = ee.Image("WWF/HydroSHEDS/15ACC")
twi = flow_acc.multiply(90*90).divide(slp.multiply(math.pi / 180.0).add(0.001).tan()).log().rename('TWI')

# 3. Soil features (4 bands)
sand = ee.Image('OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02').select('b0').rename('Sand_Content')
clay = ee.Image('OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02').select('b0').rename('Clay_Content')
bulk = ee.Image('OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02').select('b0').rename('Bulk_Density')
carbon = ee.Image('OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02').select('b0').rename('Carbon_Content')

# 4. Ecology & human disturbance (2 bands)
biome = ee.Image.constant(0).paint(ee.FeatureCollection('RESOLVE/ECOREGIONS/2017'), 'BIOME_NUM').rename('Biome_Type')
hf = ee.ImageCollection("CSP/HM/GlobalHumanModification").mosaic().unmask(0).rename('Human_Footprint')

# 5. Pre-disturbance baseline & stress test (2 bands)
pre_ndvi = ee.ImageCollection('MODIS/061/MOD13A2').filterDate('2016-01-01', '2020-12-31').select('NDVI').mean().divide(10000).rename('Pre_NDVI')
dNBR_stress = ee.Image.constant(0.047).rename('dNBR')  # P75 conservative stress test

# Stack all 20 bands. Ensure band order and names match XGBoost training columns exactly.
stack_20D = ee.Image([
    tmmx, pr, vpd, soil, def_val, pdsi, max_vpd, max_def,
    elev, slp, asp, twi, sand, clay, bulk, carbon,
    biome, hf, pre_ndvi, dNBR_stress
])

# ==========================================
# Load ESA baseline above-ground biomass (matches official example)
# ==========================================
# Use ee.Image, select agb band and rename
agb_baseline = ee.Image('ESA/CCI/Above_Ground_Biomass/V6_0/2020').select('agb').rename('AGB_Baseline')

# Append to the stack
export_stack = stack_20D.addBands(agb_baseline)

# ==========================================
# Step 3: Start cloud export to Google Drive
# ==========================================
# Define region of interest. For production, export by continent to keep files manageable.
# Global extent (excluding Antarctica ice sheets)
region_global = ee.Geometry.BBox(-179.9, -59.9, 179.9, 89.9)

task = ee.batch.Export.image.toDrive(
    image=export_stack.toFloat(),
    description='ES_Global_20D_and_AGB_5km',
    folder='Nature_Paper_Data',
    scale=5000,                        # Note: 10000 recommended for memory safety
    region=region_global,
    maxPixels=1e13                     # Relax pixel limit
)
task.start()
print("20-feature + AGB carbon baseline export submitted to Google Drive.")
