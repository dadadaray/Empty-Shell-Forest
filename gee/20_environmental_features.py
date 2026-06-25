import ee
import geemap
import math

"""Extract 20 environmental features for all 1,401 golden sites (-> 1,393 after water removal).
Site-level CSV for XGBoost training & prediction (pipeline/06-07).
Output: GTM_1393_Landsat_20D_Advanced.csv (1,393 sites x 20 features)"""

ee.Authenticate()
ee.Initialize(project='testing-496011')

sites = ee.FeatureCollection("projects/testing-496011/assets/GTM_Golden_1401")

# ==========================================
# Landsat processing pipeline with scaling, growing season, smoothing, and fallback
# ==========================================
def get_advanced_landsat(year, geom):
    y = ee.Number(year)
    lat = ee.Number(geom.coordinates().get(1))

    # Hemisphere-aware growing season logic
    start_nh = ee.Date.fromYMD(y, 4, 1)
    end_nh = ee.Date.fromYMD(y, 10, 31)
    start_sh = ee.Date.fromYMD(y, 10, 1)
    end_sh = ee.Date.fromYMD(y.add(1), 4, 30)
    start_tr = ee.Date.fromYMD(y, 1, 1)
    end_tr = ee.Date.fromYMD(y, 12, 31)

    # Dynamically assign date range based on latitude
    start = ee.Date(ee.Algorithms.If(lat.gt(23.5), start_nh, ee.Algorithms.If(lat.lt(-23.5), start_sh, start_tr)))
    end = ee.Date(ee.Algorithms.If(lat.gt(23.5), end_nh, ee.Algorithms.If(lat.lt(-23.5), end_sh, end_tr)))

    # Scaling and masking logic
    def prep_scaled(img, sensor):
        # Fix AttributeError: 'String' object has no attribute 'eq'
        is_L89 = ee.String(sensor).compareTo('L89').eq(0)
        bands = ee.List(ee.Algorithms.If(is_L89,
                        ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
                        ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']))
        new_names = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']

        # C2 standard scaling factors
        img_scaled = img.select(bands).multiply(0.0000275).add(-0.2).rename(new_names)
        qa = img.select('QA_PIXEL')
        mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
        return img_scaled.updateMask(mask)

    L9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(geom).filterDate(start, end).map(lambda img: prep_scaled(img, 'L89'))
    L8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(geom).filterDate(start, end).map(lambda img: prep_scaled(img, 'L89'))
    L7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").filterBounds(geom).filterDate(start, end).map(lambda img: prep_scaled(img, 'L57'))
    L5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").filterBounds(geom).filterDate(start, end).map(lambda img: prep_scaled(img, 'L57'))

    # Fallback dummy image for years with no satellite data (e.g., early 1980s)
    dummy = ee.Image.constant([0,0,0,0,0,0]).rename(['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']).updateMask(0)

    # Merge and compute mean
    combined_mean = L9.merge(L8).merge(L7).merge(L5).merge(ee.ImageCollection([dummy])).mean()

    # 3x3 neighborhood smoothing for gap-filling
    smoothed = combined_mean.reduceNeighborhood(
        reducer=ee.Reducer.mean(),
        kernel=ee.Kernel.square(1, 'pixels')
    )
    # reduceNeighborhood renames bands (e.g., NIR -> NIR_mean), so restore original names
    return smoothed.rename(['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2'])


# ==========================================
# 20-feature extraction
# ==========================================
def extract_20D_ultimate(feature):
    startYear = ee.Number(feature.get('event_start'))
    geom = feature.geometry()

    # 1. Climate: 6 features + 2 extremes
    recStart = ee.Date.fromYMD(startYear.add(1), 1, 1)
    recEnd = ee.Date.fromYMD(startYear.add(8), 12, 31)
    climateWindow = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").filterDate(recStart, recEnd).select(['vpd', 'pr', 'tmmx', 'soil', 'def', 'pdsi']).mean()
    extremeClimate = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").filterDate(recStart, recEnd).select(['vpd', 'def']).max().rename(['Max_VPD', 'Max_DEF'])

    # 2. Topography & soil: 8 features
    dem = ee.Image("CGIAR/SRTM90_V4")
    elevation = dem.rename('elevation')
    slope = ee.Terrain.slope(dem).rename('slope')
    aspect = ee.Terrain.aspect(dem).rename('aspect')
    flow_acc = ee.Image("WWF/HydroSHEDS/15ACC")
    twi = flow_acc.multiply(90*90).divide(slope.multiply(math.pi / 180.0).add(0.001).tan()).log().rename('TWI')

    sand = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select('b0').rename('Sand_Content')
    clay = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select('b0').rename('Clay_Content')
    bulk_density = ee.Image("OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02").select('b0').rename('Bulk_Density')
    carbon = ee.Image("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02").select('b0').rename('Carbon_Content')

    biome = ee.Image.constant(0).paint(ee.FeatureCollection("RESOLVE/ECOREGIONS/2017"), 'BIOME_NUM').rename('Biome_Type')
    humanMod = ee.ImageCollection("CSP/HM/GlobalHumanModification").mosaic().unmask(0).rename('Human_Footprint')

    # 3. Compute Pre_NDVI and dNBR from Landsat engine
    pre_img = get_advanced_landsat(startYear.subtract(1), geom)
    post_img = get_advanced_landsat(startYear.add(1), geom)

    pre_ndvi = pre_img.normalizedDifference(['NIR', 'Red']).rename('Pre_NDVI')
    pre_nbr = pre_img.normalizedDifference(['NIR', 'SWIR2'])
    post_nbr = post_img.normalizedDifference(['NIR', 'SWIR2'])
    dNBR = pre_nbr.subtract(post_nbr).rename('dNBR')

    # 4. Combine 20 bands
    combinedImage = ee.Image([
        climateWindow, extremeClimate,
        elevation, slope, aspect, twi,
        sand, clay, bulk_density, carbon, biome, humanMod,
        pre_ndvi, dNBR
    ])

    extractedDict = combinedImage.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=4000,
        maxPixels=1e9
    )
    return feature.set(extractedDict)

# ==========================================
# Submit export task
# ==========================================
final_sites = sites.map(extract_20D_ultimate)

task = ee.batch.Export.table.toDrive(
    collection=final_sites,
    description='GTM_1393_Landsat_20D_Advanced',
    folder='GEE_Forest_Recovery',
    fileFormat='CSV'
)

task.start()
print("20-feature extraction task submitted.")
