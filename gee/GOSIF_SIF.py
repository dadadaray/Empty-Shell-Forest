import ee
import geemap

"""Extract GOSIF solar-induced fluorescence (2000-2024) with GEE-based area extraction. Output: GTM_Golden_1401_GOSIF_2000_2024.csv"""

# 1. Initialize
ee.Authenticate()
ee.Initialize(project='testing-496011')

# 2. Load the golden sites
sites = ee.FeatureCollection("projects/testing-496011/assets/GTM_Golden_1401")

sites_nh = sites.filter(ee.Filter.gt('lat', 23.5))
sites_sh = sites.filter(ee.Filter.lt('lat', -23.5))
sites_tr = sites.filter(ee.Filter.And(ee.Filter.lte('lat', 23.5), ee.Filter.gte('lat', -23.5)))

# 3. Processing function (L7 scan-line gap-fill removed; handled via focal_mean downstream)
def process_landsat(image, sensor_type):
    if sensor_type == 'L8':
        bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
    else:
        bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']
    new_names = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']

    img_scaled = image.select(bands).multiply(0.0000275).add(-0.2).rename(new_names)

    qa = image.select('QA_PIXEL')
    cloud_mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    img_masked = img_scaled.updateMask(cloud_mask)

    ndvi = img_masked.normalizedDifference(['NIR', 'Red']).rename('NDVI')
    ndii = img_masked.normalizedDifference(['NIR', 'SWIR1']).rename('NDII')

    return img_masked.addBands([ndvi, ndii]).copyProperties(image, ['system:time_start'])

# 4. Assemble collections
l5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").map(lambda img: process_landsat(img, 'L5'))
l7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").map(lambda img: process_landsat(img, 'L7'))
l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").map(lambda img: process_landsat(img, 'L8'))
merged_col = l5.merge(l7).merge(l8)

# Task chunking control
# Run 1: 1984, 2002 | task_name: '...Part1'
# Run 2: 2003, 2020 | task_name: '...Part2'
START_YEAR = 2003
END_YEAR = 2020
TASK_NAME = 'GTM_Golden_1401_Landsat_NDVI_Part1'

years = ee.List.sequence(START_YEAR, END_YEAR)

def extract_yearly(year):
    y = ee.Number(year)

    img_nh = merged_col.filterDate(ee.Date.fromYMD(y, 4, 1), ee.Date.fromYMD(y, 10, 31)).mean()
    img_sh = merged_col.filterDate(ee.Date.fromYMD(y, 10, 1), ee.Date.fromYMD(y.add(1), 4, 30)).mean()
    img_tr = merged_col.filterDate(ee.Date.fromYMD(y, 1, 1), ee.Date.fromYMD(y, 12, 31)).mean()

    def get_stats(img, sites_part):
        # Apply 3x3 mean smoothing to annual average (also fills L7 scan-line gaps)
        smoothed_img = img.select(['NDVI', 'NDII']).reduceNeighborhood(
            reducer=ee.Reducer.mean(),
            kernel=ee.Kernel.square(1, 'pixels')
        )
        return smoothed_img.reduceRegions(
            collection=sites_part,
            reducer=ee.Reducer.first(),
            scale=30,
            tileScale=16
        )

    return get_stats(img_nh, sites_nh).merge(get_stats(img_sh, sites_sh)).merge(get_stats(img_tr, sites_tr)) \
           .map(lambda f: f.set('Year', y))

final_data = ee.FeatureCollection(years.map(extract_yearly)).flatten()

task = ee.batch.Export.table.toDrive(
    collection=final_data,
    description=TASK_NAME,
    folder='GEE_Forest_Recovery',
    fileFormat='CSV'
)
task.start()
print(f"Chunk task [{TASK_NAME}] for {START_YEAR}-{END_YEAR} submitted.")
