import ee
import geemap

"""Extract Landsat 5/7/8 Collection 2 NDVI & NDII (1984-2020) with 3x3 spatial smoothing, hemisphere-specific growing season aggregation. Output: GTM_Golden_1401_Landsat_NDVI_NDII_Aligned.csv"""

# 1. Initialize
ee.Authenticate()
ee.Initialize(project='testing-496011') # Confirm your Project ID

# 2. Load the golden sites
sites = ee.FeatureCollection("projects/testing-496011/assets/GTM_Golden_1401")

# Zonal logic (aligned with original code)
sites_nh = sites.filter(ee.Filter.gt('lat', 23.5))
sites_sh = sites.filter(ee.Filter.lt('lat', -23.5))
sites_tr = sites.filter(ee.Filter.And(ee.Filter.lte('lat', 23.5), ee.Filter.gte('lat', -23.5)))

# 3. Core processing function (aligned with original logic, upgraded to Collection 2)
def process_landsat(image, sensor_type):
    # Band selection and renaming
    if sensor_type == 'L8':
        bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
    else:
        bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']
    new_names = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']

    # Collection 2 scaling factors (restore actual surface reflectance)
    img_scaled = image.select(bands).multiply(0.0000275).add(-0.2).rename(new_names)

    # Collection 2 cloud masking
    qa = image.select('QA_PIXEL')
    cloud_mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    img_masked = img_scaled.updateMask(cloud_mask)

    # Landsat 7 scan line gap-filling via focal_mean
    if sensor_type == 'L7':
        filled = img_masked.focal_mean(1, 'square', 'pixels', 3)
        img_masked = filled.blend(img_masked)

    # Compute indices
    ndvi = img_masked.normalizedDifference(['NIR', 'Red']).rename('NDVI')
    ndii = img_masked.normalizedDifference(['NIR', 'SWIR1']).rename('NDII')

    # Combine and apply 3x3 mean smoothing server-side
    return img_masked.addBands([ndvi, ndii]) \
                     .reduceNeighborhood(reducer=ee.Reducer.mean(), kernel=ee.Kernel.square(1, 'pixels')) \
                     .copyProperties(image, ['system:time_start'])

# 4. Assemble three sensor collections
l5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").map(lambda img: process_landsat(img, 'L5'))
l7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").map(lambda img: process_landsat(img, 'L7'))
l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").map(lambda img: process_landsat(img, 'L8'))
merged_col = l5.merge(l7).merge(l8)

# 5. Annual growing season aggregation
years = ee.List.sequence(1984, 2020)

def extract_yearly(year):
    y = ee.Number(year)
    img_nh = merged_col.filterDate(ee.Date.fromYMD(y, 4, 1), ee.Date.fromYMD(y, 10, 31)).mean()
    img_sh = merged_col.filterDate(ee.Date.fromYMD(y, 10, 1), ee.Date.fromYMD(y.add(1), 4, 30)).mean()
    img_tr = merged_col.filterDate(ee.Date.fromYMD(y, 1, 1), ee.Date.fromYMD(y, 12, 31)).mean()

    def get_stats(img, sites_part):
        return img.select(['NDVI_mean', 'NDII_mean']).reduceRegions(
            collection=sites_part, reducer=ee.Reducer.first(), scale=30
        )

    return get_stats(img_nh, sites_nh).merge(get_stats(img_sh, sites_sh)).merge(get_stats(img_tr, sites_tr)) \
           .map(lambda f: f.set('Year', y))

# 6. Submit single unified export task
final_data = ee.FeatureCollection(years.map(extract_yearly)).flatten()
task = ee.batch.Export.table.toDrive(
    collection=final_data,
    description='GTM_Golden_1401_Landsat_NDVI_NDII_Aligned',
    folder='GEE_Forest_Recovery',
    fileFormat='CSV'
)
task.start()
print("Landsat NDVI/NDII extraction task submitted for all 1401 sites, 1984-2020.")
