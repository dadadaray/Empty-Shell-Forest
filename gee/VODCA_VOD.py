import ee
import geemap

"""Extract VODCA C-band (1987-2020) and L-band (2010-2020) vegetation optical depth. Output: GTM_Golden_1401_VODCA_CKXU_1987_2020.csv, GTM_Golden_1401_VODCA_LBAND_2010_2020.csv"""

# 1. Initialize
ee.Authenticate()
ee.Initialize(project='testing-496011')

# 2. Load the 1,401 golden sites
sites = ee.FeatureCollection("projects/testing-496011/assets/GTM_Golden_1401")

# 3. Load both VODCA versions
vodca_ckxu = ee.ImageCollection("projects/sat-io/open-datasets/VODCA/CKXU_BAND_V2").select('VOD')
vodca_l = ee.ImageCollection("projects/sat-io/open-datasets/VODCA/L_BAND_V2").select('VOD')

# 4. Extraction function factory
def create_extractor(collection, scale_meters):
    def extract_yearly(year):
        y = ee.Number(year)
        img = collection.filterDate(ee.Date.fromYMD(y, 1, 1), ee.Date.fromYMD(y, 12, 31)).mean()
        # Apply safety parameters and appropriate scale
        return img.reduceRegions(
            collection=sites,
            reducer=ee.Reducer.first(),
            scale=scale_meters,
            tileScale=4
        ).map(lambda f: f.set('Year', y))
    return extract_yearly

# 5. Task A: CKXU-Band (1987–2020 time series)
years_ckxu = ee.List.sequence(1987, 2020)
data_ckxu = ee.FeatureCollection(years_ckxu.map(create_extractor(vodca_ckxu, 25000))).flatten()

task_ckxu = ee.batch.Export.table.toDrive(
    collection=data_ckxu,
    description='GTM_Golden_1401_VODCA_CKXU_1987_2020',
    folder='GEE_Forest_Recovery',
    fileFormat='CSV'
)
task_ckxu.start()

# 6. Task B: L-Band (2010–2020 high-resolution observation)
# Note: reliable L-band data begins around 2010
years_l = ee.List.sequence(2010, 2020)
data_l = ee.FeatureCollection(years_l.map(create_extractor(vodca_l, 25000))).flatten()

task_l = ee.batch.Export.table.toDrive(
    collection=data_l,
    description='GTM_Golden_1401_VODCA_LBAND_2010_2020',
    folder='GEE_Forest_Recovery',
    fileFormat='CSV'
)
task_l.start()

print("CKXU-band and L-band extraction tasks submitted.")
