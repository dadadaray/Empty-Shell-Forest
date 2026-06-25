import ee
import geemap

"""Extract SPEI 12/24/36 month drought indices (1984-2022) with hemisphere-specific seasonal means. Output: GTM_Golden_1401_SPEI_12_24_36_1984_2022.csv"""

# 1. Initialize
ee.Authenticate()
ee.Initialize(project='testing-496011')

# 2. Load 1401 golden sites and group by latitude for growing-season SPEI extraction
sites = ee.FeatureCollection("projects/testing-496011/assets/GTM_Golden_1401")
sites_nh = sites.filter(ee.Filter.gt('lat', 23.5))
sites_sh = sites.filter(ee.Filter.lt('lat', -23.5))
sites_tr = sites.filter(ee.Filter.And(ee.Filter.lte('lat', 23.5), ee.Filter.gte('lat', -23.5)))

# 3. Load SPEI 2.10 dataset with 12/24/36 month timescales
spei_col = ee.ImageCollection("CSIC/SPEI/2_10").select(['SPEI_12_month', 'SPEI_24_month', 'SPEI_36_month'])

# 4. Latitude-aware extraction function
def extract_spei_yearly(year):
    y = ee.Number(year)

    # Compute growing-season mean images per zone
    spei_nh_img = spei_col.filterDate(ee.Date.fromYMD(y, 4, 1), ee.Date.fromYMD(y, 10, 31)).mean()
    spei_sh_img = spei_col.filterDate(ee.Date.fromYMD(y, 10, 1), ee.Date.fromYMD(y.add(1), 4, 30)).mean()
    spei_tr_img = spei_col.filterDate(ee.Date.fromYMD(y, 1, 1), ee.Date.fromYMD(y, 12, 31)).mean()

    # Extract (SPEI resolution ~55 km, use scale=55000)
    res_nh = spei_nh_img.reduceRegions(collection=sites_nh, reducer=ee.Reducer.first(), scale=55000, tileScale=4)
    res_sh = spei_sh_img.reduceRegions(collection=sites_sh, reducer=ee.Reducer.first(), scale=55000, tileScale=4)
    res_tr = spei_tr_img.reduceRegions(collection=sites_tr, reducer=ee.Reducer.first(), scale=55000, tileScale=4)

    return res_nh.merge(res_sh).merge(res_tr).map(lambda f: f.set('Year', y))

# 5. Run extraction for 1984-2022
years = ee.List.sequence(1984, 2022)
data_spei = ee.FeatureCollection(years.map(extract_spei_yearly)).flatten()

task_spei = ee.batch.Export.table.toDrive(
    collection=data_spei,
    description='GTM_Golden_1401_SPEI_12_24_36_1984_2022',
    folder='GEE_Forest_Recovery',
    fileFormat='CSV'
)
task_spei.start()

print("SPEI extraction task submitted for all 1401 sites, 1984-2022.")
