import ee
import geemap

"""Extract QA data (Hansen GFC loss year, WRI driver class, ESA WorldCover) for 1,613 sites. Output: GTM_1613_Ultimate_QA_Check.csv"""

# 1. Initialize
ee.Authenticate()
ee.Initialize(project='testing-496011') # Confirm your Project ID

# 2. Load the 1613-site feature collection
sites_1613 = ee.FeatureCollection("projects/testing-496011/assets/GTM_Full_1613")

# 3. Load the three QA datasets
# Dataset A: Hansen GFC
gfc_loss = ee.Image("UMD/hansen/global_forest_change_2025_v1_13").select('lossyear')

# Dataset B: WRI forest loss driver
drivers = ee.Image("projects/landandcarbon/assets/wri_gdm_drivers_forest_loss_1km/v1_2001_2022").select('classification').rename('driver_class')

# Dataset C: ESA WorldCover 2021
esa_worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select('Map').rename('esa_lc_2021')

# Combine three images into one for single-pass extraction
combined_qa_image = gfc_loss.addBands(drivers).addBands(esa_worldcover)

# 4. Spatial extraction: sample values at each point
def extract_qa_info(feature):
    # Use 30 m spatial resolution
    qa_values = combined_qa_image.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=feature.geometry(),
        scale=30
    )
    # Band names here match the renamed bands above
    return feature.set(
        'gfc_lossyear', qa_values.get('lossyear'),
        'driver_class', qa_values.get('driver_class'),
        'esa_lc_2021', qa_values.get('esa_lc_2021')
    )

sites_with_qa = sites_1613.map(extract_qa_info)

# 5. Export QA results to Drive
task_qa = ee.batch.Export.table.toDrive(
    collection=sites_with_qa,
    description='GTM_1613_Ultimate_QA_Check',
    folder='GEE_Forest_Recovery',
    fileFormat='CSV',
    # Include esa_lc_2021 in export fields
    selectors=['Ref_ID', 'long', 'lat', 'event_start', 'gfc_lossyear', 'driver_class', 'esa_lc_2021']
)
task_qa.start()

print("QA extraction task submitted for 1613 sites. Check GEE Tasks page for progress.")
