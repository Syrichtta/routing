import json
import rasterio
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import Point
import logging
from pathlib import Path
from tqdm import tqdm
import warnings

# Set up logging to only show CRITICAL level messages
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress warnings below CRITICAL
warnings.filterwarnings("ignore")

# Paths to your files
geojson_path = Path(__file__).parent.parent / 'davao_bounding_box_road_network.geojson'
dem_path = Path(__file__).parent.parent / 'dem'
flood_depth_path = Path(__file__).parent.parent / 'davaoFloodMap11_11_24_SRI30.tif'

# Create a transformer to convert from WGS 84 (EPSG:4326) to Web Mercator (EPSG:3857)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

# Function to get elevation and flood depth at given coordinates
def get_values(lon, lat):
    elevation = None
    flood_depth = None

    # Retrieve elevation
    with rasterio.open(dem_path) as dem_src:
        dem_array = dem_src.read(1)
        dem_transform = dem_src.transform
        row, col = ~dem_transform * (lon, lat)
        row, col = int(row), int(col)
        if 0 <= row < dem_array.shape[0] and 0 <= col < dem_array.shape[1]:
            elevation = dem_array[row, col]

    # Retrieve flood depth
    with rasterio.open(flood_depth_path) as flood_src:
        flood_array = flood_src.read(1)
        flood_transform = flood_src.transform
        row, col = ~flood_transform * (lon, lat)
        row, col = int(row), int(col)
        if 0 <= row < flood_array.shape[0] and 0 <= col < flood_array.shape[1]:
            flood_depth = flood_array[row, col]

    return elevation, flood_depth

# Load the GeoJSON data
with open(geojson_path, 'r') as f:
    geojson_data = json.load(f)

# List to hold updated features
updated_features = []

# Process each feature in the GeoJSON with tqdm for progress tracking
for feature in tqdm(geojson_data['features'], desc="Processing features"):
    coords = feature['geometry']['coordinates']
    if feature['geometry']['type'] == 'LineString':
        elevations = []
        flood_depths = []
        for coord in tqdm(coords, desc="Processing coordinates", leave=False):
            lon, lat = coord
            lon_3857, lat_3857 = transformer.transform(lon, lat)
            elevation, flood_depth = get_values(lon_3857, lat_3857)
            elevations.append(elevation)
            flood_depths.append(flood_depth)

        feature['properties']['elevations'] = elevations
        feature['properties']['flood_depths'] = flood_depths

    updated_features.append(feature)

# Create a new GeoJSON structure with updated features
updated_geojson = {
    'type': 'FeatureCollection',
    'features': updated_features
}

# Save the updated GeoJSON to a new file
updated_geojson_path = 'roads_with_elevationflood.geojson'
with open(updated_geojson_path, 'w') as f:
    json.dump(updated_geojson, f)

print(f"Updated GeoJSON saved to {updated_geojson_path}.")
