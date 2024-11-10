import json
import rasterio
from pyproj import Transformer
import logging
from pathlib import Path
from tqdm import tqdm

# Set up logging to show only critical issues
logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths to your files
geojson_path = Path(__file__).parent.parent / 'davao_bounding_box_road_network.geojson'
dem_path = Path(__file__).parent.parent / 'dem'

# Create a transformer to convert from WGS 84 (EPSG:4326) to Web Mercator (EPSG:3857)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

# Function to get elevation at given coordinates
def get_elevation(lon, lat):
    elevation = None

    # Retrieve elevation
    with rasterio.open(dem_path) as dem_src:
        dem_array = dem_src.read(1)
        dem_transform = dem_src.transform
        # Transform longitude and latitude to row and column
        row, col = ~dem_transform * (lon, lat)

        # Convert row and col to integers
        row = int(row)
        col = int(col)

        # Check if the indices are within bounds for DEM
        if 0 <= row < dem_array.shape[0] and 0 <= col < dem_array.shape[1]:
            elevation = dem_array[row, col]

    return elevation

# Load the GeoJSON data
with open(geojson_path, 'r') as f:
    geojson_data = json.load(f)

# List to hold updated features
updated_features = []

# Process each feature in the GeoJSON with tqdm for progress tracking
for feature in tqdm(geojson_data['features'], desc="Processing features"):
    # Extract coordinates from the feature
    coords = feature['geometry']['coordinates']
    if feature['geometry']['type'] == 'LineString':
        # Check elevation at each point in the LineString
        elevations = []
        for coord in coords:
            lon, lat = coord  # (longitude, latitude)
            # Transform coordinates to Web Mercator
            lon_3857, lat_3857 = transformer.transform(lon, lat)
            elevation = get_elevation(lon_3857, lat_3857)
            elevations.append(elevation)
        
        # Add elevation data to the properties
        feature['properties']['elevations'] = elevations

    # Add updated feature to the list
    updated_features.append(feature)

# Create a new GeoJSON structure with updated features
updated_geojson = {
    'type': 'FeatureCollection',
    'features': updated_features
}

# Save the updated GeoJSON to a new file
updated_geojson_path = 'roads_with_elevation.geojson'
with open(updated_geojson_path, 'w') as f:
    json.dump(updated_geojson, f)
