import json
import rasterio
from pyproj import Transformer
import geopandas as gpd
from shapely.geometry import Point
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths to your files
geojson_path = '/home/syrichta/routing/davao_specific_barangays_road_network.geojson'
dem_path = '/home/syrichta/routing/dem'
flood_depth_path = '/home/syrichta/routing/flood'

# Create a transformer to convert from WGS 84 (EPSG:4326) to Web Mercator (EPSG:3857)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

# Function to get elevation and flood depth at given coordinates
def get_values(lon, lat):
    elevation = None
    flood_depth = None

    logging.info(f"Getting values for coordinates: ({lon}, {lat})")

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
            logging.info(f"Elevation found: {elevation}")
        else:
            logging.warning(f"Row {row} and column {col} out of bounds for DEM.")

    # Retrieve flood depth
    with rasterio.open(flood_depth_path) as flood_src:
        flood_array = flood_src.read(1)
        flood_transform = flood_src.transform
        # Transform longitude and latitude to row and column
        row, col = ~flood_transform * (lon, lat)

        # Convert row and col to integers
        row = int(row)
        col = int(col)

        # Check if the indices are within bounds for flood array
        if 0 <= row < flood_array.shape[0] and 0 <= col < flood_array.shape[1]:
            flood_depth = flood_array[row, col]
            logging.info(f"Flood depth found: {flood_depth}")
        else:
            logging.warning(f"Row {row} and column {col} out of bounds for flood depth.")

    return elevation, flood_depth

# Load the GeoJSON data
logging.info(f"Loading GeoJSON data from {geojson_path}...")
with open(geojson_path, 'r') as f:
    geojson_data = json.load(f)

# List to hold updated features
updated_features = []

# Process each feature in the GeoJSON
logging.info("Processing features in GeoJSON...")
for feature in geojson_data['features']:
    # Extract coordinates from the feature
    coords = feature['geometry']['coordinates']
    if feature['geometry']['type'] == 'LineString':
        # Check elevation and flood depth at each point in the LineString
        elevations = []
        flood_depths = []
        logging.info(f"Processing LineString with {len(coords)} coordinates.")
        for coord in coords:
            lon, lat = coord  # (longitude, latitude)
            # Transform coordinates to Web Mercator
            lon_3857, lat_3857 = transformer.transform(lon, lat)
            elevation, flood_depth = get_values(lon_3857, lat_3857)
            elevations.append(elevation)
            flood_depths.append(flood_depth)
        
        # Add elevation and flood depth data to the properties
        feature['properties']['elevations'] = elevations
        feature['properties']['flood_depths'] = flood_depths

    # Add updated feature to the list
    updated_features.append(feature)

# Create a new GeoJSON structure with updated features
updated_geojson = {
    'type': 'FeatureCollection',
    'features': updated_features
}

# Save the updated GeoJSON to a new file
updated_geojson_path = 'updated_roads.geojson'
logging.info(f"Saving updated GeoJSON to {updated_geojson_path}...")
with open(updated_geojson_path, 'w') as f:
    json.dump(updated_geojson, f)

logging.info(f"Updated GeoJSON saved to {updated_geojson_path}.")
