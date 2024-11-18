import json
import networkx as nx
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
flood_path = Path(__file__).parent.parent / 'davaoFloodMap11_11_24_SRI30.tif'

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

# Function to get flood depth at given coordinates
def get_flood_depth(lon, lat):
    flood_depth = None

    # Retrieve flood depth
    with rasterio.open(flood_path) as flood_src:
        flood_array = flood_src.read(1)
        flood_transform = flood_src.transform
        # Transform longitude and latitude to row and column
        row, col = ~flood_transform * (lon, lat)

        # Convert row and col to integers
        row = int(row)
        col = int(col)

        # Check if the indices are within bounds for flood map
        if 0 <= row < flood_array.shape[0] and 0 <= col < flood_array.shape[1]:
            flood_depth = flood_array[row, col]

    return flood_depth

# Function to build a graph for betweenness centrality calculation
def build_graph_for_betweenness(geojson_data):
    G = nx.Graph()

    for feature in geojson_data['features']:
        if feature['geometry']['type'] == 'LineString':
            coordinates = feature['geometry']['coordinates']
            
            # Add edges between consecutive coordinates
            for i in range(len(coordinates) - 1):
                node1 = tuple(coordinates[i])
                node2 = tuple(coordinates[i + 1])
                
                # Calculate distance (you might want to use geodesic distance)
                from math import sqrt
                distance = sqrt(
                    (coordinates[i][0] - coordinates[i+1][0])**2 + 
                    (coordinates[i][1] - coordinates[i+1][1])**2
                )
                
                G.add_edge(node1, node2, weight=distance)

    return G

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
        # Check elevation and flood depth at each point in the LineString
        elevations = []
        flood_depths = []
        for coord in coords:
            lon, lat = coord  # (longitude, latitude)
            # Transform coordinates to Web Mercator
            lon_3857, lat_3857 = transformer.transform(lon, lat)
            elevation = get_elevation(lon_3857, lat_3857)
            flood_depth = get_flood_depth(lon_3857, lat_3857)
            elevations.append(elevation)
            flood_depths.append(flood_depth)
        
        # Add elevation and flood data to the properties
        feature['properties']['elevations'] = elevations
        feature['properties']['flood_depths'] = flood_depths

    # Add updated feature to the list
    updated_features.append(feature)

# Rebuild the graph for betweenness calculation
G = build_graph_for_betweenness({'features': updated_features})

# Calculate betweenness centrality for each node
print("Calculating betweenness centrality...")
betweenness = nx.betweenness_centrality(G, weight='weight')

# Normalize betweenness centrality
max_b = max(betweenness.values())
betweenness = {node: value / max_b for node, value in betweenness.items()}

# Add betweenness to feature properties
for feature in updated_features:
    coords = feature['geometry']['coordinates']
    feature_betweenness = [betweenness.get(tuple(coord), 0) for coord in coords]
    feature['properties']['betweenness_centrality'] = feature_betweenness

# Create a new GeoJSON structure with updated features
updated_geojson = {
    'type': 'FeatureCollection',
    'features': updated_features
}

# Save the updated GeoJSON to a new file
updated_geojson_path = 'roads_with_elevation_flood_betweenness.geojson'
with open(updated_geojson_path, 'w') as f:
    json.dump(updated_geojson, f)

print(f"Updated GeoJSON with elevation, flood, and betweenness data saved to {updated_geojson_path}")