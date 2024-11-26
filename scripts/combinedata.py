import json
import rasterio
from pyproj import Transformer
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

geojson_path = Path(__file__).parent.parent / 'davao_bounding_box_road_network.geojson'
dem_path = Path(__file__).parent.parent / 'dem'
flood_path = Path(__file__).parent.parent / 'davaoFloodMap11_11_24_SRI30.tif'

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def get_elevation(lon, lat):
    elevation = 0 

    with rasterio.open(dem_path) as dem_src:
        dem_array = dem_src.read(1)
        dem_transform = dem_src.transform
        row, col = ~dem_transform * (lon, lat)

        row = int(row)
        col = int(col)

        if 0 <= row < dem_array.shape[0] and 0 <= col < dem_array.shape[1]:
            elevation = dem_array[row, col]

            if elevation is None or elevation <= -1e+30:
                elevation = 0  

    return elevation

def get_flood_depth(lon, lat):
    flood_depth = None

    with rasterio.open(flood_path) as flood_src:
        flood_array = flood_src.read(1)
        flood_transform = flood_src.transform

        row, col = ~flood_transform * (lon, lat)

        row = int(row)
        col = int(col)

        if 0 <= row < flood_array.shape[0] and 0 <= col < flood_array.shape[1]:
            flood_depth = flood_array[row, col]

    return flood_depth

with open(geojson_path, 'r') as f:
    geojson_data = json.load(f)

updated_features = []

for feature in tqdm(geojson_data['features'], desc="Processing features"):
    coords = feature['geometry']['coordinates']
    if feature['geometry']['type'] == 'LineString':
        elevations = []
        flood_depths = []
        for coord in coords:
            lon, lat = coord 
            lon_3857, lat_3857 = transformer.transform(lon, lat)
            elevation = get_elevation(lon_3857, lat_3857)
            flood_depth = get_flood_depth(lon_3857, lat_3857)
            elevations.append(elevation)
            flood_depths.append(flood_depth)
        
        feature['properties']['elevations'] = elevations
        feature['properties']['flood_depths'] = flood_depths

    updated_features.append(feature)

updated_geojson = {
    'type': 'FeatureCollection',
    'features': updated_features
}

updated_geojson_path = 'roads_with_elevation_and_flood3.geojson'
with open(updated_geojson_path, 'w') as f:
    json.dump(updated_geojson, f)

print(f"Updated GeoJSON with elevation and flood data saved to {updated_geojson_path}")