import geopandas as gpd
from shapely.geometry import Point
from itertools import combinations

# Load your GeoJSON data
gdf = gpd.read_file("/home/syrichta/routing/updated_roads.geojson")

# Ensure GeoDataFrame is in a projected coordinate system for accurate distance measurement (e.g., meters)
gdf = gdf.to_crs(epsg=3857)  # Using WGS 84 / Pseudo-Mercator (meters)

# Extract nodes as coordinate pairs from LineString geometries
nodes = set()
for line in gdf.geometry:
    if line.geom_type == 'LineString':
        nodes.add(line.coords[0])  # Start point as (lon, lat) tuple
        nodes.add(line.coords[-1]) # End point as (lon, lat) tuple
    elif line.geom_type == 'MultiLineString':
        for subline in line:
            nodes.add(subline.coords[0])
            nodes.add(subline.coords[-1])

# Convert each node to a Point and create a GeoDataFrame for projection
points = [Point(coord) for coord in nodes]
nodes_gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")  # Original WGS84
nodes_gdf = nodes_gdf.to_crs(epsg=3857)  # Project to meters

# Check distances between pairs of nodes
found_close_nodes = False
for (coord1, point1), (coord2, point2) in combinations(zip(nodes, nodes_gdf.geometry), 2):
    distance = point1.distance(point2)
    if distance < 500:
        print(f"Nodes within 500 meters found: {coord1} and {coord2} with distance {distance:.2f} meters.")
        found_close_nodes = True
        break

if not found_close_nodes:
    print("No nodes found within 500 meters of each other.")
