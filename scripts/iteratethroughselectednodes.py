import json
import networkx as nx
import folium
import csv
from geopy.distance import geodesic
import time
from pyproj import Transformer
import rasterio
from tqdm import tqdm

# Load GeoJSON data
def load_geojson(file_path):
    with open(file_path) as f:
        return json.load(f)

# Load coordinates from a .txt file
def load_coordinates_from_txt(file_path):
    coordinates = []
    with open(file_path, 'r') as file:
        for line in file:
            lon, lat = map(float, line.strip().split(','))
            coordinates.append((lon, lat))
    return coordinates

# Build the graph from GeoJSON
def get_flood_depth(lon, lat, flood_raster_path):
    """Get flood depth at given coordinates from flood depth raster."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    
    try:
        with rasterio.open(flood_raster_path) as src:
            # Transform coordinates to Web Mercator
            lon_3857, lat_3857 = transformer.transform(lon, lat)
            
            # Get row, col indices
            row, col = ~src.transform * (lon_3857, lat_3857)
            row, col = int(row), int(col)
            
            # Check if indices are within bounds
            if 0 <= row < src.height and 0 <= col < src.width:
                depth = src.read(1)[row, col]
                # Handle nodata values
                return 0 if depth == src.nodata else depth
    except Exception as e:
        print(f"Warning: Error reading flood depth at {lon}, {lat}: {e}")
    
    return 0

def build_graph(geojson_data, flood_raster_path):
    """Build graph from GeoJSON with flood depths from raster."""
    G = nx.Graph()

    for feature in tqdm(geojson_data['features'], desc="Building graph from GeoJSON"):  # Add tqdm for progress bar
        coordinates = feature['geometry']['coordinates']
        elevations = feature['properties'].get('elevations', [0] * len(coordinates))
        
        # Get flood depths for each coordinate
        flood_depths = []
        for coord in coordinates:
            depth = get_flood_depth(coord[0], coord[1], flood_raster_path)
            flood_depths.append(depth)

        for i in range(len(coordinates) - 1):
            # Define nodes and add to the graph
            node1 = tuple(coordinates[i])
            node2 = tuple(coordinates[i+1])

            # Calculate distance between the two nodes as the edge weight
            dist = geodesic((coordinates[i][1], coordinates[i][0]), 
                          (coordinates[i+1][1], coordinates[i+1][0])).meters

            # Add edge with all attributes
            G.add_edge(node1, node2, 
                      weight=dist,
                      distance=dist,
                      elevations=(elevations[i], elevations[i+1]),
                      flood_depths=(flood_depths[i], flood_depths[i+1]))

    return G

def calculate_metrics(path, G, speed_mps):
    total_gain = total_loss = max_flood_depth = total_distance = 0
    for i in range(len(path) - 1):
        node1, node2 = path[i], path[i+1]
        edge_data = G.get_edge_data(node1, node2)
        elevations = edge_data['elevations']
        flood_depths = edge_data['flood_depths']
        
        # Check if elevations contain None values and handle them
        if elevations[0] is None or elevations[1] is None:
            elevation_diff = 0  # or some default value if applicable
        else:
            elevation_diff = elevations[1] - elevations[0]
        
        total_gain += max(0, elevation_diff)
        total_loss += max(0, -elevation_diff)
        
        # Filter out None flood depths for max calculation
        valid_flood_depths = list(filter(lambda x: x is not None, flood_depths))
        if valid_flood_depths:
            max_flood_depth = max(max_flood_depth, max(valid_flood_depths))
        
        total_distance += edge_data['distance']
    
    travel_time = total_distance / speed_mps
    return total_gain, total_loss, max_flood_depth, total_distance, travel_time


def heuristic_extended(node1, node2, G, alpha, beta, gamma, delta, epsilon):
    h_n = geodesic((node1[1], node1[0]), (node2[1], node2[0])).meters
    if G.has_edge(node1, node2):
        distance = G[node1][node2]['distance']
        elevation_change = G[node1][node2]['elevations'][1] - G[node1][node2]['elevations'][0]
        slope = elevation_change / distance if distance > 0 else 0
        inundation = G[node1][node2]['flood_depths'][1]
    else:
        distance = slope = inundation = 0
    return alpha * (distance + h_n) + beta * distance + gamma * slope + delta * inundation

def find_and_store_paths_from_txt(G, speed_mps, coordinates_file, output_csv):
    destinations = [
        (125.5657858, 7.1161489),  # Example destination
        (125.5794607, 7.0664451),
        (125.6024582, 7.0766550)
    ]
    start_nodes = load_coordinates_from_txt(coordinates_file)
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['start_node', 'destination', 'total_distance', 'travel_time', 'elevation_gain', 'elevation_loss', 'max_flood_depth', 'path'])
        writer.writeheader()
        for start_node in start_nodes:
            for dest_node in destinations:
                try:
                    path = nx.astar_path(G, source=start_node, target=dest_node, heuristic=lambda n1, n2: heuristic_extended(n1, n2, G, 0.2, 0.25, 0.2, 0.1, 0.25))
                    total_gain, total_loss, max_flood_depth, total_distance, travel_time = calculate_metrics(path, G, speed_mps)
                    writer.writerow({
                        'start_node': start_node,
                        'destination': dest_node,
                        'total_distance': f"{total_distance:.2f}",
                        'travel_time': f"{travel_time:.2f}",
                        'elevation_gain': f"{total_gain:.2f}",
                        'elevation_loss': f"{total_loss:.2f}",
                        'max_flood_depth': f"{max_flood_depth:.4f}",
                        'path': json.dumps(path)
                    })
                except nx.NetworkXNoPath:
                    print(f"No path found between {start_node} and {dest_node}")

# Main execution
if __name__ == "__main__":
    geojson_file = 'roads_with_elevation.geojson'
    flood_raster_path = 'davaoFloodMap11_11_24_SRI30.tif'
    coordinates_file = 'selected_nodes.txt'
    output_csv = 'shortest_paths_from_txt.csv'
    speed_mps = 1.4

    print("Loading GeoJSON and building graph...")
    geojson_data = load_geojson(geojson_file)
    G = build_graph(geojson_data, flood_raster_path)

    print("Finding shortest paths for nodes from txt file...")
    start_time = time.time()
    find_and_store_paths_from_txt(G, speed_mps, coordinates_file, output_csv)
    end_time = time.time()

    print(f"Computation completed in {end_time - start_time:.2f} seconds")
