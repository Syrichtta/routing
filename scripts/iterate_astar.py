import json
import networkx as nx
import folium
import random
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


# Calculate elevation gain/loss, maximum flood depth, and total distance
def calculate_metrics(path, G, speed_mps):
    total_gain = 0
    total_loss = 0
    max_flood_depth = 0
    total_distance = 0

    for i in range(len(path) - 1):
        node1 = path[i]
        node2 = path[i+1]
        edge_data = G.get_edge_data(node1, node2)
        elevations = edge_data['elevations']
        flood_depths = edge_data['flood_depths']

        # Check for valid elevation values
        elevation1 = elevations[0] if elevations[0] is not None else 0
        elevation2 = elevations[1] if elevations[1] is not None else 0

        # Calculate elevation gain/loss
        elevation_diff = elevation2 - elevation1
        if elevation_diff > 0:
            total_gain += elevation_diff
        else:
            total_loss += abs(elevation_diff)

        # Handle flood depth values
        for depth in flood_depths:
            if depth is not None:  # Only consider valid flood depth values
                max_flood_depth = max(max_flood_depth, depth)

        # Calculate total distance traveled
        total_distance += edge_data['distance']

    # Calculate travel time based on total distance and speed (in meters per second)
    travel_time = total_distance / speed_mps

    return total_gain, total_loss, max_flood_depth, total_distance, travel_time

def calculate_slope(node1, node2, G):
    if G.has_edge(node1, node2):
        edge_data = G[node1][node2]
        elevation1 = edge_data['elevations'][0]  # Elevation of node1
        elevation2 = edge_data['elevations'][1]  # Elevation of node2
        horizontal_distance = edge_data['distance']  # Horizontal distance between nodes

        # Calculate the change in elevation
        elevation_change = elevation2 - elevation1

        # Calculate slope (rise/run)
        if horizontal_distance > 0:  # Prevent division by zero
            slope = elevation_change / horizontal_distance
        else:
            slope = 0

        return slope
    else:
        return 0  # Return 0 if no edge exists



# Visualize the shortest path on a map with start/end pins and metrics
def visualize_path(geojson_data, path, output_html, total_gain, total_loss, max_flood_depth, total_distance, travel_time, start_node, end_node):
    # Create a map centered on a central point
    central_point = [7.0512, 125.5987]  # Update to your area
    m = folium.Map(location=central_point, zoom_start=15)

    # Add the GeoJSON layer to the map
    folium.GeoJson(geojson_data, name='Roads').add_to(m)

    # If a path is found, plot the path
    if path:
        path_coordinates = [(p[1], p[0]) for p in path]  # Folium expects lat, lon order
        folium.PolyLine(path_coordinates, color="blue", weight=5, opacity=0.7).add_to(m)

        # Add a marker for the start node
        folium.Marker(
            location=(start_node[1], start_node[0]),
            popup=(
                f"Start: {start_node}<br>"
                f"Elevation Gain: {total_gain:.2f} meters<br>"
                f"Max Flood Depth: {max_flood_depth:.4f} meters<br>"
                f"Total Distance: {total_distance:.2f} meters<br>"
                f"Travel Time: {travel_time:.2f} seconds"
            ),
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(m)

        # Add a marker for the end node
        folium.Marker(
            location=(end_node[1], end_node[0]),
            popup=(
                f"End: {end_node}<br>"
                f"Elevation Loss: {total_loss:.2f} meters<br>"
                f"Max Flood Depth: {max_flood_depth:.4f} meters<br>"
                f"Total Distance: {total_distance:.2f} meters<br>"
                f"Travel Time: {travel_time:.2f} seconds"
            ),
            icon=folium.Icon(color="red", icon="stop"),
        ).add_to(m)

    # Save the map to an HTML file
    m.save(output_html)
    print(f"Map with shortest path, start, and end points saved to {output_html}")

# Randomly select two connected nodes from the graph
def select_connected_nodes(G):
    nodes = list(G.nodes)
    # node1 = random.choice(nodes)
    # node2 = random.choice(nodes)
    node1 = (125.5769377, 7.0538513)
    node2 = (125.6024582, 7.0766550)
    # (125.5657858, 7.1161489), # Manila Memorial Park
    # (125.5794607, 7.0664451), # Shrine Hills
    # (125.6024582, 7.0766550), # Rizal Memorial Colleges

    # Ensure the nodes are connected
    # while not nx.has_path(G, node1, node2):
    #     node2 = random.choice(nodes)

    return node1, node2

# # Heuristic function for A*
# def heuristic(node1, node2):
#     # Calculate the straight-line distance between the two nodes
#     return geodesic((node1[1], node1[0]), (node2[1], node2[0])).meters

# New heuristic function
def heuristic_extended(node1, node2, G, alpha, beta, gamma, delta, epsilon):
    # Calculate the straight-line distance as part of the heuristic
    h_n = geodesic((node1[1], node1[0]), (node2[1], node2[0])).meters

    # Initialize metrics
    distance = 0  # Initialize distance to 0
    slope = 0     # Initialize slope to 0
    inundation = 0 # Initialize inundation to 0

    if G.has_edge(node1, node2):
        distance = G[node1][node2]['distance']  # Get distance if edge exists
        slope = calculate_slope(node1, node2, G)  # Calculate slope based on elevation
        inundation = G[node1][node2]['flood_depths'][1] if G.has_edge(node1, node2) else 0  # Define how to get inundation

    # Compute total cost with the new weights
    f_n = (alpha * (distance + h_n) +
           beta * (distance) +   # Adjust as needed
           gamma * (slope) +     # Adjust as needed
           delta * (inundation))  # Adjust as needed

    return f_n

def get_destination_nodes():
    """Returns a list of destination nodes of interest."""
    return [
        (125.5657858, 7.1161489),  # Manila Memorial Park
        (125.5794607, 7.0664451),  # Shrine Hills
        (125.6024582, 7.0766550),  # Rizal Memorial Colleges
        # Add more destinations as needed
    ]

def find_and_store_paths(G, speed_mps, output_csv):
    """
    Find shortest paths from all nodes to specified destinations and store results.
    """
    destinations = get_destination_nodes()
    
    # Get list of all nodes
    all_nodes = list(G.nodes())
    
    # Create progress bar for all combinations
    total_combinations = len(all_nodes) * len(destinations)
    pbar = tqdm(total=total_combinations, desc="Computing paths")
    
    # Open CSV file for writing results as they're computed
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['start_node', 'destination', 'total_distance', 'travel_time', 
                     'elevation_gain', 'elevation_loss', 'max_flood_depth', 'path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Iterate through all nodes as start points
        for start_node in all_nodes:
            # Store paths for current start node to sort them
            current_node_paths = []
            
            # Find paths to all destinations
            for dest_node in destinations:
                try:
                    # Skip if start and destination are the same
                    if start_node == dest_node:
                        pbar.update(1)
                        continue
                    
                    # Find path using A*
                    path = nx.astar_path(G, source=start_node, target=dest_node, 
                                       heuristic=lambda n1, n2: heuristic_extended(n1, n2, G, 0.2, 0.25, 0.2, 0.1, 0.25))
                    
                    # Calculate metrics for the path
                    total_gain, total_loss, max_flood_depth, total_distance, travel_time = calculate_metrics(path, G, speed_mps)
                    
                    # Store path info
                    path_info = {
                        'start_node': start_node,
                        'destination': dest_node,
                        'path': path,
                        'total_distance': total_distance,
                        'travel_time': travel_time,
                        'elevation_gain': total_gain,
                        'elevation_loss': total_loss,
                        'max_flood_depth': max_flood_depth
                    }
                    
                    current_node_paths.append(path_info)
                    
                except nx.NetworkXNoPath:
                    print(f"No path found between {start_node} and {dest_node}")
                
                pbar.update(1)
            
            # Sort paths for current start node by distance and write to CSV
            current_node_paths.sort(key=lambda x: x['total_distance'])
            for path_info in current_node_paths:
                writer.writerow({
                    'start_node': path_info['start_node'],
                    'destination': path_info['destination'],
                    'total_distance': f"{path_info['total_distance']:.2f}",
                    'travel_time': f"{path_info['travel_time']:.2f}",
                    'elevation_gain': f"{path_info['elevation_gain']:.2f}",
                    'elevation_loss': f"{path_info['elevation_loss']:.2f}",
                    'max_flood_depth': f"{path_info['max_flood_depth']:.4f}",
                    'path': json.dumps(path_info['path'])
                })
    
    pbar.close()
    print(f"Results have been saved to {output_csv}")

# Main execution
if __name__ == "__main__":
    geojson_file = 'roads_with_elevation.geojson'
    flood_raster_path = 'davaoFloodMap11_11_24_SRI30.tif'
    output_csv = 'shortest_paths.csv'
    speed_mps = 1.4  # Average walking speed in meters per second

    # Load the GeoJSON and build the graph
    print("Loading GeoJSON and building graph...")
    geojson_data = load_geojson(geojson_file)
    G = build_graph(geojson_data, flood_raster_path)
    
    # Find and store all paths
    print("Finding shortest paths...")
    start_time = time.time()
    find_and_store_paths(G, speed_mps, output_csv)
    end_time = time.time()
    
    print(f"Computation completed in {end_time - start_time:.2f} seconds")
