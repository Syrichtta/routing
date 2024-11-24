import json
import networkx as nx
import folium
import random
from geopy.distance import geodesic
import time
import numpy as np
from pyproj import Transformer
import rasterio
from tqdm import tqdm

# Load GeoJSON data
def load_geojson(file_path):
    with open(file_path) as f:
        return json.load(f)
    
def load_betweenness_from_json(json_file):
    with open(json_file) as f:
        return json.load(f)

# Build the graph from GeoJSON
def build_graph(geojson_data):
    G = nx.Graph()

    for feature in geojson_data['features']:
        coordinates = feature['geometry']['coordinates']
        elevations = feature['properties'].get('elevations', [0] * len(coordinates))
        flood_depths = feature['properties'].get('flood_depths', [0] * len(coordinates))

        for i in range(len(coordinates) - 1):
            node1 = tuple(coordinates[i])
            node2 = tuple(coordinates[i + 1])

            dist = geodesic((coordinates[i][1], coordinates[i][0]), (coordinates[i + 1][1], coordinates[i + 1][0])).meters
            G.add_edge(node1, node2, weight=dist, distance=dist, elevations=(elevations[i], elevations[i + 1]), flood_depths=(flood_depths[i], flood_depths[i + 1]))

    return G

def update_betweenness_from_json(G, betweenness_json):
    for node, b_prime in betweenness_json.items():
        # The node in the betweenness JSON should match the node format in the graph
        node_coordinates = eval(node)  # Convert string back to tuple (e.g., "(125.6147335, 7.0948561)" -> (125.6147335, 7.0948561)
        if node_coordinates in G.nodes:
            G.nodes[node_coordinates]['b_prime'] = round(b_prime, 4)

# Calculate elevation gain/loss, maximum flood depth, and total distance
def calculate_metrics(path, G, speed_mps):
    total_gain = 0
    total_loss = 0
    max_flood_depth = 0
    total_distance = 0

    for i in range(len(path) - 1):
        node1 = path[i]
        node2 = path[i+1]
        # print(f"node 1: {node1}")
        # print(f"node 2: {node2}")
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
    # node1 = (125.5920339, 7.1219125)
    node1 = (125.6305739, 7.0927439)
    node2 = (125.5794607, 7.0664451)

    # node1 = random.choice(nodes)
    # node2 = random.choice(nodes)

    # (125.5657858, 7.1161489), # Manila Memorial Park
    # (125.5794607, 7.0664451), # Shrine Hills
    # (125.6024582, 7.0766550), # Rizal Memorial Colleges

    # Ensure the nodes are connected
    if not nx.has_path(G, node1, node2):
        print('node path')

    return node1, node2


def heuristic_extended(node1, node2, G, alpha=1, beta=1, gamma=1, delta=1, epsilon=1):
    # Calculate straight-line distance heuristic (h(n))
    # Assuming nodes have 'pos' attribute with (latitude, longitude)
    try:
        h_n = geodesic(
            G.nodes[node1]['pos'][::-1],  # Swap order for (lon, lat)
            G.nodes[node2]['pos'][::-1]
        ).meters
    except (KeyError, Exception):
        # Fallback to Euclidean distance if geodesic fails
        try:
            h_n = np.linalg.norm(
                np.array(G.nodes[node1]['pos']) - 
                np.array(G.nodes[node2]['pos'])
            )
        except:
            h_n = float('inf')

    # Get inverse betweenness centrality
    b_prime = G.nodes[node1].get('b_prime', 0)

    # Initialize metrics
    distance, slope, flood_depth = 0, 0, 0

    # Get edge data if edge exists
    edge_data = G.get_edge_data(node1, node2)
    if edge_data:
        distance = edge_data.get('distance', 0)
        slope = calculate_slope(node1, node2, G)
        flood_depth = max(edge_data.get('flood_depths', [0]))

    # Compute total cost 
    f_n = (
        alpha * h_n +  # Only h(n) now
        beta * b_prime +
        gamma * distance +
        delta * slope +
        epsilon * flood_depth
    )
    return f_n



# Main logic to load the GeoJSON and run A*
geojson_file = 'roads_with_elevation_and_flood.geojson'
flood_raster_path = 'davaoFloodMap11_11_24_SRI30.tif'  
betweenness_path = 'betweenness_data.json'
output_html = 'shortest_path_map.html'

# Average walking speed in meters per second (adjust this as needed)
speed_mps = 1.4

# Load the GeoJSON and build the graph
geojson_data = load_geojson(geojson_file)
G = build_graph(geojson_data)  

# Load betweenness from JSON
betweenness_json = load_betweenness_from_json(betweenness_path)

# Update the graph nodes with b_prime values
update_betweenness_from_json(G, betweenness_json)

start_node, end_node = select_connected_nodes(G)
print(f"Start node: {start_node}, End node: {end_node}")

# betweenness = {}
# nodes = list(G.nodes)
# print("Calculating betweenness centrality...")
# for node in tqdm(nodes, desc="Processing nodes"):
#     betweenness[node] = sum(
#         nx.single_source_dijkstra_path_length(G, node, weight='distance').values()
#     )

# # Normalize betweenness centrality
# max_b = max(betweenness.values())
# betweenness = {node: value / max_b for node, value in betweenness.items()}


# print("Adding inverse betweenness centrality to graph nodes...")
# for node in tqdm(G.nodes, desc="Processing nodes"):
#     G.nodes[node]['b_prime'] = max_b - betweenness[node]

# Measure computation time for A* algorithm
start_time = time.time()

try:
    # Update A* call with the new heuristic
    shortest_path = nx.astar_path(G, start_node, end_node, 
                     heuristic=lambda n1, n2: heuristic_extended(n1, n2, G),
                     weight='weight')

    end_time = time.time()
    computation_time = end_time - start_time
    print(f"Shortest path computed in {computation_time:.4f} seconds")

    # print("Best path found (list of nodes):")
    # for node in shortest_path:
    #     print(f"{node},")

    # Calculate metrics
    total_gain, total_loss, max_flood_depth, total_distance, travel_time = calculate_metrics(shortest_path, G, speed_mps)

    
    
    # Visualize the path
    visualize_path(geojson_data, shortest_path, output_html, total_gain, total_loss, max_flood_depth, total_distance, travel_time, start_node, end_node)
    print(shortest_path)
    print(f"Total Distance: {total_distance:.2f} meters")
    print(f"Travel Time: {travel_time:.2f} seconds")
    print(f"Elevation Gain: {total_gain:.2f} meters, Elevation Loss: {total_loss:.2f} meters")
    print(f"Max Flood Depth: {max_flood_depth:.4f} meters")

except nx.NetworkXNoPath:
    print("No path found between the selected nodes.")
