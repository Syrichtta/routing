import json
import networkx as nx
import folium
import random
from geopy.distance import geodesic
import time
from pyproj import Transformer
import rasterio
from tqdm import tqdm

# Load GeoJSON data
def load_geojson(file_path):
    with open(file_path) as f:
        return json.load(f)

# Load GeoJSON data
def load_geojson(file_path):
    with open(file_path) as f:
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
    node1 = (125.6015325, 7.0647666)
    node2 = (125.6024582, 7.0766550)

    # node1 = random.choice(nodes)
    # node2 = random.choice(nodes)

    # (125.5657858, 7.1161489), # Manila Memorial Park
    # (125.5794607, 7.0664451), # Shrine Hills
    # (125.6024582, 7.0766550), # Rizal Memorial Colleges

    # Ensure the nodes are connected
    # while not nx.has_path(G, node1, node2):
    #     node2 = random.choice(nodes)

    return node1, node2


def heuristic_extended(node1, node2, G, alpha, beta, gamma, delta):
    # Calculate the straight-line distance (h(n))
    h_n = geodesic((node1[1], node1[0]), (node2[1], node2[0])).meters
    
    # Get g(n): cost from start node to current node
    g_n = nx.shortest_path_length(G, source=node1, target=node2, weight='distance')
    
    # Get b'(n): inverse betweenness centrality
    b_prime = G.nodes[node1]['b_prime']
    
    # Get i(n) and j(n): distance and slope parameters
    if G.has_edge(node1, node2):
        distance = G[node1][node2]['distance']
        slope = calculate_slope(node1, node2, G)
    else:
        distance, slope = 0, 0
    
    # Compute total cost
    f_n = (alpha * (g_n + h_n)) + (beta * b_prime) + (gamma * distance) + (delta * slope)
    return f_n



# Main logic to load the GeoJSON and run A*
geojson_file = 'roads_with_elevation_and_flood.geojson'
flood_raster_path = 'davaoFloodMap11_11_24_SRI30.tif'  
output_html = 'shortest_path_map.html'

# Average walking speed in meters per second (adjust this as needed)
speed_mps = 1.4

# Load the GeoJSON and build the graph
geojson_data = load_geojson(geojson_file)
G = build_graph(geojson_data)  


# Select two random connected nodes
start_node, end_node = select_connected_nodes(G)
print(f"Start node: {start_node}, End node: {end_node}")

betweenness = {}
nodes = list(G.nodes)
print("Calculating betweenness centrality...")
for node in tqdm(nodes, desc="Processing nodes"):
    betweenness[node] = sum(
        nx.single_source_dijkstra_path_length(G, node, weight='distance').values()
    )

# Normalize betweenness centrality
max_b = max(betweenness.values())
betweenness = {node: value / max_b for node, value in betweenness.items()}


print("Adding inverse betweenness centrality to graph nodes...")
for node in tqdm(G.nodes, desc="Processing nodes"):
    G.nodes[node]['b_prime'] = max_b - betweenness[node]

# Measure computation time for A* algorithm
start_time = time.time()

try:
    # Update A* call with the new heuristic
    shortest_path = nx.astar_path(
    G, 
    source=start_node, 
    target=end_node, 
    heuristic=lambda n1, n2: heuristic_extended(n1, n2, G, alpha=0.2, beta=0.25, gamma=0.2, delta=0.1), 
    weight='distance'
)

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
    
    print(f"Total Distance: {total_distance:.2f} meters")
    print(f"Travel Time: {travel_time:.2f} seconds")
    print(f"Elevation Gain: {total_gain:.2f} meters, Elevation Loss: {total_loss:.2f} meters")
    print(f"Max Flood Depth: {max_flood_depth:.4f} meters")

except nx.NetworkXNoPath:
    print("No path found between the selected nodes.")
