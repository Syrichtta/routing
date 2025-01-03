import json
import networkx as nx
import folium
import random
from geopy.distance import geodesic
import time

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

        for i in range(len(coordinates) - 1):
            # Define nodes and add to the graph
            node1 = tuple(coordinates[i])
            node2 = tuple(coordinates[i+1])

            # Calculate distance between the two nodes as the edge weight
            dist = geodesic((coordinates[i][1], coordinates[i][0]), (coordinates[i+1][1], coordinates[i+1][0])).meters

            # Add edge with distance weight and store elevation data
            G.add_edge(node1, node2, weight=dist, distance=dist, elevations=(elevations[i], elevations[i+1]))

    return G

# Calculate elevation gain/loss and total distance
def calculate_metrics(path, G, speed_mps):
    total_gain = 0
    total_loss = 0
    total_distance = 0

    for i in range(len(path) - 1):
        node1 = path[i]
        node2 = path[i+1]
        edge_data = G.get_edge_data(node1, node2)
        elevations = edge_data['elevations']

        # Check for valid elevation values
        elevation1 = elevations[0] if elevations[0] is not None else 0
        elevation2 = elevations[1] if elevations[1] is not None else 0

        # Calculate elevation gain/loss
        elevation_diff = elevation2 - elevation1
        if elevation_diff > 0:
            total_gain += elevation_diff
        else:
            total_loss += abs(elevation_diff)

        # Calculate total distance traveled
        total_distance += edge_data['distance']

    # Calculate travel time based on total distance and speed (in meters per second)
    travel_time = total_distance / speed_mps

    return total_gain, total_loss, total_distance, travel_time

def calculate_slope(node1, node2, G):
    if G.has_edge(node1, node2):
        edge_data = G[node1][node2]
        elevation1 = edge_data['elevations'][0]
        elevation2 = edge_data['elevations'][1]
        horizontal_distance = edge_data['distance']

        # Calculate the change in elevation
        elevation_change = elevation2 - elevation1

        # Calculate slope (rise/run)
        if horizontal_distance > 0:  # Prevent division by zero
            slope = elevation_change / horizontal_distance
        else:
            slope = 0

        return slope
    else:
        return 0

# Visualize the shortest path on a map with start/end pins and metrics
def visualize_path(geojson_data, path, output_html, total_gain, total_loss, total_distance, travel_time, start_node, end_node):
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
    node1 = random.choice(nodes)
    node2 = (125.6024582, 7.0766550)  # Rizal Memorial Colleges
    return node1, node2

# New heuristic function without flood depth
def heuristic_extended(node1, node2, G, alpha, beta, gamma):
    # Calculate the straight-line distance as part of the heuristic
    h_n = geodesic((node1[1], node1[0]), (node2[1], node2[0])).meters

    # Initialize metrics
    distance = 0
    slope = 0

    if G.has_edge(node1, node2):
        distance = G[node1][node2]['distance']
        slope = calculate_slope(node1, node2, G)

    # Compute total cost with weights
    f_n = (alpha * (distance + h_n) +
           beta * (distance) +
           gamma * (slope))

    return f_n

# Main logic
geojson_file = 'roads_with_elevation.geojson'
output_html = 'shortest_path_map.html'
speed_mps = 1.4  # Average walking speed in meters per second

# Load the GeoJSON and build the graph
geojson_data = load_geojson(geojson_file)
G = build_graph(geojson_data)

# Select two random connected nodes
start_node, end_node = select_connected_nodes(G)
print(f"Start node: {start_node}, End node: {end_node}")

# Measure computation time for A* algorithm
start_time = time.time()

try:
    shortest_path = nx.astar_path(G, source=start_node, target=end_node, 
                                heuristic=lambda n1, n2: heuristic_extended(n1, n2, G, 0.3, 0.35, 0.35))
    end_time = time.time()
    computation_time = end_time - start_time
    print(f"Shortest path computed in {computation_time:.4f} seconds")

    # print("Best path found (list of nodes):")
    # for node in shortest_path:
    #     print(f"{node},")

    # Calculate metrics
    total_gain, total_loss, total_distance, travel_time = calculate_metrics(shortest_path, G, speed_mps)
    
    # Visualize the path
    visualize_path(geojson_data, shortest_path, output_html, total_gain, total_loss, 
                  total_distance, travel_time, start_node, end_node)
    
    print(f"Total Distance: {total_distance:.2f} meters")
    print(f"Travel Time: {travel_time:.2f} seconds")
    print(f"Elevation Gain: {total_gain:.2f} meters, Elevation Loss: {total_loss:.2f} meters")

except nx.NetworkXNoPath:
    print("No path found between the selected nodes.")