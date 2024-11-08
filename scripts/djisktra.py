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
        flood_depths = feature['properties'].get('flood_depths', [0] * len(coordinates))

        for i in range(len(coordinates) - 1):
            node1 = tuple(coordinates[i])
            node2 = tuple(coordinates[i+1])

            # Calculate distance between nodes
            dist = geodesic((coordinates[i][1], coordinates[i][0]), (coordinates[i+1][1], coordinates[i+1][0])).meters

            # Add edge with distance as weight
            G.add_edge(node1, node2, weight=dist, distance=dist, elevations=(elevations[i], elevations[i+1]), flood_depths=(flood_depths[i], flood_depths[i+1]))

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

        elevation1 = elevations[0] if elevations[0] is not None else 0
        elevation2 = elevations[1] if elevations[1] is not None else 0

        elevation_diff = elevation2 - elevation1
        if elevation_diff > 0:
            total_gain += elevation_diff
        else:
            total_loss += abs(elevation_diff)

        for depth in flood_depths:
            if depth is not None:
                max_flood_depth = max(max_flood_depth, depth)

        total_distance += edge_data['distance']

    travel_time = total_distance / speed_mps
    return total_gain, total_loss, max_flood_depth, total_distance, travel_time

def visualize_path(geojson_data, path, output_html, total_gain, total_loss, max_flood_depth, total_distance, travel_time, start_node, end_node):
    central_point = [7.0512, 125.5987]  # Update to your area
    m = folium.Map(location=central_point, zoom_start=15)

    folium.GeoJson(geojson_data, name='Roads').add_to(m)

    if path:
        path_coordinates = [(p[1], p[0]) for p in path]
        folium.PolyLine(path_coordinates, color="blue", weight=5, opacity=0.7).add_to(m)

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

    m.save(output_html)
    print(f"Map with shortest path, start, and end points saved to {output_html}")

def select_connected_nodes(G):
    nodes = list(G.nodes)
    # node1 = random.choice(nodes)
    node1 = (125.6023788, 7.063822)
    node2 = (125.579685, 7.0656892)
    return node1, node2

# Main logic to load the GeoJSON and run Dijkstra
geojson_file = 'davao_specific_barangays_road_network.geojson'
output_html = 'shortest_path_map.html'
speed_mps = 1.4

# Load the GeoJSON and build the graph
geojson_data = load_geojson(geojson_file)
G = build_graph(geojson_data)

# Select two random connected nodes
start_node, end_node = select_connected_nodes(G)
print(f"Start node: {start_node}, End node: {end_node}")

# Measure computation time for Dijkstra's algorithm
start_time = time.time()

try:
    shortest_path = nx.dijkstra_path(G, source=start_node, target=end_node, weight='weight')
    end_time = time.time()
    computation_time = end_time - start_time
    print(f"Shortest path computed in {computation_time:.4f} seconds")

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
