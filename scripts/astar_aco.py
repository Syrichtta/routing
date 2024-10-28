import json
import networkx as nx
import random
import numpy as np
import time
import logging
from geopy.distance import geodesic
import folium
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, filename="aco_log.txt", filemode="w", format="%(message)s")

# ACO parameters
num_ants = 10
num_iterations = 100
alpha = 1.0        # Pheromone importance
beta = 5.0         # Heuristic importance
evaporation_rate = 0.5
pheromone_constant = 100.0

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

# ACO function with logging
def ant_colony_optimization(G, start_node, end_node):
    pheromone_levels = {tuple(sorted(edge)): 1.0 for edge in G.edges()}
    best_path = None
    best_path_length = float('inf')

    for iteration in range(num_iterations):
        logging.info(f"Iteration {iteration + 1}/{num_iterations}")

        paths = []
        path_lengths = []

        for ant in tqdm(range(num_ants), desc=f"Running ACO (Iteration {iteration + 1})"):
            current_node = start_node
            path = [current_node]
            path_length = 0

            while current_node != end_node:
                neighbors = list(G.neighbors(current_node))
                neighbors = [n for n in neighbors if n not in path]

                if not neighbors:
                    logging.info(f"Ant {ant + 1} stuck at node {current_node} (no valid neighbors).")
                    break

                desirability = []
                for neighbor in neighbors:
                    distance = G[current_node][neighbor]["distance"]
                    edge = tuple(sorted((current_node, neighbor)))
                    pheromone = pheromone_levels.get(edge, 1.0)  # Default to 1.0 if missing
                    desirability.append((pheromone ** alpha) * ((1.0 / distance) ** beta))

                desirability_sum = sum(desirability)
                probabilities = [d / desirability_sum for d in desirability]
                next_node = random.choices(neighbors, weights=probabilities)[0]

                path.append(next_node)
                path_length += G[current_node][next_node]["distance"]
                current_node = next_node

            logging.info(f"Ant {ant + 1} completed path: {path} with length: {path_length:.2f}")

            if current_node == end_node and path_length < best_path_length:
                best_path = path
                best_path_length = path_length
                logging.info(f"New best path found by Ant {ant + 1} with length: {best_path_length:.2f}")

            paths.append(path)
            path_lengths.append(path_length)

        for edge in pheromone_levels:
            pheromone_levels[edge] *= (1 - evaporation_rate)

        for path, length in zip(paths, path_lengths):
            if length > 0:
                pheromone_deposit = pheromone_constant / length
                for i in range(len(path) - 1):
                    edge = tuple(sorted((path[i], path[i + 1])))
                    pheromone_levels[edge] += pheromone_deposit

        logging.info(f"Pheromone levels after iteration {iteration + 1}: {pheromone_levels}")

    return best_path, best_path_length

# Calculate elevation gain/loss, maximum flood depth, and total distance
def calculate_metrics(path, G, speed_mps):
    total_gain = 0
    total_loss = 0
    max_flood_depth = 0
    total_distance = 0

    for i in range(len(path) - 1):
        node1 = path[i]
        node2 = path[i + 1]
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
            if depth is not None:  # Only consider valid flood depth values
                max_flood_depth = max(max_flood_depth, depth)

        total_distance += edge_data['distance']

    travel_time = total_distance / speed_mps

    return total_gain, total_loss, max_flood_depth, total_distance, travel_time

# Visualize the shortest path on a map with start/end pins and metrics
def visualize_path(geojson_data, path, output_html, total_gain, total_loss, max_flood_depth, total_distance, travel_time, start_node, end_node):
    central_point = [7.0512, 125.5987]  # Update to your area
    m = folium.Map(location=central_point, zoom_start=15)

    folium.GeoJson(geojson_data, name='Roads').add_to(m)

    if path:
        path_coordinates = [(p[1], p[0]) for p in path]  # Folium expects lat, lon order
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

# Heuristic function for A*
def heuristic(node1, node2):
    return geodesic((node1[1], node1[0]), (node2[1], node2[0])).meters

# Main script
geojson_file = 'updated_roads.geojson'
output_html = 'aco_path_map.html'
speed_mps = 1.4  # Average walking speed in meters per second

# Load GeoJSON and build graph
geojson_data = load_geojson(geojson_file)
G = build_graph(geojson_data)
start_node, end_node = (125.5722492, 7.089552), (125.5868572, 7.0836662)  # Update to your start and end nodes

start_time = time.time()

# Initialize pheromone trails using A*
try:
    a_star_path = nx.astar_path(G, source=start_node, target=end_node, heuristic=heuristic)
    a_star_length = sum(G[a_star_path[i]][a_star_path[i + 1]]['distance'] for i in range(len(a_star_path) - 1))
    print(f"A* path: {a_star_path}, Length: {a_star_length:.2f} meters")
except nx.NetworkXNoPath:
    print("No path found between the selected nodes.")
    exit()

# Run ACO to optimize the path
best_path, best_path_length = ant_colony_optimization(G, start_node, end_node)
end_time = time.time()

# Display results
if best_path:
    total_gain, total_loss, max_flood_depth, total_distance, travel_time = calculate_metrics(best_path, G, speed_mps)
    print(f"Best path found: {best_path}, Length: {best_path_length:.2f} meters")
    print(f"Total Elevation Gain: {total_gain:.2f} meters, Total Elevation Loss: {total_loss:.2f} meters")
    print(f"Maximum Flood Depth: {max_flood_depth:.2f} meters, Total Distance: {total_distance:.2f} meters")
    print(f"Estimated Travel Time: {travel_time:.2f} seconds")

    # Visualize the path
    visualize_path(geojson_data, best_path, output_html, total_gain, total_loss, max_flood_depth, total_distance, travel_time, start_node, end_node)

print(f"Script executed in {end_time - start_time:.2f} seconds.")
