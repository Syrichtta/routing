import json
import networkx as nx
import random
import numpy as np
import time
import logging
from geopy.distance import geodesic
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, filename="aco_log.txt", filemode="w", format="%(message)s")

# ACO parameters
num_ants = 100
num_iterations = 10000
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


# Main script
geojson_file = 'updated_roads.geojson'
output_html = 'aco_path_map.html'

# Load GeoJSON and build graph
geojson_data = load_geojson(geojson_file)
G = build_graph(geojson_data)
start_node, end_node = (125.5722492, 7.089552), (125.5868572, 7.0836662)
# Start node: (125.5722492, 7.089552), End node: (125.5868572, 7.0836662)

start_time = time.time()
best_path, best_path_length = ant_colony_optimization(G, start_node, end_node)
end_time = time.time()

# Display results
if best_path:
    print(f"ACO completed in {end_time - start_time:.2f} seconds")
    print(f"Best path length: {best_path_length:.2f} meters")
else:
    print("ACO failed to find a path between the start and end nodes.")
