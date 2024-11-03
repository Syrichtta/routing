import json
import networkx as nx
import random
import numpy as np
import time
import logging
from geopy.distance import geodesic
from tqdm import tqdm
import folium  # Make sure to install folium for visualization
import heapq  # For implementing A* algorithm

# Configure logging
logging.basicConfig(level=logging.INFO, filename="aco_log.txt", filemode="w", format="%(message)s")

# ACO parameters
num_ants = 100
num_iterations = 1
alpha = 5.0        # Pheromone importance
beta = 1.0         # Heuristic importance
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

def coordinates_equal(coord1, coord2, tolerance=0.5):
    return geodesic((coord1[1], coord1[0]), (coord2[1], coord2[0])).meters < tolerance

# A* algorithm implementation
def astar(G, start_node, end_node):
    # Priority queue to store (cost, current_node, path)
    queue = [(0, start_node, [])]
    visited = set()

    while queue:
        cost, current_node, path = heapq.heappop(queue)
        path = path + [current_node]

        if coordinates_equal(current_node, end_node):
            return path, cost  # Return the path and cost if the end node is reached

        if current_node in visited:
            continue

        visited.add(current_node)

        for neighbor in G.neighbors(current_node):
            edge_data = G[current_node][neighbor]
            distance = edge_data['distance']
            heuristic = geodesic((neighbor[1], neighbor[0]), (end_node[1], end_node[0])).meters  # Heuristic: distance to the end node
            heapq.heappush(queue, (cost + distance + heuristic, neighbor, path))

    return None, float('inf')  # Return None if no path is found

def ant_colony_optimization(G, start_node, end_node, pheromone_levels):
    best_path = None
    best_path_length = float('inf')
    path_found = False  # Track if any valid path reaches the end node
    all_paths = []  # Store all paths taken by the ants

    for iteration in range(num_iterations):
        logging.info(f"Iteration {iteration + 1}/{num_iterations}")

        paths = []
        path_lengths = []

        for ant in tqdm(range(num_ants), desc=f"Running ACO (Iteration {iteration + 1})", dynamic_ncols=False):
            current_node = start_node
            stack = [current_node]  # Use a stack for backtracking
            visited = set([current_node])  # Track visited nodes
            path_length = 0

            while stack:
                current_node = stack[-1]  # Look at the top of the stack (current node)

                if current_node == end_node:
                    path_found = True
                    path = list(stack)  # Successful path found
                    path_length = sum(G[stack[i]][stack[i + 1]]["distance"] for i in range(len(stack) - 1))
                    logging.info(f"Ant {ant + 1} completed path: {path} with length: {path_length:.2f}")

                    # Check if this is the best path
                    if path_length < best_path_length:
                        best_path = path
                        best_path_length = path_length
                        logging.info(f"New best path found by Ant {ant + 1} with length: {best_path_length:.2f}")

                    paths.append(path)
                    path_lengths.append(path_length)
                    break  # Exit while loop on success

                # Get neighbors and filter visited ones
                neighbors = list(G.neighbors(current_node))
                unvisited_neighbors = [n for n in neighbors if n not in visited]

                if unvisited_neighbors:
                    # Calculate desirability for unvisited neighbors
                    desirability = []
                    for neighbor in unvisited_neighbors:
                        distance = G[current_node][neighbor]["distance"]
                        edge = tuple(sorted((current_node, neighbor)))
                        pheromone = pheromone_levels.get(edge, 1.0)
                        desirability.append((pheromone ** alpha) * ((1.0 / distance) ** beta))

                    # Select the next node based on probabilities
                    desirability_sum = sum(desirability)
                    probabilities = [d / desirability_sum for d in desirability]
                    next_node = random.choices(unvisited_neighbors, weights=probabilities)[0]

                    # Log the traversal step for this ant
                    logging.info(f"Ant {ant + 1} moves from {current_node} to {next_node}.")

                    stack.append(next_node)  # Move to the next node
                    visited.add(next_node)    # Mark the next node as visited
                    path_length += G[current_node][next_node]["distance"]
                else:
                    # Backtrack if no unvisited neighbors
                    logging.info(f"Ant {ant + 1} stuck at node {current_node} (no unvisited neighbors). Backtracking...")
                    stack.pop()  # Backtrack by removing the last node from the stack

            all_paths.append(stack)  # Store the path taken by this ant

        # Pheromone evaporation
        for edge in pheromone_levels:
            pheromone_levels[edge] *= (1 - evaporation_rate)

        # Pheromone deposit
        for path, length in zip(paths, path_lengths):
            if length > 0:
                pheromone_deposit = pheromone_constant / length
                for i in range(len(path) - 1):
                    edge = tuple(sorted((path[i], path[i + 1])))
                    pheromone_levels[edge] += pheromone_deposit

        logging.info(f"Pheromone levels after iteration {iteration + 1}: {pheromone_levels}")
        logging.info("")  # Add a blank line for readability

    # Ensure returning three values
    if not path_found:
        return None, float('inf'), all_paths  # Return all paths even if no valid path is found

    return best_path, best_path_length, all_paths  # Return all paths

def update_pheromones_with_initial_best_path(G, pheromone_levels, initial_best_path):
    pheromone_deposit = pheromone_constant / len(initial_best_path)  # You can adjust this as needed

    # Loop through the edges in the initial best path and update the pheromone levels
    for i in range(len(initial_best_path) - 1):
        node1 = tuple(initial_best_path[i])
        node2 = tuple(initial_best_path[i + 1])
        edge = tuple(sorted((node1, node2)))

        if edge in pheromone_levels:
            pheromone_levels[edge] += pheromone_deposit
            logging.info(f"Pheromone updated on edge {edge} from initial best path.")
        else:
            logging.warning(f"Edge {edge} not found in graph.")

# Visualization of paths and the entire network
def visualize_paths(G, all_paths, start_node, end_node, output_html='aco_paths_map.html'):
    # Create a base map
    base_map = folium.Map(location=[7.0866, 125.5782], zoom_start=14)  # Center on Davao City

    # Add start and end markers
    folium.Marker(location=(start_node[1], start_node[0]), icon=folium.Icon(color='green', icon='info-sign')).add_to(base_map)
    folium.Marker(location=(end_node[1], end_node[0]), icon=folium.Icon(color='red', icon='info-sign')).add_to(base_map)

    # Visualize the entire network in blue
    for edge in G.edges(data=True):
        node1, node2, _ = edge
        folium.PolyLine(locations=[(lat, lon) for lon, lat in [node1, node2]], color='blue', weight=2.5, opacity=0.7).add_to(base_map)

    # Visualize all paths taken by the ants in red
    for path in all_paths:
        folium.PolyLine(locations=[(lat, lon) for lat, lon in path], color='red', weight=3, opacity=0.6).add_to(base_map)

    # Save the map to an HTML file
    base_map.save(output_html)

# Main script execution
if __name__ == "__main__":
    geojson_file = 'updated_roads.geojson'  # Replace with your GeoJSON file path
    start_node = (125.6217581, 7.0680991)  # Starting coordinates
    end_node = (125.6188844, 7.0671599)  # Ending coordinates
    output_html = 'aco_paths_map.html'

    # Load data and build the graph
    geojson_data = load_geojson(geojson_file)
    G = build_graph(geojson_data)

    # Initialize pheromone levels
    pheromone_levels = {}
    for edge in G.edges():
        pheromone_levels[tuple(sorted(edge))] = 1.0  # Initial pheromone level

    # Run A* algorithm to find the initial best path
    best_path, best_path_length = astar(G, start_node, end_node)

    # Define the initial best path
    initial_best_path = [
        (125.6217581, 7.0680991),
        (125.6217711, 7.0681008),
        (125.6217671, 7.0680842),
        (125.6217424, 7.0680265),
        (125.6217304, 7.0679912),
        (125.621711, 7.0679598),
        (125.6216923, 7.0679295),
        (125.6216519, 7.0678526),
        (125.6216057, 7.0677893),
        (125.6215658, 7.0677247),
        (125.6210472, 7.0679187),
        (125.6208534, 7.0679878),
        (125.6207022, 7.0675846),
        (125.6205109, 7.0676591),
        (125.6203098, 7.0677375),
        (125.620115, 7.0678135),
        (125.619957, 7.0674275),
        (125.6197762, 7.0674967),
        (125.6195992, 7.0675696),
        (125.6193847, 7.0676513),
        (125.6193492, 7.0675539),
        (125.6192407, 7.0673385),
        (125.6192072, 7.067275),
        (125.6190073, 7.0673746),
        (125.6188844, 7.0671599)
    ]

    # Update pheromones with the initial best path
    if best_path:
        update_pheromones_with_initial_best_path(G, pheromone_levels, initial_best_path)

    # Run the Ant Colony Optimization
    best_path, best_path_length, all_paths = ant_colony_optimization(G, start_node, end_node, pheromone_levels)

    # Visualize paths
    visualize_paths(G, all_paths, start_node, end_node, output_html)
