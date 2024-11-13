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
import rasterio
from pyproj import Transformer
from tqdm import tqdm


# Configure logging
logging.basicConfig(level=logging.INFO, filename="aco_log.txt", filemode="w", format="%(message)s")

# Function to load the initial best path from a JSON file
def load_initial_best_path(json_file, start_node, end_node):
    with open(json_file, 'r') as f:
        paths = json.load(f)

    for path_entry in paths:
        # Check if the start and end nodes match
        if coordinates_equal(path_entry["start_node"], start_node) and coordinates_equal(path_entry["destination"], end_node):
            logging.info(f"Found initial best path for {start_node} to {end_node}")
            return path_entry["path"]
    
    logging.warning(f"No initial best path found for {start_node} to {end_node}")
    return None


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
    # for edge in G.edges(data=True):
    #     node1, node2, _ = edge
    #     folium.PolyLine(locations=[(lat, lon) for lon, lat in [node1, node2]], color='blue', weight=2.5, opacity=0.7).add_to(base_map)

    # Visualize all paths taken by the ants in red
    for path in all_paths:
        folium.PolyLine(locations=[(lat, lon) for lat, lon in path], color='red', weight=3, opacity=0.6).add_to(base_map)

    # Save the map to an HTML file
    base_map.save(output_html)

if __name__ == "__main__":
    geojson_file = 'roads_with_elevation.geojson'  # Replace with your GeoJSON file path
    json_file = 'shortest_paths_from_txt.json'  # Path to your JSON file with initial best paths
    start_node = (125.5895695, 7.0537547)  # Starting coordinates
    end_node = (125.5657858, 7.1161489)  # Ending coordinates
    output_html = 'astar_aco_paths_map.html'
    flood_raster_path = 'davaoFloodMap11_11_24_SRI30.tif'

    # Load data from the JSON file
    with open('shortest_paths_from_txt.json', 'r') as file:
        data = json.load(file)

    # Check if the data is a list and contains at least one item
    if isinstance(data, list) and len(data) > 0:
        path_data = data[0]  # Assuming the first element contains the required information
        if 'path' in path_data:
            initial_best_path = [(coords[0], coords[1]) for coords in path_data['path']]
            print("Initial best path:", initial_best_path)
        else:
            print("Error: 'path' key is missing in the data.")
    else:
        print("Error: The JSON data is not in the expected format or is empty.")



    # print("Loading GeoJSON and building graph...")
    # geojson_data = load_geojson(geojson_file)
    # G = build_graph(geojson_data, flood_raster_path)

    # # Initialize pheromone levels
    # pheromone_levels = {}
    # for edge in G.edges():
    #     pheromone_levels[tuple(sorted(edge))] = 1.0  # Initial pheromone level

    # # Run A* algorithm to find the initial best path
    # best_path, best_path_length = astar(G, start_node, end_node)

    # # Load the initial best path from the JSON file
    # initial_best_path = load_initial_best_path(json_file, start_node, end_node)

    # if initial_best_path:
    #     # Update pheromones with the initial best path
    #     update_pheromones_with_initial_best_path(G, pheromone_levels, initial_best_path)

    # # Run the Ant Colony Optimization
    # best_path, best_path_length, all_paths = ant_colony_optimization(G, start_node, end_node, pheromone_levels)

    # # Visualize paths
    # visualize_paths(G, all_paths, start_node, end_node, output_html)


