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
from pyproj import Transformer
import rasterio

# Configure logging
logging.basicConfig(level=logging.CRITICAL, filename="aco_log.txt", filemode="w", format="%(message)s")

# ACO parameters
num_ants = 10
num_iterations = 10
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
    geojson_file = 'roads_with_elevation.geojson'  # Replace with your GeoJSON file path
    flood_raster_path = 'davaoFloodMap11_11_24_SRI30.tif'
    start_node = (125.6015325, 7.0647666)  # Starting coordinates
    end_node = (125.6024582, 7.0766550)  # Ending coordinates
    output_html = 'astar_aco_paths_map.html'

    # Load data and build the graph
    geojson_data = load_geojson(geojson_file)
    G = build_graph(geojson_data)

    # Initialize pheromone levels
    pheromone_levels = {}
    for edge in G.edges():
        pheromone_levels[tuple(sorted(edge))] = 1.0  # Initial pheromone level

    # # Run A* algorithm to find the initial best path
    # best_path, best_path_length = astar(G, start_node, end_node)

    # Define the initial best path
    initial_best_path = [
        (125.6015325, 7.0647666),
        (125.6015883, 7.0650632),
        (125.6018902, 7.0665857),
        (125.6019752, 7.0669411),
        (125.6020265, 7.0671295),
        (125.6021154, 7.0673791),
        (125.6021829, 7.0675542),
        (125.6022438, 7.0676806),
        (125.6023127, 7.0678194),
        (125.6023757, 7.0679197),
        (125.6024377, 7.0680005),
        (125.60252, 7.0680908),
        (125.6027502, 7.0682837),
        (125.6032125, 7.0686836),
        (125.6034827, 7.0689173),
        (125.6041372, 7.0694524),
        (125.6045038, 7.0697522),
        (125.6048012, 7.0699953),
        (125.604844, 7.0700303),
        (125.6048536, 7.0701051),
        (125.6048775, 7.0702905),
        (125.6049015, 7.0704764),
        (125.6049277, 7.0706797),
        (125.6049573, 7.0709099),
        (125.6050447, 7.0715901),
        (125.6050815, 7.0718762),
        (125.6051541, 7.0723902),
        (125.6051718, 7.0725155),
        (125.6051928, 7.0726641),
        (125.6051998, 7.0727191),
        (125.605227, 7.0730088),
        (125.6052502, 7.0732124),
        (125.6052062, 7.073266),
        (125.6049846, 7.073536),
        (125.6049202, 7.0736145),
        (125.6045904, 7.0740163),
        (125.6043522, 7.0743064),
        (125.6040902, 7.0746202),
        (125.6040195, 7.0747048),
        (125.6036306, 7.0751707),
        (125.6033365, 7.0755381),
        (125.6032835, 7.0756043),
        (125.6032472, 7.0756488),
        (125.6026112, 7.0764263),
        (125.6025067, 7.0765793),
        (125.6024582, 7.076655)
    ]

    # # Update pheromones with the initial best path
    # if best_path:
    update_pheromones_with_initial_best_path(G, pheromone_levels, initial_best_path)

    # Run the Ant Colony Optimization
    best_path, best_path_length, all_paths = ant_colony_optimization(G, start_node, end_node, pheromone_levels)

    # Visualize paths
    visualize_paths(G, all_paths, start_node, end_node, output_html)
