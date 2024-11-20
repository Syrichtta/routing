import json
import networkx as nx
import random
import numpy as np
import time
import logging
from geopy.distance import geodesic
from tqdm import tqdm
import folium

# Configure logging
logging.basicConfig(level=logging.INFO, filename="aco_log.txt", filemode="w", format="%(message)s")

# ACO parameters
num_ants = 10
num_iterations = 10
alpha = 1.0        # Pheromone importance
beta = 2.0         # Heuristic importance
evaporation_rate = 0.3
pheromone_constant = 100.0

# Function to calculate slope (from A* script)
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

# Heuristic function from A* script (extended version)
def heuristic_extended(node1, node2, G, alpha=0.2, beta=0.25, gamma=0.2, delta=0.1, epsilon=7):
    # Calculate the straight-line distance (h(n))
    h_n = geodesic((node1[1], node1[0]), (node2[1], node2[0])).meters

    # Get g(n): cost from start node to current node (assuming a distance weight)
    try:
        g_n = nx.shortest_path_length(G, source=node1, target=node2, weight='distance')
    except nx.NetworkXNoPath:
        g_n = float('inf')  # Handle no-path case

    # Get b'(n): inverse betweenness centrality
    b_prime = G.nodes[node1].get('b_prime', 0)

    # Initialize metrics
    distance, slope, flood_depth = 0, 0, 0

    # Get edge data if edge exists
    edge_data = G.get_edge_data(node1, node2)
    if edge_data:
        distance = edge_data.get('distance', 0)
        slope = calculate_slope(node1, node2, G)
        flood_depth = max(edge_data.get('flood_depths', [0]))

    # Compute total cost (including flood depth as a penalty)
    f_n = (
        alpha * (g_n + h_n) +
        beta * b_prime +
        gamma * distance +
        delta * slope +
        epsilon * flood_depth
    )
    return f_n

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

def ant_colony_optimization(G, start_node, end_node, initial_best_path=None):
    # If an initial best path is provided, calculate its length
    initial_best_path_length = float('inf')
    if initial_best_path:
        initial_best_path_length = sum(G[initial_best_path[i]][initial_best_path[i + 1]]["distance"] 
                                       for i in range(len(initial_best_path) - 1))
    
    # Maximum allowed path length (twice the initial best path)
    max_path_length = initial_best_path_length * 2 if initial_best_path else float('inf')

    pheromone_levels = {tuple(sorted(edge)): 1.0 for edge in G.edges()}
    best_path = initial_best_path
    best_path_length = initial_best_path_length
    path_found = False if initial_best_path is None else True
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
                    path = list(stack)  # Successful path found
                    path_length = sum(G[stack[i]][stack[i + 1]]["distance"] for i in range(len(stack) - 1))
                    
                    # Check if this path length is within the constraint
                    if path_length <= max_path_length:
                        path_found = True
                        logging.info(f"Ant {ant + 1} completed path: {path} with length: {path_length:.2f}")
                        
                        # Check if this is the best path
                        if path_length < best_path_length:
                            best_path = path
                            best_path_length = path_length
                            logging.info(f"New best path found by Ant {ant + 1} with length: {best_path_length:.2f}")

                        paths.append(path)
                        path_lengths.append(path_length)
                    else:
                        logging.info(f"Ant {ant + 1} path discarded: Length {path_length:.2f} exceeds max {max_path_length:.2f}")
                    
                    break  # Exit while loop on success

                # Get neighbors and filter visited ones
                neighbors = list(G.neighbors(current_node))
                unvisited_neighbors = [n for n in neighbors if n not in visited]

                if unvisited_neighbors:
                    # Decide which heuristic to use based on random number
                    random_choice = round(random.random(), 1)
                    
                    if random_choice > 0.5:
                        # Use A* heuristic function to choose the next node
                        desirability = []
                        for neighbor in unvisited_neighbors:
                            # Calculate heuristic value to the end node
                            heuristic_value = heuristic_extended(neighbor, end_node, G)
                            desirability.append(heuristic_value)
                        
                        # Select the next node with the lowest heuristic value
                        next_node = unvisited_neighbors[desirability.index(min(desirability))]
                        logging.info(f"Ant {ant + 1} uses A* heuristic to select node {next_node}")
                    else:
                        # Use original ACO heuristic
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
                        logging.info(f"Ant {ant + 1} uses ACO heuristic to select node {next_node}")

                    # Log the traversal step for this ant
                    logging.info(f"Ant {ant + 1} moves from {current_node} to {next_node}.")

                    # Check if adding this next node would exceed max path length
                    potential_path_length = path_length + G[current_node][next_node]["distance"]
                    if potential_path_length <= max_path_length:
                        stack.append(next_node)  # Move to the next node
                        visited.add(next_node)    # Mark the next node as visited
                        path_length = potential_path_length
                    else:
                        logging.info(f"Ant {ant + 1} prevented from adding node {next_node} to prevent exceeding max path length")
                        break
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

def calculate_metrics(path, G, speed_mps=1.5):  # Default walking speed of 1.5 m/s
    """
    Calculate path metrics including elevation gain/loss, maximum flood depth, 
    total distance, and travel time.
    
    :param path: List of nodes representing the path
    :param G: NetworkX graph
    :param speed_mps: Speed in meters per second (default is average walking speed)
    :return: Tuple of metrics (total_gain, total_loss, max_flood_depth, total_distance, travel_time)
    """
    total_gain = 0
    total_loss = 0
    max_flood_depth = 0
    total_distance = 0
    
    for i in range(len(path) - 1):
        node1 = path[i]
        node2 = path[i+1]
        
        # Get edge data between nodes
        edge_data = G.get_edge_data(node1, node2)
        
        # Extract elevations and handle potential None values
        elevations = edge_data.get('elevations', (0, 0))
        elevation1 = elevations[0] if elevations[0] is not None else 0
        elevation2 = elevations[1] if elevations[1] is not None else 0
        
        # Calculate elevation gain/loss
        elevation_diff = elevation2 - elevation1
        if elevation_diff > 0:
            total_gain += elevation_diff
        else:
            total_loss += abs(elevation_diff)
        
        # Handle flood depths
        flood_depths = edge_data.get('flood_depths', [0])
        for depth in flood_depths:
            if depth is not None:
                max_flood_depth = max(max_flood_depth, depth)
        
        # Calculate total distance
        total_distance += edge_data.get('distance', 0)
    
    # Calculate travel time based on total distance and speed
    travel_time = total_distance / speed_mps if speed_mps > 0 else float('inf')
    
    return total_gain, total_loss, max_flood_depth, total_distance, travel_time

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
        folium.PolyLine(locations=[(lat, lon) for lon, lat in path], color='red', weight=2.5, opacity=0.7).add_to(base_map)

    # Save the map to an HTML file
    base_map.save(output_html)
    print(f"Network and all paths visualized and saved to {output_html}")

# Main script
def main():
    geojson_file = 'roads_with_elevation_and_flood.geojson'
    output_html = 'aco_path_map.html'

    # Load GeoJSON and build graph
    geojson_data = load_geojson(geojson_file)
    G = build_graph(geojson_data)

    # Define start and end nodes (longitude, latitude)
    start_node = (125.6015325, 7.0647666) 
    end_node = (125.6024582, 7.0766550)

    initial_best_path = [(125.6015325, 7.0647666), (125.6015883, 7.0650632), (125.6018902, 7.0665857), (125.6019752, 7.0669411), (125.6020265, 7.0671295), (125.6021154, 7.0673791), (125.6021829, 7.0675542), (125.6022438, 7.0676806), (125.6023127, 7.0678194), (125.6023757, 7.0679197), (125.6024377, 7.0680005), (125.60252, 7.0680908), (125.6027502, 7.0682837), (125.6032125, 7.0686836), (125.6034827, 7.0689173), (125.6041372, 7.0694524), (125.6045038, 7.0697522), (125.6048012, 7.0699953), (125.604844, 7.0700303), (125.6048536, 7.0701051), (125.6048775, 7.0702905), (125.6049015, 7.0704764), (125.6049277, 7.0706797), (125.6049573, 7.0709099), (125.6050447, 7.0715901), (125.6050815, 7.0718762), (125.6051541, 7.0723902), (125.6051718, 7.0725155), (125.6051928, 7.0726641), (125.6051998, 7.0727191), (125.605227, 7.0730088), (125.6052502, 7.0732124), (125.6052062, 7.073266), (125.6049846, 7.073536), (125.6049202, 7.0736145), (125.6045904, 7.0740163), (125.6043522, 7.0743064), (125.6040902, 7.0746202), (125.6040195, 7.0747048), (125.6036306, 7.0751707), (125.6033365, 7.0755381), (125.6032835, 7.0756043), (125.6032472, 7.0756488), (125.6026112, 7.0764263), (125.6025067, 7.0765793), (125.6024582, 7.076655)]

    start_time = time.time()
    best_path, best_path_length, all_paths = ant_colony_optimization(G, start_node, end_node, initial_best_path=initial_best_path)
    end_time = time.time()

    # Display results
    if best_path:
        print(f"ACO completed in {end_time - start_time:.2f} seconds")
        # print(f"Best path: {best_path}")
        print(f"Best path length: {best_path_length:.2f} meters")

        total_gain, total_loss, max_flood_depth, total_distance, travel_time = calculate_metrics(
            best_path, 
            G, 
            speed_mps=1.5  # Average walking speed (can be adjusted)
        )
        
        print("\nPath Metrics:")
        print(f"Total Elevation Gain: {total_gain:.2f} meters")
        print(f"Total Elevation Loss: {total_loss:.2f} meters")
        print(f"Maximum Flood Depth: {max_flood_depth:.2f} meters")
        print(f"Total Path Distance: {total_distance:.2f} meters")
        print(f"Estimated Travel Time: {travel_time:.2f} seconds ({travel_time/60:.2f} minutes)")
    else:
        print("ACO failed to find a path between the start and end nodes.")

    # Call the visualization function
    visualize_paths(G, all_paths, start_node, end_node)

if __name__ == "__main__":
    main()