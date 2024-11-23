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
evaporation_rate = 0.1
pheromone_constant = 100.0

def is_within_radius(node1, node2, radius_meters):
    try:
        distance = geodesic(
            (node1[1], node1[0]),  # Convert to (lat, lon)
            (node2[1], node2[0])
        ).meters
        return distance <= radius_meters
    except:
        return False

def find_nodes_within_radius(G, center_node, radius_meters):
    nodes_within_radius = set()
    for node in G.nodes():
        if is_within_radius(node, center_node, radius_meters):
            nodes_within_radius.add(node)
    return nodes_within_radius

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
def heuristic_extended(node1, node2, G, alpha=1, beta=0.25, gamma=0.2, delta=0.1, epsilon=7):
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
        # delta * slope +
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

def ant_colony_optimization(G, start_node, end_node, forced_astar_points=None, initial_best_path=None):
    if forced_astar_points is None:
        forced_astar_points = {}
        
    # Initialize as before
    max_path_length = float('inf')
    if initial_best_path:
        initial_best_path_length = sum(G[initial_best_path[i]][initial_best_path[i + 1]]["distance"] 
                                     for i in range(len(initial_best_path) - 1))
        max_path_length = initial_best_path_length * 2

    pheromone_levels = {tuple(sorted(edge)): 1.0 for edge in G.edges()}
    
    protected_edges = set()
    if initial_best_path:
        initial_boost_factor = 10.0
        for i in range(len(initial_best_path) - 1):
            edge = tuple(sorted((initial_best_path[i], initial_best_path[i + 1])))
            pheromone_levels[edge] *= initial_boost_factor
            protected_edges.add(edge)
            logging.info(f"Protected edge {edge} with pheromone level {pheromone_levels[edge]}")

    best_path = None
    best_path_length = float('inf')
    path_found = False
    all_paths = []

    for iteration in range(num_iterations):
        logging.info(f"Iteration {iteration + 1}/{num_iterations}")
        paths = []
        path_lengths = []

        for ant in tqdm(range(num_ants), desc=f"Running ACO (Iteration {iteration + 1})", dynamic_ncols=False):
            current_node = start_node
            stack = [current_node]
            visited = set([current_node])
            path_length = 0
            force_astar = False
            current_target = end_node

            while stack:
                current_node = stack[-1]

                # Check if we've reached a forced A* transition point
                if current_node in forced_astar_points and not force_astar:
                    force_astar = True
                    current_target = forced_astar_points[current_node]
                    logging.info(f"Ant {ant + 1} reached transition point {current_node}. Switching to A* targeting {current_target}")
                    
                    try:
                        # Get A* path from current node to intermediate target
                        astar_path = nx.astar_path(G, current_node, current_target,
                                                 heuristic=lambda n1, n2: heuristic_extended(n1, n2, G),
                                                 weight='weight')
                        
                        # Remove current node to avoid duplication
                        astar_path = astar_path[1:]
                        
                        # Add A* path to stack and visited set
                        for node in astar_path:
                            if node not in visited:
                                stack.append(node)
                                visited.add(node)
                                if len(stack) > 1:
                                    path_length += G[stack[-2]][stack[-1]]["distance"]
                        
                        logging.info(f"A* path found and added to ant's path")
                        continue
                        
                    except nx.NetworkXNoPath:
                        logging.info(f"No A* path found from {current_node} to {current_target}")
                        break

                if current_node == current_target:
                    if current_target == end_node:
                        path = list(stack)
                        path_length = sum(G[stack[i]][stack[i + 1]]["distance"] for i in range(len(stack) - 1))
                        
                        if path_length <= max_path_length:
                            path_found = True
                            logging.info(f"Ant {ant + 1} completed path with length: {path_length:.2f}")
                            
                            if path_length < best_path_length:
                                best_path = path
                                best_path_length = path_length
                                logging.info(f"New best path found with length: {best_path_length:.2f}")

                            paths.append(path)
                            path_lengths.append(path_length)
                        break
                    else:
                        # We reached intermediate target, now target end_node
                        current_target = end_node
                        continue

                neighbors = list(G.neighbors(current_node))
                unvisited_neighbors = [n for n in neighbors if n not in visited]

                if unvisited_neighbors:
                    if force_astar:
                        # Use only A* heuristic when forced
                        desirability = []
                        for neighbor in unvisited_neighbors:
                            heuristic_value = heuristic_extended(neighbor, current_target, G)
                            desirability.append(heuristic_value)
                        
                        next_node = unvisited_neighbors[desirability.index(min(desirability))]
                        logging.info(f"Ant {ant + 1} uses forced A* to select {next_node}")
                    else:
                        # Normal ACO behavior
                        random_choice = round(random.random(), 1)
                        
                        if random_choice > 0.5:
                            desirability = []
                            for neighbor in unvisited_neighbors:
                                heuristic_value = heuristic_extended(neighbor, current_target, G)
                                desirability.append(heuristic_value)
                            
                            next_node = unvisited_neighbors[desirability.index(min(desirability))]
                            logging.info(f"Ant {ant + 1} uses A* heuristic to select {next_node}")
                        else:
                            desirability = []
                            for neighbor in unvisited_neighbors:
                                distance = G[current_node][neighbor]["distance"]
                                edge = tuple(sorted((current_node, neighbor)))
                                pheromone = pheromone_levels.get(edge, 1.0)
                                desirability.append((pheromone ** alpha) * ((1.0 / distance) ** beta))
                            
                            desirability_sum = sum(desirability)
                            probabilities = [d / desirability_sum for d in desirability]
                            next_node = random.choices(unvisited_neighbors, weights=probabilities)[0]
                            logging.info(f"Ant {ant + 1} uses ACO heuristic to select {next_node}")

                    potential_path_length = path_length + G[current_node][next_node]["distance"]
                    if potential_path_length <= max_path_length:
                        stack.append(next_node)
                        visited.add(next_node)
                        path_length = potential_path_length
                    else:
                        break
                else:
                    stack.pop()

            all_paths.append(stack)

        # Pheromone updates as before
        for edge in pheromone_levels:
            if edge not in protected_edges:
                pheromone_levels[edge] *= (1 - evaporation_rate)

        for path, length in zip(paths, path_lengths):
            if length > 0:
                pheromone_deposit = pheromone_constant / length
                for i in range(len(path) - 1):
                    edge = tuple(sorted((path[i], path[i + 1])))
                    pheromone_levels[edge] += pheromone_deposit

        logging.info(f"Pheromone levels after iteration {iteration + 1}: {pheromone_levels}")
        logging.info("")

    if not path_found:
        return None, float('inf'), all_paths

    return best_path, best_path_length, all_paths

def calculate_metrics(path, G, speed_mps=1.5):  # Default walking speed of 1.5 m/s
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
def visualize_paths(G, best_path, all_paths, start_node, end_node, output_html='aco_paths_map.html'):
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

    if best_path:
        folium.PolyLine(
            locations=[(lat, lon) for lon, lat in best_path], 
            color='green', 
            weight=4,  # Make it slightly thicker 
            opacity=1  # Full opacity
        ).add_to(base_map)

    # Save the map to an HTML file
    base_map.save(output_html)
    print(f"Network and all paths visualized and saved to {output_html}")

def main():
    geojson_file = 'roads_with_elevation_and_flood.geojson'
    output_html = 'aco_path_map.html'

    # Define waypoints
    waypoints = [
        # (125.5657858, 7.1161489),  # Manila Memorial Park
        (125.5794607, 7.0664451),  # Shrine Hills
        (125.6024582, 7.0766550)   # Rizal Memorial Colleges
    ]

    # Load GeoJSON and build graph
    geojson_data = load_geojson(geojson_file)
    G = build_graph(geojson_data)

    # Calculate paths between consecutive waypoints
    initial_best_path = []
    best_max_flood_depth = float('inf')
    best_start_index = 0

    # Try all possible starting points and calculate paths
    for start_index in range(len(waypoints)):
        current_path = []
        current_total_path = []
        current_max_flood_depth = 0

        # Create a path through the waypoints in order
        for i in range(start_index, start_index + len(waypoints)):
            start_node = waypoints[i % len(waypoints)]
            end_node = waypoints[(i + 1) % len(waypoints)]

            try:
                # Use A* to find the path between consecutive waypoints
                path = nx.astar_path(G, start_node, end_node, 
                                     heuristic=lambda n1, n2: heuristic_extended(n1, n2, G),
                                     weight='weight')
                
                # Calculate metrics for this path segment
                _, _, max_flood_depth, _, _ = calculate_metrics(path, G)
                current_max_flood_depth = max(current_max_flood_depth, max_flood_depth)
                
                # Extend the current total path 
                # If it's the first segment, add the whole path
                # If not, skip the first node to avoid duplicates
                current_total_path.extend(path[1:] if current_total_path else path)

            except nx.NetworkXNoPath:
                print(f"No path found between {start_node} and {end_node}")
                break

        # Check if this starting point gives the least max flood depth
        if current_max_flood_depth < best_max_flood_depth:
            best_max_flood_depth = current_max_flood_depth
            initial_best_path = current_total_path
            best_start_index = start_index

    # Set the start and end nodes based on the best path
    start_node = (125.6305739, 7.0927439)
    end_node = initial_best_path[-1]

    forced_astar_points = [
        (125.57402890, 7.08130460),
    ]

    print(f"Best initial path start index: {best_start_index}")
    print(f"Best max flood depth: {best_max_flood_depth}")
    print(f"Start node: {start_node}")
    print(f"End node: {end_node}")

    start_time = time.time()
    best_path, best_path_length, all_paths = ant_colony_optimization(
        G, 
        start_node, 
        end_node,
        forced_astar_points = forced_astar_points, 
        initial_best_path=initial_best_path
    )
    end_time = time.time()

    # Display results
    if best_path:
        print(f"ACO completed in {end_time - start_time:.2f} seconds")
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
        visualize_paths(G, best_path, all_paths, start_node, end_node)
    else:
        print("ACO failed to find a path between the start and end nodes.")
        visualize_paths(G, best_path, all_paths, start_node, end_node)

if __name__ == "__main__":
    main()