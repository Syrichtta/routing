import json
import networkx as nx
import random
import numpy as np
import time
import logging
from geopy.distance import geodesic
from tqdm import tqdm
import folium
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, filename="aco_log.txt", filemode="w", format="%(message)s")

# ACO parameters
num_ants = 5
num_iterations = 5
alpha = 1.0        # Pheromone importance
beta = 2.0         # Heuristic importance
evaporation_rate = 0.1
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
    # try:
    #     g_n = nx.shortest_path_length(G, source=node1, target=node2, weight='distance')
    # except nx.NetworkXNoPath:
    #     g_n = float('inf')  # Handle no-path case

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
        alpha * (h_n) +
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

import json
import networkx as nx
import random
import numpy as np
import time
import logging
from geopy.distance import geodesic
from tqdm import tqdm
import folium
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_ant(args):
    G, start_node, end_node, pheromone_levels, forced_astar_points, max_path_length, alpha, beta = args
    current_node = start_node
    stack = [current_node]
    visited = set([current_node])
    path_length = 0
    force_astar = False
    current_target = end_node
    
    # Pre-calculate distances and neighbors
    neighbors_cache = {node: list(G.neighbors(node)) for node in G.nodes()}
    distance_cache = {(n1, n2): G[n1][n2]["distance"] for n1, n2 in G.edges()}
    
    while stack:
        current_node = stack[-1]
        
        if current_node == current_target:
            if current_target == end_node:
                path = list(stack)
                path_length = sum(distance_cache[tuple(sorted((stack[i], stack[i + 1])))] 
                                for i in range(len(stack) - 1))
                
                if path_length <= max_path_length:
                    return path, path_length
                break
            else:
                current_target = end_node
                continue
        
        unvisited_neighbors = [n for n in neighbors_cache[current_node] if n not in visited]
        
        if unvisited_neighbors:
            random_choice = random.random()
            
            if random_choice > 0.5 or force_astar:
                # Use pre-calculated heuristic values
                desirability = [heuristic_extended(n, current_target, G) for n in unvisited_neighbors]
                next_node = unvisited_neighbors[np.argmin(desirability)]
            else:
                # Vectorized pheromone calculations
                edges = [tuple(sorted((current_node, n))) for n in unvisited_neighbors]
                distances = np.array([distance_cache[edge] for edge in edges])
                pheromones = np.array([pheromone_levels.get(edge, 1.0) for edge in edges])
                
                desirability = (pheromones ** alpha) * ((1.0 / distances) ** beta)
                probabilities = desirability / desirability.sum()
                next_node = np.random.choice(unvisited_neighbors, p=probabilities)
            
            potential_path_length = path_length + distance_cache[tuple(sorted((current_node, next_node)))]
            if potential_path_length <= max_path_length:
                stack.append(next_node)
                visited.add(next_node)
                path_length = potential_path_length
            else:
                break
        else:
            stack.pop()
    
    return None, float('inf')

def ant_colony_optimization(G, start_node, end_node, forced_astar_points=None, initial_best_path=None):
    if forced_astar_points is None:
        forced_astar_points = {}
    
    # Initialize parameters
    max_path_length = float('inf')
    if initial_best_path:
        initial_best_path_length = sum(G[initial_best_path[i]][initial_best_path[i + 1]]["distance"] 
                                     for i in range(len(initial_best_path) - 1))
        max_path_length = initial_best_path_length * 3
    
    # Pre-calculate edge tuples and initialize pheromone levels
    edges = list(G.edges())
    pheromone_levels = {tuple(sorted(edge)): 1.0 for edge in edges}

    protected_edges = set()
    if initial_best_path:
        initial_boost_factor = 10.0
        for i in range(len(initial_best_path) - 1):
            edge = tuple(sorted((initial_best_path[i], initial_best_path[i + 1])))
            pheromone_levels[edge] *= initial_boost_factor
            protected_edges.add(edge)
            logging.info(f"Protected edge {edge} with pheromone level {pheromone_levels[edge]}")

    
    # Boost initial path pheromones
    if initial_best_path:
        initial_boost_factor = 10.0
        for i in range(len(initial_best_path) - 1):
            edge = tuple(sorted((initial_best_path[i], initial_best_path[i + 1])))
            pheromone_levels[edge] *= initial_boost_factor
    
    best_path = None
    best_path_length = float('inf')
    all_paths = []
    
    # Create thread pool
    max_workers = min(num_ants, 8)  # Limit number of threads
    
    for iteration in range(num_iterations):
        ant_args = [(G, start_node, end_node, pheromone_levels.copy(), 
                    forced_astar_points, max_path_length, alpha, beta) 
                   for _ in range(num_ants)]
        
        paths = []
        path_lengths = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ant = {executor.submit(run_ant, args): i 
                           for i, args in enumerate(ant_args)}
            
            for future in as_completed(future_to_ant):
                path, length = future.result()
                if path:
                    paths.append(path)
                    path_lengths.append(length)
                    all_paths.append(path)
                    
                    if length < best_path_length:
                        best_path = path
                        best_path_length = length
        
        # Vectorized pheromone updates
        evaporation_mask = np.array([edge not in protected_edges for edge in edges])
        pheromone_array = np.array([pheromone_levels[tuple(sorted(edge))] for edge in edges])
        pheromone_array[evaporation_mask] *= (1 - evaporation_rate)
        
        # Update pheromones for successful paths
        for path, length in zip(paths, path_lengths):
            if length > 0:
                pheromone_deposit = pheromone_constant / length
                for i in range(len(path) - 1):
                    edge = tuple(sorted((path[i], path[i + 1])))
                    edge_idx = edges.index(edge)
                    pheromone_array[edge_idx] += pheromone_deposit
        
        # Update pheromone levels dictionary
        for edge, pheromone in zip(edges, pheromone_array):
            pheromone_levels[tuple(sorted(edge))] = pheromone
    
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

    # Fixed start point
    start_node = (125.6305739, 7.0927439)
    
    # Potential end points
    waypoints = [
        # (125.5794607, 7.0664451),  # Shrine Hills
        (125.5657858, 7.1161489) # Manila Memorial Park
        # (125.6024582, 7.0766550)   # Rizal Memorial Colleges
    ]

    # Load GeoJSON and build graph
    geojson_data = load_geojson(geojson_file)
    G = build_graph(geojson_data)

    # Find shortest path to each potential end point
    shortest_distance = float('inf')
    initial_best_path = None
    end_node = None

    for waypoint in waypoints:
        try:
            # Use A* to find path to this waypoint
            path = nx.astar_path(G, start_node, waypoint,
                               heuristic=lambda n1, n2: heuristic_extended(n1, n2, G),
                               weight='weight')
            
            # Calculate total distance for this path
            _, _, _, total_distance, _ = calculate_metrics(path, G)
            
            # Update if this is the shortest path found
            if total_distance < shortest_distance:
                shortest_distance = total_distance
                initial_best_path = path
                end_node = waypoint
                
        except nx.NetworkXNoPath:
            print(f"No path found between start node and {waypoint}")
            continue

    if not end_node:
        print("Could not find a path to any of the waypoints")
        return

    print(f"Selected end node: {end_node}")
    initial_gain, initial_loss, initial_flood, initial_distance, initial_time = calculate_metrics(
        initial_best_path,
        G,
        speed_mps=1.5
    )
    
    print("\nInitial Best Path Metrics:")
    print(f"Selected end node: {end_node}")
    print(f"Total Elevation Gain: {initial_gain:.2f} meters")
    print(f"Total Elevation Loss: {initial_loss:.2f} meters")
    print(f"Maximum Flood Depth: {initial_flood:.2f} meters")
    print(f"Total Path Distance: {initial_distance:.2f} meters")
    print(f"Estimated Travel Time: {initial_time:.2f} seconds ({initial_time/60:.2f} minutes)")

    start_time = time.time()
    best_path, best_path_length, all_paths = ant_colony_optimization(
        G, 
        start_node, 
        end_node,
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