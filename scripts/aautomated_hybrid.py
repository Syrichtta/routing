import json
import networkx as nx
import random
import numpy as np
import time
import logging
from geopy.distance import geodesic
from tqdm import tqdm
import folium
import pandas as pd
import math

# build graph
def build_graph(geojson_file, betweenness_file):
    
    # Load betweenness data
    with open(betweenness_file) as f:
        betweenness_data = json.load(f)

    # load GeoJSON data
    with open(geojson_file) as f:
        geojson_data = json.load(f)
    
    G = nx.Graph()
    for feature in geojson_data['features']:
        coordinates = feature['geometry']['coordinates']
        elevations = feature['properties'].get('elevations', [0] * len(coordinates))
        flood_depths = feature['properties'].get('flood_depths', [0] * len(coordinates))
        
        for i in range(len(coordinates) - 1):
            node1 = tuple(coordinates[i])
            node2 = tuple(coordinates[i + 1])
            
            # Calculate distance between nodes
            dist = geodesic((coordinates[i][1], coordinates[i][0]), 
                            (coordinates[i + 1][1], coordinates[i + 1][0])).meters
            
            # Get b_prime values if they exist
            b_prime1 = betweenness_data.get(str(node1), 0)
            b_prime2 = betweenness_data.get(str(node2), 0)
            
            # Add edge with all attributes
            G.add_edge(node1, node2, 
                       weight=dist, 
                       distance=dist, 
                       elevations=(elevations[i], elevations[i + 1]), 
                       flood_depths=(flood_depths[i], flood_depths[i + 1]),
                       b_prime=(b_prime1, b_prime2))
    
    return G

# Function to calculate slope (from A* script)
def calculate_slope(node1, node2, G):
    if G.has_edge(node1, node2):
        edge_data = G[node1][node2]

        # Get elevations and handle missing data
        elevations = edge_data.get('elevations', [None, None])
        elevation1 = elevations[0] if elevations[0] is not None else 0
        elevation2 = elevations[1] if elevations[1] is not None else 0

        # Get horizontal distance and handle missing data
        horizontal_distance = edge_data.get('distance', 0)

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
def heuristic_extended(edge, end_node, G, alpha=0.2, beta=0.25, gamma=0.2, delta=0.1, epsilon=7):
    node1, node2 = edge
    # calculate the straight-line distance (h(n))
    h_n = geodesic((node2[1], node2[0]), (end_node[1], end_node[0])).meters
    # print(h_n)

    # get b'(n): inverse betweenness centrality
    b_prime = G.nodes[node1].get('b_prime', 0)

    distance, slope, flood_depth = 0, 0, 0

    # get edge data
    edge_data = G.get_edge_data(node1,node2)
    # print(edge_data)
    if edge_data:
        distance = edge_data.get('distance', 0)
        slope = calculate_slope(node1, node2, G)
        # Handle flood depths
        flood_depths = edge_data.get('flood_depths', [0])
        filtered_flood_depths = [fd for fd in flood_depths if fd is not None]
        flood_depth = max(filtered_flood_depths, default=0)
    # print(distance)
    f_n = (
        alpha * (h_n) + 
        beta * b_prime +
        gamma * distance +
        delta * slope +
        epsilon * flood_depth
    )
    return f_n

# ACO parameters
num_ants = 25
num_iterations = 30 
alpha = 1.0        
beta = 2.0         
evaporation_rate = 0.1
pheromone_constant = 100.0

# logging for ant behavior
logging.basicConfig(level=logging.INFO, filename="astar_aco_log.txt", filemode="w", format="%(message)s")

# ACO main function
def ant_colony_optimization(G, start_node, end_node, initial_best_path=None):
        
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
                        # reached intermediate target, now target end_node
                        current_target = end_node
                        continue

                neighbors = list(G.neighbors(current_node))
                unvisited_neighbors = [n for n in neighbors if n not in visited]

                if unvisited_neighbors:
                    random_choice = round(random.random(), 1)
                        
                    if random_choice > 0.5:
                        desirability = []
                        for neighbor in unvisited_neighbors:
                            heuristic_value = heuristic_extended(tuple((current_node, neighbor)), current_target, G)
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

        # pheromone update
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
        return None, float('inf'), []

    # sort completed paths
    completed_paths = []
    completed_paths = sorted(
        [(path, sum(G[path[i]][path[i + 1]]["distance"] for i in range(len(path) - 1))) 
         for path in all_paths 
         if path[-1] == end_node],
        key=lambda x: x[1]
    )

    return best_path, best_path_length, all_paths, [path for path, _ in completed_paths]

# get path metrics
def calculate_metrics(path, G, speed_mps):
    max_slope = 0
    max_flood_depth = 0
    total_distance = 0
    
    for i in range(len(path) - 1):
        node1 = path[i]
        node2 = path[i+1]
        
        edge_data = G.get_edge_data(node1, node2)
        
        # extract elevations and handle NaN as 0
        elevations = edge_data.get('elevations', (0, 0))
        elevation1 = elevations[0] if not math.isnan(elevations[0]) else 0
        elevation2 = elevations[1] if not math.isnan(elevations[1]) else 0

        horizontal_distance = edge_data.get('distance', 0)
        
        # calculate elevation increase
        elevation_diff = elevation2 - elevation1
        if horizontal_distance > 0:  # Avoid division by zero
            slope = elevation_diff / horizontal_distance
            max_slope = max(max_slope, slope)
        
        # handle flood depths and treat NaN as 0
        flood_depths = edge_data.get('flood_depths', [0])
        for depth in flood_depths:
            depth = depth if not math.isnan(depth) else 0
            max_flood_depth = max(max_flood_depth, depth)
        
        total_distance += edge_data.get('distance', 0)
    
    return max_slope, max_flood_depth, total_distance


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
    print("runnings tests")
    geojson_file = 'roads_with_elevation_and_flood3.geojson'
    betweeness_file = 'betweenness_data.json'
    output_excel = 'aco_hybrid_results.xlsx'

    # Define start nodes and waypoints
    start_nodes = [
        (125.5520585, 7.0848135),
        (125.6204559, 7.1063729),
        (125.6155148, 7.0774605),
        (125.5895257, 7.1093835),
        (125.5860874, 7.0585475),
        (125.5679739, 7.042197),
        (125.5860753, 7.0729699),
        (125.5992493, 7.063607),
        (125.589421, 7.0956251),
        (125.5685605, 7.0716337),
        (125.5652225, 7.1037087),
        (125.6260216, 7.0744122),
        (125.6008342, 7.074818),
        (125.6092216, 7.0556508),
        (125.6151783, 7.0950408),
        (125.6023844, 7.0647222),
        (125.6122887, 7.0957643),
        (125.5878275, 7.1066098),
        (125.6155135, 7.1129291),
        (125.5748979, 7.0867518),
        (125.6119228, 7.0901529),
        (125.6305879, 7.0888433),
        (125.588251, 7.0828854),
        (125.5802075, 7.0652194),
        (125.5758011, 7.1025389),
        (125.6136619, 7.0971005),
        (125.6112873, 7.1016861),
        (125.5904933, 7.0662942),
        (125.5952838, 7.0779905),
        (125.6119783, 7.090352),
        (125.6135774, 7.0603729),
        # (125.5750689, 7.0513503),
        # (125.6204416, 7.0717227),
        # (125.6197874, 7.1082214),
        # (125.595055, 7.0839762),
        # (125.6205352, 7.1117953),
        # (125.6290101, 7.0915868),
        # (125.6004868, 7.0501385),
        # (125.6066222, 7.105572),
        # (125.616509, 7.1008208),
        # (125.6308756, 7.0864334),
        # (125.5776109, 7.1005932),
        # (125.6311633, 7.1095096),
        # (125.5882688, 7.0546153),
        # (125.5882568, 7.0552328),
        # (125.6325084, 7.1120355),
        # (125.6102425, 7.0911402),
        # (125.6259356, 7.1040573),
        # (125.6111768, 7.0554749),
        # (125.6217104, 7.0684683),
        # (125.6236336, 7.1240124),
        # (125.602972, 7.101847),
        # (125.5635374, 7.0504705),
        # (125.5850548, 7.1076778),
        # (125.613681, 7.1067177),
        # (125.6110282, 7.0867481),
        # (125.6000619, 7.0560896),
        # (125.5813632, 7.1006017),
        # (125.6202836, 7.0783214),
        # (125.6008343, 7.0679565),
        # (125.6002622, 7.1132092),
        # (125.6155904, 7.0955002),
        # (125.5918958, 7.0550804),
        # (125.5968052, 7.048139),
        # (125.5979938, 7.11038),
        # (125.5751903, 7.0905102),
        # (125.6180152, 7.0656255),
        # (125.630134, 7.097913),
        # (125.6291087, 7.0990867),
        # (125.5762927, 7.053404),
        # (125.6202769, 7.1157497),
        # (125.6144223, 7.062505),
        # (125.5699834, 7.0638791),
        # (125.6217581, 7.0680991),
        # (125.6291965, 7.1104166),
        # (125.6129826, 7.1121067),
        # (125.6131144, 7.0785856),
        # (125.5999186, 7.1060495),
        # (125.5918126, 7.084462),
        # (125.6107244, 7.0500581),
        # (125.6038221, 7.0609319),
        # (125.6227351, 7.1058975),
        # (125.5612206, 7.1120168),
        # (125.5993987, 7.0606709),
        # (125.6289825, 7.1107528),
        # (125.6248637, 7.0793785),
        # (125.6096956, 7.1074647),
        # (125.5961796, 7.0712703),
        # (125.6132924, 7.0765137),
        # (125.6090221, 7.0734291),
    ]
    
    waypoints = [
        (125.5794607, 7.0664451),  # Shrine Hills
        (125.5657858, 7.1161489),  # Manila Memorial Park
        (125.6024582, 7.0766550)   # Rizal Memorial Colleges
    ]

    # Load GeoJSON and build graph
    print("building graph")
    G = build_graph(geojson_file, betweeness_file)
    print("graph built")

    # Results storage
    results = []

    # Iterate through each start node
    for start_node in start_nodes:
        print(f"Processing for start node: {start_node}")

        # Find the best end node based on shortest path
        shortest_distance = float('inf')
        initial_best_path = None
        end_node = None

        start_time = time.time()

        for waypoint in waypoints:
            try:
                # Use A* to find path to this waypoint
                path = nx.astar_path(
                    G, start_node, waypoint,
                    # heuristic=lambda n1, n2: heuristic_extended(n1, n2, G),
                    weight='weight'
                )
                # Calculate total distance for this path
                _, _, total_distance = calculate_metrics(path, G, speed_mps=1.14)

                # Update if this is the shortest path found
                if total_distance < shortest_distance:
                    shortest_distance = total_distance
                    initial_best_path = path
                    end_node = waypoint

            except nx.NetworkXNoPath:
                print(f"No path found between start node {start_node} and waypoint {waypoint}")
                continue

        if not end_node:
            print(f"No valid end node found for start node {start_node}")
            results.append({
                "Start Node": start_node,
                "End Node": None,
                "Error": "No path to any waypoint"
            })
            continue

        
        best_path, best_path_length, all_paths, completed_paths = ant_colony_optimization(
            G, 
            start_node, 
            end_node,
            initial_best_path=initial_best_path
        )
        end_time = time.time()

        max_slope, max_flood_depth, total_distance = calculate_metrics(
            best_path, 
            G, 
            speed_mps=1.14  # Average walking speed (can be adjusted)
        )

        # Add results
        results.append({
            "Start Node": start_node,
            "End Node": end_node,
            "Maximum Slope": max_slope,
            "Maximum Flood Depth": max_flood_depth,
            "Total Distance (m)": total_distance,
            # "Travel Time (s)": travel_time,
            "Paths": len(completed_paths),
            "Time": end_time - start_time
        })

    # Save results to an Excel file
    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"Results saved to {output_excel}")

if __name__ == "__main__":
    main()