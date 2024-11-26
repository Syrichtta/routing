import json
import math
import networkx as nx
import random
from geopy.distance import geodesic
import pandas as pd
from tqdm import tqdm
import folium
import time

# ACO parameters
NUM_ANTS = 25
NUM_ITERATIONS = 30
ALPHA = 1.0        # Pheromone importance
BETA = 2.0         # Distance importance
EVAPORATION_RATE = 0.1
PHEROMONE_CONSTANT = 100.0

# Load GeoJSON data
def load_geojson(file_path):
    with open(file_path) as f:
        return json.load(f)

def build_graph(geojson_data, betweenness_file):
    # Load betweenness data
    with open(betweenness_file) as f:
        betweenness_data = json.load(f)
    
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

def ant_colony_optimization(G, start_node, end_node):
    # Initialize pheromone levels on all edges
    pheromone_levels = {tuple(sorted(edge)): 1.0 for edge in G.edges()}
    
    best_path = None
    best_path_length = float('inf')
    max_path_length = float('inf')  # Can be set to a specific value if needed
    all_paths = []

    for iteration in range(NUM_ITERATIONS):
        paths = []
        path_lengths = []

        for ant in tqdm(range(NUM_ANTS), desc=f"Iteration {iteration + 1}/{NUM_ITERATIONS}"):
            current_node = start_node
            stack = [current_node]  # Stack for backtracking
            visited = set([current_node])
            path_length = 0

            while stack:
                current_node = stack[-1]  # Get current node from top of stack

                if current_node == end_node:
                    path = list(stack)
                    path_length = sum(G[stack[i]][stack[i + 1]]["distance"] 
                                    for i in range(len(stack) - 1))
                    
                    if path_length <= max_path_length:
                        paths.append(path)
                        path_lengths.append(path_length)
                        
                        if path_length < best_path_length:
                            best_path = path
                            best_path_length = path_length
                    break

                # Get unvisited neighbors
                neighbors = list(G.neighbors(current_node))
                unvisited_neighbors = [n for n in neighbors if n not in visited]

                if unvisited_neighbors:
                    # Calculate probabilities for each unvisited neighbor
                    probabilities = []
                    for neighbor in unvisited_neighbors:
                        edge = tuple(sorted((current_node, neighbor)))
                        distance = G[current_node][neighbor]['distance']
                        pheromone = pheromone_levels[edge]
                        
                        probability = (pheromone ** ALPHA) * ((1.0 / distance) ** BETA)
                        probabilities.append(probability)

                    # Normalize probabilities
                    total = sum(probabilities)
                    if total == 0:
                        stack.pop()  # Backtrack if no valid moves
                        continue
                    probabilities = [p/total for p in probabilities]

                    # Choose next node
                    next_node = random.choices(unvisited_neighbors, weights=probabilities)[0]
                    
                    # Check if adding this node would exceed max path length
                    potential_path_length = path_length + G[current_node][next_node]["distance"]
                    if potential_path_length <= max_path_length:
                        stack.append(next_node)
                        visited.add(next_node)
                        path_length = potential_path_length
                    else:
                        stack.pop()  # Backtrack if path would be too long
                else:
                    stack.pop()  # Backtrack if no unvisited neighbors

            all_paths.append(stack)  # Store the path taken by this ant

        # Evaporate pheromones on all edges
        for edge in pheromone_levels:
            pheromone_levels[edge] *= (1 - EVAPORATION_RATE)

        # Add new pheromones for completed paths
        for path, length in zip(paths, path_lengths):
            if length > 0:
                pheromone_deposit = PHEROMONE_CONSTANT / length
                for i in range(len(path) - 1):
                    edge = tuple(sorted((path[i], path[i + 1])))
                    pheromone_levels[edge] += pheromone_deposit

    return best_path, best_path_length, all_paths

# get path metrics
def calculate_metrics(path, G, speed_mps):
    max_elevation_increase = 0
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
        
        # calculate elevation increase
        elevation_diff = elevation2 - elevation1
        if elevation_diff > 0:
            max_elevation_increase = max(max_elevation_increase, elevation_diff)
        
        # handle flood depths and treat NaN as 0
        flood_depths = edge_data.get('flood_depths', [0])
        for depth in flood_depths:
            if depth is None or (isinstance(depth, float) and math.isnan(depth)):
                depth = 0
            max_flood_depth = max(max_flood_depth, depth)
        
        # calculate total distance
        total_distance += edge_data.get('distance', 0)
    
    travel_time = total_distance / speed_mps if speed_mps > 0 else float('inf')
    
    return max_elevation_increase, max_flood_depth, total_distance, travel_time


def visualize_paths(G, best_path, all_paths, start_node, end_node, output_html='aco_basemodel_paths_map.html'):
    # Create base map centered on the start point
    base_map = folium.Map(location=[start_node[1], start_node[0]], zoom_start=14)

    # Add start and end markers
    folium.Marker(
        location=(start_node[1], start_node[0]),
        icon=folium.Icon(color='green', icon='info-sign')
    ).add_to(base_map)
    folium.Marker(
        location=(end_node[1], end_node[0]),
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(base_map)

    # Draw network edges
    for edge in G.edges():
        node1, node2 = edge
        folium.PolyLine(
            locations=[(lat, lon) for lon, lat in [node1, node2]],
            color='blue',
            weight=2,
            opacity=0.5
        ).add_to(base_map)

    # Draw all ant paths
    for path in all_paths:
        if len(path) > 1:
            folium.PolyLine(
                locations=[(lat, lon) for lon, lat in path],
                color='red',
                weight=2,
                opacity=0.3
            ).add_to(base_map)

    # Draw best path if found
    if best_path:
        folium.PolyLine(
            locations=[(lat, lon) for lon, lat in best_path],
            color='green',
            weight=4,
            opacity=1.0
        ).add_to(base_map)

    base_map.save(output_html)

def main():
    # Load and prepare data
    geojson_file = 'roads_with_elevation_and_flood3.geojson'
    betweeness_file = 'betweenness_data.json'
    output_excel = 'aco_basemodel_results.xlsx'

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
    geojson_data = load_geojson(geojson_file)
    G = build_graph(geojson_data, betweeness_file)
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

        for waypoint in waypoints:
            try:
                # Use A* to find path to this waypoint
                path = nx.astar_path(
                    G, start_node, waypoint,
                    weight='weight'
                )
                # Calculate total distance for this path
                _, _, total_distance, _ = calculate_metrics(path, G, speed_mps=1.14)

                # Update if this is the shortest path found
                if total_distance < shortest_distance:
                    shortest_distance = total_distance
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

        start_time = time.time()
        best_path, best_path_length, all_paths, = ant_colony_optimization(
            G, 
            start_node, 
            end_node,
        )
        end_time = time.time()

        max_elevation_increase, max_flood_depth, total_distance, travel_time = calculate_metrics(
            best_path, 
            G, 
            speed_mps=1.4  # Average walking speed (can be adjusted)
        )

        # Add results
        results.append({
            "Start Node": start_node,
            "End Node": end_node,
            "Maximum Elevation Increase": max_elevation_increase,
            "Maximum Flood Depth": max_flood_depth,
            "Total Distance (m)": total_distance,
            # "Travel Time (s)": travel_time,
            "Time": end_time - start_time
        })

    # Save results to an Excel file
    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False)
    print(f"Results saved to {output_excel}")

if __name__ == "__main__":
    main()