import json
import networkx as nx
import random
from geopy.distance import geodesic
from tqdm import tqdm
import folium
import time

NUM_ANTS = 25
NUM_ITERATIONS = 50
ALPHA = 1.0 
BETA = 2.0    
EVAPORATION_RATE = 0.1
PHEROMONE_CONSTANT = 100.0

def load_geojson(file_path):
    with open(file_path) as f:
        return json.load(f)

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

def calculate_metrics(path, G, speed_mps=1.14):
    total_gain = 0
    total_loss = 0
    max_flood_depth = 0
    total_distance = 0
    
    for i in range(len(path) - 1):
        node1 = path[i]
        node2 = path[i+1]
        
        edge_data = G.get_edge_data(node1, node2)
        
        elevations = edge_data.get('elevations', (0, 0))
        elevation1 = elevations[0] if elevations[0] is not None else 0
        elevation2 = elevations[1] if elevations[1] is not None else 0
        
        elevation_diff = elevation2 - elevation1
        if elevation_diff > 0:
            total_gain += elevation_diff
        else:
            total_loss += abs(elevation_diff)
        
        flood_depths = edge_data.get('flood_depths', [0])
        for depth in flood_depths:
            if depth is not None:
                max_flood_depth = max(max_flood_depth, depth)
        
        total_distance += edge_data.get('distance', 0)
    
    travel_time = total_distance / speed_mps if speed_mps > 0 else float('inf')
    
    return total_gain, total_loss, max_flood_depth, total_distance, travel_time

def visualize_paths(G, best_path, all_paths, start_node, end_node, output_html='aco_basemodel_paths_map.html'):
    base_map = folium.Map(location=[start_node[1], start_node[0]], zoom_start=14)

    folium.Marker(
        location=(start_node[1], start_node[0]),
        icon=folium.Icon(color='green', icon='info-sign')
    ).add_to(base_map)
    folium.Marker(
        location=(end_node[1], end_node[0]),
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(base_map)

    for edge in G.edges():
        node1, node2 = edge
        folium.PolyLine(
            locations=[(lat, lon) for lon, lat in [node1, node2]],
            color='blue',
            weight=2,
            opacity=0.5
        ).add_to(base_map)

    for path in all_paths:
        if len(path) > 1:
            folium.PolyLine(
                locations=[(lat, lon) for lon, lat in path],
                color='red',
                weight=2,
                opacity=0.3
            ).add_to(base_map)

    if best_path:
        folium.PolyLine(
            locations=[(lat, lon) for lon, lat in best_path],
            color='green',
            weight=4,
            opacity=1.0
        ).add_to(base_map)

    base_map.save(output_html)

def main():
    geojson_file = 'roads_with_elevation_and_flood.geojson'
    geojson_data = load_geojson(geojson_file)
    G = build_graph(geojson_data)

    start_node = (125.6305739, 7.0927439)
    end_node = (125.5657858, 7.1161489)

    print("Running Ant Colony Optimization...")
    start_time = time.time()
    best_path, best_path_length, all_paths = ant_colony_optimization(G, start_node, end_node)
    end_time = time.time()
    print(f"ACO completed in {end_time - start_time:.2f} seconds")

    if best_path:
        
        total_gain, total_loss, max_flood_depth, total_distance, travel_time = calculate_metrics(
            best_path, 
            G, 
            speed_mps=1.114
        )
        
        
        print("\nPath Metrics:")
        print(f"Total Elevation Gain: {total_gain:.2f} meters")
        print(f"Total Elevation Loss: {total_loss:.2f} meters")
        print(f"Maximum Flood Depth: {max_flood_depth:.2f} meters")
        print(f"Total Path Distance: {total_distance:.2f} meters")
        print(f"Estimated Travel Time: {travel_time:.2f} seconds ({travel_time/60:.2f} minutes)")
        visualize_paths(G, best_path, all_paths, start_node, end_node)
        
        visualize_paths(G, best_path, all_paths, start_node, end_node)
        print("Results visualized and saved to 'aco_paths_map.html'")
    else:
        print("No path found between start and end points.")
        visualize_paths(G, None, all_paths, start_node, end_node)

if __name__ == "__main__":
    main()