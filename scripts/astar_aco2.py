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
    # If an initial best path is provided, use it as a length constraint
    max_path_length = float('inf')
    if initial_best_path:
        initial_best_path_length = sum(G[initial_best_path[i]][initial_best_path[i + 1]]["distance"] 
                                       for i in range(len(initial_best_path) - 1))
        max_path_length = initial_best_path_length * 2  # Allow paths up to twice the initial path length

    pheromone_levels = {tuple(sorted(edge)): 1.0 for edge in G.edges()}
    best_path = None
    best_path_length = float('inf')
    path_found = False
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

# Main script
def main():
    geojson_file = 'roads_with_elevation_and_flood.geojson'
    output_html = 'aco_path_map.html'

    # Load GeoJSON and build graph
    geojson_data = load_geojson(geojson_file)
    G = build_graph(geojson_data)

    # Define start and end nodes (longitude, latitude)
    start_node = (125.6305739, 7.0927439)
    end_node = (125.5794607, 7.0664451)

    initial_best_path = [(125.6305739, 7.0927439), (125.6305829, 7.0926863), (125.6300046, 7.0926142), (125.6300166, 7.0925444), (125.6300334, 7.0924632), (125.6300545, 7.0923986), (125.6299858, 7.0923381), (125.6299583, 7.0923188), (125.6299281, 7.0923062), (125.6298892, 7.0922968), (125.6298128, 7.0922862), (125.6298191, 7.0922253), (125.6298205, 7.092204), (125.6298165, 7.0921877), (125.6298034, 7.0921621), (125.6296988, 7.0920826), (125.6290859, 7.0916487), (125.6290101, 7.0915868), (125.6286753, 7.0913117), (125.6283294, 7.0910235), (125.62795, 7.0907112), (125.6278171, 7.0905936), (125.6274703, 7.0902866), (125.6273391, 7.0901535), (125.6271608, 7.089917), (125.626696, 7.089337), (125.6268174, 7.0891687), (125.6271332, 7.0887914), (125.6273465, 7.0885412), (125.6270416, 7.0882821), (125.6266102, 7.0879529), (125.6265075, 7.0878713), (125.6261421, 7.0875765), (125.6242032, 7.0859534), (125.624218, 7.0858841), (125.6242322, 7.0858047), (125.6242458, 7.0857253), (125.6242586, 7.085641), (125.6242195, 7.0853481), (125.6242186, 7.085154), (125.6242206, 7.0851018), (125.6242252, 7.0850499), (125.6243312, 7.0838527), (125.6243653, 7.0834669), (125.6243886, 7.0832037), (125.6242688, 7.0831049), (125.6241694, 7.0830234), (125.6239106, 7.0828105), (125.6237361, 7.0827223), (125.6231086, 7.082212), (125.6230246, 7.0821436), (125.6226207, 7.0818103), (125.6219159, 7.0812111), (125.62186, 7.0811216), (125.6216719, 7.0809672), (125.6215495, 7.0808608), (125.6214261, 7.0807655), (125.6211268, 7.0805198), (125.6207, 7.0801693), (125.6194853, 7.0791698), (125.6194364, 7.0791279), (125.6183687, 7.0782359), (125.6182788, 7.0781596), (125.6178769, 7.0778188), (125.6175218, 7.0775347), (125.617467, 7.077493), (125.6174308, 7.0774532), (125.6174077, 7.077418), (125.6173991, 7.0773682), (125.6173978, 7.077326), (125.6173471, 7.0772962), (125.6173229, 7.0772852), (125.6173262, 7.077212), (125.6173355, 7.0770498), (125.6172507, 7.0769606), (125.6171692, 7.0768813), (125.6171691, 7.0767962), (125.6169917, 7.0767856), (125.6168389, 7.0767775), (125.6167435, 7.0767724), (125.6164674, 7.0767576), (125.6161408, 7.0767401), (125.6160492, 7.0767352), (125.6158301, 7.0767235), (125.6152282, 7.0766912), (125.6151816, 7.0766887), (125.6150403, 7.0766811), (125.6148481, 7.076671), (125.613557, 7.0766017), (125.6134517, 7.0765601), (125.6133857, 7.0765376), (125.6133392, 7.0765357), (125.6132918, 7.0765339), (125.6132038, 7.0765314), (125.6131548, 7.0765289), (125.6131036, 7.0765241), (125.613038, 7.0765108), (125.6129642, 7.07649), (125.6128869, 7.0764602), (125.6128422, 7.0764387), (125.6127862, 7.0764058), (125.6126619, 7.0763121), (125.6121345, 7.0758584), (125.6118072, 7.0755733), (125.6116907, 7.0754807), (125.6116351, 7.0754474), (125.6115728, 7.0754152), (125.6114837, 7.0753752), (125.6114303, 7.0753572), (125.6113737, 7.0753456), (125.6113099, 7.0753389), (125.6112533, 7.075333), (125.611161, 7.0753266), (125.6110723, 7.0753217), (125.6109218, 7.0753134), (125.6108891, 7.0753118), (125.6107778, 7.0753043), (125.6106557, 7.075284), (125.6104682, 7.0752426), (125.6103184, 7.0751908), (125.6101975, 7.075139), (125.6101656, 7.075122), (125.6100476, 7.0750594), (125.6099805, 7.0750146), (125.6099287, 7.07498), (125.6098341, 7.0749087), (125.609733, 7.074824), (125.6095793, 7.0746731), (125.6094526, 7.0745092), (125.6093608, 7.0743503), (125.6092833, 7.074188), (125.6092409, 7.0740553), (125.6091537, 7.0737629), (125.6090882, 7.0735781), (125.6090585, 7.0735019), (125.6090221, 7.0734291), (125.6089184, 7.0732594), (125.6088612, 7.0731958), (125.6085792, 7.072961), (125.6082892, 7.072724), (125.60715, 7.0717929), (125.6067826, 7.0714926), (125.6058699, 7.0707466), (125.6057684, 7.0706621), (125.6052888, 7.0702631), (125.6049903, 7.0700147), (125.6048775, 7.0699208), (125.6043963, 7.0695225), (125.6035767, 7.0688441), (125.6028179, 7.0682223), (125.6025897, 7.0680383), (125.6025354, 7.0679863), (125.6024717, 7.0679085), (125.602406, 7.0678047), (125.6023282, 7.0676556), (125.6022712, 7.0675112), (125.6022405, 7.0674101), (125.6021327, 7.0669692), (125.6020973, 7.066801), (125.6020454, 7.0665543), (125.6017438, 7.0650329), (125.6017076, 7.0648811), (125.6016679, 7.0647314), (125.601641, 7.0646479), (125.6016113, 7.0645667), (125.6015662, 7.0644678), (125.6014885, 7.0643339), (125.6014011, 7.0641936), (125.6013406, 7.0641141), (125.6012756, 7.064034), (125.6011788, 7.0639368), (125.6010945, 7.0638625), (125.6009912, 7.063781), (125.6009302, 7.0637375), (125.6008573, 7.0636934), (125.6007091, 7.0636148), (125.6006686, 7.0635969), (125.6005437, 7.0635499), (125.6003847, 7.0635062), (125.6002689, 7.0634779), (125.5996428, 7.0633241), (125.599459, 7.0632789), (125.5992023, 7.0632159), (125.5987314, 7.0631002), (125.5980219, 7.0629304), (125.5979825, 7.062921), (125.5968357, 7.0626466), (125.5967054, 7.0626154), (125.5963195, 7.0625231), (125.5957901, 7.0623964), (125.5956187, 7.0623553), (125.595538, 7.062336), (125.5950782, 7.062226), (125.5948098, 7.0621618), (125.5933404, 7.0618101), (125.5932656, 7.0617922), (125.5929402, 7.0617144), (125.5928223, 7.0616869), (125.5919994, 7.0614951), (125.591138, 7.0612942), (125.5909836, 7.061258), (125.5908615, 7.0612293), (125.5905919, 7.0611663), (125.5904419, 7.0611312), (125.5903595, 7.061112), (125.5896079, 7.0609374), (125.5892303, 7.0608495), (125.5891777, 7.0608373), (125.5888089, 7.0607495), (125.5880353, 7.0605646), (125.5879673, 7.0605486), (125.5877364, 7.0604941), (125.5875713, 7.0604551), (125.5873905, 7.0604125), (125.5869974, 7.0603182), (125.586266, 7.0601428), (125.5861127, 7.0601057), (125.5859567, 7.0600679), (125.5856537, 7.0599937), (125.5853698, 7.0599252), (125.5849635, 7.0598271), (125.584937, 7.0599356), (125.5840668, 7.0597297), (125.5836552, 7.0596323), (125.5835623, 7.0596197), (125.5834787, 7.0596116), (125.5832538, 7.0595968), (125.5826577, 7.0595575), (125.5824256, 7.0595346), (125.5821767, 7.0594993), (125.5821469, 7.059542), (125.5821265, 7.0595616), (125.5821026, 7.0595796), (125.5820754, 7.0595935), (125.5820486, 7.0596031), (125.5820083, 7.059617), (125.581951, 7.0596306), (125.5817527, 7.0597399), (125.5816477, 7.0597978), (125.5815667, 7.0598466), (125.5815068, 7.059893), (125.581463, 7.0599416), (125.581411, 7.0600107), (125.5812314, 7.0603139), (125.5812156, 7.0603827), (125.5811502, 7.0604887), (125.5810993, 7.0605284), (125.581014, 7.0606668), (125.5805891, 7.0613477), (125.5805406, 7.0614189), (125.5804836, 7.0614882), (125.5804172, 7.061552), (125.580462, 7.0616249), (125.5804757, 7.0616833), (125.5804822, 7.0617334), (125.5805041, 7.0618913), (125.5805243, 7.0619789), (125.5805355, 7.0620422), (125.580572, 7.0622167), (125.5806109, 7.0623088), (125.5806517, 7.0623763), (125.5807069, 7.0624532), (125.5807747, 7.0625234), (125.5808567, 7.0625947), (125.5809255, 7.0626434), (125.5809835, 7.0626832), (125.5813939, 7.0629633), (125.5811624, 7.0633352), (125.5803256, 7.0645944), (125.5800459, 7.0650195), (125.5799657, 7.064983), (125.5799113, 7.0649658), (125.5798589, 7.0649567), (125.5798117, 7.0649529), (125.5797252, 7.0649509), (125.57973, 7.0651281), (125.579734, 7.0654209), (125.5797273, 7.0654956), (125.5797131, 7.0655867), (125.579685, 7.0656892), (125.5794607, 7.0664451)]

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
        visualize_paths(G, best_path, all_paths, start_node, end_node)
    else:
        print("ACO failed to find a path between the start and end nodes.")

    # Call the visualization function

if __name__ == "__main__":
    main()