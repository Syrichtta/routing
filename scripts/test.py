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
num_ants = 100
num_iterations = 10
alpha = 5.0        # Pheromone importance
beta = 1.0         # Heuristic importance
evaporation_rate = 0.5
pheromone_constant = 100.0
initial_path_weight = 2.0  # Multiplier for initial path pheromone levels

def load_geojson(file_path):
    with open(file_path) as f:
        return json.load(f)

def build_graph(geojson_data):
    G = nx.Graph()
    
    # Add progress bar for graph building
    features = tqdm(geojson_data['features'], desc="Building graph", unit="feature")
    
    for feature in features:
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

def initialize_pheromones_with_path(G, initial_path):
    """
    Initialize pheromone levels with higher values along a predefined path.
    """
    # Initialize all edges with base pheromone level
    pheromone_levels = {tuple(sorted(edge)): 1.0 for edge in G.edges()}
    
    if not initial_path:
        return pheromone_levels
    
    # Calculate the total length of the initial path
    total_length = 0
    for i in range(len(initial_path) - 1):
        node1 = initial_path[i]
        node2 = initial_path[i + 1]
        if G.has_edge(node1, node2):
            total_length += G[node1][node2]["distance"]
    
    # Set higher pheromone levels along the initial path
    if total_length > 0:
        initial_pheromone = (pheromone_constant / total_length) * initial_path_weight
        
        # Add progress bar for pheromone initialization
        path_edges = tqdm(range(len(initial_path) - 1), desc="Initializing pheromones", unit="edge")
        
        for i in path_edges:
            node1 = initial_path[i]
            node2 = initial_path[i + 1]
            if G.has_edge(node1, node2):
                edge = tuple(sorted((node1, node2)))
                pheromone_levels[edge] = initial_pheromone
                logging.info(f"Set initial pheromone level {initial_pheromone:.2f} for edge {edge}")
    
    return pheromone_levels

def ant_colony_optimization(G, start_node, end_node, initial_path=None):
    # Initialize pheromone levels with the initial path if provided
    print("Initializing ACO algorithm...")
    pheromone_levels = initialize_pheromones_with_path(G, initial_path)
    
    best_path = initial_path if initial_path else None
    best_path_length = float('inf')
    if initial_path:
        # Calculate initial path length
        best_path_length = sum(G[initial_path[i]][initial_path[i + 1]]["distance"] 
                             for i in range(len(initial_path) - 1))
        logging.info(f"Initial path length: {best_path_length:.2f}")
    
    path_found = False
    all_paths = []

    # Main iteration progress bar
    iteration_pbar = tqdm(range(num_iterations), desc="ACO Progress", position=0)
    
    for iteration in iteration_pbar:
        logging.info(f"Iteration {iteration + 1}/{num_iterations}")

        paths = []
        path_lengths = []

        # Ant progress bar
        ant_pbar = tqdm(range(num_ants), desc=f"Ants in iteration {iteration + 1}", 
                       position=1, leave=False)
        
        for ant in ant_pbar:
            current_node = start_node
            stack = [current_node]
            visited = set([current_node])
            path_length = 0

            while stack:
                current_node = stack[-1]

                if current_node == end_node:
                    path_found = True
                    path = list(stack)
                    path_length = sum(G[stack[i]][stack[i + 1]]["distance"] for i in range(len(stack) - 1))
                    logging.info(f"Ant {ant + 1} completed path: {path} with length: {path_length:.2f}")
                    
                    if path_length < best_path_length:
                        best_path = path
                        best_path_length = path_length
                        ant_pbar.set_postfix({"Best Length": f"{best_path_length:.2f}m"})
                        logging.info(f"New best path found by Ant {ant + 1} with length: {best_path_length:.2f}")

                    paths.append(path)
                    path_lengths.append(path_length)
                    break

                neighbors = list(G.neighbors(current_node))
                unvisited_neighbors = [n for n in neighbors if n not in visited]

                if unvisited_neighbors:
                    desirability = []
                    for neighbor in unvisited_neighbors:
                        distance = G[current_node][neighbor]["distance"]
                        edge = tuple(sorted((current_node, neighbor)))
                        pheromone = pheromone_levels.get(edge, 1.0)
                        desirability.append((pheromone ** alpha) * ((1.0 / distance) ** beta))

                    desirability_sum = sum(desirability)
                    probabilities = [d / desirability_sum for d in desirability]
                    next_node = random.choices(unvisited_neighbors, weights=probabilities)[0]

                    logging.info(f"Ant {ant + 1} moves from {current_node} to {next_node}.")

                    stack.append(next_node)
                    visited.add(next_node)
                    path_length += G[current_node][next_node]["distance"]
                else:
                    logging.info(f"Ant {ant + 1} stuck at node {current_node} (no unvisited neighbors). Backtracking...")
                    stack.pop()

            all_paths.append(stack)

        # Update main progress bar with best length so far
        iteration_pbar.set_postfix({"Best Path Length": f"{best_path_length:.2f}m"})
        
        # Pheromone evaporation and deposit with progress bar
        pheromone_update = tqdm(pheromone_levels.items(), desc="Updating pheromones", 
                               position=1, leave=False)
        
        for edge in pheromone_update:
            pheromone_levels[edge[0]] *= (1 - evaporation_rate)

        # Pheromone deposit
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

def visualize_paths(G, all_paths, start_node, end_node, initial_path=None, output_html='aco_paths_map.html'):
    print("Generating visualization...")
    base_map = folium.Map(location=[7.0866, 125.5782], zoom_start=14)

    # Add start and end markers
    folium.Marker(location=(start_node[1], start_node[0]), icon=folium.Icon(color='green', icon='info-sign')).add_to(base_map)
    folium.Marker(location=(end_node[1], end_node[0]), icon=folium.Icon(color='red', icon='info-sign')).add_to(base_map)

    # Visualize the entire network in blue
    network_edges = tqdm(G.edges(data=True), desc="Drawing network edges", unit="edge")
    for edge in network_edges:
        node1, node2, _ = edge
        folium.PolyLine(locations=[(lat, lon) for lon, lat in [node1, node2]], color='blue', weight=2.5, opacity=0.7).add_to(base_map)

    # Visualize initial path in green if provided
    if initial_path:
        folium.PolyLine(locations=[(lat, lon) for lon, lat in initial_path], color='green', weight=4, opacity=0.8).add_to(base_map)

    # Visualize all paths taken by the ants in red
    ant_paths = tqdm(all_paths, desc="Drawing ant paths", unit="path")
    for path in ant_paths:
        folium.PolyLine(locations=[(lat, lon) for lon, lat in path], color='red', weight=2.5, opacity=0.7).add_to(base_map)

    base_map.save(output_html)
    print(f"Network and all paths visualized and saved to {output_html}")

# Example usage
if __name__ == "__main__":
    print("Starting ACO pathfinding process...")
    
    geojson_file = 'davao_bounding_box_road_network.geojson'
    output_html = 'aco_path_map.html'

    # Load GeoJSON and build graph
    print("Loading GeoJSON data...")
    geojson_data = load_geojson(geojson_file)
    G = build_graph(geojson_data)

    # Define start and end nodes
    start_node = (125.5834749, 7.0454768)
    end_node = (125.6024582, 7.076655)

    # Define an initial path (example coordinates)
    initial_path = [
        (125.5834749, 7.0454768),
        (125.5832365, 7.0464268),
        (125.5829941, 7.0473761),
        (125.5827517, 7.0483254),
        (125.5831928, 7.048479),
        (125.5836157, 7.0486294),
        (125.5835652, 7.0488246),
        (125.5834549, 7.0492516),
        (125.5840222, 7.0494222),
        (125.584211, 7.0495074),
        (125.5842969, 7.0495926),
        (125.5843327, 7.0496819),
        (125.584349, 7.0497529),
        (125.584452, 7.0497546),
        (125.5845521, 7.049764),
        (125.5846713, 7.0497817),
        (125.5848223, 7.0498096),
        (125.5849233, 7.0498406),
        (125.5850644, 7.0498858),
        (125.5851793, 7.0499247),
        (125.5852714, 7.0499557),
        (125.5853695, 7.0499887),
        (125.5855005, 7.0500329),
        (125.585695, 7.0500974),
        (125.5858445, 7.050147),
        (125.5859704, 7.0501888),
        (125.5860579, 7.0502178),
        (125.5862065, 7.0502677),
        (125.5863781, 7.0503254),
        (125.5864924, 7.0503641),
        (125.5866708, 7.0504245),
        (125.5867891, 7.0504645),
        (125.5869379, 7.0505149),
        (125.5869871, 7.0505316),
        (125.587053, 7.0505539),
        (125.5879247, 7.0508526),
        (125.5882177, 7.0509583),
        (125.5884504, 7.0510536),
        (125.588547, 7.0510961),
        (125.5886523, 7.0511467),
        (125.5888608, 7.0512541),
        (125.5891547, 7.0514006),
        (125.5893038, 7.0514746),
        (125.589455, 7.0515469),
        (125.5896068, 7.0516194),
        (125.5896645, 7.0516448),
        (125.5897379, 7.0516737),
        (125.5898057, 7.0516971),
        (125.5898592, 7.0517139),
        (125.5899272, 7.0517326),
        (125.589989, 7.0517493),
        (125.5900687, 7.0517682),
        (125.5901332, 7.0517834),
        (125.5913904, 7.0520795),
        (125.5918497, 7.0521876),
        (125.5919285, 7.0522065),
        (125.5921685, 7.0522639),
        (125.5923369, 7.0523045),
        (125.5928499, 7.0524288),
        (125.5938208, 7.0526595),
        (125.5947364, 7.0528771),
        (125.5958109, 7.0531324),
        (125.5969417, 7.0534011),
        (125.5975337, 7.0535471),
        (125.5977751, 7.0536066),
        (125.5979275, 7.0536441),
        (125.5979878, 7.053659),
        (125.5980671, 7.0536786),
        (125.5982534, 7.0537244),
        (125.5984211, 7.0537658),
        (125.5984601, 7.0537781),
        (125.598508, 7.0537989),
        (125.5985462, 7.0538169),
        (125.5986146, 7.0538553),
        (125.5986735, 7.053892),
        (125.5987312, 7.0539333),
        (125.5987948, 7.0539833),
        (125.5988318, 7.0540161),
        (125.5988671, 7.0540505),
        (125.5989028, 7.0540854),
        (125.5990206, 7.0542128),
        (125.5991376, 7.0543944),
        (125.5992879, 7.0546737),
        (125.5993169, 7.0547275),
        (125.5993388, 7.0547683),
        (125.5993783, 7.0548417),
        (125.5995212, 7.0551072),
        (125.5998411, 7.0556888),
        (125.6000619, 7.0560896),
        (125.6002775, 7.0564737),
        (125.6003366, 7.0565716),
        (125.6003903, 7.056659),
        (125.60046, 7.0567606),
        (125.6005393, 7.056861),
        (125.6006582, 7.056998),
        (125.6007071, 7.0570461),
        (125.6007525, 7.0570853),
        (125.6008245, 7.0571416),
        (125.6008916, 7.0571908),
        (125.600962, 7.0572383),
        (125.6010431, 7.0572853),
        (125.6010965, 7.0573121),
        (125.6011596, 7.0573416),
        (125.6012288, 7.0573712),
        (125.6012235, 7.0574225),
        (125.6012185, 7.0574918),
        (125.6012152, 7.0575602),
        (125.6012129, 7.0576207),
        (125.6012084, 7.0578028),
        (125.6011998, 7.0580671),
        (125.6011818, 7.0584798),
        (125.6011899, 7.0585994),
        (125.6012022, 7.0586806),
        (125.60122, 7.0587632),
        (125.6012747, 7.0589456),
        (125.6012817, 7.0589655),
        (125.6014983, 7.0595808),
        (125.6015179, 7.0596447),
        (125.601545, 7.0597526),
        (125.6015617, 7.059862),
        (125.6015641, 7.0599993),
        (125.6015547, 7.0601401),
        (125.6014789, 7.0604628),
        (125.6012986, 7.0612308),
        (125.6012638, 7.061374),
        (125.601123, 7.0619494),
        (125.601101, 7.0620395),
        (125.601045, 7.0622803),
        (125.6009573, 7.062657),
        (125.6009376, 7.0627418),
        (125.6007886, 7.0633589),
        (125.600779, 7.0633982),
        (125.6007767, 7.0634072),
        (125.6007737, 7.0634194),
        (125.6007716, 7.0634283),
        (125.600769, 7.0634386),
        (125.6007618, 7.063468),
        (125.6007502, 7.063512),
        (125.6008209, 7.0635401),
        (125.6008181, 7.0635664),
        (125.6008206, 7.0635948),
        (125.6008313, 7.063621),
        (125.6008501, 7.0636465),
        (125.6008762, 7.0636672),
        (125.6009888, 7.063734),
        (125.6011122, 7.0638216),
        (125.6011854, 7.0638876),
        (125.6012563, 7.0639574),
        (125.6013108, 7.0640175),
        (125.6014334, 7.0641703),
        (125.6015374, 7.064314),
        (125.6016087, 7.0644438),
        (125.6016525, 7.0645537),
        (125.6016783, 7.0646184),
        (125.6017085, 7.0647208),
        (125.6017313, 7.0647981),
        (125.6017881, 7.0650242),
        (125.6020898, 7.0665457),
        (125.6021136, 7.06667),
        (125.6021149, 7.0667067),
        (125.6021116, 7.0667479),
        (125.6020973, 7.066801),
        (125.6021327, 7.0669692),
        (125.6022405, 7.0674101),
        (125.6022712, 7.0675112),
        (125.6023282, 7.0676556),
        (125.602406, 7.0678047),
        (125.6024717, 7.0679085),
        (125.6025354, 7.0679863),
        (125.6025897, 7.0680383),
        (125.6028179, 7.0682223),
        (125.6035767, 7.0688441),
        (125.6043963, 7.0695225),
        (125.6048775, 7.0699208),
        (125.6048569, 7.0699739),
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
        (125.6024582, 7.076655),
    ]

    print("Starting ACO algorithm...")
    start_time = time.time()
    best_path, best_path_length, all_paths = ant_colony_optimization(G, start_node, end_node, initial_path)
    end_time = time.time()

    if best_path:
        print(f"\nACO completed in {end_time - start_time:.2f} seconds")
        print(f"Best path: {best_path}")
        print(f"Best path length: {best_path_length:.2f} meters")
    else:
        print("\nACO failed to find a path between the start and end nodes.")

    # Visualize the results including the initial path
    visualize_paths(G, all_paths, start_node, end_node, initial_path)