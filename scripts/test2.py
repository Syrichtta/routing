import json
import networkx as nx
import folium
import random
from geopy.distance import geodesic
import time
import numpy as np
from tqdm import tqdm

# Load functions from the original script
def load_geojson(file_path):
    with open(file_path) as f:
        return json.load(f)
    
def load_betweenness_from_json(json_file):
    with open(json_file) as f:
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

def update_betweenness_from_json(G, betweenness_json):
    for node, b_prime in betweenness_json.items():
        node_coordinates = eval(node)
        if node_coordinates in G.nodes:
            G.nodes[node_coordinates]['b_prime'] = round(b_prime, 4)

def calculate_slope(node1, node2, G):
    if G.has_edge(node1, node2):
        edge_data = G[node1][node2]
        elevation1 = edge_data['elevations'][0]
        elevation2 = edge_data['elevations'][1]
        horizontal_distance = edge_data['distance']

        elevation_change = elevation2 - elevation1
        slope = elevation_change / horizontal_distance if horizontal_distance > 0 else 0

        return slope
    else:
        return 0

def heuristic_extended(node1, node2, G, alpha, beta, gamma, delta, epsilon):
    # Calculate the straight-line distance (h(n))
    h_n = geodesic((node1[1], node1[0]), (node2[1], node2[0])).meters

    # Get g(n): cost from start node to current node
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

    # Compute total cost 
    f_n = (
        alpha * (g_n + h_n) +
        beta * b_prime +
        gamma * distance +
        delta * slope +
        epsilon * flood_depth
    )
    return f_n

class AntColonyOptimization:
    def __init__(self, graph, start_node, end_node, num_ants=10, num_iterations=100, 
                 alpha=1.0, beta=5.0, evaporation_rate=0.5, q0=0.5):
        self.graph = graph
        self.start_node = start_node
        self.end_node = end_node
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha  # pheromone importance
        self.beta = beta    # heuristic information importance
        self.evaporation_rate = evaporation_rate
        self.q0 = q0        # exploration probability
        
        # Initialize pheromone trails
        self.pheromone = {}
        for edge in self.graph.edges():
            self.pheromone[edge] = 1.0
    
    def choose_next_node(self, current_node, visited_nodes):
        # Get unvisited neighbors
        neighbors = [n for n in self.graph.neighbors(current_node) if n not in visited_nodes]
        
        if not neighbors:
            return None
        
        # Randomly decide whether to use A* heuristic or standard ACO
        use_heuristic = random.random() > 0.5
        
        if use_heuristic:
            # Use A* inspired heuristic selection
            next_node = min(neighbors, key=lambda n: heuristic_extended(
                current_node, n, self.graph, 
                alpha=0.2, beta=0.25, gamma=0.2, delta=0.1, epsilon=7
            ))
        else:
            # Standard ACO node selection with probability
            probabilities = []
            for node in neighbors:
                edge = (current_node, node)
                # Compute probability based on pheromone and distance
                pheromone = self.pheromone.get(edge, 1.0)
                distance = self.graph[current_node][node]['distance']
                
                prob = (pheromone ** self.alpha) * ((1.0 / distance) ** self.beta)
                probabilities.append(prob)
            
            # Normalize probabilities
            total = sum(probabilities)
            probabilities = [p / total for p in probabilities]
            
            # Select node based on probabilities
            next_node = neighbors[np.random.choice(len(neighbors), p=probabilities)]
        
        return next_node
    
    def construct_solution(self):
        current_node = self.start_node
        solution = [current_node]
        visited_nodes = set([current_node])
        
        while current_node != self.end_node:
            next_node = self.choose_next_node(current_node, visited_nodes)
            
            if next_node is None:
                break
            
            solution.append(next_node)
            visited_nodes.add(next_node)
            current_node = next_node
        
        return solution if solution[-1] == self.end_node else None
    
    def update_pheromones(self, solutions):
        # Evaporate existing pheromones
        for edge in self.pheromone:
            self.pheromone[edge] *= (1 - self.evaporation_rate)
        
        # Update pheromones based on solutions
        for solution in solutions:
            if solution:
                # Compute solution quality (inverse of total distance)
                total_distance = sum(self.graph[solution[i]][solution[i+1]]['distance'] 
                                     for i in range(len(solution)-1))
                pheromone_deposit = 1.0 / total_distance
                
                # Update pheromones along the path
                for i in range(len(solution) - 1):
                    edge = (solution[i], solution[i+1])
                    self.pheromone[edge] += pheromone_deposit
    
    def run(self):
        best_solution = None
        best_distance = float('inf')
        
        for _ in range(self.num_iterations):
            # Generate solutions for all ants
            solutions = [self.construct_solution() for _ in range(self.num_ants)]
            
            # Update pheromones
            self.update_pheromones(solutions)
            
            # Find best solution
            for solution in solutions:
                if solution and solution[-1] == self.end_node:
                    total_distance = sum(self.graph[solution[i]][solution[i+1]]['distance'] 
                                         for i in range(len(solution)-1))
                    if total_distance < best_distance:
                        best_solution = solution
                        best_distance = total_distance
        
        return best_solution

def main():
    # Load data
    geojson_file = 'roads_with_elevation_and_flood.geojson'
    betweenness_path = 'betweenness_data.json'
    output_html = 'ant_colony_shortest_path_map.html'

    # Load the GeoJSON and build the graph
    geojson_data = load_geojson(geojson_file)
    G = build_graph(geojson_data)  

    # Load betweenness from JSON
    betweenness_json = load_betweenness_from_json(betweenness_path)

    # Update the graph nodes with b_prime values
    update_betweenness_from_json(G, betweenness_json)

    # Set start and end nodes (you can modify these as needed)
    start_node = (125.6305739, 7.0927439)
    end_node = (125.6024582, 7.0766550)

    # Initialize and run Ant Colony Optimization
    aco = AntColonyOptimization(G, start_node, end_node)
    best_path = aco.run()

    if best_path:
        print("Best path found:")
        for node in best_path:
            print(node)
        
        # Optional: Visualize the path (you can reuse the visualization function from the original script)
        # You'll need to import the visualization function from the original script or implement it here
    else:
        print("No path found.")

if __name__ == "__main__":
    main()