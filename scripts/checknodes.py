import json
import networkx as nx
from pathlib import Path
from geopy.distance import geodesic  # Added import for geodesic

# Build the graph from GeoJSON
def build_graph(geojson_data):
    G = nx.Graph()
    for feature in geojson_data['features']:
        coordinates = feature['geometry']['coordinates']

        for i in range(len(coordinates) - 1):
            # Define nodes and add to the graph
            node1 = tuple(coordinates[i])
            node2 = tuple(coordinates[i + 1])

            # Calculate distance between the two nodes as the edge weight
            dist = geodesic((coordinates[i][1], coordinates[i][0]), (coordinates[i + 1][1], coordinates[i + 1][0])).meters
            G.add_edge(node1, node2, weight=dist, distance=dist)

    return G

# Function to check if a node exists in the graph
def check_node_in_network(graph, nodes_to_check):
    nodes_in_network = {}
    for node in nodes_to_check:
        if node in graph.nodes:
            nodes_in_network[node] = True
        else:
            nodes_in_network[node] = False
    return nodes_in_network

# Load the GeoJSON data
input_file = Path(__file__).parent.parent / "davao_bounding_box_road_network.geojson"
print(f"Loading data from {input_file}...")
with open(input_file) as f:
    geojson_data = json.load(f)
print("Data loaded successfully.")

# Build the graph
G = build_graph(geojson_data)

# List of nodes to check
nodes_to_check = [
    (125.5657858, 7.1161489), # Manila Memorial Park
    (125.5794607, 7.0664451), # Shrine Hills
    (125.6024582, 7.0766550), # Rizal Memorial Colleges
]

# Check if the nodes are in the graph
nodes_status = check_node_in_network(G, nodes_to_check)

# Output the result
for node, status in nodes_status.items():
    print(f"Node {node} is in the network: {'Yes' if status else 'No'}")
