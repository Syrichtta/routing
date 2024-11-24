import json
import networkx as nx
from tqdm import tqdm
from geopy.distance import geodesic

def calculate_betweenness_centrality(G):
    betweenness = {}
    nodes = list(G.nodes)

    print("Calculating betweenness centrality...")
    for node in tqdm(nodes, desc="Processing nodes"):
        betweenness[node] = sum(
            nx.single_source_dijkstra_path_length(G, node, weight='distance').values()
        )

    # Normalize betweenness centrality
    max_b = max(betweenness.values())
    betweenness = {node: value / max_b for node, value in betweenness.items()}
    # print(betweenness)

    # Invert for `b_prime`
    b_prime_values = {node: round(1 - value, 4) for node, value in betweenness.items()}
    # print(b_prime_values)
    return b_prime_values

def save_betweenness_to_json(betweenness_data, output_file):
    # Convert tuple keys to strings for JSON compatibility
    json_compatible_data = {str(key): value for key, value in betweenness_data.items()}
    
    with open(output_file, 'w') as f:
        json.dump(json_compatible_data, f, indent=4)
    print(f"Betweenness centrality data saved to {output_file}")

# Example Usage
geojson_file = 'roads_with_elevation_and_flood.geojson'
# geojson_file = 'small_roads.geojson' #smaller geojson for testing
betweenness_output_file = 'betweenness_data.json'

# Load GeoJSON
with open(geojson_file) as f:
    geojson_data = json.load(f)

# Build the graph
G = nx.Graph()
for feature in geojson_data['features']:
    coordinates = feature['geometry']['coordinates']
    for i in range(len(coordinates) - 1):
        node1 = tuple(coordinates[i])
        node2 = tuple(coordinates[i + 1])
        dist = geodesic((coordinates[i][1], coordinates[i][0]),
                        (coordinates[i + 1][1], coordinates[i + 1][0])).meters
        G.add_edge(node1, node2, weight=dist, distance=dist)

# Calculate betweenness centrality
b_prime_data = calculate_betweenness_centrality(G)

# Save to JSON
save_betweenness_to_json(b_prime_data, betweenness_output_file)
