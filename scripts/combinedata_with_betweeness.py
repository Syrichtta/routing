import json
import networkx as nx
from tqdm import tqdm
from geopy.distance import geodesic

def calculate_betweenness_centrality(G):
    # Use NetworkX's built-in betweenness centrality calculation
    betweenness = nx.betweenness_centrality(G, weight='distance')
    print(f"betweenness: {betweenness}")
    
    # Invert the betweenness centrality values
    b_prime_values = {node: round(1 - value, 4) for node, value in betweenness.items()}
    print(f"b_prime_values: {b_prime_values}")
    return b_prime_values

def save_betweenness_to_json(betweenness_data, output_file):
    json_compatible_data = {str(key): value for key, value in betweenness_data.items()}
    
    with open(output_file, 'w') as f:
        json.dump(json_compatible_data, f, indent=4)
    print(f"Betweenness centrality data saved to {output_file}")

geojson_file = 'roads_with_elevation_and_flood.geojson'
betweenness_output_file = 'betweenness_data2.json'

with open(geojson_file) as f:
    geojson_data = json.load(f)

G = nx.Graph()
for feature in geojson_data['features']:
    coordinates = feature['geometry']['coordinates']
    for i in range(len(coordinates) - 1):
        node1 = tuple(coordinates[i])
        node2 = tuple(coordinates[i + 1])
        dist = geodesic((coordinates[i][1], coordinates[i][0]),
                        (coordinates[i + 1][1], coordinates[i + 1][0])).meters
        G.add_edge(node1, node2, weight=dist, distance=dist)
print('graph built!')

b_prime_data = calculate_betweenness_centrality(G)

save_betweenness_to_json(b_prime_data, betweenness_output_file)
