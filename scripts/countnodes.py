import json
import networkx as nx
from geopy.distance import geodesic

# Load GeoJSON and build the graph
def load_geojson_and_build_graph(geojson_file):
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
    
    return G

# Count nodes in the graph
def count_nodes_in_graph(G):
    return G.number_of_nodes()

# Main function
if __name__ == "__main__":
    geojson_file = 'roads_with_elevation_and_flood.geojson'  # Path to your GeoJSON file
    
    # Load graph from GeoJSON
    G = load_geojson_and_build_graph(geojson_file)
    
    # Count the number of nodes in the graph
    node_count = count_nodes_in_graph(G)
    print(f"The graph has {node_count} nodes.")
