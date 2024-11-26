import json
import random
import networkx as nx
from geopy.distance import geodesic

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

# Main script
if __name__ == "__main__":
    # Load the GeoJSON file
    geojson_data = load_geojson('roads_with_elevation_and_flood.geojson')
    
    # Build the graph
    G = build_graph(geojson_data)

    # Randomly select 100 nodes
    all_nodes = list(G.nodes)
    selected_nodes = random.sample(all_nodes, min(100, len(all_nodes)))  # Ensure we don't exceed the number of nodes

    # Save selected coordinates to a .txt file
    with open('selected_nodes.txt', 'w') as f:
        for coord in selected_nodes:
            f.write(f"({coord[0]}, {coord[1]}),\n")

    print(f"{len(selected_nodes)} random nodes selected and saved to selected_nodes.txt.")
