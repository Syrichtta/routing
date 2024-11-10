import json
import folium
import networkx as nx
from geopy.distance import geodesic
from pathlib import Path

# Build the graph from GeoJSON without elevations and flood depths
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

# Load the GeoJSON data
input_file = Path(__file__).parent.parent / "davao_bounding_box_road_network.geojson"
print(f"Loading data from {input_file}...")
with open(input_file) as f:
    geojson_data = json.load(f)
print("Data loaded successfully.")

# Build the graph
G = build_graph(geojson_data)

# Get the centroid for initial map focus
nodes = list(G.nodes)
center_node = nodes[len(nodes) // 2]  # Use a middle node for centering
map_center = [center_node[1], center_node[0]]

# Initialize the folium map
m = folium.Map(location=map_center, zoom_start=13)

# Plot each node with a popup showing its coordinates on click
for node in G.nodes:
    folium.CircleMarker(
        location=(node[1], node[0]),  # latitude, longitude
        radius=3,
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=0.6
    ).add_to(m).add_child(folium.Popup(f"Coordinates: ({node[0]:.8f}, {node[1]:.8f})"))

# Save the map as an HTML file
output_file = Path(__file__).parent / "davao_road_network_nodes_on_click.html"
m.save(output_file)
print(f"Map with nodes saved to {output_file}")
