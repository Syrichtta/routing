import geopandas as gpd
import networkx as nx

# Load the road network GeoJSON as a GeoDataFrame
input_file = "davao_bounding_box_road_network.geojson"
print("Loading the road network GeoDataFrame from GeoJSON...")
gdf_edges = gpd.read_file(input_file)

# Initialize an empty graph
graph = nx.Graph()

# Add edges from GeoDataFrame to the graph
print("Adding edges to the graph...")
for _, row in gdf_edges.iterrows():
    u, v = row['u'], row['v']  # Assuming columns 'u' and 'v' represent the start and end nodes of each edge
    graph.add_edge(u, v, **row)  # Add edge with attributes

# Check connectivity of the graph
print("Checking connectivity of the road network...")
connected_components = list(nx.connected_components(graph))
num_components = len(connected_components)

# Display connectivity information
print(f"Total number of connected components: {num_components}")

# Identify and print stranded (disconnected) roads
if num_components > 1:
    print("The following components are disconnected:")
    for i, component in enumerate(connected_components):
        if len(component) < 10:  # Consider small components as stranded
            print(f"Component {i + 1} (size: {len(component)} nodes): {component}")
else:
    print("The road network is fully connected.")
