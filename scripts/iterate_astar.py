import json
import networkx as nx
import folium
from geopy.distance import geodesic
import time
from tqdm import tqdm  # For progress bar
import os

def load_geojson(file_path):
    with open(file_path) as f:
        return json.load(f)

def save_geojson(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)

def build_graph(geojson_data):
    G = nx.Graph()
    
    for feature in geojson_data['features']:
        coordinates = feature['geometry']['coordinates']
        elevations = feature['properties'].get('elevations', [0] * len(coordinates))
        
        for i in range(len(coordinates) - 1):
            node1 = tuple(coordinates[i])
            node2 = tuple(coordinates[i+1])
            
            dist = geodesic((coordinates[i][1], coordinates[i][0]), 
                          (coordinates[i+1][1], coordinates[i+1][0])).meters
            
            G.add_edge(node1, node2, weight=dist, distance=dist, 
                      elevations=(elevations[i], elevations[i+1]))
    
    return G

def calculate_metrics(path, G):
    if not path:
        return None
    
    total_gain = 0
    total_loss = 0
    total_distance = 0
    
    for i in range(len(path) - 1):
        node1 = path[i]
        node2 = path[i+1]
        edge_data = G.get_edge_data(node1, node2)
        elevations = edge_data['elevations']
        
        elevation1 = elevations[0] if elevations[0] is not None else 0
        elevation2 = elevations[1] if elevations[1] is not None else 0
        
        elevation_diff = elevation2 - elevation1
        if elevation_diff > 0:
            total_gain += elevation_diff
        else:
            total_loss += abs(elevation_diff)
            
        total_distance += edge_data['distance']
    
    return {
        'total_distance': round(total_distance, 2),
        'elevation_gain': round(total_gain, 2),
        'elevation_loss': round(total_loss, 2)
    }

def calculate_slope(node1, node2, G):
    if G.has_edge(node1, node2):
        edge_data = G[node1][node2]
        elevation1 = edge_data['elevations'][0]
        elevation2 = edge_data['elevations'][1]
        horizontal_distance = edge_data['distance']
        
        elevation_change = elevation2 - elevation1
        
        if horizontal_distance > 0:
            slope = elevation_change / horizontal_distance
        else:
            slope = 0
            
        return slope
    return 0

def heuristic_extended(node1, node2, G, alpha, beta, gamma):
    h_n = geodesic((node1[1], node1[0]), (node2[1], node2[0])).meters
    
    distance = 0
    slope = 0
    
    if G.has_edge(node1, node2):
        distance = G[node1][node2]['distance']
        slope = calculate_slope(node1, node2, G)
    
    f_n = (alpha * (distance + h_n) +
           beta * (distance) +
           gamma * (slope))
    
    return f_n

def compute_and_store_paths(geojson_data, end_nodes):
    """
    Compute shortest paths from all nodes to specified end nodes and store in GeoJSON.
    
    Args:
    geojson_data: The original GeoJSON data
    end_nodes: List of destination nodes to compute paths to
    """
    # Build the graph
    G = build_graph(geojson_data)
    
    # Get all unique nodes from the graph
    all_nodes = list(G.nodes())
    
    # Dictionary to store paths and metrics for each node
    paths_store = {}
    
    # Create a progress bar for the outer loop
    print(f"Computing paths for {len(all_nodes)} nodes to {len(end_nodes)} destinations...")
    
    # Iterate through all nodes
    for start_node in tqdm(all_nodes):
        node_paths = {}
        
        # Compute paths to each end node
        for end_node in end_nodes:
            if start_node != end_node:  # Skip if start and end are the same
                try:
                    # Compute shortest path
                    path = nx.astar_path(G, source=start_node, target=end_node,
                                       heuristic=lambda n1, n2: heuristic_extended(n1, n2, G, 0.3, 0.35, 0.35))
                    
                    # Calculate metrics for the path
                    metrics = calculate_metrics(path, G)
                    
                    # Store path and metrics
                    node_paths[str(end_node)] = {
                        'path': [list(node) for node in path],  # Convert tuples to lists for JSON serialization
                        'metrics': metrics
                    }
                except nx.NetworkXNoPath:
                    # Store None if no path exists
                    node_paths[str(end_node)] = None
        
        # Store all paths for this node
        paths_store[str(start_node)] = node_paths
    
    # Add the paths data to the GeoJSON features
    for feature in geojson_data['features']:
        coordinates = feature['geometry']['coordinates']
        for coord in coordinates:
            coord_tuple = tuple(coord)
            if str(coord_tuple) in paths_store:
                if 'properties' not in feature:
                    feature['properties'] = {}
                if 'precomputed_paths' not in feature['properties']:
                    feature['properties']['precomputed_paths'] = {}
                feature['properties']['precomputed_paths'][str(coord_tuple)] = paths_store[str(coord_tuple)]
    
    return geojson_data

def main():
    # Get the parent directory path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Input and output files with parent directory path
    input_geojson = os.path.join(parent_dir, 'davao_bounding_box_road_network.geojson')
    output_geojson = os.path.join(parent_dir, 'davao_road_network_with_paths.geojson')
    
    # Define end nodes (important destinations)
    end_nodes = [
        (125.6024582, 7.0766550),  # Rizal Memorial Colleges
        (125.5657858, 7.1161489),  # Manila Memorial Park
        (125.5794607, 7.0664451)   # Shrine Hills
        # Add more destination nodes as needed
    ]
    
    # Load the GeoJSON
    print(f"Loading GeoJSON data from {input_geojson}...")
    geojson_data = load_geojson(input_geojson)
    
    # Start timing
    start_time = time.time()
    
    # Compute and store paths
    print("Computing and storing paths...")
    updated_geojson = compute_and_store_paths(geojson_data, end_nodes)
    
    # End timing
    end_time = time.time()
    computation_time = end_time - start_time
    
    # Save the updated GeoJSON
    print(f"Saving updated GeoJSON to {output_geojson}...")
    save_geojson(updated_geojson, output_geojson)
    
    print(f"Processing completed in {computation_time:.2f} seconds")
    print(f"Updated GeoJSON saved to {output_geojson}")

if __name__ == "__main__":
    main()