import json
import networkx as nx
from tqdm import tqdm
from geopy.distance import geodesic

def calculate_betweenness_centrality(G):
    print("Calculating betweenness centrality...")
    # Use NetworkX's built-in betweenness centrality calculation
    betweenness = nx.betweenness_centrality(G, weight='distance')
    
    # Invert the betweenness centrality values
    b_prime_values = {node: round(1 - value, 4) for node, value in betweenness.items()}
    return b_prime_values

def save_betweenness_to_json(betweenness_data, output_file):
    json_compatible_data = {str(key): value for key, value in betweenness_data.items()}
    
    with open(output_file, 'w') as f:
        json.dump(json_compatible_data, f, indent=4)
    print(f"Betweenness centrality data saved to {output_file}")

def main():
    # Wrap the entire script in a tqdm progress bar
    with tqdm(total=4, desc="Overall Script Progress") as pbar:
        # Step 1: Load GeoJSON file
        geojson_file = 'roads_with_elevation_and_flood.geojson'
        pbar.set_description("Loading GeoJSON")
        with open(geojson_file) as f:
            geojson_data = json.load(f)
        pbar.update(1)

        # Step 2: Create Network Graph
        pbar.set_description("Creating Network Graph")
        G = nx.Graph()
        for feature in geojson_data['features']:
            coordinates = feature['geometry']['coordinates']
            for i in range(len(coordinates) - 1):
                node1 = tuple(coordinates[i])
                node2 = tuple(coordinates[i + 1])
                dist = geodesic((coordinates[i][1], coordinates[i][0]),
                                (coordinates[i + 1][1], coordinates[i + 1][0])).meters
                G.add_edge(node1, node2, weight=dist, distance=dist)
        pbar.update(1)

        # Step 3: Calculate Betweenness Centrality
        pbar.set_description("Computing Betweenness Centrality")
        b_prime_data = calculate_betweenness_centrality(G)
        pbar.update(1)

        # Step 4: Save Results
        betweenness_output_file = 'betweenness_data2.json'
        pbar.set_description("Saving Results")
        save_betweenness_to_json(b_prime_data, betweenness_output_file)
        pbar.update(1)

    print("Script completed successfully!")

if __name__ == "__main__":
    main()