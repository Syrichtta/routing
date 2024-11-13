import json
import networkx as nx
import random
import numpy as np
import time
import logging
import csv
import folium
from geopy.distance import geodesic

# Configure logging
logging.basicConfig(level=logging.INFO, filename="aco_log.txt", filemode="w", format="%(message)s")

# ACO parameters
num_ants = 100
num_iterations = 1
alpha = 5.0        
beta = 1.0         
evaporation_rate = 0.5
pheromone_constant = 100.0

# Load GeoJSON data
def load_geojson(file_path):
    with open(file_path) as f:
        return json.load(f)

# Load initial best path from CSV
def load_initial_best_path(csv_file):
    path = []
    with open(csv_file, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            lon, lat = map(float, row)
            path.append((lon, lat))
    return path

# Build the graph from GeoJSON (without elevation and flood depth)
def build_graph(geojson_data):
    G = nx.Graph()
    for feature in geojson_data['features']:
        coordinates = feature['geometry']['coordinates']
        
        for i in range(len(coordinates) - 1):
            node1 = tuple(coordinates[i])
            node2 = tuple(coordinates[i + 1])
            dist = geodesic((coordinates[i][1], coordinates[i][0]), (coordinates[i + 1][1], coordinates[i + 1][0])).meters
            G.add_edge(node1, node2, weight=dist, distance=dist)

    return G

# Ant Colony Optimization
def ant_colony_optimization(G, start_node, end_node, pheromone_levels):
    best_path = None
    best_path_length = float('inf')
    all_paths = []

    for _ in range(num_iterations):
        # Initialize ant paths
        ant_paths = []
        for _ in range(num_ants):
            path = [start_node]
            current_node = start_node

            while current_node != end_node:
                neighbors = list(G.neighbors(current_node))
                probabilities = []

                # Calculate transition probabilities for each neighbor
                for neighbor in neighbors:
                    pheromone = pheromone_levels.get(tuple(sorted([current_node, neighbor])), 1.0)
                    distance = G[current_node][neighbor]['distance']
                    probability = (pheromone ** alpha) * ((1 / distance) ** beta)
                    probabilities.append(probability)

                # Normalize probabilities
                total_prob = sum(probabilities)
                probabilities = [p / total_prob for p in probabilities]

                # Choose next node based on probabilities
                next_node = random.choices(neighbors, probabilities)[0]
                path.append(next_node)
                current_node = next_node

            path_length = sum(G[path[i]][path[i + 1]]['distance'] for i in range(len(path) - 1))
            ant_paths.append((path, path_length))

            # Update the best path if necessary
            if path_length < best_path_length:
                best_path = path
                best_path_length = path_length

        # Update pheromone levels based on ant paths
        pheromone_levels = update_pheromones(pheromone_levels, ant_paths, best_path)

        # Log the best path and its length
        logging.info(f"Best path: {best_path}, Length: {best_path_length}")

    return best_path, best_path_length, ant_paths

# Update pheromones after each iteration
def update_pheromones(pheromone_levels, ant_paths, best_path):
    for path, _ in ant_paths:
        for i in range(len(path) - 1):
            edge = tuple(sorted([path[i], path[i + 1]]))
            pheromone_levels[edge] = pheromone_levels.get(edge, 1.0) * (1 - evaporation_rate) + pheromone_constant

    # Intensify pheromone on the best path
    for i in range(len(best_path) - 1):
        edge = tuple(sorted([best_path[i], best_path[i + 1]]))
        pheromone_levels[edge] = pheromone_levels.get(edge, 1.0) + pheromone_constant

    return pheromone_levels

# Visualize paths on a map
def visualize_paths(G, all_paths, start_node, end_node, output_html):
    m = folium.Map(location=start_node, zoom_start=15)

    # Add start and end markers
    folium.Marker(start_node, popup="Start", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(end_node, popup="End", icon=folium.Icon(color='red')).add_to(m)

    # Add ant paths
    for path, _ in all_paths:
        path_coords = [(node[1], node[0]) for node in path]
        folium.PolyLine(path_coords, color="blue", weight=2.5, opacity=1).add_to(m)

    m.save(output_html)

if __name__ == "__main__":
    geojson_file = 'roads_with_elevation.geojson'  # Adjust the filename as needed
    csv_file = 'shortest_paths.csv'               # Path to your CSV file with paths
    start_node = (125.6030712, 7.0692329)         # Example start coordinates
    end_node = (125.5657858, 7.1161489)           # Example end coordinates
    output_html = 'astaraco_paths_map.html'

    # Load the geojson data
    geojson_data = load_geojson(geojson_file)
    
    # Build graph from the geojson data (no elevation or flood depth included)
    G = build_graph(geojson_data)

    # Initialize pheromone levels
    pheromone_levels = {tuple(sorted(edge)): 1.0 for edge in G.edges()}

    # Load initial best path from CSV
    initial_best_path = load_initial_best_path(csv_file)
    if initial_best_path:
        # If initial best path is provided, update pheromones
        update_pheromones(pheromone_levels, [(initial_best_path, 0)], initial_best_path)

    # Run Ant Colony Optimization
    best_path, best_path_length, all_paths = ant_colony_optimization(G, start_node, end_node, pheromone_levels)

    # Visualize paths using Folium
    visualize_paths(G, all_paths, start_node, end_node, output_html)
