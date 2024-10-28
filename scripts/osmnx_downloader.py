import osmnx as ox
import geopandas as gpd
import pandas as pd

# List of barangays to include
barangays = ["Bucana, Davao City, Philippines", 
             "Ma-a, Davao City, Philippines", 
             "Matina Aplaya, Davao City, Philippines", 
             "74-A Matina Crossing, Davao City, Philippines", 
             "Matina Pangi, Davao City, Philippines"]
# barangays = ["74-A Matina Crossing, Davao City, Philippines"]


# Initialize an empty list to store each barangay's GeoDataFrame
gdf_edges_list = []

print("Starting script...")

# Loop through each barangay and download its road network
for barangay in barangays:
    print(f"Downloading road network data for {barangay}...")
    graph = ox.graph_from_place(barangay, network_type="all")
    print(f"Road network data for {barangay} downloaded successfully.")
    
    # Convert the graph to GeoDataFrames
    print("Converting graph to GeoDataFrames...")
    _, gdf_edges = ox.graph_to_gdfs(graph)
    gdf_edges_list.append(gdf_edges)
    print(f"Data for {barangay} added.")

# Combine all GeoDataFrames into one
combined_gdf_edges = pd.concat(gdf_edges_list, ignore_index=True)

# Save the combined GeoDataFrame as a GeoJSON file
output_file = "davao_specific_barangays_road_network.geojson"
print(f"Saving combined GeoDataFrame to {output_file}...")
combined_gdf_edges.to_file(output_file, driver="GeoJSON")
print(f"Road network for specified barangays has been saved to {output_file}")

print("Script completed successfully.")
