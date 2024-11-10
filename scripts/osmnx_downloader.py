import osmnx as ox
import geopandas as gpd

# Define bounding box coordinates from the GeoJSON data
north, south = 7.125251, 7.021505
east, west = 125.633297, 125.550213

print("Starting script...")

# Download the road network within the bounding box
print("Downloading road network data for the specified bounding box...")
graph = ox.graph_from_bbox(north, south, east, west, network_type="all")
print("Road network data downloaded successfully.")

# Convert the graph to GeoDataFrames
print("Converting graph to GeoDataFrames...")
_, gdf_edges = ox.graph_to_gdfs(graph)

# Save the GeoDataFrame as a GeoJSON file
output_file = "davao_bounding_box_road_network.geojson"
print(f"Saving GeoDataFrame to {output_file}...")
gdf_edges.to_file(output_file, driver="GeoJSON")
print(f"Road network for the specified bounding box has been saved to {output_file}")

print("Script completed successfully.")
