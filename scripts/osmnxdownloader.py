import osmnx as ox
import geopandas as gpd

north, south = 7.125251, 7.021505
east, west = 125.633297, 125.550213

graph = ox.graph_from_bbox(north, south, east, west, network_type="all")

_, gdf_edges = ox.graph_to_gdfs(graph)

output_file = "davao_bounding_box_road_network.geojson"
gdf_edges.to_file(output_file, driver="GeoJSON")
