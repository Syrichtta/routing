import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the GeoJSON file
# input_file = "/home/syrichta/routing/davao_road_network.geojson"
input_file = Path(__file__).parent.parent / "davao_specific_barangays_road_network.geojson"
print(f"Loading data from {input_file}...")
gdf = gpd.read_file(input_file)
print("Data loaded successfully.")

# Plot the GeoDataFrame
print("Generating plot...")
fig, ax = plt.subplots(figsize=(12, 12))
gdf.plot(ax=ax, linewidth=0.5, edgecolor="blue")

# Customize plot appearance
ax.set_title("Davao City Highway Network", fontsize=16)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, which="both", linestyle="--", linewidth=0.5)

# Display the plot
plt.show()
