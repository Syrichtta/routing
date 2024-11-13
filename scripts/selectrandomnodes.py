import json
import random

# Load the updated GeoJSON with elevation and flood depth
with open('roads_with_elevationflood.geojson', 'r') as f:
    geojson_data = json.load(f)

# List to store randomly selected coordinates
selected_coords = []

# Iterate over features and randomly select coordinates
for feature in geojson_data['features']:
    if feature['geometry']['type'] == 'LineString':
        # Choose a random coordinate from the LineString
        random_coord = random.choice(feature['geometry']['coordinates'])
        selected_coords.append(random_coord)

        # Stop once you have 100 nodes
        if len(selected_coords) >= 100:
            break

# Save selected coordinates to a .txt file
with open('selected_nodes.txt', 'w') as f:
    for coord in selected_coords:
        f.write(f"{coord[0]}, {coord[1]}\n")

print("100 random nodes selected and saved to selected_nodes.txt.")
