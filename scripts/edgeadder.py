import json
from pathlib import Path

# Load existing GeoJSON data
input_file = Path("/home/syrichta/routing/davao_specific_barangays_road_network.geojson")  # Change the path to your GeoJSON file
with open(input_file) as f:
    geojson_data = json.load(f)

# Create a new edge feature
new_edge = {
    "type": "Feature",
    "properties": {
        "lanes": "2",
        "name": "New Road Segment",
        "highway": "tertiary",
        "maxspeed": "40",
        "oneway": False,
        "reversed": "0",
        "length": 100.0,  # You can adjust this based on your actual calculation
        "ref": None,
        "service": None,
        "access": None,
        "width": None,
        "bridge": None,
        "junction": None,
        "tunnel": None
    },
    "geometry": {
        "type": "LineString",
        "coordinates": [
            [125.599451, 7.063380],
            [125.600518, 7.063634]
        ]
    }
}

# Append the new edge feature to the GeoJSON features
geojson_data['features'].append(new_edge)

# Save the updated GeoJSON data back to the file
output_file = Path("appended_edges.geojson")  # Change the path for the output file
with open(output_file, 'w') as f:
    json.dump(geojson_data, f, indent=4)

print(f"Updated GeoJSON saved to {output_file}")
