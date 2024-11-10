import folium
from folium.plugins import Draw

# Set initial location and zoom level for the map (centered around Davao City)
center_lat, center_lon = 7.0731, 125.6128
zoom_start = 13

# Create a Folium map
m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

# Add drawing tools to the map with only the rectangle tool enabled
draw = Draw(export=False,
            draw_options={
                'polyline': False,
                'polygon': False,
                'circle': False,
                'marker': False,
                'circlemarker': False,
                'rectangle': True,
            })
draw.add_to(m)

# JavaScript code to log bounding box coordinates in the console
custom_js = """
function logBoundingBox(e) {
    var bounds = e.layer.getBounds();
    var north = bounds.getNorth();
    var south = bounds.getSouth();
    var east = bounds.getEast();
    var west = bounds.getWest();
    console.log("Bounding Box Coordinates:");
    console.log("North: " + north);
    console.log("South: " + south);
    console.log("East: " + east);
    console.log("West: " + west);
}

// Add event listener to log coordinates on rectangle creation
map.on('draw:created', function (e) {
    if (e.layerType === 'rectangle') {
        logBoundingBox(e);
    }                
});
"""

# Add custom JavaScript to the map
m.get_root().html.add_child(folium.Element(f"<script>{custom_js}</script>"))

# Save the map as an HTML file
output_map_file = "select_bounding_box.html"
m.save(output_map_file)

print(f"Map with bounding box selection saved as '{output_map_file}'.")
print("Open this file in a browser, draw a rectangle, and check the browser console for bounding box coordinates.")
