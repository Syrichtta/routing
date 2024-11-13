# ABC Thesis A*ACO Flood Routing
This is a thesis project that implements a  Hybrid A*ACO algorithm for dynamic routing during flood.
## File Directory
### Scripts
- **aco.py** - Ant Colony Script
- **astar.py** - A* Script
- **checkconnectivity.py** - Script to verify node connectivity
- **checknodes.py** - Script to check if certain nodes exist in the graph
- **combinedata.py** - Script to combine Flood & Elevation into the road GeoJSON
- **iteratethroughselectednodes.py** - iterate A* through selected nodes and store shortest paths
- **osmnxdownloader.py** - download the roadmap
- **selectboundingbox.py** - html file to select a bounding box
- **selectrandomnodes.py** - script to select N nodes at random
- **viewnodes.py** - view all nodes in the graph
- **viewroads.py** - view entire road network
### Files
- **aco_log.txt** - log of ant behavior during ACO
- **davao_bounding_box_road_network.geojson** - road network for the bounding box
- **davaoFloodMap11_11_24_SRI30.tif** - flood map
- **dem** - Dynamic Elevation Model
- **roads_with_elevation_and_flood.geojson** - road network with appended elevation and flood data
- **selected_nodes.txt** - selected nodes to be used for testing
