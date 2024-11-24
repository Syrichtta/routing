import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLayout, QVBoxLayout, QHBoxLayout, QWidget, QSizePolicy, QPushButton
from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineProfile
from PyQt5.QtCore import QUrl, pyqtSlot, pyqtSignal, QObject
from PyQt5.QtWebChannel import QWebChannel
from bs4 import BeautifulSoup
import os
import json
import math
from tqdm import tqdm

# screen_width = app.winfo_screenwidth()
# screen_height = app.winfo_screenheight()

map_file = "C:/Users/Admin/Documents/GitHub/routing/scripts/davao_road_network_nodes_on_click.html"
geojson_path = "C:/Users/Admin/Documents/GitHub/routing/davao_bounding_box_road_network.geojson"

def update_map_file(map_file):
    with open(map_file, 'r', encoding='utf-8') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, "html.parser")

    new_script_tag = soup.new_tag('script', type='text/javascript', src='qrc:///qtwebchannel/qwebchannel.js')
    soup.head.append(new_script_tag)

    existing_script_tag = soup.find('body').find_next('script')

    js_code = """
        let isPinningEnabled = false;

        new QWebChannel(qt.webChannelTransport, function (channel) {
            const qtObject = channel.objects.qtObject;

            qtObject.pinningEnabled.connect(function () {
                isPinningEnabled = true;
                qtObject.log_message(`Enabled map pinning set to ${isPinningEnabled}!`);
            });

            qtObject.pinningDisabled.connect(function () {
                isPinningEnabled = false;
                qtObject.log_message(`Enabled map pinning set to ${isPinningEnabled}!`);
            });

            var currentMarker = null;  // Variable to hold the current marker

            qtObject.pinMoved.connect(function (data) {
                qtObject.log_message("Moving pin");

                var coordinates = JSON.parse(data);

                var lat = coordinates.lat;
                var lon = coordinates.lon;

                if (currentMarker) {
                    map_73dbd4f2f1f6398046a4e37cb3cc066f.removeLayer(currentMarker);
                }

                // Add a new marker
                currentMarker = L.marker([lat, lon]).addTo(map_73dbd4f2f1f6398046a4e37cb3cc066f);

                var updatedData = JSON.stringify({ lat: lat, lon: lon });
                qtObject.pin_dropped(updatedData);
            });

            map_73dbd4f2f1f6398046a4e37cb3cc066f.on('click', function(event) {
                if(isPinningEnabled){
                    var lat = event.latlng.lat;
                    var lon = event.latlng.lng;
                    // Remove the existing marker, if any
                    if (currentMarker) {
                        map_73dbd4f2f1f6398046a4e37cb3cc066f.removeLayer(currentMarker);
                    }

                    // Add a new marker
                    currentMarker = L.marker([lat, lon]).addTo(map_73dbd4f2f1f6398046a4e37cb3cc066f);

                    // Send the coordinates back to Python through the registered object (qtObject)
                    var data = JSON.stringify({ lat: lat, lon: lon });
                    qtObject.pin_dropped(data);
                }
            });
        });
    """

    # Append the JavaScript code to the existing script tag
    if existing_script_tag:
        existing_script_tag.string = (existing_script_tag.string or "") + js_code
    else:
        print("No existing <script> tag found in the body.")

    original_directory = os.path.dirname(map_file)

    new_file_path = os.path.join(original_directory, "davao_road_network_nodes_on_click_select_nodes.html")

    with open(new_file_path, 'w', encoding='utf-8') as file:
        file.write(str(soup))

    print("HTML file updated and saved!")

    return new_file_path

def delete_updated_map_file():
    try:
        if os.path.exists(updated_map_file):
            os.remove(updated_map_file)
            print(f"{updated_map_file} has been deleted.")
    except Exception as e:
        print(f"Error deleting file: {e}")

class PythonBackend(QObject):
    def __init__(self):
        super().__init__()
        self.geojson_data = None  # Store geojson data to use in calculations
        self.pin_lat = None
        self.pin_lon = None

    @pyqtSlot(str)  # This will accept the serialized data as a string
    def pin_dropped(self, data):
        # Deserialize the data (JSON format)
        import json
        coords = json.loads(data)
        self.pin_lat = coords['lat']
        self.pin_lon = coords['lon']
        print(f"Pin dropped at: Latitude = {self.pin_lat}, Longitude = {self.pin_lon}")

    pinningEnabled = pyqtSignal()
    @pyqtSlot()
    def enable_pinning(self):
        self.pinningEnabled.emit()

    @pyqtSlot(str)
    def log_message(self, message):
        print(f"Log from JavaScript: {message}")

    pinningDisabled = pyqtSignal()
    @pyqtSlot()
    def disable_pinning(self):
        self.pinningDisabled.emit()
        if self.pin_lat is not None and self.pin_lon is not None:
            min_distance, nearest_coords = self.find_nearest_node_within_radius(self.pin_lat, self.pin_lon)
            if nearest_coords:
                print(f"Distance to nearest node: {min_distance} km")
                print(f"Nearest coordinates: {nearest_coords}")
            else:
                print("No node found within radius.")
        else:
            print("No pin has been dropped yet.")

    def euclidean_distance(self, lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        lat_distance = (lat2 - lat1) * 111  # Approx 111 km per degree of latitude
        lon_distance = (lon2 - lon1) * 111 * math.cos((lat1 + lat2) / 2)  # Longitude distance depends on latitude
        return math.sqrt(lat_distance ** 2 + lon_distance ** 2)

    def find_nearest_node_within_radius(self, pin_lat, pin_lon, radius_km=1.0):
        with open(geojson_path, 'r') as f:
            self.geojson_data = json.load(f)
                                       
        min_distance = float('inf')

        for feature in tqdm(self.geojson_data['features'], desc="Finding nearest node", unit="node"):
            coords = feature['geometry']['coordinates']
            if feature['geometry']['type'] == 'LineString':
                for coord in coords:
                    lon, lat = coord  # Unpack the coordinate pair (longitude, latitude)

                    # Calculate the Euclidean distance from the pin to this coordinate
                    distance = self.euclidean_distance(pin_lat, pin_lon, lat, lon)

                    # Check if this is the closest node within the radius
                    if distance <= radius_km and distance < min_distance:
                        min_distance = distance
                        nearest_coords = (lat, lon)

        if nearest_coords:
        # Send coordinates to JS after finding the nearest node
            self.move_pin_to_js(nearest_coords[0], nearest_coords[1])  # Send to JavaScript
        return min_distance, nearest_coords
    
    pinMoved = pyqtSignal(str)
    @pyqtSlot(float, float)
    def move_pin_to_js(self, lat, lon):
        # Send the coordinates to JavaScript using the Qt API
        data = json.dumps({'lat': lat, 'lon': lon})
        print(f"Moving pin to lat: {lat}, lon: {lon}")
        self.pinMoved.emit(data)

class WebViewWindow(QMainWindow):
    def __init__(self, updated_map_file):
        super().__init__()
        self.setWindowTitle("Local HTML in WebView")

        screen = QApplication.primaryScreen()  # Get the primary screen
        screen_geometry = screen.geometry()  # Get the screen's geometry (dimensions)
        
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        # Create the central widget and main_layout
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)

        # Create QVBoxLayout for parameter selection
        parameters_widget = QWidget()
        parameters_widget.setStyleSheet("background-color: lightblue;")
        parameters_layout = QVBoxLayout(parameters_widget)
        main_layout.addWidget(parameters_widget)
        parameters_widget.setMinimumWidth(600)

        # Create button to enable selecting button
        pin_button = QPushButton("Pick Location")
        pin_button.setStyleSheet("background-color: white;")
        pin_button.clicked.connect(self.enable_pinning)
        parameters_layout.addWidget(pin_button)  
        
        # Create button to confirm location
        confirm_button = QPushButton("Confirm Location")
        confirm_button.setStyleSheet("background-color: white;")
        confirm_button.clicked.connect(self.disable_pinning)
        parameters_layout.addWidget(confirm_button)

        # Create QWebEngineView and load local HTML
        self.web_view = QWebEngineView()
        self.web_view.setUrl(QUrl.fromLocalFile(updated_map_file))  # Update with your file's path
        main_layout.addWidget(self.web_view)
        self.web_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.setCentralWidget(central_widget)
        self.setMinimumSize(800, 600)
        self.resize(screen_width, screen_height)  # Adjust the size of the window
        
        self.channel = QWebChannel()
        self.backend = PythonBackend()  # Instantiate the Python backend
        self.channel.registerObject("qtObject", self.backend)  # Register the Python object
        self.web_view.page().setWebChannel(self.channel) 
    
    def enable_pinning(self):
        self.backend.enable_pinning()

    def disable_pinning(self):
        self.backend.disable_pinning()
    
if __name__ == "__main__":
    updated_map_file = update_map_file(map_file)
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(delete_updated_map_file)
    window = WebViewWindow(updated_map_file)
    window.show()
    sys.exit(app.exec_())


