import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QSizePolicy, QPushButton, QSlider, QLabel, QTableWidget, QTableWidgetItem
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl, pyqtSlot, pyqtSignal, QObject, Qt
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtGui import QFont
from bs4 import BeautifulSoup
import os
import json
import math
from tqdm import tqdm
import subprocess
from pathlib import Path

# screen_width = app.winfo_screenwidth()
# screen_height = app.winfo_screenheight()

map_file = "C:/Users/Admin/Documents/GitHub/routing/scripts/davao_road_network_nodes_on_click.html"
geojson_path = "C:/Users/Admin/Documents/GitHub/routing/davao_bounding_box_road_network.geojson"
map_name = "map_ba5e63183e28e502cea4175d40fe99e1"

def update_map_file(map_file, map_name):
    with open(map_file, 'r', encoding='utf-8') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, "html.parser")

    new_script_tag = soup.new_tag('script', type='text/javascript', src='qrc:///qtwebchannel/qwebchannel.js')
    soup.head.append(new_script_tag)

    existing_script_tag = soup.find('body').find_next('script')

    js_code_template = """
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
                    {map_name_placeholder}.removeLayer(currentMarker);
                }

                // Add a new marker
                currentMarker = L.marker([lat, lon]).addTo({map_name_placeholder});

                var updatedData = JSON.stringify({ lat: lat, lon: lon });
                qtObject.pin_dropped(updatedData);
            });

            {map_name_placeholder}.on('click', function(event) {
                if(isPinningEnabled){
                    var lat = event.latlng.lat;
                    var lon = event.latlng.lng;
                    // Remove the existing marker, if any
                    if (currentMarker) {
                        {map_name_placeholder}.removeLayer(currentMarker);
                    }

                    // Add a new marker
                    currentMarker = L.marker([lat, lon]).addTo({map_name_placeholder});

                    // Send the coordinates back to Python through the registered object (qtObject)
                    var data = JSON.stringify({ lat: lat, lon: lon });
                    qtObject.pin_dropped(data);
                }
            });
        });
    """
    js_code = js_code_template.replace("{map_name_placeholder}" , map_name)

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
        self.pin_lat = lat
        self.pin_lon = lon
        self.pinMoved.emit(data)

    outputMap = pyqtSignal(str)
    displayMetrics = pyqtSignal(dict)
    @pyqtSlot(int, int)
    def simulate_route_search(self, ant_count, iteration_count):
        print("Running path search")
        data = json.dumps({
            'lat': self.pin_lat,
            'lon': self.pin_lon,
            'ant_count': ant_count,
            'iteration_count': iteration_count
            })
        try: 
            result = subprocess.run(
                ["python", "C:/Users/Admin/Documents/GitHub/routing/scripts/astar_aco7.py", data],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                # Process the output
                print("Output from astar_aco7.py:", result.stdout)

                stdout_lines = result.stdout.strip().split("\n")
                json_part = stdout_lines[-1]
                result_data = json.loads(json_part)

                map_file_with_path = Path(__file__).parent.parent / result_data['output_html']
                self.outputMap.emit(str(map_file_with_path))
                # self.outputMap.emit(result_data['output_html'])
                self.displayMetrics.emit(result_data)
            else:
                print("Error:", result.stderr)
        except Exception as e:
            print(f"An exception occurred: {e}")

class WebViewWindow(QMainWindow):
    def __init__(self, updated_map_file):
        super().__init__()
        self.setWindowTitle("Davao Dynamic Flood Evacuation Pathfinder")
        self.setStyleSheet(
            """
                QPushButton {
                    background-color: white;
                    color: black;
                    font-size: 12px;
                    border: 1px solid black;
                    border-radius: 4px;
                    padding: 10px 15px;
                }
                QLabel {
                    font-size: 12px;
                }
            """
        )

        screen = QApplication.primaryScreen()  # Get the primary screen
        screen_geometry = screen.geometry()  # Get the screen's geometry (dimensions)
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        # Create the central widget and main_layout
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        
        # Create QVBoxLayout for user actions
        actions_widget = QWidget()
        actions_widget.setMinimumWidth(int(screen_width*.2))
        actions_widget.setMaximumWidth(int(screen_width*.4))
        actions_widget.setStyleSheet("background-color: white;")
        actions_layout = QVBoxLayout(actions_widget)
        main_layout.addWidget(actions_widget)

        # Create QVBoxLayout for button selection
        buttons_widget = QWidget()
        # buttons_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        buttons_layout = QVBoxLayout(buttons_widget)
        actions_layout.addWidget(buttons_widget)

        # Create QVBoxLayout for parameter selection
        parameters_widget = QWidget()
        parameters_layout = QVBoxLayout(parameters_widget)
        actions_layout.addWidget(parameters_widget)

        # Create QVBoxLayout for results selection
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        actions_layout.addWidget(results_widget)

        # Create button to enable selecting button
        self.pin_button = QPushButton("Pick Location")
        self.pin_button.setStyleSheet(
            """
                QPushButton:pressed {
                    background-color: #1d6fa5;  /* Darker blue when pressed */
                }
                QPushButton:hover {
                    background-color: #3498db;  /* Lighter blue when hovered */
                }
                QPushButton:disabled {
                    background-color: #d6d6d6;  /* Darker blue when pressed */
                }
            """
        )
        self.pin_button.clicked.connect(self.enable_pinning)
        buttons_layout.addWidget(self.pin_button)  
        
        # Create button to confirm location & search nearest node
        self.confirm_button = QPushButton("Confirm Location")
        self.confirm_button.setStyleSheet(
            """
                QPushButton:pressed {
                    background-color: #1d6fa5;  /* Darker blue when pressed */
                }
                QPushButton:hover {
                    background-color: #3498db;  /* Lighter blue when hovered */
                }
                QPushButton:disabled {
                    background-color: #d6d6d6;  /* Darker blue when pressed */
                }
            """
        )
        self.confirm_button.setEnabled(False)
        self.confirm_button.clicked.connect(self.disable_pinning)
        buttons_layout.addWidget(self.confirm_button)

        # Create button to calculate closest path
        self.simulate_button = QPushButton("Simulate Route")
        self.simulate_button.setStyleSheet(
            """
                QPushButton:pressed {
                    background-color: #1d6fa5;  /* Darker blue when pressed */
                }
                QPushButton:hover {
                    background-color: #3498db;  /* Lighter blue when hovered */
                }
                QPushButton:disabled {
                    background-color: #d6d6d6;  /* Darker blue when pressed */
                }
            """
        )
        self.simulate_button.setEnabled(False)
        self.simulate_button.clicked.connect(self.simulate_route_search)
        buttons_layout.addWidget(self.simulate_button)

        # Create parameter sliders
        parameter_layout_ant = QHBoxLayout()
        parameters_layout.addLayout(parameter_layout_ant)
        parameter_layout_iteration = QHBoxLayout()
        parameters_layout.addLayout(parameter_layout_iteration)
        parameter_layout_iteration = QHBoxLayout()
        parameters_layout.addLayout(parameter_layout_iteration)

        # Parameter: Ant Count
        self.label_ant = QLabel("Ant Count: 25")
        parameter_layout_ant.addWidget(self.label_ant)
        self.slider_ant = QSlider(Qt.Horizontal)
        self.slider_ant.setMinimum(10)
        self.slider_ant.setMaximum(100)
        self.slider_ant.setValue(25)
        self.slider_ant.setTickPosition(QSlider.TicksBelow)
        self.slider_ant.setTickInterval(10)   # Interval between ticks
        self.slider_ant.setSingleStep(5)
        self.slider_ant.valueChanged.connect(self.change_value_ant)
        parameter_layout_ant.addWidget(self.slider_ant)

        # Parameter: Iteration Count
        self.label_iteration = QLabel("Iterations: 50")
        parameter_layout_iteration.addWidget(self.label_iteration)
        self.slider_iteration = QSlider(Qt.Horizontal)
        self.slider_iteration.setMinimum(10)
        self.slider_iteration.setMaximum(300)
        self.slider_iteration.setValue(50)
        self.slider_iteration.setTickPosition(QSlider.TicksBelow)
        self.slider_iteration.setTickInterval(25)   # Interval between ticks
        self.slider_iteration.setSingleStep(5)
        self.slider_iteration.valueChanged.connect(self.change_value_iteration)
        parameter_layout_iteration.addWidget(self.slider_iteration)

        # Display routing of selected node to nearest shelter 👍
        # Manipulate ACO Parameters - clarify parameters editable
        # Display result metrics 👍
        # Disable buttons during certain actions 👍
        
        
        # Add loading animation during simulation
        # Add map location repicking
        # Display routing of selected node to all shelters
        # Display flood map overlay
        # If makaya display routing of surrounding nodes (to show distribution)

        # Create results section
        font_size = QFont()
        font_size.setPointSize(10)

        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(2)  # Two columns: Key and Value
        self.table_widget.setHorizontalHeaderLabels(["Metric", "Value"])
        self.table_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table_widget.setFont(font_size)
        self.table_widget.insertRow(0) 
        self.table_widget.setItem(0, 0, QTableWidgetItem("Distance Travelled"))
        self.table_widget.setItem(0, 1, QTableWidgetItem("0m"))
        self.table_widget.insertRow(1) 
        self.table_widget.setItem(1, 0, QTableWidgetItem("Travel Time"))
        self.table_widget.setItem(1, 1, QTableWidgetItem("0min 0sec"))
        self.table_widget.insertRow(2) 
        self.table_widget.setItem(2, 0, QTableWidgetItem("Computational Time"))
        self.table_widget.setItem(2, 1, QTableWidgetItem("0min 0sec"))
        self.table_widget.insertRow(3) 
        self.table_widget.setItem(3, 0, QTableWidgetItem("Total Slope Gain"))
        self.table_widget.setItem(3, 1, QTableWidgetItem("0m"))
        self.table_widget.insertRow(4) 
        self.table_widget.setItem(4, 0, QTableWidgetItem("Total Slope Loss"))
        self.table_widget.setItem(4, 1, QTableWidgetItem("0m"))
        self.table_widget.insertRow(5) 
        self.table_widget.setItem(5, 0, QTableWidgetItem("Max Flood Depth Traversed"))
        self.table_widget.setItem(5, 1, QTableWidgetItem("0m"))
        self.table_widget.resizeColumnsToContents()
        results_layout.addWidget(self.table_widget)

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

        self.backend.outputMap.connect(self.output_map)
        self.backend.displayMetrics.connect(self.display_metrics)
    
    def enable_pinning(self):
        self.simulate_button.setEnabled(False)
        self.confirm_button.setEnabled(True)
        self.backend.enable_pinning()

    def disable_pinning(self):
        self.simulate_button.setEnabled(True)
        self.backend.disable_pinning()

    def simulate_route_search(self):
        self.pin_button.setEnabled(False)
        self.confirm_button.setEnabled(False)
        ant_count = self.slider_ant.value()
        iteration_count = self.slider_iteration.value()
        self.backend.simulate_route_search(ant_count, iteration_count)
    
    @pyqtSlot(str)
    def output_map(self, map_file_with_path):
        self.web_view.setUrl(QUrl.fromLocalFile(map_file_with_path))
        print(f"Map updated to: {map_file_with_path}")

    @pyqtSlot(dict)
    def display_metrics(self, metrics):
        self.table_widget.setRowCount(0)

        self.table_widget.insertRow(0) 
        self.table_widget.setItem(0, 0, QTableWidgetItem("Distance Travelled"))
        self.table_widget.setItem(0, 1, QTableWidgetItem(f"{metrics["final_distance"]}m"))
        self.table_widget.insertRow(1) 
        self.table_widget.setItem(1, 0, QTableWidgetItem("Travel Time"))
        self.table_widget.setItem(1, 1, QTableWidgetItem(f"{int(metrics["final_time"] / 60)}min {round(metrics["final_time"] % 60, 0):.0f}sec"))
        self.table_widget.insertRow(2) 
        self.table_widget.setItem(2, 0, QTableWidgetItem("Computational Time"))
        self.table_widget.setItem(2, 1, QTableWidgetItem(f"{int(metrics["aco_duration"] / 60)}min {round(metrics["aco_duration"] % 60, 0):.0f}sec"))
        self.table_widget.insertRow(3) 
        self.table_widget.setItem(3, 0, QTableWidgetItem("Total Slope Gain"))
        self.table_widget.setItem(3, 1, QTableWidgetItem(f"{metrics["final_gain"]}m"))
        self.table_widget.insertRow(4) 
        self.table_widget.setItem(4, 0, QTableWidgetItem("Total Slope Loss"))
        self.table_widget.setItem(4, 1, QTableWidgetItem(f"{metrics["final_loss"]}m"))
        self.table_widget.insertRow(5) 
        self.table_widget.setItem(5, 0, QTableWidgetItem("Max Flood Depth Traversed"))
        self.table_widget.setItem(5, 1, QTableWidgetItem(f"{metrics["final_flood"]}m"))

        self.table_widget.resizeColumnsToContents()
    
    def change_value_ant(self, value):
        """Update the label with the slider's value."""
        self.label_ant.setText(f"Ant Count: {value}")

    def change_value_iteration(self, value):
        """Update the label with the slider's value."""
        self.label_iteration.setText(f"Iterations: {value}")

if __name__ == "__main__":
    updated_map_file = update_map_file(map_file, map_name)
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(delete_updated_map_file)
    window = WebViewWindow(updated_map_file)
    window.show()
    sys.exit(app.exec_())


