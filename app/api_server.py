import os
import uuid
from flask import Flask, jsonify, abort, render_template
from dotenv import load_dotenv
from .database import load_trip_from_db
from .map_generator import get_map_as_base64 

load_dotenv()
app = Flask(__name__, template_folder='template')

@app.route("/")
def index():
    """A simple welcome message to confirm the server is running."""
    return jsonify({"message": "Welcome to the AI Travel Planner API (Flask Version)"})

@app.route("/trip/<uuid:trip_id>", methods=["GET"])
def get_trip_json(trip_id: uuid.UUID):
    """
    Provides the raw JSON data for a specific trip.
    This is a pure data endpoint for potential future use (e.g., mobile app).
    """
    trip_data = load_trip_from_db(str(trip_id))
    if not trip_data:
        abort(404, description="Trip not found")
    return jsonify(trip_data)

@app.route("/itinerary/<uuid:trip_id>", methods=["GET"])
def show_html_itinerary(trip_id: uuid.UUID):
    """
    Finds a trip by its ID and renders its full details using the template.html file.
    The URL for this page is clean and does not contain any file extensions.
    Example: /itinerary/e2c7a96f-9773-48d4-b197-c61f9aaca88c
    """
    print(f"Received request for HTML itinerary with trip_id: {trip_id}")
    
    # 1. Load the trip data from the database
    trip_data = load_trip_from_db(str(trip_id))
    
    if not trip_data:
        # If the ID is valid but not in the DB, return a user-friendly 404 page
        # Note: You should create a simple '404.html' in your templates folder
        return render_template('404.html'), 404

    # 2. Get the Google Maps API key from environment variables
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        # This is a server configuration error, return a 500 error
        abort(500, description="Server is missing a required configuration (Google Maps API Key).")

    # 3. Generate the embedded map image data
    # This function is imported from your shared map_generator.py module
    map_image_data = get_map_as_base64(trip_data, api_key)
    
    # 4. Render the template, passing all necessary data to it
    # Flask will look for 'template.html' inside the 'app/templates/' folder
    # The 'trip' and 'map_image' variables can be used in the HTML with Jinja2 syntax
    return render_template('template.html', trip=trip_data, map_image=map_image_data)