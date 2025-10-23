import streamlit as st
import folium
import urllib.parse
import requests
import base64
import re

# --- HELPER FUNCTION FOR ROBUST COORDINATE PARSING ---
def clean_and_parse_coords(coord_str):
    """Safely extracts latitude and longitude from a potentially messy string."""
    try:
        # Find all float-like numbers (positive or negative) in the string
        numbers = re.findall(r"[-]?\d+\.\d+", str(coord_str))
        if len(numbers) >= 2:
            # Convert the first two found numbers to float for [lat, lon]
            return [float(numbers[0]), float(numbers[1])]
    except (ValueError, IndexError):
        return None
    return None

# --- STATIC MAP FUNCTIONS ---
def create_static_map_url_with_labels(trip_data, api_key):
    """Generates a Google Static Map URL ensuring Start, End, and all stops are labeled distinctly."""
    base_url = "https://maps.googleapis.com/maps/api/staticmap?"
    params = {"size": "640x480", "maptype": "roadmap", "key": api_key}
    
    start_coords = trip_data.get("start_coordinates")
    end_coords = trip_data.get("end_coordinates")
    
    intermediate_stops = [
        s for s in trip_data.get("stops", []) 
        if s.get("coordinates") and s.get("coordinates") != start_coords and s.get("coordinates") != end_coords
    ]
    
    path_points = []
    if start_coords: path_points.append(start_coords)
    path_points.extend([s.get("coordinates") for s in trip_data.get("stops", []) if s.get("coordinates")])
    if end_coords: path_points.append(end_coords)
        
    if not path_points: return "https://via.placeholder.com/640x480.png?text=No+Coordinates+Found"
        
    path_str = f"path=color:0x000000|weight:5"
    for point in path_points: path_str += f"|{point}"
        
    marker_strings = []
    if start_coords: marker_strings.append(f"markers=color:green|label:S|{start_coords}")
    if end_coords: marker_strings.append(f"markers=color:red|label:E|{end_coords}")
    for i, stop in enumerate(intermediate_stops):
        marker_strings.append(f"markers=color:red|label:{i+1}|{stop['coordinates']}")
            
    final_url = f"{base_url}{urllib.parse.urlencode(params)}&{path_str}"
    for marker in marker_strings: final_url += f"&{marker}"
    return final_url

def get_map_as_base64(trip_data, api_key):
    """
    Fetches the static map image from Google and encodes it as Base64 for embedding in HTML.
    """
    map_url = create_static_map_url_with_labels(trip_data, api_key)
    if "placeholder.com" in map_url:
        return None
    try:
        response = requests.get(map_url)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode("utf-8")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not fetch map image. {e}")
        # Use st.error if in a Streamlit context, otherwise just log.
        if 'streamlit' in st.__name__:
            st.error(f"Error fetching map image: {e}")
    return None

# In app/map_generator.py

def get_optimized_route(trip_data, api_key):
    """
    Calls the Google Maps Directions API to find the fastest route
    that connects all waypoints, considering real-time traffic.
    This version is robust and handles cases where traffic data is unavailable.
    """
    try:
        base_url = "https://maps.googleapis.com/maps/api/directions/json"
        
        origin = trip_data.get("start_coordinates")
        destination = trip_data.get("end_coordinates")
        
        if not origin or not destination:
            return {"error": "Origin or destination coordinates are missing from the trip data."}
        
        stops = trip_data.get("stops", [])
        if not stops:
            return {"error": "No intermediate stops found to optimize."}
            
        waypoints_str = "optimize:true|" + "|".join([stop['coordinates'] for stop in stops])

        params = {
            "origin": origin,
            "destination": destination,
            "waypoints": waypoints_str,
            "key": api_key,
            "departure_time": "now",
            "traffic_model": "best_guess"
        }

        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        directions = response.json()

        if directions['status'] == 'OK':
            route = directions['routes'][0]
            
            optimized_order = route.get('waypoint_order', [])
            reordered_stops = [stops[i] for i in optimized_order]
            
            # ▼▼▼ THIS IS THE ROBUST FIX ▼▼▼
            # Calculate total duration by checking for traffic data first.
            total_duration_seconds = 0
            for leg in route['legs']:
                if 'duration_in_traffic' in leg:
                    # If traffic data is available for this leg, use it.
                    total_duration_seconds += leg['duration_in_traffic']['value']
                else:
                    # Otherwise, fall back to the standard duration for this leg.
                    total_duration_seconds += leg['duration']['value']
            # ▲▲▲ END OF THE FIX ▲▲▲
            
            return {
                "success": True,
                "reordered_stops": reordered_stops,
                "optimized_duration_seconds": total_duration_seconds,
                "summary": route.get('summary', 'Optimized Route')
            }
        else:
            error_message = directions.get('error_message', directions['status'])
            return {"error": f"Directions API Error: {error_message}"}

    except requests.exceptions.RequestException as e:
        return {"error": f"Network error when calling Google Maps API: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred during route optimization: {e}"}
    
# In app/map_generator.py

def get_route_comparison(trip_data, api_key):
    """
    Calls the Google Maps Directions API twice to compare the original route
    with an optimized route, both considering real-time traffic.
    This version includes specific handling for the ZERO_RESULTS error.
    """
    try:
        base_url = "https://maps.googleapis.com/maps/api/directions/json"
        origin = trip_data.get("start_coordinates")
        destination = trip_data.get("end_coordinates")
        stops = trip_data.get("stops", [])

        if not all([origin, destination, stops]):
            return {"error": "Missing origin, destination, or stops for comparison."}
        
        waypoints_str = "|".join([stop['coordinates'] for stop in stops])

        # --- API Call 1: Get duration for the ORIGINAL route in traffic ---
        params_original = {
            "origin": origin, "destination": destination, "waypoints": waypoints_str,
            "key": api_key, "departure_time": "now", "traffic_model": "best_guess"
        }
        response_original = requests.get(base_url, params=params_original)
        response_original.raise_for_status()
        directions_original = response_original.json()

        # ▼▼▼ THIS IS THE FIX ▼▼▼
        if directions_original['status'] != 'OK':
            if directions_original['status'] == 'ZERO_RESULTS':
                return {"error": "Google Maps could not find a valid driving route connecting all the AI-suggested stops. The locations may be too far apart or inaccessible. Please try generating a new itinerary."}
            else:
                return {"error": f"Could not calculate original route: {directions_original.get('error_message', directions_original['status'])}"}
        # ▲▲▲ END OF FIX ▲▲▲
        
        original_duration = sum(leg.get('duration_in_traffic', leg['duration'])['value'] for leg in directions_original['routes'][0]['legs'])

        # --- API Call 2: Get duration for the OPTIMIZED route in traffic ---
        params_optimized = {
            "origin": origin, "destination": destination, "waypoints": f"optimize:true|{waypoints_str}",
            "key": api_key, "departure_time": "now", "traffic_model": "best_guess"
        }
        response_optimized = requests.get(base_url, params=params_optimized)
        response_optimized.raise_for_status()
        directions_optimized = response_optimized.json()

        if directions_optimized['status'] != 'OK':
             # This check is less likely to fail if the first one passed, but it's good practice
            return {"error": f"Could not calculate optimized route: {directions_optimized.get('error_message', directions_optimized['status'])}"}

        route_optimized = directions_optimized['routes'][0]
        optimized_order = route_optimized.get('waypoint_order', [])
        reordered_stops = [stops[i] for i in optimized_order]
        optimized_duration = sum(leg.get('duration_in_traffic', leg['duration'])['value'] for leg in route_optimized['legs'])

        # --- Return a full comparison dictionary ---
        return {
            "success": True,
            "original_duration_seconds": original_duration,
            "optimized_duration_seconds": optimized_duration,
            "reordered_stops": reordered_stops,
            "time_saved_seconds": original_duration - optimized_duration
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"Network error when calling Google Maps API: {e}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred during route comparison: {e}"}

# --- INTERACTIVE & EMBEDDED MAP FUNCTIONS ---
def create_interactive_map(trip_data):
    """Creates a Folium interactive map object with markers and a route line."""
    all_points = []
    start_c = clean_and_parse_coords(trip_data.get("start_coordinates"))
    if start_c: all_points.append({"name": trip_data.get("start", "Start"), "coords": start_c, "type": "start"})
    
    for i, stop in enumerate(trip_data.get("stops", [])):
        stop_c = clean_and_parse_coords(stop.get("coordinates"))
        if stop_c: all_points.append({"name": stop.get("name"), "coords": stop_c, "type": "stop", "number": i + 1})

    end_c = clean_and_parse_coords(trip_data.get("end_coordinates"))
    if end_c: all_points.append({"name": trip_data.get("end", "End"), "coords": end_c, "type": "end"})

    if not all_points:
        return None

    # Calculate map center
    avg_lat = sum(p["coords"][0] for p in all_points) / len(all_points)
    avg_lon = sum(p["coords"][1] for p in all_points) / len(all_points)

    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=7)

    # Add markers
    for point in all_points:
        lat, lon = point["coords"]
        if point["type"] == "start":
            folium.Marker([lat, lon], popup=f"Start: {point['name']}", icon=folium.Icon(color='green', icon='star')).add_to(m)
        elif point["type"] == "end":
            folium.Marker([lat, lon], popup=f"End: {point['name']}", icon=folium.Icon(color='red', icon='flag')).add_to(m)
        else:
            folium.Marker([lat, lon], popup=f"Stop {point['number']}: {point['name']}", icon=folium.Icon(color='blue', icon='info-sign')).add_to(m)

    # Add route line
    route_coords = [p["coords"] for p in all_points]
    folium.PolyLine(route_coords, color="blue", weight=5, opacity=0.8).add_to(m)

    return m

def create_stop_map_html(coords, name, api_key):
    """Creates an HTML iframe for a single stop's embedded map."""
    if not api_key: return "<p>Map requires a Google Maps API Key.</p>"
    encoded_name = urllib.parse.quote_plus(name)
    # Ensure coords are a string for the URL
    coords_str = ",".join(map(str, clean_and_parse_coords(coords)))
    return f'<iframe width="100%" height="200" style="border:0; border-radius: 12px;" loading="lazy" allowfullscreen src="https://www.google.com/maps/embed/v1/place?key={api_key}&q={encoded_name},{coords_str}"></iframe>'

# --- Keep these legacy functions for compatibility if needed ---
def create_dynamic_map_html(trip_data, api_key):
    """Generates the HTML for a dynamic map (legacy)."""
    m = create_interactive_map(trip_data)
    if m:
        return m._repr_html_()
    return "<p>Could not generate map.</p>"

def generate_google_maps_directions_link(trip_data):
    """Generates a Google Maps directions link for the entire trip."""
    base_url = "https://www.google.com/maps/dir/?api=1"
    origin = urllib.parse.quote_plus(trip_data.get("start", ""))
    destination = urllib.parse.quote_plus(trip_data.get("end", ""))
    waypoints = "|".join(
        urllib.parse.quote_plus(stop.get("name", ""))
        for stop in trip_data.get("stops", [])
    )
    return f"{base_url}&origin={origin}&destination={destination}&waypoints={waypoints}"