import google.generativeai as genai
from math import radians, sin, cos, sqrt, atan2

def get_available_models(api_key):
    """Get list of available models"""
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        return [model.name for model in models]
    except Exception as e:
        return []

def find_best_model(api_key):
    """Find the best available model for content generation"""
    try:
        models = get_available_models(api_key)
        
        # Preferred models in order of preference
        preferred_models = [
            'models/gemini-1.5-flash', 
            'models/gemini-2.5-pro', 
            # 'models/gemini-pro'
        ]
        
        for model in preferred_models:
            if model in models:
                return model
                
        # If no preferred models found, return the first available model
        if models:
            return models[0]
            
        return None
    except Exception as e:
        return None

def calculate_travel_time(start_coords, end_coords, vehicle_type="Car"):
    """Calculate approximate travel time between two coordinates"""
    try:
        if not start_coords or not end_coords:
            return 0
            
        lat1, lon1 = map(float, start_coords.split(','))
        lat2, lon2 = map(float, end_coords.split(','))
        
        # Haversine formula to calculate distance
        R = 6371  # Earth radius in km
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        distance_km = R * c
        
        # Adjust speed based on vehicle type
        speed_kmh = 60  # Default speed
        if vehicle_type == "Motorcycle":
            speed_kmh = 50
        elif vehicle_type == "SUV":
            speed_kmh = 55
        elif vehicle_type == "Bus":
            speed_kmh = 45
        elif vehicle_type == "Truck":
            speed_kmh = 40
            
        # Calculate time in hours
        travel_time = distance_km / speed_kmh
        
        # Add buffer for traffic, stops, etc.
        travel_time *= 1.3
        
        return travel_time
        
    except:
        # Fallback: return a reasonable estimate
        return 1.5  # 1.5 hours between stops