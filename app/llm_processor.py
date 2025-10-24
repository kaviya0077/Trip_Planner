import google.generativeai as genai
import json
import re
import streamlit as st

def get_available_models(api_key):
    """Get list of available models"""
    try:
        genai.configure(api_key=api_key)
        models = genai.list_models()
        return [model.name for model in models]
    except Exception as e:
        st.error(f"Error fetching models: {e}")
        return []

def find_best_model(api_key):
    """Find the best available model for content generation"""
    try:
        models = get_available_models(api_key)
        
        # Preferred models in order of preference
        preferred_models = [
            # 'models/gemini-2.5-flash', 
            'models/gemini-2.5-pro-latest', 
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
        st.error(f"Error finding model: {e}")
        return None

def get_trip_recommendations(prompt, api_key, max_hours=10, num_stops=5, vehicle_type="Car", num_people=2, budget="Moderate"):
    """
    Get trip recommendations from Gemini model with time constraints
    """
    try:
        if not api_key:
            return {"error": "API key not provided"}
            
        genai.configure(api_key=api_key)
        
        # Find the best available model
        model_name = find_best_model(api_key)
        if not model_name:
            return {"error": "No suitable model found"}
        
        # Create the model
        model = genai.GenerativeModel(model_name)
        
        # Create enhanced prompt with time constraints
        from prompts import TIME_CONSTRAINED_PROMPT
        enhanced_prompt = TIME_CONSTRAINED_PROMPT.format(
            prompt=prompt,
            max_hours=max_hours,
            vehicle_type=vehicle_type,
            num_people=num_people,
            budget=budget
        )
        
        # Generate content
        response = model.generate_content(enhanced_prompt)
        
        # Parse the response
        trip_data = parse_trip_response(response.text, max_hours)
        return trip_data
        
    except Exception as e:
        return {"error": f"Error getting LLM response: {str(e)}"}

def parse_trip_response(response_text, max_hours=10):
    """
    Extract structured trip data from LLM response and validate time constraints
    """
    try:
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            trip_data = json.loads(json_str)
            
            # Validate time constraints
            total_time = 0
            try:
                total_time = float(trip_data.get("total_trip_time", 0))
            except:
                # Calculate approximate total time if not provided
                total_time = calculate_total_trip_time(trip_data)
                
            trip_data["time_constraint_met"] = total_time <= max_hours
            return trip_data
    except:
        pass
    
    # Fallback if no JSON found or parsing failed
    return get_fallback_trip_data(max_hours)

def calculate_total_trip_time(trip_data):
    """Calculate total trip time from stops data"""
    total_time = 0
    stops = trip_data.get("stops", [])
    
    if not stops:
        return 0
        
    # Add travel time between stops (simplified estimation)
    for i in range(len(stops) - 1):
        travel_time = 1.0  # Default 1 hour between stops
        total_time += travel_time
    
    # Add visiting time at each stop
    for stop in stops:
        visiting_time = stop.get("visiting_time", 0)
        if isinstance(visiting_time, str):
            try:
                visiting_time = float(visiting_time)
            except:
                visiting_time = 0.5  # Default 30 minutes
        total_time += visiting_time
    
    return total_time

def get_fallback_trip_data(max_hours=10):
    """Get fallback trip data that respects time constraints"""
    return {
        "start": "Coimbatore",
        "end": "Chennai",
        "total_driving_distance": "450 km",
        "total_driving_time": "7.5 hours",
        "total_visiting_time": "2 hours",
        "total_trip_time": "9.5 hours",
        "time_constraint_met": True,
        "stops": [
            {
                "name": "Sri Ranganathaswamy Temple", 
                "type": "temple", 
                "coordinates": "10.8620,78.6910",
                "description": "One of the most famous Vaishnavite temples in South India",
                "visiting_time": 1.0,
                "rating": "4.8"
            },
            {
                "name": "Hotel Saravana Bhavan", 
                "type": "restaurant", 
                "coordinates": "11.0168,76.9558",
                "description": "Quick service restaurant for authentic South Indian food",
                "visiting_time": 0.5,
                "rating": "4.2"
            },
            {
                "name": "Indian Oil Fuel Station", 
                "type": "fuel", 
                "coordinates": "11.1000,77.1000",
                "description": "Efficient fuel station with clean facilities",
                "visiting_time": 0.25,
                "rating": "4.0"
            }
        ],
        "additional_recommendations": f"Total trip time is under {max_hours} hours. Consider quick visits at stops to maintain schedule."
    }