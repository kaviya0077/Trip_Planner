# In app/prompts.py

def create_travel_prompt(prompt, num_people=2, budget="Moderate"):
    """
    Create an enhanced prompt for travel planning with cost estimation and optimization.
    """
    
    optimization_instruction = ""
    
    return f"""
    You are an expert travel planner and budget analyst with extensive knowledge of routes across India. 
    Your primary task is to create a practical itinerary and a detailed cost estimation.
    
    Analyze this travel request: "{prompt}"
    
    {optimization_instruction}

    First, create the itinerary. For the start and end locations, and for each stop, provide realistic latitude and longitude coordinates.
    
    Return your complete response as a single JSON object with this exact structure:
    {{
        "start": "start_location_name",
        "end": "end_location_name",
        "start_coordinates": "lat,lng_for_start_location",
        "end_coordinates": "lat,lng_for_end_location",
        "total_driving_distance": "estimated_distance",
        "total_driving_time": "estimated_driving_time",
        "total_visiting_time": "estimated_visiting_time",
        "vehicle_suggestion": "appropriate_vehicle",
        "stops": [
            {{
                "name": "stop_name",
                "type": "stop_type", 
                "coordinates": "lat,lng_for_stop",
                "description": "brief_description",
                "visiting_time": "x.x",
                "rating": "x.x"
            }},
            ...
        ],
        "cost_estimates": {{
            "accommodation": <integer_value>,
            "food": <integer_value>,
            "fuel": <integer_value>,
            "activities": <integer_value>
        }},
        "additional_recommendations": "extra_tips_and_suggestions"
    }}
    
    Only return the JSON object and nothing else.
    Ensure all suggestions are realistic and practical for {num_people} people with a {budget} budget.
    """