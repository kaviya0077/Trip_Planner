import os
import json
import re
import pandas as pd
from dotenv import load_dotenv
import random
import time

# IMPORTANT: We need to import the functions from your existing app
from app.prompts import create_travel_prompt
from app.utils import find_best_model
import google.generativeai as genai

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

genai.configure(api_key=GEMINI_API_KEY)

# --- Define the parameters to create diverse prompts ---
CITIES = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Varanasi", "Goa", "Kochi", "Rishikesh", "Manali"]
INTERESTS = ["historical landmarks", "amazing food", "adventure sports", "spiritual sites", "beautiful nature", "shopping districts"]
BUDGETS = ["Budget", "Moderate", "Luxury"]
NUM_PEOPLE = [1, 2, 4]

def parse_trip_response(response_text):
    """A simplified parser for this script."""
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group()), True
        return None, False
    except (json.JSONDecodeError, AttributeError):
        return None, False

def generate_single_trip_data(model):
    """Generates one random trip and returns the structured data."""
    start_city, end_city = random.sample(CITIES, 2)
    interest = random.choice(INTERESTS)
    budget = random.choice(BUDGETS)
    people = random.choice(NUM_PEOPLE)
    
    prompt_text = f"A trip from {start_city} to {end_city} focusing on {interest}."
    
    print(f"Generating: {prompt_text} for {people} people on a {budget} budget...")

    try:
        # Create the full prompt
        enhanced_prompt = create_travel_prompt(prompt_text, people, budget, optimize_for_time=random.choice([True, False]))
        
        # Get response from the LLM
        response = model.generate_content(enhanced_prompt)
        
        # Parse the response
        parsed_data, success = parse_trip_response(response.text)
        
        if success and parsed_data and "cost_estimates" in parsed_data:
            # Extract key features to create a flat dictionary
            features = {
                "start_city": parsed_data.get("start"),
                "end_city": parsed_data.get("end"),
                "num_travelers": people,
                "budget_level": budget,
                "vehicle_type": parsed_data.get("vehicle_suggestion"),
                "total_distance_km": float(re.findall(r"[\d\.]+", str(parsed_data.get("total_driving_distance", "0")))[0]),
                "total_driving_hours": float(re.findall(r"[\d\.]+", str(parsed_data.get("total_driving_time", "0")))[0]),
                "total_visiting_hours": float(re.findall(r"[\d\.]+", str(parsed_data.get("total_visiting_time", "0")))[0]),
                "accommodation_cost": parsed_data["cost_estimates"].get("accommodation"),
                "food_cost": parsed_data["cost_estimates"].get("food"),
                "fuel_cost": parsed_data["cost_estimates"].get("fuel"),
                "activities_cost": parsed_data["cost_estimates"].get("activities")
            }
            return features
        else:
            print("--> Failed to parse or missing cost estimates.")
            return None
    except Exception as e:
        print(f"--> ERROR during generation: {e}")
        return None

def main():
    """Main function to generate and save the dataset."""
    print("Finding the best model...")
    model_name = find_best_model(GEMINI_API_KEY)
    if not model_name:
        print("Could not find a suitable model. Exiting.")
        return
    
    model = genai.GenerativeModel(model_name)
    print(f"Using model: {model_name}")

    all_trips_data = []
    num_trips_to_generate = 500  # You can start with a smaller number like 100

    for i in range(num_trips_to_generate):
        print(f"\n--- Trip {i+1}/{num_trips_to_generate} ---")
        trip_features = generate_single_trip_data(model)
        if trip_features:
            all_trips_data.append(trip_features)
        
        # Be respectful of API rate limits
        time.sleep(2) # Wait 2 seconds between requests

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(all_trips_data)
    
    # Save the dataset to a CSV file
    output_path = "data/trips_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSuccessfully generated and saved {len(df)} trips to {output_path}")

if __name__ == "__main__":
    main()