import google.generativeai as genai
import streamlit as st
import json
import time

# A curated list of supported languages for the UI
LANGUAGES = {
    "English": "en", "Spanish": "es", "French": "fr", "German": "de", 
    "Hindi": "hi", "Japanese": "ja", "Chinese (Simplified)": "zh-CN", 
    "Arabic": "ar", "Russian": "ru", "Portuguese": "pt", "Tamil": "ta"
}

def translate_text(text_to_translate, target_language, api_key):
    """Translates a single string of text using the Gemini API."""
    if not text_to_translate or not isinstance(text_to_translate, str):
        return text_to_translate
    try:
        genai.configure(api_key=api_key)
        # --- FIX: Updated model name from 'gemini-pro' to 'gemini-1.5-flash' ---
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Translate the following text into {target_language}. Return ONLY the translated text, with no extra explanations or context: '{text_to_translate}'"
        response = model.generate_content(prompt)
        time.sleep(1)  # Respect API rate limits
        return response.text.strip()
    except Exception as e:
        st.error(f"Translation Error: {e}")
        return text_to_translate # Return original text on failure

def translate_itinerary_data(trip_data, target_language, api_key):
    """Translates all user-facing text fields in the trip_data dictionary."""
    if not trip_data:
        return None
        
    # Create a deep copy to avoid modifying the original session state data
    translated_data = json.loads(json.dumps(trip_data))

    # Fields in the main dictionary to translate
    fields_to_translate = ['start', 'end', 'vehicle_suggestion', 'additional_recommendations']
    for field in fields_to_translate:
        if field in translated_data:
            translated_data[field] = translate_text(translated_data[field], target_language, api_key)
            
    # Translate fields within the 'stops' list of dictionaries
    if 'stops' in translated_data:
        for stop in translated_data['stops']:
            stop['name'] = translate_text(stop.get('name'), target_language, api_key)
            stop['type'] = translate_text(stop.get('type'), target_language, api_key)
            stop['description'] = translate_text(stop.get('description'), target_language, api_key)
            
    return translated_data