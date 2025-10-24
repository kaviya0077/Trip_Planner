import streamlit as st
import google.generativeai as genai
import os
import json
import re
from dotenv import load_dotenv
import joblib
import pandas as pd

# --- CORRECTED: Absolute imports from the 'app' package ---
from app.utils import find_best_model
from app.prompts import create_travel_prompt
from app.map_generator import create_static_map_url_with_labels, create_interactive_map, create_stop_map_html, get_map_as_base64, get_route_comparison
from app.voice_assistant import transcribe_audio
from app.translation import translate_itinerary_data, LANGUAGES
from app.database import (
    initialize_db,
    log_metric_event,
    save_feedback_to_db,
    save_trip_to_db,
    load_trip_from_db,
    get_all_data_for_dashboard
)

# --- DATABASE & WEB LIBRARIES ---
from jinja2 import Environment, FileSystemLoader

# --- LIBRARIES FOR DASHBOARD & INTERACTIVE MAP ---
import plotly.graph_objects as go
from streamlit_folium import folium_static
import plotly.express as px
from st_audiorec import st_audiorec

# Load environment variables
load_dotenv()

# --- DYNAMIC PAGE CONFIGURATION ---
query_params = st.query_params
if query_params.get("format") == "html":
    st.set_page_config(page_title="Printable Itinerary", layout="wide")
else:
    st.set_page_config(page_title="AI Travel Planner Pro", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")

# Initialize API keys from environment
if "GEMINI_API_KEY" not in st.session_state: st.session_state.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if "GOOGLE_MAPS_API_KEY" not in st.session_state: st.session_state.GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# --- CENTRALIZED FEATURE LIST ---
FEATURES_LIST = [
    {
        "icon": "🤖",
        "title": "AI-Powered Route Planning",
        "description": "Simply describe your desired trip in natural language. Our advanced AI understands your request to generate a fully customized itinerary just for you."
    },
    {
        "icon": "🗺️",
        "title": "Interactive & Static Maps",
        "description": "Visualize your entire journey on an interactive map, and see each recommended stop with its own dedicated static map, perfect for printing."
    },
    {
        "icon": "💾",
        "title": "Save & Share Your Itinerary",
        "description": "Your generated trip plans are saved locally. Generate a unique, clean URL to share your plan with friends and family."
    },
    {
        "icon": "💰",
        "title": "Dynamic Cost Estimation",
        "description": "Get a detailed breakdown of estimated costs, including accommodation, food, and activities, tailored to your budget level and number of travelers."
    },
    {
        "icon": "🎤",
        "title": "Voice-Activated Planning",
        "description": "No mood to type? Just speak your travel plans into our app, and our AI will transcribe and plan your journey for you, offering a hands-free experience."
    },
    {
        "icon": "🌐",
        "title": "Multi-Language Translation",
        "description": "Instantly translate your entire itinerary into one of many supported languages, making it easy to navigate and understand your plans abroad."
    }
]


# Custom CSS for Responsiveness and Theme Adaptability
st.markdown("""
<style>
    iframe { width: 100% !important; }
    /* Main styling - REMOVED fixed background to allow theme switching */
    .main-header { font-size: 3.5rem; font-weight: 800; background: red; -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; margin-bottom: 1rem; padding: 1rem; }
    .sub-header { font-size: 1.8rem; font-weight: 600; margin-bottom: 2rem; text-align: center; color: var(--text-color); /* MODIFIED: Use theme's text color */ }
    
    /* Cards with gradient backgrounds - These are okay as their text color contrasts with their specific background */
    .journey-card { background: linear-gradient(135deg, #7E57C2 0%, #5E35B1 100%); border-radius: 20px; padding: 25px; margin-bottom: 25px; box-shadow: 0 10px 30px rgba(126, 87, 194, 0.3); border: none; color: white; transition: transform 0.3s ease; min-height: 220px; }
    .inspiration-card { background: linear-gradient(135deg, #ff6a6a 0%, #ff8e53 50%, #ff758c 100%); border-radius: 20px; padding: 25px; margin-bottom: 25px; box-shadow: 0 10px 30px rgba(255, 99, 71, 0.3); border: none; color: white; transition: transform 0.3s ease; min-height: 220px; }
    .tips-card { background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); border-radius: 20px; padding: 25px; margin-bottom: 25px; box-shadow: 0 10px 30px rgba(121, 134, 203, 0.3); border: none; color: black; transition: transform 0.3s ease; min-height: 200px; font-size: 1.1rem; } /* NOTE: Black text on light gradient is fine */
    .journey-starter { background: linear-gradient(135deg, #434343 0%, #000000 100%); border-radius: 20px; padding: 40px; text-align: center; margin-top: 30px; color: white; box-shadow: 0 10px 30px rgba(25, 118, 210, 0.3); } /* MODIFIED: Darker gradient for better contrast */
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 16px; padding: 20px; text-align: center; color: white; margin: 15px 0; box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3); }
    .success-msg { background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); color: white; padding: 20px; border-radius: 16px; text-align: center; font-weight: 600; margin: 20px 0; }
    
    /* General content cards - MODIFIED to use theme variables */
    .stop-card, .feature-card {
        background: var(--secondary-background-color); /* Use theme's secondary background */
        color: var(--text-color); /* Use theme's text color */
        border-radius: 16px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        border-left: 5px solid #667eea;
        font-size: 1.1rem;
    }
    .feature-card { border-left-color: #ff8e53; min-height: 150px; }

    /* Other elements */
    .stButton>button { width: 100%; background: linear-gradient(135deg, #00bcd4 0%, #4caf50 100%); color: Black; border: none; border-radius: 12px; padding: 12px 24px; font-weight: bold; font-size: 1.1rem; transition: all 0.3s ease; }
    .stop-number { background: #667eea; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 1.2rem; margin-right: 15px; }
    
    /* Voice recorder responsiveness */
    div[data-testid="stAudioRec"] { width: 100%; }
    div[data-testid="stAudioRec"] > div > div:first-child { display: flex; justify-content: space-between; width: 100%; }
    div[data-testid="stAudioRec"] > div > div:first-child button { flex-grow: 1; margin: 0 5px; }
</style>
""", unsafe_allow_html=True)


# --- MODEL LOADING AND PREDICTION ---
@st.cache_resource
def load_model():
    """Load the trained model and columns from the 'models' directory."""
    try:
        model = joblib.load('models/cost_model.joblib')
        with open('models/training_columns.json', 'r') as f:
            columns = json.load(f)
        return model, columns
    except FileNotFoundError:
        return None, None

def predict_costs(trip_data, model, training_columns):
    """Prepares trip data and predicts costs using the trained model."""
    if not model or not training_columns:
        st.warning("Cost model is not available. Skipping custom cost prediction.")
        return trip_data.get("cost_estimates")

    try:
        features = {
            "start_city": trip_data.get("start"),
            "end_city": trip_data.get("end"),
            "num_travelers": trip_data.get("num_people"),
            "budget_level": trip_data.get("budget_level"),
            "vehicle_type": trip_data.get("vehicle_suggestion"),
            "total_distance_km": float(re.findall(r"[\d\.]+", str(trip_data.get("total_driving_distance", "0")))[0]),
            "total_driving_hours": float(re.findall(r"[\d\.]+", str(trip_data.get("total_driving_time", "0")))[0]),
            "total_visiting_hours": float(re.findall(r"[\d\.]+", str(trip_data.get("total_visiting_time", "0")))[0]),
        }
        input_df = pd.DataFrame([features])
        input_encoded = pd.get_dummies(input_df)
        input_encoded.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in input_encoded.columns]
        input_aligned = input_encoded.reindex(columns=training_columns, fill_value=0)
        prediction = model.predict(input_aligned)[0]
        cost_estimates = {
            "accommodation": abs(int(prediction[0])),
            "food": abs(int(prediction[1])),
            "fuel": abs(int(prediction[2])),
            "activities": abs(int(prediction[3]))
        }
        return cost_estimates
    except Exception as e:
        st.error(f"Error during cost prediction: {e}")
        return trip_data.get("cost_estimates")

# --- HTML RENDERING FUNCTION ---
def render_html_itinerary(trip_data, api_key, features):
    env = Environment(loader=FileSystemLoader('app/template'))
    template = env.get_template('template.html')
    map_image_data = get_map_as_base64(trip_data, api_key)
    legend_items = []
    if trip_data:
        start_coords = trip_data.get("start_coordinates")
        end_coords = trip_data.get("end_coordinates")
        legend_items.append({'label': 'S', 'name': trip_data.get('start', 'Start'), 'color': '#4CAF50'})
        intermediate_stops = [s for s in trip_data.get("stops", []) if s.get("coordinates") and s.get("coordinates") != start_coords and s.get("coordinates") != end_coords]
        for i, stop in enumerate(intermediate_stops):
            legend_items.append({'label': str(i+1), 'name': stop.get('name'), 'color': '#f44336'})
        legend_items.append({'label': 'E', 'name': trip_data.get('end', 'End'), 'color': '#2196F3'})
        
    return template.render(
        trip=trip_data, 
        map_image=map_image_data, 
        legend=legend_items, 
        features=features
    )

# --- LINK HANDLING & CORE AI FUNCTIONS ---
def generate_shareable_link(trip_id):
    base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:5000")
    return f"{base_url}/itinerary/{trip_id}"

def load_trip_from_query_params():
    query_params = st.query_params
    if "id" in query_params:
        trip_id = query_params["id"]; trip_data = load_trip_from_db(trip_id)
        if trip_data: st.session_state.trip_data = trip_data; st.session_state.plan_generated = True; st.session_state.loaded_from_link = True; st.session_state.trip_id = trip_id; st.query_params.clear()

def is_trip_data_valid(trip):
    if not (trip.get("start_coordinates") and trip.get("end_coordinates")): return False
    for stop in trip.get("stops", []):
        if not stop.get("coordinates"): return False
    return True

def get_trip_recommendations(prompt, api_key, num_people=2, budget="Moderate"):
    try:
        if not api_key: return {"error": "API key not provided"}
        genai.configure(api_key=api_key); model_name = find_best_model(api_key)
        if not model_name: return {"error": "No suitable model found"}
        model = genai.GenerativeModel(model_name); 
        enhanced_prompt = create_travel_prompt(prompt, num_people, budget)
        response = model.generate_content(enhanced_prompt)
        parsed_data, success = parse_trip_response(response.text)
        log_metric_event("ai_parse_success" if success else "ai_parse_failure")
        if success and is_trip_data_valid(parsed_data): log_metric_event("ai_valid_output")
        elif success: log_metric_event("ai_invalid_output")
        return parsed_data, success
    except Exception as e: log_metric_event("ai_parse_failure"); return {"error": f"Error getting LLM response: {str(e)}"}, False

def parse_trip_response(response_text):
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        return json.loads(json_match.group()), True
    except:
        return get_fallback_trip_data(), False

def get_fallback_trip_data():
    return { "start": "New Delhi", "end": "Agra", "start_coordinates": "28.6139,77.2090", "end_coordinates": "27.1767,78.0081", "total_driving_distance": "240 km", "total_driving_time": "4 hours", "stops": [{"name": "Mathura", "coordinates": "27.4924,77.6737"}], "additional_recommendations": "Start early.", "cost_estimates": {"accommodation": 2000, "food": 1000, "activities": 500} }

# --- App Initialization ---
if "current_prompt" not in st.session_state: st.session_state.current_prompt = ""
if "loaded_from_link" not in st.session_state: st.session_state.loaded_from_link = False
if "feedback_given" not in st.session_state: st.session_state.feedback_given = False
initialize_db()
if not st.session_state.loaded_from_link and 'plan_generated' not in st.session_state: load_trip_from_query_params()

# --- Main App Logic Begins ---
trip_id_param = query_params.get("id"); output_format = query_params.get("format")

# --- HANDLER FOR SPECIAL LINKS (HTML ONLY) ---
if trip_id_param and output_format == "html":
    trip_data = load_trip_from_db(trip_id_param)
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if trip_data and api_key:
        map_image_data_check = get_map_as_base64(trip_data, api_key)
        if not map_image_data_check:
                st.warning("⚠️ Could not generate the static map image. This is usually due to an API key issue. Please check that the 'Maps Static API' is enabled in your Google Cloud Console and that billing is active on your account.")
        html_content = render_html_itinerary(trip_data, api_key, FEATURES_LIST)
        file_name = f"itinerary_{trip_data.get('start', 'trip').replace(' ', '_')}_to_{trip_data.get('end', 'itinerary').replace(' ', '_')}.html"
        st.download_button(label="📥 Download Itinerary Page", data=html_content, file_name=file_name, mime='text/html')
        st.components.v1.html(html_content, height=1000, scrolling=True)
        st.stop()
    else:
        if not api_key: st.error("Google Maps API key is not configured on the server.")
        if not trip_data: st.error("Trip ID not found in the database.")
        st.stop()
        
# --- Main App Rendering ---
with st.sidebar:
    st.markdown('<h3>🧭 Navigation</h3>', unsafe_allow_html=True)
    page = st.radio("Select a page:",
                    ["✈️ Plan Your Trip", "📊 Trip Details", "💡 Travel Tips", "🚀 Features Overview", "📈 Admin Dashboard"],
                    label_visibility="collapsed")

st.markdown('<h1 class="main-header">✈️ AI Travel Planner Pro</h1>', unsafe_allow_html=True)
if page == "✈️ Plan Your Trip":
    st.markdown('<div class="sub-header">Plan Your Perfect Journey with AI Intelligence</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("""<div class="journey-card"><h3>🎯 Describe Your Journey</h3><p>Tell us about your dream trip and we'll create the perfect itinerary!</p></div>""", unsafe_allow_html=True)
        st.write("🎤 **Or, tell us your plan:**")
        wav_audio_data = st_audiorec()
        if wav_audio_data is not None:
            st.audio(wav_audio_data, format='audio/wav')
            with st.spinner("Transcribing your voice..."):
                transcript, success = transcribe_audio(wav_audio_data)
                if success:
                    st.session_state.current_prompt = transcript
                    st.success("Your speech has been converted to text below.")
                    st.rerun()
                else:
                    st.error(transcript)
        prompt = st.text_area("**Your Trip Details:**", value=st.session_state.current_prompt, height=100, placeholder="Example: I want to travel from Delhi to Agra...", label_visibility="collapsed")

    with col2:
        st.markdown("""<div class="inspiration-card"><h3>💡 Trip Inspiration</h3><p>Need ideas? Try one of these popular routes:</p></div>""", unsafe_allow_html=True)
        prompt_suggestions = [
            {"icon": "🕉️", "text": "Spiritual journey from Varanasi to Rishikesh"}, 
            {"icon": "🏖️", "text": "Beach hopping itinerary from Mumbai to Goa"},
            {"icon": "👑", "text": "Coastal ride from Chennai to Kanyakumari"},
            {"icon": "🏜️", "text": "Explore the Rann of Kutch from Ahmedabad"},
            {"icon": "🌧️", "text": "Monsoon magic from Shillong to Cherrapunji"},
            {"icon": "⛰️", "text": "Himalayan adventure from Manali to Leh"}
        ]
        for i, suggestion in enumerate(prompt_suggestions):
            if st.button(f"{suggestion['icon']} {suggestion['text']}", key=f"prompt_{i}", use_container_width=True): 
                st.session_state.suggestion_clicked = suggestion['text']
    
    with st.expander("⚙️ Advanced Settings", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a: num_people = st.number_input("👥 Number of Travelers", 1, 10, 2)
        with col_b: budget = st.selectbox("💰 Budget Level", ["Budget", "Moderate", "Luxury"], index=1)

    if 'suggestion_clicked' in st.session_state: 
        st.session_state.current_prompt = st.session_state.suggestion_clicked; st.session_state.generate_clicked = True; del st.session_state.suggestion_clicked
    
    if st.button("🚀 Generate Trip Itinerary", type="primary", use_container_width=True):
        st.session_state.generate_clicked = True; st.session_state.current_prompt = prompt; st.session_state.loaded_from_link = False; st.session_state.feedback_given = False

    cost_model, training_cols = load_model()

    if 'generate_clicked' in st.session_state and st.session_state.generate_clicked and not st.session_state.loaded_from_link:
        log_metric_event("generate_click")
        prompt_to_run = st.session_state.current_prompt
        if not prompt_to_run: 
            st.error("Please describe your trip first!")
        elif cost_model is None:
            st.error("Cost estimation model not found. Please run train_model.py first.")
        else:
            with st.spinner("🧠 AI is crafting your itinerary... (Step 1/2)"):
                trip_data, success = get_trip_recommendations(prompt_to_run, st.session_state.GEMINI_API_KEY, num_people, budget)
            
            if success and "error" not in trip_data:
                with st.spinner("🤖 Calculating costs with trained model... (Step 2/2)"):
                    trip_data['num_people'] = num_people
                    trip_data['budget_level'] = budget
                    model_costs = predict_costs(trip_data, cost_model, training_cols)
                    if model_costs:
                        trip_data['cost_estimates'] = model_costs
                    prompt_length = len(prompt_to_run.split())
                    trip_id = save_trip_to_db(trip_data, prompt_length)
                    if trip_id: 
                        st.session_state.trip_data = trip_data
                        st.session_state.plan_generated = True
                        st.session_state.trip_id = trip_id 
                        st.success("Trip plan and cost estimation complete!")
                    else: 
                        st.error("Could not save the trip plan.")
            elif "error" in trip_data:
                st.error(trip_data["error"])
        st.session_state.generate_clicked = False
    
    if 'plan_generated' in st.session_state and st.session_state.plan_generated:
        trip_data = st.session_state.trip_data
        st.markdown('<div class="success-msg">✅ Your Travel Plan is Ready!</div>', unsafe_allow_html=True)
        st.markdown(f'<h2 style="text-align: center; color: white;">{trip_data["start"]} to {trip_data["end"]}</h2>', unsafe_allow_html=True)
        
        # --- NOTE: The maps are placed in the same column layout, making them the same width by design ---
        map_tab1, map_tab2 = st.tabs(["📍 Interactive Map", "🗺️ Static Map"])
        
        with map_tab1:
            map_col1, map_col2, map_col3 = st.columns([1, 5, 1])
            with map_col2:
                interactive_map = create_interactive_map(trip_data)
                if interactive_map:
                    folium_static(interactive_map, height=450)
                else:
                    st.warning("Not enough data to render an interactive map.")
        with map_tab2:
            map_col1, map_col2, map_col3 = st.columns([1, 5, 1])
            with map_col2:
                static_map_url = create_static_map_url_with_labels(trip_data, st.session_state.GOOGLE_MAPS_API_KEY)
                if static_map_url:
                    st.image(static_map_url, use_container_width=True)
                else:
                    st.warning("Could not generate static map. Please check API Key and coordinates.")
        
        if 'trip_id' in st.session_state:
            clean_link = generate_shareable_link(st.session_state.trip_id)
            # --- MODIFIED: Added inline style to change the link's font color ---
            st.markdown(
                f"""
                <div style="max-width: 600px; margin: 40px auto; text-align: center;">
                    <h3>Share Your Portable Itinerary</h3>
                    <div style="background-color: #eef2ff; color: #1E90FF; padding: 1rem; border-radius: 0.5rem; font-family: monospace; overflow-wrap: break-word; font-weight: bold;">
                        {clean_link}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown("---")
        
        # Create a centered button for route optimization
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            if st.button("🚀 Optimize Route with Live Traffic", use_container_width=True, type="primary"):
                with st.spinner("Analyzing real-time traffic and finding the fastest route..."):
                    
                    comparison_result = get_route_comparison(
                        st.session_state.trip_data,
                        st.session_state.GOOGLE_MAPS_API_KEY
                    )
                    
                    if comparison_result.get("success"):
                        time_saved_seconds = comparison_result["time_saved_seconds"]
                        if time_saved_seconds > 0:
                            st.session_state.trip_data['stops'] = comparison_result['reordered_stops']
                            optimized_duration_seconds = comparison_result['optimized_duration_seconds']
                            drive_hours = optimized_duration_seconds / 3600
                            st.session_state.trip_data['total_driving_time'] = f"{drive_hours:.1f} hours (in current traffic)"
                        
                            time_saved_minutes = time_saved_seconds / 60
                            original_duration_hours = comparison_result['original_duration_seconds'] / 3600
                        
                            st.session_state.optimization_success_message = (
                                f"✅ **Route Optimized!** By reordering your stops, you can save "
                                f"**{time_saved_minutes:.0f} minutes**. "
                                f"The original route would take {original_duration_hours:.1f} hours, "
                                f"but the optimized route takes only **{drive_hours:.1f} hours**."
                            )
                            st.rerun()
                        
                        else:
                            st.info("✅ Your current itinerary is already the fastest route. No changes needed!")
                    else:
                        st.error(f"Optimization Failed: {comparison_result.get('error', 'Unknown error')}")

        if st.session_state.get("optimization_success_message"):
            st.success(st.session_state.optimization_success_message)
            del st.session_state.optimization_success_message
        
        if not st.session_state.get('feedback_given', False):
            st.markdown("---")
            with st.container():
                st.markdown("<p style='text-align: center; font-weight: bold;'>Was this itinerary helpful?</p>", unsafe_allow_html=True)
                _, btn1_col, btn2_col, _ = st.columns([0.35, 0.15, 0.15, 0.35]) 
                with btn1_col:
                    if st.button("👍 Yes", use_container_width=True, key="feedback_yes"):
                        save_feedback_to_db(st.session_state.trip_id, 1); st.rerun()
                with btn2_col:
                    if st.button("👎 No", use_container_width=True, key="feedback_no"):
                        save_feedback_to_db(st.session_state.trip_id, -1); st.rerun()
        else:
            st.markdown("---")
            st.markdown("""<div style="background-color: #e8f5e9; color: #2e7d32; padding: 20px; border-radius: 10px; text-align: center; max-width: 300px; margin: 20px auto;"><strong>Thanks for your feedback!</strong></div>""", unsafe_allow_html=True)
            
    elif not st.session_state.get('plan_generated', False): 
        st.markdown("""<div class="journey-starter"><h3>🌟 Your Journey Starts Here</h3><p>Describe your dream trip or choose from our suggestions to begin planning!</p><div style="font-size: 4rem; margin: 20px 0;">✈️</div></div>""", unsafe_allow_html=True)

elif page == "📊 Trip Details":
    st.subheader("📊 Your Trip at a Glance")
    if 'plan_generated' in st.session_state and st.session_state.plan_generated:
        
        display_data = st.session_state.get('translated_trip_data', st.session_state.trip_data)

        st.markdown("---")
        with st.container(border=True):
            st.markdown("#### 🌐 Translate Itinerary")
            trans_col1, trans_col2, trans_col3 = st.columns([2,1,1])
            with trans_col1:
                target_lang_name = st.selectbox("Select Language", options=LANGUAGES.keys())
            with trans_col2:
                if st.button("Translate", use_container_width=True):
                    with st.spinner(f"Translating to {target_lang_name}..."):
                        st.session_state.translated_trip_data = translate_itinerary_data(st.session_state.trip_data, target_lang_name, st.session_state.GEMINI_API_KEY)
                        st.rerun()
            with trans_col3:
                if 'translated_trip_data' in st.session_state:
                    if st.button("Show Original", use_container_width=True):
                        del st.session_state.translated_trip_data
                        st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        
        overall_cost = 0
        per_person_cost = 0
        cost_estimates = display_data.get("cost_estimates")

        if cost_estimates:
            overall_cost = sum(cost_estimates.values())
            num_people = display_data.get("num_people", 1)
            if num_people > 0:
                per_person_cost = overall_cost / num_people
        
        details_html = f"""
        <div style="display: flex; justify-content: center;">
            <table style="width: 80%; border-collapse: collapse; margin-top: 20px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-radius: 10px; overflow: hidden; font-size: 1.1rem;">
                <thead style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                    <tr><th style="padding: 15px; text-align: left;">Metric</th><th style="padding: 15px; text-align: left;">Details</th></tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid #ddd;"><td style="padding: 15px; font-weight: bold;">Total Stops</td><td style="padding: 15px;">{len(display_data["stops"])}</td></tr>
                    <tr style="border-bottom: 1px solid #ddd;"><td style="padding: 15px; font-weight: bold;">Total Distance</td><td style="padding: 15px;">{display_data.get("total_driving_distance", "N/A")}</td></tr>
                    <tr style="border-bottom: 1px solid #ddd;"><td style="padding: 15px; font-weight: bold;">Driving Time</td><td style="padding: 15px;">{display_data.get("total_driving_time", "N/A")}</td></tr>
                    <tr style="border-bottom: 1px solid #ddd;"><td style="padding: 15px; font-weight: bold;">Visiting Time</td><td style="padding: 15px;">{display_data.get("total_visiting_time", "N/A")}</td></tr>
                    <tr style="border-bottom: 1px solid #ddd;"><td style="padding: 15px; font-weight: bold;">Suggested Vehicle</td><td style="padding: 15px;">{display_data.get("vehicle_suggestion", "Car")}</td></tr>
                    <tr style="border-bottom: 1px solid #ddd;"><td style="padding: 15px; font-weight: bold;">Overall Estimated Budget</td><td style="padding: 15px;">₹ {overall_cost:,.0f}</td></tr>
                    <tr><td style="padding: 15px; font-weight: bold;">Est. Budget (Per Person)</td><td style="padding: 15px;">₹ {per_person_cost:,.0f}</td></tr>
                </tbody>
            </table>
        </div>
        """
        st.markdown(details_html, unsafe_allow_html=True)

        if cost_estimates:
            st.markdown("---")
            st.markdown("<h4 style='text-align: center; color: white;'>💵 Detailed Cost Breakdown</h4>", unsafe_allow_html=True)
            num_items = len(cost_estimates)
            cols = st.columns(num_items if num_items > 0 else 1)
            if num_items > 0:
                for i, (key, value) in enumerate(cost_estimates.items()):
                    with cols[i]:
                        st.metric(label=key.title(), value=f"₹ {value:,.0f}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### 🛑 Recommended Stops in {display_data.get('start', '')}")
        for i, stop in enumerate(display_data["stops"]):
            map_html = "<p>No map available</p>"
            if stop.get("coordinates"): map_html = create_stop_map_html(stop.get("coordinates", ""), stop["name"], st.session_state.GOOGLE_MAPS_API_KEY)
            # --- MODIFIED: Changed the h3 color for the stop name ---
            st.markdown(f"""<div class="stop-card"><div style="display: flex; align-items: center; margin-bottom: 15px;"><div class="stop-number">{i+1}</div><h3 style="margin: 0; color: #5E35B1;">{stop["name"]}</h3></div><div style="display: grid; grid-template-columns: 2fr 1fr; gap: 20px; align-items: start;"><div><p><strong>Type:</strong> {stop.get('type', 'N/A').title()}</p><p><strong>Time Needed:</strong> {stop.get("visiting_time", 0.5)} hours</p><p><strong>Rating:</strong> ⭐ {stop.get('rating', 'N/A')}/5</p><p>{stop.get('description', 'No description available')}</p></div><div style="height: 200px; overflow: hidden; border-radius: 12px;">{map_html}</div></div></div>""", unsafe_allow_html=True)
    else: 
        st.info("Generate a travel plan first to see the trip details here.")

elif page == "💡 Travel Tips":
    st.subheader("💡 Travel Tips & Recommendations")
    if 'plan_generated' in st.session_state and st.session_state.plan_generated:
        display_data = st.session_state.get('translated_trip_data', st.session_state.trip_data)
        st.markdown(f"""<div class="tips-card">{display_data.get("additional_recommendations", "")}</div>""", unsafe_allow_html=True)
    else: 
        st.info("Generate a travel plan first to see travel tips here.")

elif page == "🚀 Features Overview":
    st.subheader("🚀 Features Overview")
    for feature in FEATURES_LIST:
        st.markdown(
            f"""
            <div class="feature-card">
                <h3>{feature['icon']} {feature['title']}</h3>
                <p>{feature['description']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

elif page == "📈 Admin Dashboard":
    st.header("📈 Application Analytics Dashboard")
    st.info("This dashboard provides insights into user behavior and AI performance.")
    
    try:
        metrics_df, feedback_df, trips_df = get_all_data_for_dashboard()
        if trips_df.empty:
            st.warning("No data available to display analytics. Please generate some trips first.")
            st.stop()

        # --- Section 1: Key Performance Indicators (KPIs) ---
        st.markdown("---")
        st.subheader("✨ Key Performance Indicators")

        kpi_cols = st.columns(4)
        
        # KPI 1: Total Trips Generated
        total_trips = len(trips_df)
        kpi_cols[0].metric("Total Trips Generated", f"{total_trips}")

        # KPI 2: User Engagement Rate
        engagement_rate = (len(feedback_df) / total_trips) * 100 if total_trips > 0 else 0
        kpi_cols[1].metric("User Engagement Rate", f"{engagement_rate:.1f}%")

        # KPI 3: Itinerary "Helpful" Score
        helpful_score = 0
        if not feedback_df.empty:
            helpful_votes = (feedback_df['rating'] == 1).sum()
            total_votes = len(feedback_df)
            helpful_score = (helpful_votes / total_votes) * 100 if total_votes > 0 else 0
        kpi_cols[2].metric("Itinerary 'Helpful' Score", f"{helpful_score:.1f}%")

        # KPI 4: AI Reliability Score
        reliability_score = 0
        if not metrics_df.empty:
            generate_clicks = (metrics_df['event_type'] == 'generate_click').sum()
            saved_trips = (metrics_df['event_type'] == 'trip_saved').sum()
            reliability_score = (saved_trips / generate_clicks) * 100 if generate_clicks > 0 else 0
        kpi_cols[3].metric("AI Reliability Score", f"{reliability_score:.1f}%")


        # --- Section 2: Exploratory Data Analysis (EDA) ---
        st.markdown("---")
        st.subheader("📊 Exploratory Data Analysis")
        
        def safe_load_json(data):
            if isinstance(data, dict):
                return data
            elif isinstance(data, str):
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    return {}
            else:
                return {}

        trips_df['trip_data'] = trips_df['trip_data'].apply(safe_load_json)

        # Chart 1: Most Popular Travel Routes
        with st.container(border=True):
            st.markdown("#### 🗺️ Most Popular Travel Routes")
            trips_df['start_city'] = trips_df['trip_data'].apply(lambda x: x.get('start', 'Unknown'))
            trips_df['end_city'] = trips_df['trip_data'].apply(lambda x: x.get('end', 'Unknown'))
            trips_df['route'] = trips_df['start_city'] + " → " + trips_df['end_city']
            
            top_routes = trips_df['route'].value_counts().nlargest(10).sort_values(ascending=True)
            
            fig = px.bar(top_routes, x=top_routes.values, y=top_routes.index, orientation='h', 
                            title="Top 10 Most Generated Routes", labels={'x': 'Number of Trips', 'y': 'Route'})
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

        # Chart 2: Cost Estimation Analysis by Budget Level
        with st.container(border=True):
            st.markdown("#### 💰 Cost Estimation Analysis by Budget Level")
            cost_data = []
            for index, row in trips_df.iterrows():
                trip = row['trip_data']
                budget = trip.get('budget_level', 'Unknown')
                if 'cost_estimates' in trip and trip['cost_estimates']:
                    overall_cost = sum(trip['cost_estimates'].values())
                    cost_data.append({'budget_level': budget, 'overall_cost': overall_cost})
            
            if cost_data:
                cost_df = pd.DataFrame(cost_data)
                fig = px.box(cost_df, x='budget_level', y='overall_cost', 
                                title="Distribution of Overall Trip Cost by Budget Level",
                                labels={'budget_level': 'Budget Level', 'overall_cost': 'Overall Estimated Cost (₹)'},
                                category_orders={"budget_level": ["Budget", "Moderate", "Luxury"]},
                                color='budget_level')
                st.plotly_chart(fig, use_container_width=True)
                st.caption("_**Interpretation:** This chart evaluates the cost model. The boxes for each budget level should be distinct and show minimal overlap, proving the model can differentiate costs effectively._")
            else:
                st.info("No cost estimation data available to display this chart.")

    except Exception as e:
        st.error(f"Could not load analytics data. This might be because no data has been generated yet. Error: {e}")