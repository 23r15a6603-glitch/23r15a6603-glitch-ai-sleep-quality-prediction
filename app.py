import os
import time
import joblib
import pandas as pd
import streamlit as st
import numpy as np

# ------------------------------
# ✅ Secure API Key Handling
# ------------------------------
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
    except:
        api_key = None

if api_key:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        gemini_available = True
    except ImportError:
        st.warning("⚠️ Google Generative AI package not installed. Run: pip install google-generativeai")
        gemini_available = False
else:
    st.warning("⚠️ Gemini API Key not found. Please set GEMINI_API_KEY in .env (local) or Streamlit Secrets (cloud).")
    gemini_available = False

# ------------------------------
# ✅ Load ML model and scaler safely
# ------------------------------
MODEL_PATH = "xgb_sleep_quality_model.pkl"
SCALER_PATH = "scaler_sleep_quality.pkl"

model, scaler = None, None
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        st.error(f"❌ Model files not found. Please ensure {MODEL_PATH} and {SCALER_PATH} exist.")
except Exception as e:
    st.error(f"❌ Error loading model/scaler: {str(e)}")

# ------------------------------
# Streamlit Config
# ------------------------------
st.set_page_config(page_title="AI-Based Sleep Quality Prediction", layout="wide")

# ------------------------------
# 🌈 Fixed Styling for Better Visibility
# ------------------------------
st.markdown(
    """
    <style>
    /* Background with better contrast */
    .stApp {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main content container */
    .main {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
    }

    /* Titles with better contrast */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 600;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.7);
    }

    /* All text elements with proper contrast */
    .stMarkdown, p, label, .stText, .stNumberInput label, 
    .stSelectbox label, .stSlider label, .stTextInput label {
        color: #000000 !important;
        font-size: 16px;
        font-weight: 500;
    }

    /* Input fields with better visibility */
    .stTextInput input, .stNumberInput input, .stSelectbox select, 
    .stSlider div, .stTextInput>div>div, .stNumberInput>div>div {
        background: #ffffff !important;
        color: #000000 !important;
        border-radius: 8px;
        padding: 8px;
        border: 1px solid #ccc;
    }

    /* Buttons with better contrast */
    div.stButton > button {
        background-color: #ff6b35;
        color: white;
        border-radius: 12px;
        font-size: 16px;
        padding: 10px 20px;
        border: none;
        transition: 0.3s;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    div.stButton > button:hover {
        background-color: #e65c00;
        transform: scale(1.05);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }

    /* Predictor Card with better contrast */
    .predictor-card {
        background: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        margin-bottom: 25px;
        border: 1px solid #e0e0e0;
    }

    /* Chat container with better contrast */
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 15px;
        border-radius: 15px;
        background: #f9f9f9;
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
    }
    
    /* Chat messages with better contrast */
    .user-msg {
        background: #ff6b35;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 0px 18px;
        margin: 10px 0;
        text-align: right;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .bot-msg {
        background: #415a77;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 0px;
        margin: 10px 0;
        text-align: left;
        max-width: 80%;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    
    /* Form labels with better visibility */
    .stForm {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #415a77 0%, #1b263b 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    
    /* Success and error messages */
    .stSuccess {
        background-color: #4caf50;
        color: white;
    }
    .stError {
        background-color: #f44336;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# Main Title
# ------------------------------
st.markdown("<div class='section-header'>", unsafe_allow_html=True)
st.title("🌙 AI-Based Sleep Quality Prediction")
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Predictor Section (Card UI)
# ------------------------------
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.markdown('<div class="predictor-card">', unsafe_allow_html=True)
st.subheader("🔍 Sleep Quality Predictor")

with st.form("sleep_form"):
    st.write("Enter your Health and Lifestyle Factors")
    colA, colB, colC = st.columns(3)

    with colA:
        age = st.number_input("Age", 10, 100, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        sleep_duration = st.slider("Sleep Duration (hrs)", 0.0, 12.0, 7.0, 0.5)
        activity = st.slider("Physical Activity (mins/day)", 0, 180, 30)

    with colB:
        stress = st.slider("Stress Level (1–10)", 1, 10, 5)
        caffeine = st.slider("Caffeine Intake (cups/day)", 0, 10, 1)
        alcohol = st.slider("Alcohol Intake (units/day)", 0, 10, 0)
        smoker = st.selectbox("Do you smoke?", ["No", "Yes"])

    with colC:
        heart_rate = st.number_input("Heart Rate (bpm)", 40, 140, 70)
        screen_time = st.slider("Screen Time Before Bed (hrs)", 0.0, 10.0, 2.0, 0.5)
        history = st.selectbox("Sleep Disorder History", ["No", "Yes"])
        bmi = st.number_input("BMI", 10.0, 50.0, 22.0)

    colD, colE = st.columns(2)
    with colD:
        wake_consistency = st.selectbox("Wake-up Consistency", ["Consistent", "Inconsistent"])
    with colE:
        env_score = st.slider("Sleep Environment Score (1–10)", 1, 10, 7)
        water = st.slider("Daily Water Intake (litres)", 0.0, 5.0, 2.0, 0.5)

    submitted = st.form_submit_button("Predict Sleep Quality")

if submitted:
    if model and scaler:
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [1 if gender == "Male" else 0],
            'Sleep Duration (hrs)': [sleep_duration],
            'Physical Activity (mins/day)': [activity],
            'Stress Level (1–10)': [stress],
            'Caffeine Intake (cups/day)': [caffeine],
            'Alcohol Intake (units/day)': [alcohol],
            'Smoking': [1 if smoker == "Yes" else 0],
            'Heart Rate (bpm)': [heart_rate],
            'Screen Time Before Bed (hrs)': [screen_time],
            'Sleep Disorder History': [1 if history == "Yes" else 0],
            'BMI': [bmi],
            'Wake-up Consistency': [1 if wake_consistency == "Consistent" else 0],
            'Sleep Environment Score (1–10)': [env_score],
            'Daily Water Intake (litres)': [water]
        })

        try:
            scaled_input = scaler.transform(input_data)
            prediction = model.predict(scaled_input)[0]
            label_map = {0: 'Poor', 1: 'Fair', 2: 'Good'}
            result = label_map.get(prediction, "Unknown")

            st.success(f"🌙 Predicted Sleep Quality: {result}")
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
    else:
        st.error("⚠️ Prediction unavailable. Model or scaler missing.")

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------
# Chatbot Section (Bottom)
# ------------------------------
st.markdown("<div class='section-header'>", unsafe_allow_html=True)
st.subheader("💬 Sleep AI Chatbot")
st.markdown("</div>", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask your question here:", key="chat_input")

col1, col2 = st.columns([1, 1])
with col1:
    send = st.button("Send")
with col2:
    clear = st.button("Clear Chat")

if clear:
    st.session_state.chat_history = []
    st.rerun()

if send and user_input.strip():
    if not gemini_available:
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", "⚠️ Gemini API is not configured. Please check your API key."))
    else:
        try:
            st.session_state.chat_history.append(("You", user_input))
            
            # Prepare history in the format Gemini expects
            history_for_api = []
            for role, msg in st.session_state.chat_history[:-1]:  # All but the latest user message
                history_for_api.append({
                    "role": "user" if role == "You" else "model",
                    "parts": [msg]
                })
            
            # Generate response
            model_name = "gemini-pro"  # Using the stable model
            chat_model = genai.GenerativeModel(model_name)
            chat = chat_model.start_chat(history=history_for_api)
            
            with st.spinner("Thinking..."):
                response = chat.send_message(user_input)
                bot_response = response.text
            
            st.session_state.chat_history.append(("Bot", bot_response))
            
        except Exception as e:
            error_msg = f"⚠️ Error: {str(e)}"
            if "quota" in str(e).lower():
                error_msg = "⚠️ API quota exceeded. Please try again later."
            st.session_state.chat_history.append(("Bot", error_msg))

# Display chat history with styled bubbles
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(f'<div class="user-msg">🧑 {msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">🤖 {msg}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
