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

import google.generativeai as genai
if api_key:
    genai.configure(api_key=api_key)
else:
    st.warning("⚠️ Gemini API Key not found. Please set GEMINI_API_KEY in .env (local) or Streamlit Secrets (cloud).")

# ------------------------------
# ✅ Load ML model and scaler safely
# ------------------------------
MODEL_PATH = "xgb_sleep_quality_model.pkl"
SCALER_PATH = "scaler_sleep_quality.pkl"

model, scaler = None, None
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    st.error("❌ Model/scaler not found. Please upload `xgb_sleep_quality_model.pkl` and `scaler_sleep_quality.pkl`.")

# ------------------------------
# Streamlit Config
# ------------------------------
st.set_page_config(page_title="AI-Based Sleep Quality Prediction", layout="wide")

# ------------------------------
# 🌈 Custom Styling
# ------------------------------
st.markdown(
    """
    <style>
    /* Main background */
    body {
        background: linear-gradient(to right, #1e3c72, #2a5298);
        color: white;
    }

    /* Titles */
    .stApp h1, .stApp h2, .stApp h3 {
        color: #ffffff !important;
        font-family: 'Poppins', sans-serif;
    }

    /* Input styling */
    .stNumberInput, .stSelectbox, .stSlider {
        background: #ffffff10;
        border-radius: 12px;
        padding: 10px;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #ff6600;
        color: white;
        border-radius: 12px;
        font-size: 16px;
        padding: 10px 20px;
        border: none;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #e65c00;
        transform: scale(1.05);
    }

    /* Chat messages */
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        border-radius: 15px;
        background: #ffffff10;
        margin-bottom: 15px;
    }
    .user-msg {
        background: #ff6600;
        color: white;
        padding: 10px 15px;
        border-radius: 15px 15px 0px 15px;
        margin: 8px 0;
        text-align: right;
    }
    .bot-msg {
        background: #2a5298;
        color: white;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0px;
        margin: 8px 0;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------
# Main Title
# ------------------------------
st.title("🌙 AI-Based Sleep Quality Prediction")
st.markdown("---")

# ------------------------------
# Predictor Section
# ------------------------------
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

        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        label_map = {0: 'Poor', 1: 'Fair', 2: 'Good'}
        result = label_map.get(prediction, "Unknown")

        st.success(f"🌙 Predicted Sleep Quality: {result}")
    else:
        st.error("⚠️ Prediction unavailable. Model or scaler missing.")

# ------------------------------
# Chatbot Section (Bottom)
# ------------------------------
st.markdown("---")
st.subheader("💬 Sleep AI Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask your question here:", key="chat_input")

col1, col2 = st.columns([1, 1])
with col1:
    send = st.button("Send")
with col2:
    clear = st.button("Clear Chat")

if send and api_key:
    if user_input.strip():
        try:
            time.sleep(1.5)  # avoid quota hitting

            history = [
                {"role": "user" if role == "You" else "model", "parts": [msg]}
                for role, msg in st.session_state.chat_history
            ]

            chat_model = genai.GenerativeModel("gemini-2.0-pro-exp")
            chat = chat_model.start_chat(history=history)
            response = chat.send_message(user_input)

        except Exception as e:
            if "429" in str(e):  # Quota exceeded
                try:
                    chat_model = genai.GenerativeModel("gemini-2.0-flash-exp")
                    chat = chat_model.start_chat(history=history)
                    response = chat.send_message(user_input)
                except Exception:
                    st.session_state.chat_history.append(("Bot", "⚠️ Models are out of quota. Try again later."))
                    response = None
            else:
                st.session_state.chat_history.append(("Bot", f"⚠️ Chatbot error: {e}"))
                response = None

        if response:
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", response.text))

if clear:
    st.session_state.chat_history = []

# Display chat history with styled bubbles
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(f'<div class="user-msg">🧑 {msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">🤖 {msg}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
