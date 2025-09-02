import os
import time
import joblib
import pandas as pd
import streamlit as st
import numpy as np
import google.generativeai as genai

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
    genai.configure(api_key=api_key)
else:
    st.warning("⚠ Gemini API Key not found. Please set GEMINI_API_KEY in .env (local) or Streamlit Secrets (cloud).")

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
    st.error("❌ Model/scaler not found. Please upload xgb_sleep_quality_model.pkl and scaler_sleep_quality.pkl.")

# ------------------------------
# Streamlit Config
# ------------------------------
st.set_page_config(page_title="Sleep Quality App", layout="wide")

# ------------------------------
# Sidebar Navigation
# ------------------------------
with st.sidebar:
    st.title("😴 Sleep Quality App")
    st.markdown("""
    *About this app:*  
    - Predicts your sleep quality as Good, Fair, or Poor  
    - Uses health and lifestyle factors like sleep duration, stress, activity, BMI, and more  
    - Provides insights to improve your sleep habits  
    - Includes an AI chatbot to answer your sleep-related questions  
    """)
    st.markdown("---")

    # Navigation Menu
    page = st.radio("📍 Navigate", ["Sleep Quality Predictor", "Sleep AI Chatbot"])

# ------------------------------
# Page 1: Predictor
# ------------------------------
if page == "Sleep Quality Predictor":
    st.markdown("<h1 style='text-align: center; color: #6C3483;'>AI-Based Sleep Quality Prediction</h1>", unsafe_allow_html=True)
    st.markdown("---")

    with st.form("sleep_form"):
        st.subheader("Enter your Health and Lifestyle Factors")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", 10, 100, 25)
            gender = st.selectbox("Gender", ["Male", "Female"])
            sleep_duration = st.slider("Sleep Duration (hrs)", 0.0, 12.0, 7.0, 0.5)
            activity = st.slider("Physical Activity (mins/day)", 0, 180, 30)

        with col2:
            stress = st.slider("Stress Level (1–10)", 1, 10, 5)
            caffeine = st.slider("Caffeine Intake (cups/day)", 0, 10, 1)
            alcohol = st.slider("Alcohol Intake (units/day)", 0, 10, 0)
            smoker = st.selectbox("Do you smoke?", ["No", "Yes"])

        with col3:
            heart_rate = st.number_input("Heart Rate (bpm)", 40, 140, 70)
            screen_time = st.slider("Screen Time Before Bed (hrs)", 0.0, 10.0, 2.0, 0.5)
            history = st.selectbox("Sleep Disorder History", ["No", "Yes"])
            bmi = st.number_input("BMI", 10.0, 50.0, 22.0)

        st.markdown("---")

        col4, col5 = st.columns(2)
        with col4:
            wake_consistency = st.selectbox("Wake-up Consistency", ["Consistent", "Inconsistent"])
        with col5:
            env_score = st.slider("Sleep Environment Score (1–10)", 1, 10, 7)
            water = st.slider("Daily Water Intake (litres)", 0.0, 5.0, 2.0, 0.5)

        submitted = st.form_submit_button("🔍 Predict Your Sleep Quality")

    # Prediction
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

            st.success(f"🌙 *Predicted Sleep Quality:* {result}")
        else:
            st.error("⚠ Prediction unavailable. Model or scaler missing.")

# ------------------------------
# Page 2: Chatbot (Custom UI)
# ------------------------------
elif page == "Sleep AI Chatbot":
    st.markdown("<h1 style='text-align: center; color: #2E86C1;'>💬 Sleep AI Chatbot</h1>", unsafe_allow_html=True)

    # Custom CSS for chat design
    st.markdown("""
    <style>
    .chat-box {
        padding: 15px;
        height: 400px;
        overflow-y: auto;
        background: #ffffff;
        border: 1px solid #ddd;
        border-radius: 10px;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .message {
        padding: 10px 15px;
        border-radius: 10px;
        max-width: 75%;
        line-height: 1.4;
        word-wrap: break-word;
    }
    .user {
        background: #e8f0fe;
        align-self: flex-end;
    }
    .assistant {
        background: #f1f3f4;
        align-self: flex-start;
    }
    .input-area {
        display: flex;
        margin-top: 10px;
        gap: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages
    chat_html = '<div class="chat-box">'
    for role, msg in st.session_state.chat_history:
        role_class = "user" if role == "You" else "assistant"
        chat_html += f'<div class="message {role_class}">{msg}</div>'
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # Input and buttons (fixed style)
    st.markdown("###")
    with st.container():
        col1, col2, col3 = st.columns([6,1,1])
        with col1:
            user_input = st.text_input("Ask your question here:", key="chat_input", label_visibility="collapsed")
        with col2:
            send = st.button("Send")
        with col3:
            clear = st.button("Clear")

    # Handle Send
    if send and api_key:
        if user_input.strip():
            try:
                time.sleep(1.2)

                history = [
                    {"role": "user" if role == "You" else "model", "parts": [msg]}
                    for role, msg in st.session_state.chat_history
                ]

                chat_model = genai.GenerativeModel("gemini-2.0-pro-exp")
                chat = chat_model.start_chat(history=history)
                response = chat.send_message(user_input)

            except Exception as e:
                if "429" in str(e):
                    try:
                        chat_model = genai.GenerativeModel("gemini-2.0-flash-exp")
                        chat = chat_model.start_chat(history=history)
                        response = chat.send_message(user_input)
                    except Exception:
                        st.session_state.chat_history.append(("Bot", "⚠ Models are out of quota. Try again later."))
                        response = None
                else:
                    st.session_state.chat_history.append(("Bot", f"⚠ Chatbot error: {e}"))
                    response = None

            if response:
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Bot", response.text))

    if clear:
        st.session_state.chat_history = []
