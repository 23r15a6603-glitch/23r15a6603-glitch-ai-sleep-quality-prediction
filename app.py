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
# ✅ CSS Themes
# ------------------------------
light_css = """
<style>
.stApp { 
    background: linear-gradient(135deg, #f3e6f9 0%, #f9f6fb 100%); 
    color: #2c3e50; 
    transition: all 0.4s ease-in-out;
}
h1 { 
    font-family: 'Segoe UI'; 
    font-size: 3.4rem; 
    color: #6C3483; 
    text-align: center; 
    margin-bottom: 0.3em; 
}
.stPrediction { 
    font-size: 1.3rem; font-weight: 700; 
    background: linear-gradient(90deg, #6C3483, #884ea0); 
    color: white; padding: 12px 18px; border-radius: 12px; text-align: center; 
}

/* Inputs, Selects, Sliders */
input, select, textarea {
    background: #fff !important;
    color: #2c3e50 !important;
    border: 2px solid #d6bbe9 !important;
    border-radius: 8px !important;
    padding: 6px !important;
    transition: all 0.3s ease-in-out;
}
input:focus, select:focus, textarea:focus {
    border-color: #6C3483 !important;
    box-shadow: 0 0 6px rgba(108, 52, 131, 0.4) !important;
}

/* Sliders */
.stSlider > div > div > div > div {
    background: #6C3483 !important;
}
.stSlider > div > div > div {
    color: #6C3483 !important;
}

/* Buttons */
button[kind="primary"] {
    background: #6C3483 !important;
    color: white !important;
    border-radius: 10px !important;
}
button[kind="primary"]:hover {
    background: #884ea0 !important;
}

/* Chat UI */
.chat-container { background: #fff; border-radius: 15px; padding: 15px; }
.chat-bubble-user { background: #d6bbe9; color: #3a0ca3; border-radius: 15px 15px 0 15px; padding: 10px; margin: 5px 0; }
.chat-bubble-bot { background: #ece8f8; color: #4a235a; border-radius: 15px 15px 15px 0; padding: 10px; margin: 5px 0; }
</style>
"""

dark_css = """
<style>
.stApp { 
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
    color: #f5f6fa; 
    transition: all 0.4s ease-in-out;
}
h1 { 
    font-family: 'Segoe UI'; 
    font-size: 3.4rem; 
    color: #9c88ff; 
    text-align: center; 
    margin-bottom: 0.3em; 
}
.stPrediction { 
    font-size: 1.3rem; font-weight: 700; 
    background: linear-gradient(90deg, #9c88ff, #6C3483); 
    color: white; padding: 12px 18px; border-radius: 12px; text-align: center; 
}

/* Inputs, Selects, Sliders */
input, select, textarea {
    background: #2d3436 !important;
    color: #f5f6fa !important;
    border: 2px solid #9c88ff !important;
    border-radius: 8px !important;
    padding: 6px !important;
    transition: all 0.3s ease-in-out;
}
input:focus, select:focus, textarea:focus {
    border-color: #6C3483 !important;
    box-shadow: 0 0 6px rgba(156, 136, 255, 0.6) !important;
}

/* Sliders */
.stSlider > div > div > div > div {
    background: #9c88ff !important;
}
.stSlider > div > div > div {
    color: #9c88ff !important;
}

/* Buttons */
button[kind="primary"] {
    background: #9c88ff !important;
    color: #1a1a2e !important;
    border-radius: 10px !important;
}
button[kind="primary"]:hover {
    background: #6C3483 !important;
    color: white !important;
}

/* Chat UI */
.chat-container { background: #1e1e2f; border-radius: 15px; padding: 15px; }
.chat-bubble-user { background: #6c5ce7; color: white; border-radius: 15px 15px 0 15px; padding: 10px; margin: 5px 0; }
.chat-bubble-bot { background: #2d3436; color: #dfe6e9; border-radius: 15px 15px 15px 0; padding: 10px; margin: 5px 0; }
</style>
"""

# ------------------------------
# ✅ Theme Toggle
# ------------------------------
theme = st.sidebar.radio("🌗 Theme", ["Light", "Dark"])
st.markdown(light_css if theme == "Light" else dark_css, unsafe_allow_html=True)

# ------------------------------
# Main Title
# ------------------------------
st.markdown("<h1>AI-Based Sleep Quality Prediction</h1>", unsafe_allow_html=True)
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

        st.markdown(f"<div class='stPrediction'>🌙 Predicted Sleep Quality: {result}</div>", unsafe_allow_html=True)
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

# Chat container wrapper
st.markdown('<div class="chat-container" style="display:flex; flex-direction:column;">', unsafe_allow_html=True)

for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"""
        <div class="chat-bubble-user">🧑 {msg}</div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-bubble-bot">🤖 {msg}</div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
