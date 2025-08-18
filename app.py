import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai

# ------------------------------
# Load Gemini API Key
# ------------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ------------------------------
# Load ML model and scaler
# ------------------------------
model = joblib.load("xgb_sleep_quality_model.pkl")
scaler = joblib.load("scaler_sleep_quality.pkl")

# ------------------------------
# Streamlit Config
# ------------------------------
st.set_page_config(page_title="AI-Based Sleep Quality Prediction", layout="wide")

# ------------------------------
# Main Title
# ------------------------------
st.markdown("<h1 style='text-align: center; color: #6C3483;'>AI-Based Sleep Quality Prediction</h1>", unsafe_allow_html=True)
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
    result = label_map[prediction]

    st.success(f"**Predicted Sleep Quality:** {result}")

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

if send:
    if user_input.strip():
        try:
            time.sleep(2)  # ✅ avoid hitting per-minute quota

            # Prepare history
            history = [
                {"role": "user" if role == "You" else "model", "parts": [msg]}
                for role, msg in st.session_state.chat_history
            ]

            # Try PRO model first
            chat_model = genai.GenerativeModel("gemini-2.0-pro-exp")
            chat = chat_model.start_chat(history=history)
            response = chat.send_message(user_input)

        except Exception as e:
            if "429" in str(e):  # Quota exceeded
                try:
                    # ✅ Fallback to FLASH
                    chat_model = genai.GenerativeModel("gemini-2.0-flash-exp")
                    chat = chat_model.start_chat(history=history)
                    response = chat.send_message(user_input)
                except Exception as e2:
                    st.session_state.chat_history.append(
                        ("Bot", "⚠️ Both Pro and Flash models are out of quota. Please try again later.")
                    )
                    response = None
            else:
                st.session_state.chat_history.append(("Bot", f"⚠️ Chatbot error: {e}"))
                response = None

        if response:
            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("Bot", response.text))

if clear:
    st.session_state.chat_history = []

# Display chat history
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**🧑 {role}:** {msg}")
    else:
        st.markdown(f"**🤖 {role}:** {msg}")
