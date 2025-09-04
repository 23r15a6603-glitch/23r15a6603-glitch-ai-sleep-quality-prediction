import os
import time
import joblib
import pandas as pd
import streamlit as st
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
    st.warning("⚠ Gemini API Key not found. Set GEMINI_API_KEY in .env or Streamlit Secrets.")

# ------------------------------
# ✅ Load ML model and scaler
# ------------------------------
MODEL_PATH = "xgb_sleep_quality_model.pkl"
SCALER_PATH = "scaler_sleep_quality.pkl"

model, scaler = None, None
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    st.error("❌ Model/scaler not found. Upload them first.")

# ------------------------------
# ✅ Page Config
# ------------------------------
st.set_page_config(page_title="Sleep Quality Predictor", layout="centered")

# ------------------------------
# 🏷 Header + About
# ------------------------------
st.markdown("<h1 style='text-align:center; color:#6C3483;'>😴 Sleep Quality Predictor</h1>", unsafe_allow_html=True)

st.markdown("""
This app predicts your *sleep quality* based on your health and lifestyle.  

*Features:*
- Predicts sleep quality as *Fair* or *Poor*  
- Uses lifestyle factors: sleep duration, stress, BMI, caffeine, alcohol, smoking, etc.  
- Provides insights to improve your sleep  
- Includes an *AI chatbot* for questions  
---
""")

# ------------------------------
# 📝 Input Form
# ------------------------------
with st.form("sleep_form"):
    st.subheader("Enter your Health & Lifestyle Details")

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

    col4, col5 = st.columns(2)
    with col4:
        wake_consistency = st.selectbox("Wake-up Consistency", ["Consistent", "Inconsistent"])
    with col5:
        env_score = st.slider("Sleep Environment Score (1–10)", 1, 10, 7)
        water = st.slider("Daily Water Intake (litres)", 0.0, 5.0, 2.0, 0.5)

    submitted = st.form_submit_button("🔍 Predict Sleep Quality")

# ------------------------------
# 📊 Prediction (Fair / Poor)
# ------------------------------
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

        # 2-class mapping
        label_map = {0: 'Poor', 1: 'Fair'}
        result = label_map.get(prediction, "Unknown")

        # Color-coded display
        if result == "Poor":
            st.error(f"🌙 Your Predicted Sleep Quality: *{result}*")
        elif result == "Fair":
            st.warning(f"🌙 Your Predicted Sleep Quality: *{result}*")
        else:
            st.info(f"🌙 Your Predicted Sleep Quality: *{result}*")
    else:
        st.error("⚠ Model or scaler missing. Cannot predict.")

# ------------------------------
# 🤖 Chatbot Section
# ------------------------------
st.markdown("---")
st.subheader("💬 Sleep AI Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask your question:")

col1, col2 = st.columns([1, 1])
with col1:
    send = st.button("Send")
with col2:
    clear = st.button("Clear Chat")

if send and api_key and user_input.strip():
    try:
        time.sleep(1.2)  # reduce quota hit
        history = [{"role": "user" if role == "You" else "model", "parts": [msg]}
                   for role, msg in st.session_state.chat_history]

        chat_model = genai.GenerativeModel("gemini-2.0-pro-exp")
        chat = chat_model.start_chat(history=history)
        response = chat.send_message(user_input)
    except Exception as e:
        if "429" in str(e):
            try:
                chat_model = genai.GenerativeModel("gemini-2.0-flash-exp")
                chat = chat_model.start_chat(history=history)
                response = chat.send_message(user_input)
            except:
                response = None
                st.session_state.chat_history.append(("Bot", "⚠ Quota exceeded. Try later."))
        else:
            response = None
            st.session_state.chat_history.append(("Bot", f"⚠ Error: {e}"))

    if response:
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response.text))

if clear:
    st.session_state.chat_history = []

# Display chat history
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.info(f"🧑 {msg}")
    else:
        st.success(f"🤖 {msg}")
