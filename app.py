import os
import time
import joblib
import pandas as pd
import numpy as np
import streamlit as st

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
    st.error("❌ Model/scaler not found. Please upload `xgb_sleep_quality_model.pkl` and `scaler_sleep_quality.pkl` to your GitHub repo.")

# ------------------------------
# Streamlit Config
# ------------------------------
st.set_page_config(page_title="AI-Based Sleep Quality Prediction", layout="wide")

# ------------------------------
# 🌙 Theme Toggle
# ------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

toggle = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
st.session_state.dark_mode = toggle

# ------------------------------
# CSS Themes
# ------------------------------
light_css = """
<style>
.stApp { background: linear-gradient(135deg, #f3e6f9 0%, #f9f6fb 100%); }
h1 { font-family: 'Segoe UI'; font-size: 3.4rem; color: #6C3483; text-align: center;
     margin-bottom: 0.3em; animation: fadeInDown 1s ease; }
@keyframes fadeInDown { from { opacity: 0; transform: translateY(-20px);} to { opacity: 1; transform: translateY(0);} }
.stSuccess { font-size: 1.3rem; font-weight: 700; color: #fff !important;
     background: linear-gradient(90deg, #6C3483, #884ea0); border-radius: 15px; padding: 15px 20px;
     box-shadow: 0 0 18px rgba(108,52,131,0.6); }
.form-container { background: #f9f6fb; padding: 25px 30px; border-radius: 15px;
     box-shadow: 0 6px 15px rgb(108 52 131 / 0.15); margin-bottom: 30px; }
.chat-container { background: #fff; padding: 20px 25px; border-radius: 20px;
     box-shadow: 0 8px 20px rgb(108 52 131 / 0.15); max-height: 500px; overflow-y: auto;
     margin-bottom: 15px; scrollbar-width: thin; scrollbar-color: #6C3483 #f9f6fb; }
.chat-bubble-user { position: relative; background: #d6bbe9; color: #3a0ca3; padding: 12px 18px;
     border-radius: 18px 18px 0 18px; max-width: 75%; margin-bottom: 12px; font-weight: 600;
     align-self: flex-end; box-shadow: 0 2px 6px rgb(108 52 131 / 0.2); }
.chat-bubble-user::before { content: "🧑"; position: absolute; left: -30px; top: 0; font-size: 1.5rem; }
.chat-bubble-bot { position: relative; background: #ece8f8; color: #4a235a; padding: 12px 18px;
     border-radius: 18px 18px 18px 0; max-width: 75%; margin-bottom: 12px; font-weight: 500;
     align-self: flex-start; box-shadow: 0 2px 6px rgb(108 52 131 / 0.15); }
.chat-bubble-bot::before { content: "🤖"; position: absolute; left: -30px; top: 0; font-size: 1.5rem; }
</style>
"""

dark_css = """
<style>
.stApp { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #f5f6fa; }
h1 { font-family: 'Segoe UI'; font-size: 3.4rem; color: #9c88ff; text-align: center;
     margin-bottom: 0.3em; animation: fadeInDown 1s ease; }
@keyframes fadeInDown { from { opacity: 0; transform: translateY(-20px);} to { opacity: 1; transform: translateY(0);} }
.stSuccess { font-size: 1.3rem; font-weight: 700; color: #fff !important;
     background: linear-gradient(90deg, #9c88ff, #6C3483); border-radius: 15px; padding: 15px 20px;
     box-shadow: 0 0 18px rgba(156,136,255,0.6); }
.form-container { background: #222f45; padding: 25px 30px; border-radius: 15px;
     box-shadow: 0 6px 15px rgba(156,136,255,0.2); margin-bottom: 30px; }
.chat-container { background: #1e1e2f; padding: 20px 25px; border-radius: 20px;
     box-shadow: 0 8px 20px rgba(156,136,255,0.15); max-height: 500px; overflow-y: auto;
     margin-bottom: 15px; scrollbar-width: thin; scrollbar-color: #9c88ff #1a1a2e; }
.chat-bubble-user { position: relative; background: #6c5ce7; color: #fff; padding: 12px 18px;
     border-radius: 18px 18px 0 18px; max-width: 75%; margin-bottom: 12px; font-weight: 600;
     align-self: flex-end; box-shadow: 0 2px 6px rgba(156,136,255,0.3); }
.chat-bubble-user::before { content: "🧑"; position: absolute; left: -30px; top: 0; font-size: 1.5rem; }
.chat-bubble-bot { position: relative; background: #2d3436; color: #dfe6e9; padding: 12px 18px;
     border-radius: 18px 18px 18px 0; max-width: 75%; margin-bottom: 12px; font-weight: 500;
     align-self: flex-start; box-shadow: 0 2px 6px rgba(156,136,255,0.2); }
.chat-bubble-bot::before { content: "🤖"; position: absolute; left: -30px; top: 0; font-size: 1.5rem; }
</style>
"""

# Apply selected theme
st.markdown(dark_css if st.session_state.dark_mode else light_css, unsafe_allow_html=True)

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
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
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
    st.markdown('</div>', unsafe_allow_html=True)

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

        st.success(f"**🌙 Predicted Sleep Quality:** {result}")
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
            time.sleep(2)  # avoid quota hitting

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

# Chat container wrapper for scroll
st.markdown('<div class="chat-container" style="display:flex; flex-direction:column;">', unsafe_allow_html=True)

for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"""<div class="chat-bubble-user">{msg}</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="chat-bubble-bot">{msg}</div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
