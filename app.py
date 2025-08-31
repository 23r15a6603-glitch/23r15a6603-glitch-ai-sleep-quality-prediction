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
    st.warning("⚠ Gemini API Key not found. Please set GEMINI_API_KEY in .env or Streamlit Secrets.")

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
st.set_page_config(page_title="Sleep Quality Predictor", layout="wide")

# ------------------------------
# 🎨 Custom CSS (Pro Hotstar-style)
# ------------------------------
def load_custom_css():
    st.markdown("""
        <style>
        /* Reset */
        body, .stApp {
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, sans-serif;
        }

        /* Hero Section */
        .hero {
            text-align: center;
            padding: 60px 20px;
            background: linear-gradient(120deg, #6C3483, #1ABC9C);
            border-radius: 0 0 30px 30px;
            margin-bottom: 40px;
            color: white;
        }
        .hero h1 {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .hero p {
            font-size: 1.2rem;
            opacity: 0.9;
        }

        /* Cards */
        .custom-card {
            background: var(--card-bg);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.2);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .custom-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        }

        /* Result */
        .result-card {
            background: linear-gradient(135deg, #6C3483, #1ABC9C);
            color: white;
            padding: 30px;
            border-radius: 18px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            animation: fadeIn 0.8s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Buttons */
        button[kind="primary"] {
            border-radius: 12px !important;
            font-weight: bold !important;
            background: linear-gradient(90deg,#6C3483,#1ABC9C) !important;
            color: white !important;
            border: none !important;
        }

        /* Chatbox */
        .chat-container {
            border-radius: 15px;
            background: var(--card-bg);
            padding: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
            max-height: 350px;
            overflow-y: auto;
        }

        /* Light / Dark Mode */
        :root {
            --card-bg: #ffffff;
        }
        [data-theme="dark"] {
            --card-bg: #1e1e1e;
        }
        </style>
    """, unsafe_allow_html=True)

load_custom_css()

# ------------------------------
# Hero Section
# ------------------------------
st.markdown("""
<div class="hero">
    <h1>🌙 AI Sleep Quality Predictor</h1>
    <p>Understand your sleep health and get AI-driven insights</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# Predictor Section
# ------------------------------
with st.form("sleep_form"):
    st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
    st.subheader("📝 Your Health & Lifestyle Factors")

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
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Prediction
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
        label_map = {0: 'Poor', 1: 'Fair', 2: 'Good'}
        result = label_map.get(prediction, "Unknown")

        st.markdown(f"<div class='result-card'>🌙 Predicted Sleep Quality: {result}</div>", unsafe_allow_html=True)
    else:
        st.error("⚠ Prediction unavailable. Model or scaler missing.")

# ------------------------------
# Chatbot Section
# ------------------------------
st.markdown("## 💬 AI Sleep Chat Assistant")
st.markdown("<div class='custom-card chat-container'>", unsafe_allow_html=True)

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
            time.sleep(1.5)
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
                except Exception:
                    st.session_state.chat_history.append(("Bot", "⚠ Models out of quota. Try later."))
                    response = None
            else:
                st.session_state.chat_history.append(("Bot", f"⚠ Chatbot error: {e}"))
                response = None

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

st.markdown("</div>", unsafe_allow_html=True)
