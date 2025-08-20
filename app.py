import os
import time
import joblib
import pandas as pd
import streamlit as st
import numpy as np
import datetime as dt

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
# Main Title
# ------------------------------
st.title("AI-Based Sleep Quality Prediction")
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
# Chatbot Section (Swiggy/Zomato style)
# ------------------------------
st.markdown("---")

st.markdown("""
<style>
.sleep-chat-wrap { max-width: 900px; margin: 0 auto; }
.sleep-card { background: #fff; border-radius: 16px; border: 1px solid #eee;
              box-shadow: 0 6px 20px rgba(0,0,0,0.06); display: flex;
              flex-direction: column; height: 75vh; overflow: hidden; }
.sleep-header { display: flex; align-items: center; gap: 12px; padding: 14px 16px;
                background: linear-gradient(90deg,#ff7043,#ff3d00); color: #fff; }
.sleep-header .title { font-size: 1.1rem; font-weight: 700; }
.sleep-header .sub { font-size: 0.85rem; opacity: 0.85; }
.sleep-body { flex: 1; overflow-y: auto; background: #fafafa; padding: 16px 12px 90px; }
.msg { display: flex; gap: 10px; margin: 8px 0; }
.msg.agent { justify-content: flex-start; }
.msg.user  { justify-content: flex-end; }
.avatar { width: 34px; height: 34px; border-radius: 50%; display:flex; align-items:center;
          justify-content:center; color:#fff; font-weight:700; }
.avatar.agent { background:#ff5722; }
.avatar.user  { background:#37474f; }
.bubble { max-width:70vw; padding:10px 12px; border-radius:16px; font-size:0.95rem; line-height:1.45;
          box-shadow:0 2px 6px rgba(0,0,0,0.05); }
.agent .bubble { background:#fff; border:1px solid #eee; color:#333; border-top-left-radius:6px; }
.user .bubble  { background:#ffebee; border:1px solid #ffcdd2; color:#c62828; border-top-right-radius:6px; }
.meta { font-size:0.75rem; color:#999; margin-top:3px; }
.sleep-inputbar { display:flex; gap:8px; padding:10px; border-top:1px solid #eee; background:#fff; }
.sleep-inputbar .txt { flex:1; padding:10px 12px; border:1px solid #ddd; border-radius:12px; font-size:0.95rem; }
.sleep-inputbar .send { background:linear-gradient(90deg,#ff7043,#ff3d00); color:#fff;
                        border:0; border-radius:12px; padding:10px 16px; font-weight:700; cursor:pointer; }
.sleep-inputbar .clear { background:#f5f5f5; border:0; border-radius:12px; padding:10px 14px; cursor:pointer; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="sleep-chat-wrap"><div class="sleep-card">', unsafe_allow_html=True)

# Header
st.markdown("""
<div class="sleep-header">
  <div class="avatar agent">S</div>
  <div>
    <div class="title">Sleep AI Support</div>
    <div class="sub">Your 24/7 Sleep Assistant</div>
  </div>
</div>
<div class="sleep-body">
""", unsafe_allow_html=True)

# Init chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [("Bot", "Hi 👋 I’m your Sleep AI assistant. How can I help you today?")]

# Render messages
def _fmt_time():
    return dt.datetime.now().strftime("%I:%M %p")

for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"""
        <div class="msg user">
          <div class="bubble">{msg}<div class="meta">{_fmt_time()}</div></div>
          <div class="avatar user">U</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="msg agent">
          <div class="avatar agent">S</div>
          <div class="bubble">{msg}<div class="meta">{_fmt_time()}</div></div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # close body

# Input bar
col1, col2, col3 = st.columns([7,1.5,1.5])
with col1:
    user_input = st.text_input("Type your message", key="chat_input", label_visibility="collapsed", placeholder="Ask me anything about sleep…")
with col2:
    send = st.button("Send")
with col3:
    clear = st.button("Clear")

st.markdown("</div></div>", unsafe_allow_html=True)  # close card/wrap

# Logic
def make_history(hist):
    return [{"role": "user" if r == "You" else "model", "parts": [m]} for r,m in hist]

if clear:
    st.session_state.chat_history = [("Bot", "✅ Chat cleared. How can I help you next?")]

if send and user_input.strip():
    text = user_input.strip()
    st.session_state.chat_history.append(("You", text))

    if api_key:
        try:
            history = make_history(st.session_state.chat_history)
            chat_model = genai.GenerativeModel("gemini-2.0-pro-exp")
            chat = chat_model.start_chat(history=history)
            response = chat.send_message(text)
            st.session_state.chat_history.append(("Bot", response.text))
        except Exception as e:
            st.session_state.chat_history.append(("Bot", f"⚠️ Error: {e}"))
    else:
        st.session_state.chat_history.append(("Bot", "⚠️ Missing API key. Please configure it."))
