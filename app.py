import os
import time
import joblib
import pandas as pd
import streamlit as st
import numpy as np
import google.generativeai as genai

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(
    page_title="Sleep Analyzer – AI Coach",
    page_icon="😴",
    layout="wide",
)

# ------------------------------
# Global styles
# ------------------------------
STYLES = """
<style>
:root {
  --brand:#6C63FF;
  --text:#F3F4F6;
  --muted:#9CA3AF;
  --bg:#0B1020;
  --card:#1F2937;
}

body { background: var(--bg); color: var(--text); }

.hero {
  background: linear-gradient(180deg, #0B1020 0%, #111827 100%);
  border-radius: 28px;
  padding: 60px 40px;
  margin-bottom: 40px;
  text-align: left;
}
.hero h1 { font-size: 46px; font-weight: 700; margin-bottom: 12px; }
.hero p { color: var(--muted); font-size: 18px; margin-bottom: 24px; }
.cta-btn { background: var(--brand); color: white; padding: 14px 24px; border-radius: 999px; font-weight: 600; text-decoration:none; }
.kpis { display:flex; gap: 24px; flex-wrap: wrap; margin-top: 32px; }
.kpi { background: rgba(255,255,255,.05); border: 1px solid #374151; border-radius: 20px; padding: 16px 20px; min-width: 170px; }
.kpi b { font-size: 22px; }

.section { margin-top: 60px; margin-bottom: 40px; }
.section h2 { font-size: 32px; margin-bottom: 16px; }
.section p { color: var(--muted); }

.grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 20px; margin-top: 20px; }
.card { grid-column: span 4; background: var(--card); border: 1px solid #374151; border-radius: 20px; padding: 22px; }
.card h3 { margin: 10px 0; font-size: 20px; }

.badge { display:inline-block; background:#3730A3; color:#E0E7FF; padding:6px 12px; border-radius:999px; font-size:12px; font-weight:600; }

.result { border: 2px solid #374151; border-radius: 22px; padding: 20px; background: #1F2937; margin-top: 20px; }
.result.fair { border-color:#FBBF24; background:#92400E33; }
.result.poor { border-color:#EF4444; background:#7F1D1D33; }

.faq { margin-top:20px; }
.footer { color:#9CA3AF; text-align:center; margin-top: 60px; }
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)

# ------------------------------
# Secure API key (Gemini)
# ------------------------------
api_key = None
try:
    # Try to get API key from Streamlit secrets
    if hasattr(st, 'secrets') and 'GEMINI_API_KEY' in st.secrets:
        api_key = st.secrets['GEMINI_API_KEY']
    # If not in secrets, try environment variable
    if not api_key:
        api_key = os.environ.get('GEMINI_API_KEY')
except Exception:
    api_key = None

if api_key:
    try:
        genai.configure(api_key=api_key)
        gemini_available = True
    except Exception as e:
        st.error(f"Error configuring Gemini: {e}")
        gemini_available = False
else:
    gemini_available = False
    st.warning("⚠ Gemini API Key not found. AI Coach feature will be limited.")

# ------------------------------
# Load model + scaler
# ------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("xgb_sleep_quality_model.pkl")
        scaler = joblib.load("scaler_sleep_quality.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("❌ Model/scaler not found. Please upload xgb_sleep_quality_model.pkl and scaler_sleep_quality.pkl.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, scaler = load_model()

# ------------------------------
# HERO SECTION
# ------------------------------
st.markdown(
    """
    <div class="hero">
      <span class="badge">AI Sleep Analyzer</span>
      <h1>Understand your nights. Improve your days.</h1>
      <p>Measure the habits that shape your sleep and get instant, science-inspired guidance. Private. Simple. Actionable.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

colH1, colH2 = st.columns([1,1])
with colH1:
    st.button("Predict my sleep quality", on_click=lambda: st.session_state.update(scroll_to="predict"))
with colH2:
    st.button("Ask the AI sleep coach", on_click=lambda: st.session_state.update(scroll_to="coach"))

st.markdown(
    """
    <div class="kpis">
      <div class="kpi"><b>~60s</b><div>to get insights</div></div>
      <div class="kpi"><b>15+</b><div>lifestyle factors</div></div>
      <div class="kpi"><b>On-device</b><div>ML inference</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# FEATURES SECTION
# ------------------------------
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.header("Why choose AI Sleep Analyzer")
st.markdown("<div class='grid'>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="card">
      <span class="badge">Personalized</span>
      <h3>Tailored insights</h3>
      <p>Guidance based on your age, activity, stress, caffeine, and more.</p>
    </div>
    <div class="card">
      <span class="badge">Fast</span>
      <h3>Instant predictions</h3>
      <p>Our on-device model scores your sleep quality as Fair or Poor in seconds.</p>
    </div>
    <div class="card">
      <span class="badge">Private</span>
      <h3>Your data stays yours</h3>
      <p>Inputs are processed locally in the app session and not stored by default.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("</div></div>", unsafe_allow_html=True)

# ------------------------------
# HOW IT WORKS SECTION
# ------------------------------
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.header("How it works")
st.caption("Simple steps to better sleep.")
steps = [
    "Enter your daily lifestyle factors (activity, caffeine, stress, etc.)",
    "Our ML model instantly predicts Fair or Poor sleep quality",
    "Get personalized tips on improving your nightly rest",
    "Chat with the AI coach for deeper guidance"
]
for i, s in enumerate(steps, 1):
    st.markdown(f"**Step {i}:** {s}")
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# PREDICT SECTION
# ------------------------------
st.markdown("<div class='section' id='predict'>", unsafe_allow_html=True)
st.header("🔍 Predict your sleep quality")
st.caption("Takes ~1 minute • Works offline with the bundled model")

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

if st.button("Run Prediction"):
    if model is not None and scaler is not None:
        input_df = pd.DataFrame({
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
            scaled = scaler.transform(input_df)
            pred = int(model.predict(scaled)[0])
            label_map = {0: 'Poor', 1: 'Fair'}
            result = label_map.get(pred, "Unknown")

            tips = []
            if sleep_duration < 7: tips.append("Aim for 7–9 hours of sleep.")
            if stress >= 7: tips.append("Add a 5-minute wind-down: breathing, journaling, or light stretching.")
            if screen_time > 1.0: tips.append("Reduce screens ≥60 minutes before bed.")
            if caffeine >= 3: tips.append("Cut caffeine after mid-afternoon.")
            if alcohol >= 2: tips.append("Avoid alcohol within 3–4 hours of bedtime.")
            if activity < 30: tips.append("Target at least 30 minutes of light activity.")
            if bmi >= 27: tips.append("Discuss weight, snoring, or apnea risk with a professional if concerned.")
            if water < 1.5: tips.append("Hydrate earlier in the day to avoid nocturnal awakenings.")
            if env_score < 6: tips.append("Dark, cool (18–20°C), and quiet rooms improve sleep quality.")

            tone_cls = "fair" if result == "Fair" else "poor"
            st.markdown(f"<div class='result {tone_cls}'>", unsafe_allow_html=True)
            st.markdown(f"### 🌙 Predicted Sleep Quality: **{result}**")
            if tips:
                st.markdown("**Next best actions:**")
                for t in tips[:6]:
                    st.write("• ", t)
            else:
                st.caption("You're doing great. Keep routines consistent and revisit after a week of tracking.")
            st.markdown("</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.error("⚠ Prediction unavailable. Model or scaler missing.")

st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# COACH SECTION
# ------------------------------
st.markdown("<div class='section' id='coach'>", unsafe_allow_html=True)
st.header("🤖 AI Sleep Coach")
st.caption("Ask about habits, routines, or how to improve tonight.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question…", key="chat_input")
col1, col2 = st.columns([1,1])
with col1:
    send = st.button("Send")
with col2:
    clear = st.button("Clear Chat")

if clear:
    st.session_state.chat_history = []

if send and user_input.strip():
    if not gemini_available:
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", "⚠ Gemini API not configured. Please add your GEMINI_API_KEY to use the AI Coach."))
    else:
        try:
            # Add user message to chat history
            st.session_state.chat_history.append(("You", user_input))
            
            # Format history for Gemini
            history_for_gemini = []
            for role, msg in st.session_state.chat_history[:-1]:  # Exclude the latest user message
                history_for_gemini.append({
                    "role": "user" if role == "You" else "model",
                    "parts": [msg]
                })
            
            # Generate response
            model_name = "gemini-pro"  # Use a stable model name
            model = genai.GenerativeModel(model_name)
            chat = model.start_chat(history=history_for_gemini)
            
            with st.spinner("Thinking..."):
                response = chat.send_message(user_input)
                
            # Add bot response to chat history
            st.session_state.chat_history.append(("Bot", response.text))
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if "quota" in error_msg.lower():
                error_msg = "API quota exceeded. Please try again later."
            st.session_state.chat_history.append(("Bot", f"⚠ {error_msg}"))

# Display chat history
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.info(f"🧑 {msg}")
    else:
        st.success(f"🤖 {msg}")

st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# FAQ SECTION
# ------------------------------
st.markdown("<div class='section'>", unsafe_allow_html=True)
st.header("❓ Frequently Asked Questions")
faqs = {
    "Is this a medical device?": "No. This app is for educational and lifestyle purposes only.",
    "How accurate are the predictions?": "Our ML model was trained on lifestyle and sleep datasets, but it cannot replace medical evaluation.",
    "Where is my data stored?": "Inputs are processed locally in your app session and not stored by default.",
    "Do I need an internet connection?": "Only for AI coaching. Predictions work offline once the model is loaded."
}
for q, a in faqs.items():
    with st.expander(q):
        st.write(a)
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("""
<div class='footer'>
  <div>Built for education and habit-building – not a medical device.</div>
  <div>© 2025 Sleep Analyzer</div>
</div>
""", unsafe_allow_html=True)
