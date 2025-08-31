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
# Global styles (Dark Mode Fix)
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

body {
  background: var(--bg);
  color: var(--text);
}

.hero {
  background: linear-gradient(180deg, #0B1020 0%, #111827 100%);
  color: var(--text);
  border-radius: 28px;
  padding: 48px 40px;
  margin-bottom: 18px;
}
.hero h1 { font-size: 44px; line-height: 1.1; margin: 0 0 12px 0; color: var(--text); }
.hero p { color: var(--muted); font-size: 18px; margin: 0 0 20px 0; }
.cta-btn { background: var(--brand); color: white; padding: 12px 18px; border-radius: 999px; font-weight: 600; text-decoration:none; }
.subtle-btn { background: rgba(255,255,255,.08); color: #E5E7EB; padding: 12px 18px; border-radius: 999px; text-decoration:none; }
.kpis { display:flex; gap: 24px; flex-wrap: wrap; margin-top: 18px; }
.kpi { background: rgba(255,255,255,.05); border: 1px solid #374151; border-radius: 20px; padding: 14px 18px; min-width: 170px; color: var(--text); }
.kpi b { font-size: 22px; }

.section h2 { font-size: 28px; margin: 2px 0 12px; color: var(--text); }
.section p.lead { color: var(--muted); margin: 0 0 12px; }

.grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 16px; }
.card { grid-column: span 4; background: var(--card); color: var(--text); border: 1px solid #374151; border-radius: 20px; padding: 18px; }
.card h3 { margin: 6px 0 6px; color: var(--text); }

.badge { display:inline-block; background:#3730A3; color:#E0E7FF; padding:4px 10px; border-radius:999px; font-size:12px; font-weight:600; }
.rule { height:1px; background: #374151; margin: 12px 0; }

.result { border: 2px solid #374151; border-radius: 22px; padding: 18px; background: #1F2937; color: var(--text); }
.result.fair { border-color:#FBBF24; background:#92400E33; }
.result.poor { border-color:#EF4444; background:#7F1D1D33; }

.footer { color:#9CA3AF; text-align:center; margin-top: 36px; }
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)

# ------------------------------
# Secure API key (Gemini)
# ------------------------------
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
    except Exception:
        api_key = None

if api_key:
    genai.configure(api_key=api_key)
else:
    st.warning("⚠ Gemini API Key not found. Set GEMINI_API_KEY in .env or Streamlit Secrets.")

# ------------------------------
# Load model + scaler
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
# TOP HERO
# ------------------------------
with st.container():
    st.markdown(
        """
        <div class="hero">
          <div style="display:flex; align-items:flex-start; gap: 28px; flex-wrap: wrap;">
            <div style="flex:2; min-width: 280px;">
              <div class="badge">AI Sleep Analyzer</div>
              <h1>Understand your nights. Improve your days.</h1>
              <p>Measure the habits that shape your sleep and get instant, science-inspired guidance. Private. Simple. Actionable.</p>
              <div style="display:flex; gap:12px; flex-wrap: wrap;">
                <a class="cta-btn" href="#predict">Predict my sleep quality</a>
                <a class="subtle-btn" href="#coach">Ask the AI sleep coach</a>
              </div>
              <div class="kpis">
                <div class="kpi"><b>~60s</b><div>to get insights</div></div>
                <div class="kpi"><b>15+</b><div>lifestyle factors</div></div>
                <div class="kpi"><b>On‑device</b><div>ML inference</div></div>
              </div>
            </div>
            <div style="flex:1; min-width: 260px;">
              <div class="card" style="text-align:center;">
                <h3 style="margin:6px 0">Tonight's tip</h3>
                <div class="rule"></div>
                <p style="color:var(--muted); margin:0">Keep screens out of bed and aim for a consistent wake‑up time ±30 min.</p>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------
# LAYOUT TABS
# ------------------------------
overview_tab, predict_tab, coach_tab = st.tabs(["Overview", "Predict", "Coach"])

# ------------------------------
# OVERVIEW TAB
# ------------------------------
with overview_tab:
    st.subheader("Why choose AI Sleep Analyzer")
    st.caption("Designed like a premium health device page: clean, informative, and conversion‑friendly.")

    st.markdown("<div class='grid'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="card">
          <span class="badge">Personalized</span>
          <h3>Tailored insights</h3>
          <p>Get guidance based on your age, activity, stress, caffeine, and more.</p>
        </div>
        <div class="card">
          <span class="badge">Fast</span>
          <h3>Instant predictions</h3>
          <p>Our on‑device model scores your sleep quality as Fair or Poor in seconds.</p>
        </div>
        <div class="card">
          <span class="badge">Private</span>
          <h3>Your data stays yours</h3>
          <p>Inputs are processed locally in the app session and not stored by default.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# PREDICT TAB
# ------------------------------
with predict_tab:
    st.markdown("<a name='predict'></a>", unsafe_allow_html=True)
    st.subheader("Predict your sleep quality")
    st.caption("Takes ~1 minute • Works offline with the bundled model")

    with st.form("sleep_form"):
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
            wake_consistency = st.selectbox("Wake‑up Consistency", ["Consistent", "Inconsistent"])
        with col5:
            env_score = st.slider("Sleep Environment Score (1–10)", 1, 10, 7)
            water = st.slider("Daily Water Intake (litres)", 0.0, 5.0, 2.0, 0.5)

        submitted = st.form_submit_button("🔍 Predict now")

    if submitted:
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

            scaled = scaler.transform(input_df)
            pred = int(model.predict(scaled)[0])
            label_map = {0: 'Poor', 1: 'Fair'}
            result = label_map.get(pred, "Unknown")

            tips = []
            if sleep_duration < 7: tips.append("Aim for 7–9 hours of sleep.")
            if stress >= 7: tips.append("Add a 5‑minute wind‑down: breathing, journaling, or light stretching.")
            if screen_time > 1.0: tips.append("Reduce screens ≥60 minutes before bed.")
            if caffeine >= 3: tips.append("Cut caffeine after mid‑afternoon.")
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
        else:
            st.error("⚠ Prediction unavailable. Model or scaler missing.")

# ------------------------------
# COACH TAB
# ------------------------------
with coach_tab:
    st.markdown("<a name='coach'></a>", unsafe_allow_html=True)
    st.subheader("AI Sleep Coach")
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

    for role, msg in st.session_state.chat_history:
        if role == "You":
            st.info(f"🧑 {msg}")
        else:
            st.success(f"🤖 {msg}")

# ------------------------------
# Footer
# ------------------------------
st.markdown("""
<div class='footer'>
  <div>Built for education and habit‑building – not a medical device.</div>
  <div>© 2025 Sleep Analyzer</div>
</div>
""", unsafe_allow_html=True)
