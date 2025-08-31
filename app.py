# app.py
import os
import time
import joblib
import pandas as pd
import streamlit as st
import google.generativeai as genai

# ------------------------------
# Config + secure key
# ------------------------------
st.set_page_config(page_title="SleepPro • Predictor", page_icon="🌙", layout="wide")
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
    except Exception:
        api_key = None

if api_key:
    try:
        genai.configure(api_key=api_key)
    except Exception:
        # If generative API fails to configure, we still let the predictor run.
        pass

# ------------------------------
# Load model + scaler (safe)
# ------------------------------
MODEL_PATH = "xgb_sleep_quality_model.pkl"
SCALER_PATH = "scaler_sleep_quality.pkl"
model, scaler = None, None
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        st.error(f"Failed to load model/scaler: {e}")
else:
    # Only show this if running in interactive mode; don't block UI on cloud
    st.warning("Model or scaler missing. Place xgb_sleep_quality_model.pkl and scaler_sleep_quality.pkl in the app folder.")

# ------------------------------
# Theme system (clean, robust)
# ------------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def apply_theme(dark: bool):
    """Sets CSS variables for light/dark. Minimal, robust, non-invasive styling."""
    if dark:
        css = """
        <style>
        :root{
            --bg: #0b0f1a;
            --panel: #0f1724;
            --muted: #9aa3b2;
            --text: #e6eef6;
            --accent1: #6C3483;
            --accent2: #1abc9c;
            --card: #0f1724;
            --border: rgba(255,255,255,0.04);
            --shadow: 0 12px 28px rgba(2,6,23,0.65);
        }
        </style>
        """
    else:
        css = """
        <style>
        :root{
            --bg: #ffffff;
            --panel: #f6f8fb;
            --muted: #5a6473;
            --text: #0b0f1a;
            --accent1: #6C3483;
            --accent2: #1abc9c;
            --card: #ffffff;
            --border: rgba(11,15,26,0.06);
            --shadow: 0 8px 20px rgba(17,17,17,0.06);
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

apply_theme(st.session_state.dark_mode)

# ------------------------------
# Minimal, stable CSS for layout + chat bubble
# ------------------------------
st.markdown("""
<style>
/* Base */
[data-testid="stAppViewContainer"] { background: var(--bg); color: var(--text); }
[data-testid="stSidebar"] { background: var(--panel); }

/* Top nav */
.top-nav {
  display:flex; align-items:center; justify-content:space-between;
  gap: 12px;
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
  background: transparent;
}
.brand {
  display:flex; gap:10px; align-items:center; font-weight:800; font-size:1.05rem;
}
.controls { display:flex; gap:8px; align-items:center; }

/* Hero */
.hero {
  padding: 36px 26px;
  border-radius: 14px;
  background: linear-gradient(120deg, rgba(108,52,131,0.12), rgba(26,188,156,0.08));
  margin-bottom: 18px;
}
.hero h1 { margin:0; font-size:2.2rem; font-weight:800; }
.hero p { margin:6px 0 0 0; color:var(--muted); }

/* Centered main card */
.main-card {
  background: var(--card);
  border: 1px solid var(--border);
  padding: 22px;
  border-radius: 12px;
  box-shadow: var(--shadow);
  max-width: 920px;
  margin: 0 auto 18px auto;
}

/* Result */
.result {
  margin-top: 14px;
  padding: 14px;
  border-radius: 10px;
  text-align:center;
  font-weight:800;
  color: white;
  background: linear-gradient(90deg, var(--accent1), var(--accent2));
  box-shadow: 0 6px 18px rgba(0,0,0,0.12);
}

/* Small tiles row */
.tiles-row {
  display:flex; gap:14px; overflow-x:auto; padding-bottom:8px;
}
.tile {
  min-width:220px; max-width:260px;
  background: var(--card); border:1px solid var(--border);
  border-radius:12px; padding:12px;
  box-shadow: var(--shadow);
}
.tile h4 { margin:0 0 6px 0; font-size:1.05rem; font-weight:700; }
.tile p { margin:0; color:var(--muted); font-size:0.95rem; }

/* Floating chat bubble & window */
.chat-bubble {
  position: fixed;
  right: 22px;
  bottom: 22px;
  z-index: 9999;
}
.chat-button {
  width: 64px; height:64px; border-radius:50%;
  border: none; outline: none;
  background: linear-gradient(90deg, var(--accent1), var(--accent2));
  color: white; font-weight:800;
  box-shadow: 0 8px 22px rgba(0,0,0,0.2);
  cursor: pointer;
}
.chat-window {
  position: fixed;
  right: 22px;
  bottom: 100px;
  width: 360px;
  max-height: 520px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  box-shadow: 0 20px 50px rgba(2,6,23,0.45);
  z-index: 9999;
  display: flex; flex-direction: column;
  overflow: hidden;
}
.chat-header {
  display:flex; justify-content:space-between; align-items:center;
  padding: 12px; border-bottom: 1px solid var(--border);
}
.chat-messages {
  padding: 12px; overflow-y:auto; flex:1; background: transparent;
}
.msg-user { background: linear-gradient(90deg, rgba(108,52,131,0.12), rgba(26,188,156,0.07)); padding:8px 10px; border-radius:10px; margin:8px 0; max-width:85%; align-self:flex-end; }
.msg-bot { background: var(--panel); padding:8px 10px; border-radius:10px; margin:8px 0; max-width:85%; align-self:flex-start; color:var(--text); }

/* responsive */
@media (max-width: 720px) {
  .main-card { margin: 0 14px 18px 14px; width:calc(100% - 28px); }
  .chat-window { right: 12px; left: 12px; width: auto; bottom: 80px; }
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Top nav (simple)
# ------------------------------
nav_cols = st.columns([1, 3, 1])
with nav_cols[0]:
    st.markdown("<div class='top-nav'><div class='brand'>🌙 SleepPro</div></div>", unsafe_allow_html=True)
with nav_cols[1]:
    # empty center to keep nav spaced
    st.markdown("", unsafe_allow_html=True)
with nav_cols[2]:
    # theme toggle in top-right
    def toggle_theme():
        st.session_state.dark_mode = not st.session_state.dark_mode
        apply_theme(st.session_state.dark_mode)
    st.button("🌙 Dark" if not st.session_state.dark_mode else "☀ Light", on_click=toggle_theme)

# ------------------------------
# Hero
# ------------------------------
st.markdown(
    "<div class='hero'><h1>Professional Sleep Quality Predictor</h1>"
    "<p>Clean design — quick predictions — clear insights.</p></div>",
    unsafe_allow_html=True
)

# ------------------------------
# Main predictor card (centered, focused)
# ------------------------------
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("## 📝 Tell us about yourself")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    sleep_duration = st.slider("Sleep Duration (hrs)", 0.0, 12.0, 7.0, 0.5)
with col2:
    activity = st.slider("Physical Activity (mins/day)", 0, 180, 30)
    stress = st.slider("Stress Level (1–10)", 1, 10, 5)
    caffeine = st.slider("Caffeine Intake (cups/day)", 0, 10, 1)
with col3:
    alcohol = st.slider("Alcohol Intake (units/day)", 0, 10, 0)
    smoker = st.selectbox("Do you smoke?", ["No", "Yes"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)

col4, col5 = st.columns([1,1])
with col4:
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=220, value=70)
with col5:
    screen_time = st.slider("Screen Time Before Bed (hrs)", 0.0, 10.0, 2.0, 0.5)

col6, col7 = st.columns([1,1])
with col6:
    history = st.selectbox("Sleep Disorder History", ["No", "Yes"])
    wake_consistency = st.selectbox("Wake-up Consistency", ["Consistent", "Inconsistent"])
with col7:
    env_score = st.slider("Sleep Environment Score (1–10)", 1, 10, 7)
    water = st.slider("Daily Water Intake (litres)", 0.0, 5.0, 2.0, 0.5)

predict_clicked = st.button("🔍 Predict Sleep Quality")
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Prediction logic (stable)
# ------------------------------
if predict_clicked:
    if model is None or scaler is None:
        st.error("Model or scaler not available. Prediction cannot run.")
    else:
        # training mapping: Male -> 0, Female -> 1 (aligned with train_model.py)
        gender_val = 0 if gender == "Male" else 1
        smoking_val = 1 if smoker == "Yes" else 0
        hist_val = 1 if history == "Yes" else 0
        wake_val = 1 if wake_consistency == "Consistent" else 0

        input_df = pd.DataFrame({
            "Age": [age],
            "Gender": [gender_val],
            "Sleep Duration (hrs)": [sleep_duration],
            "Physical Activity (mins/day)": [activity],
            "Stress Level (1–10)": [stress],
            "Caffeine Intake (cups/day)": [caffeine],
            "Alcohol Intake (units/day)": [alcohol],
            "Smoking": [smoking_val],
            "Heart Rate (bpm)": [heart_rate],
            "Screen Time Before Bed (hrs)": [screen_time],
            "Sleep Disorder History": [hist_val],
            "BMI": [bmi],
            "Wake-up Consistency": [wake_val],
            "Sleep Environment Score (1–10)": [env_score],
            "Daily Water Intake (litres)": [water]
        })
        try:
            scaled = scaler.transform(input_df)
            pred = model.predict(scaled)[0]
            label_map = {0: "Poor", 1: "Fair", 2: "Good"}
            result = label_map.get(int(pred), "Unknown")
            st.markdown(f"<div class='result'>🌙 Predicted Sleep Quality: {result}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ------------------------------
# Insights row (simple, useful)
# ------------------------------
st.markdown("### Quick Insights")
st.markdown("<div class='tiles-row'>", unsafe_allow_html=True)
tiles = [
    ("Consistent Wake", "Sticking to a wake-up time helps circadian rhythm."),
    ("Bedroom Climate", "Cool (16-19°C), dark, and quiet improves deep sleep."),
    ("Wind-down Routine", "30-min pre-sleep routine reduces sleep latency."),
    ("Caffeine Cutoff", "Avoid caffeine 6-8 hours before bed."),
]
for title, text in tiles:
    st.markdown(f"<div class='tile'><h4>{title}</h4><p>{text}</p></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Floating chat bubble & window (collapsible)
# ------------------------------
# We use a st.empty() placeholder + session_state toggle so clicks work reliably.
chat_placeholder = st.empty()

with chat_placeholder.container():
    # When closed: show circular button in fixed position (placed by CSS)
    if not st.session_state.chat_open:
        # When the user clicks this button, we open the chat window.
        # The button is a normal Streamlit button so it updates session_state on click.
        open_button = st.button("💬", key="open_chat_button")
        if open_button:
            st.session_state.chat_open = True
    else:
        # Chat is open: show a compact chat window with history + input
        st.markdown("<div class='chat-window'>", unsafe_allow_html=True)
        # header with close action
        header_cols = st.columns([1,5,1])
        with header_cols[0]:
            st.write("")  # spacer
        with header_cols[1]:
            st.markdown("<div class='chat-header'><strong>Sleep Assistant</strong><div style='font-size:0.85rem;color:var(--muted)'>Powered by AI</div></div>", unsafe_allow_html=True)
        with header_cols[2]:
            if st.button("✕ Close", key="close_chat_button"):
                st.session_state.chat_open = False
        # messages area
        st.markdown("<div class='chat-messages'>", unsafe_allow_html=True)
        # show chat history
        for role, text in st.session_state.chat_history:
            if role == "You":
                st.markdown(f"<div class='msg-user'>{text}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='msg-bot'>{text}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # input box + send button
        input_col1, input_col2 = st.columns([4,1])
        with input_col1:
            user_message = st.text_input("Type a message", key="chat_input")
        with input_col2:
            send_clicked = st.button("Send", key="send_chat")
        # handle send
        if send_clicked and user_message and api_key:
            # append user message
            st.session_state.chat_history.append(("You", user_message))
            # simple throttle
            time.sleep(0.8)
            # call Gemini/GenAI if available, otherwise fallback canned reply
            try:
                history_struct = [
                    {"role": "user" if r == "You" else "assistant", "parts": [m]}
                    for r, m in st.session_state.chat_history
                ]
                chat_model = None
                try:
                    chat_model = genai.GenerativeModel("gemini-2.0-pro-exp")
                except Exception:
                    try:
                        chat_model = genai.GenerativeModel("gemini-2.0-flash-exp")
                    except Exception:
                        chat_model = None

                if chat_model is not None:
                    chat = chat_model.start_chat(history=history_struct)
                    response = chat.send_message(user_message)
                    reply_text = getattr(response, "text", "Sorry — no answer.")
                else:
                    reply_text = "AI service unavailable. Try later."
            except Exception as e:
                reply_text = f"Error contacting AI: {e}"

            st.session_state.chat_history.append(("Bot", reply_text))
            # Clear input for next message (streamlit text_input keeps value; we use key trick)
            st.session_state["chat_input"] = ""

        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# End footer
# ------------------------------
st.markdown("<hr style='opacity:.06'>", unsafe_allow_html=True)
st.caption("© SleepPro — Minimal, professional UI • Built with Streamlit")

