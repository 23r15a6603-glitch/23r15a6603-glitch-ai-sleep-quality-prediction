
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
    except Exception:
        api_key = None

if api_key:
    try:
        genai.configure(api_key=api_key)
    except Exception:
        pass
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
# Page Config
# ------------------------------
st.set_page_config(page_title="Sleep Quality • Pro", page_icon="🌙", layout="wide")

# ------------------------------
# Theme System (Light/Dark toggle)
# ------------------------------
if "theme_dark" not in st.session_state:
    st.session_state.theme_dark = False  # default Light

def _inject_theme(dark: bool):
    # Professional palette — tuned for contrast in both modes
    if dark:
        css = """
        <style>
        :root{
            --bg:#0b0f1a;
            --bg-soft:#121826;
            --text:#e6eaf2;
            --muted:#9aa3b2;
            --card:#0f1523;
            --border:#1f2a44;
            --accent:#1ABC9C;
            --accent-2:#6C3483;
            --shadow:0 12px 28px rgba(0,0,0,.55);
            --shadow-soft:0 8px 20px rgba(0,0,0,.35);
        }
        </style>
        """
    else:
        css = """
        <style>
        :root{
            --bg:#ffffff;
            --bg-soft:#f6f7fb;
            --text:#0b0f1a;
            --muted:#5a6473;
            --card:#ffffff;
            --border:#e6e9f2;
            --accent:#6C3483;
            --accent-2:#1ABC9C;
            --shadow:0 10px 26px rgba(17,17,17,.07);
            --shadow-soft:0 6px 16px rgba(17,17,17,.06);
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

_inject_theme = _inject_theme  # alias
_inject_theme(st.session_state.theme_dark)

# ------------------------------
# Global Styles (Hotstar-like: sticky nav, hero, carousels, cards)
# ------------------------------
st.markdown("""
<style>
/* App background & text */
[data-testid="stAppViewContainer"]{
    background: var(--bg);
    color: var(--text);
}
[data-testid="stHeader"]{ background: transparent; }

/* Sticky Navigation */
.navbar{
    position: sticky; top: 0; z-index: 1000;
    background: linear-gradient(90deg, var(--card), var(--bg-soft));
    border-bottom: 1px solid var(--border);
    box-shadow: var(--shadow-soft);
    border-radius: 0 0 18px 18px;
    padding: 12px 16px;
}
.navbrand{
    display:flex; align-items:center; gap:10px; font-weight:800; letter-spacing:.2px;
    font-size: 1.15rem;
}
.navlinks{
    display:flex; gap:18px; align-items:center; justify-content:center; flex-wrap:wrap;
}
.navlinks a{
    text-decoration:none; color:var(--muted); font-weight:600; padding:8px 10px;
    border-radius:10px;
}
.navlinks a:hover{ color:var(--text); background:var(--bg-soft); }
.theme-toggle{
    display:flex; align-items:center; justify-content:flex-end;
}

/* Hero */
.hero{
    border-radius: 28px;
    padding: 56px 28px;
    background: radial-gradient(100% 120% at 0% 0%, var(--accent) 0%, transparent 55%),
                radial-gradient(100% 120% at 100% 0%, var(--accent-2) 0%, transparent 55%),
                linear-gradient(135deg, var(--bg-soft), var(--card));
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
}
.hero h1{
    font-size: 2.6rem; line-height: 1.15; margin: 0 0 8px 0;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 900;
}
.hero p{ color: var(--muted); font-size: 1.05rem; margin: 0; }

.cta-btn button{
    border-radius: 12px;
    background: linear-gradient(90deg, var(--accent), var(--accent-2))!important;
    color: #fff!important; font-weight: 800!important;
    border: none!important;
}

/* Section headings */
.section-title{
    font-weight: 800; font-size: 1.3rem; margin: 8px 0 14px 0;
}

/* Carousel (horizontal row like Hotstar) */
.row{
    display: flex; gap: 16px; overflow-x: auto; padding: 6px 2px 4px 2px;
    scroll-snap-type: x mandatory;
}
.row::-webkit-scrollbar{ height: 8px; }
.row::-webkit-scrollbar-thumb{
    background: var(--border); border-radius: 999px;
}
.tile{
    min-width: 220px; max-width: 240px; scroll-snap-align: start;
    background: var(--card); border: 1px solid var(--border);
    border-radius: 18px; box-shadow: var(--shadow-soft);
    padding: 14px;
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
.tile:hover{
    transform: translateY(-4px);
    box-shadow: var(--shadow);
    border-color: transparent;
}
.tile .t-eyebrow{ font-size:.8rem; color: var(--muted); margin-bottom: 4px; }
.tile .t-title{ font-weight: 800; }
.tile .t-foot{ font-size:.8rem; color: var(--muted); margin-top: 6px; }

/* Card shell for forms and chat */
.card{
    background: var(--card); border: 1px solid var(--border);
    border-radius: 20px; padding: 22px; box-shadow: var(--shadow-soft);
}

/* Result banner */
.result{
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    color:#fff; border-radius: 18px; padding: 22px; text-align:center;
    font-weight: 900; font-size: 1.2rem;
    box-shadow: var(--shadow);
    animation: rise .6s ease both;
}
@keyframes rise{
  from{ opacity:0; transform: translateY(14px) }
  to{ opacity:1; transform: translateY(0) }
}

/* Buttons */
.stButton>button{
    border-radius:12px; font-weight:800;
    border:1px solid transparent;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    color:#fff;
}
.stButton>button:hover{ filter: brightness(1.05); }

/* Sidebar */
[data-testid="stSidebar"]{
    background: var(--bg-soft);
    border-right: 1px solid var(--border);
}

/* Inputs */
.css-1dp5vir, .stSelectbox, .stSlider, .stNumberInput{
    color: var(--text);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Navbar
# ------------------------------
with st.container():
    c1, c2, c3 = st.columns([1.2, 4, 1.2])
    with c1:
        st.markdown(
            "<div class='navbar'><div class='navbrand'>🌙 SleepPro</div></div>",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            "<div class='navbar' style='background:transparent; box-shadow:none; border:none;'>"
            "<div class='navlinks'>"
            "<a href='#home'>Home</a>"
            "<a href='#predictor'>Predictor</a>"
            "<a href='#insights'>Insights</a>"
            "<a href='#chatbot'>Chat</a>"
            "</div></div>",
            unsafe_allow_html=True
        )
    with c3:
        st.markdown("<div class='navbar theme-toggle'>", unsafe_allow_html=True)
        dark_now = st.toggle("🌙 Dark mode", value=st.session_state.theme_dark, help="Toggle theme (Hotstar-like)")
        st.markdown("</div>", unsafe_allow_html=True)

if dark_now != st.session_state.theme_dark:
    st.session_state.theme_dark = dark_now
    _inject_theme(st.session_state.theme_dark)

# ------------------------------
# Hero
# ------------------------------
st.markdown("<span id='home'></span>", unsafe_allow_html=True)
with st.container():
    cta1, cta2 = st.columns([2.8, 1.2])
    with cta1:
        st.markdown(
            "<div class='hero'>"
            "<h1>AI Sleep Quality — like a pro</h1>"
            "<p>Hotstar-style UI. Medical-grade features. Instant insights.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with cta2:
        st.markdown("<div class='hero'>", unsafe_allow_html=True)
        st.metric("Users", "2,000+")
        st.metric("Avg. Sleep", "6.9 hrs")
        st.metric("Models", "XGBoost")
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Content rows (Hotstar-like carousels)
# ------------------------------
st.markdown("### Explore Sleep Topics")
row1 = st.container()
with row1:
    st.markdown("<div class='row'>", unsafe_allow_html=True)
    for t in [
        ("Sleep Hygiene", "Basics to sleep better", "Start tonight"),
        ("Deep Sleep Boost", "Habits that help", "Actionable tips"),
        ("Stress & Sleep", "Manage cortisol levels", "Guided steps"),
        ("Caffeine Timing", "Cutoff & rhythms", "Daily routine"),
        ("Screen Detox", "Reduce blue light", "Bedtime ritual"),
        ("Heart Rate", "Understand your bpm", "Recovery cues"),
        ("Hydration", "Water & sleep link", "Optimal range"),
    ]:
        st.markdown(
            f"""
            <div class='tile'>
              <div class='t-eyebrow'>{t[2]}</div>
              <div class='t-title'>{t[0]}</div>
              <div class='t-foot'>{t[1]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Predictor Section
# ------------------------------
st.markdown("<span id='predictor'></span>", unsafe_allow_html=True)
with st.form("sleep_form"):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📝 Your Health & Lifestyle</div>", unsafe_allow_html=True)

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

    submit = st.form_submit_button("🔍 Predict Sleep Quality", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Prediction
# ------------------------------
if submit:
    if model is not None and scaler is not None:
        # NOTE: Gender encoding aligned with training: Male->0, Female->1
        gender_val = 0 if gender == "Male" else 1
        smoking_val = 1 if smoker == "Yes" else 0
        history_val = 1 if history == "Yes" else 0
        wake_val = 1 if wake_consistency == "Consistent" else 0

        input_df = pd.DataFrame({
            'Age': [age],
            'Gender': [gender_val],
            'Sleep Duration (hrs)': [sleep_duration],
            'Physical Activity (mins/day)': [activity],
            'Stress Level (1–10)': [stress],
            'Caffeine Intake (cups/day)': [caffeine],
            'Alcohol Intake (units/day)': [alcohol],
            'Smoking': [smoking_val],
            'Heart Rate (bpm)': [heart_rate],
            'Screen Time Before Bed (hrs)': [screen_time],
            'Sleep Disorder History': [history_val],
            'BMI': [bmi],
            'Wake-up Consistency': [wake_val],
            'Sleep Environment Score (1–10)': [env_score],
            'Daily Water Intake (litres)': [water],
        })

        try:
            scaled = scaler.transform(input_df)
            pred = model.predict(scaled)[0]
            label_map = {0: "Poor", 1: "Fair", 2: "Good"}
            result = label_map.get(int(pred), "Unknown")

            st.markdown(f"<div class='result'>🌙 Predicted Sleep Quality: {result}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.error("⚠ Prediction unavailable. Model or scaler missing.")

# ------------------------------
# Insights Row
# ------------------------------
st.markdown("<span id='insights'></span>", unsafe_allow_html=True)
st.markdown("### Personalized Insights")
with st.container():
    st.markdown("<div class='row'>", unsafe_allow_html=True)
    for t in [
        ("Circadian Rhythm", "Wake at the same time", "Consistency wins"),
        ("Bedroom Climate", "Cool, dark, quiet", "Score 7–8/10"),
        ("Wind-down Routine", "20–30 mins pre-sleep", "Screens off"),
        ("Caffeine Cutoff", "6–8 hours before bed", "Try 2pm"),
        ("Hydration Timing", "Front-load daytime", "Reduce at night"),
    ]:
        st.markdown(
            f"""
            <div class='tile'>
              <div class='t-eyebrow'>{t[2]}</div>
              <div class='t-title'>{t[0]}</div>
              <div class='t-foot'>{t[1]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Chatbot Section
# ------------------------------
st.markdown("<span id='chatbot'></span>", unsafe_allow_html=True)
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>💬 Sleep AI Chat Assistant</div>", unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_text = st.text_input("Ask your question here:")

    colA, colB = st.columns([1, 1])
    with colA:
        send = st.button("Send")
    with colB:
        clear = st.button("Clear Chat")

    if clear:
        st.session_state.chat_history = []

    if send and api_key:
        if user_text.strip():
            try:
                time.sleep(1.0)  # gentle throttle
                history = [{"role": "user" if r == "You" else "model", "parts": [m]} for r, m in st.session_state.chat_history]

                try:
                    chat_model = genai.GenerativeModel("gemini-2.0-pro-exp")
                    chat = chat_model.start_chat(history=history)
                except Exception:
                    chat_model = genai.GenerativeModel("gemini-2.0-flash-exp")
                    chat = chat_model.start_chat(history=history)

                response = chat.send_message(user_text)
                st.session_state.chat_history.append(("You", user_text))
                st.session_state.chat_history.append(("Bot", getattr(response, "text", "No response")))
            except Exception as e:
                st.session_state.chat_history.append(("Bot", f"⚠ Chatbot error: {e}"))

    for role, msg in st.session_state.chat_history:
        if role == "You":
            st.info(f"🧑 {msg}")
        else:
            st.success(f"🤖 {msg}")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Footer
# ------------------------------
st.markdown("<hr style='border:1px solid var(--border); opacity:.5'/>", unsafe_allow_html=True)
st.caption("© SleepPro • Built with Streamlit")
