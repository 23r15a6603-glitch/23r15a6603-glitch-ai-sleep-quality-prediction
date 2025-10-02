import streamlit as st
import pandas as pd
import numpy as np
import joblib

import streamlit.components.v1 as components

# Load model and scaler
model = joblib.load("xgb_sleep_quality_model.pkl")
scaler = joblib.load("scaler_sleep_quality.pkl")

# Streamlit app
st.set_page_config(page_title="Sleep Quality Predictor", layout="wide")

# Main Title
st.markdown("<h1 style='text-align: center; color: #6C3483;'>Sleep Quality Predictor</h1>", unsafe_allow_html=True)

# Input form
with st.form("sleep_form"):
    st.subheader("Enter your Health and Lifestyle Data")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 10, 100, 25)
        gender = st.selectbox("Gender", ["Male", "Female"])
        sleep_duration = st.slider("Sleep Duration (hrs)", 0.0, 12.0, 7.0, 0.5)
        activity = st.slider("Physical Activity (mins/day)", 0, 180, 30)
        stress = st.slider("Stress Level (1–10)", 1, 10, 5)
        caffeine = st.slider("Caffeine Intake (cups/day)", 0, 10, 1)
        alcohol = st.slider("Alcohol Intake (units/day)", 0, 10, 0)

    with col2:
        smoker = st.selectbox("Do you smoke?", ["No", "Yes"])
        heart_rate = st.number_input("Heart Rate (bpm)", 40, 140, 70)
        screen_time = st.slider("Screen Time Before Bed (hrs)", 0.0, 10.0, 2.0, 0.5)
        bmi = st.number_input("BMI", 10.0, 50.0, 22.0)
        wake_consistency = st.selectbox("Wake-up Consistency", ["Regular", "Irregular"])
        env_score = st.slider("Sleep Environment Score (1–10)", 1, 10, 7)
        water = st.slider("Daily Water Intake (litres)", 0.0, 5.0, 2.0, 0.5)
        history = st.selectbox("Sleep Disorder History", ["No", "Yes"])

    submitted = st.form_submit_button("Predict")

# Function to generate personalized suggestions
def generate_suggestions(user_inputs, prediction):
    suggestions = []
    
    # Sleep Duration Analysis
    if user_inputs['sleep_duration'] < 7:
        suggestions.append("💤 **Increase Sleep Duration**: Aim for 7-9 hours of sleep nightly. Consider going to bed 30 minutes earlier.")
    elif user_inputs['sleep_duration'] > 9:
        suggestions.append("💤 **Optimize Sleep Duration**: While 7-9 hours is ideal, excessive sleep can sometimes indicate underlying issues. Maintain consistent sleep patterns.")
    
    # Physical Activity
    if user_inputs['activity'] < 30:
        suggestions.append("🏃 **Boost Physical Activity**: Try to get at least 30 minutes of moderate exercise daily. Even a brisk walk can improve sleep quality.")
    elif user_inputs['activity'] > 120:
        suggestions.append("⏰ **Time Your Workouts**: Intense late-evening exercise might disrupt sleep. Consider finishing workouts 2-3 hours before bedtime.")
    
    # Stress Level
    if user_inputs['stress'] >= 7:
        suggestions.append("🧘 **Manage Stress**: Practice mindfulness, deep breathing, or meditation. Consider keeping a worry journal to clear your mind before bed.")
    
    # Caffeine Intake
    if user_inputs['caffeine'] >= 3:
        suggestions.append("☕ **Reduce Caffeine**: Limit to 1-2 cups daily and avoid caffeine after 2 PM. Try switching to herbal tea in the afternoon.")
    
    # Alcohol Consumption
    if user_inputs['alcohol'] >= 2:
        suggestions.append("🍷 **Moderate Alcohol**: Alcohol disrupts sleep architecture. Limit to 1 drink daily and avoid within 3 hours of bedtime.")
    
    # Smoking
    if user_inputs['smoker'] == "Yes":
        suggestions.append("🚭 **Quit Smoking**: Nicotine is a stimulant that interferes with sleep. Consider smoking cessation programs or nicotine replacement therapy.")
    
    # Screen Time
    if user_inputs['screen_time'] >= 2:
        suggestions.append("📱 **Reduce Screen Time**: Limit screen exposure 1 hour before bed. Use blue light filters or switch to reading a physical book.")
    
    # BMI Analysis
    if user_inputs['bmi'] >= 25:
        suggestions.append("⚖️ **Healthy Weight**: Consider weight management through balanced diet and exercise, as excess weight can contribute to sleep apnea.")
    elif user_inputs['bmi'] < 18.5:
        suggestions.append("🍎 **Nutrition Focus**: Ensure adequate nutrition, as being underweight can also affect sleep quality and energy levels.")
    
    # Wake-up Consistency
    if user_inputs['wake_consistency'] == "Irregular":
        suggestions.append("⏰ **Consistent Schedule**: Try waking up at the same time every day, even on weekends. This regulates your body's internal clock.")
    
    # Sleep Environment
    if user_inputs['env_score'] <= 5:
        suggestions.append("🌙 **Improve Sleep Environment**: Optimize your bedroom for sleep - cool, dark, and quiet. Consider blackout curtains or white noise machines.")
    
    # Water Intake
    if user_inputs['water'] < 2:
        suggestions.append("💧 **Increase Hydration**: Aim for 2-3 liters of water daily, but reduce intake 1-2 hours before bed to minimize nighttime awakenings.")
    
    # Sleep Disorder History
    if user_inputs['history'] == "Yes":
        suggestions.append("👨‍⚕️ **Professional Consultation**: Consider consulting a sleep specialist for ongoing sleep issues and proper diagnosis.")
    
    # Heart Rate
    if user_inputs['heart_rate'] > 100:
        suggestions.append("❤️ **Monitor Heart Rate**: Elevated resting heart rate may indicate stress or other health issues. Practice relaxation techniques.")
    
    # Age-specific suggestions
    if user_inputs['age'] > 50:
        suggestions.append("👴 **Age-Appropriate Routine**: As we age, sleep patterns change. Maintain good sleep hygiene and consider shorter, more frequent rest periods if needed.")
    
    # Add motivational message based on prediction
    if prediction == "Good":
        suggestions.insert(0, "🎉 **Excellent!** Your sleep habits are great! Keep maintaining these healthy routines.")
    elif prediction == "Fair":
        suggestions.insert(0, "📈 **Good foundation!** With a few adjustments, you can achieve even better sleep quality.")
    else:  # Poor
        suggestions.insert(0, "🔄 **Time for positive changes!** Small improvements can make a big difference in your sleep quality.")
    
    return suggestions

# Predict and show suggestions
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

    # Scale inputs
    scaled_input = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_input)[0]
    label_map = {0: 'Poor', 1: 'Fair', 2: 'Good'}
    result = label_map[prediction]

    # Display result
    st.success(f"Predicted Sleep Quality: **{result}**")
    
    # Generate and display personalized suggestions
    st.markdown("---")
    st.subheader("💡 Personalized Sleep Improvement Suggestions")
    
    user_inputs = {
        'sleep_duration': sleep_duration,
        'activity': activity,
        'stress': stress,
        'caffeine': caffeine,
        'alcohol': alcohol,
        'smoker': smoker,
        'screen_time': screen_time,
        'bmi': bmi,
        'wake_consistency': wake_consistency,
        'env_score': env_score,
        'water': water,
        'history': history,
        'heart_rate': heart_rate,
        'age': age
    }
    
    suggestions = generate_suggestions(user_inputs, result)
    
    # Display suggestions in a nice format
    for i, suggestion in enumerate(suggestions, 1):
        st.markdown(f"{i}. {suggestion}")
    
    # Progress tracker
    st.markdown("---")
    st.subheader("📊 Your Sleep Health Snapshot")
    
    # Create a simple progress visualization
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sleep_progress = min(sleep_duration / 8 * 100, 100)
        st.metric("Sleep Duration", f"{sleep_duration} hrs", 
                 delta="Optimal" if sleep_duration >= 7 else "Needs Improvement")
    
    with col2:
        stress_color = "🟢" if stress <= 3 else "🟡" if stress <= 6 else "🔴"
        st.metric("Stress Level", f"{stress}/10 {stress_color}")
    
    with col3:
        activity_status = "Active" if activity >= 30 else "Sedentary"
        st.metric("Physical Activity", f"{activity} mins", delta=activity_status)

st.markdown("---")
st.subheader("Get Tips With Our AI Bot")

# ... (keep your existing chatbot HTML code as is) ...

chatbot_html = """
<style>
  .chat-container {
    width: 100%;
    height: 450px;
    background: #ffffff;
    border: 1px solid #ddd;
    border-radius: 10px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    font-family: "Helvetica", "Arial", sans-serif;
  }
  .chat-header {
    background: #f0f2f6;
    padding: 12px;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
  }
  .chat-box {
    flex: 1;
    padding: 12px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .message {
    padding: 8px 12px;
    border-radius: 8px;
    max-width: 75%;
    line-height: 1.4;
    font-size: 14px;
  }
  .user {
    background: #e8f0fe;
    align-self: flex-end;
  }
  .assistant {
    background: #f1f3f4;
    align-self: flex-start;
  }
  .input-area {
    display: flex;
    border-top: 1px solid #ddd;
    padding: 10px;
    background: #fafafa;
  }
  #user-input {
    flex: 1;
    padding: 8px;
    border: 1px solid #ccc;
    border-radius: 5px;
    font-size: 14px;
  }
  #send-btn {
    margin-left: 10px;
    padding: 8px 15px;
    background: #4a90e2;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-size: 14px;
  }
  #send-btn:hover {
    background: #357ab8;
  }
</style>

<div class="chat-container">
  <div class="chat-header"></div>
  <div id="chat-box" class="chat-box"></div>
  <div class="input-area">
    <input type="text" id="user-input" placeholder="Ask your question here..." />
    <button id="send-btn">Send</button>
  </div>
</div>

<script>
  const chatBox = document.getElementById("chat-box");
  const userInput = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");

  function appendMessage(role, content) {
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message", role);
    msgDiv.innerHTML = content.replace(/\\n/g, "<br>");
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  sendBtn.addEventListener("click", () => {
    const msg = userInput.value.trim();
    if (!msg) return;
    appendMessage("user", msg);
    userInput.value = "";

    // Extended predefined multi-tip responses
    setTimeout(() => {
      let reply = "🤔 Sorry, I can only answer questions about sleep, health, and relaxation. Try asking me about stress, insomnia, routine, or diet!";
      const m = msg.toLowerCase();

      if (m.includes("hi") || m.includes("hello") || m.includes("hey")) reply = "👋 Hi there! How can I help you with your sleep today?";
  else if (m.includes("stress")) reply = "🧘 Stress relief tips: <br>1. Practice deep breathing or meditation <br>2. Write thoughts in a journal before bed <br>3. Do light stretches to relax muscles.";
      else if (m.includes("screen")) reply = "📱 Screen time advice: <br>1. Avoid phones/TV 1 hour before bed <br>2. Use blue light filters if needed <br>3. Try reading a book instead.";
      else if (m.includes("insomnia")) reply = "💡 Insomnia management: <br>1. Stick to a consistent sleep schedule <br>2. Avoid caffeine/alcohol late in the day <br>3. Use your bed only for sleep, not work.";
      else if (m.includes("caffeine")) reply = "☕ Caffeine tips: <br>1. Avoid coffee after 2 PM <br>2. Switch to herbal tea in evenings <br>3. Watch out for hidden caffeine in sodas/energy drinks.";
      else if (m.includes("alcohol")) reply = "🍷 Alcohol & sleep: <br>1. Alcohol can cause lighter, disrupted sleep <br>2. Avoid drinking right before bed <br>3. Stay hydrated with water if you drink.";
      else if (m.includes("exercise")) reply = "🏃 Exercise & sleep: <br>1. Regular workouts improve sleep quality <br>2. Best time: morning or afternoon <br>3. Avoid intense exercise right before bed.";
      else if (m.includes("food") || m.includes("diet")) reply = "🥗 Food & sleep: <br>1. Avoid heavy/spicy meals late at night <br>2. A light snack (banana, warm milk) can help <br>3. Limit sugar before bedtime.";
      else if (m.includes("temperature") || m.includes("room")) reply = "🌡️ Ideal sleep environment: <br>1. Keep your room cool (18–22°C) <br>2. Use blackout curtains <br>3. Reduce noise with earplugs/white noise.";
      else if (m.includes("routine")) reply = "📅 Routine tips: <br>1. Sleep/wake up at the same time daily <br>2. Create a calming pre-bed routine <br>3. Avoid naps too late in the day.";
      else if (m.includes("water") || m.includes("drink")) reply = "💧 Hydration advice: <br>1. Drink water throughout the day <br>2. Avoid large amounts right before bed <br>3. Herbal teas (chamomile) can help sleep.";
      else if (m.includes("wake up") || m.includes("morning")) reply = "⏰ Morning energy tips: <br>1. Place your alarm across the room <br>2. Open curtains for sunlight immediately <br>3. Do light stretches to wake your body.";
      else if (m.includes("nap")) reply = "😴 Nap tips: <br>1. Keep naps under 30 minutes <br>2. Best nap time: 1–3 PM <br>3. Avoid napping too close to bedtime.";
      else if (m.includes("dream")) reply = "💭 Dreams & sleep: <br>1. Stress can trigger vivid dreams <br>2. A regular schedule supports healthy REM sleep <br>3. Avoid heavy food before bed to reduce nightmares.";
      else if (m.includes("mental health") || m.includes("anxiety")) reply = "❤️ Mental health & sleep: <br>1. Journaling before bed can reduce racing thoughts <br>2. Try mindfulness or gratitude practice <br>3. Seek help if anxiety regularly affects sleep.";
      else if (m.includes("light")) reply = "☀️ Light & sleep: <br>1. Get sunlight in the morning <br>2. Keep evenings dim to trigger melatonin <br>3. Use blackout curtains for deep sleep.";
      else if (m.includes("melatonin")) reply = "💊 Melatonin info: <br>1. Useful for jet lag or shift work <br>2. Take only under medical guidance <br>3. Best taken 30–60 mins before sleep.";
      else if (m.includes("shift work")) reply = "🌙 Shift work survival: <br>1. Keep a consistent sleep schedule <br>2. Use blackout curtains during the day <br>3. Limit caffeine towards end of shift.";
      else if (m.includes("music")) reply = "🎶 Sleep music: <br>1. Calm instrumental or white noise helps <br>2. Avoid loud/fast music <br>3. Nature sounds can also be relaxing.";
      else if (m.includes("relax")) reply = "🛀 Relaxation ideas: <br>1. Take a warm shower <br>2. Read a light book <br>3. Try guided meditation apps.";
      else if (m.includes("pillows") || m.includes("bed")) reply = "🛏️ Bed comfort: <br>1. Choose a supportive mattress <br>2. Replace old pillows regularly <br>3. Keep bedding clean & cozy.";
      else if (m.includes("bye") || m.includes("good night")) reply = "🌌 Good night! Sleep well and wake refreshed. 🌙";

      appendMessage("assistant", reply);
    }, 600);
  });
</script>
"""

components.html(chatbot_html, height=500, scrolling=False)
