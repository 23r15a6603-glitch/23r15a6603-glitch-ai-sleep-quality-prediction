import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import streamlit.components.v1 as components

# Load model and scaler
model = joblib.load("xgb_sleep_quality_model.pkl")
scaler = joblib.load("scaler_sleep_quality.pkl")

# Streamlit app configuration
st.set_page_config(
    page_title="Sleep Quality Predictor",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #6C3483;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .good-sleep {
        background-color: #D5F5E3;
        color: #186A3B;
        border: 2px solid #27AE60;
    }
    .fair-sleep {
        background-color: #FCF3CF;
        color: #7D6608;
        border: 2px solid #F39C12;
    }
    .poor-sleep {
        background-color: #FADBD8;
        color: #943126;
        border: 2px solid #E74C3C;
    }
    .suggestion-card {
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        background-color: #F8F9FA;
        border-left: 4px solid #6C3483;
    }
    .metric-card {
        padding: 15px;
        border-radius: 8px;
        background-color: white;
        border: 1px solid #E0E0E0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Main Title
st.markdown("<h1 class='main-header'>🌙 Sleep Quality Predictor</h1>", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.header("About This App")
    st.markdown("""
    This app predicts your sleep quality based on lifestyle and health factors using machine learning.
    
    **How it works:**
    1. Fill out the form with your details
    2. Get your sleep quality prediction
    3. Receive personalized suggestions
    4. Chat with our AI sleep assistant
    
    **Ideal Ranges:**
    - Sleep: 7-9 hours
    - Activity: 30+ minutes
    - Stress: ≤ 5/10
    - Caffeine: ≤ 2 cups
    - BMI: 18.5-24.9
    """)
    
    st.header("Health Tips")
    st.info("💡 **Quick Tip**: Consistent sleep schedules improve quality more than longer, irregular sleep.")
    st.info("💧 **Hydration**: Drink water throughout the day, but reduce 2 hours before bed.")
    st.info("📱 **Digital Detox**: Avoid screens 1 hour before sleep for better rest.")

# Main content area
tab1, tab2, tab3 = st.tabs(["📊 Sleep Prediction", "📈 Health Dashboard", "💬 Sleep Assistant"])

with tab1:
    # Input form
    with st.form("sleep_form"):
        st.subheader("Enter your Health and Lifestyle Data")
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", 10, 100, 25, help="Your current age")
            gender = st.selectbox("Gender", ["Male", "Female"])
            sleep_duration = st.slider("Sleep Duration (hrs)", 0.0, 12.0, 7.0, 0.5, 
                                     help="Average hours of sleep per night")
            activity = st.slider("Physical Activity (mins/day)", 0, 180, 30,
                               help="Daily moderate to vigorous exercise")
            stress = st.slider("Stress Level (1–10)", 1, 10, 5,
                             help="Perceived stress level where 1=low, 10=high")
            caffeine = st.slider("Caffeine Intake (cups/day)", 0, 10, 1,
                               help="Coffee, tea, energy drinks equivalent")
            alcohol = st.slider("Alcohol Intake (units/day)", 0, 10, 0,
                              help="1 unit = 1 beer, 1 glass wine, or 1 shot")

        with col2:
            smoker = st.selectbox("Do you smoke?", ["No", "Yes"])
            heart_rate = st.number_input("Heart Rate (bpm)", 40, 140, 70,
                                       help="Resting heart rate")
            screen_time = st.slider("Screen Time Before Bed (hrs)", 0.0, 10.0, 2.0, 0.5,
                                  help="Screen exposure in the hour before sleep")
            bmi = st.number_input("BMI", 10.0, 50.0, 22.0, 0.1,
                                help="Body Mass Index - weight(kg)/height(m)²")
            wake_consistency = st.selectbox("Wake-up Consistency", ["Regular", "Irregular"],
                                          help="Do you wake up at consistent times?")
            env_score = st.slider("Sleep Environment Score (1–10)", 1, 10, 7,
                                help="1=poor (noisy, bright, uncomfortable), 10=excellent")
            water = st.slider("Daily Water Intake (litres)", 0.0, 5.0, 2.0, 0.5)
            history = st.selectbox("Sleep Disorder History", ["No", "Yes"])

        submitted = st.form_submit_button("Predict Sleep Quality", use_container_width=True)

    # Function to generate personalized suggestions
    def generate_suggestions(user_inputs, prediction):
        suggestions = []
        
        # Sleep Duration Analysis
        if user_inputs['sleep_duration'] < 7:
            suggestions.append({
                "icon": "💤",
                "title": "Increase Sleep Duration",
                "message": f"You're getting {user_inputs['sleep_duration']} hours. Aim for 7-9 hours. Try going to bed 30 minutes earlier each night.",
                "priority": "high"
            })
        elif user_inputs['sleep_duration'] > 9:
            suggestions.append({
                "icon": "⏰",
                "title": "Optimize Sleep Duration",
                "message": "While 7-9 hours is ideal, excessive sleep can indicate underlying issues. Maintain consistent sleep patterns.",
                "priority": "medium"
            })
        
        # Physical Activity
        if user_inputs['activity'] < 30:
            suggestions.append({
                "icon": "🏃",
                "title": "Boost Physical Activity",
                "message": f"Currently {user_inputs['activity']} minutes. Aim for at least 30 minutes daily. Even brisk walking counts!",
                "priority": "high"
            })
        elif user_inputs['activity'] > 120:
            suggestions.append({
                "icon": "⏰",
                "title": "Time Your Workouts",
                "message": "Intense late-evening exercise might disrupt sleep. Finish workouts 2-3 hours before bedtime.",
                "priority": "medium"
            })
        
        # Stress Level
        if user_inputs['stress'] >= 7:
            suggestions.append({
                "icon": "🧘",
                "title": "Manage Stress",
                "message": "High stress levels (7+) affect sleep quality. Try mindfulness, deep breathing, or journaling before bed.",
                "priority": "high"
            })
        
        # Caffeine Intake
        if user_inputs['caffeine'] >= 3:
            suggestions.append({
                "icon": "☕",
                "title": "Reduce Caffeine",
                "message": f"Currently {user_inputs['caffeine']} cups daily. Limit to 1-2 cups and avoid caffeine after 2 PM.",
                "priority": "medium"
            })
        
        # Alcohol Consumption
        if user_inputs['alcohol'] >= 2:
            suggestions.append({
                "icon": "🍷",
                "title": "Moderate Alcohol",
                "message": f"Currently {user_inputs['alcohol']} units daily. Alcohol disrupts sleep architecture. Avoid within 3 hours of bedtime.",
                "priority": "medium"
            })
        
        # Smoking
        if user_inputs['smoker'] == "Yes":
            suggestions.append({
                "icon": "🚭",
                "title": "Quit Smoking",
                "message": "Nicotine is a stimulant that interferes with sleep. Consider smoking cessation programs.",
                "priority": "high"
            })
        
        # Screen Time
        if user_inputs['screen_time'] >= 2:
            suggestions.append({
                "icon": "📱",
                "title": "Reduce Screen Time",
                "message": f"Currently {user_inputs['screen_time']} hours before bed. Limit to 1 hour and use blue light filters.",
                "priority": "medium"
            })
        
        # BMI Analysis
        if user_inputs['bmi'] >= 25:
            suggestions.append({
                "icon": "⚖️",
                "title": "Healthy Weight",
                "message": f"BMI of {user_inputs['bmi']} indicates overweight. Excess weight can contribute to sleep apnea.",
                "priority": "medium"
            })
        elif user_inputs['bmi'] < 18.5:
            suggestions.append({
                "icon": "🍎",
                "title": "Nutrition Focus",
                "message": f"BMI of {user_inputs['bmi']} indicates underweight. Ensure adequate nutrition for quality sleep.",
                "priority": "medium"
            })
        
        # Wake-up Consistency
        if user_inputs['wake_consistency'] == "Irregular":
            suggestions.append({
                "icon": "⏰",
                "title": "Consistent Schedule",
                "message": "Irregular wake-up times disrupt your body clock. Try waking at the same time daily, even on weekends.",
                "priority": "medium"
            })
        
        # Sleep Environment
        if user_inputs['env_score'] <= 5:
            suggestions.append({
                "icon": "🌙",
                "title": "Improve Sleep Environment",
                "message": f"Environment score: {user_inputs['env_score']}/10. Optimize for cool, dark, and quiet conditions.",
                "priority": "medium"
            })
        
        # Water Intake
        if user_inputs['water'] < 2:
            suggestions.append({
                "icon": "💧",
                "title": "Increase Hydration",
                "message": f"Currently {user_inputs['water']}L daily. Aim for 2-3L, but reduce 1-2 hours before bed.",
                "priority": "low"
            })
        
        # Sleep Disorder History
        if user_inputs['history'] == "Yes":
            suggestions.append({
                "icon": "👨‍⚕️",
                "title": "Professional Consultation",
                "message": "Consider consulting a sleep specialist for ongoing sleep issues and proper diagnosis.",
                "priority": "high"
            })
        
        # Heart Rate
        if user_inputs['heart_rate'] > 100:
            suggestions.append({
                "icon": "❤️",
                "title": "Monitor Heart Rate",
                "message": f"Resting heart rate of {user_inputs['heart_rate']} bpm is elevated. Practice relaxation techniques.",
                "priority": "medium"
            })
        
        # Age-specific suggestions
        if user_inputs['age'] > 50:
            suggestions.append({
                "icon": "👴",
                "title": "Age-Appropriate Routine",
                "message": "As we age, sleep patterns change. Maintain good sleep hygiene and consider shorter rest periods.",
                "priority": "low"
            })
        
        # Sort by priority
        priority_order = {"high": 1, "medium": 2, "low": 3}
        suggestions.sort(key=lambda x: priority_order[x["priority"]])
        
        return suggestions

    # Predict and show results
    if submitted:
        # Create input data for model
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

        # Scale inputs and predict
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        label_map = {0: 'Poor', 1: 'Fair', 2: 'Good'}
        result = label_map[prediction]

        # Display prediction result
        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        
        prediction_class = ""
        if result == "Good":
            prediction_class = "good-sleep"
            emoji = "😊"
        elif result == "Fair":
            prediction_class = "fair-sleep"
            emoji = "😐"
        else:
            prediction_class = "poor-sleep"
            emoji = "😔"
        
        st.markdown(f'<div class="prediction-box {prediction_class}">{emoji} Predicted Sleep Quality: {result}</div>', 
                   unsafe_allow_html=True)

        # Generate and display personalized suggestions
        st.markdown("---")
        st.subheader("💡 Personalized Improvement Suggestions")
        
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
        
        # Display suggestions in columns
        for i, suggestion in enumerate(suggestions, 1):
            with st.container():
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.markdown(f"### {suggestion['icon']}")
                    st.caption(suggestion['priority'].upper())
                with col2:
                    st.write(f"**{suggestion['title']}**")
                    st.write(suggestion['message'])
                st.markdown("---")

with tab2:
    st.header("📊 Your Sleep Health Dashboard")
    
    if submitted:
        # Create radar chart for sleep health metrics
        categories = ['Sleep Duration', 'Physical Activity', 'Stress Level', 
                     'Sleep Environment', 'Lifestyle Balance']
        
        # Normalize values for radar chart (invert stress so lower is better)
        sleep_norm = min(sleep_duration / 9 * 100, 100)
        activity_norm = min(activity / 120 * 100, 100)
        stress_norm = (10 - stress) / 9 * 100  # Invert so lower stress = higher score
        environment_norm = env_score / 10 * 100
        
        # Lifestyle balance (average of caffeine, alcohol, screen time inverses)
        lifestyle_norm = 100 - ((caffeine / 10 * 100 + alcohol / 10 * 100 + screen_time / 10 * 100) / 3)
        
        values = [sleep_norm, activity_norm, stress_norm, environment_norm, lifestyle_norm]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Your Score',
            line=dict(color='#6C3483')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title="Sleep Health Radar Chart",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Key metrics in columns
        st.subheader("📈 Health Metrics Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            sleep_status = "Optimal" if sleep_duration >= 7 else "Insufficient"
            sleep_color = "#27AE60" if sleep_duration >= 7 else "#E74C3C"
            st.metric("Sleep Duration", f"{sleep_duration} hrs", sleep_status,
                     delta_color="normal" if sleep_duration >= 7 else "inverse")
        
        with col2:
            activity_status = "Active" if activity >= 30 else "Sedentary"
            st.metric("Physical Activity", f"{activity} mins", activity_status)
        
        with col3:
            stress_color = "🟢" if stress <= 3 else "🟡" if stress <= 6 else "🔴"
            st.metric("Stress Level", f"{stress}/10 {stress_color}")
        
        with col4:
            bmi_category = "Normal" if 18.5 <= bmi <= 24.9 else "Underweight" if bmi < 18.5 else "Overweight"
            st.metric("BMI", f"{bmi:.1f}", bmi_category)
        
        # Progress bars for key metrics
        st.subheader("🎯 Progress Towards Goals")
        
        # Sleep duration progress
        sleep_goal = 8
        sleep_progress = min(sleep_duration / sleep_goal * 100, 100)
        st.write(f"**Sleep Duration Goal ({sleep_goal} hours):**")
        st.progress(sleep_progress / 100)
        st.caption(f"{sleep_duration} / {sleep_goal} hours ({sleep_progress:.1f}%)")
        
        # Activity progress
        activity_goal = 30
        activity_progress = min(activity / activity_goal * 100, 100)
        st.write(f"**Physical Activity Goal ({activity_goal} minutes):**")
        st.progress(activity_progress / 100)
        st.caption(f"{activity} / {activity_goal} minutes ({activity_progress:.1f}%)")
        
        # Stress progress (inverted)
        stress_goal = 3
        stress_progress = max(0, (10 - stress) / (10 - stress_goal) * 100)
        st.write(f"**Stress Reduction Goal (≤{stress_goal}/10):**")
        st.progress(stress_progress / 100)
        st.caption(f"Current: {stress}/10 | Target: ≤{stress_goal}/10")

    else:
        st.info("👆 Please fill out the prediction form first to see your personalized health dashboard.")

with tab3:
    st.header("💬 AI Sleep Assistant")
    st.markdown("Chat with our AI assistant for personalized sleep tips and advice!")
    
    chatbot_html = """
    <style>
      .chat-container {
        width: 100%;
        height: 500px;
        background: #ffffff;
        border: 1px solid #ddd;
        border-radius: 10px;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        font-family: "Helvetica", "Arial", sans-serif;
      }
      .chat-header {
        background: #6C3483;
        color: white;
        padding: 15px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
      }
      .chat-box {
        flex: 1;
        padding: 15px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 12px;
        background: #fafafa;
      }
      .message {
        padding: 12px 16px;
        border-radius: 12px;
        max-width: 80%;
        line-height: 1.4;
        font-size: 14px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
      }
      .user {
        background: #6C3483;
        color: white;
        align-self: flex-end;
        border-bottom-right-radius: 4px;
      }
      .assistant {
        background: white;
        border: 1px solid #e0e0e0;
        align-self: flex-start;
        border-bottom-left-radius: 4px;
      }
      .input-area {
        display: flex;
        border-top: 1px solid #ddd;
        padding: 15px;
        background: #f8f9fa;
      }
      #user-input {
        flex: 1;
        padding: 12px;
        border: 1px solid #ccc;
        border-radius: 8px;
        font-size: 14px;
        margin-right: 10px;
      }
      #send-btn {
        padding: 12px 20px;
        background: #6C3483;
        color: white;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 14px;
        font-weight: bold;
      }
      #send-btn:hover {
        background: #5B2C6F;
      }
      .timestamp {
        font-size: 11px;
        color: #888;
        margin-top: 5px;
      }
    </style>

    <div class="chat-container">
      <div class="chat-header">🛌 Sleep Health Assistant</div>
      <div id="chat-box" class="chat-box">
        <div class="message assistant">
          👋 Hello! I'm your sleep assistant. I can help you with:<br><br>
          • 💤 Sleep improvement tips<br>
          • 🏃 Lifestyle adjustments<br>
          • 🧘 Stress management<br>
          • 📱 Digital wellness<br>
          • 🍎 Diet & nutrition advice<br><br>
          What would you like to know about better sleep?
          <div class="timestamp">Just now</div>
        </div>
      </div>
      <div class="input-area">
        <input type="text" id="user-input" placeholder="Type your question about sleep..." />
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
        
        const now = new Date();
        const timeString = now.getHours().toString().padStart(2, '0') + ':' + 
                          now.getMinutes().toString().padStart(2, '0');
        
        msgDiv.innerHTML = content.replace(/\\n/g, "<br>") + 
                          '<div class="timestamp">' + timeString + '</div>';
        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
      }

      function getCurrentTime() {
        const now = new Date();
        return now.getHours().toString().padStart(2, '0') + ':' + 
               now.getMinutes().toString().padStart(2, '0');
      }

      // Enter key support
      userInput.addEventListener("keypress", function(event) {
        if (event.key === "Enter") {
          event.preventDefault();
          sendBtn.click();
        }
      });

      sendBtn.addEventListener("click", () => {
        const msg = userInput.value.trim();
        if (!msg) return;
        appendMessage("user", msg);
        userInput.value = "";

        // Extended predefined responses
        setTimeout(() => {
          let reply = "🤔 I specialize in sleep and wellness topics. Try asking me about stress management, sleep routines, diet, exercise, or relaxation techniques!";
          const m = msg.toLowerCase();

          if (m.includes("hi") || m.includes("hello") || m.includes("hey")) 
            reply = "👋 Hi there! I'm here to help you improve your sleep quality. What specific sleep challenges are you facing?";
          else if (m.includes("stress")) 
            reply = "🧘 **Stress Relief Tips:**<br><br>1. Practice 5-10 minutes of deep breathing before bed<br>2. Keep a worry journal to empty your mind<br>3. Try progressive muscle relaxation<br>4. Limit news/social media before sleep<br>5. Create a calming bedtime routine";
          else if (m.includes("screen") || m.includes("phone") || m.includes("tv")) 
            reply = "📱 **Screen Time Management:**<br><br>1. Avoid screens 1 hour before bedtime<br>2. Use blue light filters in the evening<br>3. Switch to reading physical books<br>4. Charge devices outside bedroom<br>5. Try audiobooks or podcasts instead";
          else if (m.includes("insomnia") || m.includes("can't sleep")) 
            reply = "💡 **Insomnia Management:**<br><br>1. Stick to consistent sleep/wake times<br>2. Avoid caffeine after 2 PM<br>3. Get out of bed if not asleep in 20 mins<br>4. Use bed only for sleep (no work/TV)<br>5. Practice relaxation techniques";
          else if (m.includes("caffeine") || m.includes("coffee")) 
            reply = "☕ **Caffeine Tips:**<br><br>1. Limit to 1-2 cups daily<br>2. Avoid caffeine after 2 PM<br>3. Switch to herbal tea in evenings<br>4. Watch for hidden caffeine in soda/chocolate<br>5. Try decaf alternatives";
          else if (m.includes("alcohol") || m.includes("wine") || m.includes("beer")) 
            reply = "🍷 **Alcohol & Sleep:**<br><br>1. Alcohol disrupts REM sleep<br>2. Avoid within 3 hours of bedtime<br>3. Limit to 1 drink daily<br>4. Alternate with water<br>5. Notice how it affects your sleep quality";
          else if (m.includes("exercise") || m.includes("workout")) 
            reply = "🏃 **Exercise & Sleep:**<br><br>1. 30 mins daily improves sleep quality<br>2. Morning/afternoon workouts are ideal<br>3. Avoid intense exercise 2 hours before bed<br>4. Yoga/stretching in evening can help<br>5. Consistency matters more than intensity";
          else if (m.includes("food") || m.includes("diet") || m.includes("eat")) 
            reply = "🥗 **Food & Sleep:**<br><br>1. Avoid heavy meals 3 hours before bed<br>2. Light snacks like banana or warm milk can help<br>3. Limit sugar and processed foods<br>4. Foods rich in magnesium aid sleep<br>5. Stay hydrated but reduce fluids before bed";
          else if (m.includes("temperature") || m.includes("room")) 
            reply = "🌡️ **Ideal Sleep Environment:**<br><br>1. Keep room cool (18-22°C/65-72°F)<br>2. Use breathable cotton bedding<br>3. Blackout curtains for darkness<br>4. White noise machine if noisy<br>5. Comfortable, supportive mattress";
          else if (m.includes("routine") || m.includes("schedule")) 
            reply = "📅 **Sleep Routine Tips:**<br><br>1. Consistent bedtime even on weekends<br>2. 30-minute wind-down routine<br>3. Same wake-up time daily<br>4. Morning sunlight exposure<br>5. Avoid long daytime naps";
          else if (m.includes("water") || m.includes("hydrat")) 
            reply = "💧 **Hydration Advice:**<br><br>1. Drink 2-3L water throughout day<br>2. Reduce intake 2 hours before bed<br>3. Herbal teas (chamomile) promote sleep<br>4. Avoid sugary drinks before bed<br>5. Keep water nearby but sip lightly";
          else if (m.includes("wake up") || m.m.includes("morning")) 
            reply = "⏰ **Morning Energy Tips:**<br><br>1. Place alarm across the room<br>2. Open curtains for natural light<br>3. Drink glass of water<br>4. Light stretches or walking<br>5. Consistent wake-up time";
          else if (m.includes("nap") || m.includes("napping")) 
            reply = "😴 **Nap Guidelines:**<br><br>1. Limit to 20-30 minutes<br>2. Best between 1-3 PM<br>3. Avoid napping after 4 PM<br>4. Power naps boost energy<br>5. Long naps disrupt nighttime sleep";
          else if (m.includes("dream") || m.includes("nightmare")) 
            reply = "💭 **Dreams & Sleep:**<br><br>1. Stress can cause vivid dreams<br>2. Regular schedule supports healthy REM<br>3. Avoid heavy food/alcohol before bed<br>4. Discuss recurring nightmares with doctor<br>5. Dreams are normal and help processing";
          else if (m.includes("mental health") || m.includes("anxiety") || m.includes("depression")) 
            reply = "❤️ **Mental Health & Sleep:**<br><br>1. Journaling reduces racing thoughts<br>2. Mindfulness meditation helps<br>3. Seek professional help if needed<br>4. Regular exercise boosts mood<br>5. Social connection improves sleep";
          else if (m.includes("light") || m.includes("dark")) 
            reply = "☀️ **Light Exposure:**<br><br>1. Morning sunlight regulates rhythm<br>2. Dim lights 2 hours before bed<br>3. Use amber/orange evening lighting<br>4. Complete darkness for sleep<br>5. Blackout curtains if needed";
          else if (m.includes("melatonin")) 
            reply = "💊 **Melatonin Info:**<br><br>1. Natural sleep-wake regulator<br>2. Useful for jet lag/shift work<br>3. Take 30-60 mins before bed<br>4. Consult doctor before use<br>5. Lower lights boost natural production";
          else if (m.includes("shift work") || m.includes("night shift")) 
            reply = "🌙 **Shift Work Survival:**<br><br>1. Consistent sleep schedule even on days off<br>2. Blackout curtains for daytime sleep<br>3. Limit caffeine toward end of shift<br>4. Use white noise to block daytime sounds<br>5. Strategic napping if possible";
          else if (m.includes("music") || m.includes("sound")) 
            reply = "🎶 **Sleep Sounds:**<br><br>1. Calm instrumental music helps<br>2. White noise masks disturbances<br>3. Nature sounds are relaxing<br>4. Binaural beats for some people<br>5. Volume should be low and consistent";
          else if (m.includes("relax") || m.includes("calm")) 
            reply = "🛀 **Relaxation Techniques:**<br><br>1. Warm bath 1-2 hours before bed<br>2. Light reading (physical books)<br>3. Gentle yoga or stretching<br>4. Guided meditation apps<br>5. Deep breathing exercises";
          else if (m.includes("pillow") || m.includes("mattress") || m.includes("bed")) 
            reply = "🛏️ **Bed Comfort:**<br><br>1. Replace pillows every 1-2 years<br>2. Choose mattress for sleep position<br>3. Keep bedding clean and fresh<br>4. Allergen-proof covers if needed<br>5. Comfortable sleepwear";
          else if (m.includes("snoring") || m.includes("apnea")) 
            reply = "😴 **Snoring/Sleep Apnea:**<br><br>1. Side sleeping may reduce snoring<br>2. Weight loss can help if overweight<br>3. Avoid alcohol before bed<br>4. Consult doctor for persistent issues<br>5. Consider sleep study if recommended";
          else if (m.includes("thanks") || m.includes("thank you")) 
            reply = "🌟 You're welcome! Remember that small, consistent changes make the biggest difference in sleep quality. Sweet dreams!";
          else if (m.includes("bye") || m.includes("good night")) 
            reply = "🌌 Good night! Sleep well and wake up refreshed tomorrow. Feel free to ask more anytime!";

          appendMessage("assistant", reply);
        }, 600);
      });
    </script>
    """
    
    components.html(chatbot_html, height=600, scrolling=False)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🌙 Sleep Quality Predictor | Made with Streamlit | Remember: Quality sleep is essential for health and wellbeing"
    "</div>",
    unsafe_allow_html=True
)
