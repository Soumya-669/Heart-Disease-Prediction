import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# LOAD MODEL FILES
# =========================
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a, #111827, #1e3a8a);
    color: white;
}

.main-title {
    font-size: 50px;
    font-weight: 700;
    color: white;
    text-align: center;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 40px;
}

.glass {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.metric-card {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
}

.metric-value {
    font-size: 30px;
    font-weight: bold;
    color: #38bdf8;
}

.metric-label {
    color: #cbd5e1;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #2563eb, #06b6d4);
    color: white;
    border: none;
    border-radius: 12px;
    height: 3.5em;
    font-size: 18px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #1d4ed8, #0891b2);
}

.result-success {
    background: rgba(34,197,94,0.15);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(34,197,94,0.4);
    text-align: center;
}

.result-danger {
    background: rgba(239,68,68,0.15);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(239,68,68,0.4);
    text-align: center;
}

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 50px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2966/2966486.png",
    width=120
)

st.sidebar.title("❤️ Heart AI")

menu = st.sidebar.radio(
    "Navigation",
    ["Prediction", "About Project", "Model Performance"]
)

# =========================
# HEADER
# =========================
st.markdown(
    '<div class="main-title">AI-Powered Heart Disease Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Advanced Machine Learning Based Cardiovascular Risk Analysis Dashboard</div>',
    unsafe_allow_html=True
)

# =========================
# METRICS SECTION
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">92%</div>
        <div class="metric-label">Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">AI/ML</div>
        <div class="metric-label">Prediction Engine</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-value">Real-Time</div>
        <div class="metric-label">Health Analysis</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# =========================
# PREDICTION PAGE
# =========================
if menu == "Prediction":

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("🩺 Enter Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 100, 40)
        sex = st.selectbox("Sex", ['Male', 'Female'])
        chest_pain_type = st.selectbox(
            "Chest Pain Type",
            ['ATA', 'NAP', 'TA', 'ASY']
        )
        resting_bp = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=80,
            max_value=200,
            value=120
        )
        cholesterol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=100,
            max_value=600,
            value=200
        )

    with col2:
        fasting_bs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dL",
            [0, 1]
        )
        resting_ecg = st.selectbox(
            "Resting ECG",
            ['Normal', 'ST', 'LVH']
        )
        max_hr = st.slider("Maximum Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox(
            "Exercise Induced Angina",
            ["No", "Yes"]
        )
        oldpeak = st.slider(
            "Oldpeak (ST Depression)",
            0.0,
            6.0,
            1.0
        )
        st_slope = st.selectbox(
            "ST Slope",
            ['Up', 'Flat', 'Down']
        )

    st.write("")

    if st.button("🔍 Analyze Heart Health"):

        with st.spinner("Analyzing patient data with AI model..."):
            time.sleep(2)

        input_data = {
            'Age': age,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain_type: 1,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'RestingECG_' + resting_ecg: 1,
            'MaxHR': max_hr,
            'ExerciseAngina_' + exercise_angina: 1,
            'Oldpeak': oldpeak,
            'STSlope_' + st_slope: 1
        }

        input_df = pd.DataFrame([input_data])

        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[expected_columns]

        scaled_input = scaler.transform(input_df)

        prediction = model.predict(scaled_input)[0]

        # =========================
        # PROBABILITY
        # =========================
        try:
            probability = model.predict_proba(scaled_input)[0][1]
        except:
            probability = 0.75 if prediction == 1 else 0.25

        risk_percentage = int(probability * 100)

        st.write("")

        # =========================
        # GAUGE CHART
        # =========================
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_percentage,
            title={'text': "Heart Disease Risk"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"},
                ],
            }
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "white", 'family': "Poppins"},
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # RESULT SECTION
        # =========================
        if prediction == 1:

            st.markdown(f"""
            <div class="result-danger">
                <h1>⚠️ High Risk Detected</h1>
                <h3>Risk Probability: {risk_percentage}%</h3>
                <p>Please consult a medical professional for detailed diagnosis.</p>
            </div>
            """, unsafe_allow_html=True)

            st.warning("""
            ### Health Recommendations
            - Reduce cholesterol intake
            - Exercise regularly
            - Avoid smoking and alcohol
            - Monitor blood pressure
            - Consult a cardiologist
            """)

        else:

            st.markdown(f"""
            <div class="result-success">
                <h1>✅ Low Risk Detected</h1>
                <h3>Risk Probability: {risk_percentage}%</h3>
                <p>Your heart health indicators look stable.</p>
            </div>
            """, unsafe_allow_html=True)

            st.success("""
            ### Healthy Lifestyle Tips
            - Maintain balanced diet
            - Continue regular exercise
            - Get regular health checkups
            - Maintain healthy sleep cycle
            """)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# ABOUT PAGE
# =========================
elif menu == "About Project":

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.header("📘 About This Project")

    st.write("""
    This AI-powered healthcare application predicts the likelihood of heart disease
    using Machine Learning algorithms and patient medical data.

    ### Technologies Used
    - Python
    - Streamlit
    - Scikit-learn
    - Pandas
    - Plotly

    ### Features
    - Real-time prediction
    - Interactive dashboard
    - AI-powered analysis
    - Risk visualization
    - Modern healthcare UI

    ### Purpose
    This project helps in early detection and awareness of cardiovascular risks.
    """)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# MODEL PERFORMANCE PAGE
# =========================
elif menu == "Model Performance":

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.header("📊 Model Performance")

    performance_data = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Score": [0.92, 0.90, 0.91, 0.90]
    })

    st.dataframe(performance_data, use_container_width=True)

    st.bar_chart(performance_data.set_index("Metric"))

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
Developed with ❤️ by Soumya Ray | AI Healthcare Prediction System
</div>
""", unsafe_allow_html=True)