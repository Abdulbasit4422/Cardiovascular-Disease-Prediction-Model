import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Cardiovascular Risk Prediction Model App",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for colorful interface
st.markdown("""
<style>
    .main {
        background-color: #FFF5F5;
    }
    .sidebar .sidebar-content {
        background-color: #FFE6E6;
    }
    h1 { color: #FF4B4B; }
    h2 { color: #FF7676; }
    .risk-high { color: white; background-color: #FF4B4B; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center; }
    .risk-low  { color: white; background-color: #4CAF50; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center; }
</style>
""", unsafe_allow_html=True)

# Load the saved model
@st.cache_resource
def load_model():
    model_data = joblib.load('gradient_boosting_model_with_metadata_2.joblib')
    return model_data


from joblib import load

# Load the scaler
scaler = load('scaler.joblib')

model_data = load_model()
model = model_data['model']



# Define feature names as used during training
feature_names = [
    'age', 'age_year', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi',
    'cholesterol', 'gluc', 'smoke', 'alco', 'active',
    'gender_Female', 'gender_Male'
]

# Title and description
st.title("❤️ Cardiovascular Risk Prediction Model App")
st.markdown("""
This app predicts your risk of cardiovascular disease using a Gradient Boosting Classifier model.
Please fill in your health metrics below to get your personalized risk assessment.
""")

# Sidebar
with st.sidebar:
    st.header("About the Model")
    st.markdown(f"""
    - **Model Type**: {model_data['model_name']}
    - **Training Date**: {model_data['training_date']}
    - **Accuracy**: {model_data['training_score']:.2%}
    """)


# Input form
with st.form("prediction_form"):
    st.header("Your Health Metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 0, 120, 50)
        age_unit = st.radio("Age unit", ["years", "months"], index=0)
        age_years = age if age_unit == "years" else (age / 12)
        

        height = st.slider("Height (cm)", 0, 420, 170)
        weight = st.slider("Weight (kg)", 0, 400, 70)
        bmi = weight / ((height / 100) ** 2)
        st.metric("BMI", f"{bmi:.1f}")

    with col2:
        Gender = st.selectbox(
            "Gender", ["Male", "Female"], index=0
        )

        ap_hi = st.slider("Systolic BP (mmHg)", 0, 400, 120)
        ap_lo = st.slider("Diastolic BP (mmHg)", 0, 340, 80)
        cholesterol = st.selectbox(
            "Cholesterol Level", ["Normal (<200 mg/dl)", "Above Normal(200-239 mg/dl)", "Well Above Normal(>240 mg/dl)"], index=0
        )

    with col3:
        gluc = st.selectbox(
            "Glucose Level", ["Normal (<140mg/dl)", "Above (140-200 mg/dl)", "Well Above Normal (>200mg/dl)"], index=0
        )
        smoke = st.checkbox("Smoker")
        alco = st.checkbox("Alcohol Consumer")
        active = st.checkbox("Physically Active")

    submitted = st.form_submit_button("Predict Risk")

# Prediction and results
if submitted:
    # Mapping categorical inputs to numeric codes
    cholesterol_map = {"Normal (<200 mg/dl)": 1, "Above Normal(200-239 mg/dl)": 2, "Well Above Normal(>240 mg/dl)": 3}
    gluc_map        = {"Normal (<140mg/dl)": 1, "Above Normal (140-200 mg/dl)": 2, "Well Above Normal (>200mg/dl)": 3}

    # Prepare input data with correct order and types
    input_data = {
        'age': age_years * 365,
        'height': float(height),
        'weight': float(weight),
        'ap_hi': int(ap_hi),
        'ap_lo': int(ap_lo),
        'bmi': float(bmi),
        'age_year': int(age_years),
        'cholesterol': cholesterol_map[cholesterol],
        'gluc': gluc_map[gluc],
        'smoke': int(smoke),
        'alco': int(alco),
        'active': int(active),
        'gender_Female': 0,
        'gender_Male': 1
    }
    input_df = pd.DataFrame([input_data]).astype(float)


    # Predict
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    # Display results
    st.header("Your Results")
    if prediction == 1:
        st.error("## 🚨 High Risk of Cardiovascular Disease")
        st.markdown(
            '<div class="risk-high">WARNING: You are at HIGH RISK of cardiovascular disease.<br>Please consult a healthcare professional immediately.</div>',
            unsafe_allow_html=True
        )
    else:
        st.success("## ✅ Low Risk of Cardiovascular Disease")
        st.markdown(
            '<div class="risk-low">Great news! You are at LOW RISK of cardiovascular disease.<br>Maintain your healthy lifestyle!</div>',
            unsafe_allow_html=True
        )

    # Probability gauge
    risk_percent = proba[1] * 100
    st.subheader("Risk Probability")
    st.metric("Probability of Cardiovascular Disease", f"{risk_percent:.1f}%")

    fig, ax = plt.subplots(figsize=(8, 2))
    ax.barh(['Risk'], [100], color='lightgray')
    ax.barh(['Risk'], [risk_percent], color='red' if risk_percent > 50 else 'green')
    ax.set_xlim(0, 100); ax.set_xticks([])
    ax.text(risk_percent/2, 0, f"{risk_percent:.1f}%", ha='center', va='center', color='white', fontweight='bold')
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("**Disclaimer**: For informational purposes only.")