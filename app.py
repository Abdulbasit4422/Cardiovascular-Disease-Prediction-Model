import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---- Page Configuration ----
st.set_page_config(
    page_title="CardioRisk Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom Styles ----
custom_css = '''
<style>
    .risk-high { background-color: #FF4B4B; color: white; padding: 1rem; border-radius: 0.5rem; text-align: center; font-weight: 600; }
    .risk-low  { background-color: #4CAF50; color: white; padding: 1rem; border-radius: 0.5rem; text-align: center; font-weight: 600; }
</style>
'''
st.markdown(custom_css, unsafe_allow_html=True)

# ---- Data & Model Loading ----
@st.cache_resource
def load_resources():
    model_data = joblib.load('gradient_boosting_model_with_metadata_3.joblib')
    scaler = joblib.load('scaler_2.joblib')
    return model_data['model'], scaler

model, scaler = load_resources()

# ---- Feature Preparation ----
def prepare_features(age_years, height, weight, ap_hi, ap_lo, cholesterol, gluc, 
                    smoke, alco, active, gender):
    """Prepare features matching the training data format"""
    # Calculate derived features
    bmi = weight / ((height / 100) ** 2)
    
    # Create feature DataFrame with correct column order
    features = pd.DataFrame([{
        'age': age_years * 365,
        'height': height,
        'weight': weight,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'bmi': bmi,
        'age_year': age_years,
        'cholesterol': cholesterol,
        'gluc': gluc,
        'smoke': int(smoke),
        'alco': int(alco),
        'active': int(active),
        'gender_Female': int(gender == 'Female'),
        'gender_Male': int(gender == 'Male')
    }])
    
    # Drop columns excluded during training
    return features.drop([
        'age_group_children', 'age_group_teenager', 'age_group_youth',
        'age_group_middle_age', 'age_group_elderly', 'obesity_0',
        'obesity_1', 'obesity_2', 'obesity_3', 'obesity_4', 'hypertensive'
    ], axis=1, errors='ignore')

# ---- Sidebar ----
with st.sidebar:
    st.header("About this App")
    st.markdown("This tool predicts cardiovascular risk using a Gradient Boosting model.")
    st.markdown("---")
    st.subheader("Model Details")
    st.markdown("""
    - **Algorithm:** Gradient Boosting Classifier
    - **Top Features:** Age, Blood Pressure, Cholesterol
    - **Training Accuracy:** 73.2%
    """)

# ---- Main Interface ----
st.title("❤️ Cardiovascular Risk Predictor")
st.write("Provide your health metrics below to see your estimated risk of cardiovascular disease.")

with st.form(key='input_form'):
    st.subheader("Your Health Metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        age_years = st.slider("Age (years)", 1, 120, 30)
        height = st.slider("Height (cm)", 50, 250, 170)
        weight = st.slider("Weight (kg)", 20, 200, 70)

    with col2:
        gender = st.selectbox("Gender", ["Female", "Male"])
        ap_hi = st.slider("Systolic BP (mmHg)", 80, 200, 120)
        ap_lo = st.slider("Diastolic BP (mmHg)", 40, 140, 80)
        cholesterol = st.selectbox(
            "Cholesterol Level",
            ["Normal (<200 mg/dl)", "Above Normal (200-239)", "High (>240)"],
            index=1
        )

    with col3:
        gluc = st.selectbox(
            "Glucose Level",
            ["Normal (<140 mg/dl)", "Above Normal (140-200)", "High (>200)"],
            index=1
        )
        smoke = st.checkbox("Current Smoker")
        alco = st.checkbox("Regular Alcohol Consumer")
        active = st.checkbox("Regular Physical Activity")

    submit_btn = st.form_submit_button(label='Assess Cardiovascular Risk')

# ---- Prediction Logic ----
if submit_btn:
    # Convert categorical inputs to numerical
    chol_map = {
        "Normal (<200 mg/dl)": 1, 
        "Above Normal (200-239)": 2, 
        "High (>240)": 3
    }
    gluc_map = {
        "Normal (<140 mg/dl)": 1,
        "Above Normal (140-200)": 2, 
        "High (>200)": 3
    }

    features = prepare_features(
        age_years=age_years,
        height=height,
        weight=weight,
        ap_hi=ap_hi,
        ap_lo=ap_lo,
        cholesterol=chol_map[cholesterol],
        gluc=gluc_map[gluc],
        smoke=smoke,
        alco=alco,
        active=active,
        gender=gender
    )

    # Scale features using the training scaler
    X_scaled = scaler.transform(features)
    
    # Make prediction
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1] * 100

    # Display results
    st.subheader("Risk Assessment")
    if pred == 1:
        st.markdown('<div class="risk-high">High Risk Detected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="risk-low">Low Risk Detected</div>', unsafe_allow_html=True)

    # Visualize probabilities
    fig, ax = plt.subplots(figsize=(8, 0.5))
    ax.barh([""], [proba], height=0.3, color='#FF4B4B' if pred else '#4CAF50')
    ax.barh([""], [100-proba], left=[proba], height=0.3, color='lightgrey')
    ax.set_xlim(0, 100)
    ax.text(proba/2, 0, f"{proba:.1f}%", ha='center', va='center', color='white', fontweight='bold')
    ax.text(proba + (100-proba)/2, 0, f"{100-proba:.1f}%", ha='center', va='center', color='black')
    ax.axis('off')
    st.pyplot(fig)

    # Show key factors
    st.subheader("Key Risk Factors")
    st.markdown("""
    - Systolic Blood Pressure
    - Age
    - Cholesterol Level
    - BMI
    - Physical Activity Level
    """)

# ---- Footer ----
st.markdown("---")
st.caption("⚠️ This tool provides statistical risk assessment and should not replace professional medical advice.")