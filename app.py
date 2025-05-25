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
    # Load model metadata and model
    data = joblib.load('gradient_boosting_model_with_metadata_3.joblib')
    model = data['model']
    # Load scaler
    scaler = joblib.load('scaler_2.joblib')
    return data, model, scaler

model_data, model, scaler = load_resources()

# ---- Sidebar ----
with st.sidebar:
    st.header("About this App")
    st.markdown("This tool predicts cardiovascular risk using a Gradient Boosting model.")
    st.markdown("---")
    st.subheader("Model Details")
    st.markdown(
        f"**Type:** {model_data['model_name']}  \n"
        f"**Trained on:** {model_data['training_date']}  \n"
        f"**Accuracy:** {model_data['training_score']:.2%}"
    )

# ---- Main Title ----
st.title("❤️ Cardiovascular Risk Predictor")
st.write("Provide your health metrics below to see your estimated risk of cardiovascular disease.")

# ---- Input Form ----
with st.form(key='input_form'):
    st.subheader("Your Health Metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        age_years = st.slider("Age (years)", 1, 120, 30)
        height = st.slider("Height (cm)", 50, 250, 170)
        weight = st.slider("Weight (kg)", 20, 200, 70)
        bmi = weight / ((height / 100) ** 2)
        st.metric("BMI", f"{bmi:.1f}")

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        ap_hi = st.slider("Systolic BP (mmHg)", 80, 200, 120)
        ap_lo = st.slider("Diastolic BP (mmHg)", 40, 140, 80)
        cholesterol = st.selectbox(
            "Cholesterol Level",
            ["Normal (<200 mg/dl)", "Above Normal (200-239)", "High (>240)"]
        )

    with col3:
        gluc = st.selectbox(
            "Glucose Level",
            ["Normal (<140 mg/dl)", "Above Normal (140-200)", "High (>200)"]
        )
        smoke = st.checkbox("Smoker")
        alco = st.checkbox("Alcohol Consumer")
        active = st.checkbox("Physically Active")

    submit_btn = st.form_submit_button(label='Predict Risk')

# ---- Prediction & Display ----
if submit_btn:
    # Mapping values to numeric codes
    chol_map = {"Normal (<200 mg/dl)": 1, "Above Normal (200-239)": 2, "High (>240)": 3}
    gluc_map = {"Normal (<140 mg/dl)": 1, "Above Normal (140-200)": 2, "High (>200)": 3}

    # Prepare features
    features = pd.DataFrame([{  
        'age': age_years * 365,
        'age_year': age_years,
        'height': height,
        'weight': weight,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'bmi': bmi,
        'cholesterol': chol_map[cholesterol],
        'gluc': gluc_map[gluc],
        'smoke': int(smoke),
        'alco': int(alco),
        'active': int(active),
        'gender_Female': int(gender == 'Female'),
        'gender_Male': int(gender == 'Male')
    }])

    # Scale and predict
    X_scaled = scaler.transform(features)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1] * 100

    # Display results
    st.subheader("Prediction Results")
    if pred:
        st.markdown('<div class="risk-high">High Risk detected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="risk-low">Low Risk detected</div>', unsafe_allow_html=True)

    st.subheader("Risk Probability")
    st.metric(label="Chance of Disease", value=f"{proba:.1f}%")

    # Probability bar chart
    fig, ax = plt.subplots(figsize=(6, 0.5))
    ax.barh([""], [proba], height=0.3)
    ax.barh([""], [100 - proba], left=[proba], height=0.3, color='lightgray')
    ax.set_xlim(0, 100)
    ax.axis('off')
    st.pyplot(fig)

# ---- Footer ----
st.markdown("---")
st.caption("⚠️ This app is for informational purposes only. Consult a healthcare professional for medical advice.")
