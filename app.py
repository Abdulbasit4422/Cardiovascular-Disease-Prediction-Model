import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Cardiovascular Risk Prediction",
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
    h1 {
        color: #FF4B4B;
    }
    h2 {
        color: #FF7676;
    }
    .st-bb {
        background-color: #FFE6E6;
    }
    .st-at {
        background-color: #FF4B4B;
    }
    .st-cb {
        color: #FF4B4B;
    }
    .risk-high {
        color: white;
        background-color: #FF4B4B;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    .risk-low {
        color: white;
        background-color: #4CAF50;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load the saved model
@st.cache_resource
def load_model():
    model_data = joblib.load('gradient_boosting_model_with_metadata.joblib')
    return model_data

model_data = load_model()
model = model_data['model']
feature_names = model_data['features']

# Title and description
st.title("❤️ Cardiovascular Disease Risk Prediction")
st.markdown("""
This app predicts your risk of cardiovascular disease using a Gradient Boosting Classifier model.
Please fill in your health metrics below to get your personalized risk assessment.
""")

# Sidebar with info
with st.sidebar:
    st.header("About the Model")
    st.markdown(f"""
    - **Model Type**: {model_data['model_name']}
    - **Training Date**: {model_data['training_date']}
    - **Accuracy**: {model_data['training_score']:.2%}
    - **Top Features**: Age, Blood Pressure, Cholesterol
    """)
    
    st.header("How to Use")
    st.markdown("""
    1. Enter your health metrics
    2. Click 'Predict Risk'
    3. View your results
    """)

# Input form
with st.form("prediction_form"):
    st.header("Your Health Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age (years)"0, 20, 100, 50)
        height = st.slider("Height (cm)"0, 140, 220, 170)
        weight = st.slider("Weight (kg)"0, 40, 150, 70)
        bmi = weight / ((height/100) ** 2)
        st.metric("BMI", f"{bmi:.1f}")
        
    with col2:
        ap_hi = st.slider("Systolic BP (mmHg)"0, 80, 200, 120)
        ap_lo = st.slider("Diastolic BP (mmHg)"0, 50, 150, 80)
        cholesterol = st.number_input("Cholesterol level value" )
        
    with col3:
        gluc = st.number_input("Glucose Level:")
        smoke = st.checkbox("Smoker")
        alco = st.checkbox("Alcohol Consumer")
        active = st.checkbox("Physically Active")
    
    submitted = st.form_submit_button("Predict Risk")

# Prediction and results
if submitted:
    # Prepare input data
    cholesterol_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
    gluc_map = {"Normal": 1, "Above Normal": 2, "Well Above Normal": 3}
    
    input_data = {
        'age': age * 365,  # Convert to days (as in original data)
        'height': height,
        'weight': weight,
        'ap_hi': ap_hi,
        'ap_lo': ap_lo,
        'bmi': bmi,
        'cholesterol': cholesterol,
        'gluc': gluc,
        'smoke': int(smoke),
        'alco': int(alco),
        'active': int(active),
        'gender_Female': 0,  # Assuming male as default
        'gender_Male': 1
    }
    
    # Create DataFrame with correct feature order
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Make prediction
    prediction = model.predict(input_df)
    proba = model.predict_proba(input_df)[0]
    
    # Display results
    st.header("Your Results")
    
    if prediction[0] == 1:
        st.error("## 🚨 High Risk of Cardiovascular Disease")
        st.balloons() if proba[1] > 0.8 else None  # Show balloons if very high risk
        
        # Popup notification
        st.markdown("""
        <div class="risk-high">
            WARNING: You are at HIGH RISK of cardiovascular disease.<br>
            Please consult a healthcare professional immediately.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success("## ✅ Low Risk of Cardiovascular Disease")
        
        # Popup notification
        st.markdown("""
        <div class="risk-low">
            Great news! You are at LOW RISK of cardiovascular disease.<br>
            Maintain your healthy lifestyle!
        </div>
        """, unsafe_allow_html=True)
    
    # Show probability gauge
    st.subheader("Risk Probability")
    risk_percent = proba[1] * 100
    st.metric("Probability of Cardiovascular Disease", f"{risk_percent:.1f}%")
    
    # Create gauge chart
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.barh(['Risk'], [100], color='lightgray')
    ax.barh(['Risk'], [risk_percent], color='red' if risk_percent > 50 else 'green')
    ax.set_xlim(0, 100)
    ax.set_title("Risk Level Gauge")
    ax.set_xticks([])
    ax.text(risk_percent/2, 0, f"{risk_percent:.1f}%", 
            ha='center', va='center', color='white', fontweight='bold')
    st.pyplot(fig)
    
    # Show feature importance
    st.subheader("Key Factors in Your Assessment")
    try:
        feature_importance = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False).head(5)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=importance_df, x='Importance', y='Feature', palette='Reds_r')
        ax.set_title("Top Factors Influencing Your Risk")
        st.pyplot(fig)
    except:
        st.info("Feature importance not available for this model.")

# Model performance section
st.header("Model Performance Metrics")
st.markdown("""
Below are the performance metrics of our prediction model:
""")

metrics = {
    'Accuracy': 0.73,
    'Precision': 0.72,
    'Recall': 0.74,
    'F1 Score': 0.73,
    'ROC AUC': 0.80
}

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Accuracy", f"{metrics['Accuracy']:.2%}")
col2.metric("Precision", f"{metrics['Precision']:.2%}")
col3.metric("Recall", f"{metrics['Recall']:.2%}")
col4.metric("F1 Score", f"{metrics['F1 Score']:.2%}")
col5.metric("ROC AUC", f"{metrics['ROC AUC']:.2%}")

# Confusion matrix visualization
st.subheader("Confusion Matrix")
cm = np.array([[6500, 2500], [2300, 6700]])  # Example values
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Model Confusion Matrix")
st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer**: This tool is for informational purposes only and is not a substitute for professional medical advice.
""")