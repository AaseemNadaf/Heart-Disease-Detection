import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import plotly.graph_objects as go 

# LOAD SAVED ARTIFACTS (Model, Scaler, Imputer)

try:
    model = joblib.load('best_model.joblib')
    scaler = joblib.load('scaler.joblib')
    imputer = joblib.load('imputer.joblib')
    
    feature_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]

except FileNotFoundError:
    st.error("Error: Model or preprocessing files not found.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading files: {e}")
    st.stop()

# NEW HELPER FUNCTION: Generate descriptive feedback

def generate_risk_factors(input_df):
    """Provides simple, rule-based feedback on key risk factors."""
    factors = []
    
    if input_df['age'].values[0] > 60:
        factors.append(f"**Age ({input_df['age'].values[0]}):** Advanced age is a known risk factor.")
        
    if input_df['trestbps'].values[0] > 140:
        factors.append(f"**Resting BP ({input_df['trestbps'].values[0]} mm Hg):** This is considered high (Hypertension Stage 2).")
    elif input_df['trestbps'].values[0] > 130:
        factors.append(f"**Resting BP ({input_df['trestbps'].values[0]} mm Hg):** This is considered elevated (Hypertension Stage 1).")
        
    if input_df['chol'].values[0] > 240:
        factors.append(f"**Cholesterol ({input_df['chol'].values[0]} mg/dl):** This is considered high.")
    elif input_df['chol'].values[0] > 200:
        factors.append(f"**Cholesterol ({input_df['chol'].values[0]} mg/dl):** This is borderline high.")

    if input_df['thalach'].values[0] < (220 - input_df['age'].values[0]) * 0.6:
        factors.append(f"**Max Heart Rate ({input_df['thalach'].values[0]}):** This appears low for the patient's age. (Note: This is a very general estimate).")

    if input_df['exang'].values[0] == 1:
        factors.append("**Exercise Induced Angina (Yes):** This is a strong indicator of coronary artery issues.")
        
    if input_df['oldpeak'].values[0] > 1.0:
        factors.append(f"**ST Depression ({input_df['oldpeak'].values[0]}):** A value > 1.0 can indicate abnormal blood flow to the heart.")

    if not factors:
        factors.append("Inputs do not show obvious, common risk factors based on general guidelines.")
        
    return factors

# PAGE CONFIGURATION & TITLE

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")
st.title("❤️ Heart Disease Risk Prediction")
st.write("""
This prototype uses a K-Nearest Neighbours (KNN) model to predict 
heart disease risk based on the UCI Heart Disease dataset. 
Please enter the patient's data below.
""")

# USER INPUT - (No changes in this section)

col1, col2, col3 = st.columns(3)
with col1:
    st.header("Patient Demographics")
    age = st.slider("Age", 20, 80, 50)
    sex = st.selectbox("Sex", (0, 1), format_func=lambda x: "Male" if x == 1 else "Female")
with col2:
    st.header("Vital Signs")
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    thalach = st.slider("Max. Heart Rate Achieved", 70, 220, 150)
with col3:
    st.header("Test Results")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", (0, 1), format_func=lambda x: "True" if x == 1 else "False")
    restecg_options = {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Probable Left Ventricular Hypertrophy"}
    restecg = st.selectbox("Resting ECG Results", options=list(restecg_options.keys()), format_func=lambda x: restecg_options[x])

st.divider()
st.header("Exercise & Clinical Data")
col_a, col_b, col_c, col_d, col_e = st.columns(5)
with col_a:
    cp_options = {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-Anginal Pain", 4: "Asymptomatic"}
    cp = st.selectbox("Chest Pain Type", options=list(cp_options.keys()), format_func=lambda x: cp_options[x])
with col_b:
    exang = st.selectbox("Exercise Induced Angina", (0, 1), format_func=lambda x: "Yes" if x == 1 else "No")
with col_c:
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0, 0.1)
with col_d:
    slope_options = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
    slope = st.selectbox("Slope of Peak Exercise ST", options=list(slope_options.keys()), format_func=lambda x: slope_options[x])
with col_e:
    ca = st.selectbox("Major Vessels Colored by Flourosopy", (0, 1, 2, 3))
    thal_options = {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}
    thal = st.selectbox("Thalassemia", options=list(thal_options.keys()), format_func=lambda x: thal_options[x])

# PREDICTION LOGIC 

if st.button("Get Risk Prediction", type="primary"):
    
    # 1. Collect and process inputs (same as before)
    user_input_list = [
        age, sex, cp, trestbps, chol, fbs, 
        restecg, thalach, exang, oldpeak, slope, ca, thal
    ]
    user_input_np = np.array(user_input_list).reshape(1, -1)
    user_input_df = pd.DataFrame(user_input_np, columns=feature_names)
    
    try:
        user_input_imputed = imputer.transform(user_input_df)
        user_input_scaled = scaler.transform(user_input_imputed)
    
        # 6. Make Prediction
        prediction = model.predict(user_input_scaled)
        prediction_proba = model.predict_proba(user_input_scaled)
        probability_of_disease = prediction_proba[0][1] * 100

        # 7. Display the result
        st.subheader("Prediction Result")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability_of_disease,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Probability of Disease (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps' : [
                         {'range': [0, 50], 'color': "lightgreen"},
                         {'range': [50, 75], 'color': "yellow"},
                         {'range': [75, 100], 'color': "red"}],
                }))
            
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with res_col2:
            # --- UPDATED: Main result text ---
            if prediction[0] == 1:
                st.error(f"**Result: High Risk**")
                st.write(f"The model predicts a **{probability_of_disease:.2f}%** probability of heart disease.")
                st.write("This result suggests a significant likelihood based on the provided data.")
            else:
                st.success(f"**Result: Low Risk**")
                st.write(f"The model predicts a **{probability_of_disease:.2f}%** probability of heart disease.")
                st.write("This result suggests a low likelihood based on the provided data.")
                    
        # Descriptive Feedback Section
        st.divider()
        with st.expander("Show Key Risk Factor Analysis"):
            st.write("""
            Here is a simple analysis of the inputs based on general medical guidelines. 
            This is *not* the model's reasoning, but provides context.
            """)
            
            # Get the list of risk factors
            risk_factors = generate_risk_factors(user_input_df)
            
            # Display them
            for factor in risk_factors:
                st.markdown(f"- {factor}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")