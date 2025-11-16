import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------
# Load model + scaler + columns
# -------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_logistic_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()


numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# -------------------------
# Streamlit UI
# -------------------------
st.title("Heart Disease Risk Predictor")
st.write(
    "This app uses a Logistic Regression model trained on the Cleveland Heart Disease dataset "
    "to estimate the probability of heart disease based on clinical features."
)

st.sidebar.header("Patient Input Features")

age = st.sidebar.number_input("Age (years)", min_value=20, max_value=100, value=55)
sex = st.sidebar.selectbox("Sex", options=["Female (0)", "Male (1)"])
cp = st.sidebar.selectbox(
    "Chest Pain Type (cp)",
    options=[
        "1 - Typical angina",
        "2 - Atypical angina",
        "3 - Non-anginal pain",
        "4 - Asymptomatic"
    ]
)
trestbps = st.sidebar.number_input("Resting Blood Pressure (mmHg)", 80, 250, 130)
chol = st.sidebar.number_input("Serum Cholesterol (mg/dL)", 100, 600, 250)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL (fbs)", ["No (0)", "Yes (1)"])
restecg = st.sidebar.selectbox(
    "Resting ECG (restecg)",
    [
        "0 - Normal",
        "1 - ST-T wave abnormality",
        "2 - Left ventricular hypertrophy"
    ]
)
thalach = st.sidebar.number_input("Max Heart Rate Achieved (thalach)", 60, 250, 150)
exang = st.sidebar.selectbox("Exercise-Induced Angina (exang)", ["No (0)", "Yes (1)"])
oldpeak = st.sidebar.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
slope = st.sidebar.selectbox(
    "Slope of ST Segment (slope)",
    [
        "1 - Upsloping",
        "2 - Flat",
        "3 - Downsloping"
    ]
)
ca = st.sidebar.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3])
thal = st.sidebar.selectbox(
    "Thalassemia (thal)",
    [
        "3 - Normal",
        "6 - Fixed defect",
        "7 - Reversible defect"
    ]
)

# Map string selections to numeric codes
sex_val = 1 if "Male" in sex else 0
cp_val = int(cp.split(" ")[0])
fbs_val = 1 if "Yes" in fbs else 0
restecg_val = int(restecg.split(" ")[0])
exang_val = 1 if "Yes" in exang else 0
slope_val = int(slope.split(" ")[0])
thal_val = int(thal.split(" ")[0])

# -------------------------
# Build input DataFrame in ORIGINAL format
# -------------------------
input_dict = {
    "age": [age],
    "sex": [sex_val],
    "cp": [cp_val],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs_val],
    "restecg": [restecg_val],
    "thalach": [thalach],
    "exang": [exang_val],
    "oldpeak": [oldpeak],
    "slope": [slope_val],
    "ca": [ca],
    "thal": [thal_val],
}

input_df = pd.DataFrame(input_dict)

st.subheader("Patient Input Summary")
st.write(input_df)

# -------------------------
# Match the preprocessing from training
# (get_dummies + reindex + scale numeric)
# -------------------------
# Apply same one-hot encoding
input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

# Make sure all training columns exist
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

# Scale numeric columns using the fitted scaler
input_encoded[numeric_cols] = scaler.transform(input_encoded[numeric_cols])

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Heart Disease Risk"):
    proba = model.predict_proba(input_encoded)[:, 1][0]
    pred = model.predict(input_encoded)[0]

    st.subheader("Prediction Result")
    st.write(f"**Estimated probability of heart disease:** {proba:.2%}")

    if pred == 1:
        st.error("The model predicts **heart disease is likely (class 1)**.")
    else:
        st.success("The model predicts **no heart disease (class 0)**.")

    st.caption(
        "This tool is for educational purposes only and should not be used as a substitute "
        "for professional medical diagnosis."
    )
