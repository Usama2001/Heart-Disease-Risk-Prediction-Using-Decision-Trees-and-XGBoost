import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="centered"
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_heart_disease_model.pkl"
FEATURE_NAMES_PATH = PROJECT_ROOT / "models" / "feature_names.pkl"


@st.cache_resource
def load_model_and_features():
    """
    Load saved XGBoost model and feature names.
    """
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    return model, feature_names


def predict_heart_disease_risk(patient_data, model, feature_names):
    """
    Predict heart disease risk for one patient.
    """
    patient_df = pd.DataFrame([patient_data])

    # Keep the same feature order used during model training
    patient_df = patient_df[feature_names]

    prediction = model.predict(patient_df)[0]
    probability = model.predict_proba(patient_df)[0][1]

    return int(prediction), float(probability)


# App title
st.title("❤️ Heart Disease Risk Prediction App")

st.write(
    """
    This app uses a trained **XGBoost machine learning model** to predict whether a patient may have heart disease risk based on medical features.
    """
)

st.warning(
    "Medical Disclaimer: This app is for educational and portfolio purposes only. It should not be used as a real medical diagnosis tool."
)

# Load model
try:
    model, feature_names = load_model_and_features()
except FileNotFoundError:
    st.error(
        "Model file not found. Please run the notebook up to Step 11 to generate the saved model files inside the models folder."
    )
    st.stop()


st.subheader("Enter Patient Information")

# Input fields
age = st.number_input("Age", min_value=20, max_value=100, value=54)

sex_label = st.selectbox("Sex", ["Female", "Male"])
sex = 1 if sex_label == "Male" else 0

cp_label = st.selectbox(
    "Chest Pain Type",
    [
        "Typical Angina",
        "Atypical Angina",
        "Non-Anginal Pain",
        "Asymptomatic"
    ]
)

cp_mapping = {
    "Typical Angina": 1,
    "Atypical Angina": 2,
    "Non-Anginal Pain": 3,
    "Asymptomatic": 4
}
cp = cp_mapping[cp_label]

trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=80,
    max_value=220,
    value=130
)

chol = st.number_input(
    "Cholesterol",
    min_value=100,
    max_value=600,
    value=250
)

fbs_label = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    ["No", "Yes"]
)
fbs = 1 if fbs_label == "Yes" else 0

restecg_label = st.selectbox(
    "Resting ECG Result",
    [
        "Normal",
        "ST-T Wave Abnormality",
        "Left Ventricular Hypertrophy"
    ]
)

restecg_mapping = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}
restecg = restecg_mapping[restecg_label]

thalach = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=60,
    max_value=230,
    value=150
)

exang_label = st.selectbox(
    "Exercise Induced Angina",
    ["No", "Yes"]
)
exang = 1 if exang_label == "Yes" else 0

oldpeak = st.number_input(
    "Oldpeak / ST Depression",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

slope_label = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    ["Upsloping", "Flat", "Downsloping"]
)

slope_mapping = {
    "Upsloping": 1,
    "Flat": 2,
    "Downsloping": 3
}
slope = slope_mapping[slope_label]

ca = st.selectbox(
    "Number of Major Vessels Colored by Fluoroscopy",
    [0, 1, 2, 3]
)

thal_label = st.selectbox(
    "Thalassemia",
    ["Normal", "Fixed Defect", "Reversible Defect"]
)

thal_mapping = {
    "Normal": 3,
    "Fixed Defect": 6,
    "Reversible Defect": 7
}
thal = thal_mapping[thal_label]


# Prepare patient input
patient_data = {
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}


if st.button("Predict Heart Disease Risk"):
    prediction, probability = predict_heart_disease_risk(
        patient_data,
        model,
        feature_names
    )

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Prediction: Heart Disease Risk")
    else:
        st.success("Prediction: Low Heart Disease Risk")

    st.metric(
        label="Heart Disease Risk Probability",
        value=f"{probability * 100:.2f}%"
    )

    st.progress(probability)

    with st.expander("View Input Data Sent to Model"):
        st.dataframe(pd.DataFrame([patient_data]))

    st.info(
        "Model used: XGBoost Classifier. The model was trained on the cleaned Cleveland Heart Disease dataset."
    )