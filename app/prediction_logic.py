from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_heart_disease_model.pkl"
FEATURE_NAMES_PATH = PROJECT_ROOT / "models" / "feature_names.pkl"


@st.cache_resource
def load_model_and_features():
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    return model, feature_names


def predict_heart_disease_risk(patient_data, model, feature_names):
    patient_df = pd.DataFrame([patient_data])
    patient_df = patient_df[feature_names]

    prediction = model.predict(patient_df)[0]
    probability = model.predict_proba(patient_df)[0][1]

    return int(prediction), float(probability), patient_df


def get_risk_details(prediction, probability):
    probability_percent = probability * 100

    if probability < 0.40:
        return {
            "risk_level": "Low Risk",
            "risk_message": "Prediction: Low Heart Disease Risk",
            "card_class": "prediction-card-low",
            "probability_percent": probability_percent,
        }

    if probability < 0.70:
        return {
            "risk_level": "Moderate Risk",
            "risk_message": "Prediction: Moderate Heart Disease Risk",
            "card_class": "prediction-card-medium",
            "probability_percent": probability_percent,
        }

    return {
        "risk_level": "High Risk",
        "risk_message": "Prediction: Heart Disease Risk",
        "card_class": "prediction-card-high",
        "probability_percent": probability_percent,
    }