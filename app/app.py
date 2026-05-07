from pathlib import Path
from textwrap import dedent

import pandas as pd
import streamlit as st

from footer import render_footer
from header import render_header
from prediction_logic import (
    get_risk_details,
    load_model_and_features,
    predict_heart_disease_risk,
)


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide",
)


# --------------------------------------------------
# Project Path
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown(
    dedent(
        """
        <style>
        .block-container {
            padding-top: 7.5rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }

        .fixed-header {
            position: fixed;
            top: 0;
            left: 18rem;
            right: 0;
            z-index: 999999;
            background: rgba(14, 17, 23, 0.98);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid #1F2937;
            padding: 20px 40px 18px 40px;
            text-align: center;
        }

        .main-title {
            font-size: 38px;
            font-weight: 850;
            color: #FFFFFF;
            margin-bottom: 6px;
            line-height: 1.15;
        }

        .subtitle {
            font-size: 16px;
            color: #D1D5DB;
        }

        .section-title {
            font-size: 28px;
            font-weight: 800;
            color: #FFFFFF;
            margin-top: 30px;
            margin-bottom: 16px;
        }

        .info-card {
            background: linear-gradient(135deg, #1F2937, #111827);
            padding: 28px;
            border-radius: 18px;
            border: 1px solid #374151;
            margin-bottom: 20px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.18);
        }

        .info-card h3 {
            color: #FFFFFF;
            margin-bottom: 14px;
        }

        .info-card p {
            color: #E5E7EB;
            font-size: 15.5px;
            line-height: 1.7;
        }

        .muted-text {
            color: #9CA3AF !important;
            font-size: 14px !important;
        }

        .disclaimer-card {
            background: #3F4612;
            border: 1px solid #6B721D;
            color: #FEF9C3;
            padding: 18px 22px;
            border-radius: 12px;
            font-weight: 600;
            margin-bottom: 25px;
        }

        .prediction-card-low {
            background: linear-gradient(135deg, #064E3B, #047857);
            padding: 32px;
            border-radius: 20px;
            border: 1px solid #10B981;
            color: white;
            box-shadow: 0 12px 30px rgba(16, 185, 129, 0.16);
            margin-bottom: 16px;
        }

        .prediction-card-medium {
            background: linear-gradient(135deg, #92400E, #B45309);
            padding: 32px;
            border-radius: 20px;
            border: 1px solid #F59E0B;
            color: white;
            box-shadow: 0 12px 30px rgba(245, 158, 11, 0.16);
            margin-bottom: 16px;
        }

        .prediction-card-high {
            background: linear-gradient(135deg, #7F1D1D, #B91C1C);
            padding: 32px;
            border-radius: 20px;
            border: 1px solid #EF4444;
            color: white;
            box-shadow: 0 12px 30px rgba(239, 68, 68, 0.16);
            margin-bottom: 16px;
        }

        .prediction-card-low h2,
        .prediction-card-medium h2,
        .prediction-card-high h2 {
            font-size: 30px;
            margin-bottom: 18px;
        }

        .prediction-card-low h1,
        .prediction-card-medium h1,
        .prediction-card-high h1 {
            font-size: 54px;
            margin-bottom: 10px;
        }

        .metric-card {
            background: #111827;
            border: 1px solid #374151;
            border-radius: 16px;
            padding: 18px;
            text-align: center;
            min-height: 110px;
            margin-bottom: 14px;
        }

        .metric-label {
            color: #9CA3AF;
            font-size: 13px;
            margin-bottom: 8px;
        }

        .metric-value {
            color: #FFFFFF;
            font-size: 28px;
            font-weight: 750;
        }

        .guide-card {
            background: #111827;
            border: 1px solid #374151;
            border-radius: 18px;
            padding: 28px;
            margin-top: 16px;
            margin-bottom: 26px;
        }

        .guide-card h4 {
            color: #FFFFFF;
            margin-top: 18px;
            margin-bottom: 8px;
        }

        .guide-card ul {
            color: #E5E7EB;
            line-height: 1.8;
        }

        .guide-card li {
            margin-bottom: 6px;
        }

        .why-card {
            background: linear-gradient(135deg, #172554, #1E3A8A);
            border: 1px solid #3B82F6;
            border-radius: 18px;
            padding: 24px;
            margin-top: 18px;
            margin-bottom: 30px;
        }

        .why-card h3 {
            color: #FFFFFF;
            margin-bottom: 12px;
        }

        .why-card p {
            color: #DBEAFE;
            line-height: 1.7;
        }

        .footer-card {
            background: linear-gradient(135deg, #111827, #1F2937);
            border: 1px solid #374151;
            border-radius: 22px;
            padding: 26px;
            margin-top: 40px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 24px;
        }

        .footer-image-box {
            flex-shrink: 0;
        }

        .profile-img {
            width: 92px;
            height: 92px;
            border-radius: 50%;
            object-fit: cover;
            border: 2px solid #3B82F6;
            background: white;
        }

        .profile-placeholder {
            width: 92px;
            height: 92px;
            border-radius: 50%;
            border: 2px solid #3B82F6;
            background: #E5E7EB;
            color: #111827;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 850;
            font-size: 26px;
        }

        .footer-name {
            font-size: 24px;
            font-weight: 800;
            color: #FFFFFF;
            margin-bottom: 4px;
        }

        .footer-role {
            color: #D1D5DB;
            font-size: 15px;
            margin-bottom: 10px;
        }

        .footer-links a {
            color: #60A5FA;
            text-decoration: none;
            margin-right: 18px;
            font-weight: 650;
        }

        .footer-links a:hover {
            color: #93C5FD;
            text-decoration: underline;
        }

        .footer-note {
            color: #9CA3AF;
            font-size: 13.5px;
            margin-top: 10px;
            line-height: 1.6;
        }

        div[data-testid="stSidebar"] {
            background-color: #111827;
        }

        div[data-testid="stSidebar"] h1,
        div[data-testid="stSidebar"] h2,
        div[data-testid="stSidebar"] h3 {
            color: white;
        }

        @media screen and (max-width: 900px) {
            .fixed-header {
                left: 0;
                padding: 16px 16px;
            }

            .main-title {
                font-size: 28px;
            }

            .subtitle {
                font-size: 14px;
            }

            .block-container {
                padding-top: 7rem;
            }

            .footer-card {
                flex-direction: column;
                text-align: center;
            }
        }
        </style>
        """
    ),
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------
try:
    model, feature_names = load_model_and_features()
except FileNotFoundError:
    st.error(
        "Model files not found. Please run the notebook up to Step 11 to generate model files inside the models folder."
    )
    st.stop()


# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
st.sidebar.title("Patient Input Panel")
st.sidebar.write("Enter patient medical information below.")

age = st.sidebar.number_input("Age", min_value=20, max_value=100, value=54)

sex_label = st.sidebar.selectbox("Sex", ["Female", "Male"])
sex = 1 if sex_label == "Male" else 0

cp_label = st.sidebar.selectbox(
    "Chest Pain Type",
    ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
)
cp_mapping = {
    "Typical Angina": 1,
    "Atypical Angina": 2,
    "Non-Anginal Pain": 3,
    "Asymptomatic": 4,
}
cp = cp_mapping[cp_label]

trestbps = st.sidebar.number_input(
    "Resting Blood Pressure",
    min_value=80,
    max_value=220,
    value=130,
)

chol = st.sidebar.number_input(
    "Cholesterol",
    min_value=100,
    max_value=600,
    value=250,
)

fbs_label = st.sidebar.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    ["No", "Yes"],
)
fbs = 1 if fbs_label == "Yes" else 0

restecg_label = st.sidebar.selectbox(
    "Resting ECG Result",
    ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"],
)
restecg_mapping = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2,
}
restecg = restecg_mapping[restecg_label]

thalach = st.sidebar.number_input(
    "Maximum Heart Rate Achieved",
    min_value=60,
    max_value=230,
    value=150,
)

exang_label = st.sidebar.selectbox(
    "Exercise Induced Angina",
    ["No", "Yes"],
)
exang = 1 if exang_label == "Yes" else 0

oldpeak = st.sidebar.number_input(
    "Oldpeak / ST Depression",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1,
)

slope_label = st.sidebar.selectbox(
    "Slope of Peak Exercise ST Segment",
    ["Upsloping", "Flat", "Downsloping"],
)
slope_mapping = {
    "Upsloping": 1,
    "Flat": 2,
    "Downsloping": 3,
}
slope = slope_mapping[slope_label]

ca = st.sidebar.selectbox(
    "Number of Major Vessels Colored by Fluoroscopy",
    [0, 1, 2, 3],
)

thal_label = st.sidebar.selectbox(
    "Thalassemia",
    ["Normal", "Fixed Defect", "Reversible Defect"],
)
thal_mapping = {
    "Normal": 3,
    "Fixed Defect": 6,
    "Reversible Defect": 7,
}
thal = thal_mapping[thal_label]


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
    "thal": thal,
}


# --------------------------------------------------
# Header
# --------------------------------------------------
render_header()


# --------------------------------------------------
# Project Overview
# --------------------------------------------------
st.markdown(
    dedent(
        """
        <div class="info-card">
            <h3>Project Overview</h3>
            <p>
            This application uses patient medical information such as age, chest pain type,
            blood pressure, cholesterol, maximum heart rate, and other clinical features
            to estimate the probability of heart disease risk.
            </p>
            <p class="muted-text">
            Model used: XGBoost Classifier | Dataset: Cleveland Heart Disease Dataset
            </p>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

st.markdown(
    dedent(
        """
        <div class="disclaimer-card">
            Medical Disclaimer: This app is for educational and portfolio purposes only.
            It should not be used as a real medical diagnosis tool.
        </div>
        """
    ),
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Prediction Panel
# --------------------------------------------------
st.markdown(
    '<div class="section-title">Prediction Panel</div>',
    unsafe_allow_html=True,
)

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

predict_button = st.button(
    "Generate New Prediction",
    type="primary",
    use_container_width=True,
)

if predict_button:
    prediction, probability, model_input_df = predict_heart_disease_risk(
        patient_data,
        model,
        feature_names,
    )

    st.session_state.prediction_result = {
        "prediction": prediction,
        "probability": probability,
        "model_input_df": model_input_df,
    }

if st.session_state.prediction_result is None:
    st.info(
        "Enter patient details from the left sidebar, then click **Generate New Prediction** to see the result."
    )
else:
    prediction = st.session_state.prediction_result["prediction"]
    probability = st.session_state.prediction_result["probability"]
    model_input_df = st.session_state.prediction_result["model_input_df"]

    risk_details = get_risk_details(prediction, probability)

    risk_level = risk_details["risk_level"]
    risk_message = risk_details["risk_message"]
    prediction_card_class = risk_details["card_class"]
    probability_percent = risk_details["probability_percent"]

    st.markdown(
        dedent(
            f"""
            <div class="{prediction_card_class}">
                <h2>{risk_message}</h2>
                <h1>{probability_percent:.2f}%</h1>
                <p>Estimated heart disease risk probability</p>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    st.progress(probability)

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.markdown(
            dedent(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Risk Level</div>
                    <div class="metric-value">{risk_level}</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    with metric_col2:
        st.markdown(
            dedent(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Prediction Class</div>
                    <div class="metric-value">{prediction}</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    with metric_col3:
        st.markdown(
            dedent(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Probability</div>
                    <div class="metric-value">{probability_percent:.2f}%</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    with st.expander("View Data Sent to Model"):
        st.dataframe(model_input_df, use_container_width=True)

    st.info(
        "The model uses the same feature order that was saved during training to avoid prediction errors."
    )


# --------------------------------------------------
# Feature Meaning Guide
# --------------------------------------------------
st.markdown(
    '<div class="section-title">Feature Meaning Guide</div>',
    unsafe_allow_html=True,
)

st.markdown(
    dedent(
        """
        <div class="guide-card">
            <h4>Chest Pain Type</h4>
            <ul>
                <li><b>Typical Angina:</b> chest pain commonly related to heart disease.</li>
                <li><b>Atypical Angina:</b> chest pain that is not fully typical.</li>
                <li><b>Non-Anginal Pain:</b> chest pain usually not caused by heart disease.</li>
                <li><b>Asymptomatic:</b> no clear chest pain symptoms.</li>
            </ul>

            <h4>Exercise Induced Angina</h4>
            <ul>
                <li>Shows whether chest pain occurs during exercise.</li>
            </ul>

            <h4>Oldpeak</h4>
            <ul>
                <li>ST depression value from exercise test. Higher values may show heart stress.</li>
            </ul>

            <h4>CA</h4>
            <ul>
                <li>Number of major blood vessels visible through fluoroscopy.</li>
            </ul>

            <h4>Thalassemia</h4>
            <ul>
                <li>A blood-related test category used in the dataset.</li>
            </ul>
        </div>
        """
    ),
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Model Performance Summary
# --------------------------------------------------
st.markdown(
    '<div class="section-title">Model Performance Summary</div>',
    unsafe_allow_html=True,
)

performance_data = pd.DataFrame(
    {
        "Model": ["Logistic Regression", "Decision Tree", "XGBoost"],
        "Accuracy": [0.833333, 0.766667, 0.850000],
        "Precision": [0.846154, 0.850000, 0.880000],
        "Recall": [0.785714, 0.607143, 0.785714],
        "F1-score": [0.814815, 0.708333, 0.830189],
    }
)

st.dataframe(performance_data, use_container_width=True)

st.markdown(
    dedent(
        """
        <div class="why-card">
            <h3>Why XGBoost?</h3>
            <p>
            XGBoost was selected as the final model because it achieved the best overall
            performance compared with Logistic Regression and a single Decision Tree.
            It had the highest accuracy, precision, and F1-score while maintaining the same recall
            as the Logistic Regression baseline.
            </p>
        </div>
        """
    ),
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Footer
# --------------------------------------------------
render_footer(PROJECT_ROOT)