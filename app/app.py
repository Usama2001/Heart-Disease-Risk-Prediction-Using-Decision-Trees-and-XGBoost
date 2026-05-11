from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from header import render_header
from prediction_logic import (
    load_model_and_features,
    predict_heart_disease_risk,
)


st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUNED_THRESHOLD = 0.30


# -------------------------------
# Helper: Find profile image
# -------------------------------
def find_profile_image(project_root: Path):
    image_folder = project_root / "assets"

    possible_files = [
        image_folder / "usama_profile.png",
        image_folder / "usama_profile.jpg",
        image_folder / "profile.png",
        image_folder / "profile.jpg",
        image_folder / "IMG_9152.png",
    ]

    for file_path in possible_files:
        if file_path.exists():
            return file_path

    if image_folder.exists():
        for extension in ["*.png", "*.jpg", "*.jpeg"]:
            image_files = list(image_folder.glob(extension))
            if image_files:
                return image_files[0]

    return None


# -------------------------------
# Helper: Risk details using tuned threshold
# -------------------------------
def get_app_risk_details(probability: float):
    probability_percent = probability * 100

    if probability < TUNED_THRESHOLD:
        return {
            "prediction": 0,
            "risk_level": "Low Risk",
            "risk_message": "Prediction: Low Heart Disease Risk",
            "card_class": "prediction-card-low",
            "probability_percent": probability_percent,
        }

    if probability < 0.70:
        return {
            "prediction": 1,
            "risk_level": "Moderate Risk",
            "risk_message": "Prediction: Moderate Heart Disease Risk",
            "card_class": "prediction-card-medium",
            "probability_percent": probability_percent,
        }

    return {
        "prediction": 1,
        "risk_level": "High Risk",
        "risk_message": "Prediction: Heart Disease Risk",
        "card_class": "prediction-card-high",
        "probability_percent": probability_percent,
    }


# -------------------------------
# Helper: Footer
# -------------------------------
def render_clean_footer(project_root: Path):
    profile_image = find_profile_image(project_root)

    st.markdown("---")

    with st.container(border=True):
        col_img, col_info = st.columns([0.8, 5.2], vertical_alignment="center")

        with col_img:
            if profile_image:
                st.image(str(profile_image), width=76)
            else:
                st.markdown(
                    """
                    <div style="
                        width:76px;
                        height:76px;
                        border-radius:50%;
                        background:#E5E7EB;
                        color:#111827;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        font-size:24px;
                        font-weight:800;
                        border:2px solid #3B82F6;
                    ">
                        UF
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_info:
            st.markdown("### Usama Fiaz")
            st.markdown(
                "**AI Engineer** | Machine Learning | NLP | LangGraph | AI Agents"
            )

            btn_col1, btn_col2, empty_col = st.columns([1.1, 1.1, 4])

            with btn_col1:
                st.link_button(
                    "📧 Email",
                    "mailto:usama20010101@gmail.com",
                    use_container_width=True,
                )

            with btn_col2:
                st.link_button(
                    "🔗 LinkedIn",
                    "https://www.linkedin.com/in/usama2001/",
                    use_container_width=True,
                )

            with empty_col:
                st.empty()

            st.caption(
                "Portfolio ML project built using Python, Scikit-learn, XGBoost, and Streamlit. "
                "This project demonstrates data cleaning, EDA, model comparison, threshold tuning, "
                "prediction pipeline, and interactive ML app development."
            )


# -------------------------------
# CSS
# -------------------------------
st.markdown(
    dedent(
        """
        <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }

        .sticky-header {
            position: sticky;
            top: 0;
            z-index: 9999;
            background: rgba(14, 17, 23, 0.98);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid #1F2937;
            padding: 26px 10px 22px 10px;
            margin-bottom: 30px;
            text-align: center;
        }

        .main-title {
            font-size: 38px;
            font-weight: 850;
            color: #FFFFFF;
            margin-bottom: 10px;
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
            margin-bottom: 16px;
        }

        .prediction-card-medium {
            background: linear-gradient(135deg, #92400E, #B45309);
            padding: 32px;
            border-radius: 20px;
            border: 1px solid #F59E0B;
            color: white;
            margin-bottom: 16px;
        }

        .prediction-card-high {
            background: linear-gradient(135deg, #7F1D1D, #B91C1C);
            padding: 32px;
            border-radius: 20px;
            border: 1px solid #EF4444;
            color: white;
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

        div[data-testid="stSidebar"] {
            background-color: #111827;
        }

        div[data-testid="stSidebar"] h1,
        div[data-testid="stSidebar"] h2,
        div[data-testid="stSidebar"] h3 {
            color: white;
        }
        </style>
        """
    ).strip(),
    unsafe_allow_html=True,
)


# -------------------------------
# Load model
# -------------------------------
try:
    model, feature_names = load_model_and_features()
except FileNotFoundError:
    st.error(
        "Model files not found. Please run the notebook up to Step 11 to generate model files inside the models folder."
    )
    st.stop()


# -------------------------------
# Sidebar inputs
# -------------------------------
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


# -------------------------------
# Header
# -------------------------------
render_header()


# -------------------------------
# Project overview
# -------------------------------
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
    ).strip(),
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
    ).strip(),
    unsafe_allow_html=True,
)


# -------------------------------
# Prediction panel
# -------------------------------
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
    _, probability, model_input_df = predict_heart_disease_risk(
        patient_data,
        model,
        feature_names,
    )

    risk_details = get_app_risk_details(probability)

    st.session_state.prediction_result = {
        "prediction": risk_details["prediction"],
        "probability": probability,
        "model_input_df": model_input_df,
        "risk_details": risk_details,
    }

if st.session_state.prediction_result is None:
    st.info(
        "Enter patient details from the left sidebar, then click **Generate New Prediction** to see the result."
    )
else:
    prediction = st.session_state.prediction_result["prediction"]
    probability = st.session_state.prediction_result["probability"]
    model_input_df = st.session_state.prediction_result["model_input_df"]
    risk_details = st.session_state.prediction_result["risk_details"]

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
        ).strip(),
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
            ).strip(),
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
            ).strip(),
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
            ).strip(),
            unsafe_allow_html=True,
        )

    with st.expander("View Data Sent to Model"):
        st.dataframe(
            model_input_df.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

    st.info(
        "The model uses the same feature order saved during training. A tuned threshold of 0.30 is used to improve recall for heart disease risk screening."
    )


# -------------------------------
# Feature meaning guide
# -------------------------------
st.markdown(
    '<div class="section-title">Feature Meaning Guide</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown("### Chest Pain Type")
    st.markdown(
        """
        - **Typical Angina:** chest pain commonly related to heart disease.
        - **Atypical Angina:** chest pain that is not fully typical.
        - **Non-Anginal Pain:** chest pain usually not caused by heart disease.
        - **Asymptomatic:** no clear chest pain symptoms.
        """
    )

    st.markdown("### Exercise Induced Angina")
    st.markdown("- Shows whether chest pain occurs during exercise.")

    st.markdown("### Oldpeak")
    st.markdown(
        "- ST depression value from exercise test. Higher values may show heart stress."
    )

    st.markdown("### CA")
    st.markdown("- Number of major blood vessels visible through fluoroscopy.")

    st.markdown("### Thalassemia")
    st.markdown("- A blood-related test category used in the dataset.")


# -------------------------------
# Model performance summary
# -------------------------------
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

st.dataframe(performance_data, use_container_width=True, hide_index=True)

st.markdown("### Model Performance Chart")

fig, ax = plt.subplots(figsize=(10, 5))

performance_data.set_index("Model").plot(
    kind="bar",
    ax=ax,
    edgecolor="black",
)

ax.set_title("Final Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.set_ylim(0, 1)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(loc="lower right")

plt.xticks(rotation=0)
plt.tight_layout()

st.pyplot(fig)

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
            After threshold tuning, a 0.30 threshold is used for screening because it improves recall
            and reduces false negatives.
            </p>
        </div>
        """
    ).strip(),
    unsafe_allow_html=True,
)


# -------------------------------
# Footer
# -------------------------------
render_clean_footer(PROJECT_ROOT)
