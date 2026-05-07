from textwrap import dedent

import streamlit as st


def render_header():
    st.markdown(
        dedent(
            """
            <div class="sticky-header">
                <div class="main-title">❤️ Heart Disease Risk Prediction App</div>
                <div class="subtitle">
                    A machine learning web app that predicts heart disease risk using a trained XGBoost model.
                </div>
            </div>
            """
        ).strip(),
        unsafe_allow_html=True,
    )