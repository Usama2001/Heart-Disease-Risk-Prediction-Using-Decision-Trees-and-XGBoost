import base64
from pathlib import Path
from textwrap import dedent

import streamlit as st


def image_to_base64(image_path):
    if image_path and image_path.exists():
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    return None


def find_profile_image(project_root):
    images_folder = project_root / "imges"

    preferred_files = [
        images_folder / "usama_profile.png",
        images_folder / "usama_profile.jpg",
        images_folder / "profile.png",
        images_folder / "profile.jpg",
    ]

    for file_path in preferred_files:
        if file_path.exists():
            return file_path

    if images_folder.exists():
        for extension in ["*.png", "*.jpg", "*.jpeg"]:
            image_files = list(images_folder.glob(extension))
            if image_files:
                return image_files[0]

    return None


def render_footer(project_root):
    profile_image_path = find_profile_image(project_root)
    profile_image_base64 = image_to_base64(profile_image_path)

    if profile_image_base64:
        profile_image_html = (
            f'<img class="profile-img" src="data:image/png;base64,{profile_image_base64}" alt="Usama Fiaz">'
        )
    else:
        profile_image_html = '<div class="profile-placeholder">UF</div>'

    footer_html = f"""
    <div class="footer-card">
        <div class="footer-image-box">
            {profile_image_html}
        </div>

        <div class="footer-content">
            <div class="footer-name">Usama Fiaz</div>
            <div class="footer-role">
                AI Engineer | Machine Learning | NLP | LangGraph | AI Agents
            </div>

            <div class="footer-links">
                <a href="mailto:usama20010101@gmail.com">📧 Email</a>
                <a href="https://www.linkedin.com/in/usama2001/" target="_blank">🔗 LinkedIn</a>
            </div>

            <div class="footer-note">
                Portfolio ML project using Python, Scikit-learn, XGBoost, and Streamlit.
                Built with data cleaning, EDA, model comparison, prediction pipeline, and interactive app deployment.
            </div>
        </div>
    </div>
    """

    st.markdown(dedent(footer_html), unsafe_allow_html=True)