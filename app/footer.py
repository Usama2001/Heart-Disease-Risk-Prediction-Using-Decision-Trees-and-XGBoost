from pathlib import Path

import streamlit as st


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


def render_footer(project_root: Path):
    profile_image = find_profile_image(project_root)

    st.markdown("---")

    with st.container(border=True):
        col_img, col_info = st.columns([1, 5])

        with col_img:
            if profile_image:
                st.image(str(profile_image), width=95)
            else:
                st.markdown(
                    """
                    <div style="
                        width:150px;
                        height:200px;
                        border-radius:50%;
                        background:#E5E7EB;
                        color:#111827;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        font-size:28px;
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

            link_col1, link_col2 = st.columns([1, 1])

            with link_col1:
                st.link_button(
                    "📧 Email",
                    "mailto:usama20010101@gmail.com",
                    use_container_width=True,
                )

            with link_col2:
                st.link_button(
                    "🔗 LinkedIn",
                    "https://www.linkedin.com/in/usama2001/",
                    use_container_width=True,
                )

            st.caption(
                "Portfolio ML project built using Python, Scikit-learn, XGBoost, and Streamlit. "
                "This project demonstrates data cleaning, EDA, model comparison, prediction pipeline, "
                "and interactive ML app development."
            )