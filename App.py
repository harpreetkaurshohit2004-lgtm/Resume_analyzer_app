import streamlit as st
import pandas as pd
import plotly.express as px
import os
from PIL import Image
from resume_logic import (
    extract_text,
    extract_candidate_details,
    extract_skills_from_jd,
    detect_skill_gaps,
    save_to_db,
    load_all_data,
    model,
    util,
    JOB_CATEGORIES,
    ROLE_SKILLS
)

# ---------------------- APP CONFIG ----------------------
st.set_page_config(
    page_title="Smart Resume Analyzer",
    layout="wide",
    page_icon="📄"
)

# ---------------------- THEME & STYLING ----------------------
st.markdown("""
<style>
body {
    background-color: #1e1e1e;
    color: #F5F5F5;
}
h1,h2,h3,h4 {
    color: #FFD700 !important;  /* Gold headers */
}
.stButton>button {
    background-color: #800020;  /* Burgundy */
    color: #F5F5F5;
    border-radius: 8px;
    padding: 8px 20px;
    border: none;
}
.stButton>button:hover {
    background-color: #FFD700;  /* Gold hover */
    color: #2E2E2E;
}
footer {
    color: gray;
    text-align: center;
    padding-top: 20px;
}
.stDataFrame thead th {
    background-color: #800020;
    color: #F5F5F5;
}
</style>
""", unsafe_allow_html=True)

# ---------------------- SIDEBAR NAVIGATION ----------------------
from streamlit_option_menu import option_menu
import streamlit as st

# ----- Sidebar Styling -----
with st.sidebar:

    # Logo (Optional – remove if not needed)
    st.markdown("""
        <div style="text-align:center; margin-bottom: 10px;">
            <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" width="70">
        </div>
    """, unsafe_allow_html=True)

    # Title + Subtitle
    st.markdown("""
        <h1 style='color:white; text-align:center; margin-bottom:0;'>
            Smart Resume Analyzer
        </h1>
        <h2 style='color:#FFD700; font-size:14px; text-align:center; margin-top:2px;'>
            NLP-Based Resume Analyzer
        </h2>
        <hr style='border:1px solid #333;'>
    """, unsafe_allow_html=True)

    # Option Menu
    page = option_menu(
        "",
        ["Upload & Analyze", "Dashboard", "Admin Panel"],
        icons=["cloud-upload", "bar-chart", "shield-lock"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {
                "padding": "10px",
                "background-color": "#1e1e1e",
                "border-radius": "10px"
            },
            "icon": {"color": "#FFD700", "font-size": "20px"},
            "nav-link": {
                "color": "white",
                "font-size": "16px",
                "padding": "10px",
                "margin": "8px 0",
                "background-color": "#800020",
                "border-radius": "10px",
                "transition": "0.3s",
            },
            "nav-link-hover": {
                "background-color": "#A00030",
                "color": "white"
            },
            "nav-link-selected": {
                "background-color": "#FFD700",
                "color": "#2E2E2E",
                "font-weight": "bold",
                "box-shadow": "0 0 10px #FFD700"
            },
        },
    )

    # Footer Info
    st.markdown("""
        <hr style='border:1px solid #333; margin-top:20px;'>
        <p style="color:#ccc; font-size:13px; text-align:center;">
            ✨ Powered by NLP ✨<br>
            © 2025 Smart Resume Analyzer
        </p>
    """, unsafe_allow_html=True)


# ---------------------- PAGE 1: MULTI RESUME ANALYSIS ----------------------
if page == "Upload & Analyze":
    st.title("Upload Resumes for Analysis")
    st.write("Upload multiple resumes and compare them against a specific job role using NLP-based similarity scoring.")

    col1, col2 = st.columns(2)
    with col1:
        job_category = st.selectbox("Select Job Category", list(JOB_CATEGORIES.keys()))
    with col2:
        role = st.selectbox("Select Specific Role", JOB_CATEGORIES[job_category])

    uploaded_files = st.file_uploader(
        "Upload Resumes (PDF, DOCX, or TXT)",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )

    if st.button("Start Analysis", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one resume.")
        else:
            st.info(f"Analyzing {len(uploaded_files)} resume(s) for the **{role}** role...")
            with st.spinner("Running AI model for semantic matching..."):

                jd_text = ROLE_SKILLS.get(role, "")
                jd_embedding = model.encode(jd_text, convert_to_tensor=True)
                jd_skills = extract_skills_from_jd(jd_text)

                results = []
                for file in uploaded_files:
                    text = extract_text(file)
                    if text.strip():
                        resume_embedding = model.encode(text, convert_to_tensor=True)
                        similarity = util.cos_sim(jd_embedding, resume_embedding).item()

                        info = extract_candidate_details(text)
                        matched_skills, skill_gaps = detect_skill_gaps(jd_skills, info["Skills"])

                        results.append({
                            "Name": info["Name"],
                            "Email": info["Email"],
                            "Phone": info["Phone"],
                            "Resume": file.name,
                            "Similarity": round(similarity * 100, 2),
                            "Matched Skills": ", ".join(matched_skills),
                            "Skill Gaps": ", ".join(skill_gaps)
                        })

                        save_to_db({
                            "name": info["Name"],
                            "email": info["Email"],
                            "phone": info["Phone"],
                            "filename": file.name,
                            "job_category": job_category,
                            "role": role,
                            "similarity": round(similarity * 100, 2),
                            "matched_skills": ", ".join(matched_skills),
                            "skill_gaps": ", ".join(skill_gaps)
                        })

                if results:
                    df = pd.DataFrame(results).sort_values("Similarity", ascending=False)
                    st.session_state["analysis_results"] = df
                    st.session_state["selected_role"] = role

                    st.success("Analysis Complete!")
                    st.subheader("Ranked Candidates")
                    st.dataframe(df, use_container_width=True)
                    st.download_button(
                        "Download Analysis Report (CSV)",
                        df.to_csv(index=False),
                        file_name="resume_analysis_report.csv"
                    )
                    st.info("You can now view the Dashboard or Admin Panel for deeper insights.")
                else:
                    st.warning("No valid resumes processed.")

# ---------------------- PAGE 2: DASHBOARD ----------------------
elif page == "Dashboard":
    st.title("Resume Insights Dashboard")

    if "analysis_results" not in st.session_state:
        st.info("Please upload and analyze resumes first.")
    else:
        df = st.session_state["analysis_results"]
        role = st.session_state.get("selected_role", "Selected Role")

        st.subheader(f"Insights for: {role}")
        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Candidates", len(df))
        col2.metric("Average Match Score", f"{df['Similarity'].mean():.1f}%")
        col3.metric("Top Candidate Score", f"{df['Similarity'].max():.1f}%")

        st.markdown("---")

        # --- Bar Chart: Candidate Similarity Scores ---
        fig1 = px.bar(
            df, x="Name", y="Similarity",
            title="Candidate Match Scores",
            color="Similarity",
            color_continuous_scale=["#800020", "#FFD700"],  # Burgundy to Gold
            template="plotly_dark"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # --- Skill Gap Frequency ---
        all_gaps = [gap.strip() for gaps in df["Skill Gaps"] for gap in gaps.split(",") if gap.strip()]
        if all_gaps:
            gap_df = pd.DataFrame({"Skill Gap": all_gaps})
            gap_count = gap_df["Skill Gap"].value_counts().reset_index()
            gap_count.columns = ["Skill", "Count"]
            fig2 = px.bar(
                gap_count, x="Skill", y="Count",
                title="Most Common Skill Gaps",
                color="Count",
                color_continuous_scale=["#800020", "#FFD700"],  # Burgundy → Gold
                template="plotly_dark"
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No skill gap data available.")

        # --- Pie Chart: Top 5 Candidates ---
        top5 = df.head(5)
        fig3 = px.pie(
            top5, names="Name", values="Similarity",
            title="Top 5 Candidate Match Distribution",
            color_discrete_sequence=["#800020", "#FFD700", "#B22222", "#FFD700", "#800020"]
        )
        st.plotly_chart(fig3, use_container_width=True)

# ---------------------- PAGE 3: ADMIN PANEL ----------------------
elif page == "Admin Panel":
    st.title("Recruiter Admin Panel")

    username = st.text_input("Admin Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "12345":
            st.success("Logged in successfully!")
            st.markdown("---")

            data = load_all_data()
            if data.empty:
                st.info("No candidate data available yet.")
            else:
                st.subheader("All Candidate Data")
                st.dataframe(data, use_container_width=True)

                st.download_button(
                    "Download Full Report (CSV)",
                    data.to_csv(index=False),
                    file_name="all_candidates_report.csv"
                )

                st.markdown("---")
                st.subheader("Analytics Overview")

                fig1 = px.box(data, x="role", y="similarity", color="job_category",
                              title="Role-wise Candidate Score Distribution",
                              color_discrete_sequence=["#800020", "#FFD700"],
                              template="plotly_dark")
                st.plotly_chart(fig1, use_container_width=True)

                gap_skills = []
                for s in data["skill_gaps"]:
                    for skill in s.split(","):
                        if skill.strip():
                            gap_skills.append(skill.strip())

                if gap_skills:
                    gap_df = pd.DataFrame(gap_skills, columns=["Skill Gap"])
                    gap_count = gap_df["Skill Gap"].value_counts().reset_index()
                    gap_count.columns = ["Skill", "Frequency"]
                    fig2 = px.bar(
                        gap_count, x="Skill", y="Frequency",
                        title="Most Common Skill Gaps (All Candidates)",
                        color="Frequency",
                        color_continuous_scale=["#800020", "#FFD700"],
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.error("Invalid credentials. Try again.")





