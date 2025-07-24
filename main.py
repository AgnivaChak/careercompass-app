import streamlit as st
from app import resume_parser, jd_parser, matcher, recommender
import tempfile
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Load project ideas
PROJECT_IDEA_PATH = "assets/project_ideas.json"
project_ideas = recommender.load_project_ideas(PROJECT_IDEA_PATH)

st.set_page_config(page_title="CareerCompass - Resume Matcher", layout="wide")
st.title("\U0001F680 CareerCompass: Resume vs JD Matcher")
st.markdown("Upload your resume & job description to get match score, insights, and project recommendations.")

# Upload Resume
resume_file = st.file_uploader("\U0001F4C4 Upload your Resume (PDF)", type=["pdf"], key="resume_file")
jd_input_method = st.radio("\U0001F4CB Job Description Input", ["Upload .txt file", "Paste JD text"], key="jd_input")

# Handle JD input
jd_text = ""
if jd_input_method == "Upload .txt file":
    jd_file = st.file_uploader("Upload Job Description (.txt)", type=["txt"], key="jd_file")
    if jd_file:
        jd_text = jd_file.read().decode("utf-8")
else:
    jd_text = st.text_area("Paste Job Description here", key="jd_text_area")

# Analyze Button
if st.button("\U0001F50D Analyze", key="analyze_button"):
    if resume_file and jd_text.strip():
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_resume:
            temp_resume.write(resume_file.read())
            temp_resume_path = temp_resume.name

        try:
            # Extract and preprocess
            resume_text = resume_parser.extract_text_from_pdf(temp_resume_path)
            resume_info = resume_parser.extract_basic_info(resume_text)
            resume_keywords = resume_parser.extract_resume_keywords(resume_text)
            jd_keywords = jd_parser.extract_jd_keywords(jd_text)

            # Use shared TF-IDF vectorizer
            combined_texts = [" ".join(resume_keywords), " ".join(jd_keywords)]
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(combined_texts).toarray()
            resume_vector = vectors[0]
            jd_vector = vectors[1]

            # Matching Logic
            result = matcher.compare_resume_and_jd(resume_keywords, resume_vector, jd_keywords, jd_vector)

            # Project Suggestions
            projects = recommender.recommend_projects(jd_keywords, resume_keywords, project_ideas)

            # Display Results
            st.subheader("\U0001F4CA Match Summary")
            st.markdown(f"**Match %:** `{result['match_percent']}%`")
            st.markdown(f"**Email:** {resume_info['email']} | **Phone:** {resume_info['phone']}")

            st.subheader("✅ Matched Skills")
            st.write(", ".join(result['matched_skills']) if result['matched_skills'] else "No major overlaps")

            st.subheader("❌ Missing Skills (from JD)")
            st.write(", ".join(result['missing_skills']) if result['missing_skills'] else "You're fully covered!")

            st.subheader("\U0001F4A1 Recommended Projects")
            if projects["suggested"]:
                for proj in projects["suggested"]:
                    with st.expander(f"\U0001F6E0 {proj['project_title']} ({proj['domain']}) [{proj['level']}]"):
                        st.markdown(f"- **Description:** {proj['description']}")
                        st.markdown(f"- **Tools:** {', '.join(proj['tools'])}")
                        st.markdown(f"- **Keywords:** {', '.join(proj['keywords'])}")
            else:
                st.success("✅ No suggestions needed. You're well aligned!")

            # --- Dual Visualization: Heatmap + Gauge ---
            col1, col2 = st.columns(2)

            # Heatmap
            with col1:
                fig = ff.create_annotated_heatmap(
                    z=[[result["match_percent"]]],
                    x=["JD"],
                    y=["Resume"],
                    annotation_text=[[f"{result['match_percent']}%"]],
                    colorscale="YlGnBu",
                    showscale=True,
                    font_colors=["black"],
                    zmin=0,   # Force color scale start
                    zmax=100  # Force color scale end
                )
                fig.update_layout(
                    title_text="Resume vs JD Match Heatmap",
                    title_x=0.5,
                    coloraxis_colorbar=dict(
                        title="Match %",
                        tickvals=[0, 25, 50, 75, 100]
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            # Gauge Chart
            with col2:
                st.subheader("\U0001F3AF Resume–JD Gauge Match")
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=result["match_percent"],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Match %", 'font': {"size": 22}},
                    delta={'reference': 75, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "royalblue"},
                        'bgcolor': "white",
                        'steps': [
                            {'range': [0, 50], 'color': 'tomato'},
                            {'range': [50, 75], 'color': 'gold'},
                            {'range': [75, 100], 'color': 'lightgreen'}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': result["match_percent"]
                        }
                    }
                ))
                gauge_fig.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(gauge_fig)

        except Exception as e:
            st.error(f"\u26A0\ufe0f Error during processing: {str(e)}")

        finally:
            os.remove(temp_resume_path)
    else:
        st.warning("Please upload both a resume and job description.")
