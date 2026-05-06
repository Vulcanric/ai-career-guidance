"""Streamlit frontend for the AI Career Guidance Agent.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from agent_logic import (
    build_downloadable_text_report,
    generate_guidance,
    load_knowledge_base,
    recommend_courses_for_career,
    split_semicolon_text,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


st.set_page_config(
    page_title="AI Career Guidance Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def cached_load_data():
    """Cache CSV loading so the app remains responsive."""
    return load_knowledge_base(DATA_DIR)


careers_df, courses_df, profiles_df, skills_df = cached_load_data()


# Basic CSS improves the presentation for demo and screenshots.
st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .sub-title {
            color: #555;
            font-size: 1.05rem;
            margin-bottom: 1rem;
        }
        .agent-card {
            padding: 1rem;
            border-radius: 0.8rem;
            border: 1px solid #e6e6e6;
            background: #fafafa;
            margin-bottom: 0.8rem;
        }
        .small-muted {
            color: #666;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown('<div class="main-title">🎓 AI Career Guidance Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Analyse student interests, grades and skills, then recommend career paths, courses and future scenarios.</div>',
    unsafe_allow_html=True,
)


# Sidebar contains knowledge base information and demo profile support.
with st.sidebar:
    st.header("Included Knowledge Base")
    st.write(f"Career paths: **{len(careers_df)}**")
    st.write(f"Courses: **{len(courses_df)}**")
    st.write(f"Skills: **{len(skills_df)}**")
    st.write(f"Sample profiles: **{len(profiles_df)}**")

    st.divider()
    st.header("Demo Profile")
    sample_profile_names = ["Manual entry"] + profiles_df["student_name"].tolist()
    selected_sample = st.selectbox("Load sample student", sample_profile_names)


# When a sample profile is selected, use it as default input values.
if selected_sample != "Manual entry":
    selected_profile = profiles_df[profiles_df["student_name"] == selected_sample].iloc[0].to_dict()
else:
    selected_profile = {
        "student_name": "",
        "programme": "MSc Artificial Intelligence Technology",
        "grade_percentage": 60,
        "interests": "AI agents; automation; career guidance; data analysis",
        "skills": "Python; Streamlit; SQL; communication",
        "working_style": "creative; technical; analytical",
        "career_goal": "Build intelligent automation systems",
    }


# Build dynamic options from the CSV knowledge base.
all_interest_options = sorted(
    set(
        item
        for text in careers_df["preferred_interests"].fillna("").tolist()
        for item in split_semicolon_text(text)
    )
)
all_skill_options = sorted(skills_df["skill_name"].dropna().unique().tolist())
all_style_options = sorted(
    set(
        item
        for text in careers_df["working_style"].fillna("").tolist()
        for item in split_semicolon_text(text)
    )
)


def convert_text_to_valid_defaults(text_value, options):
    """Select only values that exist in a multiselect option list."""
    text_items = split_semicolon_text(str(text_value))
    return [item for item in text_items if item in options]

st.subheader("Student Profile Input")
with st.form("student_profile_form"):
    student_name = st.text_input("Student Name", value=str(selected_profile.get("student_name", "")))
    programme = st.selectbox(
        "Programme / Course",
        [
            "MSc Artificial Intelligence Technology",
            "MSc Big Data & Data Science",
            "MSc Cyber Security Technology",
            "MSc Computing Technology",
            "MSc Cross-Disciplinary",
            "Other",
        ],
        index=[
            "MSc Artificial Intelligence Technology",
            "MSc Big Data & Data Science",
            "MSc Cyber Security Technology",
            "MSc Computing Technology",
            "MSc Cross-Disciplinary",
            "Other",
        ].index(str(selected_profile.get("programme", "MSc Artificial Intelligence Technology")))
        if str(selected_profile.get("programme", "")) in [
            "MSc Artificial Intelligence Technology",
            "MSc Big Data & Data Science",
            "MSc Cyber Security Technology",
            "MSc Computing Technology",
            "MSc Cross-Disciplinary",
            "Other",
        ] else 0,
    )
    grade_percentage = st.slider(
        "Current Average Grade (%)",
        min_value=40,
        max_value=100,
        value=int(float(selected_profile.get("grade_percentage", 60))),
        step=1,
    )

    selected_interests = st.multiselect(
        "Select Interests",
        options=all_interest_options,
        default=convert_text_to_valid_defaults(selected_profile.get("interests", ""), all_interest_options),
    )
    custom_interests = st.text_input(
        "Add extra interests separated by semicolon",
        value="" if selected_sample != "Manual entry" else "AI agents; automation",
    )

    selected_skills = st.multiselect(
        "Select Current Skills",
        options=all_skill_options,
        default=convert_text_to_valid_defaults(selected_profile.get("skills", ""), all_skill_options),
    )
    custom_skills = st.text_input(
        "Add extra skills separated by semicolon",
        value="" if selected_sample != "Manual entry" else "Streamlit; communication",
    )

    selected_working_style = st.multiselect(
        "Preferred Working Style",
        options=all_style_options,
        default=convert_text_to_valid_defaults(selected_profile.get("working_style", ""), all_style_options),
    )
    career_goal = st.text_area("Career Goal", value=str(selected_profile.get("career_goal", "")), height=80)

    submitted = st.form_submit_button("Generate AI Career Guidance", use_container_width=True)


if submitted:
    # Combine selected and custom values into strings that the matching engine can read.
    interests_combined = "; ".join(selected_interests + split_semicolon_text(custom_interests))
    skills_combined = "; ".join(selected_skills + split_semicolon_text(custom_skills))
    working_style_combined = "; ".join(selected_working_style)

    student_profile = {
        "student_name": student_name.strip() or "Student",
        "programme": programme,
        "grade_percentage": grade_percentage,
        "interests": interests_combined,
        "skills": skills_combined,
        "working_style": working_style_combined,
        "career_goal": career_goal.strip(),
    }

    st.divider()
    st.subheader("Career Guidance Output")

    try:
        resolved_api_key = os.getenv("GROQ_API_KEY", "")

        with st.spinner("Running the career matching agents and generating guidance..."):
            result = generate_guidance(
                profile=student_profile,
                careers_df=careers_df,
                courses_df=courses_df,
                api_key=resolved_api_key,
                use_llm=bool(resolved_api_key),
            )

        ranked_df = result["ranked_careers"]
        top_courses = result["top_courses"]

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Best Match", ranked_df.iloc[0]["career_path"])
        with metric_col2:
            st.metric("Match Score", f"{ranked_df.iloc[0]['match_score']}%")
        with metric_col3:
            st.metric("Skill Gaps", len(split_semicolon_text(ranked_df.iloc[0].get("missing_skills", ""))))

        st.markdown("### Career Match Ranking")
        chart_df = ranked_df[["career_path", "match_score"]].set_index("career_path")
        st.bar_chart(chart_df, height=320)

        display_columns = [
            "rank",
            "career_path",
            "category",
            "match_score",
            "interest_score",
            "skill_score",
            "grade_score",
            "matched_skills",
            "missing_skills",
        ]
        st.dataframe(ranked_df[display_columns], use_container_width=True, hide_index=True)

        st.markdown("### Top Career Details")
        for _, row in ranked_df.head(3).iterrows():
            with st.expander(f"Rank {int(row['rank'])}: {row['career_path']} — {row['match_score']}% match", expanded=int(row["rank"]) == 1):
                st.write(row["description"])
                st.write(f"**Category:** {row['category']}")
                st.write(f"**Matched skills:** {row.get('matched_skills', '') or 'Limited skill match based on provided profile'}")
                st.write(f"**Skill gaps:** {row.get('missing_skills', '') or 'No major gaps identified'}")
                st.write(f"**Suggested entry roles:** {row['entry_roles']}")
                st.write(f"**3-year scenario:** {row['three_year_scenario']}")
                st.write(f"**5-year scenario:** {row['five_year_scenario']}")

                career_courses = recommend_courses_for_career(row, courses_df, max_courses=4)
                st.write("**Recommended courses for this path:**")
                st.dataframe(
                    career_courses[["course_name", "category", "difficulty", "estimated_weeks", "portfolio_output"]],
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("### Priority Course Plan for Best Match")
        st.dataframe(
            top_courses[["course_name", "category", "skills_gained", "difficulty", "estimated_weeks", "portfolio_output"]],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("### Personalised Guidance Report")
        if result["llm_success"]:
            st.success("Personalised LLM report generated successfully.")
        elif resolved_api_key:
            st.warning("LLM report was not generated. Showing rule-based fallback report instead.")
            st.caption(result.get("llm_error", "No error details available."))
        else:
            st.info("No API key was found in the environment. Showing rule-based guidance report.")

        st.markdown(result["final_report"])

        report_text = build_downloadable_text_report(student_profile, result)
        safe_name = student_profile["student_name"].replace(" ", "_").lower()
        st.download_button(
            "Download Personalised Career Report",
            data=report_text,
            file_name=f"{safe_name}_career_guidance_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    except Exception as error:
        st.error(f"The agent could not complete the analysis: {error}")
else:
    st.divider()
    st.info("Complete the student profile and click **Generate AI Career Guidance** to run the analysis.")


st.divider()
st.caption(
    "Career and course records are stored in local CSV files for reliable academic demonstration. "
    "The LLM layer personalises the final explanation, while the matching scores are produced by transparent rule-based agent logic."
)
