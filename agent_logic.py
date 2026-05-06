"""Core agent logic for the AI Career Guidance Agent.

The Streamlit interface stays in app.py, while this file contains the reusable
career matching, course recommendation and Groq LLM generation functions.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv


# Load local .env file when available. This allows the app to read GROQ_API_KEY
load_dotenv()


GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


# A small synonym dictionary improves matching when students use simple wording
# such as "hacking" instead of "ethical hacking".
SKILL_SYNONYMS = {
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "dl": "deep learning",
    "llm": "large language model",
    "llms": "large language model",
    "bi": "business intelligence",
    "dashboard": "dashboards",
    "dashboards": "dashboards",
    "hacking": "ethical hacking",
    "security": "cybersecurity",
    "coding": "programming",
    "programming": "programming",
    "cloud": "cloud computing",
    "devops": "devops",
    "db": "database",
    "database": "database",
    "data viz": "data visualisation",
    "visualization": "visualisation",
    "visualisation": "visualisation",
}


COMMON_STOP_WORDS = {
    "and", "or", "the", "a", "an", "to", "for", "of", "in", "on", "with",
    "using", "use", "tools", "basics", "fundamentals", "work", "career",
}


def load_knowledge_base(data_dir: str | Path = "data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all CSV files used by the agent.

    Returns career paths, course catalogue, sample profiles and skills taxonomy.
    The function validates required files so setup problems are easy to fix.
    """
    data_path = Path(data_dir)
    required_files = {
        "career_paths": data_path / "career_paths.csv",
        "course_catalog": data_path / "course_catalog.csv",
        "sample_profiles": data_path / "sample_student_profiles.csv",
        "skills_taxonomy": data_path / "skills_taxonomy.csv",
    }

    missing_files = [str(path) for path in required_files.values() if not path.exists()]
    if missing_files:
        missing_text = ", ".join(missing_files)
        raise FileNotFoundError(f"Missing required CSV file(s): {missing_text}")

    careers_df = pd.read_csv(required_files["career_paths"])
    courses_df = pd.read_csv(required_files["course_catalog"])
    profiles_df = pd.read_csv(required_files["sample_profiles"])
    skills_df = pd.read_csv(required_files["skills_taxonomy"])

    careers_df["minimum_grade"] = pd.to_numeric(careers_df["minimum_grade"], errors="coerce").fillna(50)
    return careers_df, courses_df, profiles_df, skills_df


def split_semicolon_text(value: str) -> List[str]:
    """Convert semicolon or comma separated text into a clean list."""
    if pd.isna(value) or value is None:
        return []
    pieces = re.split(r"[;,|]+", str(value))
    cleaned = [piece.strip() for piece in pieces if piece.strip()]
    return cleaned


def normalise_text(value: str) -> str:
    """Normalise text for fairer rule-based matching."""
    value = str(value).lower().strip()
    value = value.replace("/", " ").replace("-", " ")
    value = re.sub(r"[^a-z0-9+.#\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return SKILL_SYNONYMS.get(value, value)


def tokenise_text(value: str) -> set:
    """Create keyword tokens from any text field."""
    normalised = normalise_text(value)
    tokens = set()

    for phrase in split_semicolon_text(normalised):
        phrase = normalise_text(phrase)
        if phrase and phrase not in COMMON_STOP_WORDS:
            tokens.add(phrase)
        for word in phrase.split():
            if len(word) > 2 and word not in COMMON_STOP_WORDS:
                tokens.add(word)

    for word in normalised.split():
        if len(word) > 2 and word not in COMMON_STOP_WORDS:
            tokens.add(SKILL_SYNONYMS.get(word, word))

    return tokens


def profile_to_tokens(profile: Dict) -> set:
    """Create a searchable token set from student interests, skills and goals."""
    combined_profile_text = " ; ".join([
        str(profile.get("interests", "")),
        str(profile.get("skills", "")),
        str(profile.get("working_style", "")),
        str(profile.get("career_goal", "")),
        str(profile.get("programme", "")),
    ])
    return tokenise_text(combined_profile_text)


def calculate_overlap_score(profile_tokens: set, target_text: str, maximum_score: float) -> Tuple[float, List[str]]:
    """Calculate keyword overlap between a profile and a target field."""
    target_tokens = tokenise_text(target_text)
    if not target_tokens:
        return 0.0, []

    matches = sorted(profile_tokens.intersection(target_tokens))
    raw_ratio = len(matches) / max(len(target_tokens), 1)
    score = min(maximum_score, raw_ratio * maximum_score)
    return round(score, 2), matches


def calculate_grade_score(student_grade: float, minimum_grade: float) -> float:
    """Score how well the student's grade fits the suggested career path."""
    if student_grade >= minimum_grade + 10:
        return 20.0
    if student_grade >= minimum_grade:
        return 16.0 + min(4.0, (student_grade - minimum_grade) * 0.4)

    grade_gap = minimum_grade - student_grade
    return max(4.0, 16.0 - grade_gap * 0.9)


def identify_skill_gaps(profile: Dict, required_skills_text: str) -> Tuple[List[str], List[str]]:
    """Return matched and missing skills for one career row."""
    profile_tokens = profile_to_tokens(profile)
    required_skills = split_semicolon_text(required_skills_text)

    matched_skills = []
    missing_skills = []

    for skill in required_skills:
        skill_tokens = tokenise_text(skill)
        if profile_tokens.intersection(skill_tokens):
            matched_skills.append(skill)
        else:
            missing_skills.append(skill)

    return matched_skills, missing_skills


def match_career_paths(profile: Dict, careers_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Score each career path and return ranked recommendations."""
    profile_tokens = profile_to_tokens(profile)
    student_grade = float(profile.get("grade_percentage", 0))
    scored_rows = []

    for _, career in careers_df.iterrows():
        interest_score, interest_matches = calculate_overlap_score(
            profile_tokens,
            f"{career.get('preferred_interests', '')}; {career.get('description', '')}",
            30.0,
        )
        skill_score, skill_matches = calculate_overlap_score(
            profile_tokens,
            career.get("required_skills", ""),
            35.0,
        )
        style_score, style_matches = calculate_overlap_score(
            profile_tokens,
            career.get("working_style", ""),
            15.0,
        )
        grade_score = calculate_grade_score(student_grade, float(career.get("minimum_grade", 50)))
        matched_skills, missing_skills = identify_skill_gaps(profile, career.get("required_skills", ""))

        # The transparent component scores are normalised into a user-friendly percentage.
        total_score = (interest_score + skill_score + style_score + grade_score) * 1.25
        total_score = round(min(100.0, total_score), 2)

        scored_row = career.to_dict()
        scored_row.update({
            "match_score": total_score,
            "interest_score": round(interest_score, 2),
            "skill_score": round(skill_score, 2),
            "style_score": round(style_score, 2),
            "grade_score": round(grade_score, 2),
            "interest_matches": "; ".join(interest_matches[:8]),
            "skill_matches": "; ".join(skill_matches[:8]),
            "style_matches": "; ".join(style_matches[:5]),
            "matched_skills": "; ".join(matched_skills),
            "missing_skills": "; ".join(missing_skills[:8]),
            "grade_gap": max(0, round(float(career.get("minimum_grade", 50)) - student_grade, 2)),
        })
        scored_rows.append(scored_row)

    ranked_df = pd.DataFrame(scored_rows).sort_values(
        by=["match_score", "skill_score", "interest_score"],
        ascending=False,
    ).head(top_n).reset_index(drop=True)
    ranked_df.insert(0, "rank", ranked_df.index + 1)
    return ranked_df


def recommend_courses_for_career(career_row: pd.Series | Dict, courses_df: pd.DataFrame, max_courses: int = 6) -> pd.DataFrame:
    """Recommend courses by matching missing and required career skills."""
    missing_skill_text = str(career_row.get("missing_skills", ""))
    required_skill_text = str(career_row.get("required_skills", ""))
    recommended_course_text = str(career_row.get("recommended_courses", ""))
    career_tokens = tokenise_text(f"{missing_skill_text}; {required_skill_text}; {recommended_course_text}")

    course_scores = []
    for _, course in courses_df.iterrows():
        course_tokens = tokenise_text(
            f"{course.get('course_name', '')}; {course.get('category', '')}; "
            f"{course.get('skills_gained', '')}; {course.get('course_description', '')}"
        )
        overlap = sorted(career_tokens.intersection(course_tokens))
        direct_name_match = normalise_text(course.get("course_name", "")) in normalise_text(recommended_course_text)
        score = len(overlap) * 8 + (25 if direct_name_match else 0)
        if score > 0:
            row = course.to_dict()
            row["course_match_score"] = score
            row["matched_keywords"] = "; ".join(overlap[:8])
            course_scores.append(row)

    if not course_scores:
        return courses_df.head(max_courses).copy()

    recommended_df = pd.DataFrame(course_scores).sort_values(
        by=["course_match_score", "difficulty"],
        ascending=[False, True],
    ).head(max_courses).reset_index(drop=True)
    return recommended_df


def build_agent_summary(profile: Dict, ranked_df: pd.DataFrame, courses_df: pd.DataFrame) -> str:
    """Create a deterministic report even when no LLM key is available."""
    student_name = profile.get("student_name", "Student") or "Student"
    programme = profile.get("programme", "Not specified") or "Not specified"
    grade = profile.get("grade_percentage", "Not specified")
    interests = profile.get("interests", "Not specified") or "Not specified"
    skills = profile.get("skills", "Not specified") or "Not specified"
    career_goal = profile.get("career_goal", "Not specified") or "Not specified"

    lines = []
    lines.append(f"AI Career Guidance Report for {student_name}")
    lines.append("=" * 70)
    lines.append(f"Programme: {programme}")
    lines.append(f"Current grade: {grade}%")
    lines.append(f"Interests: {interests}")
    lines.append(f"Current skills: {skills}")
    lines.append(f"Career goal: {career_goal}")
    lines.append("")
    lines.append("Recommended Career Paths")
    lines.append("-" * 70)

    for _, row in ranked_df.iterrows():
        lines.append(f"{int(row['rank'])}. {row['career_path']} — Match Score: {row['match_score']}%")
        lines.append(f"   Category: {row['category']}")
        lines.append(f"   Why suitable: {row['description']}")
        lines.append(f"   Matched skills: {row.get('matched_skills', '') or 'Some interests matched but skill evidence is limited'}")
        lines.append(f"   Skill gaps: {row.get('missing_skills', '') or 'No major gaps identified from the provided profile'}")
        lines.append(f"   3-year scenario: {row['three_year_scenario']}")
        lines.append(f"   5-year scenario: {row['five_year_scenario']}")
        lines.append("")

    top_career = ranked_df.iloc[0]
    top_courses = recommend_courses_for_career(top_career, courses_df, max_courses=5)
    lines.append("Priority Learning Plan for the Top Recommendation")
    lines.append("-" * 70)
    for _, course in top_courses.iterrows():
        lines.append(
            f"- {course['course_name']} ({course['difficulty']}, {course['estimated_weeks']} weeks): "
            f"{course['portfolio_output']}"
        )

    lines.append("")
    lines.append("Suggested Next Steps")
    lines.append("-" * 70)
    lines.append("1. Select the strongest career path and complete two short courses linked to missing skills.")
    lines.append("2. Build one portfolio project that proves the required technical ability.")
    lines.append("3. Update the profile after new skills are gained and re-run the agent for fresh recommendations.")
    return "\n".join(lines)


def build_llm_prompt(profile: Dict, ranked_df: pd.DataFrame, courses_df: pd.DataFrame) -> str:
    """Prepare a compact but informative prompt for the Groq model."""
    recommendations = []
    for _, row in ranked_df.head(3).iterrows():
        course_df = recommend_courses_for_career(row, courses_df, max_courses=4)
        course_lines = [f"{course['course_name']} ({course['difficulty']}, {course['estimated_weeks']} weeks)" for _, course in course_df.iterrows()]
        recommendations.append({
            "career_path": row.get("career_path", ""),
            "match_score": row.get("match_score", ""),
            "category": row.get("category", ""),
            "description": row.get("description", ""),
            "matched_skills": row.get("matched_skills", ""),
            "missing_skills": row.get("missing_skills", ""),
            "recommended_courses": course_lines,
            "three_year_scenario": row.get("three_year_scenario", ""),
            "five_year_scenario": row.get("five_year_scenario", ""),
        })

    prompt = f"""
You are an AI Career Guidance Agent for MSc computing students.
Use the structured career matching output below to write a personalised career guidance report.
Do not invent grades, skills, certifications, or career paths that are not provided.
Use a professional, supportive and practical tone.

Student profile:
- Name: {profile.get('student_name', 'Student')}
- Programme: {profile.get('programme', 'Not specified')}
- Grade percentage: {profile.get('grade_percentage', 'Not specified')}
- Interests: {profile.get('interests', 'Not specified')}
- Skills: {profile.get('skills', 'Not specified')}
- Working style: {profile.get('working_style', 'Not specified')}
- Career goal: {profile.get('career_goal', 'Not specified')}

Career matching output:
{recommendations}

Write the response using these headings:
1. Profile Analysis
2. Top Career Recommendation
3. Alternative Career Options
4. Required Courses and Skill Development Plan
5. Three-Year Career Scenario
6. Five-Year Career Scenario
7. Practical Next Steps

Keep the response concise enough for a prototype demo, but detailed enough for a university presentation.
""".strip()
    return prompt


def call_groq_llm(prompt: str, api_key: str | None = None, model: str = DEFAULT_GROQ_MODEL, temperature: float = 0.4) -> Tuple[bool, str]:
    """Call Groq's OpenAI-compatible Chat Completions endpoint.

    Returns (success, message). The Streamlit app can show a rule-based fallback
    when success is False.
    """
    resolved_api_key = api_key or os.getenv("GROQ_API_KEY", "")
    if not resolved_api_key:
        return False, "No Groq API key found. Add GROQ_API_KEY in .env or environment variables."

    headers = {
        "Authorization": f"Bearer {resolved_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a careful career guidance assistant. Ground recommendations in the provided structured data only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_completion_tokens": 1200,
    }

    try:
        response = requests.post(GROQ_CHAT_COMPLETIONS_URL, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return True, content.strip()
    except requests.exceptions.HTTPError as error:
        status_code = getattr(error.response, "status_code", "unknown")
        response_text = getattr(error.response, "text", "")
        friendly_error = (
            f"Groq API request failed with status {status_code}. "
            f"Check that the API key has access to model '{model}'. Details: {response_text[:400]}"
        )
        return False, friendly_error
    except Exception as error:
        return False, f"Groq API request failed: {error}"


def generate_guidance(profile: Dict, careers_df: pd.DataFrame, courses_df: pd.DataFrame, api_key: str | None = None, use_llm: bool = True, model: str = DEFAULT_GROQ_MODEL) -> Dict:
    """Run the full agent workflow and return all outputs for the UI."""
    ranked_df = match_career_paths(profile, careers_df, top_n=5)
    top_career = ranked_df.iloc[0]
    top_courses = recommend_courses_for_career(top_career, courses_df, max_courses=6)
    rule_based_report = build_agent_summary(profile, ranked_df, courses_df)

    llm_success = False
    llm_report = ""
    llm_error = ""

    if use_llm:
        prompt = build_llm_prompt(profile, ranked_df, courses_df)
        llm_success, llm_message = call_groq_llm(prompt, api_key=api_key, model=model)
        if llm_success:
            llm_report = llm_message
        else:
            llm_error = llm_message

    final_report = llm_report if llm_success else rule_based_report

    return {
        "ranked_careers": ranked_df,
        "top_courses": top_courses,
        "rule_based_report": rule_based_report,
        "llm_success": llm_success,
        "llm_report": llm_report,
        "llm_error": llm_error,
        "final_report": final_report,
    }


def build_downloadable_text_report(profile: Dict, result: Dict) -> str:
    """Create a text report for the Streamlit download button."""
    report_type = "LLM-enhanced report" if result.get("llm_success") else "Rule-based fallback report"
    header = [
        "AI Career Guidance Agent - Personalised Report",
        "=" * 70,
        f"Report type: {report_type}",
        f"Student: {profile.get('student_name', 'Student')}",
        f"Programme: {profile.get('programme', 'Not specified')}",
        "",
    ]
    return "\n".join(header) + result.get("final_report", "")
