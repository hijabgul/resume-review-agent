"""
app.py
------
Streamlit front-end for the Resume Review Agent.
"""

import time
import streamlit as st

from pdf_utils import extract_text_from_pdf, PDFExtractionError
from resume_crew import run_resume_review, ResumeEvaluation

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ResumeFit AI | Resume-to-Job Match Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background:
                linear-gradient(160deg, rgba(10,17,32,0.95) 0%, rgba(15,27,48,0.93) 45%, rgba(11,34,49,0.95) 100%),
                url('https://images.unsplash.com/photo-1497215728101-856f4ea42174?auto=format&fit=crop&w=1920&q=60');
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
        }

        h1, h2, h3, .hero-title {
            font-family: 'Poppins', sans-serif;
        }

        /* Hero header */
        .hero-wrap {
            padding: 2.4rem 2.2rem 2rem 2.2rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(20,184,166,0.16), rgba(59,130,246,0.10));
            border: 1px solid rgba(148,163,184,0.18);
            margin-bottom: 1.6rem;
            animation: fadeIn 0.6s ease-in-out;
        }
        .hero-title {
            font-size: 2.1rem;
            font-weight: 700;
            color: #FFFFFF !important;
            margin-bottom: 0.8rem;
        }
        .hero-subtitle {
            font-size: 1.02rem;
            color: #E5E7EB !important;
            max-width: 760px;
            line-height: 1.55;
        }
        .hero-badge {
            display: inline-block;
            padding: 0.28rem 0.75rem;
            border-radius: 999px;
            background: rgba(20,184,166,0.16);
            color: #2DD4BF !important;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            margin-bottom: 0.9rem;
            border: 1px solid rgba(45,212,191,0.35);
        }

        /* Section Headings & Label Visibility Fix */
        .section-header {
            color: #38BDF8 !important;
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Streamlit widget visibility overrides */
        .stRadio label, .stMarkdown, .stText, div[data-testid="stMarkdownContainer"] p {
            color: #F1F5F9 !important;
        }
        
        /* Textarea input contrast */
        textarea {
            background-color: rgba(15, 23, 42, 0.75) !important;
            color: #F8FAFC !important;
            border: 1px solid rgba(148, 163, 184, 0.3) !important;
            border-radius: 10px !important;
        }
        textarea:focus {
            border-color: #38BDF8 !important;
            box-shadow: 0 0 0 1px #38BDF8 !important;
        }

        /* Glass cards */
        .glass-card {
            background: rgba(15,23,42,0.65);
            border: 1px solid rgba(148,163,184,0.2);
            border-radius: 16px;
            padding: 1.5rem 1.6rem;
            backdrop-filter: blur(8px);
            transition: transform 0.25s ease, border-color 0.25s ease;
            margin-bottom: 1.1rem;
        }
        .glass-card:hover {
            transform: translateY(-2px);
            border-color: rgba(45,212,191,0.4);
        }
        .glass-card h4 {
            color: #38BDF8 !important;
            font-size: 1.1rem;
            margin-bottom: 0.8rem;
            font-weight: 600;
        }
        .glass-card ul {
            margin: 0;
            padding-left: 1.15rem;
        }
        .glass-card li {
            color: #E2E8F0 !important;
            margin-bottom: 0.45rem;
            line-height: 1.5;
            font-size: 0.94rem;
        }

        /* Score ring */
        .score-wrap {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 1.6rem 1rem;
            animation: popIn 0.5s ease-out;
        }
        .score-number {
            font-family: 'Poppins', sans-serif;
            font-size: 3.2rem;
            font-weight: 700;
            line-height: 1;
        }
        .score-label {
            color: #94A3B8 !important;
            font-size: 0.85rem;
            margin-top: 0.4rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .summary-box {
            background: rgba(30,41,59,0.7);
            border-left: 4px solid #2DD4BF;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            color: #F8FAFC !important;
            font-size: 0.98rem;
            line-height: 1.6;
            margin-bottom: 1rem;
        }

        /* Amber / Warning output overrides */
        .stAlert {
            background-color: rgba(30, 41, 59, 0.85) !important;
            color: #F8FAFC !important;
            border: 1px solid #F59E0B !important;
        }

        /* Buttons */
        div.stButton > button {
            background: linear-gradient(135deg, #14B8A6, #3B82F6);
            color: white !important;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            padding: 0.65rem 1.6rem;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            box-shadow: 0 4px 14px rgba(20,184,166,0.25);
        }
        div.stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(59,130,246,0.35);
        }

        .footer-note {
            text-align: center;
            color: #64748B !important;
            font-size: 0.8rem;
            margin-top: 2.5rem;
            padding-bottom: 1rem;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-6px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes popIn {
            from { opacity: 0; transform: scale(0.92); }
            to { opacity: 1; transform: scale(1); }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-wrap">
        <span class="hero-badge">AI RESUME ANALYST</span>
        <div class="hero-title">ResumeFit AI</div>
        <div class="hero-subtitle">
            Paste or upload a candidate's resume and a target job description.
            The agent compares them honestly — no invented skills — and gives
            you a match score plus specific, actionable improvements.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------
def get_api_key() -> str:
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        return ""

api_key = get_api_key()

if not api_key:
    st.error(
        "⚠️ No Groq API key found. Add `GROQ_API_KEY = \"your-key-here\"` to "
        "`.streamlit/secrets.toml` (locally) or to your app's Secrets in "
        "Streamlit Community Cloud settings, then rerun."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------
left_col, right_col = st.columns(2, gap="large")

with left_col:
    st.markdown('<div class="section-header">📄 Candidate Resume</div>', unsafe_allow_html=True)
    
    input_mode = st.radio(
        "How will you provide the resume?",
        ["Paste text", "Upload PDF"],
        horizontal=True,
        label_visibility="collapsed",
    )

    resume_text = ""
    if input_mode == "Paste text":
        resume_text = st.text_area(
            "Resume text",
            height=280,
            placeholder="Paste the full resume text here...",
           color=white,
        )
    else:
        uploaded_pdf = st.file_uploader("Upload resume PDF", type=["pdf"], label_visibility="collapsed")
        if uploaded_pdf is not None:
            try:
                resume_text = extract_text_from_pdf(uploaded_pdf)
                st.success(f"Extracted {len(resume_text.split())} words from the PDF.")
                with st.expander("Preview extracted text"):
                    st.text(resume_text[:2000] + ("..." if len(resume_text) > 2000 else ""))
            except PDFExtractionError as e:
                st.warning(str(e))

with right_col:
    st.markdown('<div class="section-header">🎯 Target Job Description</div>', unsafe_allow_html=True)
    job_description = st.text_area(
        "Job description",
        height=325,
        placeholder="Paste the full job description here...",
        label_visibility="collapsed",
    )

st.write("")
run_clicked = st.button("🔍 Analyze Match", use_container_width=False)

# ---------------------------------------------------------------------------
# Run the agent + display results
# ---------------------------------------------------------------------------
def score_color(score: int) -> str:
    if score >= 75:
        return "#2DD4BF"   # teal - strong match
    if score >= 50:
        return "#FBBF24"   # amber - partial match
    return "#F87171"       # red - weak match


def render_result(result: ResumeEvaluation) -> None:
    color = score_color(result.match_score)

    st.write("")
    score_col, summary_col = st.columns([1, 2.2], gap="large")

    with score_col:
        st.markdown(
            f"""
            <div class="glass-card score-wrap">
                <div class="score-number" style="color:{color};">{result.match_score}<span style="font-size:1.4rem;">/100</span></div>
                <div class="score-label">Overall Match Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with summary_col:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Summary")
        st.markdown(f'<div class="summary-box">{result.overall_summary}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        items = "".join(f"<li>{q}</li>" for q in result.matching_qualifications) or "<li>None identified.</li>"
        st.markdown(
            f'<div class="glass-card"><h4>✅ Matching Qualifications</h4><ul>{items}</ul></div>',
            unsafe_allow_html=True,
        )

        items = "".join(f"<li>{k}</li>" for k in result.ats_keywords_to_add) or "<li>None identified.</li>"
        st.markdown(
            f'<div class="glass-card"><h4>🔑 Keywords to Consider Adding</h4><ul>{items}</ul></div>',
            unsafe_allow_html=True,
        )

    with col_b:
        items = "".join(f"<li>{g}</li>" for g in result.missing_or_weak_areas) or "<li>None identified.</li>"
        st.markdown(
            f'<div class="glass-card"><h4>⚠️ Gaps vs. Job Description</h4><ul>{items}</ul></div>',
            unsafe_allow_html=True,
        )

        items = "".join(f"<li>{r}</li>" for r in result.recommendations) or "<li>None identified.</li>"
        st.markdown(
            f'<div class="glass-card"><h4>💡 Actionable Recommendations</h4><ul>{items}</ul></div>',
            unsafe_allow_html=True,
        )


if run_clicked:
    if not resume_text or not resume_text.strip():
        st.warning("Please paste or upload a resume first.")
    elif not job_description or not job_description.strip():
        st.warning("Please paste a job description first.")
    else:
        with st.spinner("Analyzing resume against job description..."):
            try:
                result = run_resume_review(resume_text, job_description, api_key)
                render_result(result)
            except Exception as e:
                err_text = str(e).lower()
                if "rate limit" in err_text or "429" in err_text:
                    st.error(
                        "🚦 The AI provider is rate-limiting requests right now. "
                        "Please wait about a minute and try again."
                    )
                elif "authentic" in err_text or "api key" in err_text or "401" in err_text or "403" in err_text:
                    st.error(
                        "🔑 The Groq API key was rejected. Double-check "
                        "`GROQ_API_KEY` in your Streamlit secrets."
                    )
                elif "timeout" in err_text:
                    st.error("⏱️ The request timed out. Please try again.")
                else:
                    st.error(f"Something went wrong while analyzing the resume: {e}")

st.markdown(
    '<div class="footer-note">ResumeFit AI · Powered by CrewAI + Groq · Built for learning purposes</div>',
    unsafe_allow_html=True,
)
