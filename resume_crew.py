"""
resume_crew.py
----------------
This file defines the "brain" of the app: a single CrewAI agent that
compares a resume against a job description and returns a structured,
honest evaluation.

Beginner note: you do NOT need to edit this file to run the app.
Everything a student needs to change (API key, model name) lives in
Streamlit secrets, not here.
"""

import json
import re
from typing import List

from crewai import Agent, Task, Crew, Process, LLM
from pydantic import BaseModel, Field, ValidationError

# ---------------------------------------------------------------------------
# 1. The shape of the answer we want back from the AI (structured output).
#    Using a schema like this means we ALWAYS get clean, predictable fields
#    to show in the UI, instead of parsing free-form paragraphs.
# ---------------------------------------------------------------------------
class ResumeEvaluation(BaseModel):
    match_score: int = Field(
        ..., ge=0, le=100,
        description="Overall match between the resume and the job description, 0-100."
    )
    overall_summary: str = Field(
        ..., description="A short, honest 3-5 sentence summary of the fit."
    )
    matching_qualifications: List[str] = Field(
        default_factory=list,
        description="Concrete qualifications FROM THE RESUME that satisfy the job description."
    )
    missing_or_weak_areas: List[str] = Field(
        default_factory=list,
        description="Job requirements that the resume does not clearly demonstrate."
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Specific, actionable edits the candidate can make to their resume."
    )
    ats_keywords_to_add: List[str] = Field(
        default_factory=list,
        description="Important keywords from the job description missing from the resume."
    )


# Model served on Groq's fast inference API (OpenAI's open-weight model).
# CrewAI routes this through LiteLLM using the "groq/" provider prefix.
GROQ_MODEL_ID = "groq/openai/gpt-oss-120b"


def build_llm(api_key: str) -> LLM:
    """Wire up the Groq-hosted model for CrewAI to use."""
    return LLM(
        model=GROQ_MODEL_ID,
        api_key=api_key,
        temperature=0.3,   # lower temperature = more consistent, less "creative"
        max_tokens=2000,
    )


def build_crew(api_key: str) -> Crew:
    """Assemble the single agent, its one task, and the crew that runs it."""
    llm = build_llm(api_key)

    analyst = Agent(
        role="Senior Resume & ATS Matching Analyst",
        goal=(
            "Objectively evaluate how well a candidate's resume matches a specific "
            "job description, using ONLY information present in the resume. Never "
            "invent, assume, or exaggerate skills, tools, or experience the candidate "
            "did not actually mention."
        ),
        backstory=(
            "You are a former technical recruiter with 12 years of experience "
            "screening thousands of resumes for tech and business roles. You are "
            "known for being fair, precise, and honest. You never flatter candidates "
            "and you never fabricate qualifications that are not written in the resume."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description=(
            "You will be given a CANDIDATE RESUME and a TARGET JOB DESCRIPTION below.\n\n"
            "=== CANDIDATE RESUME ===\n{resume_text}\n\n"
            "=== TARGET JOB DESCRIPTION ===\n{job_description}\n\n"
            "Do the following:\n"
            "1. Compare the resume against the job description's required and "
            "preferred skills, tools, qualifications, and experience.\n"
            "2. Assign match_score (0-100) representing overall fit for this specific role.\n"
            "3. List matching_qualifications: concrete evidence IN THE RESUME that "
            "satisfies the job description.\n"
            "4. List missing_or_weak_areas: job requirements the resume does not "
            "clearly demonstrate.\n"
            "5. List recommendations: specific, actionable edits the candidate can "
            "make (rewording, adding measurable results, reordering sections, "
            "clarifying titles, etc.) to better present themselves for THIS job. "
            "Never suggest inventing or lying about a qualification — only suggest "
            "better ways to present what is already true in the resume.\n"
            "6. List ats_keywords_to_add: important keywords/phrases from the job "
            "description that are missing from the resume and could reasonably be "
            "added because the candidate's real experience supports them.\n\n"
            "STRICT RULE: Never state or imply the candidate has a skill, tool, "
            "degree, certification, or years of experience that is not explicitly "
            "stated or clearly implied in the resume text. When in doubt, treat it "
            "as missing rather than assumed.\n\n"
            "OUTPUT FORMAT — this is critical:\n"
            "Respond with ONLY a single valid JSON object. No markdown code fences, "
            "no ```json, no commentary or explanation before or after it — just the "
            "raw JSON object, starting with { and ending with }. It must have exactly "
            "these keys:\n"
            '  "match_score": integer from 0 to 100\n'
            '  "overall_summary": string, 3-5 sentences\n'
            '  "matching_qualifications": array of strings\n'
            '  "missing_or_weak_areas": array of strings\n'
            '  "recommendations": array of strings\n'
            '  "ats_keywords_to_add": array of strings\n'
        ),
        expected_output=(
            "A single raw JSON object (no markdown fences, no extra text) with the "
            "keys match_score, overall_summary, matching_qualifications, "
            "missing_or_weak_areas, recommendations, and ats_keywords_to_add."
        ),
        agent=analyst,
        # Note: intentionally NOT using output_pydantic here. CrewAI's
        # output_pydantic uses function/tool-calling to force structured
        # output, and some Groq-hosted models (including gpt-oss-120b at
        # the time of writing) throw "Tool choice is none, but model
        # called a tool" errors with that path. Prompting for raw JSON
        # and parsing it ourselves (see _extract_json / run_resume_review
        # below) avoids that bug entirely and is more robust.
    )

    return Crew(
        agents=[analyst],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )


def _extract_json(raw_text: str) -> dict:
    """
    Pulls a JSON object out of the model's raw text reply, even if it
    wrapped the JSON in ```json fences or added stray text around it.
    """
    text = raw_text.strip()

    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]

    return json.loads(text)


def run_resume_review(resume_text: str, job_description: str, api_key: str) -> ResumeEvaluation:
    """
    Runs the crew once and returns a validated ResumeEvaluation object.
    Raises whatever exception CrewAI/LiteLLM raises on failure — the
    calling Streamlit code is responsible for catching and displaying
    a friendly error message.
    """
    crew = build_crew(api_key)
    result = crew.kickoff(inputs={
        "resume_text": resume_text.strip(),
        "job_description": job_description.strip(),
    })

    raw_text = getattr(result, "raw", None) or str(result)

    try:
        data = _extract_json(raw_text)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            "The AI's response wasn't valid JSON, so it couldn't be read. "
            "Please try again."
        ) from exc

    try:
        return ResumeEvaluation(**data)
    except ValidationError as exc:
        raise ValueError(
            f"The AI's response didn't match the expected format: {exc}"
        ) from exc
