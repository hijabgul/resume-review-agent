"""
resume_crew.py
----------------
This file defines the "brain" of the app: CrewAI agents that evaluate
resumes either against a target job description or as a standalone ATS audit.
"""

import json
import re
from typing import List

from crewai import Agent, Task, Crew, Process, LLM
from pydantic import BaseModel, Field, ValidationError

# ---------------------------------------------------------------------------
# 1. Structured Output Schemas
# ---------------------------------------------------------------------------

class ResumeEvaluation(BaseModel):
    """Schema for Job Description Match Evaluation."""
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


class StandaloneATSEvaluation(BaseModel):
    """Schema for Standalone ATS & Structural Quality Audit."""
    ats_score: int = Field(
        ..., ge=0, le=100,
        description="Overall ATS formatting and structural quality score from 0 to 100 based on standard industry practices."
    )
    overall_summary: str = Field(
        ..., description="Brief summary of the resume structure, strengths, and primary weaknesses."
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Key strengths found in the resume layout, content, or action-oriented writing."
    )
    formatting_issues: List[str] = Field(
        default_factory=list,
        description="Issues related to ATS readability (e.g., missing contact info, non-standard headers, complex layout, poor structure)."
    )
    impact_and_content_gaps: List[str] = Field(
        default_factory=list,
        description="Gaps in impact (e.g., lack of quantified metrics, weak action verbs, missing key professional sections)."
    )
    actionable_recommendations: List[str] = Field(
        default_factory=list,
        description="Specific steps the candidate should take to improve their overall ATS score and readability."
    )


# Model served on Groq's fast inference API (OpenAI's open-weight model).
GROQ_MODEL_ID = "groq/openai/gpt-oss-120b"


def build_llm(api_key: str) -> LLM:
    """Wire up the Groq-hosted model for CrewAI to use."""
    return LLM(
        model=GROQ_MODEL_ID,
        api_key=api_key,
        temperature=0.3,   # lower temperature = more consistent, less "creative"
        max_tokens=2000,
    )


# ---------------------------------------------------------------------------
# 2. Crew Construction for Job Match Review
# ---------------------------------------------------------------------------

def build_crew(api_key: str) -> Crew:
    """Assemble the single agent, its one task, and the crew that runs job match analysis."""
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
    )

    return Crew(
        agents=[analyst],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# 3. Crew Construction for Standalone ATS Review
# ---------------------------------------------------------------------------

def build_standalone_ats_crew(api_key: str) -> Crew:
    """Assemble the agent and task for evaluating a standalone resume without a JD."""
    llm = build_llm(api_key)

    ats_auditor = Agent(
        role="Senior ATS Compliance & Resume Auditor",
        goal=(
            "Perform a standalone evaluation of a candidate's resume for general "
            "ATS compatibility, structure, impact, action verbs, and formatting "
            "best practices without needing a specific job description."
        ),
        backstory=(
            "You are an expert ATS optimization specialist and professional resume "
            "writer. You have audited tens of thousands of resumes against major ATS "
            "parsing engines (Greenhouse, Lever, Workday, Taleo). You provide "
            "uncompromisingly honest, precise feedback on resume formatting, section headers, "
            "quantifiable impact, and phrasing."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    task = Task(
        description=(
            "You will be given a CANDIDATE RESUME below.\n\n"
            "=== CANDIDATE RESUME ===\n{resume_text}\n\n"
            "Do the following:\n"
            "1. Evaluate the overall ATS readiness, structure, readability, and impact of the resume.\n"
            "2. Assign ats_score (0-100) based on industry standards (clarity, action verbs, quantified results, contact info presentability, standard section headers).\n"
            "3. Summarize overall strengths and structural status in overall_summary (3-5 sentences).\n"
            "4. List strengths: key formatting, structural, or narrative highlights of this resume.\n"
            "5. List formatting_issues: potential ATS parsing red flags (missing sections, non-standard headers, unclear dates, contact detail issues).\n"
            "6. List impact_and_content_gaps: areas where achievements lack metrics/numbers, overused weak verbs, or missing summaries.\n"
            "7. List actionable_recommendations: concrete, practical steps to optimize the resume for any ATS scanner.\n\n"
            "OUTPUT FORMAT — this is critical:\n"
            "Respond with ONLY a single valid JSON object. No markdown code fences, "
            "no ```json, no commentary or explanation before or after it — just the "
            "raw JSON object, starting with { and ending with }. It must have exactly "
            "these keys:\n"
            '  "ats_score": integer from 0 to 100\n'
            '  "overall_summary": string, 3-5 sentences\n'
            '  "strengths": array of strings\n'
            '  "formatting_issues": array of strings\n'
            '  "impact_and_content_gaps": array of strings\n'
            '  "actionable_recommendations": array of strings\n'
        ),
        expected_output=(
            "A single raw JSON object (no markdown fences, no extra text) with the "
            "keys ats_score, overall_summary, strengths, formatting_issues, "
            "impact_and_content_gaps, and actionable_recommendations."
        ),
        agent=ats_auditor,
    )

    return Crew(
        agents=[ats_auditor],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# 4. Helper Functions & Runners
# ---------------------------------------------------------------------------

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
    Runs the job match crew once and returns a validated ResumeEvaluation object.
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


def run_standalone_ats_review(resume_text: str, api_key: str) -> StandaloneATSEvaluation:
    """
    Runs the standalone ATS audit crew once and returns a validated StandaloneATSEvaluation object.
    """
    crew = build_standalone_ats_crew(api_key)
    result = crew.kickoff(inputs={
        "resume_text": resume_text.strip(),
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
        return StandaloneATSEvaluation(**data)
    except ValidationError as exc:
        raise ValueError(
            f"The AI's response didn't match the expected format: {exc}"
        ) from exc
