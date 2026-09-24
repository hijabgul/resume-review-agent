"""
resume_crew.py
--------------
Handles crew setup and execution with strict scoring rubrics, zero-temperature 
determinism, and robust JSON/control-character sanitizing for Groq/OpenAI models.
"""

import json
import re
from typing import List, Type
from pydantic import BaseModel, Field, ValidationError

from crewai import Agent, Task, Crew, Process, LLM


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class ResumeEvaluation(BaseModel):
    match_score: int = Field(description="Score out of 100 calculated using strict rubric weights.")
    overall_summary: str = Field(description="Executive summary of candidate alignment.")
    matching_qualifications: List[str] = Field(description="Key strengths matching job requirements.")
    missing_or_weak_areas: List[str] = Field(description="Gaps vs job requirements.")
    ats_keywords_to_add: List[str] = Field(description="Missing JD keywords.")
    recommendations: List[str] = Field(description="Actionable improvement steps.")

class StandaloneATSEvaluation(BaseModel):
    ats_score: int = Field(description="General ATS compatibility score out of 100 based on rubric.")
    overall_summary: str = Field(description="Summary of overall structural/content quality.")
    strengths: List[str] = Field(description="Key strengths of resume layout and content.")
    formatting_issues: List[str] = Field(description="Formatting or parsing risks detected.")
    impact_and_content_gaps: List[str] = Field(description="Weak bullet points or missing metrics.")
    actionable_recommendations: List[str] = Field(description="Concrete steps to improve ATS readiness.")


# ---------------------------------------------------------------------------
# Robust JSON Cleaner
# ---------------------------------------------------------------------------
def clean_and_parse_json(raw_output: str, target_class: Type[BaseModel]) -> BaseModel:
    """
    Cleans raw LLM text (removing markdown blocks and conversational preambles),
    handles unescaped control characters (newlines/tabs), and parses into Pydantic.
    """
    text = str(raw_output).strip()

    # Extract JSON string inside markdown code blocks if present
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        # Search for first { and last }
        brace_match = re.search(r'\{.*\}', text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)

    # Attempt parsing with strict=False to handle raw control characters like \n
    try:
        data = json.loads(text, strict=False)
        return target_class(**data)
    except (json.JSONDecodeError, ValidationError):
        # Fallback sanitization if unescaped literal linebreaks break JSON syntax
        try:
            cleaned_text = re.sub(r'[\r\n\t]+', ' ', text)
            data = json.loads(cleaned_text, strict=False)
            return target_class(**data)
        except Exception as e:
            raise ValueError(f"Failed to parse model output into structured JSON: {str(e)}")


# ---------------------------------------------------------------------------
# Review Execution Functions
# ---------------------------------------------------------------------------
def run_resume_review(resume_text: str, job_description: str, api_key: str) -> ResumeEvaluation:
    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.0
    )

    evaluator = Agent(
        role="Strict ATS & Technical Recruiter Auditor",
        goal="Evaluate candidate fit strictly using a standardized 100-point rubric.",
        backstory="An objective recruitment algorithm focused on repeatable, bias-free evaluations.",
        verbose=False,
        llm=llm
    )

    prompt = f"""
Evaluate the candidate's resume against the target job description using this STRICT SCORING RUBRIC:

[SCORING RUBRIC - TOTAL 100 POINTS]
1. Required Core Skills & Technologies: Up to 40 points
2. Relevant Work Experience & Achievements: Up to 30 points
3. ATS Keyword Alignment & Phrasing: Up to 15 points
4. Measurable Metrics & Action Verbs: Up to 15 points

Calculate the sum of all 4 categories to produce the final "match_score".

Target Job Description:
{job_description}

Candidate Resume:
{resume_text}

CRITICAL: Output ONLY a single valid raw JSON object matching this exact structure:
{{
  "match_score": 75,
  "overall_summary": "Summary text...",
  "matching_qualifications": ["strength 1", "strength 2"],
  "missing_or_weak_areas": ["gap 1", "gap 2"],
  "ats_keywords_to_add": ["keyword 1", "keyword 2"],
  "recommendations": ["recommendation 1", "recommendation 2"]
}}
Do NOT output preambles, notes, or markdown wrappers. Output JSON only.
"""

    task = Task(
        description=prompt,
        expected_output="Valid JSON matching the required schema.",
        agent=evaluator
    )

    crew = Crew(
        agents=[evaluator],
        tasks=[task],
        process=Process.sequential
    )

    result = crew.kickoff()
    raw_text = getattr(result, "raw", str(result))
    return clean_and_parse_json(raw_text, ResumeEvaluation)


def run_standalone_ats_review(resume_text: str, api_key: str) -> StandaloneATSEvaluation:
    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.0
    )

    evaluator = Agent(
        role="Senior ATS Architecture Auditor",
        goal="Audit candidate resumes strictly using a standardized structural compliance rubric.",
        backstory="An objective ATS audit system focused on deterministic document scoring.",
        verbose=False,
        llm=llm
    )

    prompt = f"""
Perform a standalone ATS audit on this resume using this STRICT SCORING RUBRIC:

[SCORING RUBRIC - TOTAL 100 POINTS]
1. Contact Info & Essential Section Structure: Up to 25 points
2. Technical & Professional Skills Clarity: Up to 25 points
3. Work Experience Detail & Measurable Metrics: Up to 25 points
4. ATS Readability & Clean Layout: Up to 25 points

Calculate the sum of all 4 categories to produce the final "ats_score".

Candidate Resume:
{resume_text}

CRITICAL: Output ONLY a single valid raw JSON object matching this exact structure:
{{
  "ats_score": 80,
  "overall_summary": "Audit summary text...",
  "strengths": ["strength 1", "strength 2"],
  "formatting_issues": ["issue 1", "issue 2"],
  "impact_and_content_gaps": ["gap 1", "gap 2"],
  "actionable_recommendations": ["recommendation 1", "recommendation 2"]
}}
Do NOT output preambles, notes, or markdown wrappers. Output JSON only.
"""

    task = Task(
        description=prompt,
        expected_output="Valid JSON matching the required schema.",
        agent=evaluator
    )

    crew = Crew(
        agents=[evaluator],
        tasks=[task],
        process=Process.sequential
    )

    result = crew.kickoff()
    raw_text = getattr(result, "raw", str(result))
    return clean_and_parse_json(raw_text, StandaloneATSEvaluation)
