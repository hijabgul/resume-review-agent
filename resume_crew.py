"""
resume_crew.py
--------------
Handles crew setup and execution with robust JSON parsing and control-character 
sanitizing for Groq and OpenAI models.
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
    match_score: int = Field(description="Score out of 100 representing job fit match.")
    overall_summary: str = Field(description="Executive summary of the candidate's alignment.")
    matching_qualifications: List[str] = Field(description="Key strengths matching the job requirements.")
    missing_or_weak_areas: List[str] = Field(description="Gaps vs job requirements.")
    ats_keywords_to_add: List[str] = Field(description="Keywords present in JD but missing in resume.")
    recommendations: List[str] = Field(description="Actionable steps to improve the resume.")

class StandaloneATSEvaluation(BaseModel):
    ats_score: int = Field(description="General ATS compatibility score out of 100.")
    overall_summary: str = Field(description="Summary of overall structural and content quality.")
    strengths: List[str] = Field(description="Key strengths of the resume layout/content.")
    formatting_issues: List[str] = Field(description="Formatting or ATS parser risks detected.")
    impact_and_content_gaps: List[str] = Field(description="Areas where bullet points or metrics are weak.")
    actionable_recommendations: List[str] = Field(description="Concrete recommendations for improvement.")


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
    except (json.JSONDecodeError, ValidationError) as e:
        # Fallback sanitization if unescaped literal linebreaks break JSON syntax
        try:
            cleaned_text = re.sub(r'[\r\n\t]+', ' ', text)
            data = json.loads(cleaned_text, strict=False)
            return target_class(**data)
        except Exception:
            raise ValueError(f"The AI's response wasn't valid JSON ({str(e)}). Raw response: {text[:200]}...")


# ---------------------------------------------------------------------------
# Review Execution Functions
# ---------------------------------------------------------------------------
def run_resume_review(resume_text: str, job_description: str, api_key: str) -> ResumeEvaluation:
    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0.2
    )

    evaluator = Agent(
        role="Senior Technical Recruiter & ATS Specialist",
        goal="Provide exact, structured JSON evaluations comparing candidate resumes to job descriptions.",
        backstory="An expert recruiter skilled in matching talent to tech roles and ATS optimization.",
        verbose=False,
        llm=llm
    )

    prompt = f"""
Analyze the candidate's resume against the target job description.

Job Description:
{job_description}

Candidate Resume:
{resume_text}

CRITICAL: Output ONLY a single raw JSON object matching this exact structure:
{{
  "match_score": 75,
  "overall_summary": "Summary text...",
  "matching_qualifications": ["strength 1", "strength 2"],
  "missing_or_weak_areas": ["gap 1", "gap 2"],
  "ats_keywords_to_add": ["keyword 1", "keyword 2"],
  "recommendations": ["recommendation 1", "recommendation 2"]
}}
Do NOT wrap in extra prose. Respond only with the JSON object.
"""

    task = Task(
        description=prompt,
        expected_output="Valid JSON matching the required structure.",
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
        temperature=0.2
    )

    evaluator = Agent(
        role="ATS Optimization Specialist & Resume Auditor",
        goal="Audit candidate resumes for structural ATS compliance, formatting, and content strength.",
        backstory="An expert ATS engineer who audits resume structure and readability.",
        verbose=False,
        llm=llm
    )

    prompt = f"""
Perform a comprehensive standalone structural and content ATS audit on this resume:

Candidate Resume:
{resume_text}

CRITICAL: Output ONLY a single raw JSON object matching this exact structure:
{{
  "ats_score": 80,
  "overall_summary": "Audit summary text...",
  "strengths": ["strength 1", "strength 2"],
  "formatting_issues": ["issue 1", "issue 2"],
  "impact_and_content_gaps": ["gap 1", "gap 2"],
  "actionable_recommendations": ["recommendation 1", "recommendation 2"]
}}
Do NOT wrap in extra prose. Respond only with the JSON object.
"""

    task = Task(
        description=prompt,
        expected_output="Valid JSON matching the required structure.",
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
