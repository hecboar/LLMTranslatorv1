from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from ..services.llm import llm_parse
from ..config import settings

# ---------------------------------------------------------------------------
# Output Models
# ---------------------------------------------------------------------------

class TermJudgeOut(BaseModel):
    term: str
    relevant: bool = Field(..., description="True if the term is relevant in the domain")
    translations: Optional[List[str]] = Field(default=None, description="Candidate translations if available")
    reason: Optional[str] = None
    
class TermJudgeList(BaseModel):
    """Wrapper needed because OpenAI client does not support List[T] schemas directly."""
    results: List[TermJudgeOut]

# ---------------------------------------------------------------------------
# LLM Prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """Role: Domain Terminology Judge.
You receive:
- DOMAIN context: {domain}
- SOURCE text: {text}
- Candidate TERMS: {terms}

Task:
Evaluate each candidate term. Decide if it is a domain-relevant concept that should be added to the glossary.
Reject overly generic or irrelevant words.

Return JSON:
{{
  "results": [
    {{"term": "AIFMD", "valid": true, "reason": "EU directive regulating fund managers"}},
    {{"term": "Despite", "valid": false, "reason": "Generic English word"}}
  ]
}}
"""

# ---------------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------------

async def judge_terms(domain: str, text: str, terms: List[str]) -> List[TermJudgeOut]:
    """
    Uses an LLM to evaluate whether candidate terms are relevant
    for the given financial domain. Returns a list of judgments.
    """
    if not terms:
        return []

    prompt = JUDGE_PROMPT.format(
        domain=domain,
        text=text[:3000],
        terms=", ".join(terms)
    )

    out: TermJudgeList = await llm_parse(
        prompt,
        model=getattr(settings, "qa_llm_model", settings.model_review),
        schema=TermJudgeList,
    )
    return out.results
