from __future__ import annotations
from pydantic import BaseModel
from ..services.llm import llm_parse
from ..config import settings

class DomainJudgeOut(BaseModel):
    domain: str = ""
    score: float = 0.0  # 0..1
    reasons: str = ""

_PROMPT = """You are a financial domain classifier and scorer.
Given the TEXT and the EXPECTED_DOMAIN, assess how well the text aligns with that domain using industry-specific signals (metrics, ratios, regulations, jargon).
Return JSON: {{"domain":"<your best domain label>","score": <0..1>,"reasons":"..."}}

EXPECTED_DOMAIN: {expected}
TEXT:
{src}
"""

async def llm_domain_score(text: str, expected: str) -> float:
    if not settings.llm_validators_enabled:
        return 0.0
    out = await llm_parse(
        _PROMPT.format(expected=expected, src=text[:4000]),
        model=settings.qa_llm_model,
        schema=DomainJudgeOut,
        temperature=0.0
    )
    s = max(0.0, min(1.0, out.score))
    return float(s)
