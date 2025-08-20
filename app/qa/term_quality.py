# app/qa/term_quality.py
from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field
from ..services.llm import llm_parse
from ..config import settings

class TermQuality(BaseModel):
    ok: bool
    confidence: float = Field(..., ge=0, le=1)
    reasons: List[str] = []

_PROMPT = """You are a financial term auditor.
Judge if TARGET is an appropriate preferred form for SOURCE (in SOURCE_LANG) when translated to TARGET_LANG.

Return JSON: {"ok": true/false, "confidence": 0..1, "reasons": ["..."]}

SOURCE: {source}
SOURCE_LANG: {src_lang}
TARGET_LANG: {tgt_lang}
TARGET: {target}
"""

async def judge_term_quality(source: str, src_lang: str, target_lang: str, target_string: str) -> float:
    out: TermQuality = await llm_parse(
        _PROMPT.format(source=source[:120], src_lang=src_lang, tgt_lang=target_lang, target=target_string[:200]),
        model=settings.qa_llm_model,
        schema=TermQuality,
        temperature=0.0,
    )
    return float(max(0.0, min(1.0, out.confidence)))
