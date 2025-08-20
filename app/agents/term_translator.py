# app/agents/term_translator.py
from __future__ import annotations
from pydantic import BaseModel, Field
from ..services.llm import llm_parse
from ..config import settings

class TermProposal(BaseModel):
    term: str
    src_lang: str
    tgt_lang: str
    proposal: str = Field(..., description="One preferred, publication-ready target form")

_PROMPT = """You are a financial terminology adapter.
Given a SOURCE term in {src} and a TARGET language {tgt}, output ONE preferred target-language form.

Rules:
- If an established acronym exists in {tgt}, prefer 'Long form (ACRONYM)' for first mention.
- Use industry-standard casing/diacritics. No extra notes.
- Return JSON only: {{"term":"{term}", "src_lang":"{src}", "tgt_lang":"{tgt}", "proposal":"..."}}
"""

async def propose_term(term: str, src_lang: str, tgt_lang: str) -> str:
    out: TermProposal = await llm_parse(
        _PROMPT.format(term=term[:120], src=src_lang, tgt=tgt_lang),
        model=settings.model_review,
        schema=TermProposal,
        temperature=0.0,
    )
    return out.proposal.strip()
