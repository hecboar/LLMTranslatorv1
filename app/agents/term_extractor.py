# app/agents/term_extractor.py
from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field
from ..services.llm import llm_parse
from ..config import settings

class TermsOut(BaseModel):
    terms: List[str] = Field(default_factory=list)

def _make_prompt(domain: str, src: str, topk: int) -> str:
    return f"""You are a financial terminology spotter.
Goal: Extract domain-relevant KEY TERMS from the SOURCE. Prefer acronyms, ratios, regulatory names, metrics, named instruments, and idioms.

Guidelines:
- Extract **as many terms as necessary** to cover the text; **if there are too many, keep only the most salient up to {topk}**.
- Include acronyms (keep original casing) and multi-word expressions (e.g., "withholding tax", "capital call").
- Exclude generic words, stopwords, isolated numbers/dates/currencies (unless part of a named metric like "CPI", "IRR  net", etc.).
- Deduplicate; keep the canonical surface form seen in the source (preserve hyphens/slashes and casing).
- Terms should be between 2 and 80 characters.
- Output **JSON only** in the form: {{"terms": ["..."]}} — no explanations.

Expected domain: {domain}
SOURCE:
{src}"""

async def extract_terms_llm(text: str, domain: str | None = None) -> List[str]:
    dom = domain or "Finance"
    topk = max(1, int(getattr(settings, "term_cand_topk", 12)))
    prompt = _make_prompt(dom, text[:6000], topk)
    out = await llm_parse(prompt, model=settings.model_review, schema=TermsOut, temperature=0.0)

    seen, final = set(), []
    for t in out.terms:
        t = (t or "").strip()
        if not t or len(t) < getattr(settings, "term_min_len", 2):
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        final.append(t)

    # recorte defensivo por si el modelo devuelve más de topk
    return final[:topk]
