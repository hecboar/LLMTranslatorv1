from __future__ import annotations
import langid
from ..domain.taxonomy import normalize_domain, DOMAINS
from ..services.llm import llm_text
from ..config import settings

DOMAIN_PROMPT = lambda txt: (
    "Classify the PRIMARY financial domain of the text into one of: "
    f"{', '.join(DOMAINS)}. Answer with the label ONLY.\n\nTEXT:\n{txt[:2000]}"
)

def detect_lang(text: str) -> str:
    try:
        lang, _ = langid.classify(text)
        if lang in {"en","es","fr","de"}:
            return lang
    except Exception:
        pass
    # fallback simple
    return "es"

async def decide_domain(text: str, hint: str | None) -> str:
    if hint:
        dom = normalize_domain(hint)
        if dom:
            return dom
    out = await llm_text(DOMAIN_PROMPT(text), model=settings.model_classify, temperature=0.0)
    clean = normalize_domain(out.strip())
    return clean or "Wealth Management"
