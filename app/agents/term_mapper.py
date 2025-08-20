from __future__ import annotations
import re
from typing import List, Set
from ..stores.rag_store import RAGStore
from ..config import settings
from .term_extractor import extract_terms_llm

# Regex: acrónimos y ProperCase >=4
CAND_RE = re.compile(r"\b([A-Z]{2,6}|[A-Z][a-zA-Z]{3,})\b")
STOP = {"and","or","the","for","with","from","into","over","under","between","without",
        "del","de","la","el","los","las","des","le","les","von","und","der","die","das"}

BOOST = {"IRR","NAV","TVPI","DPI","MOIC","FX","AIFMD","ELTIF","UCITS","MiFID","PRIIPs","KID"}

def _regex_candidates(text: str) -> List[str]:
    cands: Set[str] = set()
    for m in CAND_RE.finditer(text or ""):
        tok = (m.group(1) or "").strip()
        if tok and tok.lower() not in STOP:
            cands.add(tok)
    return sorted(cands | BOOST)

def _filter_unknown(terms: List[str], known_terms: List[str]) -> List[str]:
    known = set((k or "").lower() for k in known_terms if k)
    out: List[str] = []
    seen: Set[str] = set()
    for t in terms:
        k = t.lower().strip()
        if not k or k in seen: 
            continue
        if k in known: 
            continue
        seen.add(k)
        out.append(t)
    return out

async def find_candidates_hybrid(text: str, known_terms: List[str], domain: str) -> List[str]:
    regex_terms = _regex_candidates(text)
    llm_terms: List[str] = []
    if settings.llm_term_extractor_enabled:
        try:
            llm_terms = await extract_terms_llm(text, domain=domain)
        except Exception:
            llm_terms = []
    merged = regex_terms + llm_terms
    return _filter_unknown(merged, known_terms)[: settings.term_cand_topk]

async def enrich_terms(terms: List[str], domain: str, client_id: str | None) -> int:
    return await RAGStore().web_backfill(terms, domain, client_id)
