from __future__ import annotations
import re, asyncio
from typing import List
from duckduckgo_search import DDGS
import trafilatura
from ..stores.rag_store import RAGStore

TERM_RE = re.compile(r"\b([A-Z]{2,6}|[A-Z][a-z]{3,})\b")

def find_candidates(text: str, known_terms: List[str]) -> List[str]:
    known_low = set(t.lower() for t in known_terms if t)
    cands = set(m.group(1) for m in TERM_RE.finditer(text or ""))
    cands = {c for c in cands if c.lower() not in known_low}
    return sorted(cands)

async def enrich_terms(terms: List[str], domain: str, client_id: str | None) -> int:
    # Busca definiciones y documentos para esos términos y los ingesta al RAG
    urls = []
    try:
        with DDGS() as ddgs:
            for t in terms:
                q = f"{t} {domain} finance definition"
                res = ddgs.text(q, max_results=3)
                for r in res:
                    href = r.get("href")
                    if href:
                        urls.append(href)
    except Exception:
        return 0
    if not urls:
        return 0
    return await RAGStore().ingest_urls(urls, domain=domain, client_id=client_id)
