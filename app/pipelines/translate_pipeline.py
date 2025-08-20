from __future__ import annotations
import asyncio, re
from typing import Dict, List, Optional

from ..agents.router import detect_lang, decide_domain
from ..domain.taxonomy import STYLE_GUIDE
from ..stores.term_store import TermStore
from ..stores.rag_store import RAGStore
from ..stores.tm_store import TMStore
from ..agents.translator import translate_text
from ..agents.adequacy_reviewer import adequacy_review
from ..agents.fluency_reviewer import fluency_review
from ..agents.editor import edit_merge
from ..config import settings

# QA (reglas + LLM híbrido)
from ..qa.validators import (
    numeric_consistency,
    terminology_coverage,
    domain_alignment_score,
    anumeric_consistency,          # LLM numeric QA (async)
    adomain_alignment_score,       # LLM domain QA (async)
)

def split_segments(text: str, max_chars: int = 1400) -> List[str]:
    parts = re.split(r"\n\s*\n|\r\n\r\n", (text or "").strip())
    segs: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) <= max_chars:
            segs.append(p)
        else:
            sents = re.split(r"(?<=[\.\!\?])\s+", p)
            buf = ""
            for s in sents:
                if len(buf) + len(s) + 1 <= max_chars:
                    buf = (buf + " " + s).strip()
                else:
                    if buf:
                        segs.append(buf)
                    buf = s
            if buf:
                segs.append(buf)
    return segs

async def _rag_for(text: str, domain: str, client_id: str) -> List[str]:
    rag = RAGStore()
    await rag.load_seed_sources()
    snippets = await rag.retrieve(text[:2000], domain, client_id, topk=settings.rag_topk)
    if not snippets:
        # backfill web con acrónimos si el corpus está vacío
        acros = sorted(set(re.findall(r"\b[A-Z]{2,6}\b", text)))[:6]
        await rag.web_backfill_if_empty(acros or ["IRR", "NAV", "MiFID", "UCITS"], domain, client_id)
        snippets = await rag.retrieve(text[:2000], domain, client_id, topk=settings.rag_topk)
    return snippets

async def _tm_hint(client_id: str, src_lang: str, tgt_lang: str, segment: str) -> Optional[str]:
    tm = TMStore()
    hits = await tm.search(client_id, src_text=segment, src_lang=src_lang, tgt_lang=tgt_lang, topk=settings.tm_topk)
    if hits and hits[0][1] > 0.92:
        return hits[0][0]
    return None

async def run_pipeline(
    *,
    text: str,
    client_id: str,
    targets: Optional[List[str]],
    src_lang_override: Optional[str],
    domain_override: Optional[str],
    enable_rag: bool,
    save_tm: bool,
) -> Dict:
    # 1) Enrutado inicial
    src_lang = src_lang_override or detect_lang(text)
    if not targets:
        targets = [l for l in ["en", "fr", "de", "es"] if l != src_lang] or ["en", "fr", "de"]
    domain = domain_override or await decide_domain(text, hint=None)

    # 2) Contexto: DNT + glosarios + RAG
    ts = TermStore()
    dnt = await ts.dnt_list(client_id)
    gl_blocks: Dict[str, str] = {}
    gl_maps: Dict[str, Dict[str, str]] = {}
    for L in targets:
        block, m = await ts.glossary_block(client_id, L)
        gl_blocks[L] = block
        gl_maps[L] = m

    rag_snips = await _rag_for(text, domain, client_id) if enable_rag else []
    segs = split_segments(text)

    results: Dict[str, Dict] = {}

    async def process_lang(tgt_lang: str):
        translated_segs: List[str] = []
        for seg in segs:
            hint = await _tm_hint(client_id, src_lang, tgt_lang, seg)
            if hint:
                base = hint
            else:
                base = await translate_text(
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    domain=domain,
                    style=STYLE_GUIDE,
                    glossary_block=gl_blocks[tgt_lang],
                    dnt=dnt,
                    rag_snippets=rag_snips,
                    source=seg,
                )
            # reviewers en paralelo
            ade, flu = await asyncio.gather(
                adequacy_review(
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    domain=domain,
                    glossary_block=gl_blocks[tgt_lang],
                    source=seg,
                    current=base,
                ),
                fluency_review(tgt_lang=tgt_lang, domain=domain, current=base),
            )
            final_seg = await edit_merge(
                tgt_lang=tgt_lang,
                domain=domain,
                adequacy_text=(ade.revised or base),
                fluency_text=(flu.revised or base),
            )
            translated_segs.append(final_seg)
            if save_tm:
                await TMStore().upsert(client_id, src_lang, tgt_lang, domain, seg, final_seg)

        # 3) QA híbrido (reglas + LLM opcional)
        full = "\n\n".join(translated_segs)
        if settings.qa_use_llm:
            num = await anumeric_consistency(text, full)
            doms = await adomain_alignment_score(domain, full)
        else:
            num = numeric_consistency(text, full)
            doms = domain_alignment_score(domain, full)
        cov = terminology_coverage(full, gl_maps[tgt_lang])

        return {
            "final": full,
            "qa": {
                "numeric_consistency": float(num),
                "term_coverage": float(cov),
                "domain_score": float(doms),
            },
        }

    outs = await asyncio.gather(*[process_lang(L) for L in targets])
    for L, data in zip(targets, outs):
        results[L] = data

    return {
        "src_lang": src_lang,
        "domain": domain,
        "rag_used": bool(rag_snips),
        "results": results,
    }
