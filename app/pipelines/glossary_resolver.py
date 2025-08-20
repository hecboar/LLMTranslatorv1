# app/pipelines/glossary_resolver.py
from __future__ import annotations
import logging, asyncio
from typing import Dict, List, Optional
from ..agents.term_mapper import find_candidates_hybrid
from ..agents.concept_canonicalizer import to_canonical
from ..agents.term_translator import propose_term
from ..qa.term_quality import judge_term_quality
from ..stores.term_store import TermStore, GlossaryItem
from ..stores.rag_store import RAGStore
from ..config import settings

log = logging.getLogger(__name__)
ALL_LANGS = ("en", "es", "fr", "de")

async def _ensure_lang_pref(
    ts: TermStore,
    rag: RAGStore,
    *,
    client_id: str,
    domain: str,
    concept_key: str,
    source_surface: str,
    src_lang: str,
    tgt_lang: str,
    enable_rag: bool
) -> str:
    # 1) lookup in store (fuzzy on ck/variants)
    existing = await ts.find_preferred_fuzzy(client_id, domain, tgt_lang, concept_key)
    if existing:
        return existing

    # 2) propose (src→tgt), QA
    proposal = await propose_term(source_surface, src_lang, tgt_lang)
    conf = await judge_term_quality(source_surface, src_lang, tgt_lang, proposal)

    # 3) if weak, enrich with RAG and retry once
    if conf < getattr(settings, "term_quality_min", 0.75) and enable_rag:
        try:
            await rag.web_backfill([source_surface, concept_key], domain=domain, client_id=client_id)
        except Exception:
            pass
        proposal = await propose_term(source_surface, src_lang, tgt_lang)
        conf = await judge_term_quality(source_surface, src_lang, tgt_lang, proposal)

    # 4) persist
    await ts.upsert_preferred(GlossaryItem(
        client_id=client_id,
        domain=domain,
        concept_key=concept_key,
        lang=tgt_lang,
        preferred=proposal
    ))
    return proposal

async def resolve_glossary_for_targets(
    *,
    source_text: str,
    src_lang: str,
    targets: List[str],
    client_id: str,
    domain: str,
    enable_rag: bool,
    fill_all_four_langs: bool = True
) -> Dict[str, str]:
    """
    Returns {tgt_lang: glossary_block_string}. Internally it:
      - extracts candidate terms (language-agnostic)
      - canonicalizes to a single concept_key across languages
      - ensures preferred in ALL 4 languages (en/es/fr/de) so the glossary is cross-linked
      - returns a per-target block forcing the translator to use those forms
    """
    ts = TermStore()
    rag = RAGStore()

    # discover candidates in the *source language*
    cands = await find_candidates_hybrid(source_text, known_terms=[], domain=domain)
    if settings.trace_prompts:
        log.info("term.candidates", extra={"cands": cands, "src_lang": src_lang})

    # Build worklist: concept_key ↔ source_surface seen
    concepts: Dict[str, str] = {}
    for t in cands:
        ck, _ = to_canonical(t)
        concepts[ck] = concepts.get(ck, t)  # keep the first surface we saw

    # Ensure entries for every concept across the 4 languages
    langs_to_fill = set(ALL_LANGS) if fill_all_four_langs else set(targets) | {src_lang}
    results_per_lang: Dict[str, Dict[str, str]] = {L: {} for L in ALL_LANGS}

    sem = asyncio.Semaphore(6)

    async def fill_for_lang(ck: str, surface_src: str, L: str):
        async with sem:
            pref = await _ensure_lang_pref(
                ts, rag,
                client_id=client_id, domain=domain,
                concept_key=ck, source_surface=surface_src,
                src_lang=src_lang, tgt_lang=L,
                enable_rag=enable_rag
            )
            results_per_lang[L][ck] = pref

    await asyncio.gather(*[
        fill_for_lang(ck, surface, L)
        for ck, surface in concepts.items()
        for L in langs_to_fill
    ])

    # Persisting is already done; now build the per-target glossary blocks
    blocks: Dict[str, str] = {}
    for tgt in targets:
        pairs = [f"- {ck}: {results_per_lang[tgt].get(ck, '')}"
                 for ck in sorted(concepts.keys()) if results_per_lang[tgt].get(ck)]
        blocks[tgt] = "\n".join(pairs)
        if settings.trace_prompts:
            log.info("glossary.block", extra={"lang": tgt, "block": blocks[tgt]})

    return blocks
