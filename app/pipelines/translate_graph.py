from __future__ import annotations
from typing import Dict, List, Optional, Any
import re, asyncio, hashlib
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from ..agents.router import detect_lang, decide_domain
from ..domain.taxonomy import STYLE_GUIDE, normalize_domain
from ..stores.term_store import TermStore, GlossaryItem
from ..stores.rag_store import RAGStore
from ..stores.tm_store import TMStore
from ..agents.translator import translate_text
from ..agents.adequacy_reviewer import adequacy_review
from ..agents.fluency_reviewer import fluency_review
from ..agents.editor import edit_merge
from ..agents.term_mapper import find_candidates_hybrid, enrich_terms
from ..agents.term_validator import judge_terms
from ..qa.validators import numeric_consistency, terminology_coverage
from ..qa.validators import domain_alignment_score as domain_alignment_score_async
from ..utils.textguards import mask, unmask
from ..config import settings
from ..telemetry import trace

def split_segments(text: str, max_chars: int = 1400) -> List[str]:
    parts = re.split(r"\n\s*\n|\r\n\r\n", text.strip())
    segs: List[str] = []
    for p in parts:
        p = p.strip()
        if not p: continue
        if len(p) <= max_chars:
            segs.append(p)
        else:
            sents = re.split(r"(?<=[\.\!\?])\s+", p)
            buf = ""
            for s in sents:
                if len(buf) + len(s) + 1 <= max_chars:
                    buf = (buf + " " + s).strip()
                else:
                    if buf: segs.append(buf)
                    buf = s
            if buf: segs.append(buf)
    return segs

class GState(TypedDict, total=False):
    text: str
    client_id: str
    targets: Optional[List[str]]
    src_lang: Optional[str]
    domain: Optional[str]
    enable_rag: bool
    save_tm: bool
    debug: bool

    # computed
    src_lang_r: str
    targets_r: List[str]
    domain_r: str
    dnt: List[str]
    gl_blocks: Dict[str, str]
    gl_maps: Dict[str, Dict[str, str]]
    rag_snips: List[str]
    masked_text: str
    mask_map: Dict[str, str]
    unknown_terms: List[str]
    results: Dict[str, Any]

async def n_detect_and_prepare(state: GState) -> GState:
    src = state.get("src_lang") or detect_lang(state["text"])
    targets = state.get("targets") or [l for l in ["en","fr","de","es"] if l != src] or ["en","fr","de"]
    ts = TermStore()
    dnt = await ts.dnt_list(state["client_id"])
    masked_text, mask_map = mask(state["text"], dnt)
    trace.log("detect", src_lang=src, targets=targets, dnt=len(dnt))
    return {**state, "src_lang_r": src, "targets_r": targets, "dnt": dnt,
            "masked_text": masked_text, "mask_map": mask_map}

async def n_decide_domain(state: GState) -> GState:
    dom = normalize_domain(state.get("domain") or await decide_domain(state["masked_text"], hint=None))
    trace.log("domain", decided=dom)
    return {**state, "domain_r": dom}

async def n_build_context(state: GState) -> GState:
    ts = TermStore()
    client_id = state["client_id"]
    gl_blocks: Dict[str, str] = {}
    gl_maps: Dict[str, Dict[str, str]] = {}
    for L in state["targets_r"]:
        block, m = await ts.glossary_block(client_id, state["domain_r"], L)
        gl_blocks[L] = block
        gl_maps[L] = m
    rag_snips: List[str] = []
    if state.get("enable_rag", True):
        await RAGStore().load_seed_sources()
        rag_snips = await RAGStore().retrieve(state["masked_text"][:2000], state["domain_r"], client_id, topk=settings.rag_topk)
    # candidatos desconocidos: regex + LLM
    known = []
    for m in gl_maps.values():
        known.extend(m.values())
    unknown = await find_candidates_hybrid(state["masked_text"], known_terms=known, domain=state["domain_r"])
    trace.log("context", glossary_terms=sum(len(m) for m in gl_maps.values()),
              rag=len(rag_snips), unknown_terms=unknown)
    return {**state, "gl_blocks": gl_blocks, "gl_maps": gl_maps, "rag_snips": rag_snips, "unknown_terms": unknown}

async def n_targeted_lookup(state: GState) -> GState:
    if not state.get("enable_rag", True) or not state["unknown_terms"]:
        return state
    # backfill web para términos desconocidos
    added = await enrich_terms(state["unknown_terms"][:10], state["domain_r"], state["client_id"])
    if added:
        rs = await RAGStore().retrieve(state["masked_text"][:2000], state["domain_r"], state["client_id"], topk=settings.rag_topk)
        trace.log("web_backfill", added=added, rag_after=len(rs))
        return {**state, "rag_snips": rs}
    return state

async def n_validate_and_update_glossary(state: GState) -> GState:
    if not state["unknown_terms"]:
        return state
    judged = await judge_terms(state["domain_r"], state["masked_text"], state["unknown_terms"][:12])
    ts = TermStore()
    # persistimos SOLO si el juez dice "relevant" y hay traducciones
    writes = 0
    for j in judged:
        if not j.relevant or not j.translations:
            continue
        for L, pref in j.translations.items():
            pref = (pref or "").strip()
            if not pref:
                continue
            await ts.upsert_preferred(GlossaryItem(
                client_id=None, domain=state["domain_r"], concept_key=j.canonical or j.translations.get("en") or "",
                lang=L, preferred=pref
            ))
            writes += 1
    trace.log("term_validation", candidates=len(state["unknown_terms"]), wrote=writes)
    return state

async def n_translate_and_review(state: GState) -> GState:
    segs = split_segments(state["masked_text"])
    src_lang = state["src_lang_r"]; domain = state["domain_r"]; dnt = state["dnt"]
    results: Dict[str, Any] = {}

    async def process_lang(tgt_lang: str):
        translated_segs: List[str] = []
        for seg in segs:
            hint = None
            hits = await TMStore().search(state["client_id"], src_text=seg, src_lang=src_lang, tgt_lang=tgt_lang, topk=1)
            if hits and hits[0][1] > 0.92:
                hint = hits[0][0]
            base = hint or await translate_text(
                src_lang=src_lang, tgt_lang=tgt_lang, domain=domain, style=STYLE_GUIDE,
                glossary_block=state["gl_blocks"][tgt_lang], dnt=dnt,
                rag_snippets=state["rag_snips"], source=seg
            )
            if settings.trace_prompts:
                trace.log("prompt:translator", tgt=tgt_lang, used_tm=bool(hint), seg_len=len(seg))
            ade, flu = await asyncio.gather(
                adequacy_review(src_lang=src_lang, tgt_lang=tgt_lang, domain=domain,
                                glossary_block=state["gl_blocks"][tgt_lang], source=seg, current=base),
                fluency_review(tgt_lang=tgt_lang, domain=domain, current=base)
            )
            merged = await edit_merge(tgt_lang=tgt_lang, domain=domain,
                                      adequacy_text=(ade.revised or base), fluency_text=(flu.revised or base))
            translated_segs.append(merged)

        masked_full = "\n\n".join(translated_segs)
        full = unmask(masked_full, state["mask_map"])

        # QA (domain score es async)
        num = numeric_consistency(state["text"], full)
        cov = terminology_coverage(full, state["gl_maps"][tgt_lang])
        from app.qa.validators import adomain_alignment_score
        doms = await adomain_alignment_score(domain, full)

        return tgt_lang, {"final": full, "qa": {"numeric_consistency": num, "term_coverage": cov, "domain_score": doms}}

    outs = await asyncio.gather(*[process_lang(L) for L in state["targets_r"]])
    for (L, data) in outs:
        results[L] = data
    return {**state, "results": results}

def build_graph():
    g = StateGraph(GState)
    g.add_node("detect_and_prepare", n_detect_and_prepare)
    g.add_node("decide_domain", n_decide_domain)
    g.add_node("build_context", n_build_context)
    g.add_node("targeted_lookup", n_targeted_lookup)
    g.add_node("validate_terms", n_validate_and_update_glossary)
    g.add_node("translate_and_review", n_translate_and_review)

    g.add_edge(START, "detect_and_prepare")
    g.add_edge("detect_and_prepare", "decide_domain")
    g.add_edge("decide_domain", "build_context")
    g.add_edge("build_context", "targeted_lookup")
    g.add_edge("targeted_lookup", "validate_terms")
    g.add_edge("validate_terms", "translate_and_review")
    g.add_edge("translate_and_review", END)
    return g.compile()

async def run_pipeline_graph(*, text: str, client_id: str, targets: Optional[List[str]],
                             src_lang_override: Optional[str],
                             domain_override: Optional[str],
                             enable_rag: bool, save_tm: bool,
                             debug: bool = False) -> Dict:
    # trazas
    tid = f"{client_id}:" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    trace.start(tid)

    app = build_graph()
    out: GState = await app.ainvoke({
        "text": text,
        "client_id": client_id,
        "targets": targets,
        "src_lang": src_lang_override,
        "domain": domain_override,
        "enable_rag": enable_rag,
        "save_tm": save_tm,
        "debug": debug,
    })

    payload = {
        "src_lang": out["src_lang_r"],
        "domain": out["domain_r"],
        "rag_used": bool(out.get("rag_snips")),
        "results": out["results"]
    }
    if debug and settings.trace_prompts:
        payload["trace"] = trace.get()
    return payload
