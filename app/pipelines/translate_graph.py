from __future__ import annotations
from typing import Dict, List, Optional, Any
import re, asyncio, hashlib
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    _ASYNC_CP = True
except Exception:
    AsyncSqliteSaver = None  # type: ignore
    _ASYNC_CP = False

from ..agents.router import detect_lang, decide_domain
from ..domain.taxonomy import STYLE_GUIDE, normalize_domain
from ..stores.term_store import TermStore
from ..stores.rag_store import RAGStore
from ..stores.tm_store import TMStore
from ..agents.translator import translate_text
from ..agents.adequacy_reviewer import adequacy_review
from ..agents.fluency_reviewer import fluency_review
from ..agents.editor import edit_merge
from ..agents.term_mapper import find_candidates, enrich_terms
from ..qa.validators import numeric_consistency, terminology_coverage, domain_alignment_score
from ..utils.textguards import mask, unmask
from ..config import settings


def split_segments(text: str, max_chars: int = 1400) -> List[str]:
    parts = re.split(r"\n\s*\n|\r\n\r\n", text.strip())
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


class GState(TypedDict, total=False):
    text: str
    client_id: str
    targets: Optional[List[str]]
    src_lang: Optional[str]
    domain: Optional[str]
    enable_rag: bool
    save_tm: bool
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
    loop: int
    results: Dict[str, Any]


async def n_detect_and_prepare(state: GState) -> GState:
    src = state.get("src_lang") or detect_lang(state["text"])
    targets = state.get("targets") or [l for l in ["en", "fr", "de", "es"] if l != src] or ["en", "fr", "de"]
    ts = TermStore()
    dnt = await ts.dnt_list(state["client_id"])
    masked_text, mask_map = mask(state["text"], dnt)
    return {
        **state,
        "src_lang_r": src,
        "targets_r": targets,
        "dnt": dnt,
        "masked_text": masked_text,
        "mask_map": mask_map,
        "loop": state.get("loop", 0),
    }


async def n_decide_domain(state: GState) -> GState:
    if state.get("domain"):
        dom = normalize_domain(state["domain"])
    else:
        dom = await decide_domain(state["masked_text"], hint=None)
    return {**state, "domain_r": dom}


async def n_build_context(state: GState) -> GState:
    ts = TermStore()
    client_id = state["client_id"]
    gl_blocks: Dict[str, str] = {}
    gl_maps: Dict[str, Dict[str, str]] = {}
    for L in state["targets_r"]:
        block, m = await ts.glossary_block(client_id, L)
        gl_blocks[L] = block
        gl_maps[L] = m
    rag_snips: List[str] = []
    if state.get("enable_rag", True):
        await RAGStore().load_seed_sources()
        rag_snips = await RAGStore().retrieve(
            state["masked_text"][:2000], state["domain_r"], client_id, topk=settings.rag_topk
        )
    known = []
    for m in gl_maps.values():
        known.extend(m.values())
    unknown = find_candidates(state["masked_text"], known_terms=known)
    return {**state, "gl_blocks": gl_blocks, "gl_maps": gl_maps, "rag_snips": rag_snips, "unknown_terms": unknown}


async def n_targeted_lookup(state: GState) -> GState:
    if not state.get("enable_rag", True):
        return state
    added = await enrich_terms(state["unknown_terms"][:10], state["domain_r"], state["client_id"])
    if added:
        rs = await RAGStore().retrieve(
            state["masked_text"][:2000], state["domain_r"], state["client_id"], topk=settings.rag_topk
        )
        return {**state, "rag_snips": rs}
    return state


async def n_translate_and_review(state: GState) -> GState:
    segs = split_segments(state["masked_text"])
    src_lang = state["src_lang_r"]
    domain = state["domain_r"]
    dnt = state["dnt"]

    async def process_lang(tgt_lang: str):
        translated_segs: List[str] = []
        for seg in segs:
            hint = None
            hits = await TMStore().search(
                state["client_id"], src_text=seg, src_lang=src_lang, tgt_lang=tgt_lang, topk=1
            )
            if hits and hits[0][1] > 0.92:
                hint = hits[0][0]
            if hint:
                base = hint
            else:
                base = await translate_text(
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    domain=domain,
                    style=STYLE_GUIDE,
                    glossary_block=state["gl_blocks"][tgt_lang],
                    dnt=dnt,
                    rag_snippets=state["rag_snips"],
                    source=seg,
                )
            ade_task = adequacy_review(
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                domain=domain,
                glossary_block=state["gl_blocks"][tgt_lang],
                source=seg,
                current=base,
            )
            flu_task = fluency_review(tgt_lang=tgt_lang, domain=domain, current=base)
            ade, flu = await asyncio.gather(ade_task, flu_task)
            merged = await edit_merge(
                tgt_lang=tgt_lang,
                domain=domain,
                adequacy_text=(ade.revised or base),
                fluency_text=(flu.revised or base),
            )
            translated_segs.append(merged)

        masked_full = "\n\n".join(translated_segs)
        full = unmask(masked_full, state["mask_map"])
        num = numeric_consistency(state["text"], full)
        cov = terminology_coverage(full, state["gl_maps"][tgt_lang])
        doms = domain_alignment_score(domain, full)
        return tgt_lang, {"final": full, "qa": {"numeric_consistency": num, "term_coverage": cov, "domain_score": doms}}

    outs = await asyncio.gather(*[process_lang(L) for L in state["targets_r"]])
    results = {L: data for (L, data) in outs}
    return {**state, "results": results}


def _qa_ok(d: Dict[str, Any]) -> bool:
    q = d["qa"]
    return (
        q["numeric_consistency"] >= settings.qa_num_min
        and q["term_coverage"] >= settings.qa_term_min
        and q["domain_score"] >= settings.qa_dom_min
    )


async def n_quality_gate(state: GState) -> GState:
    bad = [L for L, r in state["results"].items() if not _qa_ok(r)]
    if not bad or state.get("loop", 0) >= settings.qa_max_loops:
        for L, r in state["results"].items():
            if _qa_ok(r) and state.get("save_tm", True):
                await TMStore().upsert(
                    state["client_id"], state["src_lang_r"], L, state["domain_r"], state["text"], r["final"]
                )
        return state

    fixed: Dict[str, Any] = dict(state["results"])
    for L in bad:
        base = state["results"][L]["final"]
        hints = []
        for _, term in state["gl_maps"][L].items():
            if term and term.lower() not in base.lower():
                hints.append(f'Ensure term "{term}" appears.')
        extra_prompt = "NOTE: Fix to include missing preferred terms and keep all numbers exactly."
        revised = await edit_merge(
            tgt_lang=L,
            domain=state["domain_r"],
            adequacy_text=base + "\n\n" + "\n".join(hints),
            fluency_text=base + "\n\n" + extra_prompt,
        )
        num = numeric_consistency(state["text"], revised)
        cov = terminology_coverage(revised, state["gl_maps"][L])
        doms = domain_alignment_score(state["domain_r"], revised)
        fixed[L] = {"final": revised, "qa": {"numeric_consistency": num, "term_coverage": cov, "domain_score": doms}}
        if _qa_ok(fixed[L]) and state.get("save_tm", True):
            await TMStore().upsert(
                state["client_id"], state["src_lang_r"], L, state["domain_r"], state["text"], fixed[L]["final"]
            )

    return {**state, "results": fixed, "loop": state.get("loop", 0) + 1}


def build_graph():
    g = StateGraph(GState)
    g.add_node("detect_and_prepare", n_detect_and_prepare)
    g.add_node("decide_domain", n_decide_domain)
    g.add_node("build_context", n_build_context)
    g.add_node("targeted_lookup", n_targeted_lookup)
    g.add_node("translate_and_review", n_translate_and_review)
    g.add_node("quality_gate", n_quality_gate)

    g.add_edge(START, "detect_and_prepare")
    g.add_edge("detect_and_prepare", "decide_domain")
    g.add_edge("decide_domain", "build_context")
    g.add_edge("build_context", "targeted_lookup")
    g.add_edge("targeted_lookup", "translate_and_review")
    g.add_edge("translate_and_review", "quality_gate")
    g.add_edge("quality_gate", END)

    if _ASYNC_CP and AsyncSqliteSaver is not None:
        try:
            # ✅ Aquí usamos from_path en lugar de from_conn_string
            cp = AsyncSqliteSaver.from_path(settings.checkpoint_db)
            return g.compile(checkpointer=cp)
        except Exception as e:
            print(f"[WARN] Falling back to no checkpoint due to: {e}")
            pass

    return g.compile()


async def run_pipeline_graph(
    *, text: str, client_id: str, targets: Optional[List[str]], src_lang_override: Optional[str],
    domain_override: Optional[str], enable_rag: bool, save_tm: bool
) -> Dict:
    app = build_graph()
    tid = f"{client_id}:" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    cfg = {"configurable": {"thread_id": tid, "checkpoint_ns": "translate"}}
    out: GState = await app.ainvoke(
        {
            "text": text,
            "client_id": client_id,
            "targets": targets,
            "src_lang": src_lang_override,
            "domain": domain_override,
            "enable_rag": enable_rag,
            "save_tm": save_tm,
        },
        config=cfg,
    )
    return {
        "src_lang": out["src_lang_r"],
        "domain": out["domain_r"],
        "rag_used": bool(out.get("rag_snips")),
        "results": out["results"],
    }
