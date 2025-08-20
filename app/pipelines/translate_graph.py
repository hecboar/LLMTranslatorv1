from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple
import re, asyncio, hashlib, logging
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
from ..agents.term_mapper import find_candidates_hybrid
from ..qa.validators import numeric_consistency, terminology_coverage, adomain_alignment_score
from ..utils.textguards import mask, unmask
from ..config import settings
from ..telemetry import trace
from ..services.llm import llm_parse

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

# ---- Canonicalization for cross-lingual concepts (language-agnostic) ----

def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("ó","o").replace("á","a").replace("é","e").replace("í","i").replace("ú","u")
    s = re.sub(r"[^\w]+", "", s)
    return s

_ALIASES = {
    # Private Equity
    "irr": "IRR", "tir": "IRR", "tri": "IRR",
    "tvpi": "TVPI",
    "dpi": "DPI",
    "moic": "MOIC",
    "nav": "NAV", "vni": "NAV", "vna": "NAV",
    "gp": "GP", "generalpartner": "GP", "sociogeneral": "GP",
    "drypowder": "Dry powder",
    # FX / currencies
    "fx": "FX", "tipodecambio": "FX", "devises": "FX", "waehrung": "FX",
    # Real Estate
    "caprate": "cap rate", "tauxdecapitalisation": "cap rate", "kapitalisierungsrate": "cap rate",
    "noi": "NOI", "rbe": "NOI",
    "dscr": "DSCR",
    "ltv": "LTV",
    "wault": "WAULT",
    # Fiscal/Tax
    "vat": "VAT", "tva": "VAT", "mwst": "VAT",
    "withholdingtax": "withholding tax", "retenuealasource": "withholding tax", "quellensteuer": "withholding tax",
    # Wealth
    "ucits": "UCITS", "opcvm": "UCITS", "ogaw": "UCITS",
    "mifid": "MiFID",
    "priips": "PRIIPs",
    "kid": "KID",
}

def to_canonical(term: str) -> Tuple[str, bool]:
    k = _norm(term)
    if not k:
        return term, False
    if k in _ALIASES:
        return _ALIASES[k], True
    if re.fullmatch(r"[A-Za-z]{2,6}", term.strip()) and term.isupper():
        return term.strip(), True
    return term.strip(), False

# ---- Term proposal & quick QA (inline; avoids extra files) ----

from pydantic import BaseModel, Field

class TermProposal(BaseModel):
    term: str
    src_lang: str
    tgt_lang: str
    proposal: str = Field(..., description="One preferred, publication-ready target form")

class TermQuality(BaseModel):
    ok: bool
    confidence: float = Field(..., ge=0, le=1)
    reasons: List[str] = []

_PROPOSE_PROMPT = """You are a financial terminology adapter.
Given a SOURCE term in {src} and a TARGET language {tgt}, output ONE preferred target-language form.

Rules:
- If an established acronym exists in {tgt}, prefer 'Long form (ACRONYM)' for the first mention.
- Use industry-standard casing/diacritics. No extra notes.
- Return JSON only: {{"term":"{term}", "src_lang":"{src}", "tgt_lang":"{tgt}", "proposal":"..."}}
"""

_JUDGE_PROMPT = """Role: Term Quality Judge.
Return JSON:
{{
  "ok": true,
  "confidence": 0..1,
  "notes": "..."
}}
SOURCE: {source}
SRC_LANG: {src_lang}
TARGET_LANG: {tgt_lang}
TARGET: {target}
"""

async def _propose_term(term: str, src_lang: str, tgt_lang: str) -> str:
    out: TermProposal = await llm_parse(
        _PROPOSE_PROMPT.format(term=term[:120], src=src_lang, tgt=tgt_lang),
        model=settings.model_review,
        schema=TermProposal,
        temperature=0.0,
    )
    return out.proposal.strip()

async def _judge_term_quality(source: str, src_lang: str, target_lang: str, target_string: str) -> float:
    out: TermQuality = await llm_parse(
        _JUDGE_PROMPT.format(source=source[:120], src_lang=src_lang, tgt_lang=target_lang, target=target_string[:200]),
        model=settings.qa_llm_model,
        schema=TermQuality,
        temperature=0.0,
    )
    return float(max(0.0, min(1.0, out.confidence)))

# ---- Glossary resolver (per-target blocks, stores all 4 languages) ----

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
    # 1) lookup (priority scopes handled by store)
    existing = await ts.find_preferred_fuzzy(client_id, domain, tgt_lang, concept_key)
    if existing:
        return existing

    # 2) propose + QA
    proposal = await _propose_term(source_surface, src_lang, tgt_lang)
    conf = await _judge_term_quality(source_surface, src_lang, tgt_lang, proposal)

    # 3) optional retry with RAG
    min_q = float(getattr(settings, "term_quality_min", 0.75))
    if conf < min_q and enable_rag:
        try:
            await rag.web_backfill([source_surface, concept_key], domain=domain, client_id=client_id)
        except Exception:
            pass
        proposal = await _propose_term(source_surface, src_lang, tgt_lang)
        conf = await _judge_term_quality(source_surface, src_lang, tgt_lang, proposal)

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
    Returns {tgt_lang: glossary_block_string}.
    Ensures each discovered concept has a preferred form for all four languages (or only targets+src).
    """
    ts = TermStore()
    rag = RAGStore()

    # Start from existing client/domain blocks (so we don't miss pre-seeded prefs)
    base_blocks: Dict[str, str] = {}
    base_maps: Dict[str, Dict[str, str]] = {}
    for L in targets:
        b, m = await ts.glossary_block(client_id, domain, L)
        base_blocks[L] = b
        base_maps[L] = m

    # Known preferred strings to avoid proposing again
    known_strings: List[str] = []
    for m in base_maps.values():
        known_strings.extend(m.values())

    # Find new candidates in source
    cands = await find_candidates_hybrid(source_text, known_terms=known_strings, domain=domain)
    if settings.trace_prompts:
        trace.log("glossary.candidates", cands=cands, src_lang=src_lang)

    # Canonicalize
    concepts: Dict[str, str] = {}
    for t in cands:
        ck, _ = to_canonical(t)
        if ck not in concepts:
            concepts[ck] = t

    if not concepts:
        # No new concepts: return existing blocks
        return base_blocks

    # Ensure entries across languages
    langs_to_fill = set(ALL_LANGS) if fill_all_four_langs else set(targets) | {src_lang}
    sem = asyncio.Semaphore(6)

    async def fill(ck: str, surface: str, L: str):
        async with sem:
            pref = await _ensure_lang_pref(
                ts, rag,
                client_id=client_id, domain=domain,
                concept_key=ck, source_surface=surface,
                src_lang=src_lang, tgt_lang=L,
                enable_rag=enable_rag
            )
            if settings.trace_prompts:
                trace.log("glossary.ensure", ck=ck, lang=L, pref=pref)

    await asyncio.gather(*[
        fill(ck, surface, L) for ck, surface in concepts.items() for L in langs_to_fill
    ])

    # Rebuild per-target blocks (now includes newly inserted items)
    blocks: Dict[str, str] = {}
    for L in targets:
        b, _ = await ts.glossary_block(client_id, domain, L)
        blocks[L] = b
        if settings.trace_prompts:
            trace.log("glossary.block", lang=L, block=b)

    return blocks

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

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
    results: Dict[str, Any]

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

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

async def n_resolve_glossary(state: GState) -> GState:
    client_id = state["client_id"]
    domain = state["domain_r"]

    # RAG context (optional)
    rag_snips: List[str] = []
    if state.get("enable_rag", True):
        await RAGStore().load_seed_sources()
        rag_snips = await RAGStore().retrieve(
            state["masked_text"][:2000], domain, client_id, topk=settings.rag_topk
        )

    # Resolve per-target glossary blocks (also persists new prefs across languages)
    gl_blocks = await resolve_glossary_for_targets(
        source_text=state["masked_text"],
        src_lang=state["src_lang_r"],
        targets=state["targets_r"],
        client_id=client_id,
        domain=domain,
        enable_rag=state.get("enable_rag", True),
        fill_all_four_langs=True,
    )

    # Also produce the maps {concept_key: preferred} for coverage metrics
    ts = TermStore()
    gl_maps: Dict[str, Dict[str, str]] = {}
    for L in state["targets_r"]:
        _, m = await ts.glossary_block(client_id, domain, L)
        gl_maps[L] = m

    trace.log("context",
              glossary_terms=sum(len(m) for m in gl_maps.values()),
              rag=len(rag_snips))
    return {**state, "gl_blocks": gl_blocks, "gl_maps": gl_maps, "rag_snips": rag_snips}

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
                glossary_block=state["gl_blocks"].get(tgt_lang, ""), dnt=dnt,
                rag_snippets=state["rag_snips"], source=seg
            )
            if settings.trace_prompts:
                trace.log("prompt:translator", tgt=tgt_lang, used_tm=bool(hint), seg_len=len(seg))
            ade, flu = await asyncio.gather(
                adequacy_review(src_lang=src_lang, tgt_lang=tgt_lang, domain=domain,
                                glossary_block=state["gl_blocks"].get(tgt_lang, ""), source=seg, current=base),
                fluency_review(tgt_lang=tgt_lang, domain=domain, current=base)
            )
            merged = await edit_merge(tgt_lang=tgt_lang, domain=domain,
                                      adequacy_text=(ade.revised or base), fluency_text=(flu.revised or base))
            translated_segs.append(merged)

        masked_full = "\n\n".join(translated_segs)
        full = unmask(masked_full, state["mask_map"])

        # QA
        num = numeric_consistency(state["text"], full)
        cov = terminology_coverage(full, state["gl_maps"].get(tgt_lang, {}))
        doms = await adomain_alignment_score(domain, full)

        return tgt_lang, {"final": full, "qa": {"numeric_consistency": num, "term_coverage": cov, "domain_score": doms}}

    outs = await asyncio.gather(*[process_lang(L) for L in state["targets_r"]])
    for (L, data) in outs:
        results[L] = data
    return {**state, "results": results}

# ---------------------------------------------------------------------------
# Build & run
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(GState)
    g.add_node("detect_and_prepare", n_detect_and_prepare)
    g.add_node("decide_domain", n_decide_domain)
    g.add_node("resolve_glossary", n_resolve_glossary)
    g.add_node("translate_and_review", n_translate_and_review)

    g.add_edge(START, "detect_and_prepare")
    g.add_edge("detect_and_prepare", "decide_domain")
    g.add_edge("decide_domain", "resolve_glossary")
    g.add_edge("resolve_glossary", "translate_and_review")
    g.add_edge("translate_and_review", END)
    return g.compile()

async def run_pipeline_graph(*, text: str, client_id: str, targets: Optional[List[str]],
                             src_lang_override: Optional[str],
                             domain_override: Optional[str],
                             enable_rag: bool, save_tm: bool,
                             debug: bool = False) -> Dict:
    # traces
    tid = f"{client_id}:" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    trace.start(tid)
    log.info("translate_graph.start", extra={"client_id": client_id, "targets": targets,
                                             "src_override": src_lang_override, "domain_override": domain_override})

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
