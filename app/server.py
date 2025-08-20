from __future__ import annotations
from fastapi import FastAPI, HTTPException
import logging
from pydantic import BaseModel, Field, AliasChoices
from typing import List, Dict, Optional, Literal
from .logging_conf import setup_logging
from .pipelines.translate_pipeline import run_pipeline
from .pipelines.translate_graph import run_pipeline_graph
from .stores.term_store import TermStore, GlossaryItem, DNTItem
from .stores.rag_store import RAGStore
from .stores.tm_store import TMStore

setup_logging("INFO")
app = FastAPI(title="MultiAgent MT API", version="1.1.0")
logger = logging.getLogger(__name__)

class TranslateRequest(BaseModel):
    text: str
    targets: Optional[List[str]] = Field(default=None, description="ej: ['en','fr','de']")
    client_id: str = Field(default="__global__", description="identificador cliente")
    src_lang: Optional[Literal["en","es","fr","de"]] = None
    domain: Optional[str] = Field(default=None, validation_alias=AliasChoices("domain", "domain_hint"))
    enable_rag: bool = True
    save_tm: bool = True
    engine: Literal["async","graph"] = "graph"

class TranslateResponse(BaseModel):
    src_lang: str
    domain: str
    rag_used: bool
    results: Dict[str, Dict]

class GlossaryUpsert(BaseModel):
    client_id: str
    concept_key: str
    lang: str
    preferred: str
    variants: Optional[List[str]] = None

class DNTUpsert(BaseModel):
    client_id: str
    terms: List[str]

class RAGIngestRequest(BaseModel):
    urls: List[str]
    domain: str
    client_id: Optional[str] = None

class TMClearRequest(BaseModel):
    client_id: str

@app.post("/v1/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest):
    try:
        if req.engine == "graph":
            res = await run_pipeline_graph(
                text=req.text,
                client_id=req.client_id,
                targets=req.targets,
                src_lang_override=req.src_lang,
                domain_override=req.domain,
                enable_rag=req.enable_rag,
                save_tm=req.save_tm,
            )
        else:
            res = await run_pipeline(
                text=req.text,
                client_id=req.client_id,
                targets=req.targets,
                src_lang_override=req.src_lang,
                domain_override=req.domain,
                enable_rag=req.enable_rag,
                save_tm=req.save_tm,
            )
        return TranslateResponse(**res)
    except Exception as e:
        # Log full stack trace to help diagnose 500s
        logger.exception("/v1/translate failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/glossary/upsert")
async def glossary_upsert(item: GlossaryUpsert):
    try:
        ts = TermStore()
        await ts.upsert_preferred(GlossaryItem(
            client_id=item.client_id,
            concept_key=item.concept_key,
            lang=item.lang,
            preferred=item.preferred
        ))
        if item.variants:
            await ts.add_variants_global(item.concept_key, item.lang, item.variants)
        return {"status": "ok"}
    except Exception as e:
        logger.exception("/v1/glossary/upsert failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/glossary/{client_id}/export")
async def glossary_export(client_id: str):
    try:
        ts = TermStore()
        data = await ts.export_client(client_id)
        return {"client_id": client_id, "glossary": data}
    except Exception as e:
        logger.exception("/v1/glossary/%s/export failed: %s", client_id, e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/dnt/upsert")
async def dnt_upsert(req: DNTUpsert):
    try:
        ts = TermStore()
        for term in req.terms:
            await ts.add_dnt(DNTItem(client_id=req.client_id, term=term))
        return {"status": "ok", "count": len(req.terms)}
    except Exception as e:
        logger.exception("/v1/dnt/upsert failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/rag/ingest")
async def rag_ingest(req: RAGIngestRequest):
    try:
        rag = RAGStore()
        n = await rag.ingest_urls(req.urls, domain=req.domain, client_id=req.client_id)
        return {"status": "ok", "ingested": n}
    except Exception as e:
        logger.exception("/v1/rag/ingest failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/tm/clear")
async def tm_clear(req: TMClearRequest):
    try:
        tm = TMStore()
        await tm.clear_client(req.client_id)
        return {"status":"ok"}
    except Exception as e:
        logger.exception("/v1/tm/clear failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
