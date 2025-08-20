# app/api/routers/terminology.py
from fastapi import APIRouter, HTTPException
from app.models.terminology import UpsertRequest, ValidateRequest, GlossaryResponse, ValidateResponse
from app.services.terminology_service import TerminologyService

router = APIRouter(prefix="/terminology/v1", tags=["terminology"])
svc = TerminologyService()

@router.get("/{domain}/{lang}", response_model=GlossaryResponse)
def get_glossary(domain: str, lang: str, client: str | None = None):
    try:
        return svc.load_glossary(domain, lang, client)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/upsert", response_model=GlossaryResponse)
def upsert(req: UpsertRequest):
    try:
        return svc.upsert_terms(
            scope=req.scope.value,
            domain=req.domain,
            lang=req.lang,
            terms=req.terms,
            client=req.client,
            allow_create=req.allow_create_scope,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

@router.post("/clients/{client}/bootstrap")
def bootstrap(client: str, domain: str):
    return svc.bootstrap_client(client=client, domain=domain)

@router.post("/validate", response_model=ValidateResponse)
def validate(req: ValidateRequest):
    try:
        return svc.validate_text(domain=req.domain, lang=req.lang, text=req.text, client=req.client)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))