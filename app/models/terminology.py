# app/models/terminology.py
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, validator

LANGS = {"es", "fr", "de"}

class Scope(str, Enum):
    global_scope = "global"
    flanks = "flanks"
    client = "client"

class UpsertRequest(BaseModel):
    scope: Scope = Field(..., description="Dónde guardar: global | flanks | client")
    client: Optional[str] = Field(None, description="Requerido si scope=client")
    domain: str = Field(..., example="Private Equity")
    lang: str = Field(..., example="es")
    terms: Dict[str, str] = Field(..., description="Diccionario término→traducción preferida")
    allow_create_scope: bool = Field(True, description="Crea árbol y fichero si no existen")

    @validator("lang")
    def _lang_ok(cls, v):
        if v not in LANGS:
            raise ValueError(f"lang debe ser uno de {sorted(LANGS)}")
        return v

class ValidateRequest(BaseModel):
    domain: str
    lang: str
    text: str
    client: Optional[str] = None

class ValidateResponse(BaseModel):
    coverage: float
    total_terms: int
    matched_terms: List[str]
    missing_terms: List[str]

class GlossaryResponse(BaseModel):
    domain: str
    lang: str
    client: Optional[str]
    scope_paths: dict
    merged: Dict[str, str]