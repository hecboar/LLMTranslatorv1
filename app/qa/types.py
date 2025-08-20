from __future__ import annotations
from pydantic import BaseModel

class AdequacyOut(BaseModel):
    ok: bool = True
    notes: str = ""
    revised: str = ""

class FluencyOut(BaseModel):
    revised: str = ""
    notes: str = ""

class QAStats(BaseModel):
    term_coverage: float = 0.0
    numeric_consistency: float = 0.0
    domain_score: float = 0.0
