# app/qa/validators.py
from __future__ import annotations
import re
from typing import Dict, List
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Reglas rápidas (baseline)
# ---------------------------------------------------------------------------
# Captura porcentajes, divisas con símbolo delante/detrás, sufijos de magnitud
# y multiplicadores tipo "1.8x" / "1,8x". Acepta signo y paréntesis (negativos).
NUM_RE = re.compile(
    r"""
    (?<!\w)
    (                                   # grupo completo
      \(?\s*[+\-]?                      # signo opcional, con posible '('
      (?:                               # --- ALTERNATIVAS ---
        # 1) Porcentajes (7.3%)
        (?:\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?)
        \s?%
        |
        # 2) Símbolo + cantidad (€, $, £) con sufijos opcionales y/o 'x'
        [€$£]\s*
        (?:\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?)
        (?:\s?(?:k|m|b|bn|mm))?
        (?:\s?[x×])?
        |
        # 3) Cantidad + símbolo (2,4 M€ / 1.2 m$)
        (?:\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?)
        (?:\s?(?:k|m|b|bn|mm))?
        (?:\s?[x×])?
        \s?[€$£]
        |
        # 4) Cantidad suelta con sufijo y/o 'x' (2.4M, 1,8x)
        (?:\d{1,3}(?:[.\s]\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?)
        (?:\s?(?:k|m|b|bn|mm))?
        (?:\s?[x×])?
      )
      \s*\)?                            # posible ')'
    )
    (?!\w)
    """,
    re.VERBOSE | re.IGNORECASE,
)

def extract_numbers(text: str) -> List[str]:
    """Devuelve los tokens numéricos relevantes tal como aparecen en el texto."""
    return [m.group(0).strip() for m in NUM_RE.finditer(text or "")]

def normalize_number_token(tok: str) -> str:
    """
    Normaliza para comparar entre idiomas:
    - Elimina símbolos monetarios y separadores, conserva signo y '%'
    - Suprime sufijos (k/m/b/bn/mm) y 'x' para comparar magnitudes exactas
    - Quita paréntesis; conserva el signo si estaba explícito
    """
    if not tok:
        return ""
    s = tok.replace("\xa0", " ").strip().lower()
    # signo explícito si está entre paréntesis (estilo contable) o al inicio
    neg = False
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
        neg = True
    if s.startswith("-"):
        neg = True
        s = s[1:].strip()
    if s.startswith("+"):
        s = s[1:].strip()

    has_pct = "%" in s
    # moneda y multiplicadores
    s = s.replace("€", "").replace("$", "").replace("£", "")
    s = s.replace("×", "x")
    s = re.sub(r"\b(k|m|b|bn|mm)\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bx\b", "", s, flags=re.IGNORECASE)

    # deja solo dígitos, coma, punto y signo
    core = re.sub(r"[^\d,.\-]", "", s)
    # unifica separadores (quita miles y separadores ambiguos)
    core = core.replace(".", "").replace(",", "").replace(" ", "")
    if not core:
        return ""
    if neg and not core.startswith("-"):
        core = "-" + core
    return (core + "%") if has_pct else core

def numeric_consistency(src: str, tgt: str) -> float:
    """
    Porcentaje de cantidades del SOURCE que aparecen en TARGET con la misma magnitud.
    """
    s = set(filter(None, (normalize_number_token(x) for x in extract_numbers(src))))
    if not s:
        return 1.0
    t = set(filter(None, (normalize_number_token(x) for x in extract_numbers(tgt))))
    return len(s & t) / len(s) if s else 1.0

def terminology_coverage(text: str, pref_map: Dict[str, str]) -> float:
    """
    Cobertura de los términos preferidos (client > global) en el texto objetivo.
    """
    if not pref_map:
        return 1.0
    total = len(pref_map)
    if total == 0:
        return 1.0
    hits = 0
    low = text.lower()
    for _, term in pref_map.items():
        if term and term.lower() in low:
            hits += 1
    return hits / total

def domain_alignment_score(domain: str, text: str) -> float:
    """Heurística con patrones amplios y sinónimos (baseline, rápida)."""
    patterns = {
        "Private Equity": [
            r"\bnav\b",
            r"\b(ir{1,2}|internal rate of return|tir)\b",
            r"\btvpi\b", r"\bdpi\b", r"\bmoic\b", r"\bdry powder\b",
            r"\bcapital call(s)?\b", r"\bdistribution(s)?\b",
            r"\bfund(s)?\b", r"\bportfolio revaluation\b",
        ],
        "Real Estate": [
            r"\bcap rate\b", r"\blease(s)?\b", r"\bnoi\b",
            r"\bltv\b", r"\bdscr\b", r"\bwault\b", r"\brent roll\b",
            r"\bvaluation\b",
        ],
        "Fiscal/Tax": [
            r"\bwithholding\b", r"\bvat\b", r"\btreat(y|ies)\b",
            r"\bcfc\b", r"\bbeps\b", r"\btransfer pricing\b",
            r"\bpermanent establishment\b",
        ],
        "Wealth Management": [
            r"\bmifid\b", r"\bucits\b", r"\bter\b",
            r"\bsharpe\b", r"\bportfolio\b", r"\bkid\b", r"\bpriip(s)?\b",
        ],
    }
    text_l = text.lower()
    pats = patterns.get(domain, [])
    hits = sum(1 for p in pats if re.search(p, text_l))
    # Base 0.5 + 0.1 por match (capado a 1.0)
    return min(1.0, 0.5 + 0.1 * hits)

# ---------------------------------------------------------------------------
# LLM “Referee” opcional (async)
# Se usan cuando settings.qa_use_llm=True para complementar las reglas.
# ---------------------------------------------------------------------------
from ..config import settings
from ..services.llm import llm_parse

# Estructuras de parseo tipado (Pydantic)
class NumericAudit(BaseModel):
    ok: bool = Field(..., description="True si TODOS los valores y magnitudes coinciden entre SOURCE y TARGET (independiente del formato)")
    matched_ratio: float = Field(..., ge=0, le=1, description="Proporción de cantidades del SOURCE que aparecen correctamente preservadas en TARGET")
    confidence: float = Field(..., ge=0, le=1)
    issues: List[str] = Field(default_factory=list)

class DomainAudit(BaseModel):
    domain: str
    aligned: bool
    confidence: float = Field(..., ge=0, le=1)
    cues: List[str] = Field(default_factory=list)  # evidencias (términos o ideas)

NUMERIC_AUDIT_PROMPT = """Role: Financial Translation QA Referee.
Task: Check that all QUANTITIES in SOURCE are preserved in TARGET.
- Consider percentages, currencies (€, $, £), multipliers k/M/B/bn/mm, and 'x' multipliers (e.g., 1.8x).
- Locale formatting changes and currency symbol position are allowed.
- Units and magnitude MUST remain identical.

Return JSON with:
- ok: true/false
- matched_ratio: fraction of SOURCE quantities preserved in TARGET (0..1)
- confidence: 0..1
- issues: short bullet points if something mismatches.

SOURCE:
{source}

TARGET:
{target}
"""

DOMAIN_AUDIT_PROMPT = """Role: Financial Domain Auditor.
Decide if the TEXT aligns with the domain "{domain}" (one of: Private Equity, Real Estate, Fiscal/Tax, Wealth Management).

Guidance:
- Private Equity: NAV, IRR/TIR, TVPI, DPI, MOIC, capital calls, distributions, fund portfolio revaluation, dry powder.
- Real Estate: cap rate, leases, NOI, LTV, DSCR, WAULT, rent roll, valuation specifics.
- Fiscal/Tax: VAT, withholding, treaties, BEPS, transfer pricing, CFC, permanent establishment.
- Wealth Management: UCITS, MiFID, KID/PRIIPs, portfolio metrics (TER, Sharpe), retail investor disclosures.

Return JSON:
- domain: repeated input domain
- aligned: true/false
- confidence: 0..1 (semantic fit)
- cues: list of specific tokens/phrases found.

TEXT:
{text}
"""

async def anumeric_consistency(src: str, tgt: str) -> float:
    """
    Híbrido: si reglas dan 1.0 devolvemos 1.0. Si no, pedimos veredicto al LLM.
    Combinamos: score = min(1.0, (1-w)*rule + w*(0.5*matched_ratio + 0.5*confidence))
    """
    rule = numeric_consistency(src, tgt)
    if rule >= 1.0:
        return 1.0
    try:
        prompt = NUMERIC_AUDIT_PROMPT.format(source=src[:4000], target=tgt[:4000])
        audit: NumericAudit = await llm_parse(
            prompt,
            model=getattr(settings, "qa_llm_model", settings.model_review),
            schema=NumericAudit,
            temperature=0.0,
        )
        llm_score = 0.5 * float(audit.matched_ratio) + 0.5 * float(audit.confidence)
        w = float(getattr(settings, "qa_weight_llm", 0.6))
        return min(1.0, (1 - w) * rule + w * llm_score)
    except Exception:
        # fallback robusto si el LLM falla
        return rule

async def adomain_alignment_score(domain: str, text: str) -> float:
    """
    Híbrido: heurística + confianza LLM.
    score = min(1.0, max(rule, (1-w)*rule + w*confidence))
    """
    rule = domain_alignment_score(domain, text)
    try:
        prompt = DOMAIN_AUDIT_PROMPT.format(domain=domain, text=text[:4000])
        audit: DomainAudit = await llm_parse(
            prompt,
            model=getattr(settings, "qa_llm_model", settings.model_review),
            schema=DomainAudit,
            temperature=0.0,
        )
        w = float(getattr(settings, "qa_weight_llm", 0.6))
        return min(1.0, max(rule, (1 - w) * rule + w * float(audit.confidence)))
    except Exception:
        return rule
