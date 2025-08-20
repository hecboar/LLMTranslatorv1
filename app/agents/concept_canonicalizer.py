# app/agents/concept_canonicalizer.py
from __future__ import annotations
import re
from typing import Optional, Tuple

# Normalize surface (lower, strip punctuation/spaces/diacritics-lite)
def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("ó","o").replace("á","a").replace("é","e").replace("í","i").replace("ú","u")
    s = re.sub(r"[^\w]+", "", s)  # drop spaces, hyphens, slashes, dots
    return s

# Seed cross-lingual aliases → canonical concept_key (use a stable English key when possible)
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
    "fx": "FX", "tipo decambio": "FX", "devises": "FX", "waehrung": "FX",
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
    """
    Returns (concept_key, is_known). If unknown, uses the raw surface as key.
    """
    k = _norm(term)
    if not k:
        return term, False
    if k in _ALIASES:
        return _ALIASES[k], True
    # If it looks like an acronym (2–6 upper letters), use that as canonical
    if re.fullmatch(r"[A-Za-z]{2,6}", term.strip()) and term.isupper():
        return term.strip(), True
    # Fallback: return the visible surface as the key
    return term.strip(), False
