from __future__ import annotations
import re
from typing import Dict, Tuple, List

def mask(text: str, dnt: List[str]) -> Tuple[str, Dict[str, str]]:
    if not dnt:
        return text, {}
    mapping: Dict[str, str] = {}
    masked = text
    for i, term in enumerate(sorted(set(dnt), key=len, reverse=True), start=1):
        if not term.strip():
            continue
        placeholder = f"[[ENT_{i}]]"
        mapping[placeholder] = term
        # mask case-insensitive whole-word-ish
        masked = re.sub(rf"\b{re.escape(term)}\b", placeholder, masked, flags=re.IGNORECASE)
    return masked, mapping

def unmask(text: str, mapping: Dict[str, str]) -> str:
    out = text
    for ph, term in mapping.items():
        out = out.replace(ph, term)
    return out
