# app/services/terminology_service.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import re

from app.core.settings import get_settings
from app.utils.slug import slugify

BANNED = {
    "es": {
        "GP": ["Gestor de Proyectos"],
        "DPI": ["Índice de Precios al Consumidor", "IPC"],
    },
    "fr": {},
    "de": {},
}

class TerminologyService:
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or get_settings().TERMINOLOGY_BASE_DIR)

    # --- path helpers ---
    def _paths(self, domain: str, lang: str, client: Optional[str] = None) -> Tuple[Path, Path, Optional[Path]]:
        dslug = slugify(domain)
        g = self.base_dir / "global" / dslug / f"{lang}.json"
        f = self.base_dir / "flanks" / dslug / f"{lang}.json"
        c = None
        if client:
            cslug = slugify(client)
            c = self.base_dir / "clients" / cslug / dslug / f"{lang}.json"
        return g, f, c

    def _ensure_file(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("{}", encoding="utf-8")

    def _read_json(self, path: Path) -> Dict[str, str]:
        if not path or not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
                if not isinstance(data, dict):
                    return {}
                # normaliza claves a su forma exacta
                return {str(k): str(v) for k, v in data.items()}
            except json.JSONDecodeError:
                return {}

    def _write_json(self, path: Path, data: Dict[str, str]):
        self._ensure_file(path)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)

    # --- public API ---
    def load_glossary(self, domain: str, lang: str, client: Optional[str] = None):
        gp, fp, cp = self._paths(domain, lang, client)
        g = self._read_json(gp)
        f = self._read_json(fp)
        c = self._read_json(cp) if cp else {}
        merged = {**g, **f, **c}  # precedence: client > flanks > global
        return {
            "domain": domain,
            "lang": lang,
            "client": client,
            "scope_paths": {"global": str(gp), "flanks": str(fp), "client": str(cp) if cp else None},
            "merged": merged,
        }

    def upsert_terms(self, scope: str, domain: str, lang: str, terms: Dict[str, str], client: Optional[str], allow_create: bool=True):
        gp, fp, cp = self._paths(domain, lang, client)
        target = {"global": gp, "flanks": fp, "client": cp}[scope]
        if target is None:
            raise ValueError("client es obligatorio cuando scope=client")
        if not target.exists() and not allow_create:
            raise FileNotFoundError(str(target))
        self._ensure_file(target)
        existing = self._read_json(target)
        # valida prohibidos
        lang_banned = BANNED.get(lang, {})
        for k, v in terms.items():
            for bad in lang_banned.get(k, []):
                if bad.lower() in str(v).lower():
                    raise ValueError(f"Traducción prohibida para {k!r}: contiene '{bad}'")
        existing.update({str(k): str(v) for k, v in terms.items()})
        self._write_json(target, existing)
        return self.load_glossary(domain, lang, client)

    def bootstrap_client(self, client: str, domain: str, lang_list=("es", "fr", "de")):
        # crea estructura vacía para un cliente
        for lang in lang_list:
            _, _, cp = self._paths(domain, lang, client)
            self._ensure_file(cp)
        return {"client": client, "domain": domain, "created": [f"{domain}:{l}" for l in lang_list]}

    def validate_text(self, domain: str, lang: str, text: str, client: Optional[str] = None):
        data = self.load_glossary(domain, lang, client)
        glossary = data["merged"]
        keys = list(glossary.keys())
        matched = set()
        lowered = text.lower()
        for key in keys:
            # busca palabras completas; permite espacios en claves (p.ej. "capital call")
            pat = r"(?<!\w)" + re.escape(key.lower()) + r"(?!\w)"
            if re.search(pat, lowered, flags=re.IGNORECASE):
                matched.add(key)
        total = len(keys) if keys else 1
        coverage = round(len(matched) / total, 4)
        return {
            "coverage": coverage,
            "total_terms": len(keys),
            "matched_terms": sorted(matched),
            "missing_terms": sorted(set(keys) - matched),
        }