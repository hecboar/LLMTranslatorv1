from __future__ import annotations
import sqlite3, json, asyncio
from dataclasses import dataclass
from typing import List, Dict, Tuple
from ..config import settings

@dataclass
class GlossaryItem:
    client_id: str | None  # None == global
    domain: str | None     # None == cross-domain
    concept_key: str
    lang: str
    preferred: str

@dataclass
class DNTItem:
    client_id: str
    term: str

INIT_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS glossary(
  client_id TEXT,           -- NULL = global
  domain    TEXT,           -- NULL = cross-domain
  concept_key TEXT NOT NULL,
  lang TEXT NOT NULL,
  preferred TEXT NOT NULL,
  variants_json TEXT DEFAULT '[]',
  PRIMARY KEY (client_id, domain, concept_key, lang)
);

CREATE TABLE IF NOT EXISTS dnt_client(
  client_id TEXT NOT NULL,
  term TEXT NOT NULL,
  PRIMARY KEY (client_id, term)
);

-- índices auxiliares (para búsquedas rápidas y unicidad case-insensitive si lo necesitas)
CREATE INDEX IF NOT EXISTS idx_glossary_concept_lang
ON glossary(concept_key, lang);

CREATE INDEX IF NOT EXISTS idx_glossary_client_domain
ON glossary(client_id, domain);
"""

class TermStore:
    def __init__(self, path: str | None = None):
        self.path = path or settings.db_path
        with sqlite3.connect(self.path) as c:
            c.executescript(INIT_SQL)

    # -------- CRUD --------
    async def upsert_preferred(self, item: GlossaryItem):
        def _t():
            with sqlite3.connect(self.path) as c:
                c.execute(
                    """INSERT OR REPLACE INTO glossary(client_id,domain,concept_key,lang,preferred,variants_json)
                       VALUES(?,?,?,?,?,COALESCE((SELECT variants_json FROM glossary 
                               WHERE client_id IS ? AND domain IS ? 
                                 AND concept_key=? AND lang=?),'[]'))""",
                    (item.client_id, item.domain, item.concept_key, item.lang, item.preferred,
                     item.client_id, item.domain, item.concept_key, item.lang)
                )
        await asyncio.to_thread(_t)

    async def add_variants(self, concept_key: str, lang: str, variants: List[str],
                           client_id: str | None = None, domain: str | None = None):
        def _t():
            with sqlite3.connect(self.path) as c:
                row = c.execute(
                    "SELECT variants_json FROM glossary WHERE client_id IS ? "
                    "AND domain IS ? AND concept_key=? AND lang=?",
                    (client_id, domain, concept_key, lang)
                ).fetchone()
                vs = set()
                if row:
                    vs.update(json.loads(row[0] or "[]"))
                vs.update([v for v in variants if v])
                c.execute(
                    "INSERT OR REPLACE INTO glossary(client_id,domain,concept_key,lang,preferred,variants_json) "
                    "VALUES(?,?,?,?,COALESCE((SELECT preferred FROM glossary WHERE client_id IS ? "
                    "AND domain IS ? AND concept_key=? AND lang=?),''),?)",
                    (client_id, domain, concept_key, lang, client_id, domain, concept_key, lang,
                     json.dumps(sorted(vs), ensure_ascii=False))
                )
        await asyncio.to_thread(_t)

    async def add_dnt(self, item: DNTItem):
        def _t():
            with sqlite3.connect(self.path) as c:
                c.execute("INSERT OR IGNORE INTO dnt_client(client_id,term) VALUES(?,?)", (item.client_id, item.term))
        await asyncio.to_thread(_t)

    async def export_client(self, client_id: str) -> Dict[str, Dict[str, str]]:
        def _t():
            with sqlite3.connect(self.path) as c:
                rows = c.execute(
                    "SELECT domain, concept_key, lang, preferred FROM glossary WHERE client_id=?",
                    (client_id,)
                ).fetchall()
                out: Dict[str, Dict[str, str]] = {}
                for dom, ck, lang, pref in rows:
                    key = f"{(dom or 'GLOBAL')}::{ck}"
                    out.setdefault(key, {})[lang] = pref
                return out
        return await asyncio.to_thread(_t)

    async def dnt_list(self, client_id: str) -> List[str]:
        def _t():
            with sqlite3.connect(self.path) as c:
                rows = c.execute("SELECT term FROM dnt_client WHERE client_id=?", (client_id,)).fetchall()
                return [r[0] for r in rows]
        return await asyncio.to_thread(_t)

    # -------- Lectura compuesta (prioridades) --------
    async def glossary_block(self, client_id: str, domain: str | None, lang: str) -> Tuple[str, Dict[str, str]]:
        """
        Prioridad de lookup:
        1) cliente+dominio
        2) cliente (cross-domain)
        3) global+dominio
        4) global (cross-domain)
        """
        def _t():
            with sqlite3.connect(self.path) as c:
                def fetch(cid, dom):
                    return c.execute(
                        "SELECT concept_key, preferred FROM glossary WHERE client_id IS ? "
                        "AND domain IS ? AND lang=? AND preferred<>''",
                        (cid, dom, lang)
                    ).fetchall()

                m: Dict[str,str] = {}
                for cid, dom in [(client_id, domain), (client_id, None), (None, domain), (None, None)]:
                    rows = fetch(cid, dom)
                    for ck, pref in rows:
                        m[ck] = pref  # override por prioridad
                lines = [f"- {k}: {v}" for k, v in sorted(m.items())]
                return "\n".join(lines), m
        return await asyncio.to_thread(_t)
