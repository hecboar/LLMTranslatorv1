from __future__ import annotations
import sqlite3, json, asyncio
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from ..config import settings

@dataclass
class GlossaryItem:
    client_id: str
    concept_key: str
    lang: str
    preferred: str

@dataclass
class DNTItem:
    client_id: str
    term: str

INIT_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS glossary_global(
  concept_key TEXT NOT NULL,
  lang TEXT NOT NULL,
  preferred TEXT NOT NULL,
  variants_json TEXT DEFAULT '[]',
  PRIMARY KEY (concept_key, lang)
);
CREATE TABLE IF NOT EXISTS glossary_client(
  client_id TEXT NOT NULL,
  concept_key TEXT NOT NULL,
  lang TEXT NOT NULL,
  preferred TEXT NOT NULL,
  PRIMARY KEY (client_id, concept_key, lang)
);
CREATE TABLE IF NOT EXISTS dnt_client(
  client_id TEXT NOT NULL,
  term TEXT NOT NULL,
  PRIMARY KEY (client_id, term)
);
"""

class TermStore:
    def __init__(self, path: str | None = None):
        self.path = path or settings.db_path
        self._ensure()

    def _ensure(self):
        with sqlite3.connect(self.path) as c:
            c.executescript(INIT_SQL)

    async def upsert_preferred(self, item: GlossaryItem):
        def _t():
            with sqlite3.connect(self.path) as c:
                c.execute(
                    "INSERT OR REPLACE INTO glossary_client(client_id,concept_key,lang,preferred) VALUES(?,?,?,?)",
                    (item.client_id, item.concept_key, item.lang, item.preferred),
                )
        await asyncio.to_thread(_t)

    async def add_variants_global(self, concept_key: str, lang: str, variants: List[str]):
        def _t():
            with sqlite3.connect(self.path) as c:
                row = c.execute(
                    "SELECT variants_json, preferred FROM glossary_global WHERE concept_key=? AND lang=?",
                    (concept_key, lang)
                ).fetchone()
                vs = set()
                pref = ""
                if row:
                    vs.update(json.loads(row[0] or "[]"))
                    pref = row[1] or ""
                vs.update([v for v in variants if v])
                c.execute(
                    "INSERT OR REPLACE INTO glossary_global(concept_key,lang,preferred,variants_json) VALUES(?,?,?,?)",
                    (concept_key, lang, pref, json.dumps(sorted(vs), ensure_ascii=False))
                )
        await asyncio.to_thread(_t)

    async def set_global_preferred(self, concept_key: str, lang: str, preferred: str):
        def _t():
            with sqlite3.connect(self.path) as c:
                row = c.execute(
                    "SELECT variants_json FROM glossary_global WHERE concept_key=? AND lang=?",
                    (concept_key, lang)
                ).fetchone()
                vj = row[0] if row else "[]"
                c.execute(
                    "INSERT OR REPLACE INTO glossary_global(concept_key,lang,preferred,variants_json) VALUES(?,?,?,?)",
                    (concept_key, lang, preferred, vj)
                )
        await asyncio.to_thread(_t)

    async def add_dnt(self, item: DNTItem):
        def _t():
            with sqlite3.connect(self.path) as c:
                c.execute(
                    "INSERT OR IGNORE INTO dnt_client(client_id,term) VALUES(?,?)",
                    (item.client_id, item.term)
                )
        await asyncio.to_thread(_t)

    async def export_client(self, client_id: str) -> Dict[str, Dict[str, str]]:
        def _t():
            with sqlite3.connect(self.path) as c:
                rows = c.execute(
                    "SELECT concept_key, lang, preferred FROM glossary_client WHERE client_id=?",
                    (client_id,)
                ).fetchall()
                out: Dict[str, Dict[str, str]] = {}
                for ck, lang, pref in rows:
                    out.setdefault(ck, {})[lang] = pref
                return out
        return await asyncio.to_thread(_t)

    async def glossary_block(self, client_id: str, lang: str) -> Tuple[str, Dict[str, str]]:
        def _t():
            with sqlite3.connect(self.path) as c:
                cl = c.execute(
                    "SELECT concept_key, preferred FROM glossary_client WHERE client_id=? AND lang=?",
                    (client_id, lang)
                ).fetchall()
                client_map = {ck: pref for ck, pref in cl}
                gl = c.execute(
                    "SELECT concept_key, preferred FROM glossary_global WHERE lang=?",
                    (lang,)
                ).fetchall()
                gmap = {ck: pref for ck, pref in gl}
                merged = dict(gmap)
                merged.update(client_map)
                lines = [f"- {k}: {v}" for k, v in sorted(merged.items()) if v]
                return "\n".join(lines), merged
        return await asyncio.to_thread(_t)

    async def dnt_list(self, client_id: str) -> List[str]:
        def _t():
            with sqlite3.connect(self.path) as c:
                rows = c.execute("SELECT term FROM dnt_client WHERE client_id=?", (client_id,)).fetchall()
                return [r[0] for r in rows]
        return await asyncio.to_thread(_t)
