from __future__ import annotations
import sqlite3, json, asyncio
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
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
  client_id   TEXT,           -- NULL = global
  domain      TEXT,           -- NULL = cross-domain
  concept_key TEXT NOT NULL,
  lang        TEXT NOT NULL,
  preferred   TEXT NOT NULL,
  variants_json TEXT DEFAULT '[]',
  PRIMARY KEY (client_id, domain, concept_key, lang)
);

CREATE TABLE IF NOT EXISTS dnt_client(
  client_id TEXT NOT NULL,
  term      TEXT NOT NULL,
  PRIMARY KEY (client_id, term)
);

-- helper indexes
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
        """
        Upsert preferred form; preserves existing variants_json.
        """
        def _t():
            with sqlite3.connect(self.path) as c:
                c.execute(
                    """INSERT OR REPLACE INTO glossary
                       (client_id,domain,concept_key,lang,preferred,variants_json)
                       VALUES(?,?,?,?,?,
                              COALESCE((SELECT variants_json FROM glossary
                                        WHERE client_id IS ? AND domain IS ?
                                          AND concept_key=? AND lang=?),'[]'))""",
                    (item.client_id, item.domain, item.concept_key, item.lang, item.preferred,
                     item.client_id, item.domain, item.concept_key, item.lang)
                )
        await asyncio.to_thread(_t)

    async def add_variants(self, concept_key: str, lang: str, variants: List[str],
                           client_id: str | None = None, domain: str | None = None):
        """
        Merge variants for a concept/lang at a given scope.
        """
        def _t():
            with sqlite3.connect(self.path) as c:
                row = c.execute(
                    "SELECT variants_json FROM glossary WHERE client_id IS ? "
                    "AND domain IS ? AND concept_key=? AND lang=?",
                    (client_id, domain, concept_key, lang)
                ).fetchone()
                vs = set()
                if row:
                    try:
                        vs.update(json.loads(row[0] or "[]"))
                    except Exception:
                        pass
                vs.update([v for v in variants if v])
                c.execute(
                    "INSERT OR REPLACE INTO glossary(client_id,domain,concept_key,lang,preferred,variants_json) "
                    "VALUES(?,?,?,?,COALESCE((SELECT preferred FROM glossary WHERE client_id IS ? "
                    "AND domain IS ? AND concept_key=? AND lang=?),''),?)",
                    (client_id, domain, concept_key, lang,
                     client_id, domain, concept_key, lang,
                     json.dumps(sorted(vs), ensure_ascii=False))
                )
        await asyncio.to_thread(_t)

    async def add_dnt(self, item: DNTItem):
        def _t():
            with sqlite3.connect(self.path) as c:
                c.execute("INSERT OR IGNORE INTO dnt_client(client_id,term) VALUES(?,?)", (item.client_id, item.term))
        await asyncio.to_thread(_t)

    async def export_client(self, client_id: str) -> Dict[str, Dict[str, str]]:
        """
        Returns a dict: {"<DOMAIN or GLOBAL>::<concept_key>": {"en": "...", "fr": "...", ...}, ...}
        """
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

    # -------- Helpers used by tests/seed --------
    async def set_global_preferred(self, concept_key: str, lang: str, preferred: str):
        """
        Convenience used in tests/seed fixtures.
        """
        await self.upsert_preferred(GlossaryItem(
            client_id=None, domain=None, concept_key=concept_key, lang=lang, preferred=preferred
        ))

    # -------- Fuzzy lookup & blocks --------
    async def find_preferred_fuzzy(self, client_id: str | None, domain: str | None,
                                   lang: str, query_key: str) -> Optional[str]:
        """
        Priority search for a preferred term by concept_key or variants (case-insensitive):
          1) client+domain
          2) client (cross-domain)
          3) global+domain
          4) global cross-domain
        Returns preferred or None.
        """
        qkey = (query_key or "").strip()
        if not qkey:
            return None
        qkey_l = qkey.lower()

        def _like_payload(x: str) -> str:
            # crude but effective: look for `"term"` in the JSON string
            return f'%"{x}"%'

        def _t() -> Optional[str]:
            with sqlite3.connect(self.path) as c:
                c.row_factory = sqlite3.Row
                def one(cid, dom) -> Optional[str]:
                    r = c.execute(
                        """
                        SELECT preferred
                        FROM glossary
                        WHERE client_id IS ? AND domain IS ? AND lang=?
                          AND (
                                LOWER(concept_key) = ?
                             OR LOWER(variants_json) LIKE ?
                          )
                        LIMIT 1
                        """,
                        (cid, dom, lang, qkey_l, _like_payload(qkey_l))
                    ).fetchone()
                    return (r["preferred"] if r else None)

                for cid, dom in [(client_id, domain), (client_id, None), (None, domain), (None, None)]:
                    pref = one(cid, dom)
                    if pref:
                        return pref
                return None

        return await asyncio.to_thread(_t)

    async def glossary_block(self, client_id: str, domain: str | None, lang: str) -> Tuple[str, Dict[str, str]]:
        """
        Priority of lookup:
        1) client+domain
        2) client (cross-domain)
        3) global+domain
        4) global (cross-domain)
        Returns the Jinja-ready block and a map {concept_key: preferred}.
        """
        def _t():
            with sqlite3.connect(self.path) as c:
                def fetch(cid, dom):
                    return c.execute(
                        "SELECT concept_key, preferred FROM glossary WHERE client_id IS ? "
                        "AND domain IS ? AND lang=? AND preferred<>''",
                        (cid, dom, lang)
                    ).fetchall()

                m: Dict[str, str] = {}
                for cid, dom in [(client_id, domain), (client_id, None), (None, domain), (None, None)]:
                    rows = fetch(cid, dom)
                    for ck, pref in rows:
                        m[ck] = pref  # override by priority
                lines = [f"- {k}: {v}" for k, v in sorted(m.items())]
                return "\n".join(lines), m
        return await asyncio.to_thread(_t)
