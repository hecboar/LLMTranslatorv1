from __future__ import annotations
import sqlite3, asyncio
from typing import Optional, List, Tuple
import numpy as np
from ..config import settings
from ..services.embeddings import embed_texts, cosine_sim

INIT_SQL = """
CREATE TABLE IF NOT EXISTS tm_segments(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  client_id TEXT NOT NULL,
  src_lang TEXT NOT NULL,
  tgt_lang TEXT NOT NULL,
  domain TEXT,
  src_text TEXT NOT NULL,
  tgt_text TEXT NOT NULL,
  src_vec BLOB
);
CREATE INDEX IF NOT EXISTS idx_tm_client ON tm_segments(client_id);
"""

class TMStore:
    def __init__(self, path: str | None = None):
        self.path = path or settings.db_path
        with sqlite3.connect(self.path) as c:
            c.executescript(INIT_SQL)

    async def upsert(self, client_id: str, src_lang: str, tgt_lang: str, domain: str, src_text: str, tgt_text: str):
        vec = await embed_texts([src_text])
        blob = vec.tobytes()
        def _t():
            with sqlite3.connect(self.path) as c:
                c.execute(
                    "INSERT INTO tm_segments(client_id,src_lang,tgt_lang,domain,src_text,tgt_text,src_vec) VALUES(?,?,?,?,?,?,?)",
                    (client_id, src_lang, tgt_lang, domain, src_text, tgt_text, blob)
                )
        await asyncio.to_thread(_t)

    async def search(self, client_id: str, src_text: str, src_lang: str, tgt_lang: str, topk: int = 1) -> List[Tuple[str, float]]:
        vec = await embed_texts([src_text])
        v = vec[0]
        def _t():
            with sqlite3.connect(self.path) as c:
                rows = c.execute(
                    "SELECT tgt_text, src_vec FROM tm_segments WHERE client_id=? AND src_lang=? AND tgt_lang=?",
                    (client_id, src_lang, tgt_lang)
                ).fetchall()
                vecs = []
                tgts = []
                for tgt, blob in rows:
                    if blob:
                        tgts.append(tgt)
                        vecs.append(np.frombuffer(blob, dtype="float32"))
                if not vecs:
                    return []
                M = np.vstack(vecs)
                sims = cosine_sim(np.array([v]), M)[0]
                order = np.argsort(-sims)[:topk]
                return [(tgts[i], float(sims[i])) for i in order]
        return await asyncio.to_thread(_t)

    async def clear_client(self, client_id: str):
        def _t():
            with sqlite3.connect(self.path) as c:
                c.execute("DELETE FROM tm_segments WHERE client_id=?", (client_id,))
        await asyncio.to_thread(_t)
