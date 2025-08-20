from __future__ import annotations
import sqlite3, asyncio, os
from typing import List, Optional
import yaml
import trafilatura
from duckduckgo_search import DDGS
import numpy as np
from ..config import settings
from ..services.embeddings import embed_texts, cosine_sim

INIT_SQL = """
CREATE TABLE IF NOT EXISTS rag_sources(
  domain TEXT NOT NULL,
  title TEXT,
  url TEXT PRIMARY KEY,
  year INTEGER,
  notes TEXT
);
CREATE TABLE IF NOT EXISTS rag_docs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  domain TEXT,
  client_id TEXT,
  url TEXT,
  title TEXT,
  content TEXT,
  vec BLOB
);
CREATE INDEX IF NOT EXISTS idx_rag_domain ON rag_docs(domain);
"""

class RAGStore:
    def __init__(self, path: str | None = None):
        self.path = path or settings.db_path
        with sqlite3.connect(self.path) as c:
            c.executescript(INIT_SQL)

    async def load_seed_sources(self, yaml_path: str = "data/seed/domains.yaml"):
        if not os.path.exists(yaml_path):
            return 0
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        def _t():
            with sqlite3.connect(self.path) as c:
                for domain, items in data.items():
                    for it in items:
                        c.execute(
                            "INSERT OR IGNORE INTO rag_sources(domain,title,url,year,notes) VALUES(?,?,?,?,?)",
                            (domain, it.get("title"), it.get("url"), it.get("year"), it.get("notes"))
                        )
        await asyncio.to_thread(_t)
        return sum(len(v) for v in data.values())

    async def ingest_urls(self, urls: List[str], domain: str, client_id: Optional[str] = None) -> int:
        texts = []
        metas = []
        for u in urls:
            try:
                html = trafilatura.fetch_url(u, timeout=20, no_ssl=True)
                if not html:
                    continue
                text = trafilatura.extract(html, include_tables=False, include_comments=False)
                if not text:
                    continue
                texts.append(text[:12000])
                metas.append((u, u))
            except Exception:
                continue
        if not texts:
            return 0
        vecs = await embed_texts(texts)
        def _t():
            with sqlite3.connect(self.path) as c:
                for (u, title), vec, content in zip(metas, vecs, texts):
                    c.execute(
                        "INSERT INTO rag_docs(domain,client_id,url,title,content,vec) VALUES(?,?,?,?,?,?)",
                        (domain, client_id, u, title, content, vec.tobytes())
                    )
        await asyncio.to_thread(_t)
        return len(texts)

    async def retrieve(self, query: str, domain: str, client_id: Optional[str], topk: int) -> List[str]:
        qv = await embed_texts([query])
        def _t():
            with sqlite3.connect(self.path) as c:
                rows = c.execute(
                    "SELECT content, vec FROM rag_docs WHERE (domain=? OR (client_id IS NOT NULL AND client_id=?))",
                    (domain, client_id)
                ).fetchall()
                if not rows:
                    return []
                contents = [r[0] for r in rows]
                vecs = [np.frombuffer(r[1], dtype="float32") for r in rows if r[1]]
                if not vecs:
                    return []
                M = np.vstack(vecs)
                sims = cosine_sim(qv, M)[0]
                order = np.argsort(-sims)[:topk]
                return [contents[i][:1800] for i in order]
        return await asyncio.to_thread(_t)

    async def web_backfill_if_empty(self, query_terms: List[str], domain: str, client_id: Optional[str]) -> int:
        urls = []
        try:
            with DDGS() as ddgs:
                for t in query_terms[:6]:
                    res = ddgs.text(f"{t} {domain} finance definition", max_results=3)
                    for r in res:
                        href = r.get("href")
                        if href:
                            urls.append(href)
        except Exception:
            return 0
        if not urls:
            return 0
        return await self.ingest_urls(urls, domain=domain, client_id=client_id)
