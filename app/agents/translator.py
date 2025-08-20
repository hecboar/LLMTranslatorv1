from __future__ import annotations
from typing import List
from importlib import resources
from ..prompts.composer import compose_translator
from ..services.llm import llm_text
from ..config import settings

async def translate_text(*, src_lang: str, tgt_lang: str, domain: str,
                         style: str, glossary_block: str, dnt: List[str],
                         rag_snippets: List[str], source: str) -> str:
    tpl = resources.files("app.prompts.templates").joinpath("translator.j2").read_text(encoding="utf-8")
    prompt = compose_translator(tpl, src_lang=src_lang, tgt_lang=tgt_lang, domain=domain,
                                style=style, glossary_block=glossary_block, dnt=dnt,
                                rag_snippets=rag_snippets, source=source)
    out = await llm_text(prompt, model=settings.model_translate, temperature=0.2)
    return out
