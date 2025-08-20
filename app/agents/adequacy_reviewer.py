from __future__ import annotations
from importlib import resources
from ..prompts.composer import compose_adequacy
from ..services.llm import llm_parse
from ..qa.types import AdequacyOut
from ..config import settings

async def adequacy_review(*, src_lang: str, tgt_lang: str, domain: str,
                          glossary_block: str, source: str, current: str) -> AdequacyOut:
    tpl = resources.files("app.prompts.templates").joinpath("adequacy.j2").read_text(encoding="utf-8")
    prompt = compose_adequacy(tpl, src_lang=src_lang, tgt_lang=tgt_lang, domain=domain,
                              glossary_block=glossary_block, source=source, current=current)
    parsed = await llm_parse(prompt, model=settings.model_review, schema=AdequacyOut, temperature=0.0)
    return parsed
