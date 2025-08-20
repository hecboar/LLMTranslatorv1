from __future__ import annotations
from importlib import resources
from ..prompts.composer import compose_fluency
from ..services.llm import llm_parse
from ..qa.types import FluencyOut
from ..config import settings

async def fluency_review(*, tgt_lang: str, domain: str, current: str) -> FluencyOut:
    tpl = resources.files("app.prompts.templates").joinpath("fluency.j2").read_text(encoding="utf-8")
    prompt = compose_fluency(tpl, tgt_lang=tgt_lang, domain=domain, current=current)
    parsed = await llm_parse(prompt, model=settings.model_review, schema=FluencyOut, temperature=0.3)
    return parsed
