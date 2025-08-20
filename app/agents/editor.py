from __future__ import annotations
from importlib import resources
from ..prompts.composer import compose_editor
from ..services.llm import llm_text
from ..config import settings

async def edit_merge(*, tgt_lang: str, domain: str, adequacy_text: str, fluency_text: str) -> str:
    tpl = resources.files("app.prompts.templates").joinpath("editor.j2").read_text(encoding="utf-8")
    prompt = compose_editor(tpl, tgt_lang=tgt_lang, domain=domain,
                            adequacy_text=adequacy_text, fluency_text=fluency_text)
    out = await llm_text(prompt, model=settings.model_review, temperature=0.2)
    return out.strip()
