from __future__ import annotations
from jinja2 import Environment, BaseLoader
from typing import List

_env = Environment(loader=BaseLoader(), autoescape=False, trim_blocks=True, lstrip_blocks=True)

def render_template(src: str, **kwargs) -> str:
    return _env.from_string(src).render(**kwargs)

def compose_translator(tpl: str, *, src_lang: str, tgt_lang: str, domain: str,
                       style: str, glossary_block: str, dnt: List[str],
                       rag_snippets: List[str], source: str) -> str:
    return render_template(
        tpl, src_lang=src_lang, tgt_lang=tgt_lang, domain=domain, style=style,
        glossary_block=glossary_block, dnt=dnt, rag_snippets=rag_snippets, source=source
    )

def compose_adequacy(tpl: str, *, src_lang: str, tgt_lang: str, domain: str,
                     glossary_block: str, source: str, current: str) -> str:
    return render_template(
        tpl, src_lang=src_lang, tgt_lang=tgt_lang, domain=domain,
        glossary_block=glossary_block, source=source, current=current
    )

def compose_fluency(tpl: str, *, tgt_lang: str, domain: str, current: str) -> str:
    return render_template(tpl, tgt_lang=tgt_lang, domain=domain, current=current)

def compose_editor(tpl: str, *, tgt_lang: str, domain: str, adequacy_text: str, fluency_text: str) -> str:
    return render_template(tpl, tgt_lang=tgt_lang, domain=domain,
                           adequacy_text=adequacy_text, fluency_text=fluency_text)
