from __future__ import annotations
import json, asyncio
from typing import Optional, Type, Any
from pydantic import BaseModel
from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI
from ..config import settings

_client: Optional[AsyncOpenAI] = None
_llm_limiter = AsyncLimiter(settings.llm_rps, time_period=1)
_embed_limiter = AsyncLimiter(settings.embed_rps, time_period=1)

def client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI()
    return _client

async def llm_text(prompt: str, model: str, temperature: float = 0.0) -> str:
    async with _llm_limiter:
        r = await client().responses.create(model=model, input=prompt, temperature=temperature)
    return (r.output_text or "").strip()

async def llm_parse(prompt: str, model: str, schema: Type[BaseModel], temperature: float = 0.0) -> BaseModel:
    async with _llm_limiter:
        r = await client().responses.parse(
            model=model,
            input=prompt,
            temperature=temperature,
            text_format=schema,
        )
    return r.output_parsed  # pydantic instance

def extract_json_obj(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end >= 0:
                return json.loads(text[start:end+1])
        except Exception:
            pass
    return {}

def get_embed_limiter() -> AsyncLimiter:
    return _embed_limiter
