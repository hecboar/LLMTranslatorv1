# app/services/llm.py
from __future__ import annotations
import json, asyncio, logging
from time import perf_counter
from typing import Optional, Type, Any
from pydantic import BaseModel
from aiolimiter import AsyncLimiter
from openai import AsyncOpenAI
from ..config import settings
from ..telemetry import trace

_client: Optional[AsyncOpenAI] = None
_llm_limiter = AsyncLimiter(settings.llm_rps, time_period=1)
_embed_limiter = AsyncLimiter(settings.embed_rps, time_period=1)

# Logging
log = logging.getLogger(__name__)
MAX_LOG_CHARS = 1800  # cap logged prompt/response size to avoid huge logs

def _snip(s: str | None, n: int = MAX_LOG_CHARS) -> str:
    if not s:
        return ""
    s = str(s)
    return s if len(s) <= n else (s[:n] + " …[truncated]")

def client() -> AsyncOpenAI:
    """
    Lazy-initialize the OpenAI async client.
    """
    global _client
    if _client is None:
        _client = AsyncOpenAI()
    return _client

async def llm_text(prompt: str, model: str, temperature: float = 0.0) -> str:
    """
    Fire-and-return text generation helper with rate limiting, JSON logs, and trace events.
    """
    t0 = perf_counter()
    req_meta = {
        "kind": "llm.request",
        "op": "responses.create",
        "model": model,
        "temperature": temperature,
        "prompt_len": len(prompt or ""),
        "prompt_snip": _snip(prompt),
    }
    if settings.trace_prompts:
        trace.log(**req_meta)
    log.info(json.dumps(req_meta, ensure_ascii=False))

    try:
        async with _llm_limiter:
            r = await client().responses.create(
                model=model,
                input=prompt,
                temperature=temperature,
            )
        out_text = (getattr(r, "output_text", None) or "").strip()
        usage = getattr(r, "usage", None)
        in_tok = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
        out_tok = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
        rid = getattr(r, "id", None)

        t1 = perf_counter()
        resp_meta = {
            "kind": "llm.response",
            "op": "responses.create",
            "model": model,
            "request_id": rid,
            "latency_ms": round((t1 - t0) * 1000, 1),
            "in_tokens": in_tok,
            "out_tokens": out_tok,
            "response_len": len(out_text),
            "response_snip": _snip(out_text),
        }
        if settings.trace_prompts:
            trace.log(**resp_meta)
        log.info(json.dumps(resp_meta, ensure_ascii=False))
        return out_text
    except Exception as e:
        err_meta = {
            "kind": "llm.error",
            "op": "responses.create",
            "model": model,
            "error": repr(e),
        }
        if settings.trace_prompts:
            trace.log(**err_meta)
        log.exception(json.dumps(err_meta, ensure_ascii=False))
        raise

async def llm_parse(prompt: str, model: str, schema: Type[BaseModel], temperature: float = 0.0) -> BaseModel:
    """
    Structured-parse helper (Pydantic schema) with rate limiting, JSON logs, and trace events.
    """
    t0 = perf_counter()
    req_meta = {
        "kind": "llm.request",
        "op": "responses.parse",
        "model": model,
        "schema": getattr(schema, "__name__", str(schema)),
        "temperature": temperature,
        "prompt_len": len(prompt or ""),
        "prompt_snip": _snip(prompt),
    }
    if settings.trace_prompts:
        trace.log(**req_meta)
    log.info(json.dumps(req_meta, ensure_ascii=False))

    try:
        async with _llm_limiter:
            r = await client().responses.parse(
                model=model,
                input=prompt,
                temperature=temperature,
                text_format=schema,
            )
        parsed = getattr(r, "output_parsed", None)
        # Best-effort preview of raw output_text for logs (not used by caller)
        out_text = (getattr(r, "output_text", None) or "").strip()
        usage = getattr(r, "usage", None)
        in_tok = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", None)
        out_tok = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", None)
        rid = getattr(r, "id", None)

        t1 = perf_counter()
        resp_meta = {
            "kind": "llm.response",
            "op": "responses.parse",
            "model": model,
            "request_id": rid,
            "latency_ms": round((t1 - t0) * 1000, 1),
            "in_tokens": in_tok,
            "out_tokens": out_tok,
            "parsed_type": type(parsed).__name__ if parsed is not None else None,
            "raw_text_snip": _snip(out_text),
        }
        if settings.trace_prompts:
            trace.log(**resp_meta)
        log.info(json.dumps(resp_meta, ensure_ascii=False))

        return parsed  # pydantic instance
    except Exception as e:
        err_meta = {
            "kind": "llm.error",
            "op": "responses.parse",
            "model": model,
            "schema": getattr(schema, "__name__", str(schema)),
            "error": repr(e),
        }
        if settings.trace_prompts:
            trace.log(**err_meta)
        log.exception(json.dumps(err_meta, ensure_ascii=False))
        raise

def extract_json_obj(text: str) -> Any:
    """
    Utility: extract a JSON object from a possibly noisy string.
    """
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
