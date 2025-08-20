from __future__ import annotations
from contextvars import ContextVar
from typing import Any, Dict, List
from time import time

_trace: ContextVar[Dict[str, Any] | None] = ContextVar("trace_ctx", default=None)

def start(trace_id: str):
    _trace.set({"trace_id": trace_id, "events": []})

def log(kind: str, **data):
    ctx = _trace.get()
    if ctx is not None:
        ctx["events"].append({"ts": time(), "kind": kind, **data})

def get() -> Dict[str, Any]:
    return _trace.get() or {"trace_id": None, "events": []}
