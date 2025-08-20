from __future__ import annotations
from typing import List
import numpy as np
from openai import AsyncOpenAI
from ..config import settings
from .llm import get_embed_limiter

_client = AsyncOpenAI()

async def embed_texts(texts: List[str]) -> np.ndarray:
    inputs = [t[:3000] for t in texts]
    async with get_embed_limiter():
        res = await _client.embeddings.create(model=settings.model_embed, input=inputs)
    vecs = [d.embedding for d in res.data]
    return np.array(vecs, dtype="float32")

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.dot(a_norm, b_norm.T)
