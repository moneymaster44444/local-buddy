"""Embeddings via LM Studio's OpenAI-compatible /v1/embeddings endpoint (Phase 3).

Vectors are unit-normalized so the store's L2 distance ranks like cosine.
"""

from __future__ import annotations

import numpy as np
from openai import APIConnectionError, AsyncOpenAI, OpenAIError


class EmbeddingError(RuntimeError):
    """Raised when an embedding request to LM Studio fails."""


def _normalize(vec: list[float]) -> list[float]:
    arr = np.asarray(vec, dtype="float32")
    norm = float(np.linalg.norm(arr))
    if norm > 0.0:
        arr = arr / norm
    return arr.tolist()


async def embed_texts(
    client: AsyncOpenAI, model_id: str, texts: list[str], batch_size: int = 32
) -> list[list[float]]:
    """Embed ``texts`` (unit-normalized), in batches of ``batch_size``."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        try:
            response = await client.embeddings.create(model=model_id, input=batch)
        except APIConnectionError as exc:
            raise EmbeddingError(
                f"Could not reach LM Studio to embed text (model '{model_id}'). "
                "Is the server running and the embedding model available?"
            ) from exc
        except OpenAIError as exc:
            raise EmbeddingError(f"Embedding request failed (model '{model_id}'): {exc}") from exc
        vectors.extend(_normalize(item.embedding) for item in response.data)
    return vectors
