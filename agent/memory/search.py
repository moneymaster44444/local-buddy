"""Query-time retrieval: embed a query and search the store (Phase 3).

This object is injected onto ``AgentDeps`` so the read-only ``search_memory``
tool can use it. The embedder model id is resolved lazily (it may be set after
construction, e.g. on the first ``/ingest``).
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import AsyncOpenAI

from ..config import Settings
from .embeddings import embed_texts
from .store import Hit, MemoryStore


@dataclass
class MemorySearcher:
    store: MemoryStore
    client: AsyncOpenAI
    settings: Settings
    embedder_model_id: str | None = None

    async def count(self) -> int:
        return await self.store.count()

    async def search(self, query: str, k: int) -> list[Hit]:
        if not self.embedder_model_id:
            return []
        vectors = await embed_texts(
            self.client, self.embedder_model_id, [query], self.settings.embed_batch_size
        )
        return await self.store.search(vectors[0], k)
