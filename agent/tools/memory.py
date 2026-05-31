"""The search_memory tool (Phase 3): read-only retrieval over the knowledge base.

Read-only, so it is registered with ``requires_approval=False``. It only embeds
the query when the knowledge base is non-empty, keeping it VRAM-friendly.
"""

from __future__ import annotations

from pydantic_ai import RunContext

from .base import AgentDeps


async def search_memory(ctx: RunContext[AgentDeps], query: str) -> str:
    """Search the user's ingested knowledge base for passages relevant to the query.

    Use this when the question might be answered by documents the user has added
    (with /ingest). Returns the most relevant passages with their source files.
    """
    memory = ctx.deps.memory
    if memory is None:
        return "Memory is disabled; there is no knowledge base to search."
    if await memory.count() == 0:
        return "The knowledge base is empty. Ask the user to add documents with /ingest <path>."
    if not memory.embedder_model_id:
        return "No embedding model is configured this session; ask the user to run /ingest first."
    hits = await memory.search(query, ctx.deps.settings.rag_top_k)
    if not hits:
        return "No relevant passages were found in the knowledge base."
    blocks = [
        f"[{i}] source={h.source} chunk={h.chunk_index} relevance={h.score:.2f}\n{h.text}"
        for i, h in enumerate(hits, 1)
    ]
    return "\n\n".join(blocks)
