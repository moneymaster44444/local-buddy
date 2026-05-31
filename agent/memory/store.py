"""LanceDB-backed vector store for ingested document chunks (Phase 3).

Each row is ``{vector, text, source, chunk_index}``. Embeddings are stored
unit-normalized, so LanceDB's default (squared-L2) distance ranks the same as
cosine similarity. The synchronous LanceDB calls are wrapped in
``asyncio.to_thread`` so they don't block the event loop.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import lancedb

_TABLE = "chunks"


@dataclass
class Hit:
    text: str
    source: str
    chunk_index: int
    score: float  # ~cosine similarity in [0, 1]


def _escape(value: str) -> str:
    return value.replace("'", "''")


def _table_names(db) -> list[str]:
    """Table names as a plain list (LanceDB 0.33 ``list_tables()`` is paginated)."""
    result = db.list_tables()
    return list(getattr(result, "tables", result))


class MemoryStore:
    """A small async wrapper over a single LanceDB table."""

    def __init__(self, db_path: Path) -> None:
        self._path = Path(db_path)

    def _conn(self):
        # Connect per call: cheap for a local directory, and avoids sharing one
        # LanceDB connection across asyncio.to_thread worker threads.
        self._path.mkdir(parents=True, exist_ok=True)
        return lancedb.connect(str(self._path))

    # --- sync workers (executed via asyncio.to_thread) ---
    def _add_sync(self, rows: list[dict]) -> None:
        db = self._conn()
        if _TABLE in _table_names(db):
            db.open_table(_TABLE).add(rows)
        else:
            db.create_table(_TABLE, data=rows)

    def _delete_sources_sync(self, sources: list[str]) -> None:
        db = self._conn()
        if _TABLE not in _table_names(db):
            return
        table = db.open_table(_TABLE)
        for src in sources:
            table.delete(f"source = '{_escape(src)}'")

    def _search_sync(self, vector: list[float], k: int) -> list[Hit]:
        db = self._conn()
        if _TABLE not in _table_names(db):
            return []
        rows = db.open_table(_TABLE).search(vector).limit(k).to_list()
        hits: list[Hit] = []
        for r in rows:
            dist = float(r.get("_distance", 0.0))
            sim = max(0.0, min(1.0, 1.0 - dist / 2.0))  # unit vecs + sq-L2 -> cosine
            hits.append(Hit(r["text"], r["source"], int(r["chunk_index"]), sim))
        return hits

    def _count_sync(self) -> int:
        db = self._conn()
        return db.open_table(_TABLE).count_rows() if _TABLE in _table_names(db) else 0

    def _sources_sync(self) -> list[str]:
        db = self._conn()
        if _TABLE not in _table_names(db):
            return []
        col = db.open_table(_TABLE).to_arrow().column("source").to_pylist()
        return sorted(set(col))

    def _clear_sync(self) -> int:
        db = self._conn()
        if _TABLE not in _table_names(db):
            return 0
        count = db.open_table(_TABLE).count_rows()
        db.drop_table(_TABLE)
        return count

    # --- async API ---
    async def add(self, rows: list[dict]) -> None:
        await asyncio.to_thread(self._add_sync, rows)

    async def delete_sources(self, sources: list[str]) -> None:
        await asyncio.to_thread(self._delete_sources_sync, sources)

    async def search(self, vector: list[float], k: int) -> list[Hit]:
        return await asyncio.to_thread(self._search_sync, vector, k)

    async def count(self) -> int:
        return await asyncio.to_thread(self._count_sync)

    async def sources(self) -> list[str]:
        return await asyncio.to_thread(self._sources_sync)

    async def clear(self) -> int:
        return await asyncio.to_thread(self._clear_sync)
