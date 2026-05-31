"""Local RAG memory (Phase 3): chunk -> embed (LM Studio) -> LanceDB -> retrieve."""

from .embeddings import EmbeddingError, embed_texts
from .ingest import IngestResult, chunk_text, ingest_path, iter_text_files
from .search import MemorySearcher
from .store import Hit, MemoryStore

__all__ = [
    "EmbeddingError",
    "embed_texts",
    "IngestResult",
    "chunk_text",
    "ingest_path",
    "iter_text_files",
    "MemorySearcher",
    "Hit",
    "MemoryStore",
]
