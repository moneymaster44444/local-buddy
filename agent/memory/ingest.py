"""Ingest local text/Markdown files into the memory store (Phase 3).

Chunking is character-based with overlap and soft breaks on paragraph/line/
sentence boundaries. Re-ingesting a path replaces that path's prior chunks.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from .store import MemoryStore

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".text"}

# Embeds a list of texts -> a list of (unit-normalized) vectors.
EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]


@dataclass
class IngestResult:
    files: int = 0
    chunks: int = 0
    skipped: list[str] = field(default_factory=list)


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into ~``size``-char chunks overlapping by ``overlap`` chars."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:  # try to end on a clean boundary in the back half of the window
            window = text[start:end]
            brk = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "))
            if brk > size // 2:
                end = start + brk + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def iter_text_files(path: Path) -> list[Path]:
    """A single file is taken as-is; a directory is walked for text extensions."""
    if path.is_file():
        return [path]
    return [
        p for p in sorted(path.rglob("*"))
        if p.is_file() and p.suffix.lower() in TEXT_EXTENSIONS
    ]


async def ingest_path(
    store: MemoryStore,
    embed: EmbedFn,
    path: Path,
    *,
    chunk_chars: int,
    chunk_overlap: int,
) -> IngestResult:
    """Chunk, embed, and store every text file under ``path``."""
    result = IngestResult()
    files = iter_text_files(path)

    # Make re-ingest idempotent: drop any prior chunks for these sources first.
    sources = [p.resolve().as_posix() for p in files]
    if sources:
        await store.delete_sources(sources)

    for file in files:
        try:
            text = file.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            result.skipped.append(str(file))
            continue
        chunks = chunk_text(text, chunk_chars, chunk_overlap)
        if not chunks:
            continue
        vectors = await embed(chunks)
        source = file.resolve().as_posix()
        rows = [
            {"vector": v, "text": c, "source": source, "chunk_index": i}
            for i, (c, v) in enumerate(zip(chunks, vectors))
        ]
        await store.add(rows)
        result.files += 1
        result.chunks += len(rows)
    return result
