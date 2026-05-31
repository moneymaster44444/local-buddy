"""Shared tool infrastructure: injected dependencies, sandbox path resolution,
and output truncation. (Phase 2)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai import ModelRetry

from ..config import Settings

if TYPE_CHECKING:
    from ..memory import MemorySearcher


@dataclass
class AgentDeps:
    """Dependencies injected into every tool call via ``RunContext``."""

    root: Path  # absolute, resolved workspace sandbox root
    settings: Settings
    memory: "MemorySearcher | None" = None  # knowledge base for search_memory (Phase 3)


def resolve_in_root(root: Path, path: str) -> Path:
    """Resolve a model-supplied path and confine it to the workspace sandbox.

    Raises ``ModelRetry`` (fed back to the model) if the path escapes the
    sandbox, so the model can correct itself rather than the run crashing.
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ModelRetry(
            f"Path '{path}' is outside the workspace sandbox ({root}). "
            "Use a path inside the workspace."
        )
    return resolved


def truncate(text: str, limit: int) -> str:
    """Cap tool output that gets sent back to the model."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [output truncated; {len(text) - limit} more characters]"
