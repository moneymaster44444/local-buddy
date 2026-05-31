"""LocalBuddy tools (Phase 2): filesystem, shell, and web fetch.

Read-only tools run freely; everything that writes, deletes, runs a command, or
makes a network request is registered with ``requires_approval=True`` so it goes
through the approval gate (see ``approval.py``).
"""

from __future__ import annotations

from pydantic_ai import Tool

from ..config import Settings
from .base import AgentDeps
from .filesystem import delete_path, list_dir, read_file, write_file
from .memory import search_memory
from .shell import run_shell
from .webfetch import fetch_url

# Tool names that must be approved before they execute.
APPROVAL_REQUIRED: frozenset[str] = frozenset(
    {"write_file", "delete_path", "run_shell", "fetch_url"}
)

# Tools whose reach is NOT confined to the workspace sandbox (a real shell can
# touch any file; a fetch can hit any URL). These must be approved on every call
# and may never be blanket-approved for a session.
UNSANDBOXED_TOOLS: frozenset[str] = frozenset({"run_shell", "fetch_url"})


def build_tools(settings: Settings) -> list[Tool[AgentDeps]]:
    """Construct the tools with their approval requirements.

    Phase 2: filesystem, shell, web fetch. Phase 3 adds the read-only
    ``search_memory`` tool when memory is enabled.
    """
    tools: list[Tool[AgentDeps]] = [
        Tool(read_file, requires_approval=False),
        Tool(list_dir, requires_approval=False),
        Tool(write_file, requires_approval=True),
        Tool(delete_path, requires_approval=True),
        Tool(run_shell, requires_approval=True),
        Tool(fetch_url, requires_approval=True),
    ]
    if settings.enable_memory:
        tools.append(Tool(search_memory, requires_approval=False))
    return tools


__all__ = ["AgentDeps", "APPROVAL_REQUIRED", "UNSANDBOXED_TOOLS", "build_tools"]
