"""LocalBuddy tools (Phase 2): filesystem, shell, and web fetch.

Read-only tools run freely; everything that writes, deletes, runs a command, or
makes a network request is registered with ``requires_approval=True`` so it goes
through the approval gate (see ``approval.py``).
"""

from __future__ import annotations

from pydantic_ai import Tool

from .base import AgentDeps
from .filesystem import delete_path, list_dir, read_file, write_file
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


def build_tools() -> list[Tool[AgentDeps]]:
    """Construct all Phase-2 tools with their approval requirements."""
    return [
        Tool(read_file, requires_approval=False),
        Tool(list_dir, requires_approval=False),
        Tool(write_file, requires_approval=True),
        Tool(delete_path, requires_approval=True),
        Tool(run_shell, requires_approval=True),
        Tool(fetch_url, requires_approval=True),
    ]


__all__ = ["AgentDeps", "APPROVAL_REQUIRED", "UNSANDBOXED_TOOLS", "build_tools"]
