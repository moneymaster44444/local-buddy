"""Filesystem tools, scoped to the workspace sandbox (Phase 2).

``read_file`` and ``list_dir`` are read-only and run without approval.
``write_file`` and ``delete_path`` mutate the filesystem and are registered with
``requires_approval=True`` in ``tools/__init__.py``.
"""

from __future__ import annotations

from pydantic_ai import ModelRetry, RunContext

from .base import AgentDeps, resolve_in_root, truncate


async def read_file(ctx: RunContext[AgentDeps], path: str) -> str:
    """Read a UTF-8 text file. `path` is relative to the workspace root."""
    target = resolve_in_root(ctx.deps.root, path)
    if not target.exists():
        raise ModelRetry(f"File not found: {path}")
    if not target.is_file():
        raise ModelRetry(f"Not a file: {path}")
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ModelRetry(f"Could not read {path}: {exc}") from exc
    return truncate(text, ctx.deps.settings.max_tool_output_chars)


async def list_dir(ctx: RunContext[AgentDeps], path: str = ".") -> str:
    """List entries in a workspace directory (relative path; defaults to the root)."""
    target = resolve_in_root(ctx.deps.root, path)
    if not target.exists():
        raise ModelRetry(f"Directory not found: {path}")
    if not target.is_dir():
        raise ModelRetry(f"Not a directory: {path}")
    entries: list[str] = []
    for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        rel = child.relative_to(ctx.deps.root)
        if child.is_dir():
            entries.append(f"[dir]  {rel}/")
        else:
            entries.append(f"[file] {rel} ({child.stat().st_size} bytes)")
    listing = "\n".join(entries) if entries else "(empty directory)"
    return truncate(listing, ctx.deps.settings.max_tool_output_chars)


async def write_file(ctx: RunContext[AgentDeps], path: str, content: str) -> str:
    """Create or overwrite a UTF-8 text file in the workspace. Requires approval."""
    target = resolve_in_root(ctx.deps.root, path)
    if target.is_dir():
        raise ModelRetry(f"Cannot write: {path} is a directory.")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise ModelRetry(f"Could not write {path}: {exc}") from exc
    return f"Wrote {len(content)} characters to {target.relative_to(ctx.deps.root)}."


async def delete_path(ctx: RunContext[AgentDeps], path: str) -> str:
    """Delete a file or an empty directory in the workspace. Requires approval."""
    target = resolve_in_root(ctx.deps.root, path)
    if target == ctx.deps.root:
        raise ModelRetry("Refusing to delete the workspace root itself.")
    if not target.exists():
        raise ModelRetry(f"Nothing to delete at {path}.")
    try:
        if target.is_dir():
            target.rmdir()  # only removes empty directories, by design
        else:
            target.unlink()
    except OSError as exc:
        raise ModelRetry(f"Could not delete {path}: {exc} (directories must be empty)") from exc
    return f"Deleted {path}."
