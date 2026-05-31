"""Shell tool: run a command in the workspace using the platform's native shell.

Always registered with ``requires_approval=True``. Runs via ``subprocess.run`` on
a worker thread so the event loop stays responsive and we avoid platform-specific
asyncio subprocess pitfalls on Windows.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

from pydantic_ai import RunContext

from ..config import Settings
from .base import AgentDeps, truncate


def _shell_argv(settings: Settings, command: str) -> list[str]:
    """Build the argv that runs `command` in the platform's native shell."""
    if sys.platform == "win32":
        if settings.windows_shell == "powershell":
            return ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]
        comspec = os.environ.get("ComSpec", "cmd.exe")
        return [comspec, "/c", command]
    shell = os.environ.get("SHELL", "/bin/sh")
    return [shell, "-c", command]


async def run_shell(ctx: RunContext[AgentDeps], command: str) -> str:
    """Run a shell command in the workspace directory and return its output. Requires approval."""
    argv = _shell_argv(ctx.deps.settings, command)
    timeout = ctx.deps.settings.shell_timeout
    try:
        proc = await asyncio.to_thread(
            subprocess.run,
            argv,
            cwd=str(ctx.deps.root),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout:.0f}s and was killed."
    except OSError as exc:
        return f"Failed to run command: {exc}"
    combined = (proc.stdout or "") + (proc.stderr or "")
    body = truncate(combined, ctx.deps.settings.max_tool_output_chars).strip()
    header = f"(exit code {proc.returncode})"
    return f"{header}\n{body}" if body else f"{header} — no output"
