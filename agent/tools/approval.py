"""The approval gate (Phase 2).

When the model calls a tool registered with ``requires_approval=True``, the run
pauses and returns a ``DeferredToolRequests``. This module turns those pending
requests into a ``DeferredToolResults`` by asking an injected ``Asker`` callback
to approve or deny each call. The callback (and its UI) lives in the REPL, which
keeps this module UI-free and testable.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Callable

from pydantic_ai import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDenied
from pydantic_ai.messages import ToolCallPart

# Decides a single pending tool call: return True to approve, False to deny.
Asker = Callable[[ToolCallPart], Awaitable[bool]]


async def collect_approvals(requests: DeferredToolRequests, ask: Asker) -> DeferredToolResults:
    """Ask about each pending approval and build the results to resume the run."""
    approvals: dict[str, bool | ToolApproved | ToolDenied] = {}
    for call in requests.approvals:
        approved = await ask(call)
        approvals[call.tool_call_id] = (
            ToolApproved() if approved else ToolDenied("The user declined to run this action.")
        )
    return requests.build_results(approvals=approvals)
