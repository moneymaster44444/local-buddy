"""The agent step loop (Phase 2): call -> tool -> observe -> repeat.

A single user turn is driven through ``agent.iter()`` so token streaming, tool
execution, the human-in-the-loop approval gate, and a per-turn iteration cap all
live in one place.

Flow: ``iter()`` runs until it either yields a final text answer or pauses with a
``DeferredToolRequests`` (a tool needs approval). On a pause we ask the user via
the approval gate, then resume the run with the results. Read-only tools execute
inline without pausing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai import Agent, DeferredToolRequests, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
)
from pydantic_ai.usage import RunUsage
from rich.console import Console
from rich.markup import escape

from .state import Conversation
from .tools.approval import Asker, collect_approvals
from .tools.base import AgentDeps

if TYPE_CHECKING:
    from pydantic_ai.models import Model


class _Status:
    """An idempotent, restartable wrapper around ``console.status``."""

    def __init__(self, console: Console, label: str = "[dim]thinking…[/dim]") -> None:
        self._console = console
        self._label = label
        self._status = None

    def start(self) -> None:
        if self._status is None:
            self._status = self._console.status(self._label, spinner="dots")
            self._status.start()

    def stop(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None


def _text_delta(event: object) -> str | None:
    """Extract visible answer text from a model-stream event."""
    if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
        return event.part.content or None
    if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
        return event.delta.content_delta or None
    return None


def _thinking_delta(event: object) -> str | None:
    """Extract reasoning/thinking text from a model-stream event (reasoning models)."""
    if isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart):
        return event.part.content or None
    if isinstance(event, PartDeltaEvent) and isinstance(event.delta, ThinkingPartDelta):
        return event.delta.content_delta or None
    return None


def _short(text: str, limit: int = 140) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def _show_tool_event(console: Console, event: object) -> None:
    """Print a one-line note for tools that actually execute (not deferred ones)."""
    if isinstance(event, FunctionToolCallEvent):
        call = event.part
        args = call.args_as_dict()
        arg_str = ", ".join(f"{k}={_short(str(v), 60)!r}" for k, v in args.items())
        console.print(f"[dim]🔧 {escape(call.tool_name)}({escape(arg_str)})[/dim]")
    elif isinstance(event, FunctionToolResultEvent):
        part = event.part
        marker = "↺" if isinstance(part, RetryPromptPart) else "←"
        console.print(f"[dim]   {marker} {escape(_short(str(part.content)))}[/dim]")


async def run_turn(
    console: Console,
    agent: Agent,
    model: "Model",
    deps: AgentDeps,
    conversation: Conversation,
    user_prompt: str,
    *,
    ask: Asker,
    limits: UsageLimits,
) -> None:
    """Run one user turn to completion, handling tools and approvals."""
    output_type = [str, DeferredToolRequests]
    usage = RunUsage()  # shared across approval-resume legs -> a per-turn cap
    show_thinking = deps.settings.show_thinking
    spinner = _Status(console)
    prompt: str | None = user_prompt
    deferred = None
    streaming = False  # currently on an open answer line?
    thinking = False  # currently streaming a dimmed reasoning block?
    answered = False  # has any answer text been emitted this whole turn?
    # Snapshot so an aborted turn can be rolled back to a clean history (a partial
    # turn can end on unprocessed tool calls, which would break the next prompt).
    messages_at_start = list(conversation.messages)
    completed = False

    def close_line() -> None:
        """End whatever line (answer or thinking) is currently open."""
        nonlocal streaming, thinking
        if streaming or thinking:
            console.print()
            streaming = thinking = False

    def begin_answer() -> None:
        nonlocal streaming, answered
        spinner.stop()
        if thinking:
            close_line()
        if not streaming:
            console.print("[bold green]buddy ›[/bold green] ", end="")
            streaming = True
            answered = True

    def begin_thinking() -> None:
        nonlocal thinking
        spinner.stop()
        if not thinking and not streaming:
            console.print("[dim]🤔 thinking…[/dim]")
            thinking = True

    try:
        while True:
            spinner.start()
            async with agent.iter(
                prompt,
                message_history=conversation.messages,
                model=model,
                deps=deps,
                output_type=output_type,
                deferred_tool_results=deferred,
                usage_limits=limits,
                usage=usage,
            ) as run:
                async for node in run:
                    if Agent.is_model_request_node(node):
                        async with node.stream(run.ctx) as stream:
                            async for event in stream:
                                answer = _text_delta(event)
                                if answer:
                                    begin_answer()
                                    console.print(
                                        answer, end="", markup=False, highlight=False, soft_wrap=True
                                    )
                                    continue
                                if show_thinking and (reasoning := _thinking_delta(event)):
                                    begin_thinking()
                                    console.print(
                                        reasoning, end="", style="dim", markup=False,
                                        highlight=False, soft_wrap=True,
                                    )
                    elif Agent.is_call_tools_node(node):
                        spinner.stop()
                        close_line()
                        async with node.stream(run.ctx) as handle_stream:
                            async for event in handle_stream:
                                _show_tool_event(console, event)

            conversation.messages = run.result.all_messages()
            output = run.result.output

            if isinstance(output, DeferredToolRequests):
                spinner.stop()
                close_line()
                console.print("\n[yellow]⚠ the assistant needs approval to continue:[/yellow]")
                deferred = await collect_approvals(output, ask)
                prompt = None  # resume the same run with the approval results
                continue

            # Final text output.
            spinner.stop()
            if streaming or thinking:
                close_line()
            elif not answered and isinstance(output, str) and output.strip():
                # Final text that never streamed as deltas (rare): print it once.
                console.print(f"[bold green]buddy ›[/bold green] {escape(output)}")
            completed = True
            break
    except UsageLimitExceeded as exc:
        spinner.stop()
        close_line()
        console.print(f"\n[yellow]⏹ iteration cap reached ({escape(str(exc))}). Stopping this turn.[/yellow]")
    except KeyboardInterrupt:
        spinner.stop()
        close_line()
        console.print("\n[yellow]⏹ interrupted — this turn was stopped[/yellow]")
    except Exception as exc:  # noqa: BLE001 - surface errors, keep the REPL alive
        spinner.stop()
        close_line()
        console.print(f"\n[red]Error during response:[/red] {escape(str(exc))}")
    finally:
        spinner.stop()
        if not completed:
            # Roll back a failed/aborted turn so the history doesn't end on
            # unprocessed tool calls (which would block the next user prompt).
            conversation.messages = messages_at_start
