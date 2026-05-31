"""Interactive streaming chat REPL for LocalBuddy (Phase 2).

Brings together config, the LM Studio clients, the in-memory conversation, and
the tool loop:
- streams the brain model's replies token-by-token with ``rich``
- runs filesystem / shell / web-fetch tools through the agent loop (``loop.py``)
- gates risky tool calls behind an interactive approval prompt
- reads input with ``prompt_toolkit`` (history, multiline via Alt+Enter)
- supports ``/help``, ``/model``, ``/tools``, ``/clear``, ``/exit``
- condenses old turns via the utility model when the history grows too large
"""

from __future__ import annotations

import asyncio
import json
import signal
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from openai import AsyncOpenAI
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.messages import ToolCallPart
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from .config import Settings
from .llm import (
    LMStudioError,
    build_brain_agent,
    build_model,
    build_utility_agent,
    list_models,
    make_client,
)
from .loop import run_turn
from .memory import (
    EmbeddingError,
    MemorySearcher,
    MemoryStore,
    embed_texts,
    ingest_path,
    ingest_text,
)
from .state import Conversation
from .tools import UNSANDBOXED_TOOLS, AgentDeps, build_tools
from .tools.approval import Asker

if TYPE_CHECKING:
    from pydantic_ai.models import Model


@dataclass
class AppState:
    """Mutable runtime state for one REPL session (so commands can swap models)."""

    settings: Settings
    client: AsyncOpenAI
    available: list[str]
    brain_id: str
    utility_id: str
    brain_model: "Model"
    utility_model: "Model"
    brain_agent: Agent
    utility_agent: Agent
    conversation: Conversation
    deps: AgentDeps  # workspace sandbox + settings + memory, injected into tools
    memory: MemorySearcher | None = None  # knowledge base (Phase 3); None if disabled
    allowed_tools: set[str] = field(default_factory=set)  # auto-approved this session


# --- input handling ---------------------------------------------------------


def _build_key_bindings() -> KeyBindings:
    """Enter submits; Alt+Enter (Esc then Enter) inserts a newline."""
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event) -> None:  # noqa: ANN001 - prompt_toolkit callback
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    def _newline(event) -> None:  # noqa: ANN001
        event.current_buffer.insert_text("\n")

    return kb


# --- rendering helpers ------------------------------------------------------


def print_help(console: Console) -> None:
    table = Table(title="LocalBuddy commands", title_style="bold", header_style="bold")
    table.add_column("Command")
    table.add_column("Description")
    table.add_row("/help", "Show this help")
    table.add_row("/model", "Show the current brain & utility models and all available")
    table.add_row("/model brain [id|#]", "Switch the brain (reasoning) model — picker if no id given")
    table.add_row("/model utility [id|#]", "Switch the utility (summary) model — picker if no id given")
    table.add_row("/tools [reset|revoke <tool>]", "List tools & approvals; revoke one or reset all session auto-approvals")
    table.add_row("/ingest <path>", "Add a text/Markdown file or folder to the knowledge base (RAG)")
    table.add_row("/remember [text]", "Save a note (or, with no text, a summary of this conversation) to memory")
    table.add_row("/memory", "Show knowledge base stats (chunks, sources, embedder)")
    table.add_row("/forget", "Clear the knowledge base")
    table.add_row("/clear", "Clear the conversation history")
    table.add_row("/exit, /quit", "Leave LocalBuddy")
    console.print(table)
    console.print(
        "[dim]Enter sends • Alt+Enter inserts a newline • Ctrl+C cancels a reply • Ctrl+D exits[/dim]"
    )
    console.print(
        "[dim]Risky tool calls prompt: y / N / a (always this session). Choosing 'a' for shell or "
        "fetch needs a typed 'yes' confirmation, since they reach outside the sandbox. /tools reset clears it.[/dim]"
    )


def print_models(console: Console, state: AppState) -> None:
    table = Table(title="Models in LM Studio", header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Model id")
    table.add_column("Role")
    for i, model_id in enumerate(state.available, 1):
        roles = []
        if model_id == state.brain_id:
            roles.append("[green]brain[/green]")
        if model_id == state.utility_id:
            roles.append("[cyan]utility[/cyan]")
        table.add_row(str(i), model_id, " ".join(roles))
    console.print(table)


# --- model selection --------------------------------------------------------


async def pick_model(
    console: Console,
    session: PromptSession,
    role: str,
    available: list[str],
    current: str | None,
) -> str:
    """Interactively choose a model for ``role`` from the available list."""
    console.print(f"\n[bold]Select the [green]{role}[/green] model:[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right")
    table.add_column("Model id")
    table.add_column("")
    for i, model_id in enumerate(available, 1):
        marker = "[dim](current)[/dim]" if model_id == current else ""
        table.add_row(str(i), model_id, marker)
    console.print(table)

    hint = " (Enter to keep current)" if current else ""
    while True:
        choice = (await session.prompt_async(f"{role} model #/id{hint}: ")).strip()
        if not choice:
            if current:
                return current
            console.print("[red]Please choose a model.[/red]")
            continue
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(available):
                return available[index - 1]
            console.print(f"[red]Enter a number between 1 and {len(available)}.[/red]")
            continue
        if choice in available:
            return choice
        console.print(f"[red]'{choice}' is not an available model id.[/red]")


async def resolve_model(
    console: Console,
    session: PromptSession,
    role: str,
    configured: str | None,
    available: list[str],
) -> str:
    """Use the configured model id if valid, else fall back to the picker."""
    if configured:
        if configured in available:
            console.print(f"[dim]{role}:[/dim] using configured model [bold]{configured}[/bold]")
            return configured
        console.print(
            f"[yellow]Configured {role} model '{configured}' is not loaded in LM Studio.[/yellow]"
        )
    if len(available) == 1:
        only = available[0]
        console.print(f"[dim]{role}:[/dim] only one model available, using [bold]{only}[/bold]")
        return only
    return await pick_model(console, session, role, available, configured if configured in available else None)


def _apply_model(state: AppState, role: str, model_id: str) -> None:
    model = build_model(model_id, state.client)
    if role == "brain":
        state.brain_id = model_id
        state.brain_model = model
    else:
        state.utility_id = model_id
        state.utility_model = model


# --- command handling -------------------------------------------------------


async def handle_command(
    console: Console, session: PromptSession, state: AppState, line: str
) -> bool:
    """Dispatch a /command. Returns True if the REPL should exit."""
    parts = line.split()
    cmd = parts[0].lower()
    args = parts[1:]

    if cmd in ("/exit", "/quit"):
        return True
    if cmd == "/help":
        print_help(console)
        return False
    if cmd == "/clear":
        state.conversation.clear()
        console.print("[dim]Conversation history cleared.[/dim]")
        return False
    if cmd == "/model":
        await _handle_model_command(console, session, state, args)
        return False
    if cmd == "/tools":
        sub = args[0].lower() if args else ""
        if sub == "reset":
            if state.allowed_tools:
                cleared = ", ".join(sorted(state.allowed_tools))
                state.allowed_tools.clear()
                console.print(f"[dim]Revoked all session auto-approvals ({cleared}).[/dim]")
            else:
                console.print("[dim]No session auto-approvals to clear.[/dim]")
        elif sub == "revoke":
            if len(args) < 2:
                console.print("[red]Usage:[/red] /tools revoke <tool>  (e.g. /tools revoke run_shell)")
            else:
                name = args[1]
                if name in state.allowed_tools:
                    state.allowed_tools.discard(name)
                    console.print(f"[dim]Revoked auto-approval for {name}; it will prompt again.[/dim]")
                else:
                    console.print(f"[dim]{name} is not currently auto-approved this session.[/dim]")
        else:
            print_tools(console, state)
        return False
    if cmd == "/ingest":
        rest = line.split(maxsplit=1)
        await _handle_ingest(console, session, state, rest[1].strip() if len(rest) > 1 else "")
        return False
    if cmd == "/remember":
        rest = line.split(maxsplit=1)
        await _handle_remember(console, session, state, rest[1].strip() if len(rest) > 1 else "")
        return False
    if cmd == "/memory":
        await _print_memory(console, state)
        return False
    if cmd == "/forget":
        await _handle_forget(console, state)
        return False

    console.print(f"[red]Unknown command:[/red] {cmd}  [dim](try /help)[/dim]")
    return False


async def _handle_model_command(
    console: Console, session: PromptSession, state: AppState, args: list[str]
) -> None:
    if not args:
        print_models(console, state)
        return

    role = args[0].lower()
    if role not in ("brain", "utility"):
        console.print("[red]Usage:[/red] /model [brain|utility] [id|#]")
        return

    current = state.brain_id if role == "brain" else state.utility_id

    if len(args) >= 2:
        choice = args[1]
        new_id: str | None = None
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(state.available):
                new_id = state.available[index - 1]
        elif choice in state.available:
            new_id = choice
        if new_id is None:
            console.print(f"[red]'{choice}' is not an available model.[/red] Use /model to list them.")
            return
    else:
        new_id = await pick_model(console, session, role, state.available, current)

    _apply_model(state, role, new_id)
    console.print(f"[green]{role}[/green] model set to [bold]{new_id}[/bold]")


# --- approval gate UI -------------------------------------------------------


_ACTION_TITLES = {
    "write_file": "Write file",
    "delete_path": "Delete path",
    "run_shell": "Run shell command",
    "fetch_url": "Fetch URL",
}


def _render_action(console: Console, call: ToolCallPart) -> None:
    """Show the exact pending action and its arguments before asking to approve."""
    args = call.args_as_dict()
    lines: list[str] = []
    for key, value in args.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        if key == "content" and len(text) > 600:
            text = text[:600] + f"\n… (+{len(text) - 600} more chars)"
        lines.append(f"[bold]{escape(key)}[/bold]: {escape(text)}")
    body = "\n".join(lines) if lines else "[dim](no arguments)[/dim]"
    if call.tool_name in UNSANDBOXED_TOOLS:
        body += (
            "\n\n[red]⚠ This runs OUTSIDE the workspace sandbox — it can reach any "
            "file/URL on your machine. Review it carefully.[/red]"
        )
    title = _ACTION_TITLES.get(call.tool_name, call.tool_name)
    console.print(
        Panel(
            body,
            title=f"⚠ approval required — [bold]{escape(title)}[/bold]",
            border_style="yellow",
            expand=False,
        )
    )


def make_approver(console: Console, session: PromptSession, allowed: set[str]) -> Asker:
    """Build the approval callback the agent loop uses to gate risky tool calls."""

    async def ask(call: ToolCallPart) -> bool:
        if call.tool_name in allowed:
            console.print(f"[dim]↳ auto-approved {call.tool_name} (allowed this session)[/dim]")
            return True
        _render_action(console, call)
        # 'a' (blanket-allow this session) is available for every tool, but for the
        # unsandboxed ones (shell, fetch) it requires a typed confirmation, since it
        # hands the agent unrestricted reach for the rest of the session.
        needs_confirm = call.tool_name in UNSANDBOXED_TOOLS
        while True:
            try:
                choice = (await session.prompt_async(f"Approve {call.tool_name}? [y/N/a] ")).strip().lower()
            except (EOFError, KeyboardInterrupt):
                console.print("[dim](treated as no)[/dim]")
                return False
            if choice in ("y", "yes"):
                return True
            if choice in ("", "n", "no"):
                return False
            if choice in ("a", "always"):
                if needs_confirm and not await _confirm_blanket(console, session, call.tool_name):
                    console.print("[dim]Blanket approval cancelled — answer y or n for this one call.[/dim]")
                    continue
                allowed.add(call.tool_name)
                console.print(
                    f"[dim]Auto-approving {call.tool_name} for the rest of this session "
                    "(clear with /tools reset).[/dim]"
                )
                return True
            console.print("[dim]Answer y (yes), n (no), or a (always allow this tool this session).[/dim]")

    return ask


async def _confirm_blanket(console: Console, session: PromptSession, tool_name: str) -> bool:
    """Second 'are you sure?' gate before blanket-allowing an unsandboxed tool."""
    console.print(
        f"[red]⚠ This will auto-approve EVERY future {tool_name} call this session with no "
        "further prompts — including actions OUTSIDE the workspace sandbox (any file, command, "
        "or URL). The agent could act unattended.[/red]"
    )
    try:
        confirm = (await session.prompt_async(f"Type 'yes' to confirm always-allow {tool_name}: ")).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return confirm == "yes"


def print_tools(console: Console, state: AppState) -> None:
    if not state.settings.enable_tools:
        console.print("[dim]Tools are disabled (set LOCALBUDDY_ENABLE_TOOLS=true to enable).[/dim]")
        return
    table = Table(title="Tools", header_style="bold")
    table.add_column("Tool")
    table.add_column("Approval")
    table.add_column("Description")
    for tool in build_tools(state.settings):
        name = tool.function.__name__
        if name in state.allowed_tools:
            approval = "[dim]auto (session)[/dim]"
        elif tool.requires_approval:
            approval = "[yellow]required[/yellow]"
        else:
            approval = "[green]none[/green]"
        doc = tool.function.__doc__.strip().splitlines()[0] if tool.function.__doc__ else ""
        table.add_row(name, approval, doc)
    console.print(table)
    console.print(f"[dim]workspace sandbox: {state.deps.root}[/dim]")
    if state.allowed_tools:
        granted = ", ".join(sorted(state.allowed_tools))
        console.print(
            f"[dim]auto-approved this session: {granted} — downshift with "
            "/tools revoke <tool>, or /tools reset for all.[/dim]"
        )


# --- memory / RAG commands (Phase 3) ----------------------------------------


def _auto_embedder(configured: str | None, available: list[str]) -> str | None:
    """Resolve an embedder id without prompting: configured, else a lone 'embed' model."""
    if configured and configured in available:
        return configured
    candidates = [m for m in available if "embed" in m.lower()]
    return candidates[0] if len(candidates) == 1 else None


async def _ensure_embedder(console: Console, session: PromptSession, state: AppState) -> bool:
    """Ensure an embedder model is selected; auto-detect or pick interactively."""
    mem = state.memory
    if mem is None:
        console.print("[yellow]Memory is disabled (LOCALBUDDY_ENABLE_MEMORY=false).[/yellow]")
        return False
    if mem.embedder_model_id:
        return True
    auto = _auto_embedder(state.settings.embedder_model_id, state.available)
    if auto:
        mem.embedder_model_id = auto
        console.print(f"[dim]embedder: using [bold]{auto}[/bold][/dim]")
        return True
    try:
        mem.embedder_model_id = await pick_model(console, session, "embedder", state.available, None)
    except (EOFError, KeyboardInterrupt):
        console.print("[dim](cancelled)[/dim]")
        return False
    return True


async def _handle_ingest(
    console: Console, session: PromptSession, state: AppState, raw_path: str
) -> None:
    mem = state.memory
    if mem is None:
        console.print("[yellow]Memory is disabled (LOCALBUDDY_ENABLE_MEMORY=false).[/yellow]")
        return
    if not raw_path:
        console.print("[red]Usage:[/red] /ingest <path>   (relative paths are taken from the workspace)")
        return
    # Relative paths resolve against the workspace (where the agent's write_file
    # tool puts files); absolute paths are used as-is so you can index anything.
    workspace = state.settings.workspace_path()
    candidate = Path(raw_path.strip().strip('"').strip("'")).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    path = candidate.resolve()
    if not path.exists():
        console.print(f"[red]Path not found:[/red] {path}")
        console.print(
            f"[dim]Relative paths resolve inside the workspace ({workspace}); "
            "use an absolute path for files elsewhere.[/dim]"
        )
        return
    if not await _ensure_embedder(console, session, state):
        return

    async def embed(texts: list[str]) -> list[list[float]]:
        return await embed_texts(state.client, mem.embedder_model_id, texts, state.settings.embed_batch_size)

    try:
        with console.status(f"[dim]ingesting {path}…[/dim]"):
            result = await ingest_path(
                mem.store, embed, path,
                chunk_chars=state.settings.chunk_chars,
                chunk_overlap=state.settings.chunk_overlap,
            )
    except EmbeddingError as exc:
        console.print(f"[red]Ingestion failed:[/red] {exc}")
        return

    if result.chunks == 0:
        console.print("[yellow]Nothing ingested[/yellow] — no supported text files, or all empty/undecodable.")
    else:
        console.print(f"[green]Ingested[/green] {result.chunks} chunk(s) from {result.files} file(s).")
    if result.skipped:
        console.print(f"[dim]skipped {len(result.skipped)} undecodable/binary file(s).[/dim]")
    console.print(f"[dim]knowledge base now holds {await mem.count()} chunk(s).[/dim]")


_REMEMBER_PROMPT = """\
Summarize the following conversation into a concise note for long-term memory.
Capture the key facts, decisions, the user's preferences, names, identifiers, and
any conclusions, written as standalone notes that will still make sense later,
out of context. Do not add anything that is not in the conversation.

--- CONVERSATION ---
{transcript}
--- END ---

Memory note:"""


async def _handle_remember(
    console: Console, session: PromptSession, state: AppState, text: str
) -> None:
    mem = state.memory
    if mem is None:
        console.print("[yellow]Memory is disabled (LOCALBUDDY_ENABLE_MEMORY=false).[/yellow]")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if text:
        content, source, label = text, f"note:{stamp}", "note"
    else:
        if not state.conversation.messages:
            console.print("[yellow]Nothing to remember yet — the conversation is empty.[/yellow]")
            return
        try:
            with console.status("[dim]summarizing the conversation…[/dim]"):
                result = await state.utility_agent.run(
                    _REMEMBER_PROMPT.format(transcript=state.conversation.transcript()),
                    model=state.utility_model,
                    model_settings={"temperature": 0.2, "max_tokens": state.settings.summary_max_tokens},
                )
        except Exception as exc:  # noqa: BLE001 - surface, don't crash the REPL
            console.print(f"[red]Could not summarize the conversation:[/red] {exc}")
            return
        content = result.output.strip()
        source, label = f"conversation:{stamp}", "conversation summary"
        console.print(Panel(escape(content), title="remembering this summary", border_style="cyan", expand=False))

    if not content.strip():
        console.print("[yellow]Nothing to remember (empty content).[/yellow]")
        return
    if not await _ensure_embedder(console, session, state):
        return

    async def embed(texts: list[str]) -> list[list[float]]:
        return await embed_texts(state.client, mem.embedder_model_id, texts, state.settings.embed_batch_size)

    try:
        with console.status("[dim]saving to the knowledge base…[/dim]"):
            res = await ingest_text(
                mem.store, embed, content, source,
                chunk_chars=state.settings.chunk_chars,
                chunk_overlap=state.settings.chunk_overlap,
            )
    except EmbeddingError as exc:
        console.print(f"[red]Could not save:[/red] {exc}")
        return
    console.print(
        f"[green]Remembered[/green] {label} as [bold]{source}[/bold] "
        f"({res.chunks} chunk(s)). Knowledge base now holds {await mem.count()} chunk(s)."
    )


async def _handle_forget(console: Console, state: AppState) -> None:
    mem = state.memory
    if mem is None:
        console.print("[yellow]Memory is disabled.[/yellow]")
        return
    cleared = await mem.store.clear()
    console.print(f"[dim]Cleared the knowledge base ({cleared} chunk(s) removed).[/dim]")


async def _print_memory(console: Console, state: AppState) -> None:
    mem = state.memory
    if mem is None:
        console.print("[yellow]Memory is disabled (LOCALBUDDY_ENABLE_MEMORY=false).[/yellow]")
        return
    count = await mem.count()
    embedder = mem.embedder_model_id or "(auto-select on first /ingest)"
    console.print(f"[bold]Knowledge base:[/bold] {count} chunk(s)  •  embedder: [bold]{embedder}[/bold]")
    if count:
        table = Table(title="Ingested sources", header_style="bold")
        table.add_column("#", justify="right")
        table.add_column("source")
        for i, src in enumerate(await mem.store.sources(), 1):
            table.add_row(str(i), src)
        console.print(table)
    console.print(
        f"[dim]/ingest <path> to add • /forget to clear • stored in {state.settings.lancedb_dir}[/dim]"
    )


async def run_turn_interruptible(console: Console, coro) -> None:
    """Run one turn as a task so Ctrl+C cancels just the turn, not the whole app.

    During streaming (outside prompt_toolkit) a Ctrl+C SIGINT would otherwise
    propagate through ``asyncio.run()`` and exit the program. We temporarily
    route SIGINT to cancel only the current turn's task and keep the REPL alive.
    Approval prompts are handled by prompt_toolkit, which manages Ctrl+C itself.
    """
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(coro)

    def _cancel(_signum, _frame) -> None:
        loop.call_soon_threadsafe(task.cancel)

    previous = signal.signal(signal.SIGINT, _cancel)
    try:
        await task
    except asyncio.CancelledError:
        console.print("\n[yellow]⏹ interrupted — this turn was stopped[/yellow]")
    finally:
        signal.signal(signal.SIGINT, previous)


# --- bootstrap + loop -------------------------------------------------------


async def _run() -> int:
    console = Console()
    settings = Settings()
    settings.ensure_dirs()

    console.print(
        Panel.fit(
            "[bold]LocalBuddy[/bold] — local agent via LM Studio  [dim](Phase 3: tools + memory)[/dim]",
            border_style="green",
        )
    )

    client = make_client(settings)
    try:
        with console.status("[dim]Connecting to LM Studio…[/dim]"):
            available = await list_models(client)
    except LMStudioError as exc:
        console.print(f"[red]✗ {exc}[/red]")
        console.print(f"[dim]Configured base_url: {settings.base_url}[/dim]")
        await client.close()
        return 1

    if not available:
        console.print("[red]✗ No models are loaded in LM Studio.[/red] Load a model and try again.")
        await client.close()
        return 1

    console.print(f"[dim]Found {len(available)} model(s) in LM Studio.[/dim]")

    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(settings.history_file)),
        multiline=True,
        key_bindings=_build_key_bindings(),
    )

    try:
        brain_id = await resolve_model(console, session, "brain", settings.brain_model_id, available)
        utility_id = await resolve_model(console, session, "utility", settings.utility_model_id, available)
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]Setup cancelled.[/dim]")
        await client.close()
        return 130

    workspace = settings.workspace_path() if settings.enable_tools else settings.workspace_root

    memory: MemorySearcher | None = None
    if settings.enable_memory:
        memory = MemorySearcher(
            store=MemoryStore(settings.lancedb_dir),
            client=client,
            settings=settings,
            embedder_model_id=_auto_embedder(settings.embedder_model_id, available),
        )

    state = AppState(
        settings=settings,
        client=client,
        available=available,
        brain_id=brain_id,
        utility_id=utility_id,
        brain_model=build_model(brain_id, client),
        utility_model=build_model(utility_id, client),
        brain_agent=build_brain_agent(settings),
        utility_agent=build_utility_agent(),
        conversation=Conversation(settings=settings),
        deps=AgentDeps(root=workspace, settings=settings, memory=memory),
        memory=memory,
    )

    limits = UsageLimits(
        request_limit=settings.max_model_requests,
        tool_calls_limit=settings.max_tool_calls,
    )
    approver = make_approver(console, session, state.allowed_tools)

    console.print(
        f"\n[green]brain[/green]=[bold]{brain_id}[/bold]   [cyan]utility[/cyan]=[bold]{utility_id}[/bold]"
    )
    if settings.enable_tools:
        console.print(f"[dim]tools enabled • workspace sandbox: {workspace}[/dim]")
    else:
        console.print("[dim]tools disabled[/dim]")
    if settings.enable_memory and memory is not None:
        emb = memory.embedder_model_id or "auto-select on first /ingest"
        console.print(f"[dim]memory enabled • embedder: {emb} • /ingest <path> to add docs[/dim]")
    console.print("[dim]Type /help for commands. Enter sends • Alt+Enter for a newline.[/dim]")

    try:
        while True:
            try:
                line = await session.prompt_async(HTML("\n<ansicyan><b>you ›</b></ansicyan> "))
            except KeyboardInterrupt:
                console.print("[dim](type /exit to quit)[/dim]")
                continue
            except EOFError:
                break

            line = line.strip()
            if not line:
                continue

            if line.startswith("/"):
                try:
                    should_exit = await handle_command(console, session, state, line)
                except (EOFError, KeyboardInterrupt):
                    console.print("[dim](cancelled)[/dim]")
                    continue
                if should_exit:
                    break
                continue

            # Keep the context bounded *before* sending it to the brain.
            try:
                with console.status("[dim]condensing earlier conversation…[/dim]"):
                    summarized = await state.conversation.maybe_summarize(
                        state.utility_agent, state.utility_model
                    )
                if summarized:
                    console.print(
                        f"[dim]↳ summarized {summarized} older message(s) to stay within the context budget.[/dim]"
                    )
            except Exception as exc:  # noqa: BLE001 - summarization is best-effort
                console.print(f"[yellow]⚠ summarization skipped: {exc}[/yellow]")

            await run_turn_interruptible(
                console,
                run_turn(
                    console,
                    state.brain_agent,
                    state.brain_model,
                    state.deps,
                    state.conversation,
                    line,
                    ask=approver,
                    limits=limits,
                ),
            )
    finally:
        await client.close()

    console.print("\n[dim]Goodbye.[/dim]")
    return 0


def main() -> int:
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
