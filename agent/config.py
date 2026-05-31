"""Configuration for LocalBuddy (Phase 1).

Values are read from environment variables prefixed ``LOCALBUDDY_`` and an
optional ``.env`` file in the working directory. See ``.env.example`` for the
full list. Every setting has a sensible default, so LocalBuddy runs with no
configuration at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LOCALBUDDY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LM Studio (OpenAI-compatible server) ---
    base_url: str = "http://localhost:1234/v1"
    # LM Studio ignores the key, but the OpenAI client requires a non-empty value.
    api_key: str = "lm-studio"
    request_timeout: float = 120.0  # seconds per request to the local server

    # --- Model selection ---
    # Leave unset to be prompted interactively at startup. When set, the id must
    # match one returned by GET /v1/models, otherwise LocalBuddy falls back to
    # the interactive picker. Model ids are never hardcoded.
    brain_model_id: str | None = None  # large reasoning / chat model
    utility_model_id: str | None = None  # small, cheap model for summarization

    # --- Rolling history summarization ---
    # When the estimated token size of the conversation exceeds the budget, the
    # oldest turns are condensed by the utility model into a single compact
    # summary turn, so the context that gets sent to the brain stays bounded.
    history_token_budget: int = 6000  # estimated-token ceiling before summarizing
    keep_recent_turns: int = 4  # most-recent user+assistant turns kept verbatim
    chars_per_token: float = 4.0  # heuristic: ~4 chars/token (no tokenizer dependency)
    # Token budget for a summary generation call. Must leave room for a
    # reasoning/"thinking" model to think AND emit the summary text (we keep only
    # the text); too small and the model burns the budget before answering.
    summary_max_tokens: int = 2048

    # --- Tools (Phase 2) ---
    enable_tools: bool = True  # set false for pure Phase-1 chat with no tools
    show_thinking: bool = True  # stream reasoning-model "thinking" text (dimmed)
    # Filesystem tools are scoped to this root (a sandbox). Relative paths are
    # resolved against the current working directory. Reads/lists run freely;
    # writes, deletes, shell commands, and web fetches require user approval.
    workspace_root: Path = Path("workspace")
    windows_shell: Literal["powershell", "cmd"] = "powershell"  # native shell on Windows
    shell_timeout: float = 30.0  # seconds before a shell command is killed
    webfetch_timeout: float = 20.0  # seconds for a web fetch
    max_tool_output_chars: int = 8000  # truncate tool results sent back to the model
    # Per-task iteration caps (a runaway guard for the tool loop).
    max_model_requests: int = 12  # max model calls per user turn (UsageLimits.request_limit)
    max_tool_calls: int = 24  # max tool executions per user turn (UsageLimits.tool_calls_limit)

    # --- Memory / RAG (Phase 3) ---
    enable_memory: bool = True  # register the search_memory tool + /ingest command
    # Embedding model for the knowledge base. Resolved lazily (configured id, else
    # an auto-detected "embed"-named model, else an interactive pick on /ingest).
    embedder_model_id: str | None = None
    chunk_chars: int = 1200  # target characters per chunk when ingesting
    chunk_overlap: int = 200  # character overlap between consecutive chunks
    rag_top_k: int = 5  # passages returned by a search_memory call
    embed_batch_size: int = 32  # texts per embedding request

    # --- Persona ---
    system_prompt: str = (
        "You are LocalBuddy, a concise and helpful AI assistant running fully "
        "locally via LM Studio. Answer directly and accurately, and use Markdown "
        "when it helps. If you are unsure or lack information, say so plainly.\n\n"
        "You have tools for a sandboxed workspace: read, write, and list files; "
        "run shell commands; and fetch web pages. When a task needs one, CALL THE "
        "TOOL directly — do NOT merely describe the action, and do NOT ask the "
        "user for permission in your reply. Approval is handled automatically by "
        "the system: before any risky action (writing or deleting files, running "
        "commands, fetching URLs) actually runs, the user is shown the exact call "
        "and approves or denies it outside of your message. So just make the tool "
        "call; never say things like 'please approve'. If a call is denied you "
        "will receive a message saying so — then adapt. Prefer relative paths "
        "within the workspace, and only use tools when they genuinely help.\n\n"
        "You also have a search_memory tool that searches the user's ingested "
        "knowledge base (documents they added with /ingest). Use it when the "
        "question might be answered by their documents, and ground your answer in "
        "what it returns; if it returns nothing relevant, say so rather than guess."
    )

    # --- Paths ---
    # data_dir is reserved for later phases (SQLite, LanceDB); Phase 1 uses it
    # only for the REPL input history file.
    data_dir: Path = Path("data")

    @property
    def history_file(self) -> Path:
        return self.data_dir / "repl_history.txt"

    @property
    def lancedb_dir(self) -> Path:
        return self.data_dir / "lancedb"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def workspace_path(self) -> Path:
        """Return the absolute, resolved workspace root, creating it if needed."""
        root = self.workspace_root
        if not root.is_absolute():
            root = Path.cwd() / root
        root = root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root
