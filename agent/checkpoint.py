"""SQLite-backed session persistence + resume (Phase 4).

Each REPL conversation is a Session, saved to a local SQLite database after every
completed turn so it survives exit/crash. Messages are serialized with
pydantic-ai's ``ModelMessagesTypeAdapter``, so tool calls and the full message
structure round-trip faithfully. Scope is session + messages: ``/resume`` reloads
a past conversation's full context (the last *committed* turn — partial/aborted
turns are rolled back and never saved).
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    UserPromptPart,
)


def now_iso() -> str:
    # Full (microsecond) resolution so "most recently updated" ordering never ties.
    return datetime.now(timezone.utc).isoformat()


def new_session_id() -> str:
    return uuid.uuid4().hex


def derive_title(messages: list[ModelMessage], limit: int = 60) -> str:
    """A short title from the first user message, for the /sessions list."""
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    text = part.content if isinstance(part.content, str) else str(part.content)
                    text = " ".join(text.split())
                    if text:
                        return text[:limit] + ("…" if len(text) > limit else "")
    return "(no user message)"


@dataclass
class Session:
    id: str
    created_at: str
    updated_at: str
    title: str
    message_count: int

    @classmethod
    def new(cls) -> "Session":
        ts = now_iso()
        return cls(id=new_session_id(), created_at=ts, updated_at=ts, title="", message_count=0)


class SessionStore:
    """Stores conversations in a local SQLite DB (one row per session)."""

    def __init__(self, db_path: Path) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    message_count INTEGER NOT NULL DEFAULT 0,
                    messages_json TEXT NOT NULL DEFAULT '[]'
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path))
        conn.row_factory = sqlite3.Row
        return conn

    def _row_to_session(self, row: sqlite3.Row) -> Session:
        return Session(
            id=row["id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            title=row["title"],
            message_count=row["message_count"],
        )

    def save(self, session: Session, messages: list[ModelMessage]) -> None:
        blob = ModelMessagesTypeAdapter.dump_json(messages).decode("utf-8")
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO sessions (id, created_at, updated_at, title, message_count, messages_json)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       updated_at=excluded.updated_at,
                       title=excluded.title,
                       message_count=excluded.message_count,
                       messages_json=excluded.messages_json""",
                (session.id, session.created_at, session.updated_at, session.title,
                 session.message_count, blob),
            )

    def load_messages(self, session_id: str) -> list[ModelMessage]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT messages_json FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return []
        return list(ModelMessagesTypeAdapter.validate_json(row["messages_json"]))

    def list_sessions(self) -> list[Session]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, created_at, updated_at, title, message_count "
                "FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [self._row_to_session(r) for r in rows]

    def get(self, session_id: str) -> Session | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, created_at, updated_at, title, message_count "
                "FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        return self._row_to_session(row) if row else None

    def delete(self, session_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return cur.rowcount > 0
