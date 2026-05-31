"""In-memory conversation state and rolling-history summarization (Phase 1).

The conversation is stored as pydantic-ai ``ModelMessage`` objects — the same
representation the tool-calling loop will use in Phase 2. When the estimated
size of the history exceeds the configured budget, the oldest turns are
condensed by the utility model into a single summary turn and replaced, so the
context stays bounded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)

from .config import Settings

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.models import Model

_SUMMARY_PROMPT = """\
Condense the following excerpt from a conversation between a user and an AI \
assistant into concise notes. Preserve key facts, decisions, user preferences, \
names, code or identifiers, and any unresolved questions or tasks. Do not add \
anything that is not present in the excerpt.

--- CONVERSATION EXCERPT ---
{transcript}
--- END EXCERPT ---

Summary notes:"""


def _stringify(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        # Multimodal content sequences are not used in Phase 1; keep the text bits.
        return " ".join(_stringify(c) for c in content if isinstance(c, str))
    return str(content)


def message_text(message: ModelMessage) -> str:
    """Best-effort plain-text extraction from a message's textual parts."""
    chunks: list[str] = []
    for part in message.parts:
        if isinstance(part, (UserPromptPart, SystemPromptPart, TextPart)):
            text = _stringify(part.content).strip()
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def _is_user_turn(message: ModelMessage) -> bool:
    """True for a request that carries an actual user prompt (not tool returns)."""
    return isinstance(message, ModelRequest) and any(
        isinstance(part, UserPromptPart) for part in message.parts
    )


def _render_transcript(messages: list[ModelMessage]) -> str:
    lines: list[str] = []
    for message in messages:
        text = message_text(message)
        if not text:
            continue
        role = "User" if isinstance(message, ModelRequest) else "Assistant"
        lines.append(f"{role}: {text}")
    return "\n\n".join(lines)


@dataclass
class Conversation:
    """The running, in-memory conversation (persistence arrives in Phase 4)."""

    settings: Settings
    messages: list[ModelMessage] = field(default_factory=list)

    def clear(self) -> None:
        self.messages = []

    def estimated_tokens(self) -> int:
        chars = sum(len(message_text(m)) for m in self.messages)
        return int(chars / self.settings.chars_per_token)

    async def maybe_summarize(self, utility_agent: "Agent", utility_model: "Model") -> int | None:
        """Summarize and replace old turns when the history exceeds the budget.

        Returns the number of messages that were condensed, or ``None`` if no
        summarization happened (under budget, or too little history to trim).
        """
        if self.estimated_tokens() <= self.settings.history_token_budget:
            return None

        keep = max(self.settings.keep_recent_turns, 1) * 2  # turns -> messages
        if len(self.messages) <= keep:
            return None  # nothing old enough to condense

        split = len(self.messages) - keep
        # Land the kept-tail boundary on a real user turn so the injected summary
        # slots in cleanly (no orphaned tool returns or consecutive same-role msgs).
        while split > 0 and not _is_user_turn(self.messages[split]):
            split -= 1
        if split <= 0:
            return None

        head = self.messages[:split]
        tail = self.messages[split:]

        result = await utility_agent.run(
            _SUMMARY_PROMPT.format(transcript=_render_transcript(head)),
            model=utility_model,
            model_settings={
                "temperature": 0.2,
                "max_tokens": self.settings.summary_max_tokens,
            },
        )
        summary = result.output.strip()

        # Represent the summary as a normal user/assistant turn so the message
        # history stays a valid alternating sequence for strict local templates.
        summary_turn: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=(
                            "Summary of the earlier conversation (older messages were "
                            f"condensed to save context):\n\n{summary}"
                        )
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="Understood — I have that earlier context and will continue from here."
                    )
                ]
            ),
        ]
        self.messages = summary_turn + tail
        return len(head)
