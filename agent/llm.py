"""LM Studio plumbing: model discovery and pydantic-ai client construction.

This module is intentionally UI-free so it can be reused by later phases. A
single ``AsyncOpenAI`` client (pointed at LM Studio's OpenAI-compatible
endpoint) is shared for model discovery and for every model wrapper.
"""

from __future__ import annotations

from openai import APIConnectionError, AsyncOpenAI, OpenAIError
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .config import Settings
from .tools import AgentDeps, build_tools


class LMStudioError(RuntimeError):
    """Raised when LM Studio is unreachable or returns an error."""


def make_client(settings: Settings) -> AsyncOpenAI:
    """Create the shared OpenAI-compatible client pointed at LM Studio."""
    return AsyncOpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout=settings.request_timeout,
        max_retries=0,  # fail fast against a local server; the REPL surfaces errors
    )


async def list_models(client: AsyncOpenAI) -> list[str]:
    """Return the ids of all models currently loaded/available in LM Studio."""
    try:
        page = await client.models.list()
    except APIConnectionError as exc:
        raise LMStudioError(
            f"Could not reach LM Studio at {client.base_url}. "
            "Is the local server running? (LM Studio -> Developer -> Start Server)"
        ) from exc
    except OpenAIError as exc:
        raise LMStudioError(f"LM Studio returned an error while listing models: {exc}") from exc
    return sorted(model.id for model in page.data)


def build_model(model_id: str, client: AsyncOpenAI) -> OpenAIChatModel:
    """Wrap a model id as a pydantic-ai chat model backed by the shared client."""
    return OpenAIChatModel(model_id, provider=OpenAIProvider(openai_client=client))


def build_brain_agent(settings: Settings) -> Agent[AgentDeps, str]:
    """The main reasoning/chat agent, with tools registered when enabled. The
    concrete model is supplied per run, so switching the brain model at runtime
    needs no agent rebuild."""
    tools = build_tools(settings) if settings.enable_tools else []
    return Agent(deps_type=AgentDeps, instructions=settings.system_prompt, tools=tools)


def build_utility_agent() -> Agent[None, str]:
    """A small agent used for cheap utility work (Phase 1: summarization)."""
    return Agent(
        instructions=(
            "You are a summarization assistant. You produce faithful, compact "
            "summaries of conversations, preserving key facts, decisions, names, "
            "identifiers, and open questions. You never invent information that "
            "was not present in the source text."
        )
    )
