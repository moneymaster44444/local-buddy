"""Web-fetch tool: download a URL and extract its main text with trafilatura.

Always registered with ``requires_approval=True`` (it makes a network request).
"""

from __future__ import annotations

import asyncio

import httpx
import trafilatura

from pydantic_ai import RunContext

from .base import AgentDeps, truncate

_USER_AGENT = "LocalBuddy/0.1 (local CLI agent)"


async def fetch_url(ctx: RunContext[AgentDeps], url: str) -> str:
    """Fetch a web page and return its main text content. Requires approval."""
    if not url.lower().startswith(("http://", "https://")):
        return "URL must start with http:// or https://"
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=ctx.deps.settings.webfetch_timeout,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            html = response.text
    except httpx.HTTPError as exc:
        return f"Failed to fetch {url}: {exc}"

    # trafilatura.extract is CPU-bound parsing; run it off the event loop.
    extracted = await asyncio.to_thread(
        trafilatura.extract,
        html,
        include_comments=False,
        include_links=False,
        favor_recall=True,
    )
    if not extracted:
        return (
            f"Fetched {url} ({len(html)} bytes of HTML) but could not extract readable "
            "main content (it may be JS-rendered or not an article)."
        )
    return truncate(f"# Source: {url}\n\n{extracted}", ctx.deps.settings.max_tool_output_chars)
