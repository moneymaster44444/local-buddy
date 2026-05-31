# LocalBuddy

A local-first command-line AI agent that talks to models served by **LM Studio**
over its OpenAI-compatible API. **Phases 1–2** are built: an interactive,
streaming chat REPL (with model switching, reasoning/thinking display, and
rolling history summarization) plus tool calling — sandboxed filesystem, shell,
and web-fetch tools driven through an agent loop, gated by an approval prompt for
risky actions, with a per-turn iteration cap. Local RAG and durable/resumable
sessions arrive in later phases (see Roadmap).

## Requirements

- Python 3.12+ (developed on 3.14)
- [uv](https://docs.astral.sh/uv/)
- **LM Studio** running its local server (Developer → Start Server) at
  `http://localhost:1234/v1`, with **two models loaded**:
  - a larger *brain* model for chat/reasoning (e.g. a Qwen3-class ~27–30B model)
  - a small *utility* model for cheap summarization (e.g. Gemma 3n E4B)

Model ids are **never hardcoded** — LocalBuddy reads them from `GET /v1/models`
and lets you pick (or pin them via configuration).

## Setup

```bash
uv sync
```

## Run

```bash
uv run localbuddy
# equivalently:
uv run python -m agent
```

On first run (with no models pinned) LocalBuddy lists the models LM Studio
reports and asks you to choose a brain and a utility model.

## Commands

| Command | Description |
| --- | --- |
| `/help` | Show commands |
| `/model` | Show current brain & utility models and all available |
| `/model brain [id\|#]` | Switch the brain model (interactive picker if no id) |
| `/model utility [id\|#]` | Switch the utility model (interactive picker if no id) |
| `/tools [reset\|revoke <tool>]` | List tools & approvals; `revoke <tool>` downshifts one, `reset` clears all session auto-approvals |
| `/clear` | Clear the conversation |
| `/exit`, `/quit` | Quit |

Input keys: **Enter** sends, **Alt+Enter** inserts a newline, **Ctrl+C** cancels
the current reply, **Ctrl+D** exits.

## Tools & safety (Phase 2)

> ⚠️ **Experimental.** LocalBuddy can read/write files and run shell commands on
> your machine. Risky actions are gated behind an approval prompt, but the shell
> is a real shell — read each action before approving, and run it only on
> code/data you trust. No warranty; use at your own risk.

The brain model can call tools. Read-only tools run automatically; anything that
changes the world prompts you first:

| Tool | Approval? |
| --- | --- |
| `read_file`, `list_dir` | none (read-only) |
| `write_file`, `delete_path` | **required** |
| `run_shell` | **required** |
| `fetch_url` | **required** |

When a risky tool is called the run **pauses** and shows you the exact action and
its arguments. Answer **y** (once), **N** (deny — the model is told and adapts),
or **a** (always allow this tool for the rest of the session). Grants can be
**downshifted** any time: `/tools revoke <tool>` drops one back to prompting, and
`/tools reset` clears them all. For the *sandboxed* tools (`write_file`,
`delete_path`) **a** is
a single step. For `run_shell` and `fetch_url` — which reach **outside** the
sandbox — choosing **a** requires a second, typed `yes` confirmation, so blanket
approval of unrestricted tools is always deliberate (useful for longer
autonomous runs, hard to grant by accident).

- **Filesystem sandbox:** all `read_file`/`write_file`/`list_dir`/`delete_path`
  paths are confined to `LOCALBUDDY_WORKSPACE_ROOT` (default `./workspace`,
  auto-created). Paths that escape it are rejected.
- **Shell:** runs in the workspace dir using the platform's native shell
  (PowerShell on Windows by default; `$SHELL`/`/bin/sh` elsewhere). Note the shell
  is a *real* shell — it is **not** path-confined; its only guard is the per-call
  approval prompt, so read each command before approving.
- **Iteration cap:** a turn is limited to `LOCALBUDDY_MAX_MODEL_REQUESTS` model
  calls and `LOCALBUDDY_MAX_TOOL_CALLS` tool executions, as a runaway guard.

Reasoning models (e.g. Qwen3) stream their **thinking** dimmed before the answer;
set `LOCALBUDDY_SHOW_THINKING=false` to hide it.

## Configuration

Copy `.env.example` to `.env` to override defaults (or set `LOCALBUDDY_*`
environment variables). Notable settings:

- `LOCALBUDDY_BASE_URL` — LM Studio endpoint (default `http://localhost:1234/v1`)
- `LOCALBUDDY_BRAIN_MODEL_ID`, `LOCALBUDDY_UTILITY_MODEL_ID` — pin to skip the picker
- `LOCALBUDDY_HISTORY_TOKEN_BUDGET` (default `6000`) — when the *estimated* token
  size of the conversation exceeds this, the oldest turns are summarized by the
  utility model and replaced with a compact summary turn
- `LOCALBUDDY_KEEP_RECENT_TURNS` (default `4`) — recent turns always kept verbatim
- `LOCALBUDDY_ENABLE_TOOLS` (default `true`) — set `false` for pure chat, no tools
- `LOCALBUDDY_WORKSPACE_ROOT` (default `workspace`) — filesystem sandbox root
- `LOCALBUDDY_WINDOWS_SHELL` (`powershell` | `cmd`, default `powershell`)
- `LOCALBUDDY_SHELL_TIMEOUT` (default `30`), `LOCALBUDDY_WEBFETCH_TIMEOUT` (default `20`)
- `LOCALBUDDY_MAX_MODEL_REQUESTS` (default `8`), `LOCALBUDDY_MAX_TOOL_CALLS` (default `16`)
- `LOCALBUDDY_SHOW_THINKING` (default `true`) — stream reasoning-model thinking, dimmed

Token size is **estimated** with a `chars / 4` heuristic
(`LOCALBUDDY_CHARS_PER_TOKEN`), so no model-specific tokenizer dependency is
needed.

## How it works

```
agent/
  config.py     # pydantic-settings configuration
  llm.py        # shared AsyncOpenAI client → LM Studio; model discovery + agent/model builders (UI-free)
  state.py      # in-memory Conversation (pydantic-ai messages) + rolling summarization
  loop.py       # the agent step loop: stream → tools → approval → resume, with the iteration cap
  repl.py       # the REPL, commands, model picker, approval UI, bootstrap
  tools/        # filesystem, shell, webfetch tools + the approval gate + sandbox helpers
  __main__.py   # `python -m agent` entry point
data/           # repl history now; sqlite + lancedb later (gitignored)
workspace/      # filesystem sandbox for tools (gitignored)
```

The agent and tool-calling loop are built on **pydantic-ai**, connected to LM
Studio via its OpenAI-compatible provider. We depend on
`pydantic-ai-slim[openai]` rather than the full `pydantic-ai` meta-package: it's
the same library but pulls only the OpenAI-compatible provider this project
needs, avoiding ~8 unused cloud-provider SDKs.

The brain model is supplied per request (`agent.iter(..., model=...)`), so
switching models at runtime needs no agent rebuild. The system prompt is set as
agent *instructions*, which keeps it out of the stored message history (and thus
out of summarization) while always being applied.

A user turn runs through `agent.iter()` (in `loop.py`): one code path streams the
answer, executes read-only tools inline, and — when the model calls a tool marked
`requires_approval=True` — **pauses** the run (pydantic-ai's `DeferredToolRequests`
human-in-the-loop mechanism), asks via the approval gate, then **resumes** with the
results. The per-turn iteration cap is enforced with `UsageLimits`.

## Manual test checklist

1. `uv sync`; ensure LM Studio is serving two models.
2. `uv run localbuddy` → pick a brain and a utility model when prompted.
3. Ask a question → reasoning streams dimmed (if any), then the reply streams in.
4. `/model` → table shows both roles and all models; `/model utility` → re-pick.
5. `/tools` → lists the tools and their approval status, plus the workspace path.
6. **Read tool (no approval):** ask "list the files in your workspace" → it runs
   `list_dir` inline without prompting.
7. **Write tool (approval):** ask "create notes.txt with 'hello'" → you get an
   approval panel showing the path & content; **y** writes it (check
   `workspace/notes.txt`), **N** declines and the model adapts.
8. **Shell tool:** ask "run `echo hi` in the shell" → approval panel shows the
   command; approve and see the output.
9. **Iteration cap:** lower `LOCALBUDDY_MAX_MODEL_REQUESTS` and give a multi-step
   task → the turn stops with an "iteration cap reached" notice.
10. Force summarization: set `LOCALBUDDY_HISTORY_TOKEN_BUDGET=300`, then hold a
    short conversation. Once the budget is exceeded you'll see
    `↳ summarized N older message(s)…` and the context stays bounded.
11. `/clear` resets the conversation; `/exit` quits.
12. Stop the LM Studio server and start LocalBuddy → you get a clear connection
    error rather than a traceback.

## Roadmap

- **Phase 1 ✓** — streaming chat REPL, model switching, rolling summarization
- **Phase 2 ✓** — filesystem / shell / web-fetch tools, the agent loop, an
  approval gate for risky calls, and a per-turn iteration cap
- **Phase 3** — local RAG: chunk → embed (via LM Studio) → LanceDB → retrieve
- **Phase 4** — durable, resumable sessions persisted to SQLite
- **Deferred** — Phase 5 (daemon + scheduler), Phase 6 (MCP integrations)
