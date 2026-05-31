# LocalBuddy

A local-first command-line AI agent that talks to models served by **LM Studio**
over its OpenAI-compatible API. **Phases 1–3** are built: an interactive,
streaming chat REPL (model switching, reasoning/thinking display, rolling history
summarization); tool calling — sandboxed filesystem, shell, and web-fetch tools
driven through an agent loop, gated by an approval prompt for risky actions, with
a per-turn iteration cap; and local RAG — ingest your documents, embed them via
LM Studio into a LanceDB store, and let the agent retrieve from them with a
`search_memory` tool. Durable/resumable sessions arrive next (see Roadmap).

## Requirements

- Python 3.12+ (developed on 3.14)
- [uv](https://docs.astral.sh/uv/)
- **LM Studio** running its local server (Developer → Start Server) at
  `http://localhost:1234/v1`, with **two models loaded**:
  - a larger *brain* model for chat/reasoning (e.g. a Qwen3-class ~27–30B model)
  - a small *utility* model for cheap summarization (e.g. Gemma 4 E4B)
  - *(optional, for RAG)* an embedding model (e.g. `text-embedding-nomic-embed-text`)

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
| `/ingest <path>` | Add a text/Markdown file or folder to the knowledge base (RAG) |
| `/remember [text]` | Save a note — or, with no text, a summary of this conversation — to memory |
| `/memory` | Show knowledge base stats (chunks, sources, embedder) |
| `/forget` | Clear the knowledge base |
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

## Knowledge base / RAG (Phase 3)

Give LocalBuddy your own documents and let it retrieve from them:

- **Ingest:** `/ingest <path>` chunks a text/Markdown file (or every text file in
  a folder), embeds each chunk via LM Studio's embedding model, and stores it in a
  local **LanceDB** index under `data/lancedb/`. Re-ingesting a path replaces its
  previous chunks. `/memory` shows stats; `/forget` clears the index. **Relative
  paths resolve inside the workspace** (where the agent's `write_file` lands), so
  `/ingest notes.txt` finds `workspace/notes.txt`; use an absolute path for files
  elsewhere.
- **Remember:** `/remember <text>` saves a note; bare `/remember` summarizes the
  *current conversation* (via the utility model) and saves that — so a future
  session can recall the gist. Both are embedded and stored like any other entry,
  tagged `note:<time>` / `conversation:<time>`. (Note: this is long-term recall via
  RAG — *resuming* a whole session verbatim is Phase 4.)
- **Retrieve:** the agent has a read-only **`search_memory`** tool it calls when a
  question might be answered by your documents — it embeds the query, pulls the top
  matches, and grounds its answer in them (no approval needed; it's read-only).
- **Embedding model:** resolved lazily — a configured id, else an auto-detected
  `*embed*` model, else an interactive pick on your first `/ingest`. Nothing is
  embedded (and no embedding model is loaded) until you ingest or the agent
  searches, so RAG adds no startup or VRAM cost until used.

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
- `LOCALBUDDY_MAX_MODEL_REQUESTS` (default `12`), `LOCALBUDDY_MAX_TOOL_CALLS` (default `24`)
- `LOCALBUDDY_SHOW_THINKING` (default `true`) — stream reasoning-model thinking, dimmed
- `LOCALBUDDY_ENABLE_MEMORY` (default `true`) — RAG: `search_memory` tool + `/ingest`
- `LOCALBUDDY_EMBEDDER_MODEL_ID` — pin the embedding model (else auto/interactive)
- `LOCALBUDDY_CHUNK_CHARS` (default `1200`), `LOCALBUDDY_CHUNK_OVERLAP` (default `200`)
- `LOCALBUDDY_RAG_TOP_K` (default `5`) — passages returned per `search_memory` call

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
  tools/        # filesystem, shell, webfetch, search_memory tools + the approval gate
  memory/       # embeddings (LM Studio), LanceDB store, ingest, retrieval (Phase 3)
  __main__.py   # `python -m agent` entry point
data/           # repl history + LanceDB index (gitignored)
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
10. **RAG:** create a text file, `/ingest <path>` it (approve the embedder pick if
    asked), then `/memory` shows the chunk count and source. Ask a question whose
    answer is in that file → the agent calls `search_memory` (you'll see a 🔧 line)
    and grounds its answer in the retrieved passage. `/forget` clears the index.
11. Force summarization: set `LOCALBUDDY_HISTORY_TOKEN_BUDGET=300`, then hold a
    short conversation. Once the budget is exceeded you'll see
    `↳ summarized N older message(s)…` and the context stays bounded.
12. `/clear` resets the conversation; `/exit` quits.
13. Stop the LM Studio server and start LocalBuddy → you get a clear connection
    error rather than a traceback.

## Roadmap

- **Phase 1 ✓** — streaming chat REPL, model switching, rolling summarization
- **Phase 2 ✓** — filesystem / shell / web-fetch tools, the agent loop, an
  approval gate for risky calls, and a per-turn iteration cap
- **Phase 3 ✓** — local RAG: chunk → embed (via LM Studio) → LanceDB, retrieved via a `search_memory` tool + `/ingest`
- **Phase 4** — durable, resumable sessions persisted to SQLite
- **Deferred** — Phase 5 (daemon + scheduler), Phase 6 (MCP integrations)
