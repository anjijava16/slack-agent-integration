# Slack × FastAPI × LangGraph — an agent that lives in your Slack

This repo is a complete, working example of a tool-using AI agent that people talk to from Slack. You @mention it in a channel, DM it, or use a slash command; a FastAPI service receives that, hands it to a LangGraph agent, the agent searches a knowledge base / opens tickets / does math / checks the time over as many steps as it needs, and the answer comes back into the same Slack thread. The same agent is also exposed as a REST API, so your other systems can call it and optionally post the result into Slack.

It started from two small scripts that already worked — a Slack Bolt app in Socket Mode and a `chat_postMessage` sender — and grows them, one layer at a time, into something you could actually run for a team. Everything that's in the README is in the code, and everything in the code is explained in the README.

```
#it-help
  Anji:      @Ops Buddy my VPN keeps dropping every few minutes since this morning. open a ticket pls
  Ops Buddy: 👀
             ⏳ working on it…
             ↓ (edited a few seconds later)
             Opened *TCK-7F3A21* (P2) — "VPN drops every few minutes" for <@U07ANJI>.
             While you wait: switch the VPN protocol to *IPSec* in settings, that usually fixes the drops (KB-001).
             🛠 `search_knowledge_base`, `create_ticket` · ⏱ 3.4s
             [👍 Helpful] [👎 Not helpful]

  Anji:      (in the thread) how many PTO days is that per quarter if I get 21 a year?
  Ops Buddy: 21 days a year is *5.25 days per quarter* (KB-003). Up to 5 unused days roll over.
             🛠 `search_knowledge_base`, `calculator` · ⏱ 2.1s
```

The thread remembers. The ticket's requester is the real Slack user id, not whatever the model felt like writing. The 👀 → ✅ reactions show progress. None of that is magic, and the rest of this document walks through exactly how each piece works.

---

## Contents

1. [Quick start (about 10 minutes)](#1-quick-start-about-10-minutes)
2. [What's in the box](#2-whats-in-the-box)
3. [Slack from first principles](#3-slack-from-first-principles)
4. [Your two original scripts, explained](#4-your-two-original-scripts-explained)
5. [How Bolt for Python actually dispatches things](#5-how-bolt-for-python-actually-dispatches-things)
6. [Putting Bolt inside FastAPI](#6-putting-bolt-inside-fastapi)
7. [LangGraph from first principles](#7-langgraph-from-first-principles)
8. [Architecture diagrams](#8-architecture-diagrams)
9. [One request, every hop, with real payloads](#9-one-request-every-hop-with-real-payloads)
10. [Code walkthrough, file by file](#10-code-walkthrough-file-by-file)
11. [Setting up the Slack app, step by step](#11-setting-up-the-slack-app-step-by-step)
12. [Taking it to production](#12-taking-it-to-production)
13. [Testing without Slack or OpenAI](#13-testing-without-slack-or-openai)
14. [Extending it](#14-extending-it)
15. [Troubleshooting](#15-troubleshooting)
16. [Cheat sheet](#16-cheat-sheet)
17. [Feedback: 👍/👎 into PostgreSQL](#17-feedback--into-postgresql)
18. [Deploying to Google Cloud: Terraform, Cloud SQL, GitHub Actions](#18-deploying-to-google-cloud-terraform-cloud-sql-github-actions)
19. [Course FAQ bot: RAG with LlamaIndex and Qdrant](#19-course-faq-bot-rag-with-llamaindex-and-qdrant)

---

## 1. Quick start (about 10 minutes)

You need Python 3.11+, a Slack workspace where you can install apps, and an OpenAI key (or any OpenAI-compatible endpoint — OpenRouter, a LiteLLM proxy, Ollama).

```bash
unzip slack-fastapi-langgraph-agent.zip && cd slack-fastapi-langgraph-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

**Create the Slack app from the manifest** (this sets every scope, event and slash command for you):

1. Go to <https://api.slack.com/apps> → **Create New App** → **From a manifest** → pick your workspace.
2. Paste the contents of `slack_manifests/manifest.socket.yaml` → **Create**.
3. **Basic Information → App-Level Tokens → Generate Token and Scopes**, add scope `connections:write`, generate. That's your `xapp-...` token → `SLACK_APP_TOKEN` in `.env`.
4. **Install App → Install to Workspace → Allow**. Copy the **Bot User OAuth Token** (`xoxb-...`) → `SLACK_BOT_TOKEN`.
5. LLM key: if `~/.zprofile.sh` exports `OPENROUTER_API_KEY`, there's nothing to do (an `sk-or-` key points the client at OpenRouter automatically). Storage needs nothing either: locally everything (runs, 👍/👎 votes, conversation memory) goes into one SQLite file, `./data/agent.db`. Cloud environments use PostgreSQL — section 18.
6. In Slack, open a channel and type `/invite @Ops Buddy`.

**Run it:**

```bash
make test   # 85 pass offline; Postgres / Qdrant-server / real-data tests skip until you point them at one
make run    # uvicorn app.main:app --reload --port 8000
```

You should see `Slack connected via Socket Mode`. Now try, in Slack:

| Where | Type | What happens |
|---|---|---|
| Channel | `hello` | Greeting, no LLM involved |
| Channel | `@Ops Buddy when is the deploy freeze?` | Agent searches the KB, answers in a thread |
| Same thread | `@Ops Buddy open a P3 ticket to extend it for team X` | Agent remembers the thread, opens a ticket |
| DM to the bot | `what time is it in Kolkata?` | Agent calls `get_current_time` |
| Anywhere | `/agent how many PTO days per quarter?` | Agent answers publicly via the slash command |
| Anywhere | `/agent-add 2 3.5` | `5.5`, no LLM |

And from a terminal, the REST side:

```bash
./scripts/demo_requests.sh
# or open http://localhost:8000/docs for the Swagger UI
```

If your existing `.env` from the original script uses `SLACK_SOCKET_TOKEN`, that still works — it's accepted as an alias for `SLACK_APP_TOKEN`. If your existing slash commands are `/welcomeagentadd` and `/welcomeagentanalyze`, either set `SLACK_CMD_AGENT=/welcomeagentanalyze` and `SLACK_CMD_ADD=/welcomeagentadd` or create the new commands from the manifest.

---

## 2. What's in the box

```
slack-fastapi-langgraph-agent/
├── app/
│   ├── main.py                 FastAPI app factory + lifespan: starts Slack (socket or http), builds the agent
│   ├── config.py               All settings, validated by pydantic-settings
│   ├── agent/
│   │   ├── state.py            AgentState — what flows through the graph
│   │   ├── graph.py            The LangGraph: guardrail → agent ⇄ tools → finalize
│   │   ├── tools.py            6 tools: calculator, clock, KB search, create/get/list tickets
│   │   ├── prompts.py          System prompt (Slack-aware)
│   │   └── llm.py              ChatOpenAI factory (works with any OpenAI-compatible URL)
│   ├── db/
│   │   ├── schema.sql          agent_runs, agent_feedback, agent_run_scores, slack_event_claims (PostgreSQL)
│   │   ├── schema_sqlite.sql   the same tables in SQLite's dialect (local)
│   │   ├── feedback_store.py   SQLite / Postgres / in-memory stores, picked from DATABASE_URL
│   │   └── checkpointer.py     LangGraph memory in the same database (SQLite or Postgres)
│   ├── rag/                    Course FAQ bot (section 19)
│   │   ├── courses.py          config/courses.yaml: courses, Slack channels, sources
│   │   ├── qdrant_store.py     Qdrant modes (local/cloud/embedded/memory), collections, RRF fusion
│   │   ├── embeddings.py       fastembed / OpenAI-compatible / HuggingFace / hash embeddings
│   │   ├── sparse.py           BM25 sparse vectors (Qdrant applies the IDF)
│   │   ├── readers/            FAQ + GitHub repos, Slack history, YouTube, web, files
│   │   ├── indexer.py          incremental chunk → embed → upsert, prune vanished sources
│   │   ├── retriever.py        hybrid search → recency weighting → re-rank
│   │   ├── rerank.py           Cohere / local cross-encoder / none
│   │   ├── graph.py            RAG LangGraph: guardrail → condense → retrieve → generate
│   │   ├── service.py          RagService: answer, search, ingest jobs, status
│   │   ├── tools.py            search_course_materials tool for the agent
│   │   └── cli.py              python -m app.rag.cli ingest|search|ask|status|reset
│   ├── services/
│   │   ├── agent_service.py    The only thing that runs the graph. Memory, locks, timing, tool trace, run_id
│   │   └── dedup.py            TTL de-dup so Slack retries don't produce double answers
│   ├── slack/
│   │   ├── bolt_app.py         Every Slack listener: hello, DM, mention, /agent, /agent-add, upvote, downvote
│   │   ├── blocks.py           Block Kit builders (answer, "thinking…", feedback buttons)
│   │   └── formatting.py       Markdown → Slack mrkdwn, chunking to Slack's limits
│   └── api/
│       ├── routes.py           /api/agent/invoke, /api/slack/notify, /api/agent/threads/{id}, /api/feedback, /api/runs/{id}
│       ├── rag_routes.py       /api/rag/* — query router (status, courses, query, search)
│       ├── ingest_routes.py    /api/ingest/* — ingestion router (course, faq, github, slack, youtube, urls, text, files, jobs)
│       └── schemas.py          Request/response models
├── examples/
│   ├── 01_bolt_socket_mode_basics.py   Your original Bolt script, with three bugs fixed
│   ├── 02_post_message.py              Your original sender, plus Block Kit and threads
│   └── 03_feedback_postgres_sync.py    Your upvote/downvote snippet, writing to Postgres (sync, psycopg)
├── slack_manifests/
│   ├── manifest.socket.yaml    Create the Slack app in one paste (Socket Mode)
│   └── manifest.http.yaml      Same, for HTTP mode (public URL)
├── config/courses.yaml         The four DataTalks.Club Zoomcamps: channels, FAQ folder, repos, videos
├── tests/                      103 tests (+ a real-data retrieval test): agent, Slack, feedback, persistence, RAG
├── infra/
│   ├── platform/               Terraform, applied once by you: Cloud SQL, SAs, WIF, secrets, Artifact Registry
│   └── cloudrun/               Terraform, applied by CD per environment: the Cloud Run service
├── scripts/
│   ├── config.sh               Project, repo, branch, service name — edit once
│   ├── gcp-setup.sh            One-time Google Cloud setup (gcloud + terraform)
│   ├── push-secrets.sh         ~/.zprofile.sh -> Secret Manager (Slack, OpenRouter)
│   ├── github-setup.sh         GitHub variables, Docker Hub secrets, environments (gh CLI)
│   ├── slack-manifest.sh       Slack app manifest pointing at a deployed environment
│   └── demo_requests.sh        curl tour of the REST API
├── .github/workflows/          ci.yml, cd.yml (develop -> test -> prod), deploy.yml
├── Dockerfile, docker-compose.yml, Makefile, pyproject.toml (ruff + pytest config)
├── requirements.txt, requirements-dev.txt, requirements-examples.txt, .env.example
└── README.md                   You are here
```

The code is split along one rule: **Slack code doesn't know about LangGraph, and LangGraph code doesn't know about Slack.** They meet in exactly one place, `AgentService.run()`. That's why you can test the agent with no Slack, test the Slack handlers with a fake agent, and later bolt on Teams or a web UI without touching either.

---

## 3. Slack from first principles

Before any code, it helps to have the right mental model of what Slack is doing on its side. Most "my bot doesn't respond" problems come from a missing piece of this picture.

### 3.1 The cast of characters

- **Workspace** — your company's Slack (`yourco.slack.com`). Has an id like `T0123ABCD`.
- **Slack app** — a configuration object at api.slack.com/apps. It owns scopes, event subscriptions, slash commands, tokens. Id like `A0123ABCD`. An app isn't code; your code *acts on behalf of* the app.
- **Bot user** — the "person" your app shows up as in Slack (`@Ops Buddy`). Has a user id like `U0BOT1234`. When someone @mentions it, Slack writes `<@U0BOT1234>` in the message text.
- **Channels** — public `C…`, private `G…`/`C…`, DMs `D…`. Your bot only sees messages in conversations it's a member of.
- **Message `ts`** — the timestamp string like `1700000000.000100`. It's the message's primary key inside a channel. You need `channel + ts` to edit, react to or reply to a message.
- **Thread** — replies to a message carry `thread_ts` = the `ts` of the parent. To reply in a thread, you send `thread_ts`. To start a thread under a message, you send `thread_ts = that message's ts`.

### 3.2 Tokens — which one is which

| Token | Looks like | What it's for | In this repo |
|---|---|---|---|
| Bot token | `xoxb-…` | Calling the Web API *as the bot* (`chat.postMessage`, `reactions.add`, …) | `SLACK_BOT_TOKEN` |
| App-level token | `xapp-…` | Opening a Socket Mode WebSocket. Needs `connections:write`. Can't call normal Web API methods. | `SLACK_APP_TOKEN` (alias `SLACK_SOCKET_TOKEN`) |
| Signing secret | 32 hex chars | Verifying that an HTTP request really came from Slack (HMAC-SHA256) | `SLACK_SIGNING_SECRET` |
| User token | `xoxp-…` | Acting *as a human user*. Not used here, and you rarely want it for a bot. | — |

A rule of thumb: **xoxb talks to Slack, xapp listens to Slack over a socket, the signing secret checks that Slack is talking to you over HTTP.**

### 3.3 Scopes — permissions you ask for

Scopes are permissions granted at install time. If you add a scope later, **you have to reinstall the app** to get a token with it — this one trips up everyone at least once. The manifest asks for:

| Scope | Why |
|---|---|
| `app_mentions:read` | Receive `app_mention` events |
| `chat:write` | Post and edit messages |
| `chat:write.public` | Post in public channels the bot hasn't been invited to (used by `/api/slack/notify`) |
| `commands` | Slash commands |
| `im:history`, `im:read`, `im:write` | Receive and answer DMs |
| `channels:history`, `groups:history` | Receive `message` events in public/private channels (the `hello` listener) |
| `reactions:write` | The 👀 / ✅ progress reactions |

### 3.4 The four ways Slack and your code interact

```mermaid
flowchart LR
    subgraph Slack
      U[User] -->|types| SL[(Slack)]
    end
    SL -- "1. Events API<br/>app_mention, message.im, ..." --> APP[Your app]
    SL -- "2. Slash commands<br/>/agent ..." --> APP
    SL -- "3. Interactivity<br/>button clicks, modals" --> APP
    APP -- "4. Web API<br/>chat.postMessage, chat.update, reactions.add" --> SL
```

1. **Events API** — "something happened" notifications: a mention, a DM, a reaction, a channel join. You subscribe to event types in the app config. Slack pushes them to you.
2. **Slash commands** — a user types `/agent …`. Slack sends you the command text, who sent it, the channel, and a `response_url`.
3. **Interactivity** — someone clicks a button, picks from a menu, submits a modal. Slack sends you a `block_actions` (or `view_submission`) payload.
4. **Web API** — you calling Slack: post a message, update it, add a reaction, open a modal, look up a user. Plain HTTPS + bearer token.

1–3 are Slack → you. 4 is you → Slack. Your second original script (`chat_postMessage`) only uses #4, which is why it worked without any event setup.

### 3.5 How events reach you: Socket Mode vs HTTP

For 1–3 Slack has to deliver something to your code. There are two transports, and **the handler code is identical for both** — only the plumbing differs.

```mermaid
flowchart TB
    subgraph S1["Socket Mode (SLACK_MODE=socket)"]
      direction LR
      A1["Your app<br/>laptop / private network"] -- "outbound WebSocket<br/>wss://wss-primary.slack.com/..." --> B1[(Slack)]
      B1 -. "events pushed down<br/>the same socket" .-> A1
      A1 -. "ack envelope_id" .-> B1
    end
    subgraph S2["HTTP mode (SLACK_MODE=http)"]
      direction LR
      B2[(Slack)] -- "POST https://your-host/slack/events<br/>signed with X-Slack-Signature" --> A2["Your app<br/>public HTTPS URL"]
      A2 -- "HTTP 200 within 3s" --> B2
    end
```

**Socket Mode.** Your app calls `apps.connections.open` with the `xapp-` token, gets a one-time WebSocket URL, and connects *out* to Slack. Slack then pushes every event, command and interaction down that socket as an envelope; you send back an ack with the `envelope_id`. No public URL, no inbound firewall rule, no TLS certs. That's why your first script just worked on a laptop.

Things to know: an app can hold several socket connections at once (Slack spreads events across them), connections get recycled periodically and the SDK reconnects for you, and apps published to the Slack Marketplace must use HTTP.

**HTTP mode.** You give Slack a public HTTPS URL (one URL can serve events, commands and interactivity — this repo uses `/slack/events` for all three). Slack POSTs JSON (events) or form-encoded bodies (commands, interactions) to it, signed with your signing secret. You need a public endpoint: ngrok for development, then Cloud Run / ECS / Kubernetes ingress / API Gateway in production.

| | Socket Mode | HTTP |
|---|---|---|
| Public URL needed | No | Yes |
| Works behind corporate firewall | Yes (outbound only) | Needs ingress |
| Serverless (Lambda, Cloud Run scale-to-zero) | Poor fit (needs a long-lived connection) | Natural fit |
| Horizontal scaling | Each replica opens its own socket; Slack load-balances | Normal load balancer |
| Marketplace distribution | Not allowed | Required |
| Request verification | Not needed (you opened the connection) | Signing secret, mandatory |
| Good for | Dev, internal tools, on-prem | Production, multi-workspace apps |

This project supports both with one env var: `SLACK_MODE=socket` or `SLACK_MODE=http`.

### 3.6 The three-second rule

Slack waits **3 seconds** for an acknowledgement. Then:

- For **events**, Slack considers delivery failed and **retries** — up to 3 more times with backoff, adding headers `X-Slack-Retry-Num` and `X-Slack-Retry-Reason`. If your bot is slow, it answers the same question four times. If it keeps failing, Slack can eventually turn off event delivery for the app.
- For **slash commands**, the user sees *"/agent failed with the error 'operation_timeout'"*.
- For **button clicks**, the user sees a warning icon.

An LLM agent takes 2–30 seconds. So the pattern everywhere in this repo is: **ack immediately, do the work afterwards, then deliver the answer through a different channel** (`chat.update` on a placeholder message, or the command's `response_url`). Bolt makes this easy, and section 5 shows exactly how. On top of that, `TTLDeduper` drops any event whose `channel:ts` it has already seen, so a retry that sneaks through still produces one answer.

### 3.7 What an event actually looks like

This is what lands on `/slack/events` (or down the socket) when someone writes `@Ops Buddy when is the deploy freeze?` in a channel:

```json
{
  "token": "legacy-verification-token-ignore-it",
  "team_id": "T0123ABCD",
  "api_app_id": "A0123ABCD",
  "type": "event_callback",
  "event_id": "Ev07ABC123",
  "event_time": 1700000000,
  "event": {
    "type": "app_mention",
    "user": "U07ANJI",
    "text": "<@U0BOT1234> when is the deploy freeze?",
    "ts": "1700000000.000100",
    "channel": "C0ITHELP",
    "event_ts": "1700000000.000100"
  },
  "authorizations": [{ "team_id": "T0123ABCD", "user_id": "U0BOT1234", "is_bot": true }]
}
```

Three details matter for the code: the bot mention is *inside* `text` as `<@U0BOT1234>` (so we strip it before handing text to the LLM), there's no `thread_ts` because this is a top-level message (so `ts` becomes the thread root for our reply), and `event_id` + `channel:ts` identify the event for de-duplication.

The **first** request Slack ever sends to an HTTP endpoint is a handshake:

```json
{ "type": "url_verification", "challenge": "3eZbrw1aBm2rZgRNFdxV2595E9CY3gmdALWMmHkvFXO7tYXAYM8P", "token": "..." }
```

You must echo the `challenge` back. Bolt does this for you — there's a test for it in `tests/test_slack.py`.

A slash command arrives as **form-encoded** data, not JSON:

```
command=%2Fagent&text=how+many+PTO+days+per+quarter%3F&user_id=U07ANJI&channel_id=C0ITHELP
&team_id=T0123ABCD&response_url=https%3A%2F%2Fhooks.slack.com%2Fcommands%2FT0123%2F456%2Fabc&trigger_id=...
```

### 3.8 Request signing (HTTP mode)

Anyone on the internet can POST to your `/slack/events`. Slack signs each request so you can tell the real ones apart:

```
basestring = "v0:" + X-Slack-Request-Timestamp + ":" + raw_request_body
expected   = "v0=" + hex(HMAC_SHA256(signing_secret, basestring))
valid      = constant_time_equals(expected, X-Slack-Signature)
             and abs(now - timestamp) < 5 minutes        # replay protection
```

Bolt checks this before any of your code runs and answers `401` if it fails. Two practical notes: it must be computed over the **raw** body bytes (so never parse-then-reserialize before verification — Bolt's FastAPI adapter reads the raw body for you), and your server clock needs to be roughly correct. `tests/test_slack.py` builds properly signed requests by hand, which is a good way to see the algorithm in 5 lines.

### 3.9 Formatting: mrkdwn, Block Kit, and why LLM output looks wrong in Slack

Slack's `text` field and `mrkdwn` text objects use Slack's own dialect, **not** Markdown:

| You want | Markdown (what the LLM writes) | Slack mrkdwn |
|---|---|---|
| Bold | `**bold**` | `*bold*` |
| Italic | `*italic*` or `_italic_` | `_italic_` |
| Strike | `~~x~~` | `~x~` |
| Link | `[docs](https://x.io)` | `<https://x.io\|docs>` |
| Heading | `# Title` | not supported |
| List | `- item` | no list syntax; write `• item` |
| Mention a user | — | `<@U07ANJI>` |
| Mention a channel | — | `<#C0ITHELP>` |
| Code | `` `x` `` and fenced blocks | same (language hint is ignored) |

Also, the characters `&`, `<` and `>` must be escaped as `&amp;`, `&lt;`, `&gt;` — except inside Slack's own control sequences like `<@U…>`.

The project handles this two ways: the system prompt asks the model to write Slack formatting, and `app/slack/formatting.py` converts whatever Markdown slips through anyway (and leaves code blocks alone). Slack has also added a `markdown` Block Kit block that accepts standard Markdown; it's tied to Slack's AI-app features and isn't guaranteed for every app, so this repo sticks to the conversion approach, which works everywhere.

**Block Kit** is the layout system: a message's `blocks` is a list of `section`, `context`, `actions`, `header`, `divider`, `image`, … Limits worth remembering: 50 blocks per message, 3000 characters per `section` text. `blocks.py` chunks long answers to fit. Always send a plain `text` too — it's what shows in notifications and screen readers, and Slack logs a warning if it's missing.

### 3.10 Rate limits, briefly

Slack's Web API is tiered per method. The one that matters for a bot is `chat.postMessage`, which allows roughly one message per second per channel with short bursts. Exceeding it returns HTTP 429 with a `Retry-After` header; `slack_sdk` can retry those for you. An agent that answers in one edited message (placeholder → `chat.update`) instead of streaming dozens of posts stays well inside the limits.

---

## 4. Your two original scripts, explained

Both scripts are in `examples/`, with fixes. Walking through them is the fastest way to connect the theory above to code you already know works.

### 4.1 Script 1 — Bolt in Socket Mode

```python
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))          # xoxb: lets the app call the Web API
...
handler = SocketModeHandler(app, os.environ.get("SLACK_SOCKET_TOKEN"))  # xapp: opens the WebSocket
handler.start()                                               # blocks forever, dispatching events to listeners
```

| Listener | Slack surface | Needs | What Bolt gives the function |
|---|---|---|---|
| `@app.message("hello")` | Events API, `message.*` events | `channels:history` (+ event subscription) | `message`, `say` |
| `@app.command("/welcomeagentadd")` | Slash command | `commands` + command created in app config | `ack`, `respond`, `command` |
| `@app.command("/welcomeagentanalyze")` | Slash command | same | same |
| `@app.event("app_mention")` | Events API | `app_mentions:read` | `event`, `say` |

What's good about it: `ack()` is the first line in both commands, so Slack never times out; `respond()` uses the `response_url` so it works even in channels the bot isn't in.

Three things worth fixing, all fixed in `examples/01_bolt_socket_mode_basics.py`:

1. **`from slack_bolt import logger` is a module, not a logger.** `slack_bolt.logger` is a package of log-message helpers. Calling `logger.exception(...)` on it raises `AttributeError` — so the moment OpenAI failed, the error handler itself would crash and the user would never get the "Sorry…" message. Use `logging.getLogger(__name__)`.
2. **The mention handler sends `<@U0BOT1234> what is X` to the model.** The model then sees a weird token and sometimes comments on it. Strip `<@{context.bot_user_id}>` first.
3. **The mention reply goes to the channel, not a thread.** In a busy channel you want `say(text=..., thread_ts=event.get("thread_ts") or event["ts"])`.

And one design limit that the full project fixes: with the sync `App`, the `app_mention` handler calls OpenAI *inside* the listener. Bolt acks events before running listeners in Socket Mode, so you don't get retries, but each slow LLM call occupies a worker thread, there's no memory between messages, and the model can't use tools.

### 4.2 Script 2 — posting a message

```python
client = WebClient(token=slack_token)
client.chat_postMessage(channel=os.environ.get("SLACK_CHANNEL_ID"), text="@welcomeagenticapp Hello World from Python! 🎉")
```

This is pure Web API (#4 in section 3.4). It's the right tool for cron jobs, CI notifications and ETL alerts. Two things to know:

- **`@welcomeagenticapp` in plain text is not a mention.** Slack only treats `<@U0123ABCD>` (a user id in angle brackets) as a mention. Plain `@name` text is just text. If you want to mention someone from code, look up their user id (`users.lookupByEmail` is handy) and write `<@U…>`.
- **A bot can't trigger itself.** Even with a correct `<@BOTID>`, Slack doesn't deliver `app_mention` events for messages your own bot posted, and Bolt additionally ignores self-events by default. That's deliberate: it prevents infinite bot loops. If you want "code triggers the agent", call the agent directly — that's what `POST /api/agent/invoke` is for.

The server version of this script is `POST /api/slack/notify`.

### 4.3 From scripts to a service

| Script approach | What the full project does instead | Why |
|---|---|---|
| Sync `App`, threads | `AsyncApp` on FastAPI's event loop | One process serves Slack + REST; slow LLM calls don't hold threads |
| One OpenAI call per message | LangGraph agent with 6 tools and a loop | Answers grounded in tools, multi-step tasks |
| No memory | Checkpointer keyed by Slack thread | Follow-up questions work |
| Reply as new message | Placeholder → `chat.update` | One clean message per answer, visible progress |
| Plain text | Block Kit + mrkdwn conversion + chunking | Looks right, never exceeds Slack limits |
| Socket Mode only | Socket **or** HTTP via one env var | Dev on a laptop, prod behind a load balancer |
| Standalone sender script | `/api/slack/notify` and `/api/agent/invoke` | Other systems integrate over HTTP |

---

## 5. How Bolt for Python actually dispatches things

Bolt is a thin framework, but a few of its behaviours decide whether your bot feels solid or flaky.

### 5.1 The pipeline

```mermaid
flowchart LR
    R["Incoming request<br/>HTTP body or socket envelope"] --> V{"Signature valid?<br/>HTTP mode only"}
    V -- no --> X401[401]
    V -- yes --> UV{url_verification?}
    UV -- yes --> CH[echo challenge]
    UV -- no --> AU["Authorize<br/>auth.test once, cached"]
    AU --> SE{"Event from<br/>this bot itself?"}
    SE -- yes --> DROP[ignore]
    SE -- no --> GM[Global middleware]
    GM --> L1{listener 1 matches?}
    L1 -- yes --> RUN["run listener 1, stop"]
    L1 -- no --> L2{listener 2 matches?}
    L2 -- yes --> RUN2["run listener 2, stop"]
    L2 -- no --> LN["no match → 404, unhandled request warning"]
```

The step people miss: **Bolt runs the first listener that matches and stops.** Registration order is significant. In `app/slack/bolt_app.py` the order is:

1. `@app.message(regex "^hello|hi|hey")` — greeting
2. `@app.message(matchers=[human_dm_only])` — DMs go to the agent
3. `@app.event("message")` — everything else: no-op
4. `@app.event("app_mention")` — the agent, in a thread
5. `@app.command(...)`, `@app.action(...)` — different request types, no overlap

If #3 came before #1, the greeting would never fire. If you delete #3, every channel message logs an "Unhandled request" warning, because subscribing to `message.channels` means Slack sends you *every* message in channels the bot is in.

Note that a mention in a channel produces **two** events if you subscribe to both `app_mention` and `message.channels`: one of each. Here the `message` copy falls through to the no-op listener and the `app_mention` copy reaches the agent, so there's exactly one answer.

### 5.2 The helpers Bolt injects

Listener functions declare what they need by argument name, and Bolt fills them in:

| Argument | What it is | Use it for |
|---|---|---|
| `ack` | Acknowledge the request (HTTP 200 / socket ack). Optional args become the response body. | Commands, actions, views. **Must call within 3s.** Events are acked for you. |
| `say` | Post to the channel the event came from (`chat.postMessage` under the hood) | Quick replies |
| `respond` | POST to the request's `response_url` | Slash commands and button clicks. Valid 30 min, 5 uses. Works in channels the bot hasn't joined. Can be `ephemeral` or `in_channel`, can `replace_original`. |
| `client` | Full Web API client with the right token | Anything: `chat_update`, `reactions_add`, `views_open`, … |
| `event`, `message`, `command`, `action`, `body` | The parsed payload, or a slice of it | Reading input |
| `context` | Per-request data: `bot_user_id`, `team_id`, plus whatever middleware adds | Stripping the bot mention |
| `logger` | A logger scoped to the listener | Logging |

### 5.3 Acking: what `process_before_response` really does

`AsyncApp(process_before_response=False)` — the default, and what this project uses — means:

- **Events**: Bolt sends the 200/ack *first*, then runs your listener as a background task. Your listener can take 30 seconds and Slack is already happy.
- **Commands/actions**: Bolt sends the HTTP response as soon as you call `ack()`. Code after `ack()` keeps running. So `await ack("thinking…")` then `await agent.run(...)` is exactly right.

`process_before_response=True` is for FaaS (Lambda) where the process may be frozen right after the response; there you'd use Bolt's "lazy listeners" or hand work to a queue. Section 12 covers this.

### 5.4 Sync vs async

The original scripts use `App` (threads). FastAPI is asyncio, so the project uses `AsyncApp` with `AsyncWebClient` (built on aiohttp). Rules that follow from that:

- Every listener is `async def`, and every Slack call is `await client.…`.
- Never call blocking code (the sync `openai` client, `requests`, `time.sleep`) inside a listener — it freezes the whole server, including the other Slack events and REST requests. The agent uses `await llm.ainvoke(...)` for this reason.
- If you must call blocking code, wrap it: `await asyncio.to_thread(blocking_fn, ...)`.

---

## 6. Putting Bolt inside FastAPI

### 6.1 Why one process

You could run Bolt as one process and FastAPI as another. Running them together buys you:

- one deployable, one health check, one log stream;
- the Slack listeners and the REST endpoints share the **same** `AgentService`, which means the same conversation memory and the same tool backends;
- one asyncio event loop doing all the I/O, which is what agents mostly are: waiting on the LLM, waiting on tools, waiting on Slack.

### 6.2 The wiring, in `app/main.py`

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent = AgentService(llm or build_llm(settings), ...)     # 1. the brain
    bolt = create_bolt_app(settings, app.state.agent)                    # 2. Slack listeners, given the brain
    if settings.slack_mode == "socket":
        socket_handler = AsyncSocketModeHandler(bolt, settings.slack_app_token)
        await socket_handler.connect_async()                             # 3a. open WebSocket, return immediately
    else:
        app.state.bolt_http = AsyncSlackRequestHandler(bolt)             # 3b. adapter for POST /slack/events
    yield                                                                # ---- app serves requests here ----
    await socket_handler.close_async()                                   # 4. clean shutdown

@app.post("/slack/events")
async def slack_events(request: Request):
    return await request.app.state.bolt_http.handle(request)             # Bolt does verify/ack/dispatch
```

Two lines deserve a closer look.

**`connect_async()` vs `start_async()`.** `start_async()` connects and then awaits forever — fine in a script, but in a lifespan it would block startup and FastAPI would never start serving. `connect_async()` opens the socket, starts the receive loop as a background task on the current loop, and returns. That's the trick that lets Socket Mode live inside a web server.

**`AsyncSlackRequestHandler.handle(request)`** converts the Starlette request into a Bolt request (raw body for signature checks, headers, form or JSON), runs the Bolt pipeline from section 5.1, and converts Bolt's response back into a Starlette response. One route handles events, slash commands and interactivity, because Slack lets you point all three at the same URL and Bolt figures out the type from the body.

### 6.3 Request flow inside the process

```mermaid
flowchart TB
    subgraph proc["uvicorn process — one asyncio event loop"]
      direction TB
      subgraph ingress["Ingress"]
        WS["AsyncSocketModeHandler<br/>(background task, WebSocket)"]
        HTTP["POST /slack/events<br/>AsyncSlackRequestHandler"]
        REST["POST /api/agent/invoke<br/>POST /api/slack/notify<br/>GET /api/agent/threads/{id}"]
      end
      BOLT["Bolt AsyncApp<br/>verify → authorize → match listener → ack"]
      LIS["Listeners<br/>hello · DM · app_mention · /agent · /agent-add · buttons"]
      SVC["AgentService.run()<br/>per-thread lock · timing · tool trace"]
      G["LangGraph compiled graph<br/>guardrail → agent ⇄ tools → finalize"]
      CK[("Checkpointer<br/>SQLite local · Postgres cloud")]
      WC["AsyncWebClient<br/>chat.postMessage / chat.update / reactions"]
      WS --> BOLT
      HTTP --> BOLT
      BOLT --> LIS
      LIS --> SVC
      REST --> SVC
      SVC --> G
      G <--> CK
      LIS --> WC
      REST --> WC
    end
    LLM[("OpenAI-compatible LLM")]
    SLACK[(Slack)]
    G -- "ainvoke with tools" --> LLM
    WC --> SLACK
    SLACK --> WS
    SLACK --> HTTP
```

### 6.4 Workers and replicas

Run **one uvicorn worker per container** (the Dockerfile does) and scale with containers. Conversation memory, votes and Slack event claims all live in the database (SQLite locally, Cloud SQL in the cloud), so any instance can answer any thread; section 18.4 covers what is still per-instance.

---

## 7. LangGraph from first principles

### 7.1 Why a graph and not a single LLM call

The original `/welcomeagentanalyze` is one prompt → one answer. That can't look anything up, can't take an action, and can't recover from "I need more information". An agent is an LLM in a loop: look at the situation, maybe call a tool, look at the result, decide again. You can write that loop by hand in 30 lines. LangGraph gives you the same loop as an explicit graph, plus things you'd otherwise build badly yourself:

- **State** with merge rules, so nodes return partial updates and don't clobber each other.
- **Checkpointing**, so a conversation (or a paused run) is saved after every step and can be resumed by id. This is where Slack thread memory comes from.
- **Conditional routing** you can see and test, instead of `if/else` buried in a while loop.
- **Recursion limits**, so a confused model can't loop forever on your API bill.
- **Interrupts** for human-in-the-loop (section 14.4), streaming of intermediate steps, and subgraphs for multi-agent setups.

### 7.2 The building blocks

**State.** A typed dict that every node reads and writes. Ours:

```python
class AgentState(MessagesState):           # gives: messages: Annotated[list[AnyMessage], add_messages]
    user_id: str                            # from Slack, trusted
    channel_id: str
    source: str                             # slack_mention | slack_dm | slash_command | api
    blocked_reason: str                     # set by guardrail
    final_answer: str                       # set by finalize
```

**Reducers.** Each key has a rule for combining a node's update with the existing value. Plain keys use "overwrite". `messages` uses `add_messages`, which appends (and replaces a message if the ids match). That's why the agent node can return `{"messages": [response]}` — one message — and the state ends up with the whole conversation.

**Nodes.** Plain functions (sync or async) `state -> partial update`:

```python
async def agent(state: AgentState) -> dict:
    response = await llm_with_tools.ainvoke([system_prompt, *history])
    return {"messages": [response]}
```

**Edges.** Fixed (`tools → agent`) or conditional (a function that looks at the state and returns the name of the next node).

**Compile.** `builder.compile(checkpointer=...)` validates the graph and returns a runnable with `invoke/ainvoke/stream/astream/get_state`.

### 7.3 This project's graph

```mermaid
flowchart TD
    START((START)) --> guardrail
    guardrail{{"guardrail<br/>empty? too long? injection phrases?"}}
    guardrail -- "blocked_reason set" --> END1((END))
    guardrail -- ok --> agent
    agent["agent<br/>LLM + 6 tools bound<br/>system prompt with Slack user/channel"]
    agent -- "AIMessage has tool_calls" --> tools
    tools["tools (ToolNode)<br/>runs every requested tool<br/>returns ToolMessages"]
    tools --> agent
    agent -- "no tool_calls" --> finalize
    finalize["finalize<br/>final_answer = last AI text"]
    finalize --> END2((END))
```

- **guardrail** rejects empty input, giant pastes and the most obvious prompt-injection phrases without spending a token. It also resets `final_answer`/`blocked_reason`, because those are remembered from the previous turn by the checkpointer.
- **agent** builds the system prompt with who/where/when, trims history to the last 30 messages (never starting on an orphaned tool result, which OpenAI rejects), calls the model with tools bound.
- **tools** is LangGraph's prebuilt `ToolNode`: for each `tool_call` on the last AIMessage it runs the matching tool and appends a `ToolMessage` with the same `tool_call_id`. If the model asks for three tools at once, all three run.
- **finalize** copies the answer text into `final_answer` so callers don't need to know message internals.

### 7.4 The ReAct loop, traced

Question: *"VPN keeps dropping, open a ticket"*, from user `U42`.

| Step | Node | `messages` after the step |
|---|---|---|
| 0 | input | `Human("VPN keeps dropping, open a ticket")` |
| 1 | guardrail | unchanged (ok) |
| 2 | agent | + `AI(tool_calls=[search_knowledge_base(query="vpn drops")])` |
| 3 | tools | + `Tool("[KB-001] VPN access … switch the protocol to IPSec …")` |
| 4 | agent | + `AI(tool_calls=[create_ticket(title=…, description=…, priority="P2")])` |
| 5 | tools | + `Tool("Created TCK-7F3A21 (P2) … for requester U42")` |
| 6 | agent | + `AI("Opened TCK-7F3A21 … try IPSec (KB-001)")` — no tool calls |
| 7 | finalize | `final_answer` = that text |

Each "step" is one superstep; `recursion_limit` (here `AGENT_MAX_STEPS=12`) caps the total. If the limit is hit, `AgentService` catches `GraphRecursionError` and replies with a friendly "I went around in circles" instead of a stack trace.

### 7.5 Tools, and what the model actually sees

```python
@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly, e.g. "(1250 * 0.18) + 40".
    Use this for any math instead of doing it in your head."""
```

`@tool` reads the name, type hints and docstring and produces a JSON schema. `bind_tools` sends that schema to the model with each request. **The docstring is prompt** — it's how the model decides when to use the tool. Write it for the model: what it does, when to use it, what comes back.

Two safety habits shown in `tools.py`:

- **Never `eval` model output.** The calculator parses the expression with `ast` and only allows numbers and arithmetic operators. `__import__('os')` gets an error string, not a shell.
- **Return errors as strings, don't raise.** The model reads "Error: unknown timezone 'Mars/Olympus'" and tries again with a valid one. An exception would end the turn.

### 7.6 InjectedState — values the model can't fake

```python
@tool
def create_ticket(
    title: str,
    description: str,
    priority: Literal["P1", "P2", "P3", "P4"],
    requester: Annotated[str, InjectedState("user_id")],
) -> str:
```

`requester` is **removed from the schema the model sees** and filled at run time from `state["user_id"]`, which came from Slack's verified event payload. If someone writes "open a ticket and set the requester to the CEO", the model has no parameter to put that in. This is the general pattern for anything identity- or permission-shaped: tenant ids, user ids, auth tokens, row-level filters. Don't ask the LLM for them; inject them. (`InjectedToolArg` and `InjectedStore` follow the same idea for other sources.)

### 7.7 Memory: checkpointer + thread_id

```python
config = {"configurable": {"thread_id": "slack:C0ITHELP:1700000000.000100"}}
await graph.ainvoke({"messages": [HumanMessage("and how many roll over?")]}, config)
```

With a checkpointer, LangGraph saves the full state after every step under that `thread_id`. The next `ainvoke` with the same id starts from the saved state, and `add_messages` appends the new question to the old conversation. Different id, fresh conversation.

The interesting design decision is **what Slack concept maps to a thread_id**:

| Slack surface | thread_id | Effect |
|---|---|---|
| @mention in a channel | `slack:{channel}:{thread_ts or ts}` | Each Slack thread is its own conversation. Mention again inside the thread → it remembers. |
| DM, top level | `slack:dm:{channel}` | One running chat per user, like a messenger |
| DM, inside a thread | `slack:{channel}:{thread_ts}` | Side conversations stay separate |
| `/agent` | `slack:cmd:{channel}:{user}` | Per user per channel |
| REST | caller's `thread_id` or `api:{uuid}` | Caller decides |

The checkpointer follows `DATABASE_URL` (`app/db/checkpointer.py`): `AsyncSqliteSaver` on your laptop, `AsyncPostgresSaver` on Cloud Run, `InMemorySaver` only in unit tests. Restart the app, or let Cloud Run replace the instance, and the thread still remembers — `tests/test_persistence.py` proves it for both databases.

You can inspect any conversation through the API: `GET /api/agent/threads/slack:C0ITHELP:1700000000.000100`.

### 7.8 Concurrency: one lock per thread

Two people replying in the same Slack thread at the same moment would start two graph runs from the same checkpoint, and the later write would silently drop the other's messages. `AgentService` keeps an `asyncio.Lock` per `thread_id`: runs in the same thread queue up, runs in different threads go in parallel. (With multiple replicas, use a distributed lock or route a thread to one replica; see section 12.)

---

## 8. Architecture diagrams

All diagrams are Mermaid, so they render on GitHub, GitLab and in most IDEs.

### 8.1 System context

```mermaid
flowchart LR
    People(["People in Slack<br/>channels · DMs · slash commands"])
    Systems(["Your systems<br/>CI · cron · ETL · other services"])
    subgraph Svc["Agent service (this repo) — FastAPI"]
      direction TB
      Slack_In["Slack ingress<br/>Socket Mode or /slack/events"]
      API["REST API<br/>/api/*"]
      Agent["LangGraph agent"]
    end
    Slack[(Slack platform)]
    LLM[("LLM provider<br/>OpenAI / OpenRouter / LiteLLM / Ollama")]
    KB[("Knowledge base<br/>demo: in-memory")]
    Tix[("Ticketing<br/>demo: in-memory")]
    PG[("PostgreSQL<br/>agent_runs · agent_feedback")]

    People --> Slack
    Slack <--> Slack_In
    Systems --> API
    Slack_In --> Agent
    API --> Agent
    Agent --> LLM
    Agent --> KB
    Agent --> Tix
    API -- "notify / post answer" --> Slack
    Agent -- "record run" --> PG
    Slack_In -- "👍/👎 votes" --> PG
```

### 8.2 Layers and dependencies

```mermaid
flowchart TB
    subgraph L1["Transport layer"]
      main["main.py<br/>lifespan, /slack/events, /health"]
      routes["api/routes.py"]
    end
    subgraph L2["Channel adapter layer (Slack-specific)"]
      bolt["slack/bolt_app.py"]
      blocks["slack/blocks.py"]
      fmt["slack/formatting.py"]
    end
    subgraph L3["Application service"]
      svc["services/agent_service.py"]
      dedup["services/dedup.py"]
    end
    subgraph L4["Agent (framework-agnostic of Slack)"]
      graphmod["agent/graph.py"]
      state["agent/state.py"]
      tools["agent/tools.py"]
      prompts["agent/prompts.py"]
      llm["agent/llm.py"]
    end
    main --> bolt
    main --> routes
    routes --> svc
    routes --> blocks
    bolt --> svc
    bolt --> dedup
    bolt --> blocks
    blocks --> fmt
    svc --> graphmod
    graphmod --> state
    graphmod --> tools
    graphmod --> prompts
    main --> llm
```

Arrows only point downward. Nothing in `agent/` imports from `slack/` or `api/`.

### 8.3 Sequence — @mention in a channel

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant S as Slack
    participant B as Bolt AsyncApp
    participant L as handle_mention
    participant A as AgentService
    participant G as LangGraph
    participant M as LLM
    participant T as Tools

    U->>S: "@Ops Buddy VPN keeps dropping, open a ticket"
    S->>B: event_callback (app_mention) via socket or POST /slack/events
    B-->>S: ack (well under 3s)
    B->>L: dispatch (first matching listener)
    L->>L: dedup on channel+ts, strip the bot mention
    L->>S: reactions.add :eyes:
    L->>S: chat.postMessage(thread_ts) "⏳ working on it…"
    S-->>L: ts of placeholder
    L->>A: run(text, thread_id=slack:C:ts, user_id=U42)
    A->>G: ainvoke(state, config{thread_id})
    G->>G: load checkpoint for thread
    G->>G: guardrail ok
    G->>M: messages + tool schemas
    M-->>G: tool_call search_knowledge_base
    G->>T: search_knowledge_base("vpn drops")
    T-->>G: KB-001 text
    G->>M: + tool result
    M-->>G: tool_call create_ticket(title, priority)
    G->>T: create_ticket(..., requester=U42 injected)
    T-->>G: TCK-7F3A21
    G->>M: + tool result
    M-->>G: final text (no tool calls)
    G->>G: finalize, save checkpoint
    G-->>A: state
    A-->>L: AgentResult(answer, tool_calls, ms)
    L->>S: chat.update(placeholder ts, blocks)
    L->>S: reactions.add ✅, reactions.remove 👀
    S-->>U: answer appears in thread
```

### 8.4 Sequence — slash command

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant S as Slack
    participant B as Bolt
    participant H as handle_agent_command
    participant A as AgentService
    participant R as response_url

    U->>S: /agent how many PTO days per quarter?
    S->>B: command payload (form-encoded)
    B->>H: dispatch
    H-->>S: ack("⏳ Thinking about …") — ephemeral, only U sees it
    H->>A: run(question, thread_id=slack:cmd:C:U)
    A-->>H: AgentResult
    H->>R: POST response_type=in_channel, blocks
    R-->>S: message posted in channel
    S-->>U: everyone sees question + answer
```

`response_url` is why `/agent` works in channels where the bot hasn't been invited — the URL itself is the permission.

### 8.5 Sequence — your system calls the agent and posts to Slack

```mermaid
sequenceDiagram
    autonumber
    participant C as CI job / service
    participant F as FastAPI /api/agent/invoke
    participant A as AgentService
    participant W as AsyncWebClient
    participant S as Slack

    C->>F: POST {message, thread_id?, post_to_slack_channel: "C0ALERTS"}
    F->>F: check X-API-Key
    F->>A: run(message, thread_id, source=api)
    A-->>F: AgentResult
    F->>W: chat.postMessage(channel, blocks)
    W->>S: HTTPS
    S-->>W: ok, ts
    F-->>C: {answer, tool_calls, duration_ms, slack_message_ts}
```

### 8.6 Where state lives

```mermaid
flowchart LR
    subgraph per_request["Per request (gone when it ends)"]
      ev[Slack event / command payload]
      res[AgentResult]
    end
    subgraph per_thread["Per conversation (checkpointer, keyed by thread_id)"]
      msgs["messages: Human / AI / Tool"]
      meta[user_id · channel_id · source · final_answer]
    end
    subgraph per_process["Per process (in memory — move out to scale)"]
      locks[asyncio locks per thread]
      seen[dedup TTL cache]
      tickets[demo ticket store]
    end
    subgraph external["External / durable"]
      slackmsgs[("Slack messages and threads")]
      llmprov[(LLM provider)]
    end
```

---

## 9. One request, every hop, with real payloads

Here's a single `/agent` call followed all the way through, so you can match each log line to a piece of code.

**Hop 1 — Slack → you.** User types `/agent how many PTO days a quarter?` in `#general`. Slack sends (form-encoded, shown decoded):

```
command=/agent
text=how many PTO days a quarter?
user_id=U07ANJI
channel_id=C0GENERAL
team_id=T0123ABCD
response_url=https://hooks.slack.com/commands/T0123ABCD/123/abc
trigger_id=123.456.abc
```

**Hop 2 — Bolt.** Verifies the signature (HTTP mode), authorizes (cached `auth.test`), sees `command=/agent`, matches `handle_agent_command`.

**Hop 3 — ack.** `await ack(":hourglass_flowing_sand: Thinking about: _how many PTO days a quarter?_")` → the HTTP response body is `{"text": "…"}`, which Slack shows as an ephemeral message to Anji only. Elapsed: a few milliseconds.

**Hop 4 — AgentService.** `thread_id = "slack:cmd:C0GENERAL:U07ANJI"`. Takes that thread's lock. Calls:

```python
graph.ainvoke(
  {"messages": [HumanMessage("how many PTO days a quarter?")],
   "user_id": "U07ANJI", "channel_id": "C0GENERAL", "source": "slash_command"},
  {"configurable": {"thread_id": "slack:cmd:C0GENERAL:U07ANJI"}, "recursion_limit": 12},
)
```

**Hop 5 — guardrail** passes. **agent** sends to the LLM (abridged):

```json
{
  "model": "gpt-4o",
  "messages": [
    {"role": "system", "content": "You are Ops Buddy ... Slack user id: U07ANJI ... Came in through: slash_command ..."},
    {"role": "user", "content": "how many PTO days a quarter?"}
  ],
  "tools": [
    {"type": "function", "function": {"name": "calculator", "description": "Evaluate an arithmetic expression exactly...", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "search_knowledge_base", "...": "..."}},
    {"type": "function", "function": {"name": "create_ticket", "parameters": {"properties": {"title": {}, "description": {}, "priority": {"enum": ["P1","P2","P3","P4"]}}}}}
  ]
}
```

Notice `create_ticket` has no `requester` property — `InjectedState` hid it.

**Hop 6 — model asks for a tool:** `search_knowledge_base({"query": "PTO days per year"})` → `[KB-003] Paid time off … 1.75 PTO days per month (21 days per year) …`

**Hop 7 — model asks for another:** `calculator({"expression": "21 / 4"})` → `5.25`

**Hop 8 — model answers** with no tool calls: `"You get **21 days** a year, which is about **5.25 days per quarter** (KB-003)."` → `finalize` → checkpoint saved.

**Hop 9 — formatting.** `to_mrkdwn` turns `**21 days**` into `*21 days*`. `answer_blocks` builds: a `section` with the text, a `context` line `🛠 search_knowledge_base, calculator · ⏱ 2.3s`, and an `actions` block with 👍/👎.

**Hop 10 — deliver.** `await respond(response_type="in_channel", blocks=[context "<@U07ANJI> asked: …", *answer_blocks], text=fallback)` → POST to `response_url` → everyone in `#general` sees it.

**Hop 11 — feedback.** When the answer was produced, `AgentService` gave it a `run_id` (a UUID) and wrote a row to `agent_runs`; the 👍/👎 buttons carry that `run_id` as their `value`. Someone clicks 👍. Slack sends a `block_actions` payload with `action_id: "upvote"`. `handle_upvote` acks, `feedback_from_body` pulls out run, user, channel and message ts, the store upserts one row into `agent_feedback`, and the clicker gets an ephemeral "Thanks! This answer: 👍 1 👎 0". Section 17 has the whole story.

---

## 10. Code walkthrough, file by file

Read in this order and the whole thing clicks.

### `app/config.py`
One `Settings` class, loaded from env/.env, cached with `lru_cache`. Notable bits: `slack_mode: Literal["socket","http"]` (typos fail at boot), `slack_app_token` accepts both `SLACK_APP_TOKEN` and `SLACK_SOCKET_TOKEN` via `AliasChoices`, slash command names are settings because they're global per workspace and you may already have `/agent` taken, and `openai_base_url` lets you point at any OpenAI-compatible server.

### `app/agent/tools.py`
Six tools and two fake backends. The `KNOWLEDGE_BASE` list and `TicketStore` class are the seams where real systems go: replace `search_knowledge_base`'s body with a vector search or Confluence API call, replace `TicketStore` with a Jira/ServiceNow client. The tool signatures — and therefore the graph — stay the same.

### `app/agent/state.py`, `prompts.py`, `llm.py`
Covered in section 7. The prompt is deliberately Slack-aware (short, mrkdwn, no tables) and tells the model *when* to use each tool, because tool descriptions alone aren't always enough to stop a model from guessing policy.

### `app/agent/graph.py`
`build_graph(llm, checkpointer)` — takes the model as a parameter (that's what makes the fake-LLM tests possible), binds tools, defines four nodes and two routers, compiles. `_window()` keeps history bounded and valid. The agent node is `async` so the LLM call never blocks the event loop.

### `app/services/agent_service.py`
The seam. `run(text, thread_id, user_id, channel_id, source)` → `AgentResult(answer, tool_calls, duration_ms, blocked)`. Owns the checkpointer, the per-thread locks and the recursion-limit fallback, and extracts a tool trace for *this turn only* (walking back to the latest human message), which is what the 🛠 line in Slack and the `tool_calls` field in the REST response show.

### `app/services/dedup.py`
`TTLDeduper.first_time(key)` — a dict of `key → first_seen`, pruned when it grows. Ten lines that remove a whole class of "the bot answered twice" bugs.

### `app/slack/formatting.py` and `blocks.py`
`to_mrkdwn()` splits on fenced code blocks, converts everything outside them (headings, bold, links, strike, bullets, escaping), leaves code alone. `chunk()` splits at line boundaries under 2900 chars. `answer_blocks()` assembles the message; `thinking_blocks()` is the placeholder.

### `app/slack/bolt_app.py`
`create_bolt_app(settings, agent)` returns a configured `AsyncApp`. The agent comes in as an argument — the listeners close over it, there are no globals. The shared helper `run_agent_in_thread` implements the placeholder pattern: react 👀 → post placeholder → run agent → `chat.update` → react ✅ (or ❌ and an apology on failure) → remove 👀. Reactions are wrapped so a missing `reactions:write` scope never breaks an answer.

### `app/api/routes.py` and `schemas.py`
Three endpoints, an optional API key dependency (`X-API-Key`), proper HTTP errors when Slack refuses something (`not_in_channel` → 502 with the Slack error text so you can actually debug it).

### `app/main.py`
`create_app(settings=None, llm=None)` — the factory. Production calls it with nothing (env vars + real OpenAI). Tests call it with explicit settings and a scripted model. The lifespan builds everything in dependency order and tears down the socket on shutdown.

---

## 11. Setting up the Slack app, step by step

### 11.1 Socket Mode (development, or internal apps behind a firewall)

1. <https://api.slack.com/apps> → **Create New App** → **From a manifest** → workspace → paste `slack_manifests/manifest.socket.yaml` → **Next** → **Create**.
2. **Basic Information → App-Level Tokens → Generate Token and Scopes** → name it `socket`, add `connections:write` → **Generate** → copy `xapp-…` into `SLACK_APP_TOKEN`.
3. **Install App → Install to Workspace** → **Allow** → copy the `xoxb-…` **Bot User OAuth Token** into `SLACK_BOT_TOKEN`.
4. `.env`: `SLACK_MODE=socket`, plus `OPENAI_API_KEY`.
5. `make run`. Log line: `Slack connected via Socket Mode`.
6. Slack: `/invite @Ops Buddy` in a test channel. Say `hello`. Mention it. DM it (if the Messages tab says "sending messages to this app has been turned off", check **App Home → Show Tabs → Messages Tab** and the "Allow users to send…" checkbox, then reload Slack).

Prefer clicking through the UI instead of using a manifest? You need: Socket Mode on; the bot scopes from section 3.3; Event Subscriptions on with bot events `app_mention`, `message.im`, `message.channels`, `message.groups`; Interactivity on; the two slash commands; App Home messages tab enabled. Then (re)install.

### 11.2 HTTP mode (production)

1. Get a public HTTPS URL. For local testing: `make run-http` in one terminal and `ngrok http 8000` in another; note the `https://….ngrok-free.app` URL.
2. Create the app from `slack_manifests/manifest.http.yaml`, replacing every `https://YOUR-HOST` with your URL. (Or, on an existing app: turn Socket Mode off and fill in **Event Subscriptions → Request URL**, **Interactivity → Request URL**, and each slash command's **Request URL**, all as `https://<host>/slack/events`.)
3. Slack immediately sends a `url_verification` request. Your server must be running with the right `SLACK_SIGNING_SECRET` — Bolt answers the challenge and the Request URL turns green: **Verified**.
4. **Basic Information → App Credentials → Signing Secret** → `SLACK_SIGNING_SECRET`. Install the app, copy `SLACK_BOT_TOKEN`. `SLACK_APP_TOKEN` isn't needed.
5. `.env`: `SLACK_MODE=http`.

### 11.3 Finding ids

- **Channel id**: open the channel → click its name → bottom of the *About* tab. Or right-click → *Copy link*; the `C…` at the end is the id.
- **Your user id**: click your avatar → *Profile* → ⋮ → *Copy member ID*.
- **Bot user id**: logged at startup by Bolt (`auth.test`), or in the app's **App Home** page.

### 11.4 Running with Docker

```bash
docker compose up --build                 # app on SQLite, data in a named volume
docker compose --profile pg up --build    # plus a local Postgres, to rehearse the cloud setup
curl localhost:8000/health
```

Socket Mode works the same inside a container (the connection is outbound). For HTTP mode, put the container behind your load balancer / ingress and point the Slack Request URLs at it.

---

## 12. Taking it to production

The demo is honest about its shortcuts. Here's how to remove each one.

### 12.1 Durable memory

Done: the LangGraph checkpointer is stored in the same database as everything else (SQLite locally, Cloud SQL on Cloud Run). See `app/db/checkpointer.py` and section 18. Still worth adding: a retention job — Slack threads go cold after a few days, so old checkpoints can be pruned.

### 12.2 More than one replica

| Concern | Single process | On Cloud Run (this repo) |
|---|---|---|
| Conversation memory | `InMemorySaver` | Postgres checkpointer — any instance continues any thread |
| De-dup | `TTLDeduper` dict | `TTLDeduper` first, then an `INSERT … ON CONFLICT DO NOTHING` into `slack_event_claims`: only the instance whose insert lands proceeds |
| Per-thread lock | `asyncio.Lock` | Still per instance. Two people replying in the same thread at the same instant on two instances can race; rare for a Slack bot. Fix with `pg_advisory_xact_lock(hashtext(thread_id))` if it matters |
| Socket Mode | 1 connection | Needs `min_instances >= 1`; HTTP mode (the default) scales to zero |
| Demo ticket store | dict | Still a dict per instance: swap for your real ticketing API |

### 12.3 Long tasks and queues

`process_before_response=False` runs the listener after the ack, inside the web process. That's fine for answers that take seconds. For work that takes minutes (big document analysis, multi-agent research), or on serverless platforms that freeze the process after the response, decouple:

```mermaid
flowchart LR
    S[(Slack)] --> W["Web process<br/>ack + enqueue {channel, thread_ts, text, user}"]
    W --> Q[["Queue<br/>SQS · Pub/Sub · Redis · Celery"]]
    Q --> K["Worker(s)<br/>AgentService.run → chat.update"]
    K --> S
```

On AWS Lambda specifically, Bolt has *lazy listeners* (`app.command("/agent")(ack=…, lazy=[…])`) that re-invoke the function asynchronously for the slow part.

### 12.4 Streaming answers into Slack

Today the user sees "working on it…" and then the full answer. To show tokens as they arrive you have two options. The portable one: `async for chunk in graph.astream(..., stream_mode="messages")`, accumulate, and call `chat.update` on the placeholder every ~1 second (respecting rate limits). Slack has also added dedicated streaming methods for AI apps (`chat.startStream` / `chat.appendStream` / `chat.stopStream`, which take Markdown); check the current Slack docs and your SDK version before relying on them, as they're tied to Slack's AI-app features.

### 12.5 Security checklist

- **Verify every HTTP request** (Bolt does, as long as `SLACK_SIGNING_SECRET` is set — `main.py` refuses to start HTTP mode without it).
- **Secrets** in a secret manager (Vault, AWS Secrets Manager, GCP Secret Manager), injected as env vars. Never in the image, never in git. `.env` is in `.gitignore` and `.dockerignore`.
- **Least-privilege scopes.** Drop `chat:write.public` if you don't need `/api/slack/notify` in uninvited channels; drop `channels:history` if you don't need the `hello` listener (then the bot can't read channel messages that don't mention it).
- **Identity comes from Slack, not from the model.** `InjectedState` for user ids; check permissions inside tools (e.g. "can U42 close tickets for other people?") rather than trusting the prompt.
- **Prompt injection is a when, not an if.** Anything the agent reads — KB articles, ticket descriptions, other people's messages — can contain instructions. The guardrail's phrase check is a speed bump, not a wall. The real defences are tools that can't do much damage, confirmation steps for destructive actions (section 14.4), and output that never contains secrets because the agent never had them.
- **Protect the REST API.** Set `API_KEY`, or better, put it behind your gateway/mTLS/OIDC.
- **Data handling.** Slack messages may contain PII or confidential data and you're sending them to an LLM provider. Know your provider's retention terms; use a gateway (LiteLLM, an internal proxy) if you need redaction, logging or a private endpoint.
- **Don't log full message text at INFO** in regulated environments. The service logs thread ids, tool names and timings, not content.

### 12.6 Observability

What to measure from day one: latency per turn and per tool, tool error rate, recursion-limit hits, guardrail blocks, 👍/👎 ratio, LLM tokens/cost per thread. Practical options:

- **LangSmith**: set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`; every graph run shows up as a trace with each node, tool call and LLM request.
- **OpenTelemetry**: instrument FastAPI (`opentelemetry-instrumentation-fastapi`) and LangChain/LangGraph (e.g. Monocle, OpenLLMetry) and send to whatever backend you already use.
- **Feedback loop**: every 👍/👎 is already a row in Postgres joined to the exact question, answer and tool calls (section 17). Add the trace id to `agent_runs` and you have a labelled eval set built by your users.

### 12.7 Cost and model choice

The agent sends the system prompt + up to 30 messages + 6 tool schemas on every step, and a tool-using turn is usually 2–4 steps. Cheap wins: a smaller model for simple turns (route on the guardrail), shorter history window, prompt caching where your provider supports it, and `max_tokens` on the model. Because `llm.py` speaks the OpenAI protocol, switching to OpenRouter, a LiteLLM proxy with fallbacks, or a local model is an env-var change.

---

## 13. Testing without Slack or OpenAI

```bash
make test        # 49 passed, 8 skipped (the PostgreSQL ones)
make db          # start Postgres in Docker
make test-pg     # 57 passed: same, plus PostgreSQL (store, checkpointer, restart round-trip)
```

| File | What it proves |
|---|---|
| `test_tools_and_formatting.py` | Calculator is exact and refuses code; KB search ranks the right article; tickets round-trip; Markdown → mrkdwn (bold, links, headings, bullets, escaping, code blocks untouched); chunking respects limits; de-dup works |
| `test_agent_graph.py` | Full ReAct loop with two tool calls; `requester` comes from state, not the model; system prompt carries Slack context; memory within a thread; isolation across threads; guardrail blocks without calling the LLM; recursion limit gives a friendly answer |
| `test_api.py` | `/health`; invoke returns tool trace; history endpoint shows user/assistant/tool/assistant; API key enforced; 503 when Slack isn't configured |
| `test_slack.py` | Hand-signed HTTP requests: `url_verification` challenge, bad signature → 401, `/agent-add` over HTTP returns `5.5`. Listener logic: an `app_mention` event produces placeholder-in-thread → `chat.update` with converted text and tool line; the LLM received text with the bot mention stripped; a duplicate event produces exactly one answer |
| `test_persistence.py` | Restart the whole app on the same database and check the thread still remembers and the votes are still there (SQLite always, Postgres with `TEST_DATABASE_URL`, plus the Cloud Run unix-socket DSN with `POSTGRES_SOCKET_DIR`); `test`/`prod` refuse SQLite; an `sk-or-` key selects OpenRouter |
| `test_feedback.py` | Parsing a real `block_actions` payload; bad/missing run ids rejected; the full vote lifecycle (new → same click is a no-op → switch 👍 to 👎 → second voter) on the in-memory, SQLite **and** Postgres stores; cross-instance Slack event claims; unknown run → error; idempotent run inserts; summary maths; Slack end to end (mention → answer with `upvote`/`downvote` buttons carrying `run_id` → two clicks → two rows + ephemeral tallies); REST `/api/feedback`, `/api/runs/{id}`, `/api/feedback/summary` incl. 404/422 |

Two techniques worth stealing:

- **`ScriptedChatModel`** (`tests/fakes.py`) — a `BaseChatModel` that returns pre-written `AIMessage`s (including tool calls) in order and records what it was sent. You test the graph's wiring, tools and memory deterministically, for free, in milliseconds.
- **Patching `AsyncWebClient` at class level** — Bolt hands each listener a per-request copy of the client, so stubbing an instance doesn't work; `monkeypatch.setattr(AsyncWebClient, "chat_update", fake)` does. Then you dispatch a raw event with `await bolt.async_dispatch(AsyncBoltRequest(body=..., mode="socket_mode"))` and assert on the recorded calls.

For a manual end-to-end check, `scripts/demo_requests.sh` hits every REST endpoint, and the table in section 1 covers every Slack surface.

---

## 14. Extending it

### 14.1 Add a tool

```python
# app/agent/tools.py
@tool
def get_oncall(team: str) -> str:
    """Return who is on call right now for a team, e.g. "payments" or "platform"."""
    return pagerduty_client.current_oncall(team)        # your API

ALL_TOOLS = [..., get_oncall]
```

That's it — the graph binds `ALL_TOOLS`, `ToolNode` runs it. Add a line to the system prompt if the model should prefer it for certain questions, and a test with `ScriptedChatModel` that calls it.

### 14.2 Use tools from MCP servers

If you already expose systems through MCP servers (FastMCP etc.), load them as LangChain tools with `langchain-mcp-adapters` and append them to the tool list:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

mcp = MultiServerMCPClient({
    "tickets": {"url": "http://tickets-mcp:8001/mcp", "transport": "streamable_http"},
    "docs":    {"command": "python", "args": ["docs_server.py"], "transport": "stdio"},
})
mcp_tools = await mcp.get_tools()                      # in lifespan
graph = build_graph(llm, checkpointer, extra_tools=mcp_tools)   # add the parameter; bind ALL_TOOLS + extra
```

### 14.3 A smarter router up front

The `guardrail` node is a natural place to grow a classifier: a cheap model labels the request (`chitchat | policy | ticket | out_of_scope`) and the conditional edge sends chitchat straight to a no-tools reply, policy questions to a RAG subgraph, and so on. Same pattern, more nodes:

```mermaid
flowchart LR
    START((START)) --> classify
    classify -- chitchat --> small_talk --> END((END))
    classify -- policy --> rag_subgraph --> finalize
    classify -- action --> agent
    agent <--> tools
    agent --> finalize --> END
```

### 14.4 Human-in-the-loop approval with Slack buttons

For anything destructive (closing incidents, refunds, deploys) let the graph pause and ask. LangGraph's `interrupt()` stops the run and saves the checkpoint; resuming with `Command(resume=...)` continues from that exact point — even hours later, even on another replica if the checkpointer is shared.

```python
from langgraph.types import interrupt, Command

def approve_p1(state):                                  # node placed before tools when a P1 ticket is requested
    decision = interrupt({"question": "Open a P1? This pages on-call.", "ticket": ...})
    return {"approved": decision == "approve"}
```

```python
# Slack side
result = await graph.ainvoke(inputs, config)
if "__interrupt__" in result:                           # paused
    await client.chat_postMessage(channel=..., thread_ts=..., blocks=[
        section("This will page on-call. Approve?"),
        actions(button("Approve", action_id="hitl_approve", value=thread_id),
                button("Reject",  action_id="hitl_reject",  value=thread_id)),
    ])

@app.action("hitl_approve")
async def on_approve(ack, action):
    await ack()
    await graph.ainvoke(Command(resume="approve"), {"configurable": {"thread_id": action["value"]}})
```

Check in the action handler that the person clicking is allowed to approve (`body["user"]["id"]`), not just anyone in the channel.

### 14.5 Other ideas

- **App Home tab** (`@app.event("app_home_opened")` + `views.publish`): show the user's open tickets and recent conversations.
- **Modals** (`views.open` with the command's `trigger_id`): structured ticket forms instead of free text.
- **Shortcuts**: a message shortcut "Ask Ops Buddy about this message" that sends any message (and its thread) to the agent.
- **Files**: `message` events with `files` → download with the bot token → run them through a document parser (Docling, for example) as a tool.
- **Multi-workspace**: swap the single bot token for Bolt's OAuth flow + an installation store; the listeners don't change.

---

## 15. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Nothing happens when you mention the bot | Bot not in channel; `app_mention` event not subscribed; app not reinstalled after adding scopes | `/invite @Ops Buddy`; check Event Subscriptions; **reinstall** |
| `hello` works only after mentioning | `message.channels` not subscribed or `channels:history` missing | Add both, reinstall |
| DMs: "Sending messages to this app has been turned off" | Messages tab disabled | App Home → Messages Tab on + allow sending; restart Slack client |
| Bot answers 2–4 times | Slow ack → Slack retries; or two listeners both answering | This repo acks first and de-dups by `channel:ts`; if you modified listeners, check order (section 5.1) |
| `/agent failed with the error "dispatch_failed"` | Server not reachable / not running / wrong Request URL; in Socket Mode, app not connected | Check logs for `Slack connected via Socket Mode` or ngrok URL |
| `operation_timeout` on a slash command | `ack()` called after slow work | `ack()` first, always |
| `not_in_channel` | Bot isn't a member | Invite it, or add `chat:write.public` (public channels only) |
| `channel_not_found` | Wrong id, or private channel the bot isn't in | Use the `C…` id; invite the bot |
| `missing_scope` (error lists `needed`) | Scope not granted | Add it, **reinstall**, update `SLACK_BOT_TOKEN` if it changed |
| `invalid_auth` / `not_authed` | Wrong token type (`xapp` where `xoxb` expected), revoked token, wrong workspace | Re-copy tokens; xoxb for `SLACK_BOT_TOKEN` |
| HTTP mode: Request URL won't verify | Server not running, wrong signing secret, URL path wrong | `POST /slack/events`; check `SLACK_SIGNING_SECRET`; watch the server log for 401 |
| 401s in HTTP mode after a while | Server clock drift > 5 min | Fix NTP |
| `Unhandled request` warnings in logs | Event subscribed with no matching listener | Add a no-op listener or unsubscribe |
| Bold shows as `**text**` | Raw Markdown posted | Pass text through `to_mrkdwn` (the agent paths already do) |
| `invalid_blocks` | >3000 chars in a section, >50 blocks, malformed block | Use `answer_blocks` / `chunk`; test payloads in Block Kit Builder |
| Agent replies "I went around in circles" | Recursion limit hit — usually a tool returning confusing errors | Look at the trace; improve the tool's docstring or error text; raise `AGENT_MAX_STEPS` carefully |
| Memory "forgets" after restart | `DATABASE_URL` empty (in-memory) | Use the SQLite default locally; Postgres in the cloud |
| Memory inconsistent with several workers | Each worker has its own memory | One worker per container, external checkpointer to scale |
| Event loop warnings / everything freezes during an LLM call | A blocking call inside an async listener | Use async clients or `asyncio.to_thread` |
| `RuntimeError: SLACK_MODE=socket needs SLACK_APP_TOKEN` | Missing xapp token | Generate it with `connections:write` |
| Clicking 👍/👎 does nothing, log says `unhandled request` | Interactivity off in the Slack app | **Interactivity & Shortcuts → On** (HTTP mode: Request URL `…/slack/events`) |
| Vote reply says "couldn't find that answer in my records" | The answer was produced while the DB was down, or before you switched from in-memory to Postgres | Expected for old messages; new answers are recorded |
| `/ready` returns 503 `database unreachable` | Postgres down, wrong `DATABASE_URL`, or `localhost` used inside Docker | In compose the host is `postgres`; on Cloud Run see section 18.9 |
| `invalid DSN` / `unsupported scheme` | SQLAlchemy-style URL like `postgresql+asyncpg://` | Use plain `postgresql://user:pass@host:5432/db` |

When in doubt, set `LOG_LEVEL=DEBUG`: Bolt logs every request, which listener it checked, and which one ran.

---

## 16. Cheat sheet

**Bolt listeners**

```python
@app.message("text" | re.compile(...) | matchers=[fn])     # message events
@app.event("app_mention" | "message" | "reaction_added")   # any Events API type
@app.command("/name")                                       # slash commands  → ack within 3s
@app.action("action_id" | re.compile(...))                  # buttons, menus  → ack within 3s
@app.view("callback_id")                                    # modal submits   → ack within 3s
@app.shortcut("callback_id")                                # global/message shortcuts
@app.error                                                   # catch-all error handler
```

**Replying**

```python
await say(text=..., thread_ts=...)                          # same channel as the event
await respond(text=..., response_type="in_channel"|"ephemeral", replace_original=False)
await client.chat_postMessage(channel=..., text=..., blocks=..., thread_ts=...)
await client.chat_update(channel=..., ts=..., text=..., blocks=...)
await client.chat_postEphemeral(channel=..., user=..., text=...)
await client.reactions_add(channel=..., timestamp=..., name="eyes")
```

**LangGraph core**

```python
builder = StateGraph(State)
builder.add_node("name", fn)                                 # fn(state) -> partial update
builder.add_edge(START, "a"); builder.add_edge("a", "b")
builder.add_conditional_edges("a", router_fn, {"x": "x", END: END})
graph = builder.compile(checkpointer=InMemorySaver())
await graph.ainvoke(inputs, {"configurable": {"thread_id": "t1"}, "recursion_limit": 12})
await graph.aget_state({"configurable": {"thread_id": "t1"}})
async for part in graph.astream(inputs, config, stream_mode="updates" | "messages"): ...
```

**This repo's env vars**

| Var | Required | Default |
|---|---|---|
| `SLACK_MODE` | | `socket` |
| `SLACK_BOT_TOKEN` | for Slack | — |
| `SLACK_APP_TOKEN` / `SLACK_SOCKET_TOKEN` | socket mode | — |
| `SLACK_SIGNING_SECRET` | http mode | — |
| `SLACK_CHANNEL_ID` | for `/api/slack/notify` default | — |
| `SLACK_CMD_AGENT`, `SLACK_CMD_ADD` | | `/agent`, `/agent-add` |
| `OPENROUTER_API_KEY` / `OPENAI_API_KEY` / `LLM_API_KEY` | yes | — |
| `OPENAI_BASE_URL` | | OpenAI |
| `LLM_MODEL`, `LLM_TEMPERATURE` | | `gpt-4o`, `0.2` |
| `AGENT_MAX_STEPS`, `AGENT_MAX_INPUT_CHARS` | | `12`, `4000` |
| `API_KEY` | | empty = no auth |
| `APP_ENV` | | `local` (`test`/`prod` require PostgreSQL) |
| `DATABASE_URL` | | `sqlite:///./data/agent.db`; empty = in-memory |
| `DB_POOL_MIN`, `DB_POOL_MAX` | | `1`, `5` |
| `LOG_LEVEL` | | `INFO` |

**REST endpoints**

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | — | liveness: env, slack mode, model, database kind (no DB call) |
| GET | `/ready` | — | readiness: 200 if the database answers, else 503 |
| POST | `/api/agent/invoke` | `message`, `thread_id?`, `user_id?`, `post_to_slack_channel?` | answer, run_id, tool_calls, duration_ms, slack_message_ts |
| POST | `/api/slack/notify` | `text`, `channel?`, `thread_ts?` | channel, ts |
| GET | `/api/agent/threads/{thread_id}` | — | stored conversation |
| POST | `/api/feedback` | `run_id`, `feedback_type` (`upvote`/`downvote`), `user_id`, `channel_id?` | tally + `new`/`changed`/`unchanged` |
| GET | `/api/runs/{run_id}` | — | question, answer, tool calls, latency, every vote |
| GET | `/api/feedback/summary?days=7&worst=5` | — | totals, approval rate, by source, most downvoted answers |
| POST | `/slack/events` | Slack's payloads (HTTP mode) | Bolt's response |

Interactive docs at `http://localhost:8000/docs`.

---

## 17. Feedback: 👍/👎 into PostgreSQL

Every answer the agent gives in Slack ends with two buttons. A click is the cheapest labelled data you will ever get: a real user, a real question, and a verdict on the real answer. This section covers how those clicks end up as rows in PostgreSQL, joined to exactly what was asked and answered, and what you can do with them afterwards.

### 17.1 The idea in one picture

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant S as Slack
    participant L as Bolt listeners
    participant A as AgentService
    participant DB as PostgreSQL

    U->>S: @Ops Buddy when is the deploy freeze?
    S->>L: app_mention
    L->>A: run(...)
    A->>A: LangGraph answers, run_id = uuid4()
    A->>DB: INSERT INTO agent_runs (run_id, question, answer, tool_calls, ...)
    A-->>L: AgentResult(run_id, answer, ...)
    L->>S: chat.update: answer + [👍 value=run_id] [👎 value=run_id]

    U->>S: clicks 👎
    S->>L: block_actions, action_id=downvote, value=run_id
    L-->>S: ack() (immediately, before the DB)
    L->>L: feedback_from_body(body, "downvote")
    L->>DB: INSERT INTO agent_feedback ... ON CONFLICT (run_id, user_id) DO UPDATE
    L->>DB: SELECT counts for run_id
    DB-->>L: 👍 0 · 👎 1, change = new
    L->>S: chat.postEphemeral (only U sees it): "Thanks! This answer: 👍 0 👎 1"
```

The design choice that makes everything else work: **the button's `value` is a `run_id`, and the run is stored before the button exists.** A vote is then never an orphan "someone disliked something"; it's a foreign key to the question, the answer, the tools the agent used and how long it took.

### 17.2 Data model

```mermaid
erDiagram
    agent_runs ||--o{ agent_feedback : "is voted on by"
    agent_runs {
        uuid run_id PK
        text thread_id "slack:C123:1700000000.0001"
        text source "slack_mention | slack_dm | slash_command | api"
        text user_id "who asked"
        text channel_id
        text question
        text answer
        jsonb tool_calls "[{name, args}]"
        text model
        int duration_ms
        bool blocked
        timestamptz created_at
    }
    agent_feedback {
        uuid feedback_id PK
        uuid run_id FK
        text feedback_type "upvote | downvote (CHECK)"
        text user_id "who clicked"
        text team_id
        text channel_id
        text message_ts
        timestamptz created_at
        timestamptz updated_at
    }
```

The full DDL is in `app/db/schema.sql`. The parts that do real work:

- `UNIQUE (run_id, user_id)` — **one vote per person per answer.** Double-clicks, Slack retries and people changing their mind all land on the same row.
- `CHECK (feedback_type IN ('upvote','downvote'))` — the database refuses anything else, whatever the code does.
- `REFERENCES agent_runs ON DELETE CASCADE` — no votes for answers that don't exist, and deleting a run (say, for a data-deletion request) removes its votes too.
- `tool_calls JSONB` — queryable: "approval rate of answers that used `create_ticket`" is one `WHERE tool_calls @> '[{"name":"create_ticket"}]'`.
- The view `agent_run_scores` gives one row per answer with `upvotes` / `downvotes` — handy for dashboards and for pulling eval sets.

The schema is applied at startup (every statement is `IF NOT EXISTS`, guarded by a Postgres advisory lock so two replicas booting together don't race), and docker-compose also mounts it into Postgres's init directory.

### 17.3 The listeners

`app/slack/bolt_app.py`, the same shape as your snippet, async:

```python
@app.action("upvote")
async def handle_upvote(ack, body, client):
    await ack()                                   # within 3s, before touching the database
    await save_vote(body, "upvote", client)

@app.action("downvote")
async def handle_downvote(ack, body, client):
    await ack()
    await save_vote(body, "downvote", client)

async def save_vote(body, feedback_type, client):
    record = feedback_from_body(body, feedback_type)   # run_id, user, channel, message ts, team
    tally = await store.add_feedback(record)           # upsert + counts, one transaction
    await client.chat_postEphemeral(
        channel=record.channel_id, user=record.user_id,
        text=f"Thanks! This answer: :thumbsup: {tally.upvotes}  :thumbsdown: {tally.downvotes}",
    )
```

The real `save_vote` also handles the edge cases: a malformed payload is logged and ignored, a vote on a run the DB has never seen gets a polite "couldn't find that answer", and a database error tells the user to try again rather than failing silently. It also words the reply differently for a first vote, a changed vote, and a repeat click.

`feedback_from_body` is your `add_feedback` field extraction, made strict:

| Field | Comes from | Why it matters |
|---|---|---|
| `run_id` | `body["actions"][0]["value"]` | Must parse as a UUID, otherwise rejected — the button value is user-visible data in the payload, so treat it as input |
| `user_id` | `body["user"]["id"]` | Who voted. Comes from Slack's signed payload, not from anything the user typed |
| `channel_id` | `body["channel"]["id"]` | Where |
| `message_ts` | `body["message"]["ts"]` (falls back to `container.message_ts`) | Which Slack message held the buttons — lets you link back to it |
| `team_id` | `body["team"]["id"]` | Needed once the app is installed in more than one workspace |

### 17.4 The upsert, line by line

```sql
WITH prev AS (                                   -- what did this user vote before, if anything?
    SELECT feedback_type FROM agent_feedback WHERE run_id = $2 AND user_id = $4
)
INSERT INTO agent_feedback (feedback_id, run_id, feedback_type, user_id, team_id,
                            channel_id, message_ts, created_at, updated_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $8)
ON CONFLICT (run_id, user_id) DO UPDATE          -- same person, same answer -> update, don't add
   SET feedback_type = EXCLUDED.feedback_type,   -- EXCLUDED = the row we tried to insert
       message_ts    = COALESCE(EXCLUDED.message_ts, agent_feedback.message_ts),
       updated_at    = EXCLUDED.updated_at       -- created_at keeps the first click
RETURNING (xmax = 0)                AS inserted,      -- Postgres trick: xmax is 0 only for fresh inserts
          (SELECT feedback_type FROM prev) AS previous_type
```

One round trip tells the app whether this click was `new`, `changed` (👍 → 👎) or `unchanged` (a repeat). The counts are read in the same transaction, so the number the user sees includes their own vote.

Why this beats appending lines to a file:

| | `feedback.jsonl` + `threading.Lock` | PostgreSQL |
|---|---|---|
| Two replicas / two pods | Two different files, each with half the votes | One table |
| Double-click / Slack retry | Two lines | One row (`UNIQUE`) |
| User changes their mind | Both votes kept, you have to dedupe later | Row updated, `updated_at` moves |
| "Which answers are bad?" | Load the file, join to… nothing, answers weren't stored | `SELECT … FROM agent_run_scores ORDER BY downvotes DESC` |
| Concurrency | Your lock, one process only | The database's job |
| Container restart | File gone unless you mounted a volume | Durable |
| Invalid data | Whatever got written | `CHECK`, `NOT NULL`, `FOREIGN KEY` |

Two small bugs in the snippet, both fixed in `examples/03_feedback_postgres_sync.py`: `add_feedback()` calls `logger.info(...)`, but `logger` only exists as an argument of the listener functions, so it raises `NameError` unless there's also a module-level logger; and the `run_id` on the button had nothing to join to, because the answer itself was never stored.

### 17.5 Async vs sync driver

The main app uses **asyncpg** (`app/db/feedback_store.py`) because everything — FastAPI, Bolt's `AsyncApp`, LangGraph — shares one asyncio event loop. A blocking driver call inside an async listener would freeze every other Slack event and REST request while it waited on the network. If you stay with the sync `App` (like your original scripts), use **psycopg 3 with `psycopg_pool.ConnectionPool`**, which is exactly what `examples/03_feedback_postgres_sync.py` does: Bolt's sync app runs each listener in a worker thread, and the pool hands each thread its own connection.

Either way, notice the order: **`ack()` first, database second.** If Postgres is slow, the user's click still doesn't time out.

### 17.6 Running it

Locally there is nothing to start: votes go into the SQLite file next to everything else.

```bash
make run                                 # DATABASE_URL=sqlite:///./data/agent.db
curl -s localhost:8000/ready             # {"status":"ready","database":"sqlite"}
```

Ask the bot something in Slack, click 👍, then:

```bash
make feedback                            # latest votes with their questions (sqlite3 CLI)
sqlite3 data/agent.db                    # poke around: .tables, SELECT * FROM agent_run_scores;
```

To rehearse the cloud setup on PostgreSQL: `make db`, then `DATABASE_URL=postgresql://agent:agent@localhost:5432/agent make run` (and `make psql`). On Cloud Run the same tables live in Cloud SQL; section 18 covers that. The queries in 17.7 are written for PostgreSQL; on SQLite, `date_trunc('day', x)` becomes `substr(x, 1, 10)` and `jsonb_array_elements(tool_calls)` becomes `json_each(tool_calls)`.

The sync example on its own:

```bash
pip install -r requirements-examples.txt
python examples/03_feedback_postgres_sync.py
```

### 17.7 Queries you'll actually run

```sql
-- Approval rate per day
SELECT date_trunc('day', updated_at) AS day,
       COUNT(*) FILTER (WHERE feedback_type = 'upvote')   AS up,
       COUNT(*) FILTER (WHERE feedback_type = 'downvote') AS down,
       ROUND(AVG((feedback_type = 'upvote')::int), 2)     AS approval
FROM agent_feedback
GROUP BY 1 ORDER BY 1 DESC;

-- The answers to go fix first
SELECT created_at, question, left(answer, 120) AS answer, upvotes, downvotes
FROM agent_run_scores
WHERE downvotes > 0
ORDER BY downvotes DESC, created_at DESC
LIMIT 20;

-- Which tools show up in disliked answers?
SELECT t->>'name' AS tool,
       SUM(upvotes) AS up, SUM(downvotes) AS down
FROM agent_run_scores, jsonb_array_elements(tool_calls) AS t
GROUP BY 1 ORDER BY down DESC;

-- Slow answers vs approval
SELECT width_bucket(duration_ms, 0, 20000, 4) AS latency_bucket,
       SUM(upvotes) AS up, SUM(downvotes) AS down
FROM agent_run_scores GROUP BY 1 ORDER BY 1;

-- Export a labelled eval set (run from psql)
\copy (SELECT question, answer, CASE WHEN upvotes > downvotes THEN 'good' ELSE 'bad' END AS label FROM agent_run_scores WHERE upvotes + downvotes > 0) TO 'eval_set.csv' CSV HEADER
```

### 17.8 REST endpoints

| Method | Path | What for |
|---|---|---|
| `POST` | `/api/feedback` | Votes from somewhere other than Slack (a web UI, an eval harness). Same upsert, same rules. |
| `GET` | `/api/runs/{run_id}` | One answer: question, answer, tool calls, latency, and every vote on it |
| `GET` | `/api/feedback/summary?days=7&worst=5` | Totals, approval rate, split by source, the most downvoted answers |

`POST /api/agent/invoke` now returns `run_id`, so an API caller can vote on the answer it just got:

```bash
RUN=$(curl -s -X POST localhost:8000/api/agent/invoke -H 'Content-Type: application/json' \
      -d '{"message":"How many PTO days roll over?"}' | python -c 'import sys,json;print(json.load(sys.stdin)["run_id"])')
curl -s -X POST localhost:8000/api/feedback -H 'Content-Type: application/json' \
     -d "{\"run_id\":\"$RUN\",\"feedback_type\":\"upvote\",\"user_id\":\"web-1\"}"
# {"run_id":"…","upvotes":1,"downvotes":0,"changed":"new"}
curl -s localhost:8000/api/feedback/summary | python -m json.tool
```

### 17.9 Before production

- **Migrations.** Applying `schema.sql` at startup is fine for a demo. Once the schema starts changing, move it into Alembic (or Flyway/Sqitch) migrations run by your deploy pipeline, and set `apply_schema=False` on the store.
- **You're storing questions and answers.** That's the point, and also personal/confidential data. Decide retention up front (e.g. a nightly `DELETE FROM agent_runs WHERE created_at < now() - interval '90 days'` — votes go with them via `ON DELETE CASCADE`), restrict who can read the tables, and consider redacting before insert in regulated environments.
- **Connections.** Each replica holds `DB_POOL_MAX` connections. With many replicas, put PgBouncer (transaction mode) in front, or use your cloud's proxy.
- **Database down.** Answers still go out — recording a run is best-effort and logged. Votes on those answers get the "couldn't find that answer" reply. If you need zero loss, write to a queue first and insert from a worker.
- **Who can vote.** Anyone who can see the message can click. If only the asker's opinion should count, compare `body["user"]["id"]` with `agent_runs.user_id` and store the others with a flag instead of dropping them.

### 17.10 Ideas from here

- **Ask why on 👎.** In `handle_downvote`, open a modal with `client.views_open(trigger_id=body["trigger_id"], view=…)` containing a few reason checkboxes and a text box; handle `@app.view("downvote_reason")` and store the reason in a `comment` column.
- **Show live counts on the buttons.** Use `respond(replace_original=True, blocks=…)` from the action's `response_url` to rewrite the actions block as "👍 3 · 👎 1".
- **Send votes to your tracing tool** (LangSmith feedback API, Monocle, an OTel span event) alongside the Postgres row, keyed by the same `run_id`, so a bad vote opens straight onto its trace.
- **Close the loop.** A weekly job that takes the most downvoted answers from `agent_run_scores`, turns them into regression tests for the agent (the `ScriptedChatModel` pattern in `tests/` works for prompt/tool changes, real-model evals for quality), and posts a summary into a Slack channel with `/api/slack/notify`.

---

## 18. Deploying to Google Cloud: Terraform, Cloud SQL, GitHub Actions

This is the same deployment pattern as `anjijava16/glcoud-run-agents` (develop branch): Terraform under `infra/`, GitHub Actions authenticating to Google with Workload Identity Federation, the OpenRouter key in Secret Manager, Docker Hub credentials in GitHub secrets, setup scripts under `scripts/` that read your keys from `~/.zprofile.sh`. It adds the parts a Slack bot needs: PostgreSQL in the cloud, Slack secrets per environment, and a test → prod pipeline. It is designed to land in the **same Google Cloud project** and reuse what's already there (state bucket, WIF pool, `openrouter-api-key`) without touching the retail app.

### 18.1 The picture

```mermaid
flowchart LR
    dev([You]) -- "git push develop" --> GH
    subgraph GH["GitHub Actions"]
      direction TB
      CI["ci.yml<br/>ruff · pytest (SQLite + Postgres) · terraform test · docker boot"]
      BUILD["build<br/>image tagged with the commit SHA"]
      DT["deploy.yml env=test<br/>terraform apply · smoke test"]
      DP["deploy.yml env=prod<br/>only if DEPLOY_PROD=true<br/>(GitHub environment 'production')"]
      CI --> BUILD --> DT --> DP
    end
    GH -- "OIDC token → WIF<br/>(no JSON key anywhere)" --> SA["slack-agent-deployer SA"]
    BUILD -- push --> AR[("Artifact Registry<br/>slack-agent")]
    BUILD -. "optional push" .-> DH[("Docker Hub<br/>anjaiahspr/…")]
    subgraph GCP["Google Cloud project"]
      direction TB
      RT["Cloud Run<br/>slack-agent-test"]
      RP["Cloud Run<br/>slack-agent-prod"]
      SQL[("Cloud SQL · PostgreSQL 16<br/>slack-agent-pg<br/>dbs: slack_agent_test · slack_agent_prod")]
      SM[["Secret Manager<br/>openrouter-api-key (shared)<br/>slack-agent-&lt;env&gt;-db-url · -slack-bot-token<br/>-slack-signing-secret · -api-key"]]
      RT -- "unix socket /cloudsql" --> SQL
      RP -- "unix socket /cloudsql" --> SQL
      SM -. "env vars at start" .-> RT
      SM -. "env vars at start" .-> RP
    end
    DT --> RT
    DP --> RP
    SlackT[["Slack app 'Ops Buddy (test)'"]] -- "POST /slack/events" --> RT
    SlackP[["Slack app 'Ops Buddy'"]] -- "POST /slack/events" --> RP
    RT --> OR[("OpenRouter")]
    RP --> OR
```

### 18.2 Environments

| | local | test | prod |
|---|---|---|---|
| Runs on | your laptop (`make run`) | Cloud Run `slack-agent-test` | Cloud Run `slack-agent-prod` |
| Database | SQLite `./data/agent.db` | Cloud SQL db `slack_agent_test` | Cloud SQL db `slack_agent_prod` |
| Slack transport | Socket Mode (no public URL) | HTTP `…/slack/events` | HTTP `…/slack/events` |
| Slack app | your existing Socket Mode app | "Ops Buddy (test)", commands `/agent-test`, `/agent-add-test` | "Ops Buddy", commands `/agent`, `/agent-add` |
| Secrets come from | `~/.zprofile.sh` / `.env` | Secret Manager | Secret Manager |
| Deployed by | — | every push to `develop` | after test, when `DEPLOY_PROD=true` (or ticked on a manual run) |

`APP_ENV` makes the rule hard: with `APP_ENV=test` or `prod` the app **refuses to start** unless `DATABASE_URL` is PostgreSQL. A Cloud Run container's disk is in memory and private to one instance, so a SQLite file there would silently lose every vote and conversation on the next scale-down.

**Why one Slack app per environment:** a Slack app sends its events to exactly one Request URL. Your local Socket Mode app, the test service and the prod service each need their own app (and slash command names are workspace-global, hence `/agent-test`). `scripts/slack-manifest.sh test` prints a ready-to-paste manifest with the right URL and names.

### 18.3 Where every secret lives

| What | Where | Who reads it | How it gets there |
|---|---|---|---|
| OpenRouter key | Secret Manager `openrouter-api-key` (shared with glcoud-run-agents) | Cloud Run, as `LLM_API_KEY` | `gcp-setup.sh` from `OPENROUTER_API_KEY` in `~/.zprofile.sh` |
| Slack bot token | Secret Manager `slack-agent-<env>-slack-bot-token` | Cloud Run | `push-secrets.sh` from `SLACK_BOT_TOKEN` (or `TEST_…` / `PROD_…`) |
| Slack signing secret | Secret Manager `slack-agent-<env>-slack-signing-secret` | Cloud Run | `push-secrets.sh` from `SLACK_SIGNING_SECRET`, or asks you (hidden input) |
| Slack app token (`xapp-`) | Secret Manager `slack-agent-<env>-slack-app-token` | Cloud Run, only if `SLACK_MODE=socket` | `push-secrets.sh` from `SLACK_APP_TOKEN` / `SLACK_SOCKET_TOKEN` |
| `/api/*` key | Secret Manager `slack-agent-<env>-api-key` | Cloud Run, as `API_KEY` | generated once by `push-secrets.sh` |
| Database URL (user + password) | Secret Manager `slack-agent-<env>-db-url` | Cloud Run, as `DATABASE_URL` | Terraform (`random_password`), see 18.5 |
| Docker Hub user/token | GitHub **secrets** `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN` | CD build job | `github-setup.sh` from `~/.zprofile.sh` |
| GCP ids, WIF provider, SA emails | GitHub **variables** | CD | `github-setup.sh` from `.gcp-setup.env` |
| Google credentials | **nowhere** | — | GitHub's OIDC token is exchanged through WIF for a short-lived token of `slack-agent-deployer` |

Values reach Secret Manager through stdin (`gcloud secrets versions add --data-file=-`), never a command line, so they don't land in shell history or `ps`. `put_secret` only adds a version when the value changed, so re-running after rotating a token in `~/.zprofile.sh` is the rotation procedure. Cloud Run reads `latest` when an instance starts; new instances pick up a rotated value (force it with a deploy or `make restart ENV=test`, which rolls a new revision).

`SLACK_CHANNEL_ID` isn't secret, so it travels as a GitHub variable into a plain env var.

### 18.4 Decisions that are specific to Slack on Cloud Run

**HTTP mode, not Socket Mode.** Socket Mode needs a permanently open WebSocket, which means a permanently running instance (`min_instances >= 1`). HTTP mode lets Cloud Run scale to zero between messages. Terraform enforces this: `slack_mode = "socket"` with `min_instances = 0` fails the plan with an explanation. Socket Mode stays the default locally, where it needs no public URL.

**CPU always allocated (`cpu_idle = false`).** This one bites everyone. Bolt acks Slack's request immediately (Slack's 3-second rule) and runs the agent *after* the HTTP response. With Cloud Run's default "CPU only during requests", that after-the-response work is throttled to almost nothing: the 👀 reaction appears and the answer arrives minutes later or never. With CPU always allocated, instances still scale to zero when idle; you pay for CPU while an instance is alive rather than per request.

**State in the database, not the instance.** Instances come and go. Three things that used to be per-process now live in Postgres:

| State | Table | Effect |
|---|---|---|
| Conversation memory | `checkpoints`, `checkpoint_writes`, … (LangGraph) | a thread continues on whichever instance gets the next message |
| 👍/👎 and the answers they judge | `agent_runs`, `agent_feedback` | section 17 |
| "Have we handled this Slack event?" | `slack_event_claims` | a retry that lands on another instance is dropped by `INSERT … ON CONFLICT DO NOTHING` |

Still per instance, by choice: the per-thread `asyncio.Lock` (a same-instant reply race across two instances is rare for a bot; `pg_advisory_xact_lock(hashtext(thread_id))` would close it) and the demo ticket store.

**Cloud SQL over a unix socket.** The Cloud Run service mounts the instance at `/cloudsql/<project>:<region>:slack-agent-pg` (Cloud Run's built-in Cloud SQL connector). Access is authorised by IAM — the runtime SA has `roles/cloudsql.client` — and the instance's public IP has no authorised networks, so nothing else can reach it. Both drivers take the same URL:

```
postgresql://slack_agent_test:<password>@/slack_agent_test?host=/cloudsql/<project>:us-central1:slack-agent-pg
           └ user ─────────┘ └ generated ┘ └ database ─────┘ └ socket directory ───────────────────────────┘
```

`asyncpg` (votes, runs, claims) and `psycopg` (LangGraph memory) both read `?host=` as a socket directory. `tests/test_persistence.py::test_postgres_unix_socket_dsn` checks exactly this shape against a real Postgres socket.

**Connection budget.** Each instance holds two small pools (asyncpg + psycopg), `DB_POOL_MAX=3` each. With `max_instances=3` per environment that's at most 2 × 3 × 3 × 2 envs = 36 connections, under the `max_connections=50` Terraform sets on the shared-core instance. Raise `MAX_INSTANCES` and the tier together.

**Public service, authenticated requests.** Slack can't send Google credentials, so the service grants `roles/run.invoker` to `allUsers`. Every Slack request is still verified with the signing secret (the CD smoke test asserts an unsigned POST gets **401**), and `/api/*` requires `X-API-Key`. If your organisation blocks `allUsers` (Domain Restricted Sharing policy), see 18.9.

### 18.5 The Terraform, stack by stack

Two stacks, split by who applies them and how often:

| | `infra/platform` | `infra/cloudrun` |
|---|---|---|
| Applied by | you, once, via `scripts/gcp-setup.sh` (project Owner) | CD, on every deploy, once per environment |
| State | `gs://<project>-tfstate/slack-agent/platform` | `gs://<project>-tfstate/slack-agent/cloudrun/<env>` |
| Creates | APIs; Artifact Registry `slack-agent`; Cloud SQL `slack-agent-pg` (Postgres 16, `db-f1-micro`, ENTERPRISE edition, backups + PITR, deletion protection); a database + user per env; `random_password` → `slack-agent-<env>-db-url` secret; empty secret containers for Slack + API key; runtime SA (`logging`, `monitoring`, `cloudtrace`, `cloudsql.client`, accessor on its secrets and on `openrouter-api-key`); deployer SA (`run.admin`, `secretmanager.viewer`, `cloudsql.viewer`, writer on *this* AR repo only, actAs runtime SA, state bucket access); a WIF provider `slack-agent-gh` inside the existing `github-pool`, locked to `repository == anjijava16/slack-fastapi-langgraph-agent && ref == refs/heads/develop` | one `google_cloud_run_v2_service` (`slack-agent-<env>`): image, runtime SA, Cloud SQL volume, CPU always allocated, env vars, secrets by reference, `/health` probes, public invoker |
| Why separate | holds the database and the trust relationships; CD must not be able to delete or widen them | changes every deploy; least privilege for the CI identity |

A few details worth knowing:

- **Shared vs owned.** The state bucket, the `github-pool` WIF pool and the `openrouter-api-key` secret may already exist from glcoud-run-agents. `gcp-setup.sh` creates them with gcloud *only if missing*, and Terraform references them by id instead of owning them, so neither project can delete the other's resources.
- **The DB password is in Terraform state.** Terraform generates it, so it's in the platform state object (private, versioned bucket). Rotate with `terraform -chdir=infra/platform apply -replace='random_password.db["prod"]' …` and redeploy.
- **PostgreSQL 16 defaults to the Enterprise Plus edition,** which has no shared-core tiers; the instance pins `edition = "ENTERPRISE"` so `db-f1-micro` works.
- **Tests without a cloud.** `infra/*/tests/*.tftest.hcl` plan (and, for platform, apply) against mocked providers: one database per env, the exact DSN shape, the WIF condition, the env vars on the service, the socket-mode guard. `make tf-test` runs them; CI does too.

### 18.6 One-time setup, step by step

You need `gcloud` (logged in as the project owner), `terraform` ≥ 1.6, the GitHub CLI `gh`, and `openssl`.

**1. Check `scripts/config.sh`.** It defaults to the project and account from glcoud-run-agents, `GITHUB_REPO=anjijava16/slack-fastapi-langgraph-agent`, `GITHUB_BRANCH=develop`, `SERVICE=slack-agent`, `ENVIRONMENTS="test prod"`. Change `GITHUB_REPO` if you push this somewhere else — Google will only trust that exact repo and branch.

**2. Make sure `~/.zprofile.sh` has what it needs.** You already export the Slack and OpenRouter values; the scripts look for these names (shell first, then the file):

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
export SLACK_BOT_TOKEN=xoxb-...          # test env (or TEST_SLACK_BOT_TOKEN)
export SLACK_SIGNING_SECRET=...          # HTTP mode needs it; asked for if missing
export SLACK_SOCKET_TOKEN=xapp-...       # only for Socket Mode
export SLACK_CHANNEL_ID=C...
export PROD_SLACK_BOT_TOKEN=xoxb-...     # when prod gets its own Slack app
export PROD_SLACK_SIGNING_SECRET=...
export DOCKERHUB_USERNAME=anjaiahspr
export DOCKERHUB_TOKEN=dckr_pat_...      # Docker Hub access token, Read & Write
```

**3. Google Cloud.**

```bash
bash scripts/gcp-setup.sh        # or: make gcp-setup     (AUTO_APPROVE=1 to skip the "yes")
```

It checks billing, enables APIs, ensures the shared pieces, runs `terraform apply` on `infra/platform` (about 10 minutes the first time, almost all of it Cloud SQL), copies your secrets into Secret Manager, and writes `.gcp-setup.env`. Safe to re-run.

**4. GitHub.**

```bash
gh auth login                    # once
bash scripts/github-setup.sh     # or: make github-setup
```

Creates the repo if it doesn't exist (private), sets the variables and Docker Hub secrets, creates the `test` and `production` environments, and prints the git commands for the first push. It sets `DEPLOY_PROD=false`, so the first runs only deploy test.

**5. First deploy.** Push to `develop`:

```bash
git init -b develop && git remote add origin https://github.com/anjijava16/slack-fastapi-langgraph-agent.git
git add -A && git commit -m "Slack agent on Cloud Run" && git push -u origin develop
```

Watch **Actions → CD**. The deploy job's summary shows the service URL and the Slack Request URL.

**6. Point a Slack app at it.**

```bash
bash scripts/slack-manifest.sh test     # prints a manifest with the real URL
```

At api.slack.com/apps: **Create New App → From a manifest** → paste → install to the workspace. Copy that app's **Bot User OAuth Token** and **Signing Secret** into `~/.zprofile.sh` (as `TEST_SLACK_BOT_TOKEN` / `TEST_SLACK_SIGNING_SECRET`, or the plain names), then:

```bash
bash scripts/push-secrets.sh test       # new secret versions
make restart ENV=test                   # new revision -> new instances read the new values
```

(Or reuse the app you already have: open its **App Manifest**, paste, save. Note that switches it from Socket Mode to HTTP, so your laptop would need its own app.) In Slack: `/invite @Ops Buddy (test)`, then `@Ops Buddy (test) when is the deploy freeze?`.

**7. Prod.** Create the prod Slack app the same way (`slack-manifest.sh prod` after the first prod deploy — or create it now and fix the URL later), put `PROD_SLACK_*` in your profile, `bash scripts/push-secrets.sh prod`, then `gh variable set DEPLOY_PROD --body true`. From then on every push to develop goes test → prod. For a manual approval before prod, add required reviewers to the **production** environment in the repo settings; the prod job waits for a click.

### 18.7 The pipeline

```mermaid
sequenceDiagram
    autonumber
    actor Dev as You
    participant GH as GitHub Actions
    participant WIF as Google STS (WIF)
    participant AR as Artifact Registry
    participant TF as Terraform (infra/cloudrun)
    participant CR as Cloud Run
    participant SQL as Cloud SQL

    Dev->>GH: push develop
    GH->>GH: ci.yml — ruff, shellcheck, pytest with a Postgres service, terraform fmt/validate/test, docker build + boot
    GH->>WIF: OIDC token (repo=…/slack-fastapi-langgraph-agent, ref=refs/heads/develop)
    WIF-->>GH: short-lived token for slack-agent-deployer
    GH->>AR: push image :<sha12> and :latest (and Docker Hub if configured)
    GH->>TF: init (state prefix slack-agent/cloudrun/test), plan, apply image=<sha12>
    TF->>CR: new revision of slack-agent-test
    CR->>SQL: connect via /cloudsql socket, create tables if missing
    GH->>CR: GET /health, GET /ready (DB reachable), unsigned POST /slack/events → expect 401
    alt DEPLOY_PROD == true
        GH->>GH: wait for "production" environment approval (if reviewers set)
        GH->>TF: same image, env=prod (state prefix …/prod)
        TF->>CR: new revision of slack-agent-prod
        GH->>CR: same smoke test
    end
```

The image is built once and promoted, so prod runs byte-for-byte what test ran. CI runs on every PR and every non-develop push; on develop it runs as CD's first job. Deploys of the same environment queue rather than cancel (`cancel-in-progress: false`), because killing `terraform apply` midway leaves a locked state.

### 18.8 Day two

| Task | Command |
|---|---|
| Service URL | `make url ENV=test` |
| Logs | `make logs ENV=prod` or Cloud Console → Cloud Run → Logs |
| Rotate a Slack token / OpenRouter key | update `~/.zprofile.sh`, `make push-secrets ENV=prod`, then `make restart ENV=prod` |
| Change the model | `gh variable set LLM_MODEL --body anthropic/claude-sonnet-4.5`, push (or re-run CD) |
| Roll back fast | `gcloud run services update-traffic slack-agent-prod --region us-central1 --to-revisions=<previous-revision>=100` (the next deploy moves traffic forward again) |
| Roll back properly | `git revert <sha> && git push` — CD deploys the reverted code |
| psql into a cloud database | `gcloud sql connect slack-agent-pg --user=slack_agent_test --database=slack_agent_test` (password: `gcloud secrets versions access latest --secret=slack-agent-test-db-url`, the part between `:` and `@`) |
| Votes in prod | `SELECT * FROM agent_run_scores ORDER BY downvotes DESC LIMIT 20;` (section 17.7 has more) |
| Read the API key | `gcloud secrets versions access latest --secret=slack-agent-test-api-key` |
| Tear down an environment | `terraform -chdir=infra/cloudrun init -backend-config=bucket=<project>-tfstate -backend-config=prefix=slack-agent/cloudrun/test` then `terraform -chdir=infra/cloudrun destroy` with the same `-var`s as `deploy.yml` |
| Tear down everything | destroy both cloudrun states, then re-apply platform with `-var=db_deletion_protection=false`, then `terraform -chdir=infra/platform destroy` |

**Cost, roughly.** The always-on piece is Cloud SQL: a `db-f1-micro` instance with 10 GB SSD and backups is on the order of ten dollars a month (check current pricing for your region). Cloud Run with `min_instances=0` costs nothing while idle; with CPU always allocated you pay for the instance's lifetime once a message wakes it up (it lingers for a while after the last request). Artifact Registry keeps the last 15 images and deletes ones older than 30 days. OpenRouter is billed per token by model.

### 18.9 Troubleshooting the deployment

| Symptom | Cause | Fix |
|---|---|---|
| CD: `Repository variable … is not set` | `github-setup.sh` not run, or run against another repo | `bash scripts/github-setup.sh` |
| `google-github-actions/auth`: `unauthorized_client` / `attribute condition` | Workflow ran from a branch other than `develop`, or the repo name differs from `GITHUB_REPO` in `config.sh` (case matters) | Run from develop; fix `config.sh`, re-run `gcp-setup.sh` |
| `Permission 'iam.serviceAccounts.getAccessToken' denied` | WIF binding not propagated yet, or wrong repo in the binding | Wait a minute and re-run; check `GITHUB_REPO` |
| `gcp-setup.sh`: `service account … does not exist` during apply | IAM eventual consistency right after the SA was created | Run the script again |
| Deploy: `Secret … versions/latest was not found` | A secret container exists with no value (usually signing secret) | `bash scripts/push-secrets.sh <env>` |
| Deploy: `allUsers` binding rejected (`constraints/iam.allowedPolicyMemberDomains`) | Org policy Domain Restricted Sharing | Exempt the project, or put the service behind an external HTTPS load balancer / API Gateway |
| `/ready` → 503 in the smoke test | Cloud SQL not reachable: wrong `CLOUDSQL_CONNECTION_NAME`, missing `cloudsql.client`, or instance stopped | Check the variable, `gcloud sql instances describe slack-agent-pg`, container logs |
| Container exits with `APP_ENV=… requires a PostgreSQL DATABASE_URL` | Secret `slack-agent-<env>-db-url` empty or a SQLite URL | Re-apply `infra/platform` |
| `remaining connection slots are reserved` | Too many instances × pools for `db-f1-micro` | Lower `MAX_INSTANCES` / `DB_POOL_MAX`, or a bigger tier |
| Slack: "Your URL didn't respond with the value of the challenge parameter" | Wrong Request URL, or the env's signing secret doesn't match that Slack app | Use the URL from `slack-manifest.sh`; `push-secrets.sh` with the right secret, then `make restart ENV=<env>` |
| Slack: 👀 appears, answer comes very late or never | CPU throttled after the response | `cpu_idle` must be `false` (it is in this Terraform); check nobody changed it in the console |
| Slack: `/agent` "dispatch_failed" in test | Test app uses `/agent-test` | Use the env's command names; the manifest has them |
| Answers twice | Two Slack apps subscribed to the same channel events, both pointing at environments with the same bot | One app per environment; each bot only in its channels |

### 18.10 How this relates to glcoud-run-agents

| | glcoud-run-agents | this repo |
|---|---|---|
| Branch that deploys | `develop` | `develop` |
| GitHub → GCP auth | WIF provider `github-oidc` in `github-pool` | its own provider `slack-agent-gh` in the same pool |
| Deployer / runtime SAs | `github-deployer`, `<service>-run` | `slack-agent-deployer`, `slack-agent-run` |
| Terraform state | `gs://<project>-tfstate/retail-adk-agents/…` | `gs://<project>-tfstate/slack-agent/…` (same bucket) |
| OpenRouter key | `openrouter-api-key` | same secret, read access granted to `slack-agent-run` |
| Database | SQLite inside the container, `max_instances=1` + session affinity | SQLite locally only; Cloud SQL Postgres in the cloud, so it scales out |
| Environments | one (production) | test, then prod with the same image |
| Docker Hub | `DOCKERHUB_REPO` var + `DOCKERHUB_USERNAME`/`DOCKERHUB_TOKEN` secrets | same |
| Extra CI checks | ruff, pytest, terraform validate, docker boot | + shellcheck, a Postgres service container, `terraform test` with mocked providers, a Slack signature check in the smoke test |

---

## 19. Course FAQ bot: RAG with LlamaIndex and Qdrant

This part brings the idea of [aaalexlit/faq-slack-bot](https://github.com/aaalexlit/faq-slack-bot) into this project. That bot answers DataTalks.Club Zoomcamp questions in Slack from the course FAQ, old Slack threads, the course repo and video transcripts. Here it runs on Qdrant instead of Milvus/Zilliz, with hybrid (keyword + vector) search, and as two routers inside the same FastAPI service as the agent:

- **ingestion** — `/api/ingest/*`, a CLI and a weekly GitHub Actions run load sources into one Qdrant collection per course
- **query** — `/api/rag/*`, and in Slack: an @mention in a course channel is answered from that course's collection, with the sources linked under the answer and the same 👍/👎 buttons as the agent

Everything else in this README still applies: Socket Mode or HTTP, the same feedback table, the same LangGraph checkpointer for thread memory, the same Cloud Run deployment.

### 19.1 What happens when a student asks

```
#course-llm-zoomcamp
  Priya:     @Ops Buddy I get "connection refused" every time I prompt ollama in the module 1 homework
  Ops Buddy: 🔍 searching the LLM Zoomcamp FAQ, Slack threads, course repo and videos…
             ↓ (edited when ready)
             The Ollama server isn't running. Restart it from the notebook with
             `!nohup ollama serve > nohup.out 2>&1 &` and run the cell again [1].
             If you restart the notebook, rerun the cell with `ollama serve` too [1].
             ───────────
             Sources
             ❓ FAQ · module-1-rag · Connection refused error when prompting Ollama RAG
             📚 LLM Zoomcamp · hybrid search · 20→10→4 chunks · rerank: fastembed · ⏱ 3.1s
             [👍 Helpful] [👎 Not helpful]
```

(The answer comes from the real FAQ entry, <https://datatalks.club/faq/llm-zoomcamp.html#c6fc2d4d11>. The wording is the model's and will vary.)

Behind that one message:

```mermaid
sequenceDiagram
    autonumber
    actor S as Student
    participant SL as Slack
    participant B as Bolt (Socket Mode / HTTP)
    participant R as RagService
    participant G as RAG LangGraph
    participant Q as Qdrant (course collection)
    participant L as LLM (OpenRouter)
    participant DB as SQLite / Postgres

    S->>SL: @Ops Buddy <question> in #course-llm-zoomcamp
    SL->>B: app_mention
    B->>B: channel → course "llm" (config/courses.yaml)
    B->>SL: 👀 + "searching…" placeholder in the thread
    B->>R: answer(question, "llm", thread_id=slack:C:ts)
    R->>G: ainvoke (checkpointer: rag:llm:slack:C:ts)
    G->>G: guardrail → condense (follow-up? rewrite with thread history)
    G->>Q: dense query + BM25 sparse query, 20 each
    Q-->>G: two ranked lists → RRF fusion → 20
    G->>G: recency weighting (Slack threads) → 10 → re-rank → 4
    G->>L: course prompt + 4 numbered excerpts + question
    L-->>G: answer with [n] citations
    G-->>R: answer, hits
    R->>DB: agent_runs (run_id, question, answer, sources)
    R-->>B: RagAnswer
    B->>SL: chat.update: answer + Sources + 👍/👎 (value = run_id)
```

### 19.2 Architecture

```mermaid
flowchart LR
    subgraph sources["Sources"]
      FAQ[("DataTalksClub/faq<br/>_questions/&lt;course&gt;/*.md")]
      GH[("Course GitHub repos")]
      SLH[("Slack channel history")]
      YT[("YouTube transcripts")]
      UP[("URLs, files, text")]
    end
    subgraph app["FastAPI service"]
      direction TB
      ING["/api/ingest/* · CLI · weekly workflow"]
      RD["Readers → LlamaIndex Documents"]
      IX["Indexer<br/>split · content-hash ids · diff · embed new only"]
      RET["Retriever<br/>hybrid · recency · re-rank"]
      RG["RAG LangGraph"]
      QR["/api/rag/*"]
      SB["Slack: course channel @mention · /faq"]
      AG["Agent: search_course_materials tool"]
      ING --> RD --> IX
      QR --> RG
      SB --> RG
      AG --> RET
      RG --> RET
    end
    FAQ --> RD
    GH --> RD
    SLH --> RD
    YT --> RD
    UP --> ING
    IX -- "upsert dense + sparse" --> QD[("Qdrant<br/>faq_llm · faq_de · faq_ml · faq_mlops")]
    RET -- "query" --> QD
    RG --> LLM[("LLM")]
    RG --> DB[("agent_runs · agent_feedback<br/>ingest_jobs · checkpoints")]
```

Two separate paths share the collection layout: **ingestion** writes, **query** reads. They never import each other's code beyond `qdrant_store.py`, which is the only module that knows how a collection is laid out.

### 19.3 Qdrant: modes and collection layout

`QDRANT_MODE` picks how the app reaches Qdrant. Nothing else in the code changes.

| Mode | What it is | Use it for | Config |
|---|---|---|---|
| `local` | A Qdrant server you run, here the Docker container from `make qdrant` | Day-to-day development (the default) | `QDRANT_URL=http://localhost:6333` |
| `cloud` | Qdrant Cloud cluster | test / prod on Cloud Run, or local runs against the real data | `QDRANT_URL=https://<id>.<region>.cloud.qdrant.io:6333`, `QDRANT_API_KEY` |
| `embedded` | No server; Qdrant's local mode stores files in `QDRANT_PATH` | A laptop without Docker; single process only | `QDRANT_PATH=./data/qdrant` |
| `memory` | In-process, gone on exit | Unit tests | — |

`APP_ENV=test` and `prod` refuse `embedded` and `memory` for the same reason they refuse SQLite: a Cloud Run container's disk lives in memory and belongs to one instance.

One collection per course, named `<QDRANT_COLLECTION_PREFIX><course>` (`faq_llm` locally, `test_faq_llm` and `prod_faq_llm` in the cloud, so one Qdrant Cloud cluster can hold every environment):

| Part | Name | Details |
|---|---|---|
| Dense vector | `dense` | size = the embedding model's dimension (384 for bge-small), cosine |
| Sparse vector | `sparse` | BM25 term weights, **`modifier: idf`**: Qdrant computes the IDF itself |
| Payload | LlamaIndex node | `_node_content` (text + metadata), `doc_id`, and flat metadata: `course`, `source_type`, `source_key`, `source_group`, `source_url`, `title`, `timestamp`, `content_hash`, … |
| Payload indexes | `source_key`, `source_group`, `source_type`, `doc_id` (keyword), `timestamp` (float) | Filters for source-type search, diffing on re-ingest, pruning (server modes) |

The collection is created by our code (`QdrantCollections.ensure`), not by LlamaIndex, so we control the sparse modifier, the vector names and the indexes; LlamaIndex's `QdrantVectorStore` then writes and reads it. If you change `EMBED_MODEL` to one with another dimension, the app refuses with a clear "reset it and ingest again" error instead of corrupting the collection.

Open `http://localhost:6333/dashboard` to browse points while developing.

### 19.4 Hybrid search, and why it matters for course questions

Students paste error messages. Embeddings are good at meaning ("can I still enrol after it started?" ≈ "I just discovered the course, can I still join?") and bad at exact strings (`ERR_CONNECTION_REFUSED`, `pg_hba.conf`, `WinError 126`, a package called `minsearch`). Keyword search is the opposite. Hybrid runs both and fuses them.

**The keyword side is real BM25, done the Qdrant way** (`app/rag/sparse.py`):

```
document → tokens → one sparse dimension per token, value = tf·(k1+1) / (tf + k1·(1 − b + b·len/avg_len))
query    → tokens → value 1.0 per distinct token
Qdrant   → multiplies in IDF from its own document counts (modifier: idf) and sums
```

Tokens keep identifiers whole and also split them (`pg_hba.conf` → `pg_hba.conf`, `pg`, `hba`, `conf`), drop stop words and are Porter-stemmed (`running` → `run`). Token ids are a stable 32-bit hash. Nothing is downloaded; `SPARSE_PROVIDER=fastembed` swaps in a fastembed sparse model (`Qdrant/bm25`, SPLADE) if you prefer.

**Fusion is Reciprocal Rank Fusion** (`rrf_fusion` in `qdrant_store.py`, plugged into LlamaIndex's `hybrid_fusion_fn`):

```
score(chunk) = α / (60 + rank_in_dense) + (1 − α) / (60 + rank_in_keyword)        α = RAG_HYBRID_ALPHA (0.5)
```

Ranks, not raw scores, are combined, so cosine similarity and BM25 never need to share a scale, and a chunk both searches agree on beats one that only one of them found.

A real example from the tests on the actual LLM Zoomcamp FAQ + repo (top result per mode):

| Question | keyword only | dense only | hybrid |
|---|---|---|---|
| "Can I still join the course late?" | ✗ a minsearch notebook | ✓ FAQ "I just discovered the course. Can I still join?" | ✓ same |
| "minsearch" | ✓ the minsearch lesson | ✗ an unrelated leaderboard FAQ | ✓ the minsearch lesson |
| "elasticsearch connection refused error" | ✓ FAQ "Connection refused error…" | ✓ same | ✓ same |

Each side fails on a different kind of question, and hybrid gets all three right.

### 19.5 The query pipeline: 20 → 10 → 4

The same funnel as the original bot, each stage configurable:

```mermaid
flowchart LR
    Q[question] --> E[embed + BM25-encode]
    E --> H["Qdrant: dense top 20 ∥ sparse top 20<br/>RRF → 20<br/>(RAG_TOP_K)"]
    H --> T["recency weighting → 10<br/>(RAG_RECENCY_TOP_K)"]
    T --> RR["re-rank → 4<br/>(RAG_FINAL_TOP_N)"]
    RR --> P[course prompt + numbered excerpts]
    P --> LLM["LLM answer with [n] citations"]
```

**Recency weighting** (`retriever.py`). Last month's Slack thread about the current homework is worth more than the same question from two cohorts ago. Scores are normalised to 0–1, then chunks that carry a `timestamp` (Slack threads; anything you ingest with one) get

```
score += RAG_RECENCY_WEIGHT · 0.5 ^ (age_days / RAG_RECENCY_HALF_LIFE_DAYS)        defaults 0.3 and 30 days
```

FAQ entries and repo files have no timestamp and keep their relevance score. The original used LlamaIndex's `TimeWeightedPostprocessor` on `thread_ts` for the same reason; this version uses a half-life, which is easier to reason about.

**Re-ranking** (`rerank.py`) reads the question and each candidate together with a cross-encoder. That's much more accurate than comparing two separate embeddings, and too slow to run on more than a handful of chunks, which is why it comes last.

| `RERANK_PROVIDER` | What | Notes |
|---|---|---|
| `auto` (default) | `cohere` if `COHERE_API_KEY` is set, else `none` | What you get locally out of the box |
| `cohere` | Cohere Rerank API, `rerank-v3.5` | Like the original bot |
| `fastembed` | Local cross-encoder `Xenova/ms-marco-MiniLM-L-6-v2` | No key; the Docker image bakes it in; the cloud default |
| `none` | Keep the hybrid + recency order | Fastest |

If the re-ranker fails (Cohere down, rate limit), the answer still goes out with the hybrid + recency order, and the context line in Slack says `rerank: cohere (failed)`. The original bot showed the student an error in that case.

**The answer** (`graph.py`, `prompts.py`). The prompt keeps the original's rules: answer only from the excerpts, say "I don't think I have an answer for this; you'll have to ask your fellows or instructors" when they don't cover it, never invent links, Slack link syntax, and it includes the course's cohort year, start date and today's date for "is it too late?" questions. The **Sources** block under the answer is built from the retrieved chunks, not from what the model writes, so every link is real.

### 19.6 The RAG LangGraph and follow-up questions

```mermaid
flowchart TD
    START((START)) --> guardrail
    guardrail -- blocked --> END1((END))
    guardrail -- ok --> condense
    condense["condense<br/>first turn: question as is<br/>follow-up: LLM rewrites it using the thread"] --> retrieve
    retrieve["retrieve<br/>hybrid → recency → re-rank"]
    retrieve -- hits --> generate["generate<br/>course prompt + excerpts + prior turns"]
    retrieve -- nothing found --> no_hits["no_hits<br/>fixed reply, no LLM call"]
    generate --> finalize
    no_hits --> finalize
    finalize --> END2((END))
```

The graph uses the same checkpointer as the agent (SQLite locally, Postgres in the cloud), keyed `rag:<course>:slack:<channel>:<thread_ts>`. So inside a Slack thread:

```
Priya: @Ops Buddy how do I use Gemini with the OpenAI-compatible endpoint?
Bot:   Point the OpenAI client at base_url=https://generativelanguage.googleapis.com/v1beta/openai/ …
Priya: @Ops Buddy and what about qdrant, which port?
       └ condense → "Which port does the Qdrant REST API use?" → retrieve → "6333 for REST, 6334 for gRPC"
```

Without the condense step, the second search would run on "and what about qdrant, which port?". That happens to work here, but "and on Windows?" would not. The rewritten query is shown in `/api/rag/query`'s `search_query` and stored with the run.

### 19.7 Slack: who answers what

| Where / how | Who answers | Thread memory |
|---|---|---|
| @mention in a channel listed under a course in `config/courses.yaml` | **RAG** over that course | per Slack thread |
| @mention in any other channel | **Agent**; it calls `search_course_materials` when the question is about a course | per Slack thread |
| `RAG_DEFAULT_COURSE=llm` | RAG for every channel not listed | per Slack thread |
| `/faq llm how do I…` | RAG for the named course, anywhere | per user + channel |
| `/faq how do I…` in a course channel | RAG for that channel's course | per user + channel |
| DM | Agent (with the course tool) | per DM |
| 👍/👎 under a RAG answer | The same `agent_runs` / `agent_feedback` tables as the agent (section 17), `source = slack_rag` | — |

**Point it at your workspace's channels.** The channel ids in `config/courses.yaml` are the DataTalks.Club workspace's, copied from the original bot. In your workspace, either edit the YAML or set `RAG_CHANNEL_MAP=C0123ABCD=llm,C0456EFGH=de`. To find an id: channel name → *View channel details* → bottom of the *About* tab. Then `/invite @Ops Buddy` in that channel.

**New scopes and a new command.** Slack history ingestion needs `channels:history` (already there) plus `channels:read` and `groups:read` for channel names, and there's a new `/faq` command. Both manifests include them. On an existing app, paste the updated manifest (App Manifest → Save) and **reinstall** the app so the token gets the new scopes.

### 19.8 Ingestion

| Source | Reader | `source_key` (one logical source) | `source_group` (synced together) | Link shown in Slack |
|---|---|---|---|---|
| FAQ entries | `readers/github.py: load_faq` — `_questions/<faq_dir>/<module>/*.md`, YAML front matter `id`, `question` | `faq:<faq_dir>:<id>` | `faq:<faq_dir>` (pruned) | `https://datatalks.club/faq/<faq_dir>.html#<id>` |
| Course repos | `load_repo` — md, ipynb (cells only, no outputs or base64 images), py, sql, yaml, … ≤ 300 KB; `exclude_dirs` + every hidden folder skipped | `github:<owner>/<repo>:<path>` | `github:<owner>/<repo>@<branch>` (pruned) | GitHub blob URL |
| Slack threads | `readers/slack.py` — answered threads from the last `slack_days`, the bot's own replies left out | `slack:<channel>:<ts>` | `slack:<channel>` (not pruned) | Slack permalink |
| YouTube | `readers/youtube.py` — transcript in ~300-word chunks with their start time | `youtube:<video>` | `youtube:<video>` | `watch?v=…&t=<s>s` |
| Web pages | `readers/web.py` — HTML → text | `url:<url>` | `web` | the URL |
| Files | `load_bytes` — pdf (a document per page), md, txt, ipynb, html, code | `file:<name>` | `files` | — |
| Text | `/api/ingest/text` | your `source_key`, or derived from the title | `text` | your `url` |

Repos (including the FAQ repo) are fetched as **one tarball per repo**: `codeload.github.com` for public repos, or the API tarball endpoint when `GITHUB_TOKEN`/`GH_TOKEN` is set (needed for private repos). The original walked the GitHub API file by file and needed a token for its rate limits. Archives are extracted with Python's `data` filter, which blocks path traversal and links.

**Chunking.** Markdown is split on headings first (an FAQ answer or a README section stays together), then anything longer than `RAG_CHUNK_SIZE` (512 tokens, 50 overlap) goes through LlamaIndex's `SentenceSplitter`. Code and text go straight to the splitter. Titles and FAQ questions are embedded with the text; ids, URLs and timestamps are not (`HIDDEN_KEYS` in `readers/__init__.py`).

**Incremental, so re-running is cheap** (`indexer.py`):

```mermaid
flowchart LR
    D[Documents] --> S[split into chunks]
    S --> I["id = uuid5(source_key · sha256(chunk text) · n)"]
    I --> X{"id already in Qdrant<br/>for this source_key?"}
    X -- yes --> K[skip: no embedding]
    X -- no --> EM[embed + BM25 + upsert]
    Q[(ids in Qdrant for this source_key)] -- "not produced this time" --> DEL[delete stale chunks]
    G[(source_keys in the group)] -- "vanished upstream" --> PR[delete whole source]
```

A chunk's id comes from its content, so unchanged material is never re-embedded. On the real LLM Zoomcamp FAQ: the first run embeds 163 chunks, a re-run embeds **0**, editing one entry re-embeds **1** and deletes its old chunk, and removing two entries upstream deletes them. That replaces the original's Upstash Redis embedding cache: the vector store itself is the cache.

**Jobs.** Ingestion calls return `202` with a job record and run in the background; the record lives in the `ingest_jobs` table (SQLite / Postgres), so any Cloud Run instance can answer `GET /api/ingest/jobs/{id}`. Add `"wait": true` to block and get the stats back instead. One ingestion runs at a time per instance, because embedding is CPU-bound.

**Three ways to run it:**

```bash
# 1. CLI (same settings as the app)
python -m app.rag.cli ingest llm                         # faq + github + slack + youtube
python -m app.rag.cli ingest llm --sources faq,github
make ingest COURSE=de SOURCES=faq

# 2. REST
curl -X POST localhost:8000/api/ingest/course/llm -H 'Content-Type: application/json' -d '{"sources":["faq","github"]}'
curl localhost:8000/api/ingest/jobs/<job_id>

# 3. Scheduled: .github/workflows/ingest.yml, Mondays 06:17 UTC, every course, against the deployed test service
#    (or Actions → Ingest → Run workflow, pick course / environment / sources)
```

The scheduled workflow calls the deployed service's `/api/ingest` with that environment's API key. It is the only secret the workflow can read, so the Qdrant, Slack and Cohere keys never leave Secret Manager.

### 19.9 Run it locally with Qdrant in Docker

```bash
make qdrant                                   # docker compose up -d qdrant  (v1.19.0, dashboard on :6333)
cp .env.example .env                          # QDRANT_MODE=local, EMBED_PROVIDER=fastembed already set
make run                                      # first start downloads the bge-small model once, in the background

make ingest COURSE=llm SOURCES=faq,github     # 153 FAQ entries + ~200 repo files; a few minutes on a laptop CPU
make search COURSE=llm Q="ConnectionError localhost:9200"
make ask    COURSE=llm Q="I just found the course, can I still join?"
```

Then in Slack, in a channel mapped to `llm`: `@Ops Buddy how do I use Gemini with the OpenAI endpoint?`, or anywhere: `/faq llm how do I use Gemini with the OpenAI endpoint?`.

No Docker? `QDRANT_MODE=embedded` keeps the data in `./data/qdrant` with no server (one process at a time). Want to see what the model gets for a question? `POST /api/rag/search` returns the chunks with their scores, the funnel counts and the per-stage timings.

### 19.10 Qdrant Cloud (and test / prod)

1. Create a cluster at <https://cloud.qdrant.io> (the free tier is fine to start) and an API key for it.
2. **Locally**: `QDRANT_MODE=cloud`, `QDRANT_URL=https://<id>.<region>.cloud.qdrant.io:6333`, `QDRANT_API_KEY=…` in `.env`. Everything else stays the same.
3. **Cloud Run**: put the values where the scripts look for them, the same way as the Slack tokens:

   ```bash
   # ~/.zprofile.sh
   export QDRANT_URL=https://<id>.<region>.cloud.qdrant.io:6333
   export QDRANT_API_KEY=...            # or TEST_QDRANT_API_KEY / PROD_QDRANT_API_KEY
   export COHERE_API_KEY=...            # optional, only with RERANK_PROVIDER=cohere
   ```

   ```bash
   bash scripts/gcp-setup.sh            # re-run: creates the new secret containers (Terraform)
   bash scripts/push-secrets.sh         # QDRANT_API_KEY -> slack-agent-<env>-qdrant-api-key
   bash scripts/github-setup.sh         # QDRANT_URL -> GitHub variable; RERANK_PROVIDER=fastembed
   git push origin develop              # CD deploys with RAG on
   ```

RAG turns on in an environment **only when the `QDRANT_URL` variable exists** (`rag_enabled = vars.QDRANT_URL != ''` in `deploy.yml`), so deploys keep working before you've created the cluster. With RAG on, Terraform sets `QDRANT_MODE=cloud`, `QDRANT_COLLECTION_PREFIX=<env>_faq_`, injects `QDRANT_API_KEY` (and `COHERE_API_KEY` for Cohere) from Secret Manager, and `/ready` (the CD smoke test) also checks that Qdrant answers. The image bakes in the embedding model, the cross-encoder and the tokenizer, so a cold start downloads nothing; memory defaults to 2 GiB for that reason.

After the first deploy, fill the cloud collections once: run the **Ingest** workflow by hand (or `curl` the deployed `/api/ingest/course/<course>` with the env's API key, as in 18.8). It then runs every Monday.

### 19.11 API reference

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/rag/status` | — | Qdrant mode/reachability, embedding + reranker, per-course points by source type |
| GET | `/api/rag/courses` | — | the catalogue: channels, FAQ folder, repos, videos |
| POST | `/api/rag/query` | `question`, `course` *or* `channel_id`, `thread_id?` | `answer`, `sources[]`, `run_id` (vote with `/api/feedback`), `search_query`, `search{mode, funnel, timings_ms}` |
| POST | `/api/rag/search` | `query`, `course`, `mode?`, `top_k?`, `final_top_n?`, `source_types?`, `rerank?` | the chunks the LLM would see, with scores |
| POST | `/api/ingest/course/{course}` | `sources?`, `wait?` | job (202) or stats per source |
| POST | `/api/ingest/faq` · `/github` · `/slack` · `/youtube` · `/urls` | `course`, source options, `wait?` | job or stats |
| POST | `/api/ingest/text` | `course`, `title`, `text`, `url?`, `source_key?`, `timestamp?` | stats (synchronous) |
| POST | `/api/ingest/files` | multipart: `course`, `files[]`, `wait?` | job or stats |
| GET | `/api/ingest/jobs` · `/api/ingest/jobs/{id}` | `course?`, `limit?` | job records |
| POST | `/api/ingest/collections/{course}/reset` | — | drops the collection (e.g. after changing `EMBED_MODEL`) |

All of them sit behind `X-API-Key` when `API_KEY` is set. `/api/ingest/github` accepts `local_path` for a checkout on disk only when `APP_ENV=local`. `scripts/demo_requests.sh` runs through them with curl.

### 19.12 Configuration

| Variable | Default | |
|---|---|---|
| `RAG_ENABLED` | `true` | off = no Qdrant, no `/api/rag`, mentions go to the agent |
| `RAG_COURSES_FILE` | `config/courses.yaml` | |
| `RAG_CHANNEL_MAP` | — | `C123=llm,C456=de`, adds to the YAML |
| `RAG_DEFAULT_COURSE` | — | RAG in every unlisted channel |
| `SLACK_CMD_FAQ` | `/faq` | `/faq-test` in the test env |
| `QDRANT_MODE` | `local` | `local` · `cloud` · `embedded` · `memory` |
| `QDRANT_URL` / `QDRANT_API_KEY` | `http://localhost:6333` / — | |
| `QDRANT_COLLECTION_PREFIX` | `faq_` | `<env>_faq_` in the cloud |
| `EMBED_PROVIDER` / `EMBED_MODEL` | `fastembed` / `BAAI/bge-small-en-v1.5` | also `openai` (+ `EMBED_BASE_URL`, `EMBED_API_KEY`), `huggingface`, `hash` (tests) |
| `SPARSE_PROVIDER` | `bm25` | or `fastembed` (+ `SPARSE_MODEL`) |
| `RAG_SEARCH_MODE` | `hybrid` | `dense` (creates collections without the sparse vector) · `sparse` |
| `RAG_TOP_K` / `RAG_RECENCY_TOP_K` / `RAG_FINAL_TOP_N` | 20 / 10 / 4 | must be decreasing |
| `RAG_HYBRID_ALPHA` | 0.5 | 1 = dense only, 0 = keyword only |
| `RAG_RECENCY_HALF_LIFE_DAYS` / `RAG_RECENCY_WEIGHT` | 30 / 0.3 | |
| `RERANK_PROVIDER` / `COHERE_API_KEY` / `RERANK_MODEL` | `auto` / — / per provider | |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | 512 / 50 | |
| `GITHUB_TOKEN` (or `GH_TOKEN`) | — | private repos, higher limits |
| `FAQ_REPO` / `FAQ_LOCAL_PATH` | `DataTalksClub/faq@main` / — | |
| `SLACK_WORKSPACE_URL` | auto (`auth.test`) | base of Slack permalinks |

### 19.13 Does it find the right answer? Tests and real-data results

The RAG code has its own test layers, all in `make test` except the last:

| Test file | Covers |
|---|---|
| `test_rag_core.py` | BM25 tokenizer and weights, RRF, the catalogue (and the shipped `courses.yaml`), every reader (FAQ front matter, repo filters and notebooks, tarball download with/without token, path-traversal guard, Slack thread selection, YouTube timestamps, HTML, PDF), incremental indexing (re-run, edit, prune), dimension-mismatch guard, hybrid vs keyword results, recency ordering, re-ranker reorder + failure fallback, init failure back-off |
| `test_rag_flow.py` | The service and graph (excerpts in the prompt, real sources, run recorded), follow-up condensing with thread memory, no-hits path without an LLM call, both routers end to end (background job → poll → query → vote → reset), course-channel mention → RAG with Sources and 👍/👎 (and the vote saved), other channel → agent, `/faq` with and without a course, the agent's tool |
| `test_rag_real_data.py` | Ingests the **real** LLM Zoomcamp FAQ + course repo and checks retrieval on 10 real student-style questions (opt-in: `make test-real`) |

Qdrant-backed tests run on in-memory Qdrant by default and on a real server with `make test-qdrant` (`TEST_QDRANT_URL`). They were run against Qdrant 1.19 and PostgreSQL 16.

On the real data (153 FAQ entries → 163 chunks, 214 repo files → 1,103 chunks), with the offline `hash` embedding that the tests use, the right FAQ entry is in the top 4 for **10 of 10 questions, MRR 0.93**. That's a floor, not a ceiling: the hash model has no notion of meaning, so these numbers come mostly from BM25. With the default bge-small model, paraphrased questions do better. To measure your own changes, add cases to `CASES` in that file; with the 👍/👎 data in `agent_run_scores` (section 17.7), downvoted answers are good candidates.

### 19.14 Coming from the original faq-slack-bot

| faq-slack-bot | Here |
|---|---|
| Milvus locally, Zilliz Cloud in production, two clusters | Qdrant: Docker locally, Qdrant Cloud in the cloud, one cluster with per-env prefixes; or embedded / in-memory |
| `MilvusVectorStore`, dense only | LlamaIndex `QdrantVectorStore` with dense + BM25 sparse, RRF fusion |
| HuggingFace `bge-base-en-v1.5` (torch) | fastembed `bge-small-en-v1.5` (ONNX, no torch, baked into the image); `EMBED_PROVIDER=huggingface` for bge-base |
| Upstash Redis embedding cache | content-hash chunk ids; unchanged chunks are never re-embedded |
| `GithubRepositoryReader` (API per file, token) | one tarball per repo; token optional |
| `TimeWeightedPostprocessor(thread_ts, decay 0.4)` | half-life recency boost on any chunk with a timestamp |
| `CohereRerank(top_n=4)`, error to the user if it fails | Cohere, a local cross-encoder, or none; falls back silently if it fails |
| Four query engines, routed by hard-coded channel lists | `config/courses.yaml` + `RAG_CHANNEL_MAP`; one graph, course as a parameter |
| No follow-ups | Thread memory + a condense step |
| Feedback counts on the buttons via LangSmith | Votes in your own database, joined to question, answer and sources (section 17) |
| Ingestion scripts per course, cron per course | `/api/ingest/*`, CLI, one weekly workflow for all courses, jobs in the DB |
| Socket Mode on Fly.io | Socket Mode locally, HTTP on Cloud Run, Terraform + GitHub Actions (section 18) |

### 19.15 Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Slack: "Course search is unavailable right now" | Qdrant unreachable or the embedding model couldn't load | `GET /api/rag/status` shows `init_error` and `qdrant.reachable`; `make qdrant`; check `QDRANT_URL` |
| `init_error: Could not load model … from any source` | No access to huggingface.co (proxy, offline) | Allow it once (the model is cached), use the Docker image (model baked in), or `EMBED_PROVIDER=openai` |
| Startup log: `Qdrant client version … is incompatible with server version` | Server more than one minor version away from `qdrant-client` | Use the pinned `qdrant/qdrant:v1.19.0`, or upgrade the cluster |
| `Collection faq_llm stores 384-dim vectors but the embedding model makes 768-dim ones` | `EMBED_MODEL` changed | `POST /api/ingest/collections/llm/reset`, then ingest again |
| "I couldn't find anything about that in the … materials" | The collection is empty, or this course has nothing on it | `GET /api/rag/status` → `points`; run the ingestion |
| Mention in a course channel answered by the agent instead | Channel id not in `courses.yaml` / `RAG_CHANNEL_MAP` | Add it; ids from *View channel details* |
| Slack ingestion: `not_in_channel` / `missing_scope` | Bot not in the channel, or no `channels:history` / `channels:read` | `/invite` the bot; update the manifest and **reinstall** |
| GitHub ingestion: `404 … not found (private repo?)` | Private repo or wrong branch | `GITHUB_TOKEN`, check `branch` (`master` for the ML Zoomcamp repos) |
| YouTube: `TranscriptsDisabled` / `RequestBlocked` | No captions, or YouTube blocks the IP (common from cloud IPs) | Run that source from your machine (`make ingest SOURCES=youtube`) against the cloud cluster |
| Answers ignore a fresh FAQ edit | The weekly ingestion hasn't run yet | Run **Ingest** by hand, or `POST /api/ingest/faq` |
| `/ready` → 503 `qdrant unreachable` after deploy | Wrong `QDRANT_URL`, a bad key, or a suspended cluster (free-tier clusters can be suspended after a period of inactivity) | Check the variable and the secret; resume the cluster in the Qdrant console |
