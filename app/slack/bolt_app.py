"""
Slack side of the house: an AsyncApp (Bolt for Python) with every listener.

Bolt picks the FIRST listener that matches an incoming request and stops, so
the order of registration below matters:

    1. message  "hello" / "hi" / "hey"     -> greeting (no LLM)
    2. message  in a DM (channel_type=im)  -> LangGraph agent
    3. message  anything else               -> ignored (silences "unhandled" warnings)
    4. event    app_mention                 -> course channel? RAG over that course's Qdrant collection
                                               otherwise the LangGraph agent. Replies in thread.
    5. command  /agent <question>          -> LangGraph agent via response_url
    6. command  /agent-add 2 3             -> the original adder demo
    7. command  /faq [course] <question>   -> RAG for a course (named, or the channel's course)
    8. action   upvote / downvote           -> 👍/👎 buttons (agent AND RAG answers), saved to the DB

The same AsyncApp object is driven either by Socket Mode (WebSocket) or by
FastAPI's /slack/events route (HTTP) — see app/main.py. Handlers don't care.
"""

from __future__ import annotations

import logging
import re

from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

from app.config import Settings
from app.db.feedback_store import (
    FeedbackStore,
    FeedbackType,
    InMemoryFeedbackStore,
    UnknownRunError,
    feedback_from_body,
)
from app.rag.service import RagService, RagUnavailable
from app.services.agent_service import AgentService
from app.services.dedup import TTLDeduper
from app.slack.blocks import answer_blocks, fallback_text, rag_answer_blocks, rag_thinking_blocks, thinking_blocks
from app.slack.formatting import to_mrkdwn

log = logging.getLogger(__name__)


def to_fallback(text: str) -> str:
    out = to_mrkdwn(text)
    return out if len(out) <= 3000 else out[:2997] + "..."


_GREETING = re.compile(r"^\s*(hello|hi|hey)\b", re.IGNORECASE)


def _is_human_dm(message: dict) -> bool:
    return (
        message.get("channel_type") == "im"
        and not message.get("subtype")  # skips edits, deletes, joins, bot_message...
        and not message.get("bot_id")
    )


def create_bolt_app(
    settings: Settings,
    agent: AgentService,
    deduper: TTLDeduper | None = None,
    store: FeedbackStore | None = None,
    rag: RagService | None = None,
) -> AsyncApp:
    deduper = deduper or TTLDeduper()
    store = store or InMemoryFeedbackStore()

    app = AsyncApp(
        token=settings.slack_bot_token,
        signing_secret=settings.slack_signing_secret or None,
        # In Socket Mode there are no HTTP requests to verify; in HTTP mode the
        # signing secret is mandatory and Bolt rejects unsigned requests with 401.
        request_verification_enabled=settings.slack_mode == "http",
        # Let Bolt ack first and run the listener afterwards (needed to stay under
        # Slack's 3-second limit while the LLM thinks for 5-20 seconds).
        process_before_response=False,
    )

    # ------------------------------------------------------------------ helpers
    async def first_delivery(key: str) -> bool:
        """Two-level de-dup: the in-process cache catches repeats on this instance
        for free; the database claim catches repeats that land on another Cloud Run
        instance. If the DB is down we still answer (better twice than never)."""
        if not deduper.first_time(key):
            return False
        try:
            return await store.claim_event(key)
        except Exception:  # noqa: BLE001
            log.warning("event claim failed for %s; answering anyway", key, exc_info=True)
            return True

    async def run_agent_in_thread(
        client: AsyncWebClient, *, channel: str, thread_ts: str, text: str, user: str, source: str, reply_ts_anchor: str
    ) -> None:
        """Post a placeholder, run the agent, then edit the placeholder with the answer.
        Editing (chat.update) instead of posting a second message keeps the thread clean."""
        await _react(client, channel, reply_ts_anchor, "eyes")
        placeholder = await client.chat_postMessage(
            channel=channel, thread_ts=thread_ts, text="Working on it…", blocks=thinking_blocks(user)
        )
        try:
            result = await agent.run(
                text,
                thread_id=f"slack:{channel}:{thread_ts}",
                user_id=user,
                channel_id=channel,
                source=source,
            )
            await client.chat_update(
                channel=channel, ts=placeholder["ts"], text=fallback_text(result), blocks=answer_blocks(result)
            )
            await _react(client, channel, reply_ts_anchor, "white_check_mark")
        except Exception:
            log.exception("agent failed channel=%s thread=%s", channel, thread_ts)
            await client.chat_update(
                channel=channel,
                ts=placeholder["ts"],
                text=":warning: Something broke on my side while working on that. Try again in a minute.",
                blocks=None,
            )
            await _react(client, channel, reply_ts_anchor, "x")
        finally:
            await _unreact(client, channel, reply_ts_anchor, "eyes")

    async def run_rag_in_thread(
        client: AsyncWebClient, *, channel: str, thread_ts: str, text: str, user: str, course_key: str, anchor_ts: str
    ) -> None:
        """Course channel: same placeholder -> update pattern, answered from the course collection."""
        course = rag.catalog.get(course_key)
        await _react(client, channel, anchor_ts, "eyes")
        placeholder = await client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text="Searching the course materials…",
            blocks=rag_thinking_blocks(course.name, user),
            unfurl_links=False,
        )
        try:
            res = await rag.answer(
                text,
                course_key,
                thread_id=f"slack:{channel}:{thread_ts}",
                user_id=user,
                channel_id=channel,
                source="slack_rag",
            )
            await client.chat_update(
                channel=channel,
                ts=placeholder["ts"],
                text=to_fallback(res.answer),
                blocks=rag_answer_blocks(res, course.name),
                unfurl_links=False,
                unfurl_media=False,
            )
            await _react(client, channel, anchor_ts, "white_check_mark")
        except RagUnavailable as exc:
            log.error("RAG unavailable: %s", exc)
            await client.chat_update(
                channel=channel,
                ts=placeholder["ts"],
                blocks=None,
                text=":warning: Course search is unavailable right now (the search index can't be reached). Try again later.",
            )
            await _react(client, channel, anchor_ts, "x")
        except Exception:
            log.exception("RAG answer failed channel=%s thread=%s", channel, thread_ts)
            await client.chat_update(
                channel=channel,
                ts=placeholder["ts"],
                blocks=None,
                text=":warning: Something broke on my side while searching. Try again in a minute.",
            )
            await _react(client, channel, anchor_ts, "x")
        finally:
            await _unreact(client, channel, anchor_ts, "eyes")

    # ------------------------------------------------------------------ 1. greeting
    @app.message(_GREETING)
    async def say_hello(message, say):
        # `say` posts to the same channel the event came from. thread_ts keeps it
        # in a thread if the greeting was inside one.
        await say(
            text=f"Hello <@{message['user']}> :wave: Mention me, DM me, or try `{settings.slack_cmd_agent} <question>`.",
            thread_ts=message.get("thread_ts"),
        )

    # ------------------------------------------------------------------ 2. DMs -> agent
    async def human_dm_only(message) -> bool:
        return _is_human_dm(message)

    @app.message(matchers=[human_dm_only])
    async def handle_dm(message, client: AsyncWebClient):
        if not await first_delivery(f"{message['channel']}:{message['ts']}"):
            return
        # In a DM, a top-level message continues one long "main" conversation
        # (feels like a chat). A reply inside a DM thread gets its own memory.
        if message.get("thread_ts"):
            await run_agent_in_thread(
                client,
                channel=message["channel"],
                thread_ts=message["thread_ts"],
                text=message.get("text", ""),
                user=message["user"],
                source="slack_dm",
                reply_ts_anchor=message["ts"],
            )
        else:
            await _run_dm_main(client, message)

    async def _run_dm_main(client: AsyncWebClient, message: dict) -> None:
        channel, user = message["channel"], message["user"]
        placeholder = await client.chat_postMessage(channel=channel, text="Working on it…", blocks=thinking_blocks())
        try:
            result = await agent.run(
                message.get("text", ""),
                thread_id=f"slack:dm:{channel}",
                user_id=user,
                channel_id=channel,
                source="slack_dm",
            )
            await client.chat_update(
                channel=channel, ts=placeholder["ts"], text=fallback_text(result), blocks=answer_blocks(result)
            )
        except Exception:
            log.exception("agent failed in DM channel=%s", channel)
            await client.chat_update(
                channel=channel, ts=placeholder["ts"], text=":warning: Something broke on my side. Try again shortly."
            )

    # ------------------------------------------------------------------ 3. everything else
    @app.event("message")
    async def ignore_other_messages():
        # message.channels delivers every message in channels the bot is in.
        # We only act on mentions (app_mention) and greetings, so drop the rest.
        return

    # ------------------------------------------------------------------ 4. @mentions -> agent
    @app.event("app_mention")
    async def handle_mention(event, client: AsyncWebClient, context):
        if not await first_delivery(f"{event['channel']}:{event['ts']}"):
            return
        bot_user_id = context.get("bot_user_id") or ""
        # Strip only the bot's own mention; other people's <@U...> stay so the
        # agent knows who else is being talked about.
        text = event.get("text", "").replace(f"<@{bot_user_id}>", "").strip()
        # Reply in the thread. If the mention started a new thread, its own ts is the thread root.
        thread_ts = event.get("thread_ts") or event["ts"]
        course = rag.catalog.for_channel(event["channel"]) if rag else None
        if course is not None:
            # Course channel -> RAG, exactly like the original faq-slack-bot's channel routing
            await run_rag_in_thread(
                client,
                channel=event["channel"],
                thread_ts=thread_ts,
                text=text,
                user=event.get("user", "unknown"),
                course_key=course.key,
                anchor_ts=event["ts"],
            )
            return
        await run_agent_in_thread(
            client,
            channel=event["channel"],
            thread_ts=thread_ts,
            text=text,
            user=event.get("user", "unknown"),
            source="slack_mention",
            reply_ts_anchor=event["ts"],
        )

    # ------------------------------------------------------------------ 5. /agent <question>
    @app.command(settings.slack_cmd_agent)
    async def handle_agent_command(ack, command, respond):
        question = (command.get("text") or "").strip()
        if not question:
            await ack(f"Usage: `{settings.slack_cmd_agent} how many PTO days do I get per year?`")
            return
        # ack() within 3 seconds or Slack shows "dispatch_failed". The text here is
        # an ephemeral message only the caller sees.
        await ack(f":hourglass_flowing_sand: Thinking about: _{question}_")
        try:
            result = await agent.run(
                question,
                thread_id=f"slack:cmd:{command['channel_id']}:{command['user_id']}",
                user_id=command["user_id"],
                channel_id=command["channel_id"],
                source="slash_command",
            )
            # respond() POSTs to command["response_url"] — valid for 30 minutes, up to 5 times.
            # It works even in channels the bot hasn't joined, unlike chat.postMessage.
            await respond(
                response_type="in_channel",
                text=fallback_text(result),
                blocks=[
                    {
                        "type": "context",
                        "elements": [{"type": "mrkdwn", "text": f"<@{command['user_id']}> asked: _{question}_"}],
                    },
                    *answer_blocks(result),
                ],
                replace_original=False,
            )
        except Exception:
            log.exception("slash command failed")
            await respond(response_type="ephemeral", text=":warning: Something broke while answering. Try again.")

    # ------------------------------------------------------------------ 7. /faq [course] <question>
    @app.command(settings.slack_cmd_faq)
    async def handle_faq_command(ack, command, respond):
        if rag is None:
            await ack("Course search is turned off on this deployment (RAG_ENABLED=false).")
            return
        words = (command.get("text") or "").strip().split(maxsplit=1)
        course = None
        if words and words[0].lower() in rag.catalog:
            course = rag.catalog.get(words[0].lower())
            question = words[1] if len(words) > 1 else ""
        else:
            course = rag.catalog.for_channel(command["channel_id"])
            question = " ".join(words)
        if not question or course is None:
            keys = ", ".join(f"`{k}`" for k in rag.catalog.keys())
            await ack(f"Usage: `{settings.slack_cmd_faq} [course] question` — courses: {keys}")
            return
        await ack(f":mag: Searching the {course.name} materials for: _{question}_")
        try:
            res = await rag.answer(
                question,
                course.key,
                thread_id=f"slack:faq:{command['channel_id']}:{command['user_id']}",
                user_id=command["user_id"],
                channel_id=command["channel_id"],
                source="slack_rag",
            )
            await respond(
                response_type="in_channel",
                text=to_fallback(res.answer),
                blocks=[
                    {"type": "context", "elements": [{"type": "mrkdwn", "text": f"<@{command['user_id']}> asked: _{question}_"}]},
                    *rag_answer_blocks(res, course.name),
                ],
                replace_original=False,
            )
        except RagUnavailable:
            await respond(response_type="ephemeral", text=":warning: Course search is unavailable right now.")
        except Exception:
            log.exception("/faq failed")
            await respond(response_type="ephemeral", text=":warning: Something broke while searching. Try again.")

    # ------------------------------------------------------------------ 6. /agent-add 2 3
    @app.command(settings.slack_cmd_add)
    async def handle_add_command(ack, command):
        try:
            a, b = map(float, (command.get("text") or "").split())
            total = a + b
            await ack(f"{a:g} + {b:g} = *{total:g}*")
        except ValueError:
            await ack(f"Give me two numbers, e.g. `{settings.slack_cmd_add} 2 3`")

    # ------------------------------------------------------------------ 7a. 👍 upvote
    @app.action("upvote")
    async def handle_upvote(ack, body, client: AsyncWebClient):
        await ack()  # within 3s, before touching the database
        await save_vote(body, "upvote", client)

    # ------------------------------------------------------------------ 7b. 👎 downvote
    @app.action("downvote")
    async def handle_downvote(ack, body, client: AsyncWebClient):
        await ack()
        await save_vote(body, "downvote", client)

    # ------------------------------------------------------------------ shared vote logic
    async def save_vote(body: dict, feedback_type: FeedbackType, client: AsyncWebClient) -> None:
        """Parse the click, upsert it into agent_feedback, tell the clicker (only them) it landed."""
        user_id = (body.get("user") or {}).get("id")
        channel_id = (body.get("channel") or {}).get("id")
        try:
            record = feedback_from_body(body, feedback_type)
            tally = await store.add_feedback(record)
        except UnknownRunError:
            log.warning("vote for unknown run user=%s", user_id)
            text = "I couldn't find that answer in my records (it may be older than the database). Thanks anyway!"
        except ValueError as exc:
            log.warning("malformed feedback payload: %s", exc)
            return
        except Exception:
            log.exception("could not save feedback user=%s", user_id)
            text = ":warning: I couldn't save your vote just now. Please try again in a moment."
        else:
            log.info(
                "feedback saved type=%s run=%s user=%s change=%s tally=+%s/-%s",
                feedback_type,
                record.run_id,
                user_id,
                tally.changed,
                tally.upvotes,
                tally.downvotes,
            )
            if tally.changed == "unchanged":
                lead = "You'd already voted that way, so nothing changed."
            elif tally.changed == "changed":
                lead = "Changed your vote."
            else:
                lead = "Thanks, noted :pray:" if feedback_type == "upvote" else "Thanks, that helps me get better."
            text = f"{lead}  This answer: :thumbsup: {tally.upvotes}  :thumbsdown: {tally.downvotes}"

        if channel_id and user_id:
            try:
                await client.chat_postEphemeral(channel=channel_id, user=user_id, text=text)
            except Exception:  # noqa: BLE001 - e.g. bot not in channel for a response_url message
                log.debug("could not post ephemeral feedback ack", exc_info=True)

    # ------------------------------------------------------------------ errors
    @app.error
    async def global_error_handler(error, body):
        log.exception("unhandled Bolt error: %s | body keys=%s", error, list((body or {}).keys()))

    return app


async def _react(client: AsyncWebClient, channel: str, ts: str, name: str) -> None:
    """Reactions are a nice 'I saw it' signal, but never worth failing over
    (missing reactions:write scope, already reacted, etc.)."""
    try:
        await client.reactions_add(channel=channel, timestamp=ts, name=name)
    except Exception:  # noqa: BLE001
        pass


async def _unreact(client: AsyncWebClient, channel: str, ts: str, name: str) -> None:
    try:
        await client.reactions_remove(channel=channel, timestamp=ts, name=name)
    except Exception:  # noqa: BLE001
        pass
