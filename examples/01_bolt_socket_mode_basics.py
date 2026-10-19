"""
Step 1 of the journey: plain Slack Bolt (sync) in Socket Mode, no FastAPI.

This is the original working script, with three small fixes:
  1. `from slack_bolt import logger` imports a *module*, not a Logger, so
     logger.exception(...) raised AttributeError inside the except block.
     We use the standard logging module instead.
  2. The usage hint for the add command said "/add" — it now names the real command.
  3. The app_mention handler strips the "<@BOTID>" token before sending text to
     the model and replies in a thread.

Run:  python examples/01_bolt_socket_mode_basics.py
Needs: SLACK_BOT_TOKEN, SLACK_SOCKET_TOKEN (xapp-...), OPENAI_API_KEY in .env
"""

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bolt-basics")

client = OpenAI()  # reads OPENAI_API_KEY
app = App(token=os.environ["SLACK_BOT_TOKEN"])


# 1. Keyword trigger — fires on any message containing "hello" in channels the bot is in
@app.message("hello")
def message_hello(message, say):
    say(f"Hello <@{message['user']}>")


# 2. Slash command — deterministic, no LLM
@app.command("/welcomeagentadd")
def handle_add_command(ack, respond, command):
    ack()
    try:
        a, b = map(float, command["text"].split())
        respond(f"{a:g} + {b:g} = {a + b:g}")
    except ValueError:
        respond("Please provide two numbers, e.g. /welcomeagentadd 2 3")


# 3. Slash command — single LLM call
@app.command("/welcomeagentanalyze")
def handle_analyze_command(ack, respond, command):
    ack()  # within 3 seconds, before the slow LLM call
    content = (command.get("text") or "").strip()
    if not content:
        respond("Please provide content to analyze.\nExample: /welcomeagentanalyze Explain LangGraph nodes and edges")
        return
    try:
        response = client.responses.create(
            model="gpt-4o",
            instructions=(
                "You are a content analysis assistant. Analyze the user's content and give a clear "
                "summary, key points, insights and recommendations. Distinguish facts from assumptions. "
                "Format for Slack: *bold*, bullets with •, no markdown headings."
            ),
            input=content,
        )
        respond(response.output_text or "No analysis was generated.")
    except Exception:
        log.exception("Content analysis failed")
        respond("Sorry, I couldn't analyze the content. Please try again.")


# 4. @mention -> LLM, reply in thread
@app.event("app_mention")
def handle_app_mention_events(event, say, context):
    text = event.get("text", "").replace(f"<@{context.bot_user_id}>", "").strip()
    response = client.responses.create(model="gpt-4o", input=text or "Say hi briefly.")
    say(text=response.output_text, thread_ts=event.get("thread_ts") or event["ts"])


# Silence "Unhandled request" warnings for other channel messages
@app.event("message")
def ignore_other_messages():
    pass


if __name__ == "__main__":
    SocketModeHandler(app, os.environ.get("SLACK_SOCKET_TOKEN") or os.environ["SLACK_APP_TOKEN"]).start()
