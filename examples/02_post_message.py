"""
The other direction: your code -> Slack, using only the Web API (no events).

Good for cron jobs, CI pipelines, ETL alerts. The server version of this is
POST /api/slack/notify in the main app.

Note: writing "@welcomeagenticapp" as plain text does NOT mention anyone.
A real mention is "<@U0123ABCD>" (the bot's user id). And Slack never
delivers a bot's own messages back to it as app_mention events, so a bot
can't trigger itself this way — which is a good thing (no infinite loops).

Run:  python examples/02_post_message.py
"""

import os

from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

load_dotenv()
client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
channel = os.environ["SLACK_CHANNEL_ID"]

try:
    # Plain text + Block Kit. `text` is the notification/fallback; `blocks` is what people see.
    resp = client.chat_postMessage(
        channel=channel,
        text="Hello World from Python! :tada:",
        blocks=[
            {"type": "header", "text": {"type": "plain_text", "text": "Hello World from Python :tada:"}},
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": "*Pipeline:* nightly-etl\n*Status:* :white_check_mark: success"},
            },
            {"type": "context", "elements": [{"type": "mrkdwn", "text": "sent by examples/02_post_message.py"}]},
        ],
    )
    print(f"Message sent: channel={resp['channel']} ts={resp['ts']}")

    # Reply in a thread under the message we just sent
    client.chat_postMessage(channel=channel, thread_ts=resp["ts"], text="Details go in the thread :thread:")
except SlackApiError as e:
    # not_in_channel -> /invite @YourBot, or add chat:write.public scope for public channels
    # channel_not_found -> wrong id, or private channel the bot isn't in
    # missing_scope -> add the scope under OAuth & Permissions and REINSTALL the app
    print(f"Error sending message: {e.response['error']}")
