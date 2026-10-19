from app.agent.tools import TICKETS, calculator, get_current_time, search_knowledge_base
from app.services.dedup import TTLDeduper
from app.slack.formatting import chunk, to_mrkdwn


def test_calculator_does_math():
    assert calculator.invoke({"expression": "(1250 * 0.18) + 40"}) == "265"
    assert calculator.invoke({"expression": "21 / 4"}) == "5.25"


def test_calculator_refuses_code():
    out = calculator.invoke({"expression": "__import__('os').system('ls')"})
    assert out.startswith("Error")


def test_time_tool_handles_bad_zone():
    assert "Error" in get_current_time.invoke({"tz": "Mars/Olympus"})
    assert "202" in get_current_time.invoke({"tz": "America/New_York"})


def test_kb_search_finds_pto():
    out = search_knowledge_base.invoke({"query": "how many PTO days per year"})
    assert out.startswith("[KB-003]")


def test_kb_search_no_match():
    assert "No knowledge base articles" in search_knowledge_base.invoke({"query": "zzzz qqqq"})


def test_ticket_store_roundtrip():
    TICKETS.clear()
    t = TICKETS.create("VPN down", "drops every 5 min", "P2", "U123")
    assert TICKETS.get(t["id"].lower())["requester"] == "U123"
    assert len(TICKETS.list_open()) == 1


def test_markdown_to_mrkdwn():
    md = "# Summary\n**Bold** and [docs](https://x.io) and ~~old~~\n- one\n- two\nA < B & C <@U123>"
    out = to_mrkdwn(md)
    assert out.splitlines()[0] == "*Summary*"
    assert "*Bold*" in out
    assert "<https://x.io|docs>" in out
    assert "~old~" in out
    assert "• one" in out
    assert "A &lt; B &amp; C <@U123>" in out


def test_code_blocks_untouched():
    out = to_mrkdwn("see\n```python\nx = a**2\n```")
    assert "```\nx = a**2\n```" in out


def test_chunking_respects_limit():
    pieces = chunk("line\n" * 2000, size=500)
    assert all(len(p) <= 500 for p in pieces)
    assert sum(p.count("line") for p in pieces) == 2000


def test_dedup():
    d = TTLDeduper(ttl_seconds=60)
    assert d.first_time("C1:123.4")
    assert not d.first_time("C1:123.4")
