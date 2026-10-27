"""User notes memory (F-EXP-9)."""
import json

from forgemind.memory.notes import SIGNALS
from forgemind.providers.base import ChatResult


class NoteWriter:
    name, supports_vision = "fake", False

    def __init__(self):
        self.calls = []

    def model_for(self, role):
        return f"fake-{role}"

    def chat(self, messages, tools=None, model=None, json_mode=False):
        self.calls.append((model, json_mode, messages[-1]["content"]))
        if json_mode:
            return ChatResult(content=json.dumps({"notes": [
                {"kind": "goal", "text": "wants to finish the divider today"},
                {"kind": "preference", "text": "keeps hex keys in the drawer"}]}))
        return ChatResult(content="Sounds good.")


def test_signal_gate():
    assert SIGNALS.search("I want to finish the divider today")
    assert SIGNALS.search("the probe keeps slipping off")
    assert SIGNALS.search("I keep my hex keys in the drawer")
    assert not SIGNALS.search("where is the multimeter?")
    assert not SIGNALS.search("set R2 to 10k")


def test_notes_extracted_in_background_and_shown_to_the_agent(rt):
    rt.settings.notes = True
    llm = rt.llm = rt.ctx.llm = NoteWriter()
    rt.agent.tools.run("start_session", {"name": "divider"})
    rt.agent.handle("I want to finish the divider today, and I keep hex keys in the drawer")
    rt.agent.last_notes_thread.join(5)
    assert [(c[0], c[1]) for c in llm.calls] == [("fake-agent", False), ("fake-worker", True)]
    notes = rt.memory.notes.recent()
    assert {n["text"] for n in notes} == {"wants to finish the divider today",
                                          "keeps hex keys in the drawer"}
    from forgemind.agent import prompts
    sit = prompts.situation(rt.ctx)
    assert "goal: wants to finish the divider today" in sit
    r = rt.agent.tools.run("resume_session", {"name": "divider"})
    assert "You mentioned: wants to finish the divider today." in r["summary"]
    # duplicates are not stored twice
    rt.agent.handle("I want to finish the divider today")
    rt.agent.last_notes_thread.join(5)
    assert len(rt.memory.notes.recent()) == 2


def test_no_extraction_without_signal_or_when_disabled(rt):
    rt.settings.notes = True
    llm = rt.llm = rt.ctx.llm = NoteWriter()
    rt.agent.handle("where is the multimeter?")
    rt.agent.last_notes_thread.join(5)
    assert len(llm.calls) == 1                       # only the agent call
    rt.settings.notes = False
    rt.agent.handle("I want to finish today")
    assert len(llm.calls) == 2 and rt.memory.notes.recent() == []


def test_notes_api_lists_and_forgets(rt):
    from fastapi.testclient import TestClient

    from forgemind.server.app import create_app
    nid = rt.memory.notes.add("preference", "keeps hex keys in the drawer")
    h = {"Authorization": "Bearer secret-token"}
    with TestClient(create_app(rt), base_url="https://testserver") as c:
        assert [n["text"] for n in c.get("/api/notes", headers=h).json()["notes"]] == \
            ["keeps hex keys in the drawer"]
        assert c.delete(f"/api/notes/{nid}", headers=h).json() == {"ok": True}
        assert c.get("/api/notes", headers=h).json()["notes"] == []
