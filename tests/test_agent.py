"""Agent flows with the mock model, plus the real OpenAI-compatible provider
against a fake Meta endpoint (so day-one API wiring is already exercised)."""
import json
import threading
import time

import httpx

from conftest import jpeg, scene
from forgemind.agent.tools import schemas
from forgemind.providers.base import ChatResult, ToolCall
from forgemind.providers.openai_compat import OpenAICompatModel


def say(rt, text):
    return rt.agent.handle(text)


def test_session_flow_and_report(rt):
    r = say(rt, "Start a session called voltage divider")
    assert r["trace"][0]["tool"] == "start_session" and "voltage divider" in r["reply"]
    say(rt, "The steps are: wire the divider; measure output voltage; swap R2")
    r = say(rt, "Log wire the divider")
    assert "Marked 'wire the divider' done" in r["reply"]
    say(rt, "Set R2 to 10k")
    r = say(rt, "I measured output voltage is 3.3 V")
    assert r["trace"][0]["tool"] == "log_measurement" and "3.3" in r["reply"]
    r = say(rt, "just checking, the probe contact")
    assert "not logged as a step" in r["reply"]
    r = say(rt, "write it up")
    assert r["trace"][0]["tool"] == "generate_report"
    md = next(rt.memory.reports_dir.glob("*.md")).read_text()
    assert "| output voltage | 3.3 | V | you said" in md
    assert "R2 | 10k" in md and "Not done: swap R2" in md and "[E" in md
    assert "Tried but not committed: the probe contact" in md


def test_repeat_session_carries_steps_and_compares(rt):
    say(rt, "start a session called voltage divider")
    say(rt, "steps are: measure output voltage; swap R2")
    say(rt, "set R2 to 10k")
    say(rt, "measured output voltage is 3.3 V")
    say(rt, "end the session")
    r = say(rt, "start a new session called voltage divider")
    assert "carried over" in r["reply"]
    say(rt, "set R2 to 4.7k")
    say(rt, "measured output voltage is 2.1 V")
    r = say(rt, "what changed?")
    assert "R2 changed from 10k to 4.7k" in r["reply"]
    assert "from 3.3 to 2.1" in r["reply"]


def test_resume(rt):
    say(rt, "start a session called solder headers")
    say(rt, "steps are: tin the pads; solder pin 1; check continuity")
    say(rt, "log tin the pads")
    say(rt, "end the session")
    r = say(rt, "where did we leave off?")
    assert r["trace"][0]["tool"] == "resume_session"
    assert "Next up: solder pin 1" in r["reply"]


def test_where_is_and_look_request(rt):
    rt.perception.process("static", scene([("multimeter", .4, .4, .6, .6)]), ts=time.time() - 30)
    r = say(rt, "where is the multimeter?")
    assert "on the bench" in r["reply"] and "bench camera" in r["reply"]
    r = say(rt, "where's my unicorn")
    assert "haven't seen" in r["reply"]
    r = say(rt, "look at the drawer")
    assert r["trace"][0]["tool"] == "go_look_at"
    assert rt.hub.pending_looks()[0]["zone"] == "drawer"
    rt.perception.process("mobile", scene([], tag=2))          # user points phone at drawer
    assert rt.hub.pending_looks() == []
    assert any(m["type"] == "look_done" for m in rt.hub.outbox)


def test_safety_unattended_alert(rt):
    now = time.time()
    rt.safety.last_present = now - 600
    rt.perception.process("static", scene([("soldering iron", .1, .4, .3, .6)]), ts=now)
    fired = rt.safety.evaluate(now)
    assert fired and "soldering iron" in fired[0]["message"]
    assert rt.safety.evaluate(now + 1) == []                       # cooldown
    rt.perception.process("static", scene([("soldering iron", .1, .4, .3, .6),
                                           ("person", .5, .1, .9, .9)]), ts=now + 2, force=True)
    assert rt.safety.user_away_for(now + 2) == 0


class FakeVision:
    """Model that asks for a close-up, then reads '4.98 V' from it."""
    name, supports_vision = "fake", True

    def model_for(self, role):
        return f"fake-{role}"

    def chat(self, messages, tools=None, model=None, json_mode=False):
        if json_mode:
            return ChatResult(content=json.dumps({"value": 4.98, "unit": "V", "text": "4.98",
                                                  "confidence": 0.9}))
        if messages[-1]["role"] == "tool":
            return ChatResult(content=json.loads(messages[-1]["content"])["summary"])
        return ChatResult(content=None, tool_calls=[ToolCall(
            "c1", "read_closeup", {"target": "multimeter display", "quantity": "supply voltage"})])


def test_read_closeup_logs_instrument_measurement(rt):
    rt.llm = rt.ctx.llm = FakeVision()
    rt.memory.experiments.start("supply check")

    def phone():
        for _ in range(100):
            req = [m for m in rt.hub.outbox if m["type"] == "capture_request"]
            if req:
                rt.hub.deliver_capture(req[-1]["id"], jpeg(scene([("multimeter", .2, .2, .8, .8)])))
                return
            time.sleep(0.02)
    threading.Thread(target=phone).start()
    r = rt.agent.handle("read the multimeter")
    assert "reads 4.98 V" in r["reply"]
    m = rt.memory.experiments.measurements(rt.memory.experiments.active()["id"])[0]
    assert (m["quantity"], m["value"], m["source"]) == ("supply voltage", 4.98, "instrument")
    assert m["keyframe"]


def test_openai_compat_provider_against_fake_meta_endpoint(rt):
    """Exercises the real client code path: tools in, tool_calls out, loop, reply."""
    seen = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        seen.append(body)
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        if body["messages"][-1]["role"] == "tool":
            msg = {"role": "assistant", "content": "It's on the bench, left side."}
        else:
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "where_is", "arguments": json.dumps({"query": "multimeter"})}}]}
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})

    llm = OpenAICompatModel("https://api.meta.ai/v1", "test-key", "muse-wonder-1.0",
                            "muse-spark-1.3", name="meta",
                            http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    rt.llm = rt.ctx.llm = llm
    rt.perception.process("static", scene([("multimeter", .05, .4, .2, .6)]))
    r = rt.agent.handle("where's the multimeter?")
    # where_is's summary is the answer: spoken directly, no second model call
    assert len(seen) == 1
    assert r["reply"].startswith("The multimeter was last seen on the bench (left side)")
    assert r["trace"][0]["tool"] == "where_is"
    assert seen[0]["model"] == "muse-wonder-1.0"
    assert {t["function"]["name"] for t in seen[0]["tools"]} >= {"where_is", "go_look_at"}
    # the system prompt is the same every turn (cacheable); the situation rides on the user turn
    from forgemind.agent import prompts
    assert seen[0]["messages"][0] == {"role": "system", "content": prompts.SYSTEM}
    user = seen[0]["messages"][-1]["content"]
    assert "Known objects: multimeter @ bench" in user and user.endswith("where's the multimeter?")

    # a tool that needs phrasing still gets the second call, with the tool result
    seen.clear()
    def handler2(request):
        body = json.loads(request.content)
        seen.append(body)
        if body["messages"][-1]["role"] == "tool":
            msg = {"role": "assistant", "content": "Just the multimeter, on the left."}
        else:
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_2", "type": "function",
                "function": {"name": "inspect_scene", "arguments": json.dumps({"camera": "static"})}}]}
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
    llm.client._client._transport = httpx.MockTransport(handler2)
    r = rt.agent.handle("what's on the bench?")
    assert len(seen) == 2 and r["reply"] == "Just the multimeter, on the left."
    tool_msg = seen[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "call_2"


def test_tool_schemas_are_valid_and_validation_rejects_bad_args(rt):
    names = [s["function"]["name"] for s in schemas()]
    assert len(names) == len(set(names)) == 18 and "alert" not in names   # alert hidden from the agent
    for s in schemas():
        assert s["function"]["parameters"]["type"] == "object"
    out = rt.agent.tools.run("log_measurement", {"quantity": "v"})
    assert "invalid arguments" in out["error"]
    out = rt.agent.tools.run("log_step", {"text": "x"})
    assert "no active session" in out["error"]
    assert "unknown tool" in rt.agent.tools.run("rm_rf", {})["error"]


def test_measurement_completes_its_planned_step_once(rt):
    tb = rt.agent.tools
    tb.run("start_session", {"name": "divider"})
    tb.run("plan_steps", {"steps": ["wire the divider", "measure the output", "swap R2"]})
    tb.run("log_step", {"text": "wired the divider"})
    out = tb.run("log_measurement", {"quantity": "output voltage", "value": 3.3, "unit": "V"})
    assert out["completed_step"] == "measure the output"
    assert out["summary"] == "Logged output voltage = 3.3 V. Marked 'measure the output' done. Next: swap R2."
    again = tb.run("log_step", {"text": "measured the output"})       # the model logs it too
    assert again.get("already_done")
    st = rt.memory.experiments.state(rt.memory.experiments.active()["id"])
    assert st["done"].count("measure the output") == 1
    assert st["next"] == "swap R2"


def test_gaps_are_reported_not_filled(rt):
    tb = rt.agent.tools
    tb.run("start_session", {"name": "divider"})
    tb.run("plan_steps", {"steps": ["wire the divider", "measure the output", "measure the input",
                                    "swap R2"]})
    tb.run("log_step", {"text": "wire the divider"})
    tb.run("log_step", {"text": "measured the output"})          # done, but no value logged
    tb.run("log_measurement", {"quantity": "input current", "value": None})
    r = tb.run("resume_session", {})
    assert r["missing"] == ["input current (logged without a value)",
                            "a value for 'measure the output' (step done, nothing logged)"]
    assert "Missing: input current (logged without a value); a value for 'measure the output'" in r["summary"]
    assert "Next up: measure the input." in r["summary"]
    rep = tb.run("generate_report", {})
    assert len(rep["missing"]) == 2 and "flags 2 missing values" in rep["summary"]


def test_where_is_reports_stale(rt):
    rt.perception.process("static", scene([("multimeter", .05, .4, .2, .6)]), ts=time.time() - 7200)
    out = rt.agent.tools.run("where_is", {"query": "multimeter"})
    assert out["stale"] is True and out["confidence"] < 0.4
    assert "may have moved" in out["summary"]


def test_resume_yesterdays_session_states_pending_step(rt):
    """Acceptance check #2: resume a previous day's session with the right next step."""
    tb = rt.agent.tools
    tb.run("start_session", {"name": "voltage divider test"})
    tb.run("plan_steps", {"steps": ["wire the divider", "measure the output", "swap R2 for 22k"]})
    tb.run("log_step", {"text": "wired the divider"})
    tb.run("log_measurement", {"quantity": "output voltage", "value": 3.3, "unit": "V"})
    tb.run("end_session", {})
    db = rt.memory.db
    for table in ("events", "steps", "measurements"):          # it all happened yesterday
        db.execute(f"UPDATE {table} SET ts = ts - 86400")
    tb.run("start_session", {"name": "something else"})       # another session in between
    r = say(rt, "let's continue the voltage divider test")
    assert r["trace"][0]["tool"] == "resume_session"
    assert "Resuming 'voltage divider test', last active 1 day ago" in r["reply"]
    assert "Next up: swap R2 for 22k." in r["reply"]
    assert "output voltage = 3.3 V" in r["reply"]


class _Scripted:
    """Fake model returning queued replies; records what it was sent."""
    name, supports_vision = "scripted", False

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def model_for(self, role):
        return "scripted"

    def chat(self, messages, tools=None, model=None, json_mode=False):
        from forgemind.providers.base import ChatResult, ToolCall
        self.seen.append(messages)
        r = self.replies.pop(0)
        if isinstance(r, tuple):
            return ChatResult(content=None, tool_calls=[ToolCall(id="c1", name=r[0], arguments=r[1])])
        return ChatResult(content=r)


def _use(rt, llm):
    rt.llm = rt.ctx.llm = llm
    return llm


def test_guard_retries_a_false_action_claim_and_the_tool_runs(rt):
    llm = _use(rt, _Scripted(["Started 'voltage divider test'.",
                              ("start_session", {"name": "voltage divider test"})]))
    r = rt.agent.handle("start a session called voltage divider test")
    assert [t["tool"] for t in r["trace"]] == ["start_session"]
    assert r["reply"] == "Started session 'voltage divider test'."
    assert any("called no tool" in (m.get("content") or "") for m in llm.seen[1])   # the nudge
    assert rt.memory.experiments.active()["name"] == "voltage divider test"


def test_guard_never_speaks_a_false_confirmation_twice(rt):
    _use(rt, _Scripted(["Logged: swap R2 for 22k.", "Okay, logged it."]))
    r = rt.agent.handle("set R2 to 10k")
    assert r["trace"] == [] and r["reply"].startswith("Sorry, I didn't actually record that")
    ev = rt.memory.db.all("SELECT payload FROM events WHERE type='guard'")
    assert len(ev) == 2


def test_guard_leaves_recall_answers_alone(rt):
    llm = _use(rt, _Scripted(["You measured 3.3 V on the output earlier."]))
    r = rt.agent.handle("what was the output voltage?")
    assert r["reply"] == "You measured 3.3 V on the output earlier." and len(llm.seen) == 1


def test_compound_request_runs_both_tools(rt):
    """Found by the simulated-user eval: the report request after a measurement was dropped."""
    llm = _use(rt, _Scripted([("log_measurement", {"quantity": "output voltage", "value": 3.3, "unit": "V"}),
                              ("generate_report", {}),
                              "The divider was tested.",          # the report's summary paragraph
                              "Done: logged it and wrote the report."]))
    rt.agent.tools.run("start_session", {"name": "divider"})
    r = rt.agent.handle("I measured 3.3 volts at the output. Please write up a report.")
    assert [t["tool"] for t in r["trace"]] == ["log_measurement", "generate_report"]
    assert "Logged output voltage = 3.3 V." in r["reply"] and "report" in r["reply"]
    assert len(llm.seen) == 4          # tool, tool (+ report summary), then "finished"


def test_same_reading_twice_is_one_measurement(rt):
    tb = rt.agent.tools
    tb.run("start_session", {"name": "divider"})
    tb.run("log_measurement", {"quantity": "output voltage", "value": 3.3, "unit": "V"})
    again = tb.run("log_measurement", {"quantity": "Output voltage", "value": 3.3, "unit": "V"})
    assert again["summary"].startswith("Already logged")
    assert len(rt.memory.experiments.measurements(rt.memory.experiments.active()["id"])) == 1
