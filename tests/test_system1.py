"""System 1 fast path: decisions without Spark, argument filling, background check + undo."""
import time

import pytest

from forgemind.agent.system1 import OTHER, System1, parse_measurement
from forgemind.providers.base import ChatResult, ToolCall
from conftest import scene


class StubLaya:
    """Stands in for the fine-tuned Laya: tool answers from a table, objects by substring."""

    def __init__(self, table):
        self.table = table          # text -> (tool, conf, multi)

    def predict(self, text, questions):
        if "tool" in questions:
            tool, conf, multi = self.table[text]
            return {"answers": {"tool": {"choice": tool, "answer_confidence": conf},
                                "multi": {"noul": multi}}}
        crit = questions["x"]["criteria"]
        pick = next((k for k in crit if k != OTHER and k.lower() in text.lower()), None)
        if pick is None and "most recent" in crit:
            pick = "most recent"
        return {"answers": {"x": {"choice": pick or OTHER, "answer_confidence": 0.95}}}


class SparkStub:
    """Records calls; answers the background check with a fixed tool call (or none)."""

    def __init__(self, check_calls=()):
        self.calls, self.check_calls = 0, list(check_calls)
        self.supports_vision = False

    def model_for(self, role):
        return "stub"

    def chat(self, messages, tools=None, model=None, **kw):
        self.calls += 1
        return ChatResult(content="" if self.check_calls else "Sure.",
                          tool_calls=[ToolCall(f"c{i}", n, a) for i, (n, a) in enumerate(self.check_calls)])


def fast_agent(rt, table, spark=None):
    rt.agent.ctx.llm = spark or SparkStub()
    rt.agent.system1 = System1(rt.settings, rt.memory, backend=StubLaya(table))
    return rt.agent


def test_where_is_answered_without_spark(rt):
    rt.perception.process("static", scene([("hex key", .4, .4, .5, .5)]), ts=time.time() - 5)
    agent = fast_agent(rt, {"where is my hex key?": ("where_is", 0.99, 0.01)})
    out = agent.handle("where is my hex key?")
    assert out["fast"] and out["trace"][0]["args"] == {"query": "hex key"}
    assert "bench" in out["reply"].lower()
    assert agent.ctx.llm.calls == 0


@pytest.mark.parametrize("text,decision", [
    ("where is my hex key?", ("where_is", 0.90, 0.01)),          # unsure
    ("start a session called x", ("start_session", 0.99, 0.0)),   # not a fast tool
    ("where is my hex key?", ("where_is", 0.99, 0.7)),           # looks like two requests
    ("where's my flux pen?", ("where_is", 0.99, 0.0)),           # object not in memory
])
def test_falls_back_to_spark(rt, text, decision):
    rt.perception.process("static", scene([("hex key", .4, .4, .5, .5)]), ts=time.time() - 5)
    agent = fast_agent(rt, {text: decision})
    out = agent.handle(text)
    assert not out.get("fast") and agent.ctx.llm.calls >= 1


def test_compound_never_takes_the_fast_path(rt):
    text = "I measured 3.3 V. Please write up a report."
    agent = fast_agent(rt, {text: ("log_measurement", 0.99, 0.01)})
    assert not agent.handle(text).get("fast")


def test_fast_measurement_confirmed_by_spark(rt):
    rt.memory.experiments.start("divider")
    text = "I measured 3.3 volts on the output"
    spark = SparkStub([("log_measurement", {"quantity": "output voltage", "value": 3300, "unit": "mV"})])
    agent = fast_agent(rt, {text: ("log_measurement", 0.98, 0.01)}, spark)
    out = agent.handle(text)
    assert out["fast"] and out["reply"].startswith("Logged output voltage = 3.3 V")
    agent.last_verify_thread.join(5)
    sid = rt.memory.experiments.active()["id"]
    assert [m["value"] for m in rt.memory.experiments.measurements(sid)] == [3.3]
    assert not [m for m in rt.hub.outbox if m["type"] == "say"]


def test_fast_measurement_undone_when_spark_disagrees(rt):
    s = rt.memory.experiments.start("divider")
    rt.memory.experiments.plan(s["id"], ["measure R1 resistance"])
    text = "R1 is 4.7k"
    spark = SparkStub([("set_config", {"key": "R1", "value": "4.7k"})])
    agent = fast_agent(rt, {text: ("log_measurement", 0.97, 0.01)}, spark)
    out = agent.handle(text)
    assert out["fast"] and "r1 resistance" in out["reply"].lower()
    agent.last_verify_thread.join(5)
    assert rt.memory.experiments.measurements(s["id"]) == []
    assert rt.memory.experiments.config(s["id"])["R1"]["value"] == "4.7k"
    steps = rt.memory.experiments.steps(s["id"])
    assert steps[0]["status"] == "planned"             # the tick from the measurement is undone
    said = [m["text"] for m in rt.hub.outbox if m["type"] == "say"]
    assert said and said[0].startswith("Correction: I removed r1 resistance.") and "R1 is 4.7k" in said[0]


def test_spark_ticking_the_same_step_is_agreement(rt):
    s = rt.memory.experiments.start("divider")
    rt.memory.experiments.plan(s["id"], ["measure the output"])
    text = "I measured 3.3 volts on the output"
    spark = SparkStub([("log_measurement", {"quantity": "output voltage", "value": 3.3, "unit": "V"}),
                       ("log_step", {"text": "measure the output"})])
    agent = fast_agent(rt, {text: ("log_measurement", 0.98, 0.01)}, spark)
    assert agent.handle(text)["fast"]
    agent.last_verify_thread.join(5)
    assert len(rt.memory.experiments.measurements(s["id"])) == 1
    assert not [m for m in rt.hub.outbox if m["type"] == "say"]


def test_spark_extra_request_is_added_not_undone(rt):
    s = rt.memory.experiments.start("divider")
    text = "I measured 3.3 volts on the output"
    spark = SparkStub([("log_measurement", {"quantity": "output voltage", "value": 3.3, "unit": "V"}),
                       ("set_config", {"key": "R2", "value": "10k"})])
    agent = fast_agent(rt, {text: ("log_measurement", 0.98, 0.01)}, spark)
    agent.handle(text)
    agent.last_verify_thread.join(5)
    assert len(rt.memory.experiments.measurements(s["id"])) == 1
    said = [m["text"] for m in rt.hub.outbox if m["type"] == "say"]
    assert said == ["Also: Noted: R2 is 10k."]


def test_fast_watch_undone_when_spark_disagrees(rt):
    text = "keep an eye on the soldering iron"
    before = len(rt.safety.watches())
    agent = fast_agent(rt, {text: ("watch_bench", 0.99, 0.0)}, SparkStub([("inspect_scene", {})]))
    out = agent.handle(text)
    assert out["fast"] and out["trace"][0]["args"]["concept"] == "soldering iron"
    assert len(rt.safety.watches()) == before + 1
    agent.last_verify_thread.join(5)
    assert len(rt.safety.watches()) == before
    said = [m["text"] for m in rt.hub.outbox if m["type"] == "say"]
    assert said and said[0].startswith("Correction: I stopped watching the soldering iron.")


@pytest.mark.parametrize("text,args", [
    ("where did we leave off?", {}),
    ("let's continue the soldering headers job", {"name": "soldering headers"}),
])
def test_resume_session_by_words(rt, text, args):
    rt.memory.experiments.start("soldering headers")
    rt.memory.experiments.start("divider")
    rt.memory.experiments.end()
    agent = fast_agent(rt, {text: ("resume_session", 0.99, 0.0)})
    out = agent.handle(text)
    assert out["fast"] and out["trace"][0]["args"] == args


@pytest.mark.parametrize("text,fast", [("that's it, end the session", True),
                                       ("thanks, that's all for now", False)])
def test_end_session_only_when_named(rt, text, fast):
    rt.memory.experiments.start("divider")
    agent = fast_agent(rt, {text: ("end_session", 0.99, 0.0)})
    assert bool(agent.handle(text).get("fast")) is fast


@pytest.mark.parametrize("text,want", [
    ("I measured 3.3 volts on the output", ("output voltage", 3.3, "V")),
    ("current draw is 120 milliamps", ("current draw", 120.0, "mA")),
    ("resistance reads 9.8k", ("resistance", 9.8, "kΩ")),
    ("got 2.1 volts across the LED", ("led voltage", 2.1, "V")),
    ("the supply went from 5 V to 4.8 V", None),        # two numbers: Spark
    ("3.3", None),                                      # no unit: Spark
])
def test_parse_measurement(text, want):
    assert parse_measurement(text) == want


def test_prefetched_decision_is_reused_for_the_final_transcript(rt):
    rt.perception.process("static", scene([("hex key", .4, .4, .5, .5)]), ts=time.time() - 5)
    calls = []

    class Counting(StubLaya):
        def predict(self, text, questions):
            calls.append(text)
            return super().predict(text, questions)
    agent = fast_agent(rt, {})
    agent.system1.backend = Counting({"where is my hex key": ("where_is", 0.99, 0.01),
                                      "Where is my hex key?": ("where_is", 0.99, 0.01)})
    agent.system1.prefetch("where is my hex key")
    n = len(calls)
    out = agent.handle("Where is my hex key?")
    assert out["fast"] and len(calls) == n and agent.system1.prefetched
