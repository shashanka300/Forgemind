"""Citation check on report summaries (PRD F-OUT-5, acceptance check #6)."""
import re

from forgemind import reports
from forgemind.providers.base import ChatResult


class Writer:
    """Fake worker model: returns queued drafts, records prompts."""
    name, supports_vision = "fake", False

    def __init__(self, *drafts):
        self.drafts, self.calls = list(drafts), []

    def model_for(self, role):
        return "fake-worker"

    def chat(self, messages, tools=None, model=None, json_mode=False):
        self.calls.append(messages)
        return ChatResult(content=self.drafts.pop(0))


def _session(rt):
    tb = rt.agent.tools
    tb.run("start_session", {"name": "divider"})
    tb.run("plan_steps", {"steps": ["wire the divider", "measure the output", "swap R2 for 22k"]})
    tb.run("log_step", {"text": "wired the divider"})
    tb.run("set_config", {"key": "R2", "value": "10k"})
    tb.run("log_measurement", {"quantity": "output voltage", "value": 3.3, "unit": "V"})
    sid = rt.memory.experiments.active()["id"]
    facts = reports.build_facts(rt.memory.experiments, sid)
    ev = {m["quantity"]: m["event"] for m in facts["measurements"]}
    cfg = {c["key"]: c["event"] for c in facts["config"]}
    return sid, facts, ev["output voltage"], cfg["R2"]


def test_check_accepts_cited_values_counts_and_step_numbers(rt):
    _, facts, e_v, e_r2 = _session(rt)
    text = (f"The output read 3.3 V [{e_v}] with R2 at 10k [{e_r2}]. "
            "2 of 3 steps are done. The swap of R2 for 22k is still pending.")
    sentences, problems = reports.check_citations(text, facts)
    # "22" in the last sentence has no citation and isn't a count -> flagged
    assert [i for i, _ in problems] == [2]
    ok = (f"The output read 3.3 V [{e_v}] with R2 at 10k [{e_r2}]. 2 of 3 steps are done.")
    assert reports.check_citations(ok, facts)[1] == []


def test_invented_value_is_sent_back_then_removed(rt):
    sid, facts, e_v, _ = _session(rt)
    bad = f"The output read 3.5 V [{e_v}]. Wiring went well."
    w = Writer(bad)                                         # the rewrite repeats the mistake
    narrative, note, problems = reports.checked_narrative(w, facts, bad)
    assert len(w.calls) == 1                               # exactly one rewrite request
    assert "states 3.5" in w.calls[0][-1]["content"]
    assert "3.5" not in narrative and narrative == "Wiring went well."
    assert note.startswith("_1 sentence removed") and problems


def test_rewrite_that_fixes_it_is_kept(rt):
    _, facts, e_v, _ = _session(rt)
    w = Writer(f"The output read 3.3 V [{e_v}].")
    narrative, note, problems = reports.checked_narrative(w, facts, "The output read 3.5 V [E999].")
    assert narrative == f"The output read 3.3 V [{e_v}]." and note is None and not problems


def test_report_end_to_end_contains_only_logged_numbers(rt):
    sid, facts, e_v, _ = _session(rt)
    w = Writer(f"Output voltage was 3.3 V [{e_v}], below the 5 V target.",   # 5 is invented
               f"Output voltage was 3.3 V [{e_v}]. The input voltage is missing.")
    out = reports.generate(rt.memory.experiments, w, sid, rt.memory.reports_dir)
    md = out["markdown"]
    assert "## Summary" in md and "5 V target" not in md and "input voltage is missing" in md
    summary = md.split("## Summary")[1].split("##")[0]
    logged = {float(x) for x in re.findall(r"\d+(?:\.\d+)?", "3.3 10 22 2 3 1")}
    for n in re.findall(r"(?<![\w.\[E])\d+(?:\.\d+)?", reports._CITE.sub("", summary)):
        assert float(n) in logged
