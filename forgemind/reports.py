"""Experiment report (PRD F-OUT-1).

The facts (configuration, steps, measurements, open items) are rendered
deterministically from memory, each with its event id. A model may add a
narrative summary, but only from those facts and citing them, so a report
never contains a number the system did not log.
"""
import json
import re
import time
from datetime import datetime

from forgemind.memory.registry import fmt_ago

NARRATIVE_PROMPT = """You write the summary section of a lab/workshop experiment report.
Use ONLY the facts in the JSON below. Cite each fact with its event id in
brackets, like [E12]. Do not invent values, causes or steps. If something
needed for a conclusion is missing, say it is missing. 4-8 sentences."""


def _t(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "-"


def _src(source):
    return {"stated": "you said", "instrument": "read from display", "observed": "camera",
            "inferred": "model", "carried": "carried over"}.get(source, source or "-")


def build_facts(experiments, sid):
    s = experiments.get(sid)
    return {
        "session": {"id": s["id"], "name": s["name"], "objective": s["objective"],
                    "started": _t(s["started_at"]), "ended": _t(s["ended_at"]),
                    "status": s["status"]},
        "config": [{"key": k, "value": v["value"], "event": f"E{v['event_id']}"}
                   for k, v in experiments.config(sid).items()],
        "steps": [{"text": st["text"], "status": st["status"], "source": st["source"],
                   "time": _t(st["ts"]), "event": f"E{st['event_id']}"}
                  for st in experiments.steps(sid)],
        "measurements": [{"quantity": m["quantity"], "value": m["value"], "unit": m["unit"],
                          "source": m["source"], "time": _t(m["ts"]), "note": m["note"],
                          "event": f"E{m['event_id']}"} for m in experiments.measurements(sid)],
        "comparison": experiments.compare(sid),
    }


# -- citation check (PRD F-OUT-5) ----------------------------------------------

_NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?")
_CITE = re.compile(r"\[E(\d+)\]")


def _numbers(text):
    return {float(n) for n in _NUM.findall(str(text or ""))}


def fact_index(facts):
    """E-id -> the numbers that fact contains (values, step text, times)."""
    idx = {}
    for group in ("config", "steps", "measurements"):
        for f in facts.get(group, []):
            nums = set()
            for k, v in f.items():
                if k != "event":
                    nums |= _numbers(v)
            idx[f["event"]] = nums
    return idx


def _derived(facts):
    """Counts a summary may state without a citation (\"2 of 3 steps\")."""
    steps = facts.get("steps", [])
    by = {}
    for st in steps:
        by[st["status"]] = by.get(st["status"], 0) + 1
    return {float(x) for x in (len(steps), len(facts.get("measurements", [])),
                               len(facts.get("config", [])), *by.values())}


def check_citations(text, facts):
    """Per sentence: cited ids must exist; every number must be in a cited fact.

    Returns (sentences, problems) where problems is a list of
    (sentence index, reason). Numbers that are only counts of steps,
    measurements or settings are allowed without a citation.
    """
    idx, derived = fact_index(facts), _derived(facts)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s]
    problems = []
    for i, s in enumerate(sentences):
        cites = [f"E{n}" for n in _CITE.findall(s)]
        unknown = [c for c in cites if c not in idx]
        if unknown:
            problems.append((i, f"cites {', '.join(unknown)}, which are not in the facts"))
            continue
        allowed = set().union(*(idx[c] for c in cites)) if cites else set()
        bare = _CITE.sub(" ", s)
        for n in sorted(_numbers(bare)):
            if n in allowed or (n in derived and n == int(n)):
                continue
            where = f"[{', '.join(cites)}]" if cites else "no citation"
            problems.append((i, f"states {n:g}, which is not in {where}"))
            break
    return sentences, problems


FIX_PROMPT = """Your summary failed the citation check:
{problems}
Rewrite the summary. Every number must come from the fact you cite next to it,
and only cite ids that exist in the JSON. If a value is missing, say it is missing."""


def checked_narrative(llm, facts, draft):
    """Check the draft; one rewrite if needed; then drop sentences that still fail."""
    sentences, problems = check_citations(draft, facts)
    if problems:
        listing = "\n".join(f"- \"{sentences[i]}\": {why}" for i, why in problems)
        res = llm.chat([{"role": "system", "content": NARRATIVE_PROMPT},
                        {"role": "user", "content": json.dumps(facts, default=str)},
                        {"role": "assistant", "content": draft},
                        {"role": "user", "content": FIX_PROMPT.format(problems=listing)}],
                       model=llm.model_for("worker"))
        draft = res.content or ""
        sentences, problems = check_citations(draft, facts)
    bad = {i for i, _ in problems}
    kept = " ".join(s for i, s in enumerate(sentences) if i not in bad)
    note = None
    if bad:
        note = (f"_{len(bad)} sentence{'s' * (len(bad) > 1)} removed from this summary: "
                "stated values that are not in the cited records._")
    return kept, note, [why for _, why in problems]


def render(facts, narrative=None, note=None):
    s = facts["session"]
    L = [f"# Experiment report: {s['name']}", "",
         f"*Generated {_t(time.time())} by ForgeMind. Started {s['started']}, "
         f"status: {s['status']}. Facts cite their memory event as [E#].*", ""]
    L += ["## Objective", "", s["objective"] or "_Not stated._", ""]
    if narrative:
        L += ["## Summary", "", narrative.strip(), ""]
    if note:
        L += [note, ""]
    L += ["## Configuration", ""]
    if facts["config"]:
        L += ["| Setting | Value | Ref |", "|---|---|---|"]
        L += [f"| {c['key']} | {c['value']} | [{c['event']}] |" for c in facts["config"]]
    else:
        L += ["_No configuration recorded._"]
    L += ["", "## Method", ""]
    done = [st for st in facts["steps"] if st["status"] == "done"]
    if done:
        L += [f"{i}. {st['text']} ({st['time']}, {_src(st['source'])}) [{st['event']}]"
              for i, st in enumerate(done, 1)]
    else:
        L += ["_No completed steps recorded._"]
    L += ["", "## Results", ""]
    if facts["measurements"]:
        L += ["| Quantity | Value | Unit | Source | Time | Ref |", "|---|---|---|---|---|---|"]
        for m in facts["measurements"]:
            val = "**missing**" if m["value"] is None else m["value"]
            L.append(f"| {m['quantity']} | {val} | {m['unit'] or ''} | {_src(m['source'])} "
                     f"| {m['time']} | [{m['event']}] |")
    else:
        L += ["_No measurements recorded._"]
    comp = facts.get("comparison")
    if comp and comp.get("previous"):
        L += ["", f"## Compared with \"{comp['previous']['name']}\" (session {comp['previous']['id']})", ""]
        if comp["config_changes"]:
            L += [f"- {c['key']}: {c['previous']} → {c['current']}" for c in comp["config_changes"]]
        for m in comp["measurements"]:
            if "delta" in m:
                L.append(f"- {m['quantity']}: {m['previous']} → {m['current']} {m['unit'] or ''} "
                         f"(Δ {m['delta']:+g})")
        if not comp["config_changes"] and not any("delta" in m for m in comp["measurements"]):
            L.append("_No comparable differences recorded._")
    L += ["", "## Deviations and open items", ""]
    pending = [st for st in facts["steps"] if st["status"] == "planned"]
    tentative = [st for st in facts["steps"] if st["status"] == "tentative"]
    missing = [m for m in facts["measurements"] if m["value"] is None]
    if pending:
        L += [f"- Not done: {st['text']} [{st['event']}]" for st in pending]
    if tentative:
        L += [f"- Tried but not committed: {st['text']} [{st['event']}]" for st in tentative]
    if missing:
        L += [f"- Missing value: {m['quantity']} [{m['event']}]" for m in missing]
    if not (pending or tentative or missing):
        L += ["_None recorded._"]
    return "\n".join(L) + "\n"


def generate(experiments, llm, sid, out_dir, use_model=True):
    facts = build_facts(experiments, sid)
    narrative, note, problems = None, None, []
    if use_model and getattr(llm, "name", "mock") != "mock":
        try:
            res = llm.chat([{"role": "system", "content": NARRATIVE_PROMPT},
                            {"role": "user", "content": json.dumps(facts, default=str)}],
                           model=llm.model_for("worker"))
            narrative, note, problems = checked_narrative(llm, facts, res.content or "")
        except Exception as e:  # the deterministic report still stands
            narrative = f"_Summary unavailable: {e}_"
    md = render(facts, narrative, note)
    slug = "".join(c if c.isalnum() else "-" for c in facts["session"]["name"].lower()).strip("-")
    name = f"{slug or 'session'}-{sid}.md"
    (out_dir / name).write_text(md)
    return {"path": str(out_dir / name), "url": f"/reports/{name}", "markdown": md,
            "facts": facts, "generated": fmt_ago(0), "citation_problems": problems}
