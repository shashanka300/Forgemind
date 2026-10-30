"""Generate System 1 training phrasings with local Muse Glimmer (no API spend).

    python eval/gen_decisions.py --out eval/decisions_train.jsonl

Glimmer (llama.cpp, OpenAI-compatible on :8081) writes varied things a person at an
electronics bench would say for each action; near-misses that belong to another label are
spelled out so the boundaries are learned. The hand-written turns in laya_zero_shot.py are
the held-out test and are filtered out of the training data.
"""
import argparse
import concurrent.futures as cf
import json
import random
import re
import sys
import urllib.request

sys.path.insert(0, __file__.rsplit("eval", 1)[0]); sys.path.insert(0, __file__.rsplit("gen_decisions", 1)[0])

URL = "http://127.0.0.1:8081/v1/chat/completions"

LABELS = {
    "where_is": ("asks where an object is, was left, or went; says they can't find something",
                 "not: asking to see it highlighted on screen (highlight), asking what's on the bench in general (inspect_scene)"),
    "inspect_scene": ("asks what the bench camera or the phone sees right now, or what is on the bench at the moment",
                      "not: asking where one specific object is (where_is)"),
    "go_look_at": ("tells the assistant to go look at / check a place or object with the phone camera",
                   "not: asking where something is (where_is)"),
    "read_closeup": ("asks the assistant to read an instrument, meter, display, screen or label for them",
                     "not: the user telling a value they read themselves (log_measurement)"),
    "highlight": ("asks to highlight, mark, point out or show an object on the phone's live view",
                  "not: asking where it is (where_is)"),
    "watch_bench": ("asks to watch, monitor or keep an eye on something for safety, or to warn them if something happens (iron left on, smoke)",
                    "not: asking where something is"),
    "teach_object": ("introduces the object in front of the phone so it is remembered under a name ('this is my ...')",
                     "not: asking where it is"),
    "start_session": ("starts a new experiment, job or work session, often giving it a name",
                      "not: continuing an earlier one (resume_session)"),
    "resume_session": ("wants to continue an earlier session, or asks where they left off, what's next, or which step they're on",
                       "not: comparing with an earlier run (compare_sessions)"),
    "end_session": ("ends, closes, stops or finishes the current session",
                    "not: thanks or goodbye without mentioning the session (chat)"),
    "plan_steps": ("lists the steps they plan to do in this session, or adds planned steps",
                   "not: reporting a step already done (log_step)"),
    "log_step": ("reports a work step they just did or finished, or says to not log something they are only trying",
                 "not: a number they measured (log_measurement), a value they set (set_config)"),
    "log_measurement": ("reports a value they measured themselves: volts, amps, ohms, temperature, frequency",
                        "not: asking the assistant to read the meter (read_closeup), a value they chose or set (set_config)"),
    "set_config": ("states or sets a configuration: a component value, part choice, supply or iron setting they chose",
                   "not: a value they measured (log_measurement)"),
    "compare_sessions": ("asks to compare this run with an earlier run or session, or what changed since last time",
                         "not: continuing an earlier session (resume_session)"),
    "generate_report": ("asks to write up, document or make a report of the experiment",
                        "not: ending the session (end_session)"),
    "chat": ("small talk, greetings, thanks, or a general knowledge or electronics question that needs no action on their bench, sessions or objects",
             "not: anything asking the assistant to find, record, read, watch or start something"),
}

STYLES = [
    "short and casual, like spoken out loud while working",
    "voice transcript style: lowercase, no punctuation, filler words like um or okay",
    "polite full sentences",
    "terse, 2 to 5 words",
    "indirect, e.g. 'I wonder ...', 'any idea ...', 'could you ...'",
]
THINGS = ("hex key, allen key, multimeter, DMM, soldering iron, wire cutters, flush cutters, pliers, tweezers, "
          "breadboard, arduino, ESP32 board, scope probe, solder spool, flux pen, heat gun, screwdriver, "
          "magnet wire, 10k resistor, LED strip, bench supply, calipers, crimper, desolder pump")

def ask(label, style, n=25, seed=0, meaning=None):
    definition, near = meaning or LABELS[label]
    prompt = (f"You write test data for a voice assistant at an electronics workbench. Write {n} different things "
              f"a person might say that mean: the user {definition}. ({near}.)\n"
              f"Style: {style}. Vary wording, objects and numbers; objects can include: {THINGS}. "
              f"Each must be one utterance, no numbering. Reply with only a JSON array of strings.")
    body = {"model": "glimmer", "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.0, "seed": seed, "max_tokens": 3000}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        text = json.load(r)["choices"][0]["message"]["content"]
    m = re.search(r"\[.*\]", text, re.S)
    try:
        return [s.strip() for s in json.loads(m.group(0)) if isinstance(s, str) and s.strip()] if m else []
    except json.JSONDecodeError:
        return []

def ask_compound(style, n=25, seed=0):
    names = ", ".join(LABELS)
    prompt = (f"You write test data for a voice assistant at an electronics workbench. Its actions are: {names}. "
              f"Write {n} different single utterances in which the user asks for TWO different actions at once "
              f"(e.g. reports a measurement and asks for a report; ends the session and asks where the iron is). "
              f"Style: {style}. Reply with only a JSON array of strings.")
    body = {"model": "glimmer", "messages": [{"role": "user", "content": prompt}],
            "temperature": 1.0, "seed": seed, "max_tokens": 3000}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        text = json.load(r)["choices"][0]["message"]["content"]
    m = re.search(r"\[.*\]", text, re.S)
    try:
        return [s.strip() for s in json.loads(m.group(0)) if isinstance(s, str)] if m else []
    except json.JSONDecodeError:
        return []

def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/decisions_train.jsonl")
    ap.add_argument("--per-style", type=int, default=25)
    a = ap.parse_args()
    from laya_zero_shot_data import held_out
    test = {norm(u) for u in held_out()}
    jobs = [("single", lab, st, i) for i, (lab, st) in enumerate((l, s) for l in LABELS for s in STYLES)]
    jobs += [("compound", None, st, 1000 + i) for i, st in enumerate(STYLES * 2)]
    out, seen = [], set()
    with cf.ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(ask if k == "single" else ask_compound,
                          *((lab, st) if k == "single" else (st,)), a.per_style, seed): (k, lab)
                for k, lab, st, seed in jobs}
        for f in cf.as_completed(futs):
            kind, lab = futs[f]
            try:
                got = f.result()
            except Exception as e:                     # one bad call shouldn't sink the run
                print("failed:", kind, lab, e)
                continue
            for u in got:
                key = norm(u)
                if not key or key in seen or key in test:
                    continue
                seen.add(key)
                out.append({"text": u, "label": lab, "compound": kind == "compound"})
            print(f"{kind:8s} {lab or '-':16s} +{len(got):3d}  total {len(out)}", flush=True)
    random.Random(0).shuffle(out)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("wrote", len(out), "->", a.out)

if __name__ == "__main__":
    main()
