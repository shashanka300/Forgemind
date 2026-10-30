"""Second data round: phrasings for the decisions the first model confused.

    python eval/gen_confusions.py      # appends to eval/decisions_train.jsonl

Each case pairs a label with the look-alike it was confused with (eval/system1_eval.py on
the hand-written turns), so the generated phrasings sit right on the boundary.
"""
import concurrent.futures as cf
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import gen_decisions as g
from laya_zero_shot_data import held_out

CASES = [
    ("highlight", "asks to show, mark or highlight an object on the phone's screen or live view "
                  "(the object is already in view; the user is NOT asking you to go and look somewhere)"),
    ("go_look_at", "asks the assistant to go look at, check or inspect a PLACE (shelf, drawer, left of the bench) "
                   "with the phone camera, not to highlight a specific object on screen"),
    ("set_config", "states which component value or part is used, in the form '<part> is <value>' or "
                   "'<part> = <value>' (e.g. 'R3 is 2.2k', 'C3 is 100 nF', 'the regulator is a 7805'): a design "
                   "choice, not something they measured"),
    ("log_measurement", "reports a reading they just took with a meter, using words like measured, reads, got, "
                        "shows, came out at (e.g. 'R3 measures 2.18k', 'I got 1.9 volts across C2')"),
    ("watch_bench", "asks to be warned or told if something happens on the bench, such as smoke, the iron being "
                    "left on, or something overheating ('tell me if anything starts smoking', 'shout if the board overheats')"),
    ("inspect_scene", "asks what is on the bench or what the camera sees right now (a one-off look, not a request "
                      "to keep watching)"),
    ("resume_session", "asks what they were doing, which step they are on, or what to do next in their work "
                       "session ('which step was I up to', 'remind me where I stopped', 'what comes after this')"),
    ("log_step", "reports a step they just finished ('done with the pads', 'flashed it', 'replaced the cap'), "
                 "not asking which step they are on"),
    ("read_closeup", "asks the assistant to read a meter or display for them right now ('what's the DMM showing', "
                     "'tell me the number on the display')"),
    ("chat", "asks a general question about earlier results or electronics that needs an explanation, such as "
             "'what did the LED voltage come to earlier?' or 'why is the reading so low?'"),
]


def ask_case(label, definition, style, seed):
    return g.ask(label, style, 20, seed, meaning=(definition, "be precise about this meaning"))


def main():
    path = os.path.join(os.path.dirname(__file__), "decisions_train.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    seen = {g.norm(r["text"]) for r in rows}
    test = {g.norm(u) for u in held_out()}
    jobs = [(lab, d, st, 2000 + i) for i, ((lab, d), st) in
            enumerate((c, s) for c in CASES for s in g.STYLES[:4])]
    added = {}
    with cf.ThreadPoolExecutor(4) as ex:
        for (lab, *_), got in zip(jobs, ex.map(lambda j: ask_case(*j), jobs)):
            for u in got:
                k = g.norm(u)
                if k and k not in seen and k not in test:
                    seen.add(k)
                    rows.append({"text": u, "label": lab, "compound": False, "round": 2})
                    added[lab] = added.get(lab, 0) + 1
            print(lab, "+", len(got), flush=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print("added", added, "total", len(rows))


if __name__ == "__main__":
    main()
