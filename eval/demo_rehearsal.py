"""The demo script end to end, on saved frames: real Spark, SAM and Perception Encoder.

    python eval/demo_rehearsal.py forgemind-data.bak-phonetest-20261031

No cameras or mics. Speech goes in as text through the same wake-word and turn path as voice.
Prints each turn's tools, time, prompt size and reply, in a throwaway memory; costs a few cents.
"""
import json
import os
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image

from forgemind.config import Settings
from forgemind.providers import build_embedder
from forgemind.runtime import Runtime

# (what is said, tools that are right; None = must be ignored)
SCRIPT = [
    ("Hey Forge, this is my clippers.", [{"teach_object"}]),
    ("so I think I'll check this later", None),
    ("Forge, where are my clippers?", [{"where_is"}]),
    ("Forge, what do you see?", [{"inspect_scene"}]),
    ("Forge, start a session called voltage divider test.", [{"start_session"}]),
    ("Forge, the steps are wire the divider, measure the output, swap R2.", [{"plan_steps"}]),
    ("Forge, I wired the divider.", [{"log_step"}]),
    ("Forge, I measured 3.3 volts on the output.", [{"log_measurement"}, {"log_measurement", "log_step"}]),
    ("Forge, watch the soldering iron while I'm away.", [{"watch_bench"}]),
    ("Forge, that's not the clippers, it's my pliers.", [{"correct_object"}]),
    ("Forge, reply on the phone only from now on.", [set()]),
    ("Forge, where did we leave off?", [{"resume_session"}]),
    ("Forge, write it up.", [{"generate_report"}]),
]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "forgemind-data.bak-phonetest-20261031"
    db = sqlite3.connect(f"file:{root}/forgemind.sqlite?mode=ro", uri=True)
    t_clip = next(ts for ts, p in db.execute("SELECT ts,payload FROM events WHERE type='taught' ORDER BY ts")
                  if json.loads(p)["name"] == "clippers")
    frames = sorted(Path(root, "keyframes").glob("mobile-*.jpg"))
    near = [f for f in frames if -6 <= int(f.stem.split("-")[1]) / 1000 - t_clip <= 2]

    real = Settings.from_env(token="off")
    s = Settings.from_env(token="off", data_dir=tempfile.mkdtemp(prefix="fm-demo-"), system1=real.system1,
                          tts="", vector_backend="numpy", confirm_new=True, wake_word=True, report_async=False)
    rt = Runtime(s, embedder=build_embedder(real), warm=False)
    rt.ctx.capture_timeout = 1.0
    if rt.agent.system1:
        rt.agent.system1.warm_up()
    show = lambda: [rt.perception.process("mobile", Image.open(f).convert("RGB"), force=True) for f in near[-4:]]
    rows, ok_n = [], 0
    for text, want in SCRIPT:
        show()                                  # the phone keeps looking at the object
        said = rt.agent.addressed(text)
        if said is None:
            ok = want is None
            print(f"{'OK ' if ok else 'XX '}  ignored | {text}")
            ok_n += ok
            continue
        if want is None:
            rt.agent._follow_until = 0        # still inside the follow-up window: try again
            ok = rt.agent.addressed(text) is None
            print(f"{'OK ' if ok else 'XX '}  in follow-up window, then ignored | {text}")
            ok_n += ok
            continue
        r = rt.agent.handle(said, source="phone")
        if rt.agent.last_verify_thread:
            rt.agent.last_verify_thread.join(20)
        tools = [t["tool"] for t in r["trace"]]
        ok = set(tools) in want
        ok_n += ok
        ev = [e for e in rt.memory.db.recent_events(30) if e["type"] == "turn"][0]["payload"]
        rows.append(ev)
        print(f"{'OK ' if ok else 'XX '} {r['ms']:5d} ms {'S1' if r.get('fast') else '  '} model={ev['model_ms']} "
              f"prompt={ev.get('prompt_tokens')} {tools} | {text}\n      -> {r['reply'][:140]}")
    spark = [e["model_ms"] for e in rows if e["model_ms"]]
    print(f"\n{ok_n}/{len(SCRIPT)} right; Spark median {statistics.median(spark)} ms, max {max(spark)} ms; "
          f"prompt tokens max {max(e.get('prompt_tokens') or 0 for e in rows)}")
    rt.close()


if __name__ == "__main__":
    main()
