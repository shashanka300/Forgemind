"""Sentences the user really said, on saved frames, with real Spark, SAM and Perception Encoder.

    python eval/real_turns.py forgemind-data.bak-phonetest-20261031 [out.jpg]

No cameras or mics; a throwaway memory; a few cents. Prints what it does and says, to read by eye:
the point is how it behaves on real speech, not a score.
"""
import json
import os
import sqlite3
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

# (frames to show: a taught object's name, then what is said)
SCRIPT = [
    ("clippers", "These are my clippers."),
    ("clippers", "what are you looking at?"),
    ("clippers", "where are my clippers?"),
    ("keyboard", "do you see my clippers?"),
    ("clippers", "did you forget about it?"),
    ("keyboard", "What do you see on my screen?"),
    ("keyboard", "my secondary desk."),
    ("keyboard", "remember that my generic books are on the first shelf and ML books on the second."),
    ("keyboard", "where do I keep my ML books?"),
    ("switch", "and going forward recognize it as a paper tape."),
    ("switch", "going"),
    ("switch", "So don't"),
    ("switch", "about it."),
    ("switch", "It's my scissor."),
    ("switch", "Do you remember the paper tape?"),
    ("switch", "Why are you not able to find it?"),
    ("timer", "I'm holding a timer in my hand."),
    ("timer", "that's not the timer, it's my stopwatch"),
    ("timer", "Do you remember what you're built for?"),
    ("timer", "reply on the phone only"),
]


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "forgemind-data.bak-phonetest-20261031"
    db = sqlite3.connect(f"file:{root}/forgemind.sqlite?mode=ro", uri=True)
    frames = sorted(Path(root, "keyframes").glob("mobile-*.jpg"), key=lambda f: int(f.stem.split("-")[1]))
    times = {}
    for ts, p in db.execute("SELECT ts,payload FROM events WHERE type='taught' ORDER BY ts"):
        times.setdefault(json.loads(p)["name"], ts)
    near = {n: [f for f in frames if -6 <= int(f.stem.split("-")[1]) / 1000 - t <= 2][-3:] for n, t in times.items()}

    real = Settings.from_env(token="off")
    s = Settings.from_env(token="off", data_dir=tempfile.mkdtemp(prefix="fm-real-"), system1=real.system1, tts="",
                          vector_backend="numpy", report_async=False, wake_word=False)
    rt = Runtime(s, embedder=build_embedder(real), warm=False)
    rt.agent.system1.warm_up()
    for shown, text in SCRIPT:
        for f in near[shown]:
            rt.perception.process("mobile", Image.open(f).convert("RGB"), force=True)
        t0 = time.perf_counter()
        r = rt.agent.handle(text, source="phone")
        if rt.agent.last_verify_thread:
            rt.agent.last_verify_thread.join(20)
        tools = [t["tool"] for t in r["trace"]]
        flag = "ignored" if r.get("ignored") else ("S1" if r.get("fast") else "")
        print(f"[{shown:8s}] {text}\n   {time.perf_counter() - t0:4.1f}s {flag:7s} frame={rt.agent._frame_sent!s:5s} {tools}\n"
              f"   -> {r['reply'][:170]}")
    print("notes:", [n["text"] for n in rt.memory.notes.recent(10)])
    print("taught:", [v["label"] for v in rt.memory.registry.list() if v["taught"]])
    rt.close()


if __name__ == "__main__":
    main()
