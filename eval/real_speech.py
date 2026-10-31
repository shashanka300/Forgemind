"""What the fast path does with real speech: the sentences said in a phone test.

    python eval/real_speech.py [eval/phone_test_oct31.txt]

No labels: it prints every sentence that would take the fast path, so a person can read
them. The memory holds a few objects so that where-is questions can be answered.
"""
import os
import sys
import tempfile
import types

os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from forgemind.agent.orchestrator import is_compound, is_fragment
from forgemind.agent.system1 import System1
from forgemind.config import Settings
from forgemind.memory import WorldMemory
from forgemind.providers.embedders import HashEmbedder


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "eval/phone_test_oct31.txt"
    lines = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]
    s = Settings.from_env(data_dir=tempfile.mkdtemp(prefix="fm-real-"), embedder="hash",
                          vector_backend="numpy", meta_api_key="")
    mem = WorldMemory(s, HashEmbedder())
    for name in ("baby's biter", "computer mouse", "keyboard", "headphones", "pliers"):
        mem.registry.teach(name, [types.SimpleNamespace()] if False else [_img(name)], "object")
    mem.experiments.start("demo")
    s1 = System1(s, mem)
    s1.warm_up()
    fast, frag, compound = [], 0, 0
    for t in lines:
        if is_fragment(t):
            frag += 1
            continue
        compound += is_compound(t)
        act = s1.decide(t)
        if act:
            fast.append((act.tool, act.args, round(act.conf, 3), t))
    print(f"{len(lines)} sentences: {frag} fragments ignored, {compound} compound, {len(fast)} on the fast path")
    for tool, args, conf, t in fast:
        print(f"  {tool:16s} {conf} {args} | {t}")


def _img(name):
    from PIL import Image
    return Image.new("RGB", (32, 32), tuple(hash(name + c) % 255 for c in "rgb"))


if __name__ == "__main__":
    main()
