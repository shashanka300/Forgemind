"""Replay saved phone frames: teach objects when the user really did, see what gets labelled.

    python eval/replay_teach.py forgemind-data.bak-phonetest-20261031 --names clippers,switch

No cameras, no API: real SAM and Perception Encoder on the stored keyframes, in a throwaway
memory. For each taught name it reports the frames after teaching that show the label, and
saves a contact sheet of them to look at (a label on the wrong thing is only visible there).
"""
import argparse
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
from PIL import Image, ImageDraw

from forgemind.config import Settings
from forgemind.providers import build_embedder
from forgemind.runtime import Runtime


def frames_of(root):
    k = Path(root) / "keyframes"
    out = sorted((int(f.stem.split("-")[1]) / 1000, f) for f in k.glob("mobile-*.jpg"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--names", default="")
    ap.add_argument("--out", default="replay_teach.jpg")
    ap.add_argument("--after", type=float, default=40, help="seconds of frames to check after teaching")
    a = ap.parse_args()
    db = sqlite3.connect(f"file:{a.root}/forgemind.sqlite?mode=ro", uri=True)
    want = [n.strip().lower() for n in a.names.split(",") if n.strip()]
    teach, seen = [], set()
    for ts, p in db.execute("SELECT ts,payload FROM events WHERE type='taught' ORDER BY ts"):
        n = json.loads(p)["name"]
        if (not want or n.lower() in want) and n.lower() not in seen:
            seen.add(n.lower())
            teach.append((ts, n))
    real = Settings.from_env(token="off")
    tmp = tempfile.mkdtemp(prefix="fm-replay-")
    s = Settings.from_env(token="off", data_dir=tmp, llm_provider="mock", meta_api_key="", system1="",
                          tts="", vector_backend="numpy", confirm_new=True)
    rt = Runtime(s, embedder=build_embedder(real), warm=False)
    all_frames = frames_of(a.root)
    if not teach:
        raise SystemExit("nothing to teach")
    start, end = teach[0][0] - 12, teach[-1][0] + a.after
    frames = [(t, f) for t, f in all_frames if start <= t <= end]
    print(f"{len(frames)} frames, teaching {[n for _, n in teach]}")
    pending, hits, ms, sheet = list(teach), {}, [], []
    for t, f in frames:
        while pending and pending[0][0] <= t:
            _, name = pending.pop(0)
            rt.agent.tools.turn_frames = rt.perception.recent_frames(6.0, now=t)   # as a live turn does
            out = rt.agent.tools.run("teach_object", {"name": name})
            rt.agent.tools.turn_frames = None
            box = rt.memory.registry.find(name)[0]["box"] if not out.get("error") else None
            print(f"teach {name!r}: {out.get('error') or out['summary']} views={out.get('views')}")
            hits[name] = {"t": t, "frames": 0, "shown": 0, "box": box}
            if box:
                sheet.append((f"TAUGHT {name}", f, [(box, name)]))
        t0 = time.perf_counter()
        r = rt.perception.process("mobile", Image.open(f).convert("RGB"), ts=t)
        ms.append((time.perf_counter() - t0) * 1000)
        labels = [(d["box"], d.get("label") or d["concept"]) for d in r.get("detections", []) if d.get("kind") == "object"]
        for name, h in hits.items():
            if t - h["t"] <= a.after:
                h["frames"] += 1
                if any(l == name for _, l in labels):
                    h["shown"] += 1
                    if h["shown"] <= 2:
                        sheet.append((f"{name} +{t - h['t']:.0f}s", f, [x for x in labels if x[1] == name]))
    print(f"frame time median {statistics.median(ms):.0f} ms, p90 {sorted(ms)[int(.9 * len(ms))]:.0f} ms")
    for name, h in hits.items():
        print(f"  {name}: labelled in {h['shown']}/{h['frames']} frames in the next {a.after:.0f} s")
    ents = rt.memory.registry.list()
    print(len(ents), "objects:", sorted(v["label"] for v in ents))
    W, H, cols = 240, 320, 6
    rows = (len(sheet) + cols - 1) // cols or 1
    o = Image.new("RGB", (W * cols, H * rows), (20, 20, 20))
    for i, (title, f, boxes) in enumerate(sheet):
        im = Image.open(f).convert("RGB").resize((W, H))
        d = ImageDraw.Draw(im)
        for b, l in boxes:
            d.rectangle((b[0] * W, b[1] * H, b[2] * W, b[3] * H), outline=(80, 255, 120), width=3)
        d.rectangle((0, 0, W, 14), fill=(0, 0, 0))
        d.text((3, 1), title, fill=(255, 255, 255))
        o.paste(im, ((i % cols) * W, (i // cols) * H))
    o.save(a.out, quality=85)
    print("sheet:", a.out)
    rt.close()


if __name__ == "__main__":
    main()
