"""Can YOLOE find a taught object from one example image? Saved frames only, no cameras.

    python eval/yoloe_probe.py forgemind-data.bak-phonetest-20261031 out.jpg

Each object: an example box on the frame where it was taught, then frames after it, and frames from
when another object was being shown (where it should mostly stay quiet).
"""
import json
import os
import sqlite3
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_VERBOSE", "false")
import numpy as np
import torch
from PIL import Image, ImageDraw
from ultralytics import YOLOE

WEIGHTS = "F:/models/yoloe/yoloe-26l-seg.pt"
CONCEPTS = ["multimeter", "soldering iron", "screwdriver", "pliers", "circuit board", "breadboard", "wall charger",
            "copper wire", "headphones", "phone", "computer mouse", "person", "hand", "smoke"]
# example boxes (0..1) on the frame at teach time, checked by eye on the replay sheet
OBJECTS = {"clippers": (.28, .125, .70, .73), "timer": (.37, .366, .675, .59), "switch": (.46, .30, .69, .46)}
CONF = 0.25


def main():
    root, out = sys.argv[1], sys.argv[2]
    db = sqlite3.connect(f"file:{root}/forgemind.sqlite?mode=ro", uri=True)
    frames = sorted(Path(root, "keyframes").glob("mobile-*.jpg"), key=lambda f: int(f.stem.split("-")[1]))
    ft = [int(f.stem.split("-")[1]) / 1000 for f in frames]
    times = {}
    for ts, p in db.execute("SELECT ts,payload FROM events WHERE type='taught' ORDER BY ts"):
        times.setdefault(json.loads(p)["name"], ts)
    around = lambda t, a, b: [f for f, x in zip(frames, ft) if a <= x - t <= b]

    Path(WEIGHTS).parent.mkdir(parents=True, exist_ok=True)
    os.chdir(Path(WEIGHTS).parent)            # ultralytics downloads next to the working directory
    model = YOLOE(Path(WEIGHTS).name)
    model.to("cuda")

    # 1. text prompts need Ultralytics' CLIP package; skipped when it isn't installed
    try:
        import clip  # noqa: F401
        sample = [Image.open(f).convert("RGB") for f in around(times["clippers"], -4, 20)[:12]]
        model.set_classes(CONCEPTS, model.get_text_pe(CONCEPTS))
        model.predict(sample[0], verbose=False, conf=CONF)
        ms, n = [], []
        for im in sample:
            t = time.perf_counter()
            r = model.predict(im, verbose=False, conf=CONF, imgsz=640)[0]
            ms.append((time.perf_counter() - t) * 1000)
            n.append([CONCEPTS[int(c)] for c in r.boxes.cls.tolist()])
        print(f"text prompts: {statistics.median(ms):.0f} ms median; e.g. {n[0]} | {n[len(n) // 2]}")
    except ImportError:
        print("text prompts: skipped, no CLIP package")

    # 2. one example image -> other frames
    sheet = []
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
    for name, box in OBJECTS.items():
        t0 = times[name]
        ref_path = min(frames, key=lambda f: abs(int(f.stem.split("-")[1]) / 1000 - t0))
        ref = Image.open(ref_path).convert("RGB")
        W, H = ref.size
        prompts = {"bboxes": np.array([[box[0] * W, box[1] * H, box[2] * W, box[3] * H]]), "cls": np.array([0])}
        pos = around(t0, 1, 25)[::3][:6]
        neg = [f for other, t in times.items() if other in OBJECTS and other != name for f in around(t, 0, 8)[::4][:3]]
        for label, group in (("same object", pos), ("other object", neg)):
            hits, ms = 0, []
            for f in group:
                im = Image.open(f).convert("RGB")
                t = time.perf_counter()
                r = model.predict(im, refer_image=ref, visual_prompts=prompts, predictor=YOLOEVPSegPredictor,
                                  verbose=False, conf=CONF, imgsz=640)[0]
                ms.append((time.perf_counter() - t) * 1000)
                bs = r.boxes.xyxyn.tolist() if len(r.boxes) else []
                cs = r.boxes.conf.tolist() if len(r.boxes) else []
                hits += bool(bs)
                if len(sheet) < 30:
                    sheet.append((f"{name} / {label} {max(cs) if cs else 0:.2f} n={len(bs)}", f, bs))
            print(f"{name:9s} {label:13s}: found in {hits}/{len(group)} frames, {statistics.median(ms) if ms else 0:.0f} ms")
        model = YOLOE(Path(WEIGHTS).name)     # visual predict replaces the predictor
        model.to("cuda")
    print("GPU memory GB", round(torch.cuda.max_memory_allocated() / 1e9, 2))

    w, h, cols = 240, 320, 6
    o = Image.new("RGB", (w * cols, h * ((len(sheet) + cols - 1) // cols)), (20, 20, 20))
    for i, (title, f, bs) in enumerate(sheet):
        im = Image.open(f).convert("RGB").resize((w, h))
        d = ImageDraw.Draw(im)
        for b in bs:
            d.rectangle((b[0] * w, b[1] * h, b[2] * w, b[3] * h), outline=(80, 255, 120), width=3)
        d.rectangle((0, 0, w, 14), fill=(0, 0, 0))
        d.text((3, 1), title, fill=(255, 255, 255))
        o.paste(im, ((i % cols) * w, (i // cols) * h))
    o.save(out, quality=85)


if __name__ == "__main__":
    main()
