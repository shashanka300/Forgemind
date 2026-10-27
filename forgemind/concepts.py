"""Probe SAM concept phrasings on sample frames (tuning FM_CONCEPTS).

    forgemind concepts bench.jpg "walkthrough.mp4@24" --phrases "multimeter,digital multimeter,stopwatch"

For every phrase and frame it records the detections above the cutoff, prints
a table (count and best score per frame) and saves one annotated image per
frame with each phrase in its own colour, so a person can see which phrasing
finds the object and which ones fire on the wrong thing.
"""
import colorsys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_source(src):
    """'path.jpg' or 'video.mp4@seconds' -> (label, PIL image)."""
    if "@" in src and Path(src.rsplit("@", 1)[0]).suffix.lower() in (".mp4", ".mov", ".mkv", ".avi", ".webm"):
        path, t = src.rsplit("@", 1)
        import cv2
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ok, f = cap.read()
        if not ok:
            raise SystemExit(f"no frame at {t}s in {path}")
        return f"{Path(path).stem}@{t}", Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return Path(src).stem, Image.open(src).convert("RGB")


def _colours(n):
    return [tuple(int(255 * c) for c in colorsys.hsv_to_rgb(i / max(1, n), 0.9, 1.0)) for i in range(n)]


def probe(segmenter, sources, phrases, min_score=0.4, out_dir=None, log=print):
    out_dir = Path(out_dir) if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    colours = dict(zip(phrases, _colours(len(phrases))))
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:                                   # older Pillow
        font = ImageFont.load_default()
    table = {}
    for src in sources:
        label, img = load_source(src)
        W, H = img.size
        drawn = img.copy()
        d = ImageDraw.Draw(drawn)
        for ph in phrases:
            dets = [x for x in segmenter.segment(img, [ph]) if x.score >= min_score]
            table.setdefault(ph, {})[label] = (len(dets), max((x.score for x in dets), default=0.0))
            for x in dets:
                b = (x.box[0] * W, x.box[1] * H, x.box[2] * W, x.box[3] * H)
                d.rectangle(b, outline=colours[ph], width=3)
                d.text((b[0] + 3, b[1] + 2), f"{ph} {x.score:.2f}", fill=colours[ph], font=font)
        if out_dir:
            drawn.save(out_dir / f"{label}.jpg", quality=88)
    labels = [load_label for load_label in dict.fromkeys(
        lbl for ph in table for lbl in table[ph])]
    width = max(len(p) for p in phrases) + 2
    log(" " * width + "".join(f"{lbl[:14]:>16s}" for lbl in labels))
    for ph in phrases:
        cells = "".join(f"{n:>9d} x {s:.2f}" if n else f"{'-':>16s}"
                        for n, s in (table[ph].get(lbl, (0, 0.0)) for lbl in labels))
        log(f"{ph:<{width}s}{cells}")
    return table
