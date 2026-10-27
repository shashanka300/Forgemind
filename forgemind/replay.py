"""Replay a recorded video through the real perception pipeline (PRD §15.1).

    forgemind replay bench.mp4 --camera static --fps 1

Frames go through the same path as live ones (change gate, segmenter,
embedder, registry, events), with the mock model so no API is called. The
report says how many objects memory believed in, which events fired, and
writes one contact-sheet row per entity (its sightings cropped from the saved
keyframes), so a person can count duplicates (one object split into several
entities) and merges (several objects in one entity).
"""
import json
import shutil
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

SHEET_CELL = 150
SHEET_PER_ROW = 6


def frames(path, fps):
    """(t_seconds, PIL image) sampled at `fps` from a video file."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"can't open {path}")
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = 1.0 / fps
    t = 0.0
    while t * native < n:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, f = cap.read()
        if not ok:
            break
        yield t, Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        t += step
    cap.release()


def entity_sheet(rt, out_path):
    """One row per entity: label, then crops of up to SHEET_PER_ROW sightings."""
    reg, kdir = rt.memory.registry, rt.memory.keyframes_dir
    ents = reg.list(limit=500)
    if not ents:
        return None
    w = 190 + SHEET_CELL * SHEET_PER_ROW
    sheet = Image.new("RGB", (w, SHEET_CELL * len(ents)), (25, 25, 25))
    d = ImageDraw.Draw(sheet)
    for r, e in enumerate(ents):
        y = r * SHEET_CELL
        d.text((6, y + 50), f"#{e['id']} {e['label']}", fill=(255, 230, 0))
        d.text((6, y + 68), f"{e['sightings']} sightings", fill=(200, 200, 200))
        rows = rt.memory.db.all(
            "SELECT keyframe, box FROM sightings WHERE entity_id=? AND keyframe IS NOT NULL"
            " ORDER BY ts", (e["id"],))
        if len(rows) > SHEET_PER_ROW:              # spread picks over the whole track
            k = len(rows) / SHEET_PER_ROW
            rows = [rows[int(i * k)] for i in range(SHEET_PER_ROW)]
        for c, s in enumerate(rows):
            p = kdir / s["keyframe"]
            if not p.exists() or not s["box"]:
                continue
            img = Image.open(p).convert("RGB")
            W, H = img.size
            x0, y0, x1, y1 = (float(v) for v in s["box"].split(","))
            crop = img.crop((int(x0 * W), int(y0 * H), max(int(x1 * W), int(x0 * W) + 2),
                             max(int(y1 * H), int(y0 * H) + 2)))
            crop.thumbnail((SHEET_CELL - 6, SHEET_CELL - 6))
            sheet.paste(crop, (190 + c * SHEET_CELL, y + 3))
    sheet.save(out_path, quality=88)
    return out_path


def run(settings, video, camera="static", fps=1.0, out_dir=None, log=print):
    from forgemind.providers.mock_llm import MockModel
    from forgemind.runtime import Runtime
    out_dir = Path(out_dir or settings.data_dir / "replay" / Path(video).stem)
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    settings.data_dir = out_dir / "data"
    rt = Runtime(settings, llm=MockModel())
    t0 = time.time() - 3600                       # pretend it was recorded an hour ago
    events, processed, skipped, ms = Counter(), 0, 0, []
    try:
        for t, img in frames(video, fps):
            res = rt.perception.process(camera, img, ts=t0 + t)
            if res.get("skipped"):
                skipped += 1
                continue
            processed += 1
            ms.append(res.get("ms", 0))
            for e in res["events"]:
                events[e["type"]] += 1
                log(f"  {t:6.1f}s {e['type']:12s} {e.get('label') or e.get('target')}")
        ents = rt.memory.registry.list(limit=500)
        by_concept = Counter(e["concept"] for e in ents)
        sheet = entity_sheet(rt, out_dir / "entities.jpg")
        report = {
            "video": str(video), "camera": camera, "fps": fps,
            "frames": {"processed": processed, "skipped_by_change_gate": skipped},
            "ms_per_frame": round(sum(ms) / len(ms)) if ms else None,
            "entities": len(ents), "entities_by_concept": dict(by_concept),
            "events": dict(events),
            "settings": {"embedder": settings.embedder, "segmenter": settings.segmenter,
                         "teach_min_sim": settings.teach_min_sim,
                         "match_min_sim": settings.match_min_sim,
                         "concepts": settings.concepts},
            "entity_list": [{"id": e["id"], "label": e["label"], "sightings": e["sightings"],
                             "zone": e["zone"], "region": e["region"]} for e in ents],
            "sheet": str(sheet) if sheet else None,
        }
        (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"{processed} frames processed ({skipped} skipped by the change gate), "
            f"~{report['ms_per_frame']} ms/frame; {len(ents)} entities "
            f"{dict(by_concept)}; events {dict(events)}")
        log(f"report: {out_dir / 'report.json'}  sheet: {sheet}")
        return report
    finally:
        rt.close()
