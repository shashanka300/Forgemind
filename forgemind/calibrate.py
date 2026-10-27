"""Embedder crop test and threshold calibration (PRD Day 3, F-OBJ-2/3).

Photos go in one folder per real object; the file name says which camera:

    photos/
      hex_key-2.5mm/     phone_1.jpg phone_2.jpg bench_1.jpg bench_2.jpg
      hex_key-3mm/       ...            <- look-alikes are the point
      screwdriver-red/   ...
      multimeter/        ...

The folder name before "-" (underscores as spaces) is the concept SAM is
asked for when cropping ("hex key"). Phone photos play the teaching crops,
bench photos the later sightings, as in the app.

For each embedder it reports:
  * identification: each bench crop vs the phone crops of every object, best
    match wins if its similarity clears the threshold (the registry's taught
    rule); precision/recall over thresholds, and the lowest threshold with
    precision >= target -> FM_TEACH_MIN_SIM
  * same-concept pairs (hard negatives: two screwdrivers): the threshold that
    keeps look-alikes apart at the same precision -> FM_MATCH_MIN_SIM
  * AUC of same-object vs different-object pairs
"""
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def concept_of(folder_name):
    return folder_name.split("-")[0].replace("_", " ").strip()


def camera_of(file_name):
    n = file_name.lower()
    return "bench" if n.startswith(("bench", "static")) else \
        "phone" if n.startswith(("phone", "mobile")) else "unknown"


def load_photos(root):
    items = []
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMAGE_EXT:
                items.append({"object": d.name, "concept": concept_of(d.name),
                              "camera": camera_of(f.name), "path": str(f)})
    return items


def crop_items(items, segmenter=None, pad=0.05):
    """Crop each photo to its object with the segmenter; full image if not found."""
    from forgemind.perception.pipeline import Perception
    for it in items:
        img = ImageOps.exif_transpose(Image.open(it["path"])).convert("RGB")
        it["cropped"] = False
        if segmenter is not None:
            dets = segmenter.segment(img, [it["concept"]])
            if dets:
                d = max(dets, key=lambda d: d.score)
                img = Perception.crop(img, d.box, pad=pad)
                it["cropped"], it["seg_score"] = True, round(d.score, 3)
        it["image"] = img
    return items


class TimmEmbedder:
    """Image-only embedders for the comparison (DINOv2 / DINOv3)."""

    def __init__(self, name, model_id):
        import timm
        import torch
        self.name = name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = timm.create_model(model_id, pretrained=True, num_classes=0).eval().to(self.device)
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.tf = timm.data.create_transform(**cfg)

    def embed_images(self, images):
        import torch
        x = torch.stack([self.tf(i.convert("RGB")) for i in images]).to(self.device)
        with torch.no_grad():
            v = self.model(x).float().cpu().numpy()
        return [r / (np.linalg.norm(r) or 1.0) for r in v]


EMBEDDERS = {
    "clip": lambda s: __import__("forgemind.providers.embedders", fromlist=["x"]).ClipEmbedder(
        cache_dir=str(s.data_dir / "models")),
    "pe": lambda s: __import__("forgemind.providers.embedders", fromlist=["x"]).PerceptionEncoderEmbedder(
        cache_dir=str(s.data_dir / "models")),
    "dinov2": lambda s: TimmEmbedder("dinov2", "vit_base_patch14_reg4_dinov2.lvd142m"),
    "dinov3": lambda s: TimmEmbedder("dinov3", "vit_base_patch16_dinov3.lvd1689m"),
}


def auc(pos, neg):
    """P(random positive > random negative), ties count half."""
    if not pos or not neg:
        return None
    pos, neg = np.asarray(pos), np.asarray(neg)
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):                      # average ranks of ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def identification(vecs, items, query_cam="bench", ref_cam="phone"):
    """(score, correct) for each query crop's best match among other objects' refs."""
    refs = [i for i, it in enumerate(items) if it["camera"] == ref_cam]
    queries = [i for i, it in enumerate(items) if it["camera"] == query_cam]
    if not refs or not queries:      # no camera labels: leave-one-out over everything
        refs = queries = list(range(len(items)))
    out = []
    for q in queries:
        best_obj, best = None, -1.0
        for r in refs:
            if r == q:
                continue
            s = float(vecs[q] @ vecs[r])
            if s > best:
                best, best_obj = s, items[r]["object"]
        if best_obj is not None:
            out.append((best, best_obj == items[q]["object"]))
    return out


def threshold_for_precision(results, target=0.90):
    """Lowest threshold whose accepted matches reach `target` precision."""
    if not results:
        return None
    for t in sorted({round(s, 3) for s, _ in results}):
        acc = [ok for s, ok in results if s >= t]
        if acc and sum(acc) / len(acc) >= target:
            return {"threshold": t, "precision": round(sum(acc) / len(acc), 3),
                    "recall": round(sum(acc) / len(results), 3), "accepted": len(acc)}
    return None


def pair_threshold(pos, neg, target=0.90):
    """Lowest threshold with precision >= target on pairs (pos = same object)."""
    for t in sorted({round(v, 3) for v in pos + neg}):
        tp = sum(v >= t for v in pos)
        fp = sum(v >= t for v in neg)
        if tp and tp / (tp + fp) >= target:
            return {"threshold": t, "precision": round(tp / (tp + fp), 3),
                    "recall": round(tp / len(pos), 3)}
    return None


def evaluate(items, embedder, target=0.90):
    embedder.embed_images([items[0]["image"]])          # load the model before timing
    t0 = time.perf_counter()
    vecs = [np.asarray(v, np.float32) for v in embedder.embed_images([it["image"] for it in items])]
    ms = (time.perf_counter() - t0) * 1000 / max(1, len(items))
    pos, neg, hard = [], [], []
    for i, j in combinations(range(len(items)), 2):
        s = float(vecs[i] @ vecs[j])
        a, b = items[i], items[j]
        if a["object"] == b["object"]:
            pos.append(s)
        else:
            neg.append(s)
            if a["concept"] == b["concept"]:
                hard.append(s)
    ident = identification(vecs, items)
    return {
        "embedder": embedder.name, "ms_per_crop": round(ms, 1),
        "pairs": {"same_object": len(pos), "different": len(neg), "look_alike": len(hard)},
        "mean_sim": {"same_object": _mean(pos), "different": _mean(neg), "look_alike": _mean(hard)},
        "auc": _r(auc(pos, neg)), "auc_look_alike": _r(auc(pos, hard)),
        "identification": {
            "queries": len(ident),
            "top1_accuracy": _r(sum(ok for _, ok in ident) / len(ident)) if ident else None,
            "at_target": threshold_for_precision(ident, target),
        },
        "look_alike_split": pair_threshold(pos, hard, target) if hard else None,
    }


def recommend(result):
    t = (result["identification"]["at_target"] or {}).get("threshold")
    m = (result["look_alike_split"] or {}).get("threshold")
    return {"FM_TEACH_MIN_SIM": t, "FM_MATCH_MIN_SIM": m if m is not None else t}


TEXT_CAPABLE = {"clip", "pe"}       # can also embed text (where_is by description)
IMAGE_ONLY_MARGIN = 0.10             # recall an image-only model must win by


def pick_embedder(results):
    """Best embedder: recall at the target precision, then look-alike AUC.

    Text-capable models are preferred: an image-only model (DINO) would need a
    second vector per crop for text search, so it must win by a clear margin.
    """
    def key(r):
        return ((r["identification"]["at_target"] or {}).get("recall", 0),
                r["auc_look_alike"] or 0)
    text = [r for r in results if r["embedder"] in TEXT_CAPABLE]
    best = max(results, key=key)
    if text and best["embedder"] not in TEXT_CAPABLE:
        best_text = max(text, key=key)
        if key(best)[0] - key(best_text)[0] <= IMAGE_ONLY_MARGIN:
            return best_text
    return best


def _mean(xs):
    return round(float(np.mean(xs)), 3) if xs else None


def _r(x):
    return None if x is None else round(x, 3)


def run(settings, photos, embedders=("clip", "pe", "dinov2"), target=0.90, crop=True,
        segmenter=None, out=None, log=print):
    items = load_photos(photos)
    if len(items) < 4 or len({it["object"] for it in items}) < 2:
        raise SystemExit(f"need photos of at least 2 objects in {photos} (found {len(items)})")
    if crop and segmenter is None:
        from forgemind.providers import build_segmenter
        segmenter = build_segmenter(settings, None)
    crop_items(items, segmenter if crop else None)
    n_cropped = sum(it["cropped"] for it in items)
    log(f"{len(items)} photos of {len({it['object'] for it in items})} objects, "
        f"{n_cropped} cropped by the segmenter")
    report = {"photos": str(photos), "target_precision": target, "results": [],
              "uncropped": [it["path"] for it in items if not it["cropped"]]}
    for name in embedders:
        res = evaluate(items, EMBEDDERS[name](settings), target)
        res["recommend"] = recommend(res)
        report["results"].append(res)
        idn = res["identification"]
        at = idn["at_target"] or {}
        log(f"{name:7s} auc {res['auc']}  look-alike auc {res['auc_look_alike']}  "
            f"top-1 {idn['top1_accuracy']}  @P>={target}: thr {at.get('threshold')} "
            f"recall {at.get('recall')}  ({res['ms_per_crop']} ms/crop)")
    if out:
        Path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"report: {out}")
    return report
