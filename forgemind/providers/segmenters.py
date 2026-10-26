"""Concept segmenters: text concepts in, boxes (and masks) out.

No model runs on the user's machine. The options are:

* ClipGridSegmenter  - stand-in that works today: CLIP zero-shot over a grid
  of tiles on the server's CPU. Coarse, tile-sized boxes; enough to wire up
  memory, zones, safety and the overlay until hosted SAM is up.
* RemoteSamSegmenter - SAM 3 / 3.1 on a hosted GPU (deploy/modal_sam.py on
  Modal, or any endpoint speaking the same small JSON protocol).
* MetaApiSegmenter   - placeholder in case SAM is served through the Meta
  Model API during the hackathon.
* Sam3Segmenter      - the SAM 3 wrapper that runs *inside* the hosted GPU
  worker. Not used on your machine.
"""
import base64
import io
import threading
import time

import numpy as np
from PIL import Image

from forgemind.providers.base import Detection

BACKGROUND_PROMPTS = ["a photo of an empty table", "a photo of a wall",
                      "a photo of a floor", "a blurry photo", "a photo of a room",
                      # the zone markers themselves
                      "a black and white printed QR code on paper", "a paper sign"]


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(dets, iou=0.5):
    out = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all(o.concept != d.concept or _iou(o.box, d.box) < iou for o in out):
            out.append(d)
    return out


class NullSegmenter:
    name = "none"

    def segment(self, image, concepts):
        return []


class ClipGridSegmenter:
    name = "clip_grid"

    def __init__(self, embedder, grids=(2, 3), min_prob=0.55, min_cos=0.20,
                 temperature=100.0):
        self.emb = embedder
        self.grids = grids
        self.min_prob = min_prob
        self.min_cos = min_cos
        self.t = temperature
        self._text_cache = {}
        self._lock = threading.Lock()

    def _text_matrix(self, prompts):
        key = tuple(prompts)
        with self._lock:
            if key not in self._text_cache:
                self._text_cache[key] = np.asarray(
                    [self.emb.embed_text(p) for p in prompts], dtype=np.float32)
            return self._text_cache[key]

    def segment(self, image, concepts):
        concepts = list(concepts)
        img = image.convert("RGB")
        w, h = img.size
        boxes, tiles = [], []
        for g in self.grids:
            for gy in range(g):
                for gx in range(g):
                    b = (gx / g, gy / g, (gx + 1) / g, (gy + 1) / g)
                    boxes.append(b)
                    tiles.append(img.crop((int(b[0] * w), int(b[1] * h),
                                           int(b[2] * w), int(b[3] * h))))
        T = self._text_matrix([f"a photo of a {c}" for c in concepts] + BACKGROUND_PROMPTS)
        V = np.asarray(self.emb.embed_images(tiles), dtype=np.float32)
        cos = V @ T.T                                  # tiles x labels
        logits = self.t * cos
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        dets = []
        n = len(concepts)
        for i, b in enumerate(boxes):
            j = int(np.argmax(probs[i]))
            if j < n and probs[i, j] >= self.min_prob and cos[i, j] >= self.min_cos:
                dets.append(Detection(concept=concepts[j], box=b, score=float(probs[i, j])))
        return nms(dets, iou=0.3)


class Sam3Segmenter:
    """SAM 3 / 3.1 concept segmentation. Runs inside the hosted GPU worker
    (deploy/modal_sam.py), which installs torch + the sam3 package and pulls
    the gated checkpoint with a Hugging Face token.

    TODO(day 1): confirm the import paths and output keys against the
    version the worker installs; SAM 3.1 may ship a different builder name.
    """
    name = "sam3"

    def __init__(self, checkpoint="", device="cuda", min_score=0.5):
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as e:  # pragma: no cover - needs the GPU stack
            raise RuntimeError(
                "FM_SEGMENTER=sam3 but the sam3 package is not installed. "
                "See docs/META_INTEGRATION.md, or set FM_SEGMENTER=clip_grid.") from e
        kw = {"checkpoint_path": checkpoint} if checkpoint else {}
        self.model = build_sam3_image_model(**kw)
        if hasattr(self.model, "to"):
            self.model = self.model.to(device)
        self.processor = Sam3Processor(self.model)
        self.min_score = min_score
        self._lock = threading.Lock()

    def segment(self, image, concepts):  # pragma: no cover - needs the GPU stack
        img = image.convert("RGB")
        w, h = img.size
        dets = []
        with self._lock:
            state = self.processor.set_image(img)
            for concept in concepts:
                out = self.processor.set_text_prompt(state=state, prompt=concept)
                masks, boxes, scores = out["masks"], out["boxes"], out["scores"]
                for k in range(len(scores)):
                    s = float(scores[k])
                    if s < self.min_score:
                        continue
                    x0, y0, x1, y1 = [float(v) for v in boxes[k]]   # pixel xyxy
                    m = masks[k]
                    m = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
                    dets.append(Detection(concept=concept, score=s, mask=m.squeeze() > 0.5,
                                          box=(x0 / w, y0 / h, x1 / w, y1 / h)))
        return nms(dets)


class RemoteSamSegmenter:
    """Client for a hosted SAM endpoint.

    Protocol (what deploy/modal_sam.py serves):
        POST <FM_SAM_URL>   Authorization: Bearer <FM_SAM_TOKEN>
        {"image": "<base64 JPEG>", "concepts": [...], "min_score": 0.5}
        -> {"detections": [{"concept", "box": [x0,y0,x1,y1] (0..1), "score"}], "ms"}

    On a network/endpoint error it returns no detections (memory, chat and
    sessions keep working) and reports the error in `last_error`.
    """
    name = "remote_sam"

    def __init__(self, url, token="", min_score=0.5, timeout=20.0, max_px=1024,
                 http_client=None):
        import httpx
        if not url:
            raise ValueError("FM_SEGMENTER=remote_sam needs FM_SAM_URL in .env")
        self.url, self.min_score, self.max_px = url, min_score, max_px
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.client = http_client or httpx.Client(timeout=timeout)
        self.headers = headers
        self.last_error = None
        self.last_ms = None
        self._last_log = 0.0

    def segment(self, image, concepts):
        img = image.convert("RGB").copy()
        img.thumbnail((self.max_px, self.max_px))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        body = {"image": base64.b64encode(buf.getvalue()).decode(),
                "concepts": list(concepts), "min_score": self.min_score}
        t0 = time.perf_counter()
        try:
            r = self.client.post(self.url, json=body, headers=self.headers)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            if time.time() - self._last_log > 30:
                self._last_log = time.time()
                print(f"remote SAM unavailable ({self.last_error}); returning no detections")
            return []
        self.last_error = None
        self.last_ms = round((time.perf_counter() - t0) * 1000)
        dets = [Detection(concept=d["concept"], box=tuple(float(v) for v in d["box"]),
                          score=float(d["score"]))
                for d in data.get("detections", []) if d.get("concept") in concepts]
        return nms(dets)


class MetaApiSegmenter:
    """Placeholder: SAM served through the Meta Model API.

    TODO(when access arrives): if the hackathon exposes SAM 3.1 as an API,
    send the frame + concept list here and map the response to Detection.
    """
    name = "meta_sam"

    def __init__(self, settings):
        self.settings = settings

    def segment(self, image, concepts):
        raise NotImplementedError(
            "SAM via Meta Model API is not wired yet - see docs/META_INTEGRATION.md. "
            "Use FM_SEGMENTER=remote_sam (hosted) or clip_grid meanwhile.")
