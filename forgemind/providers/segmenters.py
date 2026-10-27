"""Concept segmenters: text concepts in, boxes (and masks) out.

The options are:

* ClipGridSegmenter  - stand-in that works today: CLIP zero-shot over a grid
  of tiles on the server's CPU. Coarse, tile-sized boxes; enough to wire up
  memory, zones, safety and the overlay until hosted SAM is up.
* Sam3Segmenter      - SAM 3 / 3.1 open weights on the local GPU (continuous
  perception); also what the Modal worker runs.
* MetaApiSegmenter   - SAM 3.1 through the Meta Model API, one billed image
  per concept: for on-demand checks, not per frame.
* RemoteSamSegmenter - fallback: SAM on a hosted GPU (deploy/modal_sam.py on
  Modal, or any endpoint speaking the same small JSON protocol).
"""
import base64
import io
import re
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
    """SAM 3 / 3.1 concept segmentation in-process: on the local GPU
    (FM_SEGMENTER=sam3), or inside the hosted Modal worker (deploy/modal_sam.py).
    Needs torch + the sam3 package and the gated checkpoint (Hugging Face login).

    `checkpoint` is "sam3.1" (default), "sam3", or a path to a .pt file. The
    per-image model loads the *detector* half of the checkpoint; for "sam3.1"
    that is the retrained 3.1 detector inside sam3.1_multiplex.pt (its video
    tracker is not used here). The sam3 package then reports four missing
    `vision_backbone.convs.3` keys: that neck level is computed but dropped by
    the image model (scalp=1), so the warning is harmless.

    The model expects bfloat16 autocast on CUDA, as in Meta's examples;
    without it the first linear layer fails on a dtype mismatch.
    """
    name = "sam3"
    HF_VERSIONS = ("sam3.1", "sam3")

    def __init__(self, checkpoint="sam3.1", device="cuda", min_score=0.5):
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as e:  # pragma: no cover - needs the GPU stack
            raise RuntimeError(
                "FM_SEGMENTER=sam3 but the sam3 package is not installed. "
                "See docs/META_INTEGRATION.md, or set FM_SEGMENTER=clip_grid.") from e
        checkpoint = checkpoint or "sam3.1"
        if checkpoint in self.HF_VERSIONS:
            from sam3.model_builder import download_ckpt_from_hf
            checkpoint = download_ckpt_from_hf(version=checkpoint)
        self.checkpoint = checkpoint
        self.model = build_sam3_image_model(checkpoint_path=checkpoint, load_from_HF=False)
        if hasattr(self.model, "to"):
            self.model = self.model.to(device)
        self.processor = Sam3Processor(self.model)
        self.device = device
        self.last_error = None
        self.min_score = min_score
        self._lock = threading.Lock()

    def segment(self, image, concepts):  # pragma: no cover - needs the GPU stack
        """All concepts in one batched grounding pass over the shared image features.

        SAM 3's grounding takes (image id, text id) query pairs, so N prompts on one
        image run as a batch: ~170 ms for 14 concepts on an RTX 5090 instead of
        ~575 ms prompt by prompt, with the same detections. Boxes only (no masks:
        nothing downstream stores them). Falls back to the per-prompt path.
        """
        import contextlib

        import torch
        concepts = list(concepts)
        if not concepts:
            return []
        amp = (torch.autocast("cuda", dtype=torch.bfloat16) if str(self.device).startswith("cuda")
               else contextlib.nullcontext())
        with self._lock, torch.inference_mode(), amp:
            try:
                return nms(self._batched(image.convert("RGB"), concepts))
            except Exception as e:                       # keep working on API changes
                self.last_error = f"batched grounding failed, per-prompt fallback: {e}"
                return nms(self._sequential(image.convert("RGB"), concepts))

    def _batched(self, img, concepts):
        import torch
        from sam3.model import box_ops
        from sam3.model.data_misc import FindStage
        proc, model, n = self.processor, self.model, len(concepts)
        state = proc.set_image(img)
        state["backbone_out"].update(model.backbone.forward_text(concepts, device=proc.device))
        find = FindStage(img_ids=torch.zeros(n, device=proc.device, dtype=torch.long),
                         text_ids=torch.arange(n, device=proc.device, dtype=torch.long),
                         input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
                         input_points=None, input_points_mask=None)
        out = model.forward_grounding(backbone_out=state["backbone_out"], find_input=find,
                                      geometric_prompt=model._get_dummy_prompt(num_prompts=n),
                                      find_target=None)
        probs = (out["pred_logits"].sigmoid()
                 * out["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1).float()
        boxes = box_ops.box_cxcywh_to_xyxy(out["pred_boxes"]).float().clamp(0, 1)
        dets = []
        for i, concept in enumerate(concepts):
            keep = probs[i] >= self.min_score
            for b, sc in zip(boxes[i][keep].tolist(), probs[i][keep].tolist()):
                dets.append(Detection(concept=concept, score=float(sc), box=tuple(b)))
        return dets

    def _sequential(self, img, concepts):
        w, h = img.size
        dets = []
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
                m = (m.detach().float().cpu().numpy() if hasattr(m, "detach")
                     else np.asarray(m))
                dets.append(Detection(concept=concept, score=s, mask=m.squeeze() > 0.5,
                                      box=(x0 / w, y0 / h, x1 / w, y1 / h)))
        return dets


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
    """SAM 3.1 through the Meta Model API (`POST /v1/responses`, model sam-3.1).

    The API takes one noun phrase per request, so a frame costs one billed
    image per concept ($2.50 per 1k). Use it on demand (checks, close-ups),
    never per frame; continuous perception runs SAM 3.1 locally (`sam3`).

    The answer is special-token text, one line per image:
        <0f>0<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..;c=..|><|mask;...|>,1<|box;...|>...
    Boxes are pixel xyxy on the image we sent; `c` (0..1) is present because
    we ask for it with metadata.include_confidence. Masks are not decoded.
    """
    name = "meta_sam"
    model = "sam-3.1"

    def __init__(self, settings, http_client=None, timeout=30.0, max_px=1024, workers=4):
        import httpx
        if not settings.meta_api_key:
            raise ValueError("FM_SEGMENTER=meta_sam needs MODEL_API_KEY in .env")
        self.url = settings.meta_base_url.rstrip("/") + "/responses"
        self.headers = {"Authorization": f"Bearer {settings.meta_api_key}"}
        self.min_score = settings.sam_min_score
        self.client = http_client or httpx.Client(timeout=timeout)
        self.max_px, self.workers = max_px, workers
        self.last_error = None
        self.last_ms = None
        self.images_billed = 0

    def _request(self, data_url, concept):
        body = {"model": self.model, "metadata": {"include_confidence": "true"},
                "input": [{"role": "user", "content": [
                    {"type": "input_text", "text": concept},
                    {"type": "input_image", "image_url": data_url}]}]}
        r = self.client.post(self.url, json=body, headers=self.headers)
        r.raise_for_status()
        return output_text(r.json())

    def segment(self, image, concepts):
        from concurrent.futures import ThreadPoolExecutor
        img = image.convert("RGB").copy()
        img.thumbnail((self.max_px, self.max_px))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        data_url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        concepts = list(concepts)
        t0 = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                texts = list(pool.map(lambda c: self._request(data_url, c), concepts))
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return []
        self.last_error = None
        self.last_ms = round((time.perf_counter() - t0) * 1000)
        self.images_billed += len(concepts)
        dets = []
        for concept, text in zip(concepts, texts):
            for rec in parse_sam_text(text):
                score = rec["c"] if rec["c"] is not None else 1.0
                if score < self.min_score or not rec["w"] or not rec["h"]:
                    continue
                w, h = rec["w"], rec["h"]
                dets.append(Detection(concept=concept, score=score, track_id=rec["id"],
                                      box=(rec["x1"] / w, rec["y1"] / h,
                                           rec["x2"] / w, rec["y2"] / h)))
        return nms(dets)


def output_text(resp):
    """Text of a Responses API reply (the `output_text` shortcut or the message parts)."""
    if isinstance(resp.get("output_text"), str):
        return resp["output_text"]
    parts = []
    for item in resp.get("output") or []:
        for c in item.get("content") or []:
            if c.get("type") == "output_text":
                parts.append(c.get("text", ""))
    return "\n".join(parts)


_MASK = re.compile(r"<\|mask;.*?\|>")
_RECORD = re.compile(r"(?:^|,|<\d+f>)\s*(\d+)<\|box;([^|]*)\|>")


def parse_sam_text(text):
    """Box records from SAM 3.1 special-token text (all frames)."""
    out = []
    for line in (text or "").splitlines():
        line = _MASK.sub("", line)     # mask payloads are base85 and may contain commas
        m = re.match(r"\s*<(\d+)f>", line)
        frame = int(m.group(1)) if m else 0
        for rid, fields in _RECORD.findall(line):
            kv = dict(f.split("=", 1) for f in fields.split(";") if "=" in f)
            try:
                rec = {k: float(kv[k]) for k in ("x1", "y1", "x2", "y2", "w", "h")}
            except (KeyError, ValueError):
                continue
            rec.update(id=int(rid), frame=frame,
                       c=float(kv["c"]) if "c" in kv else None)
            out.append(rec)
    return out
