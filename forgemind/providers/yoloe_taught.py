"""Finding taught objects with YOLOE: the example box from teaching is the prompt.

SAM needs a word for an object, and comparing crops afterwards misses a lot. YOLOE turns the
example into an embedding once, then each frame checks all of them in one pass. Examples are kept
on disk (a frame and its box) so they survive a restart.

Needs `ultralytics` (AGPL-3.0). Text prompts aren't used, so its CLIP package isn't needed.
"""
import json
import threading
import time
from pathlib import Path


class TaughtDetector:
    name = "yoloe"

    def __init__(self, weights, store, conf=0.3, imgsz=640, device=None):
        self.weights, self.store, self.conf, self.imgsz = str(weights), Path(store), conf, imgsz
        self.device = device
        self.store.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._maker = None          # unfused model, makes embeddings
        self._det = None            # model with the current classes
        self.views = []             # [{"eid", "vpe"}], index = YOLOE class
        self._load_saved()

    def _model(self):
        import torch
        from ultralytics import YOLOE
        m = YOLOE(self.weights)
        m.to(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        return m

    def _embed(self, image, box):
        import numpy as np
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
        if self._maker is None:
            self._maker = self._model()
        W, H = image.size
        prompts = {"bboxes": np.array([[box[0] * W, box[1] * H, box[2] * W, box[3] * H]]), "cls": np.array([0])}
        p = YOLOEVPSegPredictor(overrides={"task": "segment", "mode": "predict", "save": False, "batch": 1,
                                           "verbose": False, "imgsz": self.imgsz, "conf": self.conf},
                                _callbacks=self._maker.callbacks)
        p.set_prompts(prompts)
        p.setup_model(model=self._maker.model, verbose=False)
        return p.get_vpe(image).detach().cpu()

    def _rebuild(self):
        import torch
        self._det = None
        if not self.views:
            return
        m = self._model()
        vpe = torch.cat([v["vpe"] for v in self.views], dim=1)
        m.set_classes([f"view{i}" for i in range(len(self.views))], vpe.to(next(m.model.parameters()).device))
        self._det = m

    def _load_saved(self):
        from PIL import Image
        for meta in sorted(self.store.glob("*.json")):
            d = json.loads(meta.read_text(encoding="utf-8"))
            try:
                vpe = self._embed(Image.open(meta.with_suffix(".jpg")).convert("RGB"), d["box"])
            except Exception as e:
                print(f"skipping taught view {meta.name}: {e}")
                continue
            self.views.append({"eid": d["eid"], "vpe": vpe, "file": meta})
        if self.views:
            self._rebuild()

    def add(self, examples):
        """Teach from [(eid, image, box)]; the detector is rebuilt once."""
        with self._lock:
            for eid, image, box in examples:
                vpe = self._embed(image.convert("RGB"), box)
                meta = self.store / f"{eid}-{int(time.time() * 1000)}-{len(self.views)}.json"
                image.convert("RGB").save(meta.with_suffix(".jpg"), "JPEG", quality=85)
                meta.write_text(json.dumps({"eid": eid, "box": list(box)}), encoding="utf-8")
                self.views.append({"eid": eid, "vpe": vpe, "file": meta})
            self._rebuild()

    def remove_like(self, eid, is_same):
        """Drop this object's examples whose picture `is_same(crop)` says shows the wrong thing."""
        from PIL import Image
        with self._lock:
            dropped = 0
            for v in [v for v in self.views if v["eid"] == eid]:
                box = json.loads(v["file"].read_text(encoding="utf-8"))["box"]
                im = Image.open(v["file"].with_suffix(".jpg")).convert("RGB")
                W, H = im.size
                if is_same(im.crop((box[0] * W, box[1] * H, box[2] * W, box[3] * H))):
                    v["file"].unlink(missing_ok=True)
                    v["file"].with_suffix(".jpg").unlink(missing_ok=True)
                    self.views.remove(v)
                    dropped += 1
            if dropped:
                self._rebuild()
            return dropped

    def remove(self, eid):
        with self._lock:
            for v in [v for v in self.views if v["eid"] == eid]:
                v["file"].unlink(missing_ok=True)
                v["file"].with_suffix(".jpg").unlink(missing_ok=True)
                self.views.remove(v)
            self._rebuild()

    def find(self, image):
        """[(eid, box, conf)], the best box per taught object."""
        with self._lock:
            if self._det is None:
                return []
            r = self._det.predict(image, verbose=False, conf=self.conf, imgsz=self.imgsz)[0]
            best = {}
            for b, c, k in zip(r.boxes.xyxyn.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                eid = self.views[int(k)]["eid"]
                if c > best.get(eid, (None, 0))[1]:
                    best[eid] = (tuple(b), c)
            return [(eid, b, c) for eid, (b, c) in best.items()]
