"""Frames in, memory events out.

    static camera : change-gated, fixed zone, left/centre/right regions,
                    presence + hazard signals for the safety watch,
                    "disappeared" when a known object stops being seen
                    (not while a hand or arm covers where it was)
    both cameras  : a new object is created only after it is seen in two
                    processed frames in a row; on the bench camera an object
                    in a hand can update a known entity but never creates one
                    (held objects are blurred and half covered by fingers)
    mobile phone  : zone from AprilTags, every frame matched against memory,
                    resolves pending look requests, feeds the live overlay
"""
import collections
import io
import threading
import time

import numpy as np
from PIL import Image, ImageOps

from forgemind.config import HAZARD_CONCEPTS, PRESENCE_CONCEPTS
from forgemind.perception.zones import ZoneTracker

KEYFRAME_PX = 1024
MOBILE_KEYFRAME_EVERY_S = 2.0
STATIC_FORCE_EVERY_S = 15.0
# A frame "changed" when this fraction of 96x72 pixels moved by > PIXEL_DELTA
# grey levels: a small object moving trips it, webcam noise doesn't.
CHANGE_FRACTION = 0.004
PIXEL_DELTA = 25
MISSES_BEFORE_GONE = 3
# ...and at least this long unseen (an arm can cover things for a few seconds).
GONE_AFTER_S = 10.0
# Fraction of an object's box covered by a hand/person box for it to count as
# held (new detections) or hidden (known objects that weren't detected).
HELD_OVERLAP = 0.25
OCCLUDED_OVERLAP = 0.20
# A candidate new object must reappear within this long to be confirmed.
CONFIRM_WITHIN_S = 5.0
CONFIRM_IOU = 0.3
# Move linking: a same-concept object the bench camera saw go missing within
# MOVE_LINK_S is matched at a relaxed similarity (match_min_sim - MOVE_RELAX):
# "it left its spot and one like it appeared over there" is most likely a move.
MOVE_LINK_S = 120.0
MOVE_RELAX = 0.15
MAX_CROP_PX = 448
# Recent phone frames kept for teaching (several views of one object).
RECENT_FRAMES = 8


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / (np.linalg.norm(v) or 1.0)


def covered(box, others):
    """Largest fraction of `box` covered by any box in `others`."""
    area = max(1e-9, (box[2] - box[0]) * (box[3] - box[1]))
    best = 0.0
    for o in others:
        w = min(box[2], o[2]) - max(box[0], o[0])
        h = min(box[3], o[3]) - max(box[1], o[1])
        if w > 0 and h > 0:
            best = max(best, w * h / area)
    return best


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def region_of(box):
    cx = (box[0] + box[2]) / 2
    return "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "centre"


class ChangeGate:
    def __init__(self, fraction=CHANGE_FRACTION, delta=PIXEL_DELTA):
        self.fraction, self.delta = fraction, delta
        self.prev = None

    def changed(self, image):
        small = np.asarray(image.convert("L").resize((96, 72)), dtype=np.float32)
        if self.prev is None:
            self.prev = small
            return True
        moved = float((np.abs(small - self.prev) > self.delta).mean())
        if moved >= self.fraction:
            self.prev = small
            return True
        return False


class Perception:
    def __init__(self, settings, memory, segmenter, embedder, hub, safety):
        self.s, self.mem, self.seg, self.emb = settings, memory, segmenter, embedder
        self.hub, self.safety = hub, safety
        self.phone_zone = ZoneTracker(settings.zones)
        self.gate = ChangeGate()
        self.latest = {}             # camera -> {"ts", "image", "detections", "zone"}
        self.recent = collections.deque(maxlen=RECENT_FRAMES)   # phone frames, newest last
        self.last_processed = {}     # camera -> ts
        self.last_keyframe = {}      # camera -> ts
        self.misses = {}             # entity id -> consecutive static misses
        self.pending = {}            # camera -> candidate new objects from the last frame
        self.gone = set()
        self.paused = False
        self.private = False
        self._lock = threading.Lock()

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def decode(data):
        return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")

    def _save_keyframe(self, camera, image, ts, force=False):
        if self.private:
            return None
        if (not force and camera == "mobile"
                and ts - self.last_keyframe.get(camera, 0) < MOBILE_KEYFRAME_EVERY_S):
            return None
        self.last_keyframe[camera] = ts
        name = f"{camera}-{int(ts * 1000)}.jpg"
        kf = image.copy()
        kf.thumbnail((KEYFRAME_PX, KEYFRAME_PX))
        kf.save(self.mem.keyframes_dir / name, "JPEG", quality=82)
        return name

    @staticmethod
    def crop(image, box, pad=0.05):
        w, h = image.size
        x0, y0, x1, y1 = box
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(1, x1 + pad), min(1, y1 + pad)
        c = image.crop((int(x0 * w), int(y0 * h), max(int(x1 * w), int(x0 * w) + 2),
                        max(int(y1 * h), int(y0 * h) + 2)))
        c.thumbnail((MAX_CROP_PX, MAX_CROP_PX))
        return c

    def zone_for(self, camera, ts=None):
        return self.s.static_zone if camera == "static" else self.phone_zone.current(ts)

    def recent_frames(self, max_age=4.0, now=None):
        """Phone frames from the last `max_age` seconds, newest first."""
        now = now or time.time()
        return [f for f in reversed(self.recent) if now - f["ts"] <= max_age]

    def current_image(self, camera):
        item = self.latest.get(camera)
        return item["image"] if item else None

    # -- main entry -----------------------------------------------------------

    def process(self, camera, data, ts=None, force=False):
        if camera not in ("static", "mobile"):
            raise ValueError("camera must be 'static' or 'mobile'")
        if self.paused:
            return {"paused": True, "detections": []}
        ts = ts or time.time()
        image = data if isinstance(data, Image.Image) else self.decode(data)
        with self._lock:
            return self._process(camera, image, ts, force)

    def _process(self, camera, image, ts, force):
        t0 = time.perf_counter()
        zone_changed = False
        if camera == "mobile":
            zone, zone_changed = self.phone_zone.update(image, ts)
        else:
            zone = self.s.static_zone
            changed = self.gate.changed(image)
            stale = ts - self.last_processed.get(camera, 0) >= STATIC_FORCE_EVERY_S
            if not (changed or stale or force):
                prev = self.latest.get(camera, {})
                return {"camera": camera, "zone": zone, "skipped": True, "events": [],
                        "detections": prev.get("detections", []), "keyframe": None}
        self.last_processed[camera] = ts
        dets = self.seg.segment(image, self.s.concepts)
        keyframe = None if self.private else self._save_keyframe(camera, image, ts)

        out_dets, events, claimed = [], [], set()
        present = False
        occluders = [d.box for d in dets if d.concept in PRESENCE_CONCEPTS]
        candidates = []
        for d in sorted(dets, key=lambda d: -d.score):
            item = {"concept": d.concept, "box": [round(b, 4) for b in d.box],
                    "score": round(d.score, 3)}
            if d.concept in PRESENCE_CONCEPTS:
                present = True
                item["kind"] = "presence"
            elif d.concept in HAZARD_CONCEPTS:
                item["kind"] = "hazard"
                if camera == "static":
                    self.safety.seen(d.concept, ts)
            elif not self.private:
                region = region_of(d.box) if camera == "static" else None
                if keyframe is None and camera == "mobile":
                    # make sure a first sighting/move has a picture to show
                    keyframe = self._save_keyframe(camera, image, ts, force=True)
                vec = _unit(self.emb.embed_image(self.crop(image, d.box)))
                eid, _ = self.mem.registry.match(d.concept, vec, camera, zone, region, ts,
                                                 exclude=claimed)
                if eid is None:
                    held = camera == "static" and covered(d.box, occluders) >= HELD_OVERLAP
                    if not held:
                        eid = self._moved_from(d.concept, vec, ts, claimed)
                if eid is None:
                    if held or (self.s.confirm_new and not self._confirmed(camera, d, vec, ts)):
                        if not held:
                            candidates.append({"concept": d.concept, "box": d.box,
                                               "vec": vec, "ts": ts})
                        item.update(kind="object", entity_id=None, label=d.concept,
                                    pending=not held, held=held)
                        out_dets.append(item)
                        continue
                r = self.mem.registry.observe(
                    camera=camera, zone=zone, region=region, concept=d.concept, box=d.box,
                    score=d.score, crop=None, keyframe=keyframe, ts=ts,
                    exclude=claimed, vec=vec, entity_id=eid)
                claimed.add(r["entity_id"])
                item.update(entity_id=r["entity_id"], label=r["label"], kind="object")
                if r["event"]:
                    events.append({"type": r["event"], "entity_id": r["entity_id"],
                                   "label": r["label"], "zone": zone})
                if camera == "static":
                    self.safety.seen(d.concept, ts)
            out_dets.append(item)

        self.pending[camera] = candidates
        if present and camera == "static":
            self.safety.present(ts)
        if camera == "mobile" and zone == self.s.static_zone:
            self.safety.present(ts)          # carrying the phone at the bench
        if camera == "static" and not self.private:
            events += self._track_absences(claimed, ts, occluders)
        if camera == "mobile":
            labels = [i.get("label") or i["concept"] for i in out_dets]
            for look in self.hub.check_looks(zone, labels, claimed):
                self.mem.db.log_event("look_done", "observed", {"target": look["target"]},
                                      camera="mobile", zone=zone, ts=ts)
                events.append({"type": "look_done", "target": look["target"]})
            if zone_changed and zone:
                self.mem.db.log_event("phone_zone", "observed", {"zone": zone},
                                      camera="mobile", zone=zone, ts=ts)

        self.latest[camera] = {"ts": ts, "image": image, "detections": out_dets, "zone": zone,
                               "keyframe": keyframe}
        if camera == "mobile":
            self.recent.append(self.latest[camera])
        return {"camera": camera, "zone": zone, "detections": out_dets, "events": events,
                "keyframe": keyframe, "ms": round((time.perf_counter() - t0) * 1000)}

    def _moved_from(self, concept, vec, ts, claimed):
        """A same-concept entity that recently went missing from the bench, if similar enough."""
        missing = [eid for eid, n in self.misses.items() if n > 0 or eid in self.gone]
        if not missing:
            return None
        best, best_sim = None, self.s.match_min_sim - MOVE_RELAX
        for eid in missing:
            if eid in claimed:
                continue
            e = self.mem.registry.get(eid)
            if (not e or e["concept"] != concept or not e["proto"]
                    or ts - (e["last_seen"] or 0) > MOVE_LINK_S):
                continue
            from forgemind.memory.db import blob_to_vec
            sim = float(np.dot(vec, blob_to_vec(e["proto"])))
            if sim >= best_sim:
                best, best_sim = eid, sim
        return best

    def _confirmed(self, camera, d, vec, ts):
        """Was this unmatched detection also a candidate in the previous frame?"""
        for c in self.pending.get(camera, []):
            if c["concept"] != d.concept or ts - c["ts"] > CONFIRM_WITHIN_S:
                continue
            same_place = camera == "static" and iou(c["box"], d.box) >= CONFIRM_IOU
            if same_place or float(np.dot(c["vec"], vec)) >= self.s.match_min_sim:
                return True
        return False

    def _track_absences(self, seen_ids, ts, occluders=()):
        """Known bench objects the static camera stopped seeing.

        A miss doesn't count while a hand/person box covers the object's last
        position; "gone" needs MISSES_BEFORE_GONE misses and GONE_AFTER_S.
        """
        events = []
        rows = self.mem.db.all(
            "SELECT id, last_box, last_seen FROM entities WHERE deleted=0 AND last_camera='static'"
            " AND last_zone=? AND last_seen > ?", (self.s.static_zone, ts - 3600))
        for r in rows:
            eid = r["id"]
            if eid in seen_ids:
                self.misses[eid] = 0
                self.gone.discard(eid)
                continue
            box = [float(v) for v in r["last_box"].split(",")] if r["last_box"] else None
            if box and covered(box, occluders) >= OCCLUDED_OVERLAP:
                continue                       # hidden, not gone
            self.misses[eid] = self.misses.get(eid, 0) + 1
            if (self.misses[eid] >= MISSES_BEFORE_GONE and ts - r["last_seen"] >= GONE_AFTER_S
                    and eid not in self.gone):
                self.gone.add(eid)
                self.mem.registry.mark_disappeared(eid, "static", self.s.static_zone, ts)
                events.append({"type": "disappeared", "entity_id": eid,
                               "label": self.mem.registry.label(eid)})
        return events

    # -- summaries for the agent ----------------------------------------------

    def scene(self, camera):
        item = self.latest.get(camera)
        if not item:
            return {"camera": camera, "available": False}
        return {"camera": camera, "available": True, "zone": item["zone"],
                "age_s": round(time.time() - item["ts"], 1),
                "objects": [self.mem.registry.label(d["entity_id"]) or d["concept"]
                            for d in item["detections"] if d.get("kind") == "object"],
                "people_present": any(d.get("kind") == "presence" for d in item["detections"]),
                "hazards": [d["concept"] for d in item["detections"] if d.get("kind") == "hazard"]}
