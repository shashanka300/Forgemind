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
from forgemind.providers.base import Detection
from forgemind.perception import bench
from forgemind.perception.zones import ZoneTracker

KEYFRAME_PX = 1024
MOBILE_KEYFRAME_EVERY_S = 2.0
STATIC_FORCE_EVERY_S = 15.0
# The bench camera sends a few frames a second for the live preview; detection stays at
# about one a second however much the scene is moving.
STATIC_MIN_EVERY_S = 1.0
# SAM only finds what it is asked for, and FM_CONCEPTS can't list everything the user owns. So
# once something is taught, each frame also asks for its concept and for "small object". These
# candidates never create objects (they fire on everything): one counts only if it looks like
# a taught object, more strictly than a named detection does.
CANDIDATE_CONCEPT = "small object"
CANDIDATE_MIN_SIM = 0.90
MAX_CANDIDATES = 24
MAX_TAUGHT_CONCEPTS = 10
# One frame of "smoke" isn't smoke (phone test, Oct 31: a false critical alert).
HAZARD_MIN_FRAMES = 3
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
# Object boxes larger than this are scene-level labels, not objects (SAM boxes
# the whole photo for "soldering station").
MAX_OBJECT_AREA = 0.6
# After a look request at a zone resolves, phone frames in that zone without the
# object count towards "it's not there" (PRD acceptance #5).
ABSENT_AFTER_FRAMES = 3
ABSENT_CHECK_S = 30.0
# Two different object concepts on one box (SAM said "phone" 0.90 and
# "multimeter" 0.95 for the same meter): keep the higher score.
CROSS_CONCEPT_IOU = 0.7
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


def box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


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


def one_label_per_object(dets):
    """Drop scene-sized object boxes and duplicate labels of the same object."""
    keep, objects = [], []
    for d in sorted(dets, key=lambda d: -d.score):
        if d.concept in PRESENCE_CONCEPTS or d.concept in HAZARD_CONCEPTS:
            keep.append(d)
            continue
        if d.area > MAX_OBJECT_AREA:
            continue
        if any(iou(d.box, o.box) >= CROSS_CONCEPT_IOU for o in objects):
            continue
        objects.append(d)
        keep.append(d)
    return keep


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
        self.newest = {}
        self.hazard_frames = {}
        self.learning = {}                   # taught entity -> until when the phone adds views
        self.bench_dirty = False
        self.latest = {}             # camera -> {"ts", "image", "detections", "zone"}
        self.recent = collections.deque(maxlen=RECENT_FRAMES)   # phone frames, newest last
        self.last_processed = {}     # camera -> ts
        self.last_keyframe = {}      # camera -> ts
        self.misses = {}             # entity id -> consecutive static misses
        self.pending = {}            # camera -> candidate new objects from the last frame
        self.absence_checks = {}     # entity id -> {"zone", "label", "misses", "until"}
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
        if camera == "static":
            image = bench.apply(image, self.s.bench_roi, self.s.bench_masks)
        self.newest[camera] = image          # the live preview shows this, processed or not
        with self._lock:
            return self._process(camera, image, ts, force)

    def _process(self, camera, image, ts, force):
        t0 = time.perf_counter()
        zone_changed = False
        if camera == "mobile":
            zone, zone_changed = self.phone_zone.update(image, ts)
        else:
            zone = self.s.static_zone
            self.bench_dirty = self.gate.changed(image) or self.bench_dirty
            changed = self.bench_dirty
            since = ts - self.last_processed.get(camera, 0)
            too_soon = 0 <= since < STATIC_MIN_EVERY_S
            stale = since >= STATIC_FORCE_EVERY_S
            # a new object waiting for its second sighting: look again even if
            # nothing changed, or a still scene never confirms it
            confirming = bool(self.pending.get(camera))
            if not force and (too_soon or not (changed or stale or confirming)):
                prev = self.latest.get(camera, {})
                for d in prev.get("detections", []):   # unchanged scene: still there
                    if d.get("kind") == "object" or (d.get("kind") == "hazard" and
                                                     self.hazard_frames.get(d["concept"], 0) >= HAZARD_MIN_FRAMES):
                        self.safety.seen(d["concept"], ts)
                        if d.get("label") and d["label"] != d["concept"]:
                            self.safety.seen(d["label"], ts)
                    elif d.get("kind") == "presence":
                        self.safety.present(ts)
                return {"camera": camera, "zone": zone, "skipped": True, "events": [],
                        "detections": prev.get("detections", []), "keyframe": None}
        self.last_processed[camera] = ts
        if camera == "static":
            self.bench_dirty = False
        extra = []
        if not self.private:
            extra = [c for c in self.mem.registry.taught_concepts() if c not in self.s.concepts]
            extra = extra[:MAX_TAUGHT_CONCEPTS]
            if self.mem.registry.any_taught():
                extra.append(CANDIDATE_CONCEPT)
        found = self.seg.segment(image, list(self.s.concepts) + extra)
        dets = one_label_per_object([d for d in found if d.concept not in extra])
        candidates_in = [d for d in found if d.concept in extra]
        keyframe = None if self.private else self._save_keyframe(camera, image, ts)

        out_dets, events, claimed = [], [], set()
        present = False
        occluders = [d.box for d in dets if d.concept in PRESENCE_CONCEPTS]
        candidates = []
        dets = sorted(dets, key=lambda d: -d.score)
        hazards_ok = (self._hazard_confirmed({d.concept for d in dets if d.concept in HAZARD_CONCEPTS})
                      if camera == "static" else set())
        # one batched embedding call for every object crop in the frame
        objs = [d for d in dets if d.concept not in PRESENCE_CONCEPTS
                and d.concept not in HAZARD_CONCEPTS] if not self.private else []
        taken = self._taught_candidates(image, candidates_in, objs)
        if taken:
            dets = dets + [d for d, _ in taken]
            objs = objs + [d for d, _ in taken]
        forced = {id(d): eid for d, eid in taken}
        vecs = dict(zip(map(id, objs), self.emb.embed_images([self.crop(image, d.box)
                                                              for d in objs]))) if objs else {}
        for d in dets:
            item = {"concept": d.concept, "box": [round(b, 4) for b in d.box],
                    "score": round(d.score, 3)}
            if d.concept in PRESENCE_CONCEPTS:
                present = True
                item["kind"] = "presence"
            elif d.concept in HAZARD_CONCEPTS:
                item["kind"] = "hazard"
                if camera == "static" and d.concept in hazards_ok:
                    self.safety.seen(d.concept, ts)
            elif not self.private:
                region = region_of(d.box) if camera == "static" else None
                if keyframe is None and camera == "mobile":
                    # make sure a first sighting/move has a picture to show
                    keyframe = self._save_keyframe(camera, image, ts, force=True)
                vec = _unit(vecs[id(d)])
                if id(d) in forced:
                    eid = forced[id(d)]
                else:
                    eid, _ = self.mem.registry.match(d.concept, vec, camera, zone, region, ts,
                                                     exclude=claimed | set(forced.values()))
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
                if camera == "mobile" and ts < self.learning.get(r["entity_id"], 0):
                    self.mem.registry.add_view(r["entity_id"], vec)
                item.update(entity_id=r["entity_id"], label=r["label"], kind="object")
                if r["event"]:
                    events.append({"type": r["event"], "entity_id": r["entity_id"],
                                   "label": r["label"], "zone": zone})
                if camera == "static":
                    # watches match the taught name too: SAM never says "soldering
                    # iron", but the user's taught iron is labelled that way
                    self.safety.seen(d.concept, ts)
                    if r["label"] and r["label"] != d.concept:
                        self.safety.seen(r["label"], ts)
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
            boxes = {i["entity_id"]: i["box"] for i in out_dets if i.get("entity_id")}
            for f in self.hub.check_seeks(boxes, now=ts):
                self.mem.db.log_event("found", "observed", {"label": f["label"]}, camera="mobile",
                                      zone=zone, entity_id=f["entity_id"], ts=ts)
                events.append({"type": "found", "entity_id": f["entity_id"], "label": f["label"]})
            for look in self.hub.check_looks(zone, labels, claimed):
                self.mem.db.log_event("look_done", "observed", {"target": look["target"]},
                                      camera="mobile", zone=zone, ts=ts)
                events.append({"type": "look_done", "target": look["target"]})
                eid = look.get("entity_id")
                if eid and eid not in claimed and zone:
                    self.absence_checks[eid] = {"zone": zone, "misses": 0, "until": ts + ABSENT_CHECK_S,
                                                "label": self.mem.registry.label(eid)}
            events += self._check_absences(zone, claimed, ts)
            if zone:
                self.safety.phone_at(zone)
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

    def _check_absences(self, zone, seen_ids, ts):
        """The user is looking where an object should be: is it there?"""
        events = []
        for eid, c in list(self.absence_checks.items()):
            if eid in seen_ids or ts > c["until"]:
                del self.absence_checks[eid]          # found it (seek/registry handle that)
                continue
            if zone != c["zone"]:
                continue
            c["misses"] += 1
            if c["misses"] >= ABSENT_AFTER_FRAMES:
                del self.absence_checks[eid]
                self.mem.db.log_event("not_found", "observed", {"label": c["label"], "zone": zone},
                                      camera="mobile", zone=zone, entity_id=eid, ts=ts)
                self.hub.say(f"I don't see the {c['label']} on the {zone}. It must have moved.")
                events.append({"type": "not_found", "entity_id": eid, "label": c["label"],
                               "zone": zone})
        return events

    def _taught_candidates(self, image, candidates, named):
        """[(detection, entity id)]: the best-looking candidate per taught object."""
        fresh = sorted((d for d in candidates if box_area(d.box) <= MAX_OBJECT_AREA
                        and not any(iou(d.box, n.box) >= 0.7 for n in named)),
                       key=lambda d: -d.score)[:MAX_CANDIDATES]
        if not fresh:
            return []
        vecs = self.emb.embed_images([self.crop(image, d.box) for d in fresh])
        best = {}
        for d, v in zip(fresh, vecs):
            v = _unit(v)
            eid, sim = self.mem.registry.match_taught(v, CANDIDATE_MIN_SIM)
            if eid is not None and sim > best.get(eid, (0, None))[0]:
                best[eid] = (sim, d)
        out = []
        for eid, (_, d) in best.items():
            e = self.mem.registry.get(eid)
            out.append((Detection(e["concept"], d.box, d.score), eid))
        return out

    def _hazard_confirmed(self, concepts_now):
        """Hazards seen on the bench for HAZARD_MIN_FRAMES processed frames in a row."""
        self.hazard_frames = {c: self.hazard_frames.get(c, 0) + 1 for c in concepts_now}
        return {c for c, n in self.hazard_frames.items() if n >= HAZARD_MIN_FRAMES}

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
