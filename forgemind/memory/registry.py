"""Object registry: one entity per real object, sightings from either camera.

Matching a new detection to an entity (PRD F-OBJ-2):
  1. taught objects first: max cosine to any of its reference crops must
     reach FM_TEACH_MIN_SIM (the gallery app's "teach a name" rule),
     regardless of the detected concept (a coarse detector may call a hex
     key a screwdriver; the taught identity wins);
  2. otherwise untaught entities of the same concept, by cosine to their
     running prototype, with a small bonus for "same camera, same place,
     seen a moment ago" (that is what a tracker would tell us);
  3. no match -> a new entity.

Memory is written on change (appeared / moved), plus a heartbeat sighting
per entity per minute, never on every frame.
"""
import math
import time

import numpy as np

from forgemind.memory.db import blob_to_vec, vec_to_blob

HEARTBEAT_S = 60
REAPPEAR_S = 300
CONTINUITY_BONUS = 0.10
# Default floor for text->image search; embedders set their own (text_min_sim).
VISUAL_MIN = 0.22


def _norm(v):
    v = np.asarray(v, np.float32)
    return v / (np.linalg.norm(v) or 1.0)


def fmt_ago(seconds):
    s = max(0, int(seconds))
    if s < 60:
        return "just now" if s < 10 else f"{s} seconds ago"
    if s < 3600:
        return f"{s // 60} minute{'s' if s // 60 != 1 else ''} ago"
    if s < 86400:
        return f"{s // 3600} hour{'s' if s // 3600 != 1 else ''} ago"
    return f"{s // 86400} day{'s' if s // 86400 != 1 else ''} ago"


UNLEARN_MIN_SIM = 0.93


class TeachConflict(Exception):
    """The views being taught look like a different taught object."""

    def __init__(self, label):
        super().__init__(label)
        self.label = label


class Registry:
    def __init__(self, db, index, embedder, settings):
        self.db, self.index, self.emb, self.s = db, index, embedder, settings

    # -- matching -------------------------------------------------------------

    def _candidates(self):
        ents = self.db.all("SELECT * FROM entities WHERE deleted=0")
        refs = {}
        for r in self.db.all("SELECT entity_id, vec FROM entity_refs"):
            refs.setdefault(r["entity_id"], []).append(blob_to_vec(r["vec"]))
        return ents, refs

    def match(self, concept, vec, camera=None, zone=None, region=None, ts=None,
              exclude=()):
        """Best (entity_id, similarity) for a crop vector, or (None, best_sim)."""
        ts = ts or time.time()
        ents, refs = self._candidates()
        best, best_sim = None, 0.0
        for e in ents:                                   # 1. taught identities
            if not e["taught"] or e["id"] in exclude or e["id"] not in refs:
                continue
            sim = max(float(vec @ r) for r in refs[e["id"]])
            if sim >= self.s.teach_min_sim and sim > best_sim:
                best, best_sim = e["id"], sim
        if best is not None:
            return best, best_sim
        for e in ents:                                   # 2. same-concept objects
            if e["taught"] or e["concept"] != concept or e["id"] in exclude:
                continue
            proto = blob_to_vec(e["proto"])
            if proto is None:
                continue
            sim = float(vec @ proto)
            if (camera and e["last_camera"] == camera and e["last_zone"] == zone
                    and e["last_region"] == region and ts - (e["last_seen"] or 0) < 120):
                sim += CONTINUITY_BONUS
            if sim >= self.s.match_min_sim and sim > best_sim:
                best, best_sim = e["id"], sim
        return best, best_sim

    def match_taught(self, vec, min_sim, exclude=()):
        """Best taught identity for a crop, or (None, best_sim)."""
        ents, refs = self._candidates()
        best, best_sim = None, 0.0
        for e in ents:
            if e["taught"] and e["id"] not in exclude and e["id"] in refs:
                sim = max(float(vec @ r) for r in refs[e["id"]])
                if sim > best_sim:
                    best, best_sim = e["id"], sim
        return (best, best_sim) if best_sim >= min_sim else (None, best_sim)

    def unit_embedding(self, crop):
        return _norm(self.emb.embed_images([crop])[0])

    def similarity_to(self, eid, vec):
        rows = self.db.all("SELECT vec FROM entity_refs WHERE entity_id=?", (eid,))
        return max((float(vec @ blob_to_vec(r["vec"])) for r in rows), default=0.0)

    def any_taught(self):
        return bool(self.db.one("SELECT 1 AS x FROM entities WHERE deleted=0 AND taught=1 LIMIT 1"))

    def taught_concepts(self):
        """The words SAM can look for taught objects by ("object" says nothing)."""
        rows = self.db.all("SELECT DISTINCT concept FROM entities WHERE deleted=0 AND taught=1")
        return [r["concept"] for r in rows if r["concept"] and r["concept"] != "object"]

    # -- writes ---------------------------------------------------------------

    def observe(self, *, camera, zone, region, concept, box, score, crop,
                keyframe=None, ts=None, exclude=(), vec=None, entity_id=None):
        """Record one detection. Returns {entity_id, event, label, sim}.

        `entity_id` forces the match (the caller already decided, e.g. a move).
        """
        ts = ts or time.time()
        vec = _norm(vec if vec is not None else self.emb.embed_image(crop))  # crop unused if vec
        if entity_id is not None:
            proto = self.db.one("SELECT proto FROM entities WHERE id=?", (entity_id,))["proto"]
            eid, sim = entity_id, float(vec @ blob_to_vec(proto)) if proto else 0.0
        else:
            eid, sim = self.match(concept, vec, camera, zone, region, ts, exclude)
        event = None
        if eid is None:
            eid = self.db.execute(
                "INSERT INTO entities(concept,created_at,proto) VALUES(?,?,?)",
                (concept, ts, vec_to_blob(vec)))
            prev, event = None, "appeared"
        else:
            prev = self.db.one("SELECT * FROM entities WHERE id=?", (eid,))
            if zone is None:
                # The phone saw it but doesn't know where it is (no tag seen yet):
                # keep the last known zone rather than moving it to "nowhere".
                zone = prev["last_zone"]
            if prev["last_zone"] != zone:
                event = "moved"
            elif camera == "static" and prev["last_region"] != region:
                event = "moved"
            elif ts - (prev["last_seen"] or 0) > REAPPEAR_S:
                event = "appeared"
            if not prev["taught"]:
                proto = _norm(0.9 * blob_to_vec(prev["proto"]) + 0.1 * vec)
                self.db.execute("UPDATE entities SET proto=? WHERE id=?",
                                (vec_to_blob(proto), eid))
        self.db.execute(
            "UPDATE entities SET last_seen=?, last_camera=?, last_zone=?, last_region=?,"
            " last_box=?, last_score=?, sightings=sightings+1,"
            " last_keyframe=COALESCE(?, last_keyframe) WHERE id=?",
            (ts, camera, zone, region, ",".join(f"{b:.4f}" for b in box), score,
             keyframe, eid))
        last_written = self.db.one(
            "SELECT MAX(ts) AS t FROM sightings WHERE entity_id=?", (eid,))["t"] or 0
        if event or ts - last_written >= HEARTBEAT_S:
            self.db.execute(
                "INSERT INTO sightings(entity_id,ts,camera,zone,region,box,score,sim,keyframe)"
                " VALUES(?,?,?,?,?,?,?,?,?)",
                (eid, ts, camera, zone, region, ",".join(f"{b:.4f}" for b in box),
                 score, sim, keyframe))
            self.index.upsert(vec.tolist(), {"kind": "crop", "entity_id": eid,
                                             "camera": camera, "zone": zone or "",
                                             "ts": ts, "keyframe": keyframe or ""})
        if event:
            payload = {"concept": concept, "region": region, "box": list(box)}
            if prev is not None:
                payload.update({"from_zone": prev["last_zone"], "from_region": prev["last_region"]})
            self.db.log_event(event, "observed", payload, camera=camera, zone=zone,
                              entity_id=eid, ts=ts)
        label = self.label(eid)
        return {"entity_id": eid, "event": event, "label": label, "sim": round(sim, 3)}

    def mark_disappeared(self, eid, camera, zone, ts=None):
        self.db.log_event("disappeared", "observed", {"label": self.label(eid)},
                          camera=camera, zone=zone, entity_id=eid, ts=ts)

    def _keep_own_views(self, vecs, own_id):
        """Drop views that look like another taught object more than like this one.

        Phone test, Oct 31: re-teaching with a mouse in view stored mouse crops under
        "baby's biter", and from then on the mouse matched as the biter.
        """
        ents, refs = self._candidates()
        others = [e for e in ents if e["taught"] and e["id"] != own_id and e["id"] in refs]
        keep, rival = [], None
        for v in vecs:
            own = max((float(v @ r) for r in refs.get(own_id, [])), default=0.0)
            best, best_sim = None, 0.0
            for e in others:
                sim = max(float(v @ r) for r in refs[e["id"]])
                if sim > best_sim:
                    best, best_sim = e, sim
            if best is not None and best_sim >= self.s.teach_min_sim and best_sim > own:
                rival = best
            else:
                keep.append(v)
        if not keep:
            raise TeachConflict(rival["name"] or rival["concept"])
        return keep

    def teach(self, name, crops, concept="object", keyframe=None, force=False):
        """Name an object from 1+ example crops (gallery's TEACH, for objects).

        `force` skips the look-alike check (the user just corrected a wrong label).
        """
        name = " ".join(name.split())[:60]
        vecs = [_norm(v) for v in self.emb.embed_images(crops)]
        existing = self.db.one(
            "SELECT * FROM entities WHERE deleted=0 AND lower(name)=lower(?)", (name,))
        if not force:
            vecs = self._keep_own_views(vecs, existing["id"] if existing else None)
        if existing:
            eid = existing["id"]
        else:
            eid, sim = self.match(concept, vecs[0])
            if eid is not None and self.db.one("SELECT taught FROM entities WHERE id=?",
                                               (eid,))["taught"]:
                eid = None              # don't steal another taught identity
            if eid is None:
                eid = self.db.execute(
                    "INSERT INTO entities(concept,created_at) VALUES(?,?)", (concept, time.time()))
        for v in vecs:
            self.db.execute("INSERT INTO entity_refs(entity_id,vec,keyframe,created_at)"
                            " VALUES(?,?,?,?)", (eid, vec_to_blob(v), keyframe, time.time()))
        proto = _norm(np.mean(vecs, axis=0))
        self.db.execute("UPDATE entities SET name=?, taught=1, proto=?,"
                        " last_keyframe=COALESCE(last_keyframe, ?) WHERE id=?",
                        (name, vec_to_blob(proto), keyframe, eid))
        self.db.log_event("taught", "stated", {"name": name, "examples": len(vecs)},
                          entity_id=eid)
        v = self.view(self.get(eid))
        v["examples"] = self.db.one("SELECT COUNT(*) AS n FROM entity_refs WHERE entity_id=?",
                                    (eid,))["n"]
        return v

    def saw_taught(self, eid, zone, box, keyframe=None):
        """The phone was looking at it while it was taught: that's a sighting."""
        self.db.execute(
            "UPDATE entities SET last_seen=?, last_camera='mobile',"
            " last_zone=COALESCE(?, last_zone), last_region=NULL, last_box=?, last_score=0.9,"
            " sightings=sightings+1, last_keyframe=COALESCE(?, last_keyframe) WHERE id=?",
            (time.time(), zone, ",".join(f"{b:.4f}" for b in box), keyframe, eid))

    def add_view(self, eid, vec, max_views=12, min_own=0.75):
        """Another view of a taught object: not a near-copy, and still looking like itself."""
        rows = self.db.all("SELECT vec FROM entity_refs WHERE entity_id=?", (eid,))
        sims = [float(vec @ blob_to_vec(r["vec"])) for r in rows]
        if len(rows) >= max_views or not sims or max(sims) > 0.97 or max(sims) < min_own:
            return False
        self.db.execute("INSERT INTO entity_refs(entity_id,vec,keyframe,created_at) VALUES(?,?,?,?)",
                        (eid, vec_to_blob(vec), None, time.time()))
        return True

    def same_name(self, name):
        """An existing taught name that is this one ("scissors" = "scissor"), else `name`."""
        stem = lambda n: " ".join(w.rstrip("s") for w in n.lower().split())
        for r in self.db.all("SELECT name FROM entities WHERE deleted=0 AND taught=1 AND name IS NOT NULL"):
            if stem(r["name"]) == stem(name):
                return r["name"]
        return name

    def unlearn(self, eid, crop):
        """Forget the stored views of `eid` that look like `crop` (a wrong label).

        An identity left with no views was entirely wrong, so it is removed.
        """
        v = _norm(self.emb.embed_images([crop])[0])
        rows = self.db.all("SELECT id, vec FROM entity_refs WHERE entity_id=?", (eid,))
        # Only nearly the same picture: a similar bottle under another name stays (test 5: each
        # correction wiped out the object before it).
        gone = [r["id"] for r in rows if float(v @ blob_to_vec(r["vec"])) >= UNLEARN_MIN_SIM]
        for rid in gone:
            self.db.execute("DELETE FROM entity_refs WHERE id=?", (rid,))
        left = [blob_to_vec(r["vec"]) for r in rows if r["id"] not in gone]
        if left:
            self.db.execute("UPDATE entities SET proto=? WHERE id=?",
                            (vec_to_blob(_norm(np.mean(left, axis=0))), eid))
        elif rows:
            self.delete(eid)
        return len(gone)

    def correct(self, eid, name=None, zone=None):
        if name:
            self.db.execute("UPDATE entities SET name=? WHERE id=?", (name, eid))
        if zone:
            self.db.execute("UPDATE entities SET last_zone=?, last_camera='user',"
                            " last_seen=? WHERE id=?", (zone, time.time(), eid))
        self.db.log_event("corrected", "stated", {"name": name, "zone": zone}, entity_id=eid)
        return self.view(self.get(eid))

    def delete(self, eid):
        self.db.execute("UPDATE entities SET deleted=1 WHERE id=?", (eid,))
        self.db.log_event("forgotten", "stated", {}, entity_id=eid)

    # -- reads ----------------------------------------------------------------

    def get(self, eid):
        return self.db.one("SELECT * FROM entities WHERE id=? AND deleted=0", (eid,))

    def label(self, eid):
        e = self.db.one("SELECT name, concept FROM entities WHERE id=?", (eid,))
        return (e["name"] or e["concept"]) if e else None

    def confidence(self, e, now=None):
        if not e or not e["last_seen"]:
            return 0.0
        age = (now or time.time()) - e["last_seen"]
        quality = 0.6 + 0.4 * min(1.0, e["last_score"] or 0.5)
        return round(quality * math.pow(0.5, age / self.s.confidence_half_life), 3)

    def view(self, e, now=None):
        if not e:
            return None
        now = now or time.time()
        conf = self.confidence(e, now)
        box = [float(x) for x in e["last_box"].split(",")] if e["last_box"] else None
        return {
            "id": e["id"], "label": e["name"] or e["concept"], "name": e["name"],
            "concept": e["concept"], "taught": bool(e["taught"]),
            "zone": e["last_zone"], "region": e["last_region"], "camera": e["last_camera"],
            "last_seen": e["last_seen"],
            "ago": fmt_ago(now - e["last_seen"]) if e["last_seen"] else "never",
            "confidence": conf, "stale": conf < self.s.stale_confidence,
            "keyframe": f"/keyframes/{e['last_keyframe']}" if e["last_keyframe"] else None,
            "snapshot": f"/api/entities/{e['id']}/snapshot" if e["last_keyframe"] else None,
            "box": box, "sightings": e["sightings"],
        }

    def describe(self, v):
        if not v["last_seen"]:
            return f"I know the {v['label']} but haven't seen it yet."
        where = f"on the {v['zone']}" if v["zone"] else "somewhere I couldn't place"
        if v["region"] and v["camera"] == "static":
            where += f" ({v['region']} side)" if v["region"] != "centre" else " (centre)"
        who = {"static": "the bench camera", "mobile": "your phone",
               "user": "you"}.get(v["camera"], v["camera"])
        s = f"The {v['label']} was last seen {where} by {who}, {v['ago']}."
        miss = self.db.one("SELECT ts, zone FROM events WHERE type='not_found' AND entity_id=?"
                           " AND ts > ? ORDER BY ts DESC LIMIT 1", (v["id"], v["last_seen"]))
        if miss:
            s += (f" But when you checked the {miss['zone']} {fmt_ago(time.time() - miss['ts'])},"
                  " it wasn't there.")
        elif v["stale"]:
            s += " That's a while ago, so it may have moved."
        return s

    def list(self, limit=200):
        rows = self.db.all("SELECT * FROM entities WHERE deleted=0"
                           " ORDER BY last_seen DESC NULLS LAST LIMIT ?", (limit,))
        return [self.view(r) for r in rows]

    def history(self, eid, limit=10):
        return self.db.all("SELECT ts,camera,zone,region,keyframe FROM sightings"
                           " WHERE entity_id=? ORDER BY ts DESC LIMIT ?", (eid, limit))

    def find(self, query, limit=3):
        """Entities matching a name/description: names first, then visual search."""
        q = " ".join(query.lower().split())
        q = q.removeprefix("the ").removeprefix("my ")
        ents = self.db.all("SELECT * FROM entities WHERE deleted=0")
        scored = []
        head = q.split()[-1] if q else ""
        single_word = len(q.split()) == 1
        for e in ents:
            label = (e["name"] or "").lower()
            concept = e["concept"].lower()
            s = 0.0
            if label and (label == q or label in q or q in label):
                s = 3.0
            elif concept == q or concept in q or q in concept:
                s = 2.0
            elif single_word and head in (label + " " + concept).split():
                s = 1.0                 # "tape" finds "paper tape"; "measurement tape" doesn't
            if s:
                scored.append((s + (e["last_seen"] or 0) * 1e-12, e))
        if scored:
            scored.sort(key=lambda x: -x[0])
            return [self.view(e) for _, e in scored[:limit]]
        # Visual fallback: CLIP text tower against stored object crops.
        hits = self.index.search(self.emb.embed_text(f"a photo of a {q}"), limit=40,
                                 where={"kind": "crop"})
        seen, out = set(), []
        floor = getattr(self.emb, "text_min_sim", VISUAL_MIN)
        for _, score, p in hits:
            eid = p.get("entity_id")
            if score < floor or eid in seen:
                continue
            seen.add(eid)
            v = self.view(self.get(eid))
            if v:
                v["match_score"] = round(score, 3)
                out.append(v)
            if len(out) >= limit:
                break
        return out
