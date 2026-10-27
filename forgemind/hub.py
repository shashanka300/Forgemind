"""The link between the agent and the human body carrying the phone.

* push()             - send a message to every connected phone (WebSocket)
* request_look()     - "point the phone at the drawer"; resolved by the
                       perception pipeline when that zone/object is seen.
                       A robot base would take this request instead.
* request_capture()  - ask the phone for a high-res close-up and wait for it
* seek()             - "where is X?" keeps looking: the first phone frame that
                       contains X gets it highlighted live
"""
import asyncio
import collections
import itertools
import threading
import time

SEEK_TTL_S = 120.0


class Hub:
    def __init__(self):
        self.clients = set()
        self.loop = None
        self.outbox = collections.deque(maxlen=100)    # also read by tests / polling
        self.looks = {}
        self.captures = {}
        self.seeks = {}              # entity id -> {"label", "until"}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    # -- push -----------------------------------------------------------------

    def attach_loop(self, loop):
        self.loop = loop

    def push(self, msg):
        msg = dict(msg, ts=time.time())
        self.outbox.append(msg)
        if not self.loop:
            return
        for ws in list(self.clients):
            fut = asyncio.run_coroutine_threadsafe(ws.send_json(msg), self.loop)
            fut.add_done_callback(lambda f, ws=ws: f.exception() and self.clients.discard(ws))

    def say(self, text):
        self.push({"type": "say", "text": text})

    # -- active perception ----------------------------------------------------

    def request_look(self, target, reason="", zone=None, entity_id=None):
        with self._lock:
            lid = next(self._ids)
            look = {"id": lid, "target": target, "zone": zone, "entity_id": entity_id,
                    "reason": reason, "created": time.time(), "status": "pending"}
            self.looks[lid] = look
        self.push({"type": "look_request", **look})
        return look

    def pending_looks(self):
        return [l for l in self.looks.values() if l["status"] == "pending"]

    def check_looks(self, zone, labels, entity_ids):
        """Called per phone frame; resolves requests the user has fulfilled."""
        done = []
        labels = {l.lower() for l in labels}
        for look in self.pending_looks():
            hit = ((look["zone"] and zone == look["zone"])
                   or (look["entity_id"] and look["entity_id"] in entity_ids)
                   or (look["target"] and look["target"].lower() in labels))
            if hit:
                look["status"] = "done"
                look["done"] = time.time()
                done.append(look)
                self.push({"type": "look_done", "id": look["id"], "target": look["target"]})
        return done

    def seek(self, entity_id, label, ttl=SEEK_TTL_S):
        """Highlight `entity_id` on the phone as soon as a phone frame contains it."""
        with self._lock:
            self.seeks[entity_id] = {"label": label, "until": time.time() + ttl}
        self.push({"type": "seeking", "entity_id": entity_id, "label": label, "ttl": ttl})

    def check_seeks(self, boxes_by_entity, now=None):
        """Called per phone frame with {entity_id: box}; returns the entities found."""
        now = now or time.time()
        found = []
        with self._lock:
            for eid in list(self.seeks):
                s = self.seeks[eid]
                if now > s["until"]:
                    del self.seeks[eid]
                elif eid in boxes_by_entity:
                    del self.seeks[eid]
                    found.append({"entity_id": eid, "label": s["label"],
                                  "box": boxes_by_entity[eid]})
        for f in found:
            self.push({"type": "highlight", **f, "found": True})
            self.say(f"There's your {f['label']}.")
        return found

    def dismiss_look(self, lid):
        if lid in self.looks:
            self.looks[lid]["status"] = "dismissed"

    # -- close-up capture -----------------------------------------------------

    def request_capture(self, target, timeout=30.0):
        """Ask the phone for a close-up; block until it arrives or times out."""
        with self._lock:
            cid = next(self._ids)
            ev = threading.Event()
            self.captures[cid] = {"event": ev, "data": None, "target": target}
        self.push({"type": "capture_request", "id": cid, "target": target,
                   "timeout": timeout})
        ev.wait(timeout)
        item = self.captures.pop(cid, None)
        return item["data"] if item else None

    def deliver_capture(self, cid, data):
        item = self.captures.get(cid)
        if not item:
            return False
        item["data"] = data
        item["event"].set()
        return True
