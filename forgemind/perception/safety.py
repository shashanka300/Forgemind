"""Safety watch (PRD 8.8): rules evaluated against what the bench camera sees.

    unattended  - <concept> is on the bench and nobody has been there for
                  <seconds> (the soldering iron case)
    present     - <concept> is visible at all (smoke, flame)

Alerts go to every connected phone, wherever the user is. Assistive only:
this is not a certified safety system.
"""
import threading
import time

COOLDOWN_S = {"unattended": 180, "present": 60}
SEEN_WINDOW_S = {"unattended": 20, "present": 10}


class Safety:
    def __init__(self, settings, db, hub):
        self.s, self.db, self.hub = settings, db, hub
        self.last_present = time.time()
        self.last_seen = {}          # concept -> ts (static camera)
        self._lock = threading.Lock()
        if settings.default_watch and not db.one(
                "SELECT id FROM watches WHERE kind='unattended' AND concept=?",
                (settings.default_watch,)):
            self.add_watch("unattended", settings.default_watch, settings.unattended_seconds)
        for hazard in ("smoke",):
            if hazard in settings.concepts and not db.one(
                    "SELECT id FROM watches WHERE kind='present' AND concept=?", (hazard,)):
                self.add_watch("present", hazard, 0)

    def add_watch(self, kind, concept, seconds=None):
        if kind not in COOLDOWN_S:
            raise ValueError(f"unknown watch kind {kind}")
        seconds = self.s.unattended_seconds if seconds is None else seconds
        wid = self.db.execute("INSERT INTO watches(kind,concept,seconds,active,created_at)"
                              " VALUES(?,?,?,1,?)", (kind, concept, seconds, time.time()))
        return {"id": wid, "kind": kind, "concept": concept, "seconds": seconds}

    def watches(self):
        return self.db.all("SELECT * FROM watches WHERE active=1")

    def present(self, ts=None):
        with self._lock:
            self.last_present = max(self.last_present, ts or time.time())

    def seen(self, concept, ts=None):
        """`concept` is a detector concept or a taught object's name (case-insensitive)."""
        key = concept.lower().strip()
        with self._lock:
            self.last_seen[key] = max(self.last_seen.get(key, 0), ts or time.time())

    def user_away_for(self, now=None):
        return max(0.0, (now or time.time()) - self.last_present)

    def evaluate(self, now=None):
        now = now or time.time()
        fired = []
        for w in self.watches():
            kind, concept = w["kind"], w["concept"]
            seen = self.last_seen.get(concept.lower().strip(), 0)
            if now - seen > SEEN_WINDOW_S[kind]:
                continue
            if w["last_fired"] and now - w["last_fired"] < COOLDOWN_S[kind]:
                continue
            if kind == "unattended":
                away = self.user_away_for(now)
                if away < (w["seconds"] or 0):
                    continue
                msg = (f"The {concept} is on the {self.s.static_zone} and nobody has been "
                       f"there for {int(away // 60)} min {int(away % 60)} s.")
                severity = "warning"
            else:
                msg = f"I can see {concept} on the {self.s.static_zone}."
                severity = "critical"
            self.db.execute("UPDATE watches SET last_fired=? WHERE id=?", (now, w["id"]))
            self.db.log_event("alert", "observed", {"watch": w["id"], "message": msg,
                                                    "severity": severity},
                              camera="static", zone=self.s.static_zone, ts=now)
            alert = {"type": "alert", "severity": severity, "message": msg, "watch_id": w["id"]}
            self.hub.push(alert)
            fired.append(alert)
        return fired
