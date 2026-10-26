"""Experiment memory: sessions, planned/done steps, measurements, configuration.

Every write also appends an event, and the event id is the provenance
handle ("E42") that reports cite next to each fact.
"""
import re
import time

from forgemind.memory.registry import fmt_ago


def _words(s):
    return set(re.findall(r"[a-z0-9.]+", (s or "").lower())) - {"the", "a", "an", "to", "of"}


class Experiments:
    def __init__(self, db):
        self.db = db

    # -- sessions -------------------------------------------------------------

    def active(self):
        return self.db.one("SELECT * FROM sessions WHERE status='active'"
                           " ORDER BY started_at DESC LIMIT 1")

    def get(self, sid):
        return self.db.one("SELECT * FROM sessions WHERE id=?", (sid,))

    def find(self, name):
        if not name:
            return self.db.one("SELECT * FROM sessions ORDER BY started_at DESC LIMIT 1")
        rows = self.db.all("SELECT * FROM sessions ORDER BY started_at DESC")
        want = _words(name)
        best, best_score = None, 0.0
        for r in rows:
            have = _words(r["name"])
            score = len(want & have) / (len(want) or 1)
            if score > best_score:
                best, best_score = r, score
        return best if best_score >= 0.5 else None

    def list(self, limit=50):
        return self.db.all("SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?", (limit,))

    def _pause_others(self, keep=None):
        self.db.execute("UPDATE sessions SET status='paused' WHERE status='active' AND id IS NOT ?",
                        (keep,))

    def start(self, name, objective=None):
        self._pause_others()
        now = time.time()
        sid = self.db.execute("INSERT INTO sessions(name,objective,started_at,status)"
                              " VALUES(?,?,?, 'active')", (name, objective, now))
        self.db.log_event("session_started", "stated", {"name": name, "objective": objective},
                          session_id=sid)
        prev = self.db.one("SELECT * FROM sessions WHERE id<>? AND lower(name)=lower(?)"
                           " ORDER BY started_at DESC LIMIT 1", (sid, name))
        if prev:  # carry the open checklist forward ("repeat yesterday's test")
            for st in self.db.all("SELECT text FROM steps WHERE session_id=? AND status='planned'"
                                  " ORDER BY ord", (prev["id"],)):
                self._insert_step(sid, st["text"], "planned", "carried")
        return self.get(sid)

    def resume(self, name=None):
        s = self.find(name) if name else (self.active() or self.find(None))
        if not s:
            return None
        self._pause_others(keep=s["id"])
        self.db.execute("UPDATE sessions SET status='active', ended_at=NULL WHERE id=?", (s["id"],))
        self.db.log_event("session_resumed", "stated", {"name": s["name"]}, session_id=s["id"])
        return self.state(s["id"])

    def end(self, sid=None):
        s = self.get(sid) if sid else self.active()
        if not s:
            return None
        self.db.execute("UPDATE sessions SET status='ended', ended_at=? WHERE id=?",
                        (time.time(), s["id"]))
        self.db.log_event("session_ended", "stated", {"name": s["name"]}, session_id=s["id"])
        return self.get(s["id"])

    # -- steps, measurements, configuration ------------------------------------

    def _insert_step(self, sid, text, status, source, keyframe=None):
        ev = self.db.log_event("step", source, {"text": text, "status": status}, session_id=sid)
        n = self.db.one("SELECT COUNT(*) AS n FROM steps WHERE session_id=?", (sid,))["n"]
        return self.db.execute(
            "INSERT INTO steps(session_id,ord,ts,text,status,source,event_id,keyframe)"
            " VALUES(?,?,?,?,?,?,?,?)", (sid, n, time.time(), text, status, source, ev, keyframe))

    def plan(self, sid, steps):
        return [self._insert_step(sid, t, "planned", "stated") for t in steps]

    def log_step(self, sid, text, tentative=False, source="stated", keyframe=None):
        """Mark a matching planned step done, or record a new one."""
        if not tentative:
            want = _words(text)
            for st in self.db.all("SELECT * FROM steps WHERE session_id=? AND status='planned'"
                                  " ORDER BY ord", (sid,)):
                have = _words(st["text"])
                if have and len(want & have) / len(have) >= 0.6:
                    ev = self.db.log_event("step", source, {"text": text, "status": "done",
                                                            "plan_step": st["id"]}, session_id=sid)
                    self.db.execute("UPDATE steps SET status='done', ts=?, event_id=?,"
                                    " keyframe=COALESCE(?, keyframe) WHERE id=?",
                                    (time.time(), ev, keyframe, st["id"]))
                    return {"step_id": st["id"], "matched_plan": st["text"], "event_id": ev}
        sid_ = self._insert_step(sid, text, "tentative" if tentative else "done", source, keyframe)
        ev = self.db.one("SELECT event_id FROM steps WHERE id=?", (sid_,))["event_id"]
        return {"step_id": sid_, "matched_plan": None, "event_id": ev}

    def log_measurement(self, sid, quantity, value, unit=None, source="stated",
                        keyframe=None, note=None):
        ev = self.db.log_event("measurement", source, {"quantity": quantity, "value": value,
                                                       "unit": unit, "note": note}, session_id=sid)
        mid = self.db.execute(
            "INSERT INTO measurements(session_id,ts,quantity,value,unit,source,event_id,keyframe,note)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (sid, time.time(), quantity, value, unit, source, ev, keyframe, note))
        return {"measurement_id": mid, "event_id": ev}

    def set_config(self, sid, key, value, source="stated"):
        ev = self.db.log_event("config", source, {"key": key, "value": value}, session_id=sid)
        self.db.execute("INSERT INTO configs(session_id,ts,key,value,event_id) VALUES(?,?,?,?,?)",
                        (sid, time.time(), key, str(value), ev))
        return {"event_id": ev}

    # -- reads ----------------------------------------------------------------

    def config(self, sid):
        out = {}
        for r in self.db.all("SELECT key,value,event_id FROM configs WHERE session_id=? ORDER BY ts",
                             (sid,)):
            out[r["key"]] = {"value": r["value"], "event_id": r["event_id"]}
        return out

    def measurements(self, sid):
        return self.db.all("SELECT * FROM measurements WHERE session_id=? ORDER BY ts", (sid,))

    def steps(self, sid):
        return self.db.all("SELECT * FROM steps WHERE session_id=? ORDER BY ord", (sid,))

    def state(self, sid):
        s = self.get(sid)
        steps = self.steps(sid)
        last = self.db.one("SELECT MAX(ts) AS t FROM events WHERE session_id=?", (sid,))["t"]
        pending = [st["text"] for st in steps if st["status"] == "planned"]
        return {
            "session": {"id": s["id"], "name": s["name"], "status": s["status"],
                        "objective": s["objective"]},
            "last_activity": fmt_ago(time.time() - last) if last else None,
            "done": [st["text"] for st in steps if st["status"] == "done"],
            "pending": pending,
            "tentative": [st["text"] for st in steps if st["status"] == "tentative"],
            "measurements": [{"quantity": m["quantity"], "value": m["value"], "unit": m["unit"],
                              "source": m["source"]} for m in self.measurements(sid)][-10:],
            "config": {k: v["value"] for k, v in self.config(sid).items()},
            "next": pending[0] if pending else None,
        }

    def compare(self, a=None, b=None):
        """Diff two sessions (default: active/latest vs the one before it)."""
        sa = self.get(a) if a else (self.active() or self.find(None))
        if not sa:
            return None
        if b:
            sb = self.get(b)
        else:
            sb = (self.db.one("SELECT * FROM sessions WHERE id<>? AND lower(name)=lower(?)"
                              " ORDER BY started_at DESC LIMIT 1", (sa["id"], sa["name"]))
                  or self.db.one("SELECT * FROM sessions WHERE started_at<? ORDER BY started_at"
                                 " DESC LIMIT 1", (sa["started_at"],)))
        if not sb:
            return {"current": sa["name"], "previous": None}
        ca, cb = self.config(sa["id"]), self.config(sb["id"])
        cfg = []
        for k in sorted(set(ca) | set(cb)):
            va, vb = ca.get(k, {}).get("value"), cb.get(k, {}).get("value")
            if va != vb:
                cfg.append({"key": k, "previous": vb, "current": va})

        def latest(sid):
            out = {}
            for m in self.measurements(sid):
                out[m["quantity"].lower()] = m
            return out
        ma, mb = latest(sa["id"]), latest(sb["id"])
        meas = []
        for q in sorted(set(ma) | set(mb)):
            x, y = ma.get(q), mb.get(q)
            row = {"quantity": q, "previous": y and y["value"], "current": x and x["value"],
                   "unit": (x or y)["unit"]}
            if x and y and x["value"] is not None and y["value"] is not None:
                row["delta"] = round(x["value"] - y["value"], 6)
            meas.append(row)
        return {"current": {"id": sa["id"], "name": sa["name"]},
                "previous": {"id": sb["id"], "name": sb["name"]},
                "config_changes": cfg, "measurements": meas}
