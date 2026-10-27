"""Experiment memory: sessions, planned/done steps, measurements, configuration.

Every write also appends an event, and the event id is the provenance
handle ("E42") that reports cite next to each fact.
"""
import re
import time

from forgemind.memory.registry import fmt_ago


STOP_WORDS = {"the", "a", "an", "to", "of", "i", "my", "on", "for"}


def _stem(w):
    """Crude suffix strip so 'wired'/'wire'/'wiring' and 'measured'/'measure' match."""
    if len(w) >= 4 and not any(c.isdigit() for c in w):
        w = re.sub(r"(ing|ed|es|e|s)$", "", w)
        if len(w) >= 4 and w[-1] == w[-2] and w[-1] not in "aeiouls":
            w = w[:-1]                                   # tinned -> tinn -> tin
    return w


def _words(s):
    return {_stem(w) for w in re.findall(r"[a-z0-9.]+", (s or "").lower())} - STOP_WORDS


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
        state = self.state(s["id"])          # before logging: "last active" means real work
        self._pause_others(keep=s["id"])
        self.db.execute("UPDATE sessions SET status='active', ended_at=NULL WHERE id=?", (s["id"],))
        self.db.log_event("session_resumed", "stated", {"name": s["name"]}, session_id=s["id"])
        return state

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
            # The model sometimes logs a step the code already completed (e.g. after a
            # measurement): don't record the same step twice within a minute.
            for st in self.db.all("SELECT * FROM steps WHERE session_id=? AND status='done'"
                                  " AND ts > ?", (sid, time.time() - 60)):
                have = _words(st["text"])
                if have and len(want & have) / len(have) >= 0.6:
                    return {"step_id": st["id"], "matched_plan": st["text"], "event_id": st["event_id"],
                            "already_done": True}
        sid_ = self._insert_step(sid, text, "tentative" if tentative else "done", source, keyframe)
        ev = self.db.one("SELECT event_id FROM steps WHERE id=?", (sid_,))["event_id"]
        return {"step_id": sid_, "matched_plan": None, "event_id": ev}

    def complete_measure_step(self, sid, quantity, source="stated"):
        """Mark the planned 'measure ...' step a measurement completes; returns its text.

        Done in code rather than left to the model: with reasoning_effort=low,
        Spark logged the value but skipped ticking off "measure the output".
        """
        want = _words(quantity) - _words("voltage current value reading")
        for st in self.db.all("SELECT * FROM steps WHERE session_id=? AND status='planned'"
                              " ORDER BY ord", (sid,)):
            words = _words(st["text"])
            if any(w.startswith("measur") or w in ("read", "check") for w in words) and want & words:
                self.log_step(sid, st["text"], source=source)
                return st["text"]
        return None

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

    def gaps(self, sid):
        """What the record is missing (PRD F-AGT-8), as short phrases.

        * a quantity whose latest measurement was logged without a value
        * a done "measure/read/check X" step with no measurement of X at all
        Planned steps aren't gaps: they just haven't happened yet.
        """
        missing = []
        latest = {}
        for m in self.measurements(sid):
            latest[m["quantity"]] = m
        for q, m in latest.items():
            if m["value"] is None:
                missing.append(f"{q} (logged without a value)")
        measured = [_words(q) for q in latest]
        for st in self.steps(sid):
            words = _words(st["text"])
            if st["status"] != "done" or not any(
                    w.startswith("measur") or w in ("read", "check") for w in words):
                continue
            topic = words - _words("measure measured read check value reading voltage current")
            if topic and not any(topic & q for q in measured):
                missing.append(f"a value for '{st['text']}' (step done, nothing logged)")
        return missing

    def state(self, sid):
        s = self.get(sid)
        steps = self.steps(sid)
        last = self.db.one("SELECT MAX(ts) AS t FROM events WHERE session_id=? AND type NOT IN"
                           " ('session_resumed', 'session_paused', 'session_ended')", (sid,))["t"]
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
            "missing": self.gaps(sid),
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
