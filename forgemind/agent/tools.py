"""Agent tools (PRD section 9).

Each tool has a pydantic argument model (its JSON schema is what the model
sees) and an implementation. The model proposes; `ToolBox.run` validates the
arguments and the current state before anything executes, and every result
carries a one-line `summary` that can be spoken as-is.

The embodied tools (go_look_at, read_closeup, highlight) talk to the Hub.
Today a person carrying the phone fulfils them; a robot base or arm can
implement the same requests later without changing the agent.
"""
import json
import re
import threading
import time
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from forgemind import reports
from forgemind.memory.registry import TeachConflict
from forgemind.providers.openai_compat import image_part

READ_PROMPT = """You read instrument displays from a photo. Target: {target}.
Answer with JSON only: {{"value": number or null, "unit": string or null,
"text": the exact characters on the display, "confidence": 0..1,
"problem": null or a short reason when value is null}}.
value is null when: the display is not readable, it is blank or off, it shows
OL/overload or a dash, or more than one instrument display is visible (then say
so in "problem"). Only give a unit you can see on the display. Never guess digits."""
# Teaching takes the object from up to TEACH_VIEWS phone frames of the last
# TEACH_WINDOW_S seconds (the phone sends a frame every ~0.5-1 s).
TEACH_VIEWS = 3
TEACH_WINDOW_S = 4.0
# Tools whose `summary` needs the model to phrase an answer (it lists or
# compares things). Every other tool's summary is already the spoken reply, so
# the agent skips the second model call (PRD: spoken answer under 4 s).
NEEDS_PHRASING = {"inspect_scene", "compare_sessions"}


# -- argument models ----------------------------------------------------------

class WhereIs(BaseModel):
    query: str = Field(..., description="Object name or description, e.g. '2.5 mm hex key'")


class InspectScene(BaseModel):
    camera: Literal["static", "mobile", "both"] = "both"


class GoLookAt(BaseModel):
    target: str = Field(..., description="A zone name (bench, shelf, drawer...) or an object")
    reason: str = Field("", description="Short reason shown to the user")


class ReadCloseup(BaseModel):
    target: str = Field(..., description="What to read, e.g. 'multimeter display'")
    quantity: str | None = Field(None, description="Quantity to log the reading as")


class Highlight(BaseModel):
    query: str


class WatchBench(BaseModel):
    concept: str
    kind: Literal["unattended", "present"] = "unattended"
    seconds: float | None = Field(None, description="For 'unattended': seconds away before alerting")


class TeachObject(BaseModel):
    name: str = Field(..., description="Personal name for the object in front of the phone")
    concept: str | None = None


def name_said(name, text):
    """Did the user say at least half of the words of this name in `text`?"""
    def words(t):
        return {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", t.lower())
                if w not in ("my", "the", "a", "an", "s")}
    want = words(name)
    return not want or len(want & words(text)) * 2 >= len(want)


class CorrectObject(BaseModel):
    wrong_name: str = Field(..., description="The wrong label, e.g. 'baby's biter'")
    right_name: str = Field(..., description="What the object in front of the phone really is, e.g. 'mouse'")


class StartSession(BaseModel):
    name: str
    objective: str | None = None


class ResumeSession(BaseModel):
    name: str | None = Field(None, description="Session name; omit for the most recent")


class Empty(BaseModel):
    pass


class PlanSteps(BaseModel):
    steps: list[str] = Field(..., min_length=1)


class LogStep(BaseModel):
    text: str
    tentative: bool = Field(False, description="True for 'just checking, don't log that'")


class LogMeasurement(BaseModel):
    quantity: str
    value: float | None = Field(..., description="null if the user did not give a number")
    unit: str | None = None
    source: Literal["stated", "instrument"] = "stated"
    note: str | None = None


class SetConfig(BaseModel):
    key: str
    value: str


class CompareSessions(BaseModel):
    session_a: int | None = None
    session_b: int | None = None


class GenerateReport(BaseModel):
    session_name: str | None = None


class Alert(BaseModel):
    message: str
    severity: Literal["info", "warning", "critical"] = "info"


SPECS = {
    "where_is": (WhereIs, "Find where an object was last seen (zone, camera, time, confidence, keyframe). Highlights it if it is in the phone's view now."),
    "inspect_scene": (InspectScene, "What the bench camera and/or the phone see right now."),
    "go_look_at": (GoLookAt, "Ask the user to point the phone at a zone or object, e.g. when memory is stale or uncertain. Non-blocking; memory updates when they do."),
    "read_closeup": (ReadCloseup, "Ask for a close-up photo and read an instrument display or label from it. Logs the reading as a measurement if a session is active."),
    "highlight": (Highlight, "Highlight an object on the phone's live view, or show its last-seen card if it isn't in view."),
    "watch_bench": (WatchBench, "Start a safety watch when the user asks you to watch, keep an eye on or monitor something: 'unattended' (object on the bench while the user is away, e.g. the soldering iron) or 'present' (e.g. smoke). Not for finding objects."),
    "teach_object": (TeachObject, "Remember the object in front of the phone under a name the user says in THIS message ('this is my X'). Never for thanks, remarks or follow-ups."),
    "correct_object": (CorrectObject, "The user says an object you labelled is really something else ('that's not X, it's Y'): removes the wrong label from the object in front of the phone and teaches the right one."),
    "start_session": (StartSession, "Start a new experiment/work session (repeats carry over unfinished steps)."),
    "resume_session": (ResumeSession, "Resume a session: last state, done and pending steps, next step."),
    "end_session": (Empty, "End the active session."),
    "plan_steps": (PlanSteps, "Set the checklist of planned steps for the active session."),
    "log_step": (LogStep, "Record a step the user did (or a tentative one they asked not to log as done)."),
    "log_measurement": (LogMeasurement, "Record a measurement. Never invent values."),
    "set_config": (SetConfig, "Record a configuration value (component, setting)."),
    "compare_sessions": (CompareSessions, "Compare configuration and measurements between two sessions (default: current vs previous run)."),
    "generate_report": (GenerateReport, "Write the experiment report (Markdown) with provenance for every fact."),
    "alert": (Alert, "Send an alert to the phone. Only for safety or when the user asked for a reminder."),
}


# Not offered to the model in conversation: alerts come from the safety watch,
# and in a conversation the spoken reply already reaches the user. With
# reasoning_effort=low, Spark called `alert` for small talk ("Sent.").
AGENT_HIDDEN = {"alert"}


def schemas():
    out = []
    for name, (model, desc) in SPECS.items():
        if name in AGENT_HIDDEN:
            continue
        params = model.model_json_schema()
        params.pop("title", None)
        out.append({"type": "function",
                    "function": {"name": name, "description": desc, "parameters": params}})
    return out


class ToolBox:
    def __init__(self, ctx):
        self.ctx = ctx

    @property
    def mem(self):
        return self.ctx.memory

    # What the user said this turn (set by the agent); None for direct calls like the
    # Teach button, which name the object explicitly.
    utterance = None

    def run(self, name, arguments):
        if name not in SPECS:
            return {"error": f"unknown tool {name}"}
        model, _ = SPECS[name]
        try:
            args = model(**(arguments or {}))
        except ValidationError as e:
            return {"error": f"invalid arguments for {name}: {e.errors()[:3]}"}
        try:
            result = getattr(self, name)(args)
        except ToolError as e:
            return {"error": str(e), "summary": str(e)}
        self.mem.db.log_event("tool", "inferred", {"tool": name, "args": args.model_dump(),
                                                   "summary": result.get("summary")})
        return result

    def _session(self):
        s = self.mem.experiments.active()
        if not s:
            raise ToolError("There's no active session. Start or resume one first.")
        return s

    # -- perception & memory --------------------------------------------------

    def where_is(self, a: WhereIs):
        hits = self.mem.registry.find(a.query)
        if not hits:
            return {"found": False, "summary": f"I haven't seen anything like '{a.query}' yet."}
        top = hits[0]
        in_view = self._in_mobile_view(top["id"])
        if in_view:
            self.ctx.hub.push({"type": "highlight", "entity_id": top["id"], "label": top["label"],
                               "box": in_view["box"]})
            summary = f"The {top['label']} is right there in your view - I've highlighted it."
        else:
            self.ctx.hub.push({"type": "last_seen", "entity": top})
            summary = self.mem.registry.describe(top)
            if "match_score" in top:        # only looks a bit like it: say so
                summary = f"I don't know a {a.query} by name. The closest thing I've seen: " + summary
            if top["last_seen"]:
                self.ctx.hub.seek(top["id"], top["label"])
                if top["stale"] and top["zone"]:
                    # Memory is old: ask the user to look (PRD acceptance #5). The phone
                    # then either finds it (seek -> highlight) or records it as gone.
                    self.ctx.hub.request_look(top["zone"], f"checking for the {top['label']}",
                                              zone=top["zone"], entity_id=top["id"])
                    self.mem.db.log_event("look_request", "inferred",
                                          {"target": top["zone"], "zone": top["zone"],
                                           "reason": "stale memory", "entity": top["label"]})
                    summary += (f" Could you point the phone at the {top['zone']}"
                                " so I can check it's still there?")
                else:
                    near = f"get to the {top['zone']}" if top["zone"] else "are near it"
                    summary += f" When you {near}, point the phone around and I'll highlight it."
            # (describe() already says when the sighting is stale)
        return {"found": True, "results": hits, "in_view": bool(in_view), "summary": summary,
                "stale": top["stale"], "confidence": top["confidence"],
                "seeking": not in_view and bool(top["last_seen"]),
                "history": self.mem.registry.history(top["id"], 5)}

    def _in_mobile_view(self, eid):
        latest = self.ctx.perception.latest.get("mobile")
        if not latest or time.time() - latest["ts"] > 5:
            return None
        return next((d for d in latest["detections"] if d.get("entity_id") == eid), None)

    def inspect_scene(self, a: InspectScene):
        cams = ["static", "mobile"] if a.camera == "both" else [a.camera]
        scenes = [self.ctx.perception.scene(c) for c in cams]
        parts = []
        for sc in scenes:
            who = "bench camera" if sc["camera"] == "static" else "phone"
            if not sc["available"]:
                parts.append(f"The {who} isn't streaming.")
                continue
            objs = ", ".join(sc["objects"]) or "nothing I recognise"
            where = f" ({sc['zone']})" if sc.get("zone") else ""
            parts.append(f"The {who}{where} sees {objs}.")
        return {"scenes": scenes, "summary": " ".join(parts)}

    def go_look_at(self, a: GoLookAt):
        zones = {z.lower() for z in self.ctx.settings.zones.values()}
        target = a.target.lower().removeprefix("the ")
        zone = target if target in zones else None
        eid = None
        if not zone:
            hits = self.mem.registry.find(target, limit=1)
            if hits:
                eid, zone = hits[0]["id"], hits[0]["zone"]
        look = self.ctx.hub.request_look(a.target, a.reason, zone=zone, entity_id=eid)
        self.mem.db.log_event("look_request", "inferred", {"target": a.target, "zone": zone,
                                                           "reason": a.reason})
        where = f" (the {zone})" if zone and zone != target else ""
        return {"look_id": look["id"], "zone": zone,
                "summary": f"Could you point the phone at the {target}{where}?"}

    def read_closeup(self, a: ReadCloseup):
        data = self.ctx.hub.request_capture(a.target, timeout=self.ctx.capture_timeout)
        if data is None:
            return {"value": None, "summary": f"I didn't get a close-up of the {a.target}."}
        img = self.ctx.perception.decode(data)
        ts = time.time()
        keyframe = self.ctx.perception._save_keyframe("mobile", img, ts, force=True)
        llm = self.ctx.llm
        reading = {"value": None, "unit": None, "text": None, "confidence": 0}
        if getattr(llm, "supports_vision", False):
            res = llm.chat([{"role": "user", "content": [
                {"type": "text", "text": READ_PROMPT.format(target=a.target)},
                image_part(img, max_px=1600, quality=90)]}],
                model=llm.model_for("worker"), json_mode=True)
            try:
                reading.update(json.loads(res.content or "{}"))
            except json.JSONDecodeError:
                pass
        if reading.get("value") is None:
            why = ("reading displays needs a vision model (mock mode)"
                   if not getattr(llm, "supports_vision", False)
                   else reading.get("problem") or "I couldn't read it clearly")
            return {"value": None, "keyframe": keyframe,
                    "summary": f"I got the close-up, but {why}. Tell me the value and I'll log it."}
        out = {"reading": reading, "keyframe": keyframe}
        s = self.mem.experiments.active()
        val = f"{reading['value']} {reading.get('unit') or ''}".strip()
        if s:
            q = a.quantity or a.target
            out.update(self.mem.experiments.log_measurement(
                s["id"], q, float(reading["value"]), reading.get("unit"), "instrument", keyframe,
                note=f"display read '{reading.get('text')}' conf {reading.get('confidence')}"))
            out["summary"] = f"The {a.target} reads {val}. Logged as {q}, read from the display."
        else:
            out["summary"] = f"The {a.target} reads {val}."
        return out

    def highlight(self, a: Highlight):
        hits = self.mem.registry.find(a.query, limit=1)
        if not hits:
            return {"summary": f"I don't know the {a.query} yet."}
        top = hits[0]
        d = self._in_mobile_view(top["id"])
        if d:
            self.ctx.hub.push({"type": "highlight", "entity_id": top["id"], "label": top["label"],
                               "box": d["box"]})
            return {"in_view": True, "summary": f"Highlighted the {top['label']}."}
        self.ctx.hub.push({"type": "last_seen", "entity": top})
        return {"in_view": False, "entity": top,
                "summary": "It's not in view. " + self.mem.registry.describe(top)}

    def watch_bench(self, a: WatchBench):
        w = self.ctx.safety.add_watch(a.kind, a.concept, a.seconds)
        if a.kind == "unattended":
            s = f"Watching the {a.concept}: I'll alert you if it's on the bench and you're away for {int(w['seconds'])} seconds."
        else:
            s = f"I'll alert you if I see {a.concept} on the bench."
        return {"watch": w, "summary": s}

    def _teach_target(self, frame, concept):
        """(box, concept, how) of the object being shown in one phone frame.

        Prefer a detection near the centre (of the named concept if given);
        if the frame has none, ask the segmenter for the concept directly;
        the centre of the view is the last resort.
        """
        objs = [d for d in frame["detections"] if d.get("kind") == "object"]
        if concept:
            objs = [d for d in objs if d["concept"] == concept] or objs
        if objs:
            d = min(objs, key=_off_centre)
            return d["box"], concept or d["concept"], "detected"
        if concept:
            try:
                dets = self.ctx.perception.seg.segment(frame["image"], [concept])
            except Exception:
                dets = []
            if dets:
                d = min(dets, key=lambda d: _off_centre({"box": d.box}))
                return d.box, concept, "asked"
        return (0.25, 0.25, 0.75, 0.75), concept or "object", "centre"

    def teach_object(self, a: TeachObject):
        """Name the object in front of the phone, from up to TEACH_VIEWS recent frames."""
        p = self.ctx.perception
        frames = p.recent_frames(TEACH_WINDOW_S)
        if not frames:
            if p.current_image("mobile") is None:
                raise ToolError("Point the phone's camera at the object first.")
            frames = [p.latest["mobile"]]
        frames = frames[:TEACH_VIEWS]
        if self.utterance is not None and not name_said(a.name, self.utterance):
            # Phone test: "thanks for locating it" re-taught whatever was in view.
            raise ToolError(f"To teach the {a.name}, hold it up and say its name.")
        crops, hows, boxes, concept = [], [], [], a.concept
        for f in frames:
            box, c, how = self._teach_target(f, concept)
            if how == "centre":             # the middle of the view may be the background
                continue
            concept = concept or c
            crops.append(p.crop(f["image"], box))
            hows.append(how)
            boxes.append(box)
        if not crops:
            raise ToolError(f"I can't pick out the {a.name}. Hold it closer, in the middle of "
                            "the view, and say its name again.")
        keyframe = p._save_keyframe("mobile", frames[0]["image"], time.time(), force=True)
        try:
            v = self.mem.registry.teach(a.name, crops, concept or "object", keyframe)
        except TeachConflict as e:
            raise ToolError(f"That looks like your {e.label}. If it's really the {a.name}, "
                            f"say: that's not the {e.label}, it's the {a.name}.")
        self.mem.registry.saw_taught(v["id"], p.phone_zone.current(time.time()), boxes[0], keyframe)
        summary = f"Got it - that's your {a.name} ({len(crops)} view{'s' * (len(crops) > 1)})."
        if v.get("examples", 0) < 3:
            summary += " Turn it a little and say it again so I can recognise it from other angles."
        return {"entity": v, "views": hows, "summary": summary}

    def correct_object(self, a: CorrectObject):
        """Move the object in front of the phone from a wrong label to the right one."""
        p, reg = self.ctx.perception, self.mem.registry
        latest = p.latest.get("mobile")
        if not latest or time.time() - latest["ts"] > 5:
            raise ToolError(f"Point the phone at the {a.right_name} and tell me again.")
        objs = [d for d in latest["detections"] if d.get("kind") == "object"]
        wrong = " ".join(a.wrong_name.lower().split())
        labelled = [d for d in objs if (d.get("label") or "").lower() == wrong]
        if not (labelled or objs):
            raise ToolError(f"I can't pick out the {a.right_name}. Hold it in the middle of the view.")
        target = min(labelled or objs, key=_off_centre)
        crop = p.crop(latest["image"], target["box"])
        row = self.mem.db.one("SELECT id FROM entities WHERE deleted=0 AND lower(name)=?", (wrong,))
        removed = reg.unlearn(row["id"], crop) if row else 0
        v = reg.teach(a.right_name, [crop], target["concept"], latest.get("keyframe"), force=True)
        reg.saw_taught(v["id"], p.phone_zone.current(time.time()), target["box"], latest.get("keyframe"))
        return {"entity": v, "removed_views": removed,
                "summary": f"Fixed: that's your {a.right_name}, not the {a.wrong_name}."}

    # -- sessions -------------------------------------------------------------

    def start_session(self, a: StartSession):
        s = self.mem.experiments.start(a.name, a.objective)
        st = self.mem.experiments.state(s["id"])
        extra = f" I carried over {len(st['pending'])} unfinished steps; next is: {st['next']}." if st["pending"] else ""
        return {"state": st, "summary": f"Started session '{a.name}'.{extra}"}

    def resume_session(self, a: ResumeSession):
        st = self.mem.experiments.resume(a.name)
        if not st:
            raise ToolError(f"I can't find a session called '{a.name}'." if a.name
                            else "There are no sessions yet.")
        name = st["session"]["name"]
        bits = [f"Resuming '{name}', last active {st['last_activity']}."]
        if st["done"]:
            bits.append(f"Done: {len(st['done'])} steps, last was '{st['done'][-1]}'.")
        if st["next"]:
            bits.append(f"Next up: {st['next']}.")
        elif not st["done"]:
            bits.append("No steps recorded yet.")
        else:
            bits.append("No pending steps.")
        if st["measurements"]:
            m = st["measurements"][-1]
            if m["value"] is not None:
                bits.append(f"Last measurement: {m['quantity']} = {m['value']:g} {m['unit'] or ''}.".replace(" .", "."))
        if st["missing"]:
            bits.append("Missing: " + "; ".join(st["missing"]) + ".")
        notes = [n for n in self.mem.notes.recent(20)
                 if n["session_id"] == st["session"]["id"] and n["kind"] in ("goal", "obstacle")]
        if notes:
            bits.append("You mentioned: " + "; ".join(n["text"] for n in notes[:2]) + ".")
        return {"state": st, "missing": st["missing"], "notes": notes, "summary": " ".join(bits)}

    def end_session(self, a: Empty):
        s = self.mem.experiments.end()
        if not s:
            raise ToolError("There's no active session.")
        return {"session": s, "summary": f"Ended session '{s['name']}'."}

    def plan_steps(self, a: PlanSteps):
        s = self._session()
        self.mem.experiments.plan(s["id"], a.steps)
        return {"summary": f"Added {len(a.steps)} planned steps. First: {a.steps[0]}."}

    def log_step(self, a: LogStep):
        s = self._session()
        kf = (self.ctx.perception.latest.get("static") or {}).get("keyframe")
        r = self.mem.experiments.log_step(s["id"], a.text, a.tentative, keyframe=kf)
        if a.tentative:
            return {**r, "summary": "Okay, noted as just a check - not logged as a step."}
        if r["matched_plan"]:
            nxt = self.mem.experiments.state(s["id"])["next"]
            tail = f" Next: {nxt}." if nxt else " That was the last planned step."
            return {**r, "summary": f"Marked '{r['matched_plan']}' done.{tail}"}
        return {**r, "summary": f"Logged: {a.text}."}

    def log_measurement(self, a: LogMeasurement):
        s = self._session()
        r = self.mem.experiments.log_measurement(s["id"], a.quantity, a.value, a.unit, a.source,
                                                 note=a.note)
        if a.value is None:
            return {**r, "summary": f"I logged {a.quantity} as missing - tell me the value when you have it."}
        summary = f"Logged {a.quantity} = {a.value:g} {a.unit or ''}".rstrip() + "."
        if r.get("already_logged"):
            summary = summary.replace("Logged", "Already logged", 1)
        done = self.mem.experiments.complete_measure_step(s["id"], a.quantity, a.source)
        if done:
            nxt = self.mem.experiments.state(s["id"])["next"]
            summary += f" Marked '{done}' done." + (f" Next: {nxt}." if nxt else "")
        return {**r, "completed_step": done, "summary": summary}

    def set_config(self, a: SetConfig):
        s = self._session()
        r = self.mem.experiments.set_config(s["id"], a.key, a.value)
        return {**r, "summary": f"Noted: {a.key} is {a.value}."}

    def compare_sessions(self, a: CompareSessions):
        c = self.mem.experiments.compare(a.session_a, a.session_b)
        if not c or not c.get("previous"):
            raise ToolError("There's no earlier session to compare with.")
        bits = [f"Compared with '{c['previous']['name']}':"]
        for ch in c["config_changes"][:4]:
            bits.append(f"{ch['key']} changed from {ch['previous']} to {ch['current']}.")
        for m in c["measurements"]:
            if "delta" in m:
                bits.append(f"{m['quantity']} went from {m['previous']:g} to {m['current']:g} {m['unit'] or ''}.".replace(" .", "."))
            elif m["current"] is None:
                bits.append(f"{m['quantity']} hasn't been measured this time.")
        if len(bits) == 1:
            bits.append("no differences recorded yet.")
        missing = [m["quantity"] for m in c["measurements"] if m["current"] is None]
        return {"comparison": c, "missing": missing, "summary": " ".join(bits)}

    def generate_report(self, a: GenerateReport):
        s = self.mem.experiments.find(a.session_name) if a.session_name else (
            self.mem.experiments.active() or self.mem.experiments.find(None))
        if not s:
            raise ToolError("There's no session to report on.")
        missing = self.mem.experiments.gaps(s["id"])
        flags = (f" It flags {len(missing)} missing value{'s' * (len(missing) > 1)}."
                 if missing else "")

        def write():
            r = reports.generate(self.mem.experiments, self.ctx.llm, s["id"], self.mem.reports_dir)
            self.ctx.hub.push({"type": "report", "url": r["url"], "name": s["name"]})
            return r

        if getattr(self.ctx.settings, "report_async", False):
            # The model-written summary takes ~8 s (Spark reasons ~1,300 tokens):
            # answer now, deliver the report card when it's done.
            def background():
                try:
                    write()
                    self.ctx.hub.say(f"The report for {s['name']} is ready on your phone.")
                except Exception as e:
                    self.ctx.hub.say(f"Sorry, the report failed: {e}")
            threading.Thread(target=background, daemon=True, name="report").start()
            return {"url": None, "missing": missing, "pending": True,
                    "summary": f"Writing the report for '{s['name']}' now; it'll be on your phone"
                               f" in a few seconds.{flags}"}
        r = write()
        return {"url": r["url"], "missing": missing,
                "summary": f"The report for '{s['name']}' is ready on your phone.{flags}"}

    def alert(self, a: Alert):
        self.ctx.hub.push({"type": "alert", "severity": a.severity, "message": a.message})
        self.mem.db.log_event("alert", "inferred", {"message": a.message, "severity": a.severity})
        return {"summary": "Sent."}


def _off_centre(d):
    b = d["box"]
    return abs((b[0] + b[2]) / 2 - 0.5) + abs((b[1] + b[3]) / 2 - 0.5)


class ToolError(Exception):
    pass
