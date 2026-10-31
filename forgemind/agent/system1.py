"""System 1: answer simple turns without Muse Spark.

A fine-tuned Laya decision model (eval/system1_train.py) picks the tool in ~20 ms; the
arguments come from closed choices (objects in memory, session names) or plain parsing
(a measurement's number and unit). Anything compound, uncertain, or outside FAST_TOOLS
returns None and goes to the Spark agent as before. Measured on hand-written turns
(eval/system1_eval.py) and on real speech (eval/real_speech.py).

Writes made on the fast path are re-checked by Spark in the background (orchestrator), and
undone with a spoken correction if Spark disagrees; `undo` here reverses each one.
"""
import os
import re
import threading
import time
from dataclasses import dataclass, field

# Tools whose arguments can be filled without generating text. The rest
# (plan_steps, log_step, set_config, start_session, teach_object, ...) need Spark.
# resume_session is out: on real speech (phone test, Oct 31) remarks like "I'm getting back
# to work" and "no I was asking about ..." scored 0.98 as resume.
FAST_TOOLS = {"where_is", "highlight", "read_closeup", "end_session", "watch_bench",
              "log_measurement"}
# Laya is confident even on sentences that aren't commands, so a fast action also needs the
# words of that request. Anything else goes to Spark.
INTENT = {
    "where_is": re.compile(r"\b(where|find|locate|seen|lost|missing)\b", re.I),
    "highlight": re.compile(r"\b(highlight|mark|point out|show me)\b", re.I),
    "read_closeup": re.compile(r"\b(read|reading|say|says|display|screen|showing)\b", re.I),
    "end_session": re.compile(r"\b(end|close|stop|finish|wrap up|done with)\b", re.I),
    "watch_bench": re.compile(r"\b(watch|monitor|keep an eye|warn|alert|let me know|tell me if|shout if)\b", re.I),
    "log_measurement": re.compile(r"\d"),
}
# "this is my X" names the object in hand: taught at once, while it is still being held.
_TEACH = re.compile(r"^\W*(?:(?:so|okay|ok|alright|yeah|yes|and|hey|right|now|um|uh)\W+)*"
                    r"(?:(?:this|that|these|those)(?:'s|\s+is|\s+are)|here(?:'s|\s+is|\s+are)|"
                    r"(?:the\s+)?next\s+(?:one\s+|object\s+)?is|this\s+one\s+is)\s+(?:my|the|our|a|an)\s+"
                    r"(?P<name>[\w'\- ]{2,40}?)\W*$", re.I)
_NOT_A_NAME = re.compile(r"\b(?:which|that|where|when|because|for|on|in|at|and|with|behind|i|it|to|"
                         r"of|problem|mistake|issue|question|joke|session|experiment|test|bit|"
                         r"thing|way|point|reason|idea|one)\b", re.I)


_SWITCH = re.compile(r"\b(?:switch|change|go|move)\b.*\b(?:to|back to)\s+(?:the\s+|my\s+)?"
                     r"(?P<cam>desk|bench|mounted|fixed|static|phone|mobile)\b", re.I)


def switch_target(text):
    """"desk" or "phone" if this sentence asks to change cameras, else None."""
    m = _SWITCH.search(text)
    return (None if not m else "phone" if m["cam"].lower() in ("phone", "mobile") else "desk")


def teach_name(text):
    """The object's name in a plain "this is my X" sentence, else None."""
    m = _TEACH.match(text)
    if not m or len(m["name"].split()) > 4 or _NOT_A_NAME.search(m["name"]):
        return None
    return m["name"]


# Corrections and pushback need thinking: "no I was asking about ...", "that's not ..."
PUSHBACK = re.compile(r"^\W*(?:(?:uh|um|see|so|well)\W+)*(?:no|nope|not|wrong)\b|\b(?:that's|it's|that is|it is) not\b", re.I)
# Fast actions that change memory: Spark re-checks them after the reply.
WRITES = {"log_measurement", "end_session", "watch_bench"}
MULTI_MAX = 0.5
ARG_MIN_CONF = 0.8
OTHER = "something else"

TOOL_CRITERIA = {
    "where_is": "find where an object is or was last seen", "inspect_scene": "describe what the cameras see right now",
    "go_look_at": "go and look at a place or object with the phone", "read_closeup": "read an instrument display or meter",
    "highlight": "highlight or point out an object on the phone screen",
    "watch_bench": "watch or monitor something for safety while the user is away",
    "teach_object": "remember the object in front of the phone under a name",
    "start_session": "start a new experiment or work session", "resume_session": "continue a previous session, or say what's next",
    "end_session": "end or close the current session", "plan_steps": "list the planned steps for the session",
    "log_step": "the user reports a step they did", "log_measurement": "the user reports a measured value",
    "set_config": "the user sets or states a configuration value or component",
    "compare_sessions": "compare this run with an earlier run", "generate_report": "write the experiment report",
    "chat": "small talk or a general question that needs no workshop action"}
QUESTIONS = {
    "tool": {"type": "choice", "instructions": "Which action should the workshop assistant take for what the user just said?",
             "criteria": TOOL_CRITERIA},
    "multi": {"type": "noul", "instructions": "Does the user ask for two or more different actions in this message?"},
}


@dataclass
class FastAction:
    tool: str
    args: dict
    conf: float
    detail: dict = field(default_factory=dict)


# -- measurements -------------------------------------------------------------

# spoken unit -> (unit as logged, quantity kind, scale to the base unit)
UNITS = [
    (r"millivolts?|mv", "mV", "voltage", 1e-3), (r"kilovolts?|kv", "kV", "voltage", 1e3),
    (r"volts?|v", "V", "voltage", 1),
    (r"milliamps?|milliamperes?|ma", "mA", "current", 1e-3), (r"microamps?|ua|µa", "µA", "current", 1e-6),
    (r"amps?|amperes?|a", "A", "current", 1),
    (r"megohms?|m(?:ega)?\s?ohms?|mω", "MΩ", "resistance", 1e6),
    (r"kilohms?|k\s?ohms?|kω|k", "kΩ", "resistance", 1e3), (r"ohms?|ω", "Ω", "resistance", 1),
    (r"degrees?(?: c(?:elsius)?)?|°c|celsius", "°C", "temperature", 1),
    (r"megahertz|mhz", "MHz", "frequency", 1e6), (r"kilohertz|khz", "kHz", "frequency", 1e3),
    (r"hertz|hz", "Hz", "frequency", 1),
    (r"milliwatts?|mw", "mW", "power", 1e-3), (r"watts?|w", "W", "power", 1),
]
_UNIT_RE = "|".join(f"(?P<u{i}>{p})" for i, (p, *_rest) in enumerate(UNITS))
_NUM_UNIT = re.compile(r"(?<![\w.])(?P<num>-?\d+(?:\.\d+)?|-?\.\d+)\s*(?:" + _UNIT_RE + r")(?![a-zµω])", re.I)
_ANY_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?|(?<![\w.])-?\.\d+")
_KINDS = ("voltage", "current", "resistance", "temperature", "frequency", "power")
_SUBJECT = re.compile(r"^(?:so |okay |ok |um |uh |and )*(?:the |my |our )?(?P<s>[a-z0-9 ]+?)\s+"
                      r"(?:is|was|reads|reading|measures|measured|came out at|comes out at|shows)\b", re.I)
_PLACE = re.compile(r"\b(?:on|at|across|of|through|from)\s+(?:the\s+)?(?P<p>[a-z0-9 ]{1,30}?)\s*(?:[.,!?]|$)", re.I)
_NOT_SUBJECT = {"i", "it", "that", "this", "there", "we", "got", "meter", "multimeter", "dmm", "reading"}


def parse_measurement(text):
    """(quantity, value, unit) from 'I measured 3.3 volts on the output', or None.

    Exactly one number, and it must carry a unit: anything else is left to Spark.
    """
    t = text.strip()
    hits = list(_NUM_UNIT.finditer(t))
    if len(hits) != 1 or len(_ANY_NUM.findall(t)) != 1:
        return None
    m = hits[0]
    i = next(int(k[1:]) for k, v in m.groupdict().items() if k != "num" and v)
    unit, kind, _scale = UNITS[i][1:]
    value = float(m.group("num"))
    subject = (_SUBJECT.match(t[:m.start()]) or None) and _SUBJECT.match(t[:m.start()]).group("s").strip().lower()
    place = _PLACE.search(t[m.end():])
    place = place.group("p").strip().lower() if place else None
    if subject and subject.split()[-1] not in _NOT_SUBJECT and subject not in _NOT_SUBJECT:
        quantity = subject if any(k in subject for k in _KINDS) else f"{subject} {kind}"
    elif place:
        quantity = f"{place} {kind}"
    else:
        quantity = kind
    return quantity, value, unit


def base_value(value, unit):
    """Value in the base unit (9.8 kΩ -> 9800), for comparing with Spark's version."""
    if value is None:
        return None
    u = (unit or "").strip()
    for p, symbol, _k, scale in UNITS:
        if u == symbol or re.fullmatch(p, u.lower()):
            return value * scale
    return value


# A question about a value ("is 68 degrees too hot?") isn't a reading to log.
_QUESTION = re.compile(r"\?\s*$|^\W*(?:um\s+|so\s+)?(?:what|why|how|is|are|should|could|can|would|does|do|will)\b"
                       r"|\btoo (?:high|low|hot|cold|much)\b|\b(?:normal|okay|ok|safe) for\b", re.I)
_NAMES_SESSION = re.compile(r"\b(?:session|experiment|job|run|test)\b", re.I)
_SECONDS = re.compile(r"(\d+)\s*(seconds?|secs?|minutes?|mins?)\b", re.I)


# A decision made on a partial transcript is reused for this long: memory (objects,
# sessions) can change after that.
PREFETCH_TTL_S = 10.0


def _norm(text):
    return " ".join(re.sub(r"[^a-z0-9.' ]", " ", text.lower()).split())


class System1:
    def __init__(self, settings, memory, backend=None):
        self.s, self.mem = settings, memory
        self.backend = backend
        self.min_conf = settings.system1_min_conf
        self._lock = threading.Lock()
        self.loading = False
        self._cache = {}                     # normalised text -> (ts, FastAction | None, reason, decision)

    def _predict(self, text, questions):
        with self._lock:
            if self.backend is None:
                os.environ.setdefault("USE_TF", "0")
                import laya
                self.backend = laya.load(self.s.system1, device="cuda")
            return self.backend.predict(text, questions)["answers"]

    def warm_up(self):
        self.loading = True
        try:
            self._predict("where is my multimeter?", QUESTIONS)
        finally:
            self.loading = False

    def _choose(self, text, instructions, options):
        q = {"x": {"type": "choice", "instructions": instructions, "criteria": options}}
        a = self._predict(text, q)["x"]
        return a["choice"], a["answer_confidence"]

    def _object(self, text, extra=()):
        labels = list(dict.fromkeys([e["label"] for e in self.mem.registry.list(60) if e.get("label")]
                                    + list(extra)))
        if not labels:
            return None
        crit = {l: l for l in labels}
        crit[OTHER] = "an object not in this list, or no object"
        choice, conf = self._choose(text, "Which object is the user talking about?", crit)
        return choice if choice != OTHER and conf >= ARG_MIN_CONF else None

    def prefetch(self, text):
        """Decide on a partial transcript while the user is still talking.

        The final transcript usually equals the last partial; `decide` then reuses this
        result instead of running the model again.
        """
        if self.loading or not text.strip():
            return
        key = _norm(text)
        if key in self._cache:
            return
        act = self._decide(text)
        self._cache[key] = (time.time(), act, self.last_reason, getattr(self, "last_decision", None))
        if len(self._cache) > 64:
            for k in sorted(self._cache, key=lambda k: self._cache[k][0])[:32]:
                self._cache.pop(k, None)

    def decide(self, text):
        """FastAction, or None (with self.last_reason) when Spark should take the turn."""
        hit = self._cache.pop(_norm(text), None)
        if hit and time.time() - hit[0] < PREFETCH_TTL_S:
            _, act, self.last_reason, self.last_decision = hit
            self.prefetched = True
            return act
        self.prefetched = False
        return self._decide(text)

    def _decide(self, text):
        from forgemind.agent.orchestrator import is_compound
        self.last_reason = None
        if self.loading:                      # still loading at startup: Spark takes the turn
            self.last_reason = "loading"
            return None
        if is_compound(text.strip()):
            self.last_reason = "compound"
            return None
        cam = switch_target(text)
        if cam:
            self.last_decision = {"tool": "switch_camera", "conf": 1.0, "multi": 0.0}
            return FastAction("switch_camera", {"camera": cam}, 1.0, dict(self.last_decision))
        name = teach_name(text)
        if name:
            self.last_decision = {"tool": "teach_object", "conf": 1.0, "multi": 0.0}
            return FastAction("teach_object", {"name": name}, 1.0, dict(self.last_decision))
        a = self._predict(text, QUESTIONS)
        tool, conf, multi = a["tool"]["choice"], a["tool"]["answer_confidence"], a["multi"]["noul"]
        self.last_decision = {"tool": tool, "conf": round(conf, 3), "multi": round(multi, 3)}
        if tool not in FAST_TOOLS or conf < self.min_conf or multi >= MULTI_MAX:
            self.last_reason = "not fast" if tool not in FAST_TOOLS else "unsure"
            return None
        if PUSHBACK.search(text) or not INTENT[tool].search(text):
            self.last_reason = "not a plain request"
            return None
        args = self._args(tool, text)
        if args is None:
            self.last_reason = "arguments"
            return None
        return FastAction(tool, args, conf, dict(self.last_decision))

    def _args(self, tool, text):
        low = text.lower()
        if tool in ("where_is", "highlight"):
            obj = self._object(text)
            return {"query": obj} if obj else None
        if tool == "read_closeup":
            return {"target": "multimeter display"}
        if tool == "end_session":
            # "thanks, that's all for now" may be a goodbye, not an ending: only when the
            # session itself is named (agent-eval: Spark disagreed and it was undone)
            return {} if _NAMES_SESSION.search(text) else None
        if tool == "watch_bench":
            secs = _SECONDS.search(low)
            seconds = None
            if secs:
                seconds = float(secs.group(1)) * (60 if secs.group(2).startswith("min") else 1)
            if "smoke" in low:
                return {"concept": "smoke", "kind": "present"}
            obj = self._object(text, extra=[self.s.default_watch])
            return {"concept": obj, "kind": "unattended", "seconds": seconds} if obj else None
        if tool == "log_measurement":
            p = None if _QUESTION.search(text) else parse_measurement(text)
            if not p:
                return None
            quantity, value, unit = p
            return {"quantity": quantity, "value": value, "unit": unit, "source": "stated"}
        return None


# -- undo (when Spark disagrees with a fast write) ---------------------------------

def undo(ctx, action, out):
    """Reverse a fast write; returns a short phrase for the spoken correction."""
    db, exp = ctx.memory.db, ctx.memory.experiments
    if action.tool == "log_measurement":
        if out.get("already_logged"):
            return None
        row = db.one("SELECT session_id FROM measurements WHERE id=?", (out["measurement_id"],))
        db.execute("DELETE FROM measurements WHERE id=?", (out["measurement_id"],))
        db.execute("DELETE FROM events WHERE id=?", (out["event_id"],))
        if out.get("completed_step") and row:
            db.execute("UPDATE steps SET status='planned', event_id=NULL WHERE session_id=? AND text=?"
                       " AND status='done'", (row["session_id"], out["completed_step"]))
        return f"I removed {action.args['quantity']}"
    if action.tool == "end_session":
        exp.resume(out["session"]["name"])
        return f"the session '{out['session']['name']}' is open again"
    if action.tool == "watch_bench":
        db.execute("UPDATE watches SET active=0 WHERE id=?", (out["watch"]["id"],))
        return f"I stopped watching the {action.args['concept']}"
    return None
