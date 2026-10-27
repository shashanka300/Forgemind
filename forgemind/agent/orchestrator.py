"""The agent loop: situation summary + conversation -> model -> validated tools -> reply.

Muse Wonder (FM_AGENT_MODEL) handles user turns; workers (FM_WORKER_MODEL)
are used inside tools for reading displays and writing reports.
"""
import json
import re
import threading
import time
from dataclasses import dataclass

from forgemind.agent import prompts
from forgemind.agent.tools import NEEDS_PHRASING, ToolBox, schemas
from forgemind.providers.openai_compat import image_part

MAX_STEPS = 6
# A reply that opens like a completed action ("Logged ...", "Started ...") with
# no tool call this turn is a false confirmation (local Glimmer did this 4/15
# times in agent-eval). Recall answers ("You measured 3.3 V") don't match.
ACTION_CLAIM = re.compile(
    r"^\W*(?:ok(?:ay)?|sure|done|got it)?[\s,.!-]*"
    r"(?:i(?:'ve| have)? )?(?:logged|started|noted|marked|recorded|saved|added|ended|stopped|"
    r"set |watching|taught|remembered|deleted|forgot|changed|updated)\b",
    re.I)
NUDGE = ("Your last reply said you did something, but you called no tool, so nothing "
         "happened. If the user asked for an action, call the tool now. If you were only "
         "answering from memory, reply again without claiming a new action.")
UNDONE = "Sorry, I didn't actually record that. Could you say it again?"
# More than one request in one utterance ("I measured 3.3 V. Write it up."):
# don't end the turn after the first tool's summary; let the model continue.
# Found by the simulated-user eval (the report request was dropped).
_ACTIONS = (r"(write|log|start|read|set|watch|plan|end|stop|teach|show|record|compare|"
            r"generate|make|measure|check|remember|look|highlight|resume|note)")
COMPOUND = re.compile(
    r"[.!?;]\s+(?:(?:please|can you|could you|now|then|and|also|ok(?:ay)?)[\s,]+)*" + _ACTIONS + r"\b"
    r"|\b(and then|after that)\b"
    r"|,?\s+(?:and|then|also)\s+(?:please\s+|can you\s+)?" + _ACTIONS + r"\b",
    re.I)
_TAIL_REQUEST = re.compile(r",\s*(?:can you|could you|please)\s+" + _ACTIONS + r"\b", re.I)


def is_compound(text):
    """Two requests in one utterance? ("I measured 5 V, can you write it up?")"""
    if COMPOUND.search(text):
        return True
    m = _TAIL_REQUEST.search(text)          # but "Hey, can you start a session" is one request
    return bool(m) and len(re.findall(r"[a-z]{2,}", text[:m.start()].lower())) >= 2


HISTORY_TURNS = 8


@dataclass
class Context:
    settings: object
    memory: object
    perception: object
    safety: object
    hub: object
    llm: object
    capture_timeout: float = 30.0


class Agent:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.tools = ToolBox(ctx)
        self.schemas = schemas()
        self._lock = threading.Lock()

    def _history(self):
        rows = self.ctx.memory.db.all(
            "SELECT role, content FROM conversation ORDER BY id DESC LIMIT ?",
            (HISTORY_TURNS * 2,))
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def _remember(self, role, content):
        self.ctx.memory.db.execute("INSERT INTO conversation(ts,role,content) VALUES(?,?,?)",
                                   (time.time(), role, content))

    def _user_message(self, text, prefix=""):
        llm, p = self.ctx.llm, self.ctx.perception
        if not (self.ctx.settings.send_frames and getattr(llm, "supports_vision", False)):
            return {"role": "user", "content": prefix + text}
        parts = [{"type": "text", "text": prefix + text}]
        for cam, label in (("static", "bench camera"), ("mobile", "phone camera")):
            item = p.latest.get(cam)
            if item and time.time() - item["ts"] < 30:
                parts.append({"type": "text", "text": f"[latest {label} frame]"})
                parts.append(image_part(item["image"], max_px=768, quality=75))
        return {"role": "user", "content": parts}

    def _extract_notes(self, text):
        """User notes (F-EXP-9) in the background: never delays the reply."""
        mem, llm = self.ctx.memory, self.ctx.llm
        if not getattr(self.ctx.settings, "notes", True):
            return
        s = mem.experiments.active()
        t = threading.Thread(target=mem.notes.extract, args=(llm, text, s["id"] if s else None),
                             daemon=True, name="notes")
        t.start()
        self.last_notes_thread = t          # tests join it

    def handle(self, text):
        t0 = time.perf_counter()
        with self._lock:
            llm = self.ctx.llm
            # The system prompt stays identical every turn so the provider can cache
            # it (with the tool schemas); the changing situation rides on the user turn.
            situation = "[Current situation]\n" + prompts.situation(self.ctx) + "\n[User says]\n"
            messages = [{"role": "system", "content": prompts.SYSTEM}, *self._history(),
                        self._user_message(text, prefix=situation)]
            self._remember("user", text)
            self.ctx.memory.db.log_event("speech", "stated", {"text": text})
            trace, reply = [], None
            nudged = False
            for _ in range(MAX_STEPS):
                res = llm.chat(messages, tools=self.schemas, model=llm.model_for("agent"))
                if not res.tool_calls:
                    reply = (res.content or "").strip()
                    if trace and all(t["summary"] and not t["error"] and t["tool"] not in NEEDS_PHRASING
                                     for t in trace):
                        # compound turn finished: speak the tools' own (accurate) summaries
                        reply = " ".join(dict.fromkeys(t["summary"] for t in trace))
                        break
                    if not trace and ACTION_CLAIM.match(reply):
                        self.ctx.memory.db.log_event("guard", "inferred",
                                                     {"claim": reply[:200], "retried": not nudged})
                        if not nudged:
                            nudged = True
                            messages += [{"role": "assistant", "content": reply},
                                         {"role": "user", "content": NUDGE}]
                            reply = None
                            continue
                        reply = UNDONE
                    break
                messages.append({"role": "assistant", "content": res.content or "",
                                 "tool_calls": [{"id": c.id, "type": "function",
                                                 "function": {"name": c.name,
                                                              "arguments": json.dumps(c.arguments)}}
                                                for c in res.tool_calls]})
                step = []
                for c in res.tool_calls:
                    out = self.tools.run(c.name, c.arguments)
                    step.append({"tool": c.name, "args": c.arguments,
                                 "summary": out.get("summary"), "error": out.get("error")})
                    messages.append({"role": "tool", "tool_call_id": c.id,
                                     "content": json.dumps(out, default=str)[:12000]})
                trace += step
                if (all(s["summary"] and not s["error"] and s["tool"] not in NEEDS_PHRASING
                        for s in step) and not is_compound(text.strip())):
                    # The tools' summaries are the answer: speak them, skip a model call.
                    reply = " ".join(dict.fromkeys(s["summary"] for s in trace))
                    break
            if reply is None:
                reply = trace[-1]["summary"] if trace else "Sorry, I got stuck on that."
            self._remember("assistant", reply)
            self._extract_notes(text)
            return {"reply": reply, "trace": trace,
                    "ms": round((time.perf_counter() - t0) * 1000)}
