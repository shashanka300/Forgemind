"""The agent loop: situation summary + conversation -> model -> validated tools -> reply.

Muse Wonder (FM_AGENT_MODEL) handles user turns; workers (FM_WORKER_MODEL)
are used inside tools for reading displays and writing reports.
"""
import json
import threading
import time
from dataclasses import dataclass

from forgemind.agent import prompts
from forgemind.agent.tools import ToolBox, schemas
from forgemind.providers.openai_compat import image_part

MAX_STEPS = 6
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

    def _user_message(self, text):
        llm, p = self.ctx.llm, self.ctx.perception
        if not (self.ctx.settings.send_frames and getattr(llm, "supports_vision", False)):
            return {"role": "user", "content": text}
        parts = [{"type": "text", "text": text}]
        for cam, label in (("static", "bench camera"), ("mobile", "phone camera")):
            item = p.latest.get(cam)
            if item and time.time() - item["ts"] < 30:
                parts.append({"type": "text", "text": f"[latest {label} frame]"})
                parts.append(image_part(item["image"], max_px=768, quality=75))
        return {"role": "user", "content": parts}

    def handle(self, text):
        t0 = time.perf_counter()
        with self._lock:
            llm = self.ctx.llm
            system = prompts.SYSTEM + "\n\nCurrent situation:\n" + prompts.situation(self.ctx)
            messages = [{"role": "system", "content": system}, *self._history(),
                        self._user_message(text)]
            self._remember("user", text)
            self.ctx.memory.db.log_event("speech", "stated", {"text": text})
            trace, reply = [], None
            for _ in range(MAX_STEPS):
                res = llm.chat(messages, tools=self.schemas, model=llm.model_for("agent"))
                if not res.tool_calls:
                    reply = (res.content or "").strip()
                    break
                messages.append({"role": "assistant", "content": res.content or "",
                                 "tool_calls": [{"id": c.id, "type": "function",
                                                 "function": {"name": c.name,
                                                              "arguments": json.dumps(c.arguments)}}
                                                for c in res.tool_calls]})
                for c in res.tool_calls:
                    out = self.tools.run(c.name, c.arguments)
                    trace.append({"tool": c.name, "args": c.arguments,
                                  "summary": out.get("summary"), "error": out.get("error")})
                    messages.append({"role": "tool", "tool_call_id": c.id,
                                     "content": json.dumps(out, default=str)[:12000]})
            if reply is None:
                reply = trace[-1]["summary"] if trace else "Sorry, I got stuck on that."
            self._remember("assistant", reply)
            return {"reply": reply, "trace": trace,
                    "ms": round((time.perf_counter() - t0) * 1000)}
