"""Rule-based stand-in for the reasoning model.

It speaks the same chat-completions protocol as the real provider: it reads
the last user message, picks a tool by pattern, and after the tool result
comes back it answers with the tool's `summary`. This lets the whole
pipeline (tools, memory, phone UI, voice) run before you have API access.
It does not reason; swap in FM_LLM_PROVIDER=meta for that.
"""
import json
import re
import uuid

from forgemind.providers.base import ChatResult, ToolCall

ARTICLE = r"(?:the |my |a |an |our )?"


def _clean(s):
    return re.sub(r"[?.!,]+$", "", s.strip())


RULES = [
    (rf"^(?:where(?:'s| is| are)|where did i (?:put|leave)|find|have you seen) {ARTICLE}(.+)",
     lambda m: ("where_is", {"query": _clean(m.group(1))})),
    (rf"(?:this is|remember this as|call this|teach(?: this as)?) {ARTICLE}(.+)",
     lambda m: ("teach_object", {"name": _clean(m.group(1))})),
    (r"(?:where did we leave off|resume|continue|pick up)(?: (?:the|with|from))?(?: (.+?))?(?: from yesterday| session| experiment)?$",
     lambda m: ("resume_session", {"name": _clean(m.group(1) or "") or None})),
    (r"(?:start|begin|new) (?:a )?(?:new )?(?:session|experiment|test)(?: (?:called|named|for|on))? ?(.*)",
     lambda m: ("start_session", {"name": _clean(m.group(1)) or "untitled session"})),
    (r"(?:end|stop|finish|close) (?:the )?(?:session|experiment)",
     lambda m: ("end_session", {})),
    (r"(?:plan|steps are|the steps are|checklist)[: ]+(.+)",
     lambda m: ("plan_steps", {"steps": [s.strip() for s in re.split(r";|,| then ", m.group(1)) if s.strip()]})),
    (r"(?:i )?measured (?:the )?(.+?) (?:is|was|=|at|as) (-?[\d.]+) ?([a-zA-Zµ%Ω]*)",
     lambda m: ("log_measurement", {"quantity": m.group(1), "value": float(m.group(2)),
                                    "unit": m.group(3) or None, "source": "stated"})),
    (r"(?:set|configure|change) (?:the )?(.+?) to (.+)",
     lambda m: ("set_config", {"key": _clean(m.group(1)), "value": _clean(m.group(2))})),
    (r"^(?:log|note|record)(?: that)?[: ]+(.+)",
     lambda m: ("log_step", {"text": _clean(m.group(1))})),
    (r"(?:just checking|don't log|do not log)[,: ]*(.*)",
     lambda m: ("log_step", {"text": _clean(m.group(1)) or "checking", "tentative": True})),
    (rf"read {ARTICLE}(.+)",
     lambda m: ("read_closeup", {"target": _clean(m.group(1))})),
    (rf"(?:look at|check|go to|show me) {ARTICLE}(.+)",
     lambda m: ("go_look_at", {"target": _clean(m.group(1)), "reason": "you asked"})),
    (rf"(?:highlight|point (?:to|at)) {ARTICLE}(.+)",
     lambda m: ("highlight", {"query": _clean(m.group(1))})),
    (rf"(?:watch|keep an eye on) {ARTICLE}(.+?)(?: for me)?$",
     lambda m: ("watch_bench", {"concept": _clean(m.group(1))})),
    (r"(?:what(?:'s| is) on|what do you see|look around|what's here)",
     lambda m: ("inspect_scene", {"camera": "both"})),
    (r"(?:compare|what changed)",
     lambda m: ("compare_sessions", {})),
    (r"(?:write (?:it|this) up|report|summari[sz]e)",
     lambda m: ("generate_report", {})),
]


class MockModel:
    name = "mock"
    supports_vision = False

    def model_for(self, role):
        return f"mock-{role}"

    def chat(self, messages, tools=None, model=None, json_mode=False):
        last = messages[-1]
        if last.get("role") == "tool":
            # Answer from the tool results of this turn.
            says = []
            for m in reversed(messages):
                if m.get("role") != "tool":
                    break
                try:
                    data = json.loads(m.get("content") or "{}")
                except json.JSONDecodeError:
                    data = {}
                says.append(data.get("summary") or data.get("error") or "Done.")
            return ChatResult(content=" ".join(reversed(says)))

        text = last.get("content")
        if isinstance(text, list):
            text = " ".join(p.get("text", "") for p in text if p.get("type") == "text")
        utter = (text or "").split("[User says]\n")[-1].strip()   # drop the situation block
        # Drop a leading wake word / politeness.
        utter = re.sub(r"^(?:hey |ok )?(?:forge(?:mind)?|jarvis)[,:]?\s*", "", utter, flags=re.I)
        utter = re.sub(r"^(?:can you |could you |please |let's |lets )+", "", utter, flags=re.I)
        utter = utter.rstrip("?.! ")
        allowed = {t["function"]["name"] for t in tools or []}
        if json_mode:
            return ChatResult(content=json.dumps({"value": None, "unit": None,
                                                  "note": "mock model cannot read images"}))
        for pattern, build in RULES:
            m = re.search(pattern, utter, flags=re.I)
            if m:
                name, args = build(m)
                if name in allowed:
                    args = {k: v for k, v in args.items() if v is not None}
                    return ChatResult(content=None, tool_calls=[
                        ToolCall(id="call_" + uuid.uuid4().hex[:10], name=name, arguments=args)])
        return ChatResult(content=(
            "I'm running in mock mode, so I only understand fixed phrases like "
            "'where is the multimeter', 'start a session called voltage divider', "
            "'measured output voltage is 3.3 V', 'what's on the bench' or 'write it up'. "
            "Set FM_LLM_PROVIDER=meta once you have a key."))
