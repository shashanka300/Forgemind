"""OpenAI-compatible reasoning provider.

One class serves two configurations:

* ``meta``   - Meta Model API (https://api.meta.ai/v1, key in MODEL_API_KEY),
               Muse Wonder for the agent and Muse Spark for workers.
* ``compat`` - any hosted OpenAI-compatible endpoint serving Meta open
               weights (Fireworks, Baseten, Modal...).

Both use /chat/completions with `tools`, which Meta documents for the Muse
models. Images go in as data URLs (`image_url` parts).
"""
import base64
import io
import json
import uuid

from PIL import Image

from forgemind.providers.base import ChatResult, ToolCall


def image_part(image: Image.Image, max_px=1024, quality=80):
    """A chat content part carrying a JPEG data URL."""
    img = image.convert("RGB").copy()
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


class OpenAICompatModel:
    supports_vision = True

    def __init__(self, base_url, api_key, agent_model, worker_model, name="compat",
                 reasoning_effort="", http_client=None, timeout=120):
        from openai import OpenAI
        if not api_key:
            raise ValueError(f"{name}: no API key configured")
        # api_key is passed explicitly so an ambient OPENAI_API_KEY is never
        # sent to a different provider's endpoint.
        kw = {"base_url": base_url, "api_key": api_key, "timeout": timeout}
        if http_client is not None:
            kw["http_client"] = http_client
        self.client = OpenAI(**kw)
        self.name = name
        self.base_url = base_url
        self.models = {"agent": agent_model, "worker": worker_model or agent_model}
        self.reasoning_effort = reasoning_effort

    def model_for(self, role):
        return self.models.get(role) or self.models["agent"]

    def chat(self, messages, tools=None, model=None, json_mode=False):
        kw = {"model": model or self.models["agent"], "messages": messages}
        if tools:
            kw["tools"] = tools
            kw["tool_choice"] = "auto"
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            kw["reasoning_effort"] = self.reasoning_effort
        resp = self.client.chat.completions.create(**kw)
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append(ToolCall(id=tc.id or uuid.uuid4().hex[:12],
                                  name=tc.function.name, arguments=args))
        usage = resp.usage.model_dump() if getattr(resp, "usage", None) else None
        return ChatResult(content=msg.content, tool_calls=calls, raw=resp, usage=usage)

    def list_models(self):
        return [m.id for m in self.client.models.list().data]
