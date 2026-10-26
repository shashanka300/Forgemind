"""Primary reasoning model with a fallback (PRD F-AGT-7).

The Meta Model API is the primary; Muse Glimmer on the local GPU (served as an
OpenAI-compatible endpoint, FM_COMPAT_*) takes over when the API errors, times
out or rate-limits. After a failure the fallback is used for `cooldown`
seconds before the primary is tried again, so a dead API costs one timeout,
not one per turn.
"""
import threading
import time


class FallbackModel:
    def __init__(self, primary, fallback, cooldown=60.0):
        self.primary, self.fallback = primary, fallback
        self.cooldown = cooldown
        self.name = f"{primary.name}+{fallback.name}"
        self.supports_vision = primary.supports_vision and fallback.supports_vision
        self.last_error = None
        self.active = primary.name
        self._down_until = 0.0
        self._lock = threading.Lock()

    def model_for(self, role):
        return self.primary.model_for(role)

    def _fallback_model(self, model):
        """Translate a primary model id to the fallback's id for the same role."""
        for role in ("agent", "worker"):
            if model == self.primary.model_for(role):
                return self.fallback.model_for(role)
        return self.fallback.model_for("agent")

    def chat(self, messages, tools=None, model=None, json_mode=False):
        model = model or self.primary.model_for("agent")
        if time.time() >= self._down_until:
            try:
                res = self.primary.chat(messages, tools=tools, model=model, json_mode=json_mode)
                self.active = self.primary.name
                return res
            except Exception as e:
                with self._lock:
                    self.last_error = f"{type(e).__name__}: {e}"
                    self._down_until = time.time() + self.cooldown
                print(f"{self.primary.name} failed ({self.last_error}); "
                      f"using {self.fallback.name} for {self.cooldown:.0f}s")
        self.active = self.fallback.name
        return self.fallback.chat(messages, tools=tools, model=self._fallback_model(model),
                                  json_mode=json_mode)

    def list_models(self):
        return self.primary.list_models()
