"""Wires providers, memory, perception, safety and the agent together."""
import threading
import time

from forgemind.agent.orchestrator import Agent, Context
from forgemind.hub import Hub
from forgemind.memory import WorldMemory
from forgemind.perception.pipeline import Perception
from forgemind.perception.safety import Safety
from forgemind.providers import build_embedder, build_llm, build_segmenter
from forgemind.server.asr import VoiceTranscribeRelay

SAFETY_EVERY_S = 5.0


class Runtime:
    def __init__(self, settings, llm=None, segmenter=None, embedder=None, warm=True):
        self.settings = settings
        self.embedder = embedder or build_embedder(settings)
        if warm and hasattr(self.embedder, "warm_up"):
            self.embedder.warm_up()
        self.memory = WorldMemory(settings, self.embedder)
        self.hub = Hub()
        self.safety = Safety(settings, self.memory.db, self.hub)
        self.segmenter = segmenter or build_segmenter(settings, self.embedder)
        self.perception = Perception(settings, self.memory, self.segmenter, self.embedder,
                                     self.hub, self.safety)
        self.llm = llm or build_llm(settings)
        self.ctx = Context(settings, self.memory, self.perception, self.safety, self.hub, self.llm)
        self.agent = Agent(self.ctx)
        self.asr = VoiceTranscribeRelay(settings)
        self._stop = threading.Event()

    def describe(self):
        return {"llm": getattr(self.llm, "name", "?"),
                "agent_model": self.llm.model_for("agent"),
                "worker_model": self.llm.model_for("worker"),
                "segmenter": self.segmenter.name,
                "segmenter_error": getattr(self.segmenter, "last_error", None),
                "embedder": self.embedder.name,
                "asr": "muse-voice-transcribe" if self.asr.available else "browser",
                "vector_backend": self.memory.index.name,
                "concepts": self.settings.concepts,
                "zones": self.settings.zones, "static_zone": self.settings.static_zone}

    def start_background(self):
        def loop():
            while not self._stop.wait(SAFETY_EVERY_S):
                try:
                    self.safety.evaluate()
                except Exception as e:  # keep watching
                    print(f"safety loop: {e}")
        threading.Thread(target=loop, daemon=True, name="safety").start()

    def close(self):
        self._stop.set()
        time.sleep(0.05)
        self.memory.close()
