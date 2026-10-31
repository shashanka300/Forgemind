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
PRUNE_EVERY_TICKS = 120          # unreferenced keyframes are pruned every 10 minutes


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
        self.desk_mic_enabled = True       # the phone's Desk mic button
        self.desk_mic_open = 0
        self.speaker = None
        if settings.tts == "kokoro":
            from forgemind.server.tts import Speaker
            self.speaker = Speaker(settings)
            if warm:        # ~20 s to load: not on the first spoken reply
                threading.Thread(target=self.speaker.warm_up, daemon=True, name="tts-warm").start()
        if warm and self.agent.system1:
            # loading Laya takes ~10-50 s: do it now, not on the first spoken turn
            threading.Thread(target=self._warm_system1, daemon=True, name="system1-warm").start()
        from forgemind.providers import get_ledger
        self.ledger = get_ledger(settings) if settings.meta_api_key else None
        if self.ledger:
            self.ledger.on_warn = lambda m: self.hub.push({"type": "budget", "text": m})
        self.asr = VoiceTranscribeRelay(settings, ledger=self.ledger)
        self._stop = threading.Event()

    def _warm_system1(self):
        try:
            self.agent.system1.warm_up()
        except Exception as e:          # the Spark path still works
            print(f"System 1 unavailable: {e}")
            self.agent.system1 = None

    def describe(self):
        return {"llm": getattr(self.llm, "name", "?"),
                "agent_model": self.llm.model_for("agent"),
                "worker_model": self.llm.model_for("worker"),
                "segmenter": self.segmenter.name,
                "segmenter_error": getattr(self.segmenter, "last_error", None),
                "embedder": self.embedder.name,
                "asr": "muse-voice-transcribe" if self.asr.available else "browser",
                "system1": self.settings.system1 if self.agent.system1 else None,
                "tts": (f"kokoro:{self.speaker.voice}" if self.speaker and self.speaker.ready
                        and not self.speaker.error else None),
                "vector_backend": self.memory.index.name,
                "concepts": self.settings.concepts,
                "zones": self.settings.zones, "static_zone": self.settings.static_zone}

    def start_background(self):
        def loop():
            ticks = 0
            while not self._stop.wait(SAFETY_EVERY_S):
                ticks += 1
                try:
                    self.safety.evaluate()
                except Exception as e:  # keep watching
                    print(f"safety loop: {e}")
                if ticks % PRUNE_EVERY_TICKS == 0:
                    try:
                        self.memory.prune_keyframes()
                    except Exception as e:
                        print(f"keyframe prune: {e}")
        threading.Thread(target=loop, daemon=True, name="safety").start()

    def close(self):
        self._stop.set()
        time.sleep(0.05)
        self.memory.close()
