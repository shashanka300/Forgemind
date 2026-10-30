"""Spoken replies with a natural voice: Kokoro-82M on the local GPU.

The phone's built-in speech synthesis sounds robotic, and the Meta Model API has no
text-to-speech model, so replies are voiced locally (~90 ms for 8 s of speech on the
5090) and sent to the phone as MP3 (`POST /api/tts`). If the voice isn't available the
phone falls back to its own speech synthesis.

    FM_TTS=kokoro          # "" = phone's built-in voice
    FM_TTS_VOICE=af_heart  # af_bella, am_michael, bf_emma, ...
"""
import io
import os
import re
import threading
from collections import OrderedDict

SAMPLE_RATE = 24000
CACHE_SIZE = 64            # acknowledgements and repeated phrases are voiced once


class Speaker:
    def __init__(self, settings):
        self.voice = settings.tts_voice
        self.speed = settings.tts_speed
        self.lang = self.voice[0] if self.voice else "a"     # Kokoro voices start with their language
        self._pipe = None
        self._lock = threading.Lock()
        self._cache = OrderedDict()
        self.loading = False
        self.error = None

    @property
    def ready(self):
        return self._pipe is not None and not self.loading and not self.error

    def warm_up(self):
        self.loading = True
        try:
            self.speak("Ready.")
        except Exception as e:           # the phone keeps its own voice
            self.error = str(e)
        finally:
            self.loading = False

    def _load(self):
        if self._pipe is None:
            os.environ.setdefault("USE_TF", "0")
            from kokoro import KPipeline
            import torch
            self._pipe = KPipeline(lang_code=self.lang, repo_id="hexgrad/Kokoro-82M",
                                   device="cuda" if torch.cuda.is_available() else "cpu")
        return self._pipe

    def speak(self, text, voice=None):
        """MP3 bytes for `text`."""
        text = _clean(text)
        if not text:
            raise ValueError("nothing to say")
        key = (voice or self.voice, text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            import numpy as np
            import soundfile as sf
            pipe = self._load()
            parts = [a.numpy() if hasattr(a, "numpy") else a
                     for _, _, a in pipe(text, voice=key[0], speed=self.speed)]
            audio = np.concatenate(parts) if parts else np.zeros(SAMPLE_RATE // 10, "float32")
            buf = io.BytesIO()
            sf.write(buf, audio, SAMPLE_RATE, format="MP3", subtype="MPEG_LAYER_III")
            data = buf.getvalue()
            self._cache[key] = data
            if len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
            return data


def _clean(text):
    """What a listener should hear: no markdown, no citation tags, units spoken."""
    t = re.sub(r"\[E\d+\]", "", text or "")
    t = re.sub(r"[*_`#>]+", "", t)
    t = (t.replace("kΩ", " kilohms").replace("MΩ", " megohms").replace("Ω", " ohms")
         .replace("°C", " degrees"))
    t = " ".join(t.split())
    return re.sub(r"\s+([.,!?;:])", r"\1", t)
