"""Desk mic client: the bench webcam's microphone -> ForgeMind, while the user is at the bench.

    forgemind desk-mic                 # uses the first input device named like FM_DESK_MIC

One mic at a time: this one is only open while the phone's mic is off (for PHONE_IDLE_S) and
the phone's Desk mic button is on, cameras aren't paused or private, the bench camera is
streaming, and it saw the user within NEAR_S. Running both at once mixed the conversation
(phone test, Oct 31). Replies to this mic play on the PC speakers (muted meanwhile).
"""
import asyncio
import io
import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np

NEAR_S = 120.0
PHONE_IDLE_S = 5.0         # a phone reconnecting isn't a phone gone
RATE = 16000
FRAME_S = 0.02
AFTER_SPEECH_S = 1.2       # output lags the end of playback
POLL_S = 2.0


def should_listen(state):
    """May the mic be open? `state` is /api/state."""
    desk = state.get("desk_mic") or {}
    bench = (state.get("scenes") or {}).get("static") or {}
    phone = state.get("phone_mic") or {}
    if phone.get("listening", True) or phone.get("idle_s", 0) < PHONE_IDLE_S:
        return False
    return bool(desk.get("enabled") and not state.get("paused") and not state.get("private")
                and bench.get("available") and bench.get("age_s", 1e9) < 60
                and state.get("user_away_s", 1e9) < NEAR_S)


def to_pcm16(block, rate):
    """Float samples at the device's rate -> 16 kHz mono PCM16 bytes."""
    mono = block.mean(axis=1) if block.ndim > 1 else block
    if rate != RATE:
        n = int(round(len(mono) * RATE / rate))
        mono = np.interp(np.linspace(0, len(mono), n, endpoint=False), np.arange(len(mono)), mono)
    return (np.clip(mono, -1, 1) * 32767).astype("<i2").tobytes()


def find_device(name):
    """Index and native rate of the input device named like `name` (WASAPI preferred)."""
    import sounddevice as sd
    hits = [(i, d) for i, d in enumerate(sd.query_devices())
            if d["max_input_channels"] > 0 and name.lower() in d["name"].lower()]
    if not hits:
        return None
    hits.sort(key=lambda x: "WASAPI" not in sd.query_hostapis(x[1]["hostapi"])["name"])
    return hits[0][0], int(hits[0][1]["default_samplerate"])


class DeskMic:
    def __init__(self, server, token, device):
        self.server, self.token = server.rstrip("/"), token
        self.device, self.rate = device
        self.quiet_until = 0.0

    def _req(self, path, data=None):
        req = urllib.request.Request(self.server + path, data=data, headers={
            "Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=15)

    def state(self):
        return json.load(self._req("/api/state"))

    def _speak(self, text):
        """The reply on the PC speakers; the mic sends silence until it has finished."""
        import sounddevice as sd
        import soundfile as sf
        self.quiet_until = time.time() + 60
        try:
            audio, rate = sf.read(io.BytesIO(self._req("/api/tts", json.dumps({"text": text}).encode()).read()))
            sd.play(audio, rate)
            sd.wait()
        except Exception as e:
            print(f"couldn't play the reply: {e}")
        self.quiet_until = time.time() + AFTER_SPEECH_S

    async def _session(self):
        """One open-mic period; ends when should_listen turns false or the link drops."""
        import sounddevice as sd
        import websockets
        ws_url = (self.server.replace("http", "ws", 1) +
                  f"/ws/asr?mode=hands_free&server_turns=1&source=desk&t={self.token}")
        loop, q = asyncio.get_running_loop(), asyncio.Queue()

        def on_audio(block, frames, when, status):
            pcm = to_pcm16(block.copy(), self.rate)
            if time.time() < self.quiet_until:
                pcm = b"\x00" * len(pcm)
            loop.call_soon_threadsafe(q.put_nowait, pcm)

        async with websockets.connect(ws_url, max_size=2 ** 22) as ws:
            async def send():
                while True:
                    await ws.send(await q.get())

            async def receive():
                while True:
                    m = json.loads(await ws.recv())
                    if m["type"] == "turn":
                        print(f"heard: {m['text']}")
                    elif m["type"] == "reply" and m.get("reply"):
                        print(f"reply: {m['reply']}")
                        threading.Thread(target=self._speak, args=(m["reply"],), daemon=True).start()
                    elif m["type"] == "error":
                        print(f"speech error: {m['text']}")

            async def watch():
                while True:
                    await asyncio.sleep(POLL_S)
                    if not should_listen(await asyncio.to_thread(self.state)):
                        return

            with sd.InputStream(device=self.device, channels=1, samplerate=self.rate, dtype="float32",
                                blocksize=int(self.rate * FRAME_S), callback=on_audio):
                print("desk mic open")
                tasks = [asyncio.create_task(c) for c in (send(), receive(), watch())]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
            print("desk mic closed")

    def run(self):
        while True:
            try:
                if should_listen(self.state()):
                    asyncio.run(self._session())
            except KeyboardInterrupt:
                return
            except (urllib.error.URLError, OSError, ValueError) as e:
                print(f"waiting for the server: {e}")
            except Exception as e:           # a dropped link or a busy device: try again
                print(f"desk mic stopped: {type(e).__name__}: {e}")
            time.sleep(POLL_S)
