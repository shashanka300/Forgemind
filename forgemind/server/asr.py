"""Speech-to-text relay: phone microphone -> Muse Voice Transcribe -> text.

The phone opens /ws/asr while *Hold* is pressed and streams raw PCM
(16-bit little-endian mono, 16 kHz) as binary frames, then sends
{"type": "end"} on release. The server keeps MODEL_API_KEY to itself and
relays the audio to the Model API's realtime endpoint in PUSH_TO_TALK mode:

    -> {"authorization": {"accessToken": "Bearer <key>"}, "model": ..., ...}
    <- {"sessionId": ...}                      (ack, no "type")
    -> binary PCM frames
    <- {"type": "transcript", "transcript": ..., "final": false}   (cumulative)
    -> {"type": "endStream"}
    <- {"type": "transcript", "final": true, ...}, then close 1000

The phone gets {"type": "partial"|"final"|"error", "text": ...} back and
sends the final text to /api/chat like typed text.

Hands-free (/ws/asr?mode=hands_free) keeps one session open in ENDPOINTING
mode: the transcriber decides when the user has finished a sentence, and each
completed turn reaches the phone as {"type": "turn", "text": ...} (plus
"speech_start" when they begin talking). The phone keeps streaming, sending
silence while it speaks itself so the transcriber never hears the assistant.

With `TurnHooks` (the phone opens ?mode=hands_free&server_turns=1) the server answers
each turn itself instead of the phone posting it to /api/chat, saving a round trip:
    <- {"type": "turn", "text": ..., "server": true}   (show it, speak the acknowledgement)
    <- {"type": "reply", "text": ..., "reply": ..., "trace": [...]}
Partial transcripts are decided on while the user talks (System 1 prefetch), so a
fast-path answer is ready a few ms after the end of turn. Answering before the end of
turn doesn't work: the transcriber sends the last word only ~70 ms before it (measured
with eval/voice_latency.py), ~0.6 s after the user stops.
"""
import asyncio
import json
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

MODEL = "muse-voice-transcribe-1.0"
ACK_TIMEOUT = 10.0
FINAL_TIMEOUT = 15.0


@dataclass
class TurnHooks:
    partial: Optional[Callable[[str], None]] = None          # quick; schedules work
    turn: Optional[Callable[[str], Awaitable[dict]]] = None  # the agent's answer
    addressed: Optional[Callable[[str], Optional[str]]] = None  # the request, or None: not for us


def realtime_url(base_url):
    """wss://api.meta.ai/v1/asr/realtime from META_BASE_URL."""
    base = base_url.rstrip("/")
    for a, b in (("https://", "wss://"), ("http://", "ws://")):
        if base.startswith(a):
            base = b + base[len(a):]
    return base + "/asr/realtime"


async def default_connect(url):
    import websockets
    return await websockets.connect(url, max_size=2 ** 22, open_timeout=10)


class VoiceTranscribeRelay:
    def __init__(self, settings, connect=default_connect, ledger=None):
        self.url = realtime_url(settings.meta_base_url)
        self.key = settings.meta_api_key
        self.connect = connect
        self.ledger = ledger
        self._bytes = 0

    @property
    def available(self):
        return bool(self.key)

    def handshake(self, mode="PUSH_TO_TALK"):
        return {"authorization": {"accessToken": f"Bearer {self.key}"},
                "model": MODEL, "audioEncoding": "PCM_16KHZ", "mode": mode,
                "partialMode": "CUMULATIVE", "emitAudioProgress": False}

    async def run(self, phone, hands_free=False, hooks=None):
        """Relay one push-to-talk utterance, or a hands-free session of many turns.

        `phone` is a Starlette WebSocket.
        """
        try:
            if self.ledger:
                self.ledger.check()
            upstream = await self.connect(self.url)
        except Exception as e:
            await phone.send_json({"type": "error", "text": f"transcriber unreachable: {e}"})
            return None
        try:
            await upstream.send(json.dumps(self.handshake("ENDPOINTING" if hands_free
                                                          else "PUSH_TO_TALK")))
            ack = json.loads(await asyncio.wait_for(upstream.recv(), ACK_TIMEOUT))
            if ack.get("type") == "error" or "sessionId" not in ack:
                raise RuntimeError(ack.get("message") or f"bad handshake reply: {ack}")
            await phone.send_json({"type": "ready", "hands_free": hands_free})
            pump = asyncio.create_task(self._pump_audio(phone, upstream))
            if hands_free:
                await self._read_turns(phone, upstream, pump, hooks)
                return None
            final = await self._read_transcripts(phone, upstream, pump)
            await phone.send_json({"type": "final", "text": final})
            return final
        except Exception as e:
            await phone.send_json({"type": "error", "text": f"{type(e).__name__}: {e}"})
            return None
        finally:
            try:
                await upstream.close()
            except Exception:
                pass
            self._bill()

    def _bill(self):
        """16 kHz 16-bit mono = 32,000 bytes per second; billed per whole second."""
        secs, self._bytes = self._bytes // 32000, 0
        if self.ledger and secs:
            from forgemind.providers.budget import ASR_HOUR_USD
            self.ledger.add("asr", MODEL, secs, secs / 3600 * ASR_HOUR_USD)

    async def _pump_audio(self, phone, upstream):
        while True:
            msg = await phone.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes"):
                await upstream.send(msg["bytes"])
                self._bytes += len(msg["bytes"])
            elif msg.get("text") and json.loads(msg["text"]).get("type") == "end":
                break
        await upstream.send(json.dumps({"type": "endStream"}))

    async def _read_turns(self, phone, upstream, pump, hooks=None):
        """Hands-free: forward (or, with hooks, answer) every completed turn."""
        last_partial, tasks = "", set()
        try:
            while True:
                timeout = FINAL_TIMEOUT if pump.done() else None
                raw = await asyncio.wait_for(upstream.recv(), timeout)
                if isinstance(raw, bytes):
                    continue
                m = json.loads(raw)
                kind = m.get("type")
                if kind == "error":
                    raise RuntimeError(m.get("message") or "transcriber error")
                if kind == "speechStart":
                    await phone.send_json({"type": "speech_start"})
                elif kind == "transcript" and not m.get("final"):
                    text = m.get("transcript") or ""
                    await phone.send_json({"type": "partial", "text": text})
                    if text != last_partial and hooks and hooks.partial:
                        hooks.partial(text)
                    last_partial = text
                elif kind == "speechComplete":
                    text = (m.get("transcript") or "").strip()
                    last_partial = ""
                    if not text:
                        continue
                    if hooks and hooks.turn:
                        said = hooks.addressed(text) if hooks.addressed else text
                        if said:
                            await self._serve_turn(phone, said, hooks, tasks)
                        else:
                            await phone.send_json({"type": "heard", "text": text})
                    else:
                        await phone.send_json({"type": "turn", "text": text})
        except asyncio.TimeoutError:
            return
        except Exception as e:
            if type(e).__name__.startswith("ConnectionClosed"):
                return
            raise
        finally:
            pump.cancel()

    async def _serve_turn(self, phone, text, hooks, tasks):
        await phone.send_json({"type": "turn", "text": text, "server": True})

        async def answer():
            try:
                r = await hooks.turn(text)
            except Exception as e:
                r = {"reply": f"Something went wrong talking to the model: {e}", "trace": []}
            try:
                await phone.send_json({"type": "reply", "text": text, **r})
            except Exception:
                pass                      # phone went away
        t = asyncio.create_task(answer())
        tasks.add(t)
        t.add_done_callback(tasks.discard)

    async def _read_transcripts(self, phone, upstream, pump):
        latest = ""
        try:
            while True:
                timeout = FINAL_TIMEOUT if pump.done() else None
                raw = await asyncio.wait_for(upstream.recv(), timeout)
                if isinstance(raw, bytes):
                    continue
                m = json.loads(raw)
                kind = m.get("type")
                if kind == "error":
                    raise RuntimeError(m.get("message") or "transcriber error")
                if kind == "transcript":
                    latest = m.get("transcript") or ""     # cumulative: replace
                    if m.get("final"):
                        return latest
                    await phone.send_json({"type": "partial", "text": latest})
                elif kind == "speechComplete":
                    return m.get("transcript") or latest
        except asyncio.TimeoutError:
            return latest
        except Exception as e:
            if type(e).__name__.startswith("ConnectionClosed"):
                return latest           # server closed (1000) after the last transcript
            raise
        finally:
            pump.cancel()
