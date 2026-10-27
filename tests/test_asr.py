"""Voice Transcribe relay against a fake realtime endpoint."""
import asyncio
import json

from fastapi.testclient import TestClient

from forgemind.server.app import create_app
from forgemind.server.asr import realtime_url


class FakeUpstream:
    """Speaks the realtime protocol: ack, cumulative partials, final after endStream."""

    def __init__(self, fail_handshake=False):
        self.sent, self.audio = [], b""
        self.fail_handshake = fail_handshake
        self.inbox = asyncio.Queue()
        self.closed = False

    async def send(self, data):
        if isinstance(data, bytes):
            self.audio += data
            await self.inbox.put({"type": "transcript", "final": False,
                                  "transcript": f"heard {len(self.audio)} bytes"})
            return
        m = json.loads(data)
        self.sent.append(m)
        if "authorization" in m:
            await self.inbox.put({"type": "error", "message": "bad key"} if self.fail_handshake
                                 else {"sessionId": "s1"})
        elif m.get("type") == "endStream":
            await self.inbox.put({"type": "transcript", "final": True,
                                  "transcript": "where is the multimeter"})

    async def recv(self):
        return json.dumps(await self.inbox.get())

    async def close(self):
        self.closed = True


def test_realtime_url():
    assert realtime_url("https://api.meta.ai/v1/") == "wss://api.meta.ai/v1/asr/realtime"


def _client(rt, up):
    rt.settings.meta_api_key = "k"
    rt.asr.key = "k"

    async def connect(url):
        assert url == "wss://api.meta.ai/v1/asr/realtime"
        return up
    rt.asr.connect = connect
    return TestClient(create_app(rt), base_url="https://testserver")


def test_asr_relay_partials_and_final(rt):
    up = FakeUpstream()
    with _client(rt, up) as c:
        with c.websocket_connect("/ws/asr?t=secret-token") as ws:
            assert ws.receive_json() == {"type": "ready", "hands_free": False}
            ws.send_bytes(b"\x00\x01" * 800)
            assert ws.receive_json() == {"type": "partial", "text": "heard 1600 bytes"}
            ws.send_text(json.dumps({"type": "end"}))
            assert ws.receive_json() == {"type": "final", "text": "where is the multimeter"}
    hs = up.sent[0]
    assert hs["authorization"] == {"accessToken": "Bearer k"}
    assert hs["model"] == "muse-voice-transcribe-1.0" and hs["mode"] == "PUSH_TO_TALK"
    assert hs["audioEncoding"] == "PCM_16KHZ"
    assert up.sent[-1] == {"type": "endStream"} and up.audio == b"\x00\x01" * 800
    assert up.closed


def test_asr_handshake_error_reaches_phone(rt):
    with _client(rt, FakeUpstream(fail_handshake=True)) as c:
        with c.websocket_connect("/ws/asr?t=secret-token") as ws:
            m = ws.receive_json()
            assert m["type"] == "error" and "bad key" in m["text"]


def test_asr_without_key_says_use_browser(rt):
    with TestClient(create_app(rt), base_url="https://testserver") as c:
        with c.websocket_connect("/ws/asr?t=secret-token") as ws:
            assert ws.receive_json()["type"] == "error"
        assert c.get("/api/state", headers={"Authorization": "Bearer secret-token"}
                     ).json()["providers"]["asr"] == "browser"


class FakeEndpointing(FakeUpstream):
    """ENDPOINTING mode: speechStart / partials / speechComplete per turn."""

    async def send(self, data):
        if isinstance(data, bytes):
            self.audio += data
            n = len(self.audio)
            if n == 3200:
                await self.inbox.put({"type": "speechStart", "turnId": 1})
                await self.inbox.put({"type": "transcript", "final": False, "transcript": "where is"})
            elif n == 6400:
                await self.inbox.put({"type": "speechComplete", "turnId": 1,
                                      "transcript": "where is the multimeter"})
            elif n == 9600:
                await self.inbox.put({"type": "speechComplete", "turnId": 2, "transcript": "  "})
            elif n == 12800:
                await self.inbox.put({"type": "speechComplete", "turnId": 3,
                                      "transcript": "thanks"})
            return
        await super().send(data)


def test_hands_free_forwards_each_turn(rt):
    up = FakeEndpointing()
    with _client(rt, up) as c:
        with c.websocket_connect("/ws/asr?t=secret-token&mode=hands_free") as ws:
            assert ws.receive_json() == {"type": "ready", "hands_free": True}
            for _ in range(4):
                ws.send_bytes(b"\x00\x00" * 1600)
            got = [ws.receive_json() for _ in range(4)]
            ws.send_text(json.dumps({"type": "end"}))
    assert got == [{"type": "speech_start"}, {"type": "partial", "text": "where is"},
                   {"type": "turn", "text": "where is the multimeter"},
                   {"type": "turn", "text": "thanks"}]                  # empty turn dropped
    assert up.sent[0]["mode"] == "ENDPOINTING"
