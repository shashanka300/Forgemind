"""Natural voice endpoint (Kokoro itself is stubbed: no model download in tests)."""
from fastapi.testclient import TestClient

from forgemind.server.app import create_app
from forgemind.server.tts import _clean


class StubSpeaker:
    voice, loading, error, ready = "af_heart", False, None, True

    def __init__(self):
        self.said = []

    def speak(self, text, voice=None):
        self.said.append(text)
        return b"ID3fake-mp3"


def _client(rt):
    c = TestClient(create_app(rt), base_url="https://testserver")
    c.get("/?t=secret-token")                     # sets the session cookie
    return c


def test_tts_returns_mp3(rt):
    rt.speaker = StubSpeaker()
    with _client(rt) as c:
        r = c.post("/api/tts", json={"text": "Your multimeter is on the bench."})
        assert r.status_code == 200 and r.headers["content-type"] == "audio/mpeg"
        assert r.content == b"ID3fake-mp3"
        assert c.get("/api/state").json()["providers"]["tts"] == "kokoro:af_heart"


def test_tts_unavailable_means_phone_voice(rt):
    rt.speaker = None
    with _client(rt) as c:
        assert c.post("/api/tts", json={"text": "hi"}).status_code == 503
        assert c.get("/api/state").json()["providers"]["tts"] is None


def test_text_is_cleaned_for_listening():
    assert _clean("Logged R1 = 9.8 kΩ [E12]. **Tip** at 350 °C") == "Logged R1 = 9.8 kilohms. Tip at 350 degrees"
