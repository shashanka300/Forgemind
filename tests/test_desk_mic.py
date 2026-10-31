"""Desk mic: when it may be open, audio conversion, the phone's on/off button."""
import numpy as np
from fastapi.testclient import TestClient

from forgemind.cameras.desk_mic import should_listen, to_pcm16
from forgemind.server.app import create_app

NEAR = {"phone_mic": {"listening": False, "idle_s": 60}, "desk_mic": {"enabled": True}, "paused": False, "private": False, "user_away_s": 5,
        "scenes": {"static": {"available": True, "age_s": 2}}}


def test_open_only_while_the_user_is_at_the_bench():
    assert should_listen(NEAR)
    for change in ({"desk_mic": {"enabled": False}}, {"paused": True}, {"private": True},
                   {"user_away_s": 500}, {"scenes": {"static": {"available": False}}},
                   {"scenes": {"static": {"available": True, "age_s": 300}}},
                   {"phone_mic": {"listening": True, "idle_s": 60}},        # one mic at a time
                   {"phone_mic": {"listening": False, "idle_s": 1}}):       # phone reconnecting
        assert not should_listen({**NEAR, **change}), change
    assert not should_listen({})


def test_audio_becomes_16khz_pcm():
    pcm = to_pcm16(np.zeros((960, 1), "float32") + 0.5, 48000)
    a = np.frombuffer(pcm, "<i2")
    assert len(a) == 320 and abs(a[0] - 16383) < 2


def test_phone_button_turns_it_off(rt):
    with TestClient(create_app(rt), base_url="https://testserver") as c:
        c.get("/?t=secret-token")
        assert c.get("/api/state").json()["desk_mic"] == {"enabled": True, "listening": False}
        assert c.post("/api/desk-mic", json={"on": False}).json()["enabled"] is False
        assert not should_listen({**NEAR, **c.get("/api/state").json()})
        assert 'id="deskBtn"' in c.get("/").text
