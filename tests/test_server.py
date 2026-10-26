import time

from fastapi.testclient import TestClient

from conftest import jpeg, scene
from forgemind.server.app import create_app


def client(rt):
    return TestClient(create_app(rt), base_url="https://testserver")


def test_token_cookie_flow(rt):
    with client(rt) as c:
        assert c.get("/healthz").json() == {"ok": True}
        r = c.get("/", follow_redirects=False)
        assert r.status_code == 401 and "locked" in r.text
        assert c.get("/api/state").status_code == 401
        r = c.get("/?t=wrong", follow_redirects=False)
        assert r.status_code == 401
        r = c.get("/?t=secret-token&role=static", follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/?role=static"
        assert "forgemind_token" in r.headers["set-cookie"] and "HttpOnly" in r.headers["set-cookie"]
        assert c.get("/").status_code == 200          # cookie now set
        assert "ForgeMind" in c.get("/").text


def test_frames_chat_and_inspector(rt):
    h = {"Authorization": "Bearer secret-token"}
    with client(rt) as c:
        r = c.post("/api/frame?camera=static", content=jpeg(scene([("multimeter", .4, .4, .6, .6)])),
                   headers={**h, "Content-Type": "image/jpeg"})
        assert r.status_code == 200
        out = r.json()
        assert out["zone"] == "bench" and out["detections"][0]["label"] == "multimeter"
        kf = out["keyframe"]
        assert c.get(f"/keyframes/{kf}", headers=h).status_code == 200
        assert c.get("/keyframes/..%2Fforgemind.sqlite", headers=h).status_code == 404
        r = c.post("/api/chat", json={"text": "where is the multimeter?"}, headers=h).json()
        assert "bench" in r["reply"]
        ents = c.get("/api/entities", headers=h).json()["entities"]
        eid = ents[0]["id"]
        r = c.post(f"/api/entities/{eid}", json={"name": "Fluke"}, headers=h).json()
        assert r["entity"]["label"] == "Fluke"
        assert c.get(f"/api/entities/{eid}", headers=h).json()["history"]
        st = c.get("/api/state", headers=h).json()
        assert st["providers"]["llm"] == "mock" and st["scenes"]["static"]["objects"] == ["Fluke"]
        assert c.post("/api/privacy", json={"paused": True}, headers=h).json()["paused"]
        r = c.post("/api/frame?camera=static", content=jpeg(scene()), headers=h).json()
        assert r["paused"]
        assert c.delete(f"/api/entities/{eid}", headers=h).json()["ok"]
        assert c.get("/api/entities", headers=h).json()["entities"] == []


def test_websocket_pushes_safety_alert(rt):
    with client(rt) as c:
        c.get("/?t=secret-token")
        with c.websocket_connect("/ws") as ws:
            assert ws.receive_json()["type"] == "hello"
            now = time.time()
            rt.safety.last_present = now - 600
            rt.perception.process("static", scene([("soldering iron", .1, .4, .3, .6)]), ts=now)
            rt.safety.evaluate(now)
            msg = ws.receive_json()
            assert msg["type"] == "alert" and "soldering iron" in msg["message"]


def test_websocket_rejects_without_token(rt):
    import pytest
    from starlette.websockets import WebSocketDisconnect
    with client(rt) as c:
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect("/ws") as ws:
                ws.receive_json()
