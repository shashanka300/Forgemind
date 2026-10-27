"""Acceptance check #1 (PRD §14): a taught object moved from bench to shelf is
found by where_is with the right zone and camera, and highlighted live when the
phone gets there."""
import time

from fastapi.testclient import TestClient

from conftest import scene
from forgemind.server.app import create_app

H = {"Authorization": "Bearer secret-token"}


def pushed(rt, kind):
    return [m for m in rt.hub.outbox if m["type"] == kind]


def test_taught_object_moved_bench_to_shelf_found_and_highlighted_on_arrival(rt):
    p, now = rt.perception, time.time()
    # 1. teach it on the phone at the bench (tag 0)
    p.process("mobile", scene([("hex key", .4, .4, .55, .6)], tag=0), ts=now - 60)
    rt.agent.tools.run("teach_object", {"name": "2.5 mm hex key"})
    # 2. the bench camera sees it lying on the bench
    p.process("static", scene([("hex key", .1, .4, .25, .6)]), ts=now - 50)
    v = rt.memory.registry.find("2.5 mm hex key")[0]
    assert (v["zone"], v["camera"]) == ("bench", "static")
    # 3. it is carried away; later the phone sees it on the shelf (tag 1)
    for i in range(4):
        p.process("static", scene([]), ts=now - 40 + i * 5, force=True)
    out = p.process("mobile", scene([("hex key", .2, .5, .35, .7)], tag=1), ts=now - 10)
    assert ("moved", "2.5 mm hex key") in [(e["type"], e.get("label")) for e in out["events"]]
    # 4. the user is at the drawer (tag 2, nothing there) and asks
    p.process("mobile", scene([], tag=2), ts=now - 5)
    rt.hub.outbox.clear()
    r = rt.agent.handle("where is my 2.5 mm hex key?")
    assert "shelf" in r["reply"] and "your phone" in r["reply"]
    assert "highlight" in r["reply"]
    card = pushed(rt, "last_seen")[0]["entity"]
    assert card["zone"] == "shelf" and card["snapshot"].endswith("/snapshot")
    assert pushed(rt, "seeking")[0]["label"] == "2.5 mm hex key"
    assert pushed(rt, "highlight") == []
    # 5. the user walks to the shelf: the first phone frame with it highlights it
    out = p.process("mobile", scene([("hex key", .6, .3, .75, .5)], tag=1), ts=time.time())
    assert [e["type"] for e in out["events"]].count("found") == 1
    hl = pushed(rt, "highlight")[0]
    assert hl["found"] and hl["label"] == "2.5 mm hex key" and hl["box"][0] == 0.6
    assert pushed(rt, "say")[-1]["text"] == "There's your 2.5 mm hex key."
    # ...only once
    out = p.process("mobile", scene([("hex key", .6, .3, .75, .5)], tag=1), ts=time.time() + 1)
    assert "found" not in [e["type"] for e in out["events"]]


def test_where_is_in_view_highlights_now_and_does_not_seek(rt):
    p = rt.perception
    p.process("mobile", scene([("multimeter", .4, .4, .6, .6)], tag=0), ts=time.time())
    rt.hub.outbox.clear()
    out = rt.agent.tools.run("where_is", {"query": "multimeter"})
    assert out["in_view"] and not out["seeking"]
    assert pushed(rt, "highlight") and not pushed(rt, "seeking")


def test_seek_expires(rt):
    rt.hub.seek(99, "thing", ttl=1)
    assert rt.hub.check_seeks({}, now=time.time() + 5) == []
    assert rt.hub.check_seeks({99: [0, 0, 1, 1]}, now=time.time() + 6) == []


def test_snapshot_endpoint_zooms_on_the_object(rt):
    rt.perception.process("static", scene([("multimeter", .45, .4, .6, .6)], size=(1280, 960)),
                          ts=time.time())
    eid = rt.memory.registry.list()[0]["id"]
    with TestClient(create_app(rt), base_url="https://testserver") as c:
        r = c.get(f"/api/entities/{eid}/snapshot", headers=H)
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        from io import BytesIO

        from PIL import Image
        img = Image.open(BytesIO(r.content))
        assert max(img.size) <= 480 and img.size != (480, 360)      # cropped, not the full frame
        assert c.get("/api/entities/9999/snapshot", headers=H).status_code == 404


def test_phone_frame_without_zone_keeps_last_known_zone(rt):
    p = rt.perception
    p.process("static", scene([("multimeter", .4, .4, .6, .6)]), ts=time.time() - 30)
    out = p.process("mobile", scene([("multimeter", .3, .3, .5, .5)]), ts=time.time())  # no tag
    assert out["zone"] is None
    assert "moved" not in [e["type"] for e in out["events"]]
    v = rt.memory.registry.find("multimeter")[0]
    assert (v["zone"], v["camera"]) == ("bench", "mobile")
