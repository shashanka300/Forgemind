import time

from conftest import scene


def events_of(out):
    return [(e["type"], e.get("label")) for e in out["events"]]


def test_static_camera_creates_entities_once(rt):
    img = scene([("soldering iron", .05, .4, .25, .6), ("multimeter", .45, .4, .6, .6)])
    out = rt.perception.process("static", img, ts=1000.0)
    assert sorted(events_of(out)) == [("appeared", "multimeter"), ("appeared", "soldering iron")]
    again = rt.perception.process("static", img, ts=1001.0, force=True)
    assert again["events"] == []
    assert len(rt.memory.registry.list()) == 2


def test_move_within_bench_then_to_shelf_via_phone(rt):
    rt.perception.process("static", scene([("multimeter", .05, .4, .2, .6)]), ts=1000.0)
    out = rt.perception.process("static", scene([("multimeter", .8, .4, .95, .6)]), ts=1010.0)
    assert ("moved", "multimeter") in events_of(out)
    v = rt.memory.registry.find("multimeter")[0]
    assert (v["zone"], v["region"], v["camera"]) == ("bench", "right", "static")

    # The phone sees the shelf tag and the same multimeter.
    out = rt.perception.process("mobile", scene([("multimeter", .3, .5, .5, .8)], tag=1), ts=1020.0)
    assert out["zone"] == "shelf"
    assert ("moved", "multimeter") in events_of(out)
    v = rt.memory.registry.find("the multimeter")[0]
    assert (v["zone"], v["camera"]) == ("shelf", "mobile")
    assert len(rt.memory.registry.list()) == 1      # same entity across cameras


def test_disappeared_after_misses(rt):
    rt.perception.process("static", scene([("soldering iron", .05, .4, .25, .6),
                                           ("multimeter", .5, .4, .6, .6)]), ts=time.time() - 100)
    gone = []
    for i in range(4):
        out = rt.perception.process("static", scene([("multimeter", .5, .4, .6, .6)]),
                                    ts=time.time() - 50 + i, force=True)
        gone += [e for e in out["events"] if e["type"] == "disappeared"]
    assert [g["label"] for g in gone] == ["soldering iron"]


def test_teach_and_find_by_name(rt):
    rt.perception.process("mobile", scene([("hex key", .4, .4, .6, .6)], tag=2), ts=time.time())
    out = rt.agent.tools.run("teach_object", {"name": "2.5 mm hex key"})
    assert out["entity"]["taught"] and out["entity"]["name"] == "2.5 mm hex key"
    hits = rt.memory.registry.find("my 2.5 mm hex key")
    assert hits[0]["name"] == "2.5 mm hex key"
    # Seen again -> matched to the taught entity, not a new one.
    rt.perception.process("mobile", scene([("hex key", .1, .1, .3, .3)], tag=1), ts=time.time())
    v = rt.memory.registry.find("2.5 mm hex key")[0]
    assert v["zone"] == "shelf"
    assert len([e for e in rt.memory.registry.list() if e["concept"] == "hex key"]) == 1


def test_confidence_decays(rt):
    rt.perception.process("static", scene([("multimeter", .4, .4, .6, .6)]),
                          ts=time.time() - 4 * rt.settings.confidence_half_life)
    v = rt.memory.registry.find("multimeter")[0]
    assert v["stale"] and v["confidence"] < 0.1
    assert "may have moved" in rt.memory.registry.describe(v)


def test_private_mode_stores_nothing(rt):
    rt.perception.private = True
    out = rt.perception.process("static", scene([("multimeter", .4, .4, .6, .6)]), ts=1000.0)
    assert out["detections"] and out["events"] == [] and out["keyframe"] is None
    assert rt.memory.registry.list() == []
