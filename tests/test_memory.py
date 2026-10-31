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


def test_teach_uses_several_recent_views(rt):
    now = time.time()
    for i, x in enumerate((.40, .42, .38)):
        rt.perception.process("mobile", scene([("hex key", x, .4, x + .2, .6)], tag=2),
                              ts=now - 2 + i)
    out = rt.agent.tools.run("teach_object", {"name": "2.5 mm hex key"})
    assert out["views"] == ["detected"] * 3 and out["entity"]["examples"] == 3
    assert "3 views" in out["summary"] and "Turn it" not in out["summary"]
    assert out["entity"]["concept"] == "hex key"


def test_teach_asks_segmenter_when_frame_missed_it(rt):
    rt.perception.process("mobile", scene([("hex key", .4, .4, .6, .6)], tag=2), ts=time.time())
    rt.perception.recent[-1]["detections"] = []          # the live detector missed it
    out = rt.agent.tools.run("teach_object", {"name": "small key", "concept": "hex key"})
    assert out["views"] == ["asked"] and out["entity"]["concept"] == "hex key"
    assert "Turn it" in out["summary"]


def test_teach_refuses_when_it_cannot_pick_out_the_object(rt):
    rt.perception.process("mobile", scene([], tag=2), ts=time.time())
    out = rt.agent.tools.run("teach_object", {"name": "mystery part"})
    assert "can't pick out" in out["error"] and rt.memory.registry.find("mystery part") == []


def test_small_talk_does_not_teach(rt):
    rt.perception.process("mobile", scene([("hex key", .4, .4, .6, .6)], tag=2), ts=time.time())
    rt.agent.tools.utterance = "thanks for locating it"
    assert "say its name" in rt.agent.tools.run("teach_object", {"name": "baby's biter"})["error"]
    rt.agent.tools.utterance = "these are my hex keys"
    assert "error" not in rt.agent.tools.run("teach_object", {"name": "hex key"})


def test_just_taught_object_can_be_found(rt):
    rt.perception.process("mobile", scene([("hex key", .4, .4, .6, .6)], tag=1), ts=time.time())
    v = rt.agent.tools.run("teach_object", {"name": "small key"})["entity"]
    found = rt.memory.registry.find("small key")[0]
    assert found["last_seen"] and found["camera"] == "mobile" and found["id"] == v["id"]


def test_teaching_a_lookalike_of_another_taught_object_is_refused(rt):
    rt.perception.process("mobile", scene([("hex key", .4, .4, .6, .6)], tag=2), ts=time.time())
    rt.agent.tools.run("teach_object", {"name": "mouse"})
    out = rt.agent.tools.run("teach_object", {"name": "biter"})
    assert "looks like your mouse" in out["error"]
    assert rt.memory.registry.find("biter") == []


def test_correct_object_moves_the_label(rt):
    img = scene([("hex key", .4, .4, .6, .6)], tag=2)
    rt.perception.process("mobile", img, ts=time.time())
    rt.agent.tools.run("teach_object", {"name": "biter"})
    rt.perception.process("mobile", img, ts=time.time(), force=True)
    assert rt.perception.latest["mobile"]["detections"][0]["label"] == "biter"
    out = rt.agent.tools.run("correct_object", {"wrong_name": "biter", "right_name": "mouse"})
    assert out["summary"].startswith("Fixed") and out["removed_views"] >= 1
    assert rt.memory.registry.find("biter") == []            # nothing else was ever taught as it
    rt.perception.process("mobile", img, ts=time.time(), force=True)
    assert rt.perception.latest["mobile"]["detections"][0]["label"] == "mouse"


def test_weak_word_overlap_is_not_a_match(rt):
    rt.perception.process("mobile", scene([("hex key", .4, .4, .6, .6)], tag=2), ts=time.time())
    rt.agent.tools.run("teach_object", {"name": "baby's biter"})
    assert not [v for v in rt.memory.registry.find("baby's hand") if v["label"] == "baby's biter"]
    assert rt.memory.registry.find("biter")[0]["label"] == "baby's biter"


def test_teach_ignores_old_frames(rt):
    rt.perception.process("mobile", scene([("hex key", .4, .4, .6, .6)], tag=2),
                          ts=time.time() - 30)
    out = rt.agent.tools.run("teach_object", {"name": "old key"})
    assert len(out["views"]) == 1          # falls back to the latest frame only


def test_new_object_needs_two_frames_when_confirming(rt):
    rt.settings.confirm_new = True
    img = scene([("multimeter", .45, .4, .6, .6)])
    first = rt.perception.process("static", img, ts=1000.0)
    assert first["events"] == [] and first["detections"][0]["pending"]
    second = rt.perception.process("static", img, ts=1001.0, force=True)
    assert events_of(second) == [("appeared", "multimeter")]
    # a one-frame blip elsewhere never becomes an entity
    rt.perception.process("static", scene([("multimeter", .45, .4, .6, .6),
                                           ("hex key", .05, .05, .15, .15)]), ts=1002.0, force=True)
    rt.perception.process("static", img, ts=1003.0, force=True)
    assert [e["concept"] for e in rt.memory.registry.list()] == ["multimeter"]


def test_still_scene_still_confirms_new_object(rt):
    # the second, identical frame would be gate-skipped; a pending candidate
    # must still be looked at again or it never becomes an entity
    rt.settings.confirm_new = True
    img = scene([("multimeter", .45, .4, .6, .6)])
    rt.perception.process("static", img, ts=1000.0)
    second = rt.perception.process("static", img, ts=1001.0)
    assert not second.get("skipped")
    assert events_of(second) == [("appeared", "multimeter")]
    assert rt.perception.process("static", img, ts=1002.0).get("skipped")


def test_held_object_never_creates_entity_on_bench(rt):
    # yellow "person" block (a hand/arm) covers most of the hex key
    img = scene([("person", .38, .38, .60, .60), ("hex key", .40, .40, .50, .50)])
    out = rt.perception.process("static", img, ts=1000.0)
    held = [d for d in out["detections"] if d["concept"] == "hex key"][0]
    assert held["held"] and held["entity_id"] is None
    assert rt.memory.registry.list() == []


def test_covered_object_is_not_reported_gone(rt):
    t = time.time() - 100
    rt.perception.process("static", scene([("multimeter", .5, .4, .6, .6)]), ts=t)
    gone = []
    for i in range(1, 16):          # an arm (person block) covers it for 15 s
        out = rt.perception.process("static", scene([("person", .45, .35, .7, .7)]),
                                    ts=t + i, force=True)
        gone += [e for e in out["events"] if e["type"] == "disappeared"]
    assert gone == []
    for i in range(16, 30):         # arm gone, multimeter really missing
        out = rt.perception.process("static", scene([]), ts=t + i, force=True)
        gone += [e for e in out["events"] if e["type"] == "disappeared"]
    assert [g["label"] for g in gone] == ["multimeter"]


def test_untaught_object_moved_on_bench_is_linked_not_duplicated(rt, monkeypatch):
    # The object looks a bit different after the move (below match_min_sim but
    # above the relaxed move-link threshold): moved, not new + disappeared.
    from forgemind.perception import pipeline as pl
    t = time.time() - 100
    rt.perception.process("static", scene([("multimeter", .05, .4, .2, .6)]), ts=t)
    eid = rt.memory.registry.list()[0]["id"]
    rt.perception.process("static", scene([]), ts=t + 2, force=True)      # left its spot
    real_match = rt.memory.registry.match
    monkeypatch.setattr(rt.memory.registry, "match",
                        lambda *a, **k: (None, 0.8))                       # strict match fails
    out = rt.perception.process("static", scene([("multimeter", .75, .4, .9, .6)]),
                                ts=t + 4, force=True)
    monkeypatch.setattr(rt.memory.registry, "match", real_match)
    assert ("moved", "multimeter") in events_of(out)
    assert [e["id"] for e in rt.memory.registry.list()] == [eid]


def test_one_label_per_object_and_no_scene_boxes():
    from forgemind.perception.pipeline import one_label_per_object
    from forgemind.providers.base import Detection
    dets = [Detection("phone", (0.1, 0.1, 0.3, 0.3), 0.90),
            Detection("multimeter", (0.1, 0.1, 0.31, 0.3), 0.95),
            Detection("soldering station", (0.0, 0.0, 1.0, 0.95), 0.68),
            Detection("hand", (0.0, 0.0, 0.9, 0.9), 0.9)]
    assert sorted(d.concept for d in one_label_per_object(dets)) == ["hand", "multimeter"]


def test_watch_matches_taught_name_not_just_detector_concept(rt):
    # SAM calls the iron a "screwdriver"; the user taught it as "soldering iron"
    from conftest import COLORS
    COLORS["screwdriver"] = COLORS["soldering iron"]
    try:
        rt.settings.concepts = ["screwdriver", "person"]
        rt.perception.process("mobile", scene([("screwdriver", .4, .4, .6, .6)], tag=0),
                              ts=time.time())
        rt.agent.tools.run("teach_object", {"name": "Soldering Iron"})
        now = time.time()
        rt.safety.last_present = now - 600                 # user away for 10 min
        rt.perception.process("static", scene([("screwdriver", .05, .4, .25, .6)]), ts=now)
        alerts = rt.safety.evaluate(now)
        assert [a["severity"] for a in alerts] == ["warning"]
        assert "soldering iron" in alerts[0]["message"]
    finally:
        del COLORS["screwdriver"]


def test_prune_keeps_referenced_and_recent_keyframes(rt):
    import os
    mem = rt.memory
    for i in range(3):                              # three frames; only the last is referenced
        rt.perception.process("static", scene([("multimeter", .3 + .1 * i, .4, .45 + .1 * i, .6)]),
                              ts=time.time() - 3600 + i, force=True)
    files = sorted(mem.keyframes_dir.glob("*.jpg"))
    assert len(files) == 3
    old = time.time() - 3600
    for p in files:
        os.utime(p, (old, old))
    (mem.keyframes_dir / "fresh.jpg").write_bytes(b"x")       # recent, unreferenced
    removed = mem.prune_keyframes(older_than_s=600)
    left = {p.name for p in mem.keyframes_dir.glob("*.jpg")}
    referenced = {v for q in mem.KEYFRAME_REFS for r in mem.db.all(q) for v in r.values() if v}
    assert "fresh.jpg" in left and referenced <= left and removed == 3 - len(referenced & {p.name for p in files})
