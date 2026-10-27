"""Acceptance checks #4 (unattended iron alert) and #5 (look request on stale memory)."""
import time

from conftest import scene

SAFETY_EVERY_S = 5.0          # runtime loop period
LATENCY_TARGET_S = 10.0       # PRD §11: alert within 10 s of the unattended threshold


def test_unattended_iron_alerts_while_user_is_in_another_zone(rt):
    """#4: iron on the bench, user walks to the drawer with the phone."""
    p, s = rt.perception, rt.safety
    t0 = time.time()               # simulated clock runs forward from now
    # user at the bench with the iron, then leaves
    p.process("static", scene([("soldering iron", .1, .4, .3, .6), ("person", .5, .1, .9, .9)]), ts=t0)
    # the user walks off: a real departure changes the frame (the synthetic yellow
    # "person" is nearly the background's brightness, so mark it as a change)
    left = t0 + 1
    p.process("static", scene([("soldering iron", .1, .4, .3, .6)]), ts=left, force=True)
    p.process("mobile", scene([], tag=2), ts=t0 + 5)            # phone sees the drawer tag
    threshold = t0 + rt.settings.unattended_seconds      # last seen at the bench at t0
    fired_at, t = None, t0 + 2
    while t < t0 + 120 and fired_at is None:
        # bench camera keeps sending (mostly unchanged) frames of the iron, ~1/s
        p.process("static", scene([("soldering iron", .1, .4, .3, .6)]), ts=t)
        if (t - t0) % SAFETY_EVERY_S < 1:                         # safety loop tick
            if s.evaluate(t):
                fired_at = t
        t += 1
    assert fired_at is not None
    print(f"alert {fired_at - threshold:.0f} s after the unattended threshold")
    assert 0 <= fired_at - threshold <= LATENCY_TARGET_S
    alert = [m for m in rt.hub.outbox if m["type"] == "alert"][-1]
    assert "soldering iron" in alert["message"] and "Your phone is in the drawer." in alert["message"]


def test_unchanged_bench_frames_keep_the_iron_seen(rt):
    """Skipped (unchanged) frames still count as seeing what was there."""
    p, s = rt.perception, rt.safety
    t0 = time.time() - 100
    img = scene([("soldering iron", .1, .4, .3, .6)])
    p.process("static", img, ts=t0)
    for i in range(1, 14):
        out = p.process("static", img, ts=t0 + i)              # identical -> skipped by the gate
        assert out.get("skipped")
    assert s.last_seen["soldering iron"] == t0 + 13


def _stale_multimeter_on_bench(rt):
    p = rt.perception
    p.process("static", scene([("multimeter", .4, .4, .55, .6)]), ts=time.time() - 7200)
    rt.hub.outbox.clear()
    return rt.agent.tools.run("where_is", {"query": "multimeter"})


def test_stale_memory_asks_to_look_and_records_absence(rt):
    """#5: stale -> look request; user looks; it isn't there -> memory says so."""
    out = _stale_multimeter_on_bench(rt)
    assert out["stale"] and "Could you point the phone at the bench" in out["summary"]
    look = [m for m in rt.hub.outbox if m["type"] == "look_request"][0]
    assert look["zone"] == "bench"
    p, now = rt.perception, time.time()
    evs = []
    for i in range(4):                                         # phone at the bench, no multimeter
        evs += p.process("mobile", scene([], tag=0), ts=now + i)["events"]
    types = [e["type"] for e in evs]
    assert "look_done" in types and types.count("not_found") == 1
    assert [m["text"] for m in rt.hub.outbox if m["type"] == "say"][-1] == \
        "I don't see the multimeter on the bench. It must have moved."
    again = rt.agent.tools.run("where_is", {"query": "multimeter"})
    assert "But when you checked the bench" in again["summary"] and "it wasn't there" in again["summary"]


def test_stale_memory_look_finds_it_and_refreshes(rt):
    """#5, other branch: it is still there -> highlighted, memory fresh again."""
    _stale_multimeter_on_bench(rt)
    out = rt.perception.process("mobile", scene([("multimeter", .3, .3, .5, .5)], tag=0),
                                ts=time.time())
    types = [e["type"] for e in out["events"]]
    assert "found" in types and "not_found" not in types
    v = rt.memory.registry.find("multimeter")[0]
    assert not v["stale"] and v["camera"] == "mobile"
