"""System prompt and the compact situation summary the agent sees each turn."""
import time
from datetime import datetime

from forgemind.memory.registry import fmt_ago

SYSTEM = """You are ForgeMind, a workshop assistant with eyes, ears and memory.
You see the user's bench through a fixed camera and the rest of their space
through the phone they carry; you remember objects, sessions, steps and
measurements over days. The person carrying the phone is your body: when you
need to see something, ask them with go_look_at instead of guessing.

Rules:
- Your replies are spoken aloud. Keep them to 1-3 short sentences.
- Use tools for anything about objects, sessions, measurements or the scene.
  Do not answer from general knowledge when memory can answer.
- Never invent measurements or readings. If a value is missing, ask for it.
- Say where facts came from when it matters: seen by the bench camera, seen
  by the phone, said by the user, read from an instrument display.
- If an object's memory is stale (low confidence) or the bench changed,
  offer to check, or call go_look_at.
- "Just checking" / "don't log that" means log_step with tentative=true.
- Don't narrate what you see unprompted. Speak first only for safety.
- Physical help is virtual for now: highlight things, point to them, read
  them; you cannot pick things up.
"""


def situation(ctx):
    now = time.time()
    p, mem, safety = ctx.perception, ctx.memory, ctx.safety
    L = [f"Now: {datetime.now().strftime('%a %Y-%m-%d %H:%M')}"]
    phone_zone = p.phone_zone.current(now)
    L.append(f"Phone: {'in the ' + phone_zone if phone_zone else 'location unknown'}.")
    away = safety.user_away_for(now)
    L.append("User at the bench." if away < 30 else f"User away from the bench for {fmt_ago(away).replace(' ago', '')}.")
    s = mem.experiments.active()
    if s:
        st = mem.experiments.state(s["id"])
        L.append(f"Active session: '{st['session']['name']}' (id {st['session']['id']}); "
                 f"{len(st['done'])} steps done; next: {st['next'] or 'none planned'}.")
    else:
        L.append("No active session.")
    for cam in ("static", "mobile"):
        sc = p.scene(cam)
        name = "Bench camera" if cam == "static" else "Phone view"
        if sc["available"] and sc["age_s"] < 60:
            L.append(f"{name} now: {', '.join(sc['objects']) or 'nothing recognised'}"
                     + (f"; hazards: {', '.join(sc['hazards'])}" if sc["hazards"] else "") + ".")
    looks = ctx.hub.pending_looks()
    if looks:
        L.append("Waiting for the user to look at: " + ", ".join(l["target"] for l in looks) + ".")
    ev = [e for e in mem.db.recent_events(40)
          if e["type"] in ("appeared", "moved", "disappeared", "alert", "measurement", "step",
                           "look_done", "phone_zone")][:10]
    if ev:
        L.append("Recent events:")
        for e in ev:
            label = mem.registry.label(e["entity_id"]) if e["entity_id"] else ""
            pl = e["payload"]
            detail = {"moved": f"{label} moved {pl.get('from_zone') or ''}->{e['zone']}",
                      "appeared": f"{label} seen on {e['zone']}",
                      "disappeared": f"{label} gone from {e['zone']}",
                      "alert": pl.get("message", ""),
                      "measurement": f"{pl.get('quantity')} = {pl.get('value')} {pl.get('unit') or ''}",
                      "step": f"step '{pl.get('text')}' ({pl.get('status')})",
                      "look_done": f"user looked at {pl.get('target')}",
                      "phone_zone": f"phone entered {pl.get('zone')}"}.get(e["type"], e["type"])
            L.append(f"- {fmt_ago(now - e['ts'])}: {detail} [{e['source']}{', ' + e['camera'] if e['camera'] else ''}]")
    known = mem.registry.list(15)
    if known:
        L.append("Known objects: " + "; ".join(
            f"{v['label']} @ {v['zone'] or '?'} {v['ago']} (conf {v['confidence']})" for v in known))
    return "\n".join(L)
