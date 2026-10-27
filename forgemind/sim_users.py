"""Simulated users (PRD §15.1): a local model plays a person with a goal, in
their own words, against the real agent; success is judged in code from the
tools called and the memory state, never by a model.

    forgemind sim-users            # needs Muse Glimmer (or any compat model) as the user

The user model runs locally (default: Glimmer on llama.cpp at
http://127.0.0.1:8081/v1, FM_SIM_USER_URL) so only the agent's turns cost
API money. The world is agent_eval's fixed world (hex key taught and on the
bench, multimeter, iron; yesterday's "soldering headers" job).
"""
import json
import os
import time

from forgemind.agent_eval import build_world

PERSONAS = [
    {"name": "resume yesterday",
     "brief": ("You are a hobbyist at your workbench. Yesterday you started a job called "
               "'soldering headers' and didn't finish. Ask the assistant to pick it up and find "
               "out what's next. Then tell it you soldered the other pins."),
     "check": lambda tools, rt: ("resume_session" in tools and "log_step" in tools,
                                 "resumed and logged the next step")},
    {"name": "lost hex key",
     "brief": ("You can't find your 2.5 mm hex key. Ask the assistant where it is, in your own "
               "casual words. If it tells you where, thank it and stop."),
     "check": lambda tools, rt: ("where_is" in tools, "asked memory where the hex key is")},
    {"name": "measure and report",
     "brief": ("You are testing a voltage divider. Start a session for it, tell the assistant you "
               "measured 3.3 volts at the output, then ask it to write up a report."),
     "check": lambda tools, rt: (_measured(rt, 3.3) and "generate_report" in tools,
                                 "logged 3.3 V and generated a report")},
    {"name": "leaving the bench",
     "brief": ("You're about to step away to the kitchen and you left the soldering iron on the "
               "bench. Ask the assistant to keep an eye on it while you're gone."),
     "check": lambda tools, rt: ("watch_bench" in tools, "set a watch on the iron")},
]

USER_PROMPT = """You are role-playing a person talking to a voice assistant in their workshop.
Your situation and goal: {brief}
Speak in short, natural spoken sentences (one or two), the way people talk to a voice
assistant. Don't explain that you are role-playing. When your goal is done, or the
assistant can't help, reply with exactly: DONE"""


def _measured(rt, value):
    rows = rt.memory.db.all("SELECT value FROM measurements")
    return any(r["value"] is not None and abs(r["value"] - value) < 1e-6 for r in rows)


def user_model(url=None, model=None):
    from forgemind.providers.openai_compat import OpenAICompatModel
    url = url or os.environ.get("FM_SIM_USER_URL", "http://127.0.0.1:8081/v1")
    m = model or os.environ.get("FM_SIM_USER_MODEL", "glimmer")
    return OpenAICompatModel(url, "local", m, m, name="sim-user")


def run_persona(settings, persona, user, max_turns=6, log=print):
    rt, llm = build_world(settings)
    convo, tools, turns = [], [], []
    try:
        for _ in range(max_turns):
            msgs = [{"role": "system", "content": USER_PROMPT.format(brief=persona["brief"])}]
            for who, text in convo:                    # roles flipped: the agent is the "user"
                msgs.append({"role": "assistant" if who == "user" else "user", "content": text})
            if not convo:
                msgs.append({"role": "user", "content": "(The assistant is listening.)"})
            said = (user.chat(msgs).content or "").strip().strip('"')
            if not said or said.upper().startswith("DONE"):
                break
            t0 = time.perf_counter()
            r = rt.agent.handle(said)
            ms = round((time.perf_counter() - t0) * 1000)
            tools += [t["tool"] for t in r["trace"]]
            turns.append({"user": said, "agent": r["reply"], "tools": [t["tool"] for t in r["trace"]],
                          "ms": ms})
            log(f"   user: {said}\n  agent: {r['reply'][:150]}  [{ms} ms, {[t['tool'] for t in r['trace']]}]")
            convo += [("user", said), ("agent", r["reply"])]
        ok, what = persona["check"](tools, rt)
        guards = len(rt.memory.db.all("SELECT id FROM events WHERE type='guard'"))
        usd = round(sum(_cost(c) for c in llm.calls), 4)
        return {"persona": persona["name"], "success": bool(ok), "criterion": what,
                "turns": len(turns), "guard_events": guards, "usd": usd, "transcript": turns}
    finally:
        rt.close()


def _cost(call):
    from forgemind.providers.budget import token_cost
    return token_cost(call["model"], call["usage"])


def run(settings, out=None, log=print, personas=PERSONAS):
    user = user_model()
    results = []
    for p in personas:
        log(f"\n== {p['name']}")
        r = run_persona(settings, p, user, log=log)
        log(f"   -> {'PASS' if r['success'] else 'FAIL'}: {r['criterion']} ({r['turns']} turns)")
        results.append(r)
    summary = {"personas": len(results), "passed": sum(r["success"] for r in results),
               "turns": sum(r["turns"] for r in results),
               "guard_events": sum(r["guard_events"] for r in results),
               "usd": round(sum(r["usd"] for r in results), 4)}
    log(json.dumps(summary))
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
    return summary, results
