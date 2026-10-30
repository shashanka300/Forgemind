import sys, time, runpy
sys.argv = ["x"]
ns = {}
src = open(sys.argv[0] if False else __file__.replace("laya_zero_shot_variants.py", "laya_zero_shot.py"), encoding="utf-8").read()
src = src.split("agent = laya.load")[0]          # reuse data + descriptions only
exec(src, ns)
import laya
agent = laya.load("convaiinnovations/laya", device="cuda")
DATA, AMBIG, SHORT = ns["DATA"], ns["AMBIG"], ns["SHORT"]
items = [(u, {lab}) for lab, us in DATA.items() for u in us] + AMBIG

# A: one yes/no question per tool, pick the highest
NOUL = {t: {"type": "noul", "instructions": f"Does the user want the assistant to {d}?" if t != "chat"
            else "Is this small talk or a general question that needs no workshop action?"} for t, d in SHORT.items()}
NOUL["log_step"]["instructions"] = "Is the user reporting a step of work they just did?"
NOUL["log_measurement"]["instructions"] = "Is the user reporting a value they measured?"
NOUL["set_config"]["instructions"] = "Is the user stating or setting a component value or setting?"
ok_n, ms, rows = 0, [], []
for u, ok in items:
    t = time.perf_counter(); r = agent.predict(u, NOUL)["answers"]; ms.append((time.perf_counter() - t) * 1000)
    p = sorted(((a["noul"], k) for k, a in r.items()), reverse=True)
    rows.append((u, ok, p[0][1], p[0][0], p[0][0] - p[1][0]))
n = len(rows)
print(f"A per-tool yes/no: {sum(c in ok for _, ok, c, *_ in rows)}/{n}, median {sorted(ms)[n // 2]:.0f} ms")
for thr in (0.5, 0.8, 0.9):
    sel = [c in ok for _, ok, c, p, m in rows if p >= thr and m >= 0.3]
    print(f"   top>={thr} & margin>=0.3: covers {len(sel)}/{n}, wrong {len(sel) - sum(sel)}")

# B: two levels
CAT = {"objects": "finding, showing, looking at or describing objects and what the cameras see",
       "instrument": "reading a meter or instrument display",
       "safety": "watching or monitoring something for safety",
       "teach": "naming or remembering a new object",
       "session": "starting, continuing, ending, planning, comparing or reporting an experiment session",
       "record": "the user tells what they did, measured or set",
       "chat": "small talk or a general question"}
CAT_OF = {"where_is": "objects", "inspect_scene": "objects", "go_look_at": "objects", "highlight": "objects",
          "read_closeup": "instrument", "watch_bench": "safety", "teach_object": "teach",
          "start_session": "session", "resume_session": "session", "end_session": "session", "plan_steps": "session",
          "compare_sessions": "session", "generate_report": "session",
          "log_step": "record", "log_measurement": "record", "set_config": "record", "chat": "chat"}
cq = {"cat": {"type": "choice", "instructions": "What kind of request is this?", "criteria": CAT}}
cat_ok, tool_ok, wrong_conf = 0, 0, []
for u, ok in items:
    a = agent.predict(u, cq)["answers"]["cat"]
    want = {CAT_OF[o] for o in ok}
    cat_ok += a["choice"] in want
    sub = {t: SHORT[t] for t, c in CAT_OF.items() if c == a["choice"]}
    if len(sub) == 1:
        pick = next(iter(sub))
    else:
        pick = agent.predict(u, {"t": {"type": "choice", "instructions": "Which action?", "criteria": sub}})["answers"]["t"]["choice"]
    tool_ok += pick in ok
    if a["choice"] not in want: wrong_conf.append((round(a["answer_confidence"], 2), a["choice"], u))
print(f"B two-level: category {cat_ok}/{n}, tool {tool_ok}/{n}")
print("   category errors:", sorted(wrong_conf, reverse=True)[:12])
