"""Zero-shot Laya on ForgeMind turns: tool choice, compound detection, object choice."""
import os, sys, time, json
os.environ["USE_TF"] = "0"
sys.path.insert(0, "F:/Forgemind")
import laya
from forgemind.agent.tools import SPECS, AGENT_HIDDEN

# (utterance, acceptable tools); "chat" = answer without a tool
DATA = {
    "where_is": ["where is my hex key?", "where did I leave the multimeter", "have you seen my soldering iron",
                 "I can't find the wire cutters", "where's the 10k resistor", "do you know where I put my screwdriver",
                 "find my pliers", "any idea where the breadboard went"],
    "inspect_scene": ["what's on the bench right now?", "what do you see on the bench", "what can the phone see",
                      "is there anything on the bench", "tell me what's in front of the camera"],
    "go_look_at": ["take a look at the shelf", "go check the drawer for me", "look over at the left side of the bench"],
    "read_closeup": ["read the multimeter", "what does the meter say", "can you read the display",
                     "read the value off the meter for me", "what's the reading on the screen"],
    "highlight": ["show me the hex key on the screen", "highlight the pliers", "point out the multimeter in the view",
                  "mark the screwdriver on my phone"],
    "watch_bench": ["watch the soldering iron while I'm away", "keep an eye on the iron", "let me know if there's smoke",
                    "monitor the bench while I grab coffee", "warn me if I leave the iron on the bench"],
    "teach_object": ["this is my soldering iron", "remember this as my favourite screwdriver", "call this the blue multimeter",
                     "this one's my 2.5 mm hex key", "learn this, it's my flux pen"],
    "start_session": ["start a session called voltage divider test", "new experiment: LED driver",
                      "let's begin a new job, soldering headers", "start logging a session for the motor test"],
    "resume_session": ["let's continue the soldering headers job from yesterday", "where did we leave off?", "what's next?",
                       "pick up the divider test again", "what was I doing yesterday", "what step am I on"],
    "end_session": ["that's it, end the session", "wrap up the session", "I'm done with this experiment, close it",
                    "stop the session"],
    "plan_steps": ["the steps are: wire the divider, measure the output, swap R2 for 22k",
                   "plan: tin the pads, solder the headers, check continuity",
                   "first I'll strip the wires, then crimp them, then test", "add these steps: flash firmware, test LEDs"],
    "log_step": ["okay, I wired the divider", "done soldering the headers", "I finished tinning the pads",
                 "just checking the probe, don't log that", "flashed the firmware", "swapped R2"],
    "log_measurement": ["I measured 3.3 volts on the output", "the output is 4.98 V", "current draw is 120 milliamps",
                        "resistance reads 9.8k", "got 2.1 volts across the LED"],
    "set_config": ["set R2 to 10k", "I'm using a 22k resistor for R1", "iron temperature is set to 330",
                   "change the input to 9 V", "R1 is 4.7k"],
    "compare_sessions": ["compare this run with yesterday's", "how does this compare to the last session",
                         "what changed since the previous run", "is the output different from last time"],
    "generate_report": ["write it up", "generate the report", "make a report of today's experiment",
                        "can you write up the divider test"],
    "chat": ["what's the capital of France?", "how does a voltage divider work", "what's a good temperature for leaded solder",
             "hello", "tell me a joke", "why would an LED not light up"],
}
AMBIG = [("thanks, that's all for now", {"chat", "end_session"}),
         ("what was the output voltage again?", {"chat", "resume_session"}),
         ("the supply is at 12 volts", {"set_config", "log_measurement"})]
COMPOUND = ["I measured 3.3 V. Please write up a report.", "end the session and write the report",
            "where's my multimeter? and start a session called LED test",
            "log that I wired the divider and set R2 to 10k", "read the meter, then compare it with yesterday"]
OBJECTS = ["hex key", "multimeter", "soldering iron", "wire cutters", "pliers", "breadboard", "circuit board",
           "headphones", "screwdriver", "copper wire"]
OBJ_TESTS = [("where is my hex key?", "hex key"), ("where did I leave the multimeter", "multimeter"),
             ("have you seen my soldering iron", "soldering iron"), ("I can't find the wire cutters", "wire cutters"),
             ("find my pliers", "pliers"), ("any idea where the breadboard went", "breadboard"),
             ("where's the arduino", "circuit board"), ("where's my allen key", "hex key"),
             ("where did my DMM go", "multimeter"), ("where's the spool of magnet wire", "copper wire"),
             ("where are my snips", "wire cutters"), ("find my headset", "headphones")]

SHORT = {"where_is": "find where an object is or was last seen", "inspect_scene": "describe what the cameras see right now",
         "go_look_at": "go and look at a place or object with the phone", "read_closeup": "read an instrument display or meter",
         "highlight": "highlight or point out an object on the phone screen",
         "watch_bench": "watch or monitor something for safety while the user is away",
         "teach_object": "remember the object in front of the phone under a name",
         "start_session": "start a new experiment or work session", "resume_session": "continue a previous session, or say what's next",
         "end_session": "end or close the current session", "plan_steps": "list the planned steps for the session",
         "log_step": "the user reports a step they did", "log_measurement": "the user reports a measured value",
         "set_config": "the user sets or states a configuration value or component",
         "compare_sessions": "compare this run with an earlier run", "generate_report": "write the experiment report",
         "chat": "small talk or a general question that needs no workshop action"}
LONG = {n: d for n, (_, d) in SPECS.items() if n not in AGENT_HIDDEN}
LONG["chat"] = SHORT["chat"]

agent = laya.load("convaiinnovations/laya", device="cuda")
INSTR = "Which action should the workshop assistant take for what the user just said?"
MULTI = {"type": "noul", "instructions": "Does the user ask for two or more different actions in this message?"}

def run(criteria):
    q = {"tool": {"type": "choice", "instructions": INSTR, "criteria": criteria}, "multi": MULTI}
    rows, ms = [], []
    items = [(u, {lab}) for lab, us in DATA.items() for u in us] + AMBIG
    for u, ok in items:
        t = time.perf_counter(); r = agent.predict(u, q); ms.append((time.perf_counter() - t) * 1000)
        a = r["answers"]["tool"]
        rows.append((u, ok, a["choice"], a["answer_confidence"], r["answers"]["multi"]["noul"]))
    comp = [agent.predict(u, q)["answers"] for u in COMPOUND]
    return rows, comp, ms

def report(name, rows, comp, ms):
    n = len(rows); correct = sum(c in ok for _, ok, c, _, _ in rows)
    print(f"\n== {name}: top-1 {correct}/{n} = {correct / n:.2f}; median {sorted(ms)[len(ms) // 2]:.0f} ms")
    for thr in (0.5, 0.7, 0.8, 0.9, 0.95):
        sel = [(c in ok) for _, ok, c, conf, _ in rows if conf >= thr]
        acc = sum(sel) / len(sel) if sel else 0
        print(f"  conf>={thr}: covers {len(sel)}/{n} ({len(sel) / n:.0%}), accuracy {acc:.2f}, wrong {len(sel) - sum(sel)}")
    print("  multi on single turns: max", round(max(m for *_, m in rows), 3), " >0.5:", sum(m > 0.5 for *_, m in rows))
    print("  multi on compound turns:", [round(a["multi"]["noul"], 2) for a in comp],
          " their tool picks:", [(a["tool"]["choice"], round(a["tool"]["answer_confidence"], 2)) for a in comp])
    print("  errors:")
    for u, ok, c, conf, _ in rows:
        if c not in ok:
            print(f"    {conf:.2f} {c:16s} want {'/'.join(sorted(ok)):16s} | {u}")

for name, crit in (("short descriptions", SHORT), ("tool descriptions", LONG)):
    report(name, *run(crit))

oq = {"object": {"type": "choice", "instructions": "Which object is the user asking about?",
                 "criteria": {o: o for o in OBJECTS}}}
hits = []
for u, want in OBJ_TESTS:
    a = agent.predict(u, oq)["answers"]["object"]
    hits.append(a["choice"] == want)
    if a["choice"] != want:
        print(f"  object miss: {u!r} -> {a['choice']} ({a['answer_confidence']:.2f}), want {want}")
print(f"\n== object choice: {sum(hits)}/{len(hits)}")
