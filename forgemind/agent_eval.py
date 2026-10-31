"""Agent eval: scripted user turns against the configured model (PRD §15.1).

    forgemind agent-eval            # uses FM_LLM_PROVIDER / models from .env

A small fixed world (bench camera sees a multimeter, a soldering iron and a
taught hex key; the phone is on the shelf) is built with a stand-in segmenter
and embedder, so only the reasoning model is real. Each turn records which
tools the model called, whether that matches the expected tool, the reply,
latency per model call and tokens, and an estimated cost.
"""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from forgemind.providers.base import Detection
from forgemind.providers.embedders import HashEmbedder

COLORS = {"soldering pen": (220, 30, 30), "multimeter": (30, 60, 220),
          "hex key": (30, 190, 60), "person": (240, 220, 40)}

# (user text, acceptable tool sets): the turn passes if the tools called match
# one of them exactly (as a set). {} = answering without a tool.
SCRIPT = [
    ("let's continue the soldering headers job from yesterday", [{"resume_session"}]),
    ("where is my hex key?", [{"where_is"}]),
    ("where did I leave the multimeter", [{"where_is"}, {"where_is", "highlight"}]),
    ("start a session called voltage divider test", [{"start_session"}]),
    ("the steps are: wire the divider, measure the output, swap R2 for 22k", [{"plan_steps"}]),
    ("okay, I wired the divider", [{"log_step"}]),
    # the measurement ticks off its planned step in code, so log_step alone is optional
    ("I measured 3.3 volts on the output", [{"log_measurement", "log_step"}, {"log_measurement"}]),
    ("set R2 to 10k", [{"set_config"}]),
    ("just checking the probe, don't log that", [{"log_step"}]),
    ("what's on the bench right now?", [{"inspect_scene"}]),
    ("watch the soldering iron while I'm away", [{"watch_bench"}]),
    ("where did we leave off?", [{"resume_session"}]),
    ("what was the output voltage again?", [set(), {"resume_session"}]),
    ("what's the capital of France?", [set()]),
    ("thanks, that's all for now", [set(), {"end_session"}]),
    # from the phone test: remarks must not act, corrections must correct
    ("thanks for locating it", [set()]),
    ("okay fine then.", [set()]),
    ("the logs for further improvement.", [set()]),
    ("that's not the multimeter, it's my stopwatch", [{"correct_object"}, {"correct_object", "go_look_at"}]),
]

# USD per 1M tokens (input, output), from dev.meta.ai pricing (Oct 26, 2026)
PRICES = {"muse-spark-1.3": (1.25, 4.25), "muse-spark-1.2": (1.25, 4.25),
          "muse-wonder-1.0": (4.0, 20.0)}


class ColorSegmenter:
    name = "color"

    def segment(self, image, concepts):
        a = np.asarray(image.convert("RGB")).astype(int)
        h, w = a.shape[:2]
        out = []
        for concept, rgb in COLORS.items():
            if concept not in concepts:
                continue
            m = (np.abs(a - np.array(rgb)).sum(axis=2) < 60)
            if m.sum() < 0.005 * h * w:
                continue
            ys, xs = np.nonzero(m)
            out.append(Detection(concept, (xs.min() / w, ys.min() / h, (xs.max() + 1) / w,
                                           (ys.max() + 1) / h), 0.9))
        return out


def scene(blocks=(), tag=None, size=(640, 480)):
    img = Image.new("RGB", size, (200, 200, 200))
    a = np.asarray(img).copy()
    w, h = size
    for concept, x0, y0, x1, y1 in blocks:
        a[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = COLORS[concept]
    if tag is not None:
        import cv2
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        m = cv2.aruco.generateImageMarker(d, tag, 120)
        pad = np.full((160, 160), 255, np.uint8)
        pad[20:140, 20:140] = m
        a[10:170, w - 170:w - 10] = np.stack([pad] * 3, axis=2)
    return Image.fromarray(a)


class TimedLLM:
    """Wraps the model to record each call's latency and usage."""

    def __init__(self, llm):
        self.llm, self.calls = llm, []
        self.name, self.supports_vision = llm.name, llm.supports_vision

    def model_for(self, role):
        return self.llm.model_for(role)

    def chat(self, messages, tools=None, model=None, json_mode=False):
        t0 = time.perf_counter()
        res = self.llm.chat(messages, tools=tools, model=model, json_mode=json_mode)
        self.calls.append({"model": model, "ms": round((time.perf_counter() - t0) * 1000),
                           "usage": res.usage or {}, "tools": [c.name for c in res.tool_calls]})
        return res


def build_world(settings, frames=False):
    from forgemind.providers import build_llm
    from forgemind.runtime import Runtime
    settings.confirm_new = False
    settings.concepts = list(COLORS)
    settings.send_frames = frames
    llm = TimedLLM(build_llm(settings))
    rt = Runtime(settings, llm=llm, segmenter=ColorSegmenter(), embedder=HashEmbedder(), warm=False)
    rt.ctx.capture_timeout = 2.0
    now = time.time()
    p = rt.perception
    p.process("mobile", scene([("hex key", .4, .4, .55, .6)], tag=0), ts=now - 600)
    rt.agent.tools.run("teach_object", {"name": "2.5 mm hex key"})
    p.process("static", scene([("multimeter", .45, .4, .6, .6), ("soldering pen", .05, .4, .25, .6),
                               ("hex key", .7, .4, .8, .55)]), ts=now - 300)
    p.process("mobile", scene([], tag=1), ts=now - 20)          # phone on the shelf
    tb = rt.agent.tools                                          # yesterday's unfinished job
    tb.run("start_session", {"name": "soldering headers"})
    tb.run("plan_steps", {"steps": ["tin the pads", "solder pin 1", "solder the other pins",
                                    "check continuity"]})
    tb.run("log_step", {"text": "tinned the pads"})
    tb.run("log_step", {"text": "soldered pin 1"})
    tb.run("end_session", {})
    for table in ("events", "steps", "measurements"):
        rt.memory.db.execute(f"UPDATE {table} SET ts = ts - 86400 WHERE session_id IS NOT NULL"
                             if table == "events" else f"UPDATE {table} SET ts = ts - 86400")
    return rt, llm


def cost(calls):
    usd = 0.0
    for c in calls:
        pin, pout = PRICES.get(c["model"], (1.25, 4.25))
        u = c["usage"]
        cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        usd += ((u.get("prompt_tokens", 0) - cached) * pin + cached * 0.15
                + u.get("completion_tokens", 0) * pout) / 1e6
    return usd


def run(settings, out=None, log=print, script=SCRIPT, frames=False):
    rt, llm = build_world(settings, frames)
    if rt.agent.system1:
        rt.agent.system1.warm_up()           # load time isn't a turn's latency
    rows = []
    try:
        for text, expected in script:
            n0 = len(llm.calls)
            t0 = time.perf_counter()
            if frames:                   # keep the camera frames fresh (< 30 s) like live use
                t = time.time()
                for cam, item in rt.perception.latest.items():
                    item["ts"] = t
            r = rt.agent.handle(text)
            ms = round((time.perf_counter() - t0) * 1000)
            fast = bool(r.get("fast"))
            if fast and rt.agent.last_verify_thread:
                rt.agent.last_verify_thread.join(30)  # Spark's background check: after the reply
            calls = llm.calls[n0:]
            tools = [t["tool"] for t in r["trace"]]
            ok = set(tools) in expected
            rows.append({"text": text, "expected": [sorted(e) for e in expected], "tools": tools, "ok": ok,
                         "reply": r["reply"], "ms": ms, "calls": len(calls), "fast": fast,
                         "call_ms": [c["ms"] for c in calls],
                         "prompt_tokens": [c["usage"].get("prompt_tokens") for c in calls],
                         "usd": round(cost(calls), 5)})
            log(f"{'OK ' if ok else 'XX '} {ms:5d} ms {'S1' if fast else '  '} {len(calls)} calls {tools!s:<34} {text!r}\n"
                f"      -> {r['reply'][:160]}")
    finally:
        rt.close()
    total = sum(r["usd"] for r in rows)
    ms = sorted(r["ms"] for r in rows)
    summary = {"turns": len(rows), "tool_accuracy": f"{sum(r['ok'] for r in rows)}/{len(rows)}",
               "median_ms": ms[len(ms) // 2], "max_ms": ms[-1],
               "mean_calls": round(sum(r["calls"] for r in rows) / len(rows), 2),
               "fast_turns": sum(r["fast"] for r in rows),
               "usd_total": round(total, 4)}
    log(json.dumps(summary))
    if out:
        Path(out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    return summary, rows
