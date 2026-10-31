"""HTTP + WebSocket server: phone page, camera frames, chat, memory inspector.

    POST /api/frame?camera=static|mobile     JPEG body -> detections + events
    POST /api/frame?capture_id=N             close-up answering read_closeup
    POST /api/chat        {"text": "..."}     -> {"reply", "trace"}
    GET  /api/state                           situation for the phone UI
    GET  /api/entities    POST/DELETE /api/entities/{id}
    GET  /api/sessions    GET /api/sessions/{id}
    GET  /api/events      POST /api/privacy   POST /api/teach
    WS   /ws                                  alerts, look/capture requests, highlights

Access: the token from the printed link is swapped for a 30-day HttpOnly
cookie (same scheme as the gallery app). Scripts may send
`Authorization: Bearer <token>` instead.
"""
import asyncio
import hmac
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse
from pydantic import BaseModel

STATIC = Path(__file__).resolve().parent / "static"
COOKIE = "forgemind_token"
MAX_FRAME = 8 * 1024 * 1024
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
LOCKED = """<!doctype html><meta name=viewport content="width=device-width">
<body style="font:16px system-ui;background:#0f1012;color:#e9e6df;padding:40px 24px">
<h2>ForgeMind is locked</h2>__ERROR__<form method="post" action="/login">
<p><input name="token" type="password" placeholder="Access token" autofocus autocomplete="current-password"
 style="font:16px system-ui;padding:10px;width:100%;max-width:320px;box-sizing:border-box"></p>
<button style="font:16px system-ui;padding:10px 20px">Unlock</button></form></body>"""


class ChatIn(BaseModel):
    text: str


BENCH_PAGE = """<!doctype html><meta charset="utf-8"><title>ForgeMind bench camera</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{margin:0;background:#111;color:#ddd;font:14px system-ui;text-align:center}
img{max-width:100%;max-height:92vh}#s{padding:6px}</style>
<div id="s">waiting for the bench camera...</div><img id="f" alt="">
<script>
const f = document.getElementById("f"), s = document.getElementById("s");
async function tick() {
  const r = await fetch("/api/cameras/static/latest.jpg", {credentials: "same-origin"}).catch(() => null);
  if (r && r.ok) { const u = URL.createObjectURL(await r.blob()), old = f.src; f.src = u; if (old) URL.revokeObjectURL(old); }
  const st = await fetch("/api/state", {credentials: "same-origin"}).then(r => r.json()).catch(() => null);
  const b = st && st.scenes && st.scenes.static;
  if (b && b.available) s.textContent = "Live. Boxes update when the bench changes (last " + Math.round(b.age_s) + " s ago): " + (b.objects.join(", ") || "nothing recognised");
  else if (r && !r.ok) s.textContent = "No frames: the bench camera isn't streaming.";
}
tick(); setInterval(tick, 400);
</script>"""


class SayIn(BaseModel):
    text: str


class TeachIn(BaseModel):
    name: str
    concept: str | None = None


class EntityIn(BaseModel):
    name: str | None = None
    zone: str | None = None


class PrivacyIn(BaseModel):
    paused: bool | None = None
    private: bool | None = None


def create_app(rt):
    token = rt.settings.token

    @asynccontextmanager
    async def lifespan(app):
        rt.hub.attach_loop(asyncio.get_running_loop())
        rt.start_background()
        yield

    app = FastAPI(title="ForgeMind", lifespan=lifespan)

    def authorized(cookies, headers, query):
        if not token:
            return True
        given = (cookies.get(COOKIE) or query.get("t")
                 or (headers.get("authorization") or "").removeprefix("Bearer ").strip())
        return bool(given) and hmac.compare_digest(given, token)

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if request.url.path in ("/healthz", "/login") or not token:
            return await call_next(request)
        q = request.query_params
        if hmac.compare_digest(request.cookies.get(COOKIE, ""), token):
            return await call_next(request)
        bearer = (request.headers.get("authorization") or "").removeprefix("Bearer ").strip()
        if bearer and hmac.compare_digest(bearer, token):
            return await call_next(request)
        if q.get("t") and hmac.compare_digest(q["t"], token):
            # Trade the URL token for a cookie and drop it from the address bar.
            rest = "&".join(f"{k}={v}" for k, v in q.items() if k != "t")
            resp = RedirectResponse(request.url.path + (f"?{rest}" if rest else ""), 303)
            resp.set_cookie(COOKIE, token, max_age=2592000, httponly=True, samesite="lax",
                            secure=request.headers.get("x-forwarded-proto") == "https")
            return resp
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": "locked"}, 401)
        return HTMLResponse(LOCKED.replace("__ERROR__", ""), 401)

    @app.post("/login")
    async def login(request: Request):
        """The locked page's form: the right token gets the same 30-day cookie as ?t=."""
        from urllib.parse import parse_qs
        body = (await request.body()).decode("utf-8", "replace")
        given = (parse_qs(body).get("token") or [""])[0].strip()
        if token and hmac.compare_digest(given.encode(), token.encode()):
            resp = RedirectResponse("/", 303)
            resp.set_cookie(COOKIE, token, max_age=2592000, httponly=True, samesite="lax",
                            secure=request.headers.get("x-forwarded-proto") == "https")
            return resp
        await asyncio.sleep(1)             # no fast guessing
        return HTMLResponse(LOCKED.replace("__ERROR__", "<p>That token isn't right.</p>"), 401)

    # -- pages and files ------------------------------------------------------

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "phone.html", headers={"Cache-Control": "no-store"})

    @app.get("/bench")
    def bench_page():
        """Live preview of the bench camera, with what it detects."""
        return HTMLResponse(BENCH_PAGE, headers={"Cache-Control": "no-store"})

    @app.get("/api/cameras/{camera}/latest.jpg")
    def latest_frame(camera: str):
        """The newest processed frame of a camera, with its detections drawn."""
        from io import BytesIO

        from fastapi.responses import Response
        from PIL import ImageDraw
        item = rt.perception.latest.get(camera) if camera in ("static", "mobile") else None
        newest = rt.perception.newest.get(camera)
        if not item and not newest:
            raise HTTPException(404, "no frame yet")
        item = item or {}
        img = (newest or item["image"]).convert("RGB")
        img.thumbnail((1280, 1280))
        W, H = img.size
        draw = ImageDraw.Draw(img)
        for d in item.get("detections") or []:
            x0, y0, x1, y1 = d["box"]
            color = (240, 200, 60) if d.get("kind") == "presence" else (95, 211, 138)
            draw.rectangle((x0 * W, y0 * H, x1 * W, y1 * H), outline=color, width=2)
            draw.text((x0 * W + 4, y0 * H + 3), d.get("label") or d["concept"], fill=color)
        buf = BytesIO()
        img.save(buf, "JPEG", quality=80)
        return Response(buf.getvalue(), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    def _safe(root, name):
        if not SAFE_NAME.match(name):
            raise HTTPException(404)
        path = root / name
        if not path.is_file():
            raise HTTPException(404)
        return path

    @app.get("/keyframes/{name}")
    def keyframe(name: str):
        return FileResponse(_safe(rt.memory.keyframes_dir, name),
                            headers={"Cache-Control": "max-age=31536000, immutable"})

    @app.get("/reports/{name}")
    def report(name: str, raw: int = 0):
        md = _safe(rt.memory.reports_dir, name).read_text(encoding="utf-8")
        if raw:
            return PlainTextResponse(md, media_type="text/markdown; charset=utf-8")
        from forgemind.reports import to_html
        return HTMLResponse(to_html(md, title=name))

    # -- cameras --------------------------------------------------------------

    @app.post("/api/frame")
    async def frame(request: Request, camera: str = "mobile", capture_id: int | None = None):
        data = await request.body()
        if not data or len(data) > MAX_FRAME:
            raise HTTPException(413 if data else 400, "frame missing or too large")
        if capture_id is not None:
            ok = rt.hub.deliver_capture(capture_id, data)
            return {"delivered": ok}
        try:
            out = await run_in_threadpool(rt.perception.process, camera, data)
        except ValueError as e:
            raise HTTPException(400, str(e))
        return out

    # -- agent ----------------------------------------------------------------

    @app.post("/api/chat")
    async def chat(body: ChatIn):
        text = body.text.strip()
        if not text:
            raise HTTPException(400, "empty message")
        try:
            return await run_in_threadpool(rt.agent.handle, text)
        except Exception as e:
            return JSONResponse({"reply": f"Something went wrong talking to the model: {e}",
                                 "error": str(e), "trace": []}, 502)

    @app.post("/api/tts")
    async def tts(body: SayIn):
        """A reply in the natural voice (MP3); 503 -> the phone uses its own voice."""
        from fastapi.responses import Response
        sp = rt.speaker
        if not sp or sp.loading or sp.error:
            raise HTTPException(503, "voice not available")
        try:
            t0 = time.perf_counter()
            audio = await run_in_threadpool(sp.speak, body.text[:1500])
            rt.memory.db.log_event("voice", "inferred", {"ms": round((time.perf_counter() - t0) * 1000),
                                                         "chars": len(body.text)})
        except ValueError as e:
            raise HTTPException(400, str(e))
        return Response(audio, media_type="audio/mpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/teach")
    async def teach(body: TeachIn):
        from forgemind.agent.tools import TeachObject
        out = await run_in_threadpool(rt.agent.tools.run, "teach_object",
                                      TeachObject(name=body.name, concept=body.concept).model_dump())
        return out

    # -- state and memory inspector -------------------------------------------

    @app.get("/api/state")
    def state():
        now = time.time()
        s = rt.memory.experiments.active()
        return {
            "providers": rt.describe(),
            "spend": ({"usd": round(rt.ledger.total(), 2), "cap": rt.ledger.cap}
                      if rt.ledger else None),
            "phone_zone": rt.perception.phone_zone.current(now),
            "user_away_s": round(rt.safety.user_away_for(now)),
            "session": rt.memory.experiments.state(s["id"]) if s else None,
            "scenes": {c: rt.perception.scene(c) for c in ("static", "mobile")},
            "looks": rt.hub.pending_looks(),
            "watches": rt.safety.watches(),
            "paused": rt.perception.paused, "private": rt.perception.private,
        }

    @app.get("/api/entities")
    def entities():
        return {"entities": rt.memory.registry.list()}

    @app.get("/api/entities/{eid}")
    def entity(eid: int):
        v = rt.memory.registry.view(rt.memory.registry.get(eid))
        if not v:
            raise HTTPException(404)
        return {"entity": v, "history": rt.memory.registry.history(eid, 20)}

    @app.get("/api/entities/{eid}/snapshot")
    def snapshot(eid: int):
        """The entity's last keyframe, zoomed to the object, with its box drawn."""
        from io import BytesIO

        from fastapi.responses import Response
        from PIL import Image, ImageDraw
        v = rt.memory.registry.view(rt.memory.registry.get(eid))
        if not v or not v["keyframe"]:
            raise HTTPException(404)
        path = rt.memory.keyframes_dir / Path(v["keyframe"]).name
        if not path.exists():
            raise HTTPException(404)
        img = Image.open(path).convert("RGB")
        W, H = img.size
        if v["box"]:
            x0, y0, x1, y1 = v["box"]
            ImageDraw.Draw(img).rectangle((x0 * W, y0 * H, x1 * W, y1 * H),
                                          outline=(95, 211, 138), width=max(3, W // 200))
            cx, cy = (x0 + x1) / 2 * W, (y0 + y1) / 2 * H      # context of ~3x the object
            half = max((x1 - x0) * W, (y1 - y0) * H, min(W, H) * 0.25) * 1.5
            img = img.crop((int(max(0, cx - half)), int(max(0, cy - half)),
                            int(min(W, cx + half)), int(min(H, cy + half))))
        img.thumbnail((480, 480))
        buf = BytesIO()
        img.save(buf, "JPEG", quality=82)
        return Response(buf.getvalue(), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.post("/api/entities/{eid}")
    def correct(eid: int, body: EntityIn):
        if not rt.memory.registry.get(eid):
            raise HTTPException(404)
        return {"entity": rt.memory.registry.correct(eid, body.name, body.zone)}

    @app.delete("/api/entities/{eid}")
    def forget(eid: int):
        rt.memory.registry.delete(eid)
        return {"ok": True}

    @app.get("/api/notes")
    def notes():
        return {"notes": rt.memory.notes.recent(50)}

    @app.delete("/api/notes/{nid}")
    def forget_note(nid: int):
        rt.memory.notes.forget(nid)
        rt.memory.db.log_event("note_forgotten", "stated", {"id": nid})
        return {"ok": True}

    @app.get("/api/events")
    def events(limit: int = 50):
        return {"events": rt.memory.db.recent_events(min(limit, 500))}

    @app.get("/api/sessions")
    def sessions():
        return {"sessions": rt.memory.experiments.list()}

    @app.get("/api/sessions/{sid}")
    def session(sid: int):
        if not rt.memory.experiments.get(sid):
            raise HTTPException(404)
        return {"state": rt.memory.experiments.state(sid),
                "steps": rt.memory.experiments.steps(sid),
                "measurements": rt.memory.experiments.measurements(sid)}

    @app.post("/api/looks/{lid}/dismiss")
    def dismiss(lid: int):
        rt.hub.dismiss_look(lid)
        return {"ok": True}

    @app.post("/api/privacy")
    def privacy(body: PrivacyIn):
        if body.paused is not None:
            rt.perception.paused = body.paused
        if body.private is not None:
            rt.perception.private = body.private
        rt.memory.db.log_event("privacy", "stated", body.model_dump())
        return {"paused": rt.perception.paused, "private": rt.perception.private}

    # -- push channel ---------------------------------------------------------

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        if not authorized(socket.cookies, socket.headers, dict(socket.query_params)):
            await socket.close(code=4401)
            return
        await socket.accept()
        rt.hub.clients.add(socket)
        try:
            await socket.send_json({"type": "hello", "providers": rt.describe()})
            while True:
                msg = await socket.receive_json()
                if msg.get("type") == "ping":
                    await socket.send_json({"type": "pong"})
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            rt.hub.clients.discard(socket)

    @app.websocket("/ws/asr")
    async def ws_asr(socket: WebSocket):
        """One push-to-talk utterance: PCM16 16 kHz in, transcripts out."""
        if not authorized(socket.cookies, socket.headers, dict(socket.query_params)):
            await socket.close(code=4401)
            return
        await socket.accept()
        if not rt.asr.available:
            await socket.send_json({"type": "error", "text": "no MODEL_API_KEY: use browser speech"})
            await socket.close()
            return
        hands_free = socket.query_params.get("mode") == "hands_free"
        hooks = None
        if hands_free and socket.query_params.get("server_turns") == "1":
            from forgemind.server.asr import TurnHooks
            source = socket.query_params.get("source") or "phone"
            hooks = TurnHooks(partial=rt.agent.prefetch,
                              turn=lambda text: run_in_threadpool(rt.agent.handle, text, source))
        try:
            # without hooks the phone sends each turn to /api/chat itself
            await rt.asr.run(socket, hands_free, hooks)
        except WebSocketDisconnect:
            pass
        finally:
            try:
                await socket.close()
            except (RuntimeError, WebSocketDisconnect):
                pass                       # the phone already went away

    return app
