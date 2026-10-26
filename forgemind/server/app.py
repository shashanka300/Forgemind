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
<h2>ForgeMind is locked</h2><p>Open the full link printed on the computer,
including the <code>?t=</code> access token.</p></body>"""


class ChatIn(BaseModel):
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
        if request.url.path == "/healthz" or not token:
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
        return HTMLResponse(LOCKED, 401)

    # -- pages and files ------------------------------------------------------

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "phone.html", headers={"Cache-Control": "no-store"})

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
    def report(name: str):
        return PlainTextResponse(_safe(rt.memory.reports_dir, name).read_text(),
                                 media_type="text/markdown; charset=utf-8")

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

    @app.post("/api/entities/{eid}")
    def correct(eid: int, body: EntityIn):
        if not rt.memory.registry.get(eid):
            raise HTTPException(404)
        return {"entity": rt.memory.registry.correct(eid, body.name, body.zone)}

    @app.delete("/api/entities/{eid}")
    def forget(eid: int):
        rt.memory.registry.delete(eid)
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

    return app
