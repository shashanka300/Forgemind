# ForgeMind

A workshop assistant with eyes, ears and memory. A fixed camera watches your bench, the
phone you carry is its moving eye, and a persistent world memory remembers where things
are and what you were doing. An agent on Meta's Muse models finds things, resumes
unfinished work, logs experiments, reads instrument displays and warns you when something
is unsafe.

> Today **you are the robot's body**: when the agent needs to see something it asks you to
> point the phone there. The same requests can drive a robot later.

Built for the Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026). Full spec:
[`docs/PRD.md`](docs/PRD.md). Plugging in Meta's models: [`docs/META_INTEGRATION.md`](docs/META_INTEGRATION.md).

## Status

The whole loop runs **without any Meta API access** (mock model + CLIP-grid detector):
cameras → perception → memory → agent tools → phone UI with voice, overlays and alerts.
Adding keys is a `.env` change, then `forgemind check`.

**Where the models run** (details in [`docs/PRD.md`](docs/PRD.md) §7.3): anything that
runs on every frame, or that the Model API doesn't offer, runs on the local GPU
workstation (RTX 5090). Reasoning and speech recognition go to the Meta Model API.

| Meta Model API | Local GPU |
|---|---|
| Muse Spark 1.3 (default agent, workers) and Muse Wonder (hard turns) | SAM 3.1 open weights: continuous segmentation + tracking |
| Muse Voice Transcribe: streaming speech-to-text | Embeddings: CLIP today, Perception Encoder if it wins the crop test |
| SAM 3.1: on-demand checks only, never per frame | Muse Glimmer 30B: fallback agent if the API is unavailable |
| | Text-to-speech: browser today, Kokoro optional |

Modal-hosted SAM (`deploy/modal_sam.py`) is a fallback if the workstation is unavailable.

## Quick start

```bash
cd Forgemind
uv venv --python 3.12 && source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e ".[vision,dev]"
cp .env.example .env
pytest -q                       # no downloads, no network
forgemind serve --zrok          # prints the phone link (HTTPS, needed for the camera)
```

- **Phone (carried):** open the printed `https://…/?t=…` link, tap *Start camera*, hold
  *Hold* to talk (Chrome on Android has the best speech recognition).
- **Bench camera, option A (USB webcam on the computer):**
  `forgemind static-cam --device 0 --token <t value>`
- **Bench camera, option B (old phone):** open the same link with `&role=static`, fix it
  over the bench, keep it plugged in.
- **Zones:** `forgemind tags` writes `zone-tags.png`; print it and stick each tag in its
  zone (bench, shelf, drawer…). The phone knows where it is from the last tag it saw.

zrok works exactly as in the gallery app (`zrok enable <account token>` once). On the same
Wi-Fi without zrok the page loads, but browsers block the live camera over plain HTTP.

## Things to say (mock mode understands these phrasings)

| Say | Tool |
|---|---|
| "Start a session called voltage divider" | `start_session` |
| "The steps are: wire the divider; measure output; swap R2" | `plan_steps` |
| "Log wire the divider" / "just checking, the probe" | `log_step` |
| "Set R2 to 10k" | `set_config` |
| "I measured output voltage is 3.3 V" | `log_measurement` |
| "Read the multimeter display" | `read_closeup` (needs a vision model to read digits) |
| "Where is the multimeter?" | `where_is` |
| "This is my 2.5 mm hex key" (or the *Teach object* button) | `teach_object` |
| "Look at the drawer" | `go_look_at` |
| "What's on the bench?" | `inspect_scene` |
| "Watch the soldering iron" | `watch_bench` |
| "Where did we leave off?" / "What changed?" / "Write it up" | `resume_session` / `compare_sessions` / `generate_report` |

With `FM_LLM_PROVIDER=meta` you can say anything; the model chooses the tools.

## Layout

```
forgemind/
  config.py              settings from .env
  runtime.py             wires everything together
  hub.py                 phone push channel + look/capture requests ("the body")
  providers/             model slots: mock, Meta/OpenAI-compatible, CLIP, CLIP-grid, local/hosted/API SAM
  memory/                SQLite event store, Qdrant Edge visual index, object registry, experiments
  perception/            frame pipeline, AprilTag zones, safety watch
  agent/                 tools (pydantic-validated), prompts, orchestrator loop
  reports.py             experiment report with [E#] provenance
  server/                FastAPI app, phone page, zrok tunnel
  cameras/static_cam.py  USB/RTSP webcam client
tests/                   memory, agent (incl. fake Meta endpoint), server + WebSocket
deploy/                  modal_sam.py (fallback hosted SAM 3.1), Dockerfile + compose for the app server
```

## What came from the gallery app (Multi_meta)

- CLIP ViT-B/32 image + text encoders via FastEmbed/ONNX, same pinned versions, so the
  0.80 "teach a name" threshold carries over (`FM_TEACH_MIN_SIM`).
- Qdrant Edge as the visual store (`memory/vectors.py`), now holding object crops with
  camera, zone, entity and time.
- "Teach a name" becomes **teach an object identity**, matched across both cameras.
- Live viewfinder → phone camera stream; shutter → `read_closeup` high-res capture.
- Access token → 30-day HttpOnly cookie, and the zrok HTTPS share.

## Design rules (from the PRD)

1. Memory is written on change (appeared / moved / disappeared) plus a one-per-minute
   heartbeat, never every frame.
2. The agent sees a compact situation summary, not raw streams.
3. The model proposes, the app validates (`ToolBox.run`).
4. Every fact has provenance: observed (which camera), stated, instrument, inferred.
5. Robot-ready interfaces: `go_look_at`, `highlight` and `read_closeup` go through the Hub.
6. People are only "present"; they are never stored as identities. Pause and private
   mode are on the phone.

## Known limits of the skeleton

- `clip_grid` boxes are tile-sized and it confuses look-alike tools; it's there to wire
  things up until SAM 3.1 runs locally (`FM_SEGMENTER=sam3`). It's ~1 s per frame on a CPU.
- Built but only tested against fake endpoints until keys arrive: the SAM 3.1 API adapter
  (`meta_sam`), Voice Transcribe streaming, and automatic fallback to Muse Glimmer.
- Local SAM 3 imports on Windows but hasn't run yet (needs `facebook/sam3` access); the
  Perception Encoder embedder runs, but thresholds aren't recalibrated for it. See
  [`docs/META_INTEGRATION.md`](docs/META_INTEGRATION.md).
- The mock model only matches fixed phrasings and cannot read displays.
- Disappearance is only inferred by the bench camera (the phone sees too little at once).
