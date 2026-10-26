# PRD: ForgeMind — a workshop assistant with eyes, ears and memory

**Event:** Meta Global AI Developer Hackathon (Cerebral Valley), Oct 26 – Nov 4, 2026
**Track:** Indie developer · **Theme:** Push the Frontier (primary), Build for Your World (secondary)
**Owner:** Shashanka B R · **Status:** Draft v2 (Oct 27: API vs local model split, §7.3) · **Working name:** ForgeMind (original name; inspired by, but not branded as, any film character)

---

## 1. One-liner

A personal AI assistant that watches your workbench through a fixed camera, follows you around through the phone in your hand, remembers where everything is and what you were doing, and helps you pick up unfinished work: finding things, guiding the next step, logging the experiment and warning you when something is unsafe.

Today **you are the robot's body**: the agent decides where it needs to look, and you carry its eye there. The same tool interface later drives a real mobile robot or arm.

---

## 2. Starting point: what already exists

This project extends the **gallery memory app** already built on the DeepLearning.AI `sc-on-device-ai-memory` repo (`Documents/Assets/Multi_meta`, repo `daily_logs`). It is the foundation, not a throwaway.

| Existing component (gallery mode) | What it does today | How ForgeMind reuses it |
|---|---|---|
| `robot/gallery/store.py` | CLIP ViT-B/32 image + text embeddings (FastEmbed/ONNX) in a Qdrant Edge shard | Becomes the **visual memory store**: every object crop and keyframe is embedded and stored with time, camera and zone |
| Sync with content-hash dedupe | Avoids re-indexing the same photo | Dedupe of near-identical keyframes from continuous streams |
| Text search, "more like this" | Find photos by meaning or similarity | `where_is("flux bottle")`, "find the last time I saw this" |
| Teach a name (3–5 examples, `RecommendBestScore`, floor 0.80) | Personal names for people/things | **Teach an object identity**: "this is my 2.5 mm hex key" → stable personal entity across cameras |
| `/api/recognize` + live viewfinder (~3 fps; ~65 ms per frame measured in earlier test) | Embed live frames without storing, match to tagged items | Mobile camera live pipeline; extended to send frames to perception and show highlights |
| Shutter → 2048 px capture into index | Save a high-res shot | `read_closeup()`: high-res frame for reading labels, meter displays, part numbers |
| Access token + 30-day cookie, zrok HTTPS tunnel | Secure public link; HTTPS lets the browser use the camera | Same for both camera clients and the assistant UI |
| Docker + Colima deploy (`deploy/gallery/`) | Reproducible run | Extended compose with perception + agent services |
| Original repo pieces (YOLOE, Whisper, Nomic embed) | Robot perception, speech, text embedding | Superseded by §7.3: YOLOE-26 as fast fallback detector; Whisper/Parakeet only as speech-to-text fallback |

**What is new:** a second (static) camera, zones and localization, SAM 3.1 concept segmentation and tracking, an object/sighting registry, experiment memory, a reasoning agent on Meta's Muse models, voice in/out, safety watch, and report generation.

---

## 3. Problem

People doing hands-on work (electronics, lab experiments, repairs, making things) lose time and make mistakes because:

1. **Things go missing.** Tools and parts get moved, and searching breaks focus.
2. **Context evaporates.** After a break or the next day, it takes effort to reconstruct where the work stood, what changed and why.
3. **Records are an afterthought.** Steps, settings and measurements are rarely written down while working, so lab notes and SOPs are incomplete.
4. **Hazards go unnoticed when attention moves.** A soldering iron left on while you are in another room.

Existing assistants only know what you type or the single photo you upload. They don't observe the workspace over time, and they can't connect what they see now with what happened before.

---

## 4. Goals and non-goals

### Goals (hackathon scope)
- **G1. Persistent spatial memory** across a static and a mobile camera: where each known object was last seen, with time, source and confidence.
- **G2. Work continuity:** record an experiment session (steps, configuration, measurements, open items) from speech and observation, and resume it later ("where did we leave off?").
- **G3. Active perception:** the agent can ask the user to bring the phone to a zone or point it at something when its memory is stale or uncertain.
- **G4. Embodied guidance without hardware:** highlight objects on the live phone view and speak directions, instead of physically handing things over.
- **G5. Safety watch** on the bench while the user is away, with alerts delivered to the phone.
- **G6. Useful output:** an experiment report and SOP generated from the session, with provenance for every fact.

### Non-goals
- Physical manipulation (no arm or mobile robot in v1; the interface is designed for it later).
- Full SLAM or 3D reconstruction; zones are defined by printed markers.
- Multi-user, multi-site deployments.
- Medical, industrial-certified or safety-critical guarantees. Alerts are assistive, not a safety system.
- Fully on-device inference on the phone.

---

## 5. Users and primary scenario

**Primary user:** a maker, engineer or researcher working at a home or lab bench, moving between the bench, shelves and other rooms.

**Primary scenario (the demo):**
1. Phone on the move, static camera over the bench. User starts: "Let's continue the voltage-divider test from yesterday."
2. Assistant recalls yesterday's session, compares it with the current bench view, and says which step is pending.
3. User: "Where's the 2.5 mm hex key?" → not on the bench; last seen on the shelf 20 minutes ago by the phone. Shows the frame; when the user walks there, the key is highlighted live.
4. User narrates a configuration change and reads a meter; the phone close-up captures the display value and logs it as a measurement.
5. User leaves the iron on and walks to the kitchen. Static camera sees the iron unattended; the phone speaks an alert.
6. Assistant notices the bench changed and memory of the multimeter is stale → "Can you point the phone at the drawer?"
7. User: "What changed versus yesterday, and write it up." → report with methods, results, deviations and open questions, each fact tagged with its source.

---

## 6. Hackathon fit

| Judging lens | How ForgeMind answers it |
|---|---|
| Creativity | Human-as-robot-body framing; two-camera world model; memory that follows you around |
| Technical ambition | Multi-camera concept tracking, cross-camera identity, event-sourced memory, agent with active perception |
| Does it work | Single end-to-end demo on real hardware (two cameras + workstation), no faked steps |
| Theme: Push the Frontier | Gives a "machine" persistent physical-world awareness; produces egocentric data a future robot could learn from |
| Meta models used | Via Model API: Muse Wonder (agent), Muse Spark 1.3 (workers: display reading, reports), Muse Voice Transcribe (speech in), SAM 3.1 (on-demand checks). Meta open weights on the local GPU: SAM 3.1 (continuous segmentation + tracking), Perception Encoder (visual memory embeddings), Muse Glimmer (fallback agent / cheap worker). Muse Image optional (annotated SOP figures). See §7.3 |

---

## 7. System overview

### 7.1 Devices and network

| Device | Role | Connection |
|---|---|---|
| **Static camera** (USB webcam on the host, or an old phone fixed over the bench) | World camera for the bench; always on, low frame rate | Same Wi-Fi/LAN. USB webcam via OpenCV on the host (no browser HTTPS issue), or old phone opening the app link with `?role=static` |
| **Mobile phone** (carried by user) | Body camera + mic + speaker + main UI | Same Wi-Fi when home; zrok HTTPS link works anywhere and is required for browser camera access |
| **App server + GPU workstation** (Windows/WSL, RTX 5090 32 GB, 64 GB RAM) | Memory services, agent orchestration, local models: SAM 3.1, Perception Encoder, optional Muse Glimmer and TTS | LAN; outbound to Meta Model API |
| **Meta Model API** (`api.meta.ai/v1`) | Muse Wonder / Spark reasoning, Muse Voice Transcribe, on-demand SAM 3.1 | HTTPS, OpenAI-compatible SDK; WebSocket for streaming speech |
| **Hosted SAM 3.1 on Modal** (fallback only) | Segmentation if the workstation is unavailable | HTTPS endpoint (`deploy/modal_sam.py`) |

Models are split by how often they run (§7.3): high-frequency, per-frame work runs on the local GPU; reasoning and speech recognition go to the Meta Model API. The workstation has to stay awake during the demo.

### 7.2 Architecture

```text
 STATIC CAM ──(1–2 fps, change-gated)──┐
                                       │
 PHONE CAM ──(5–10 fps while moving)───┼──► Ingest gateway (WebSocket / OpenCV)
 PHONE MIC ──(audio stream)────────────┼──► Muse Voice Transcribe (API) ─► text
                                       │   ├─ frame dedupe (hash / embedding sim)
                                       │   └─ zone detection (AprilTags)
                                       ▼
                   Perception service (local GPU)
               SAM 3.1 open weights: concept segmentation + tracking
               Perception Encoder crop embeddings (existing store)
               [SAM 3.1 via Model API for on-demand checks]
                                       │  observations
                                       ▼
                         World memory (event-sourced)
        ┌───────────────┬────────────────┬────────────────┬───────────────┐
        │ Object        │ Sightings      │ Experiment     │ Visual store  │
        │ registry      │ (time, cam,    │ sessions,      │ Qdrant Edge   │
        │ (entities)    │  zone, conf)   │ steps, values  │ (crops, keys) │
        └───────────────┴────────────────┴────────────────┴───────────────┘
                                       │ compact situation summary
                                       ▼
              Agent orchestrator (Muse Wonder / Spark; Glimmer fallback)
          tools: where_is · go_look_at · read_closeup · watch_bench ·
                 highlight · log_step · log_measurement · compare_sessions ·
                 resume_session · alert · generate_report
                 subagents (Muse Spark): logging, summarizing, report drafting
                                       │ validated actions
                                       ▼
          Phone UI: voice reply (browser TTS / Kokoro) · live overlay · alerts ·
                    session timeline · memory inspector (view / correct / delete)
```

**Design principles**
1. **Memory is written on meaningful events, not every frame** (object appears, moves, disappears; step starts/ends; measurement read; hazard state changes).
2. **The agent sees a compact situation summary**, not raw streams: current zone, recent events, relevant entities, a few keyframes.
3. **Reasoning proposes, the app validates.** Tool calls are checked against schemas and current state before execution.
4. **Provenance on every fact:** observed (which camera), user-stated, instrument reading, or model inference, each with a timestamp and confidence.
5. **Interfaces are robot-ready:** `go_look_at(zone)` is executed by the user today and by a robot later, with no change to the agent.

### 7.3 Model placement: API vs local

**Rule:** anything that runs on every frame, or that the API doesn't offer, runs on the local GPU. Reasoning and speech recognition use the Meta Model API. Every slot stays behind the provider interfaces in `forgemind/providers/`, so moving a slot between API and local is a config change.

**Meta Model API** (pricing from dev.meta.ai/docs/pricing-rate-limits, checked Oct 26, 2026)

| Slot | Model | Price | Used for |
|---|---|---|---|
| Agent | Muse Wonder `muse-wonder-1.0` | $4 / $20 per 1M tokens in / out | Hard user turns and decisions |
| Worker / default agent | Muse Spark `muse-spark-1.3` | $1.25 / $4.25 per 1M tokens | Most user turns, `read_closeup` display reading, report drafting |
| Speech-to-text | Muse Voice Transcribe `muse-voice-transcribe-1.0` | $0.18 per audio hour | Streaming over `wss://api.meta.ai/v1/asr/realtime`; replaces browser speech recognition (unreliable outside Chrome on Android) |
| Segmentation, on demand | SAM 3.1 `sam-3.1` | $2.50 per 1k images; $0.20 per 1k video frames (min 50) | Confirming a `where_is` result, close-ups, teaching. **Not per-frame**: the API takes one concept per request, so 12 concepts × one bench frame every 2 s is about $54/hour |
| Image generation | Muse Image | $0.01 per image | Optional (F-OUT-3) |

Contributor mode (Spark at about 12× cheaper) lets Meta train on prompts and images. Our images are frames of the user's bench, so it stays **off** unless we decide otherwise.

The API has **no embeddings endpoint and no text-to-speech**. Those slots must be local.

**Local GPU (RTX 5090, 32 GB)**

| Slot | Choice | Alternatives | Notes |
|---|---|---|---|
| Continuous segmentation + tracking | **SAM 3.1 open weights** (`facebook/sam3.1`), ~4 GB FP16 | YOLOE-26 as a fast open-vocabulary gate (AGPL) | Same model as the API, so behaviour matches. Reuses the existing `Sam3Segmenter` (`FM_SEGMENTER=sam3`). Replaces `clip_grid`; Modal deploy becomes fallback only |
| Visual memory embeddings (image + text) | **Meta Perception Encoder PE-Core-L14-336**, ~1 GB | SigLIP 2; DINOv2/v3 (image-only, same-object matching); current CLIP ViT-B/32 | Evidence on same-object matching is mixed (DINOv3 weak on a 2026 kitchen re-ID study). Decide with a test on ~50 crops of our own tools. Switching means recalibrating `FM_TEACH_MIN_SIM` / `FM_MATCH_MIN_SIM` |
| Fallback agent / cheap worker | **Muse Glimmer 30B** (Apache 2.0), Q4 ~17 GB, ~75–84 tok/s on a 5090 via llama.cpp | — | Keeps the agent working if the API is slow, down or rate-limited. Cap context ~32K so it fits beside SAM + PE |
| Text-to-speech | Browser `speechSynthesis` (today) | **Kokoro-82M** (CPU, Apache); Qwen3-TTS (GPU, best low-latency quality); Chatterbox-Turbo (voice cloning) | Upgrade only if the browser voice hurts the demo |
| Speech-to-text fallback | Only if Voice Transcribe fails from India | Parakeet TDT 0.6B v3 (fastest, streaming); Qwen3-ASR-1.7B / Canary-Qwen 2.5B (most accurate English); Whisper large-v3-turbo | Speech/TTS rankings come from vendor blogs with conflicting numbers; verify locally |
| Non-ML | AprilTags (OpenCV), change gate, SQLite, Qdrant Edge | — | Unchanged |

**VRAM budget:** SAM 3.1 (~4 GB) + PE (~1 GB) + Glimmer Q4 with ~32K context (~20–24 GB) stays under 32 GB. If Glimmer isn't loaded there is plenty of headroom.

---

## 8. Functional requirements

Priority: **P0** = required for the demo, **P1** = should have, **P2** = stretch.

### 8.1 Camera ingestion
| ID | Requirement | Priority |
|---|---|---|
| F-CAM-1 | Accept the static camera stream from a USB webcam (OpenCV) or a browser client with `role=static` | P0 |
| F-CAM-2 | Accept the mobile phone stream from the existing live viewfinder, extended to send frames to the perception service | P0 |
| F-CAM-3 | Static camera runs at 1–2 fps and only forwards frames when pixel/embedding change exceeds a threshold | P0 |
| F-CAM-4 | Mobile camera forwards 5–10 fps while moving or when the agent requests a view; idle otherwise | P0 |
| F-CAM-5 | High-res capture on demand (reuse shutter, 2048 px) for close-ups | P0 |
| F-CAM-6 | Each frame tagged with camera id, timestamp, zone (if known) | P0 |

### 8.2 Zones and localization
| ID | Requirement | Priority |
|---|---|---|
| F-ZONE-1 | Printed AprilTag/ArUco markers identify zones (bench, shelf, drawer, cabinet; 3–5 zones) | P0 |
| F-ZONE-2 | Phone's current zone = most recently seen marker, with a timeout after which zone becomes "unknown" | P0 |
| F-ZONE-3 | Static camera has a fixed zone (bench) and sub-regions (left, centre, right, tray) defined once by drawing boxes in a setup page | P1 |
| F-ZONE-4 | Static camera detects the user (person/hand) at the bench, used to decide "user present / away" | P0 |

### 8.3 Perception
| ID | Requirement | Priority |
|---|---|---|
| F-PER-1 | SAM 3.1 (open weights on the local GPU) segments and tracks a configurable concept list (e.g. multimeter, soldering iron, screwdriver, hex key, PCB, breadboard, flux, hand, smoke) | P0 |
| F-PER-2 | Each detected object crop is embedded (Perception Encoder, or CLIP until the crop test decides) and stored in the existing Qdrant Edge store with metadata | P0 |
| F-PER-3 | Track-level events emitted: appeared, moved (sub-region change), disappeared, interacted (hand overlap) | P0 |
| F-PER-4 | Hazard states: soldering iron present + powered indicator/smoke + no hand nearby for N seconds | P1 |
| F-PER-5 | Fallback detector (YOLOE-26) if local SAM latency is too high for the static stream | P2 |
| F-PER-6 | On-demand SAM 3.1 via Model API for verification and close-ups; never per frame | P1 |

### 8.4 Object registry and cross-camera identity
| ID | Requirement | Priority |
|---|---|---|
| F-OBJ-1 | One entity per real object; each entity has a list of sightings from either camera | P0 |
| F-OBJ-2 | Match new observations to entities using concept label + crop embedding similarity + zone + recency | P0 |
| F-OBJ-3 | **Teach identity** (extends "teach a name"): user shows an object and names it; 3–5 crops become its reference set | P0 |
| F-OBJ-4 | Confidence decays with time since last sighting; the bench is "source of truth" from the static camera unless the phone has a newer, closer view | P0 |
| F-OBJ-5 | User can correct identity or location via voice or the memory inspector | P1 |

### 8.5 Experiment memory (work continuity)
| ID | Requirement | Priority |
|---|---|---|
| F-EXP-1 | Start/resume/end sessions by voice ("let's continue the voltage-divider test") | P0 |
| F-EXP-2 | Log steps from narration and observation, with timestamps and keyframes | P0 |
| F-EXP-3 | Log measurements from speech or from close-up reading of instrument displays, tagged by source | P0 |
| F-EXP-4 | Store configuration (components, values) as structured fields | P0 |
| F-EXP-5 | Distinguish "don't log that, just checking" (tentative actions) from committed steps | P1 |
| F-EXP-6 | Compare two sessions: configuration diff, measurement diff, deviations | P0 |
| F-EXP-7 | Resume: reconstruct last state, list incomplete steps, propose the next action | P0 |
| F-EXP-8 | Learn a procedure (SOP) from a completed session and check later sessions against it | P1 |

### 8.6 Agent
| ID | Requirement | Priority |
|---|---|---|
| F-AGT-1 | Muse Wonder / Spark as the main agent with the transcript, keyframes and the situation summary as input | P0 |
| F-AGT-7 | Fall back to Muse Glimmer on the local GPU when the Model API errors, times out or rate-limits | P1 |
| F-AGT-2 | Muse Spark subagents for logging, summarizing and report drafting | P1 |
| F-AGT-3 | Tool calls validated against JSON schemas and state before execution | P0 |
| F-AGT-4 | Active perception: when an entity's confidence is below threshold or the bench changed, the agent may call `go_look_at` | P0 |
| F-AGT-5 | Never invent measurements; ask for them when missing | P0 |
| F-AGT-6 | Proactive speech limited to: safety alerts, answers to the user, and requested reminders | P0 |

### 8.7 Phone UI and voice
| ID | Requirement | Priority |
|---|---|---|
| F-UI-1 | Push-to-talk and a wake word option; phone streams audio to the server, which relays it to Muse Voice Transcribe; local Parakeet/Whisper fallback | P0 |
| F-UI-2 | Spoken replies (browser speech synthesis; Kokoro on the server if the browser voice is poor) plus on-screen text | P0 |
| F-UI-3 | Live overlay: highlight a target object's mask/box on the phone view; arrow toward it if off-screen but in the current zone | P0 |
| F-UI-4 | "Last seen" card: keyframe, zone, time, camera, confidence | P0 |
| F-UI-5 | Session timeline (steps, measurements, alerts) | P1 |
| F-UI-6 | Memory inspector: view, correct and delete entities, sightings and logs | P1 |
| F-UI-7 | Privacy controls: pause camera(s), private mode (no storage) | P0 |

### 8.8 Safety watch
| ID | Requirement | Priority |
|---|---|---|
| F-SAF-1 | Detect "iron on and unattended" (iron present, user away for N seconds) | P0 |
| F-SAF-2 | Detect smoke/flame concepts on the bench | P1 |
| F-SAF-3 | Alert on the phone (speech + vibration + banner) wherever the user is | P0 |
| F-SAF-4 | Optional smart-plug cut-off after confirmation or timeout | P2 |

### 8.9 Outputs
| ID | Requirement | Priority |
|---|---|---|
| F-OUT-1 | Experiment report: objective, setup, method, results, deviations, open questions; every fact links to its source event | P0 |
| F-OUT-2 | SOP from a session with keyframes per step | P1 |
| F-OUT-3 | Annotated figures from keyframes with Muse Image | P2 |
| F-OUT-4 | Export Markdown/PDF | P1 |

---

## 9. Agent tools

| Tool | Input | Output | Notes |
|---|---|---|---|
| `where_is` | object name or description | entity, last sightings, zone, time, camera, confidence, keyframe | Text search + teach-identity match over the visual store |
| `go_look_at` | zone or object, reason | request id; completes when the phone sees the zone marker or object | Executed by the user today; by a robot later |
| `read_closeup` | target (e.g. "multimeter display") | high-res frame + extracted value + confidence | Prompts the user to hold the phone close |
| `inspect_scene` | camera (static/mobile) | current objects, regions, recent events | Compact summary, not raw frames |
| `highlight` | entity id | overlay drawn on phone | Fails gracefully if not in view; offers `where_is` instead |
| `watch_bench` | condition (e.g. iron unattended), duration | watch id | Registers a rule evaluated by the perception service |
| `log_step` | text, tentative flag | step id | Spark subagent normalizes the wording |
| `log_measurement` | quantity, value, unit, source | measurement id | Source: spoken / read from display / instrument |
| `resume_session` | session name or "last" | last state, incomplete steps, next suggestion | |
| `compare_sessions` | two session ids | configuration and result diff | |
| `alert` | severity, message | delivered to phone | Rate-limited |
| `generate_report` | session id, format | Markdown/PDF report | Spark subagent drafts; Wonder reviews |

---

## 10. Data model (SQLite + Qdrant Edge)

```text
zones(id, name, marker_id, camera_fixed BOOL)
cameras(id, role[static|mobile], device_label)
entities(id, label, concept, user_name NULL, created_at)
entity_refs(entity_id, crop_vector_id)           -- teach-identity examples
sightings(id, entity_id, camera_id, zone_id, sub_region, bbox, mask_ref,
          keyframe_ref, vector_id, ts, confidence)
events(id, ts, type[appeared|moved|disappeared|interacted|hazard|speech|step|measure],
       entity_id NULL, camera_id NULL, payload JSON, source[observed|stated|instrument|inferred])
sessions(id, name, started_at, ended_at, status)
steps(id, session_id, ts, text, tentative BOOL, keyframe_ref, event_id)
measurements(id, session_id, ts, quantity, value, unit, source, keyframe_ref)
configs(id, session_id, ts, json)
watches(id, condition JSON, active BOOL, created_at)
```

Qdrant Edge collections: `crops` (object crops, image vectors from the chosen embedder, §7.3), `keyframes` (scene frames), `notes` (text memory). The collection records which embedder wrote it; changing embedder means re-embedding. Every vector payload carries `camera`, `zone`, `ts`, `entity_id`.

---

## 11. Non-functional requirements

| Area | Target |
|---|---|
| `where_is` answer latency | Under 2 s for memory lookup; spoken answer under 4 s |
| Live highlight latency on phone | Under 500 ms from frame to overlay at the bench |
| Safety alert latency | Under 10 s after the unattended threshold |
| Static camera compute | Change-gated; local SAM runs only on changed frames |
| Model cost | No per-frame API calls. Spark by default, Wonder for hard turns; Voice Transcribe ~$0.18/h; SAM API on demand only. Target: whole build week under ~$50 of API spend |
| Reliability | Pipelines restart independently; memory survives restarts (SQLite + Qdrant Edge on disk) |
| Security | Reuse access token + cookie; zrok link treated as a password; no frames sent to the cloud except those the agent needs (continuous perception is local); Contributor mode off |
| Privacy | Pause and private mode; user can delete any memory; faces are not identified or stored as identities |

---

## 12. Reuse vs new work

| Component | Reuse | New |
|---|---|---|
| Visual store (Qdrant Edge) | ✅ | metadata fields for camera, zone, entity; embedder swap (CLIP → PE) and threshold recalibration |
| Teach a name → teach identity | ✅ | multi-camera matching rules |
| Live viewfinder + recognize | ✅ | streaming to perception, overlay rendering |
| Token, zrok, Docker | ✅ | extra services in compose |
| Static camera ingest | | ✅ |
| AprilTag zones | | ✅ |
| SAM 3.1 perception service | | ✅ |
| Entity/sighting registry, event store | | ✅ |
| Experiment memory | | ✅ |
| Agent + tools on Muse models | | ✅ |
| Voice in/out | partly (browser TTS) | ✅ Voice Transcribe streaming relay |
| Local model hosting (SAM 3.1, PE, Glimmer) | `Sam3Segmenter` already written | ✅ |
| Safety watch | | ✅ |
| Reports | | ✅ |

---

## 13. Milestones (Oct 26 – Nov 4; submission due Nov 4, 9 PM ET = Nov 5, 7:30 AM IST)

| Day | Date | Deliverable |
|---|---|---|
| 1 | Oct 26 | Kickoff; Meta Model API key tested from India; both camera streams reach the host; AprilTags printed |
| 2 | Oct 27 | API smoke tests (`forgemind check` + one SAM 3.1 call + one Voice Transcribe stream); SAM 3.1 open weights running locally on both streams for 6–8 concepts; zone detection |
| 3 | Oct 28 | Embedder crop test (CLIP vs PE-Core vs DINOv2 on ~50 crops of our tools) → pick embedder and recalibrate thresholds; entity/sighting registry; teach identity |
| 4 | Oct 29 | `where_is` end to end with keyframe card and live highlight |
| 5 | Oct 30 | Agent on Muse Spark/Wonder with voice in (Voice Transcribe) and out; tools wired with validation; Glimmer fallback if time allows |
| 6 | Oct 31 | Experiment memory: start/resume, steps, measurements, `read_closeup` |
| 7 | Nov 1 | `go_look_at` active perception; user present/away; safety watch + phone alerts |
| 8 | Nov 2 | `compare_sessions`, report generation (Spark subagents); memory inspector |
| 9 | Nov 3 | Integration, latency tuning, demo rehearsal, fallback paths |
| 10 | Nov 4 | Demo video, README, architecture diagram, submission |

---

## 14. Acceptance criteria (demo must pass)

1. Taught object moved from bench to shelf is found by `where_is` with correct zone and camera, and highlighted live when the phone arrives.
2. Assistant resumes a previous day's session and states the correct pending step.
3. Measurement read from a meter display by close-up is logged with source "instrument (read from display)".
4. Iron left unattended with the user in another zone triggers a phone alert within the latency target.
5. Agent issues a `go_look_at` request when memory is stale, and updates memory after the user complies.
6. Report contains only facts traceable to logged events; missing values are flagged, not invented.

## 15. Success metrics

- `where_is` correct-zone rate on a scripted set of 20 queries: ≥ 85%
- Cross-camera identity match precision on taught objects: ≥ 90%
- False safety alerts during a 30-minute session: ≤ 1
- Report facts with valid provenance links: 100%

---

## 16. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Meta Model API access from India or rate limits | Test on day 1; ask on Discord; Spark for most calls; Muse Glimmer on the local GPU as fallback agent; hosted open weights on partner credits (Fireworks/Baseten/Modal) |
| SAM 3.1 API cost if used per frame (one concept per request) | Continuous perception runs locally; API only on demand (F-PER-6) |
| Local SAM 3.1 latency on the 5090 | Change-gating; fewer concepts per pass; YOLOE-26 gate; Modal or API fallback |
| Workstation sleeps or crashes during the demo | Keep-awake during demo; pipelines restart independently; memory on disk |
| Embedder swap breaks tuned thresholds | Decide on Day 3 with a crop test; keep CLIP as default until then |
| Voice Transcribe unavailable from India | Browser speech recognition (current) or local Parakeet/Whisper |
| Small objects (hex keys) hard to segment | Taught identities, close camera framing, tool tray zones; choose demo objects of reasonable size |
| Browser camera needs HTTPS | Phone uses the zrok HTTPS link (already built); static camera via USB/OpenCV avoids the browser |
| Cross-camera mismatch | Combine concept + embedding similarity + zone + recency; user correction in the inspector |
| Agent talks too much | Strict proactive-speech policy (F-AGT-6) |
| Privacy concerns with always-on cameras | Pause/private mode, local storage, deletion, no face identities |
| Scope creep | P0 list is the demo; P1/P2 only after acceptance criteria pass |
| Employer/IP constraints | Check official rules and IBM outside-activity guidance before building |

---

## 17. Open questions

1. ~~SAM 3.1: API or Modal?~~ **Resolved (§7.3):** open weights on the local GPU for continuous perception; Model API on demand; Modal as fallback only. Still measure local per-frame latency on Day 2.
2. ~~Streaming audio?~~ **Resolved:** Muse Voice Transcribe has a streaming WebSocket endpoint. Open: confirm it works from India and its latency for push-to-talk.
3. Static camera: USB webcam on the host, or an old phone? (USB is simpler for always-on.)
4. ~~Mac or cloud VM?~~ **Resolved:** the RTX 5090 workstation (Windows/WSL) runs the app server and local models.
5. Which demo build: voltage-divider breadboard test, a small soldering job, or a Raspberry Pi assembly?
6. Embedder: PE-Core vs CLIP vs DINOv2 for same-object matching on our tools (Day 3 crop test).
7. Does the SAM 3.1 API return confidence scores? The docs don't mention them; `FM_SAM_MIN_SCORE` may not apply to the API path.
8. Is Kokoro worth adding, or is browser TTS good enough on the demo phone?

---

## 18. Beyond the hackathon

- **Physical embodiment:** `go_look_at`, `highlight` and a future `fetch` map onto a mobile robot base and an arm; the agent and memory don't change.
- **Learning from you:** sessions record egocentric video, your actions and the agent's requests — the kind of paired data a robot policy needs.
- **Other domains:** software and hardware debugging sessions, home repairs, kitchen, lab research notebooks.
- **Memory research:** plug in the event-sourced, merge-invariant memory work for syncing across devices.
