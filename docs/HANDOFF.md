# Handoff

Where the build stands and what to do next. Update this at the end of each work session.

**Last updated:** Oct 27, 2026 (Day 6). **Deadline:** Nov 5, 7:30 AM IST (Nov 4, 9 PM ET).
**Branch:** `main`. Check `git status -sb` for unpushed commits.

## Status

| Piece | State | Verified how |
|---|---|---|
| Full loop in mock mode (cameras → memory → agent → phone) | Works | 37 tests; phone page loads, chat round trip in browser |
| Object registry + teach identity (F-OBJ-1..5) | Existed in skeleton; teaching improved | Teach now takes up to 3 views from the last 4 s of phone frames, asks SAM for the named concept if the frame's detections missed it, and says when it fell back to the centre of the view. Tests only (colour-block fixtures) |
| Replay test (`forgemind replay`) | **Built + run** | Video → real pipeline (gate, SAM 3.1, PE, registry), mock model, no API. Fixed cam (overhead, 70 s): 16 entities / 19 false "disappeared" before the fixes → **9 entities, all real; 1 correct move (LCD); 1 correct disappearance (headset band picked up)**. Handheld walk-through as phone frames: 27 entities in a cluttered room, ~4 duplicates (a steel bottle split in 3, 1-sighting fragments) |
| Hand-aware perception | **Built** | New objects need 2 frames in a row (`FM_CONFIRM_NEW`); on the bench an object overlapping a hand/person box never creates an entity; a covered object isn't "missing"; "gone" needs 3 misses and 10 s; move linking: a same-concept object the bench saw go missing within 120 s matches at `match_min_sim - 0.15` |
| Voice, Option A (Day 5) | **Built** | Hands-free (`/ws/asr?mode=hands_free`, Voice Transcribe ENDPOINTING): verified live, each sentence becomes a turn ~0.4–1 s after speech ends. Phone streams silence while it talks (no self-transcription), auto-reconnects, tap Hold to interrupt. Instant spoken ack on the phone ("Let me check." / "Okay." / "One moment.") with the answer queued behind it. Not yet tried on a real phone. Duplex models investigated: NVIDIA VoiceChat needs an 80 GB GPU; MiniCPM-o 4.5 fits but has no tool calling (stretch goal) |
| Muse Glimmer local (Option A item 4) | **Tested, not used for live turns** | `F:\models\glimmer\Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf` on llama.cpp b11934 CUDA 13.4 (`F:\llama.cpp11934\llama-server.exe -m <gguf> -c 32768 -ngl 99 -fa on --jinja --port 8081 --alias glimmer`), ~18 GB VRAM, ~80 tok/s, prompt cache ~0.2 s. Agent eval: **9/15 tools, median 4.5 s, max 16 s**; writes 300–500 reasoning tokens before tool calls, and **claims actions without calling tools** ("Started …", "Logged: swap R2 …"). Spark stays primary. As F-AGT-7 fallback it would need a guard that rejects action claims with no tool call; not wired in `.env` |
| Experiment memory (Day 6) | **Acceptance #2, #3 pass** | #2: a session from yesterday resumes with "last active 1 day ago … Next up: …" (unit test + real Spark eval turn). #3: `read_closeup` with real Spark vision logs `source=instrument` and keeps the raw display text; 7 test photos: correct reading (.0060 → 0.006 V), null for OL / blank / no display / stopwatch, and two visible meters are refused as ambiguous. F-AGT-8: `missing` from `experiments.gaps()` (values logged without a number; done "measure X" steps with nothing logged) in resume/compare/report summaries; `stale` + `confidence` in where_is. "Last active" ignores resume/end bookkeeping events |
| Agent on Muse Spark (Day 5) | **Under target** | `forgemind agent-eval`: 14 scripted turns on a fixed world, real Spark. Baseline median 8.2 s / max 12.9 s / $0.149. Now **median 2.2–2.5 s, max 3.6–6.3 s, 13–14/14 tools, $0.03–0.04 per 14 turns** (`--frames` adds ~700 tokens, no measurable delay). Changes: speak a tool's summary directly unless it needs phrasing (`NEEDS_PHRASING` = inspect_scene, compare_sessions) → ~1.07 calls/turn; static system prompt (situation moved to the user turn) so the prefix caches; `FM_REASONING_EFFORT=low`; `alert` hidden from the agent (low effort called it for small talk); a measurement completes its planned "measure …" step in code; step matching stems words (wired/wire, tinned/tin). Spoken answer ≈ 0.25 s transcript + ~2.5 s agent. Wonder: not used for live turns (slower, 3–5x cost); candidate for report review (Day 8) |
| `where_is` end to end (acceptance #1) | **Built, tested** | Answers zone + camera + time; "last seen" card shows a server-rendered snapshot (`/api/entities/{id}/snapshot`, keyframe zoomed on the object, box drawn); if not in view it **seeks** for 120 s: the first phone frame containing the object pushes a live highlight + "There's your X." (event `found`). Test `tests/test_where_is.py` runs the full bench → shelf → ask → arrive scenario. Verified in the browser pane with SAM 3.1 + PE on the workbench photo (cards, snapshot, highlight). Not yet on the real phone |
| Embedder calibration (`forgemind calibrate`) | **Decided: PE** | Real cross-camera test on Assembly101 (see below). `.env` now `FM_EMBEDDER=pe`, teach 0.835, match 0.90 (provisional). Text-search floor per embedder (PE 0.19, CLIP 0.22) |
| SAM 3.1 via Model API (`FM_SEGMENTER=meta_sam`) | **Works** | Live call on the workbench photo: reply format matches the parser, `c` scores and object ids come back, boxes match local SAM. ~2.3 s per concept (3 in parallel: 3.2 s) |
| Voice Transcribe relay (`/ws/asr`, phone AudioWorklet → server → Meta) | **Works server-side** | Live: synthetic 16 kHz speech streamed through the relay; exact transcript ("2.5 millimeter", "3.3 volts"), final 0.25 s after release, first partial ~1 s in, connect+handshake ~1.3 s (phone buffers meanwhile). `endStream` shape accepted. Not yet tried with a real phone mic over zrok. The file endpoint (`/asr/transcribe`) returned "Malformed request" to a quick multipart test; unused by the app |
| LLM fallback to Glimmer (`FM_FALLBACK_PROVIDER=compat`) | Built | Stub models only; no Glimmer server yet |
| Perception Encoder embedder (`FM_EMBEDDER=pe`) | Built | Runs on the 5090: 16 crops ~60 ms, 2.7 GB VRAM. Thresholds not recalibrated |
| Local SAM 3 (`FM_SEGMENTER=sam3`) | Runs on the 5090 | Two bench photos: ~490 ms for 12 concepts, 80 ms for one, 4.1 GB VRAM. SAM 3.1 detector weights give the same detections. Misses: soldering iron read as "screwdriver" in one photo; irons, stations and pliers missed in the other |
| Muse Spark / Wonder via Model API | **Works from India** | `forgemind check` OK: key sees 9 models (wonder, spark 1.1–1.3 + contributor, sam-3.1, voice-transcribe, image). Both return a correct `where_is` tool call; ~2.5 s Spark, ~4.5 s Wonder per call (~2.3k prompt tokens) |

## Blockers

1. ~~SAM access~~ Granted for both `facebook/sam3` and `facebook/sam3.1`.
2. ~~`MODEL_API_KEY`~~ In `F:\Forgemind\.env` (git-ignored; never print it). `.env` also sets
   `FM_LLM_PROVIDER=meta`, `FM_AGENT_MODEL=muse-spark-1.3`, `FM_SEGMENTER=sam3`.

## Next steps (in order)

1. **SAM follow-ups:** the 3.1 checkpoint is now the default (`FM_SAM_CHECKPOINT=sam3.1`;
   the "missing convs.3 keys" message at load is harmless, see `Sam3Segmenter`). Tune concept names and the 0.5 cutoff on photos of
   the user's own bench ("soldering iron" vs "soldering station"). Benchmark script:
   `sam_bench.py` (scratchpad; move into the repo if kept). `Sam3Segmenter` now runs
   under bf16 autocast (required).
2. **API follow-up:** try push-to-talk from the real phone over zrok (mic permission,
   AudioWorklet at 16 kHz on the phone's browser). Agent latency is now under target
   (see Status); re-run `forgemind agent-eval` after prompt or tool changes (~$0.04).
3. **Embedder decision (Day 3): done on public data.** Assembly101 sample, one recording
   (`nusar-2021_action_both_9011-a01…`, CC BY-NC 4.0, internal testing only; downloaded to
   the session scratchpad, not the repo). At t=139.5 s, 9 parts hand-matched across 5 fixed
   cameras (C10115 + C10404 overhead = "bench", C10095/C10119/C10390 = "phone"); 45 crops,
   18 bench queries. Includes a true look-alike pair (two identical tracks).
   | embedder | top-1 | recall @ P≥0.9 (thr) | look-alike AUC | text search |
   |---|---|---|---|---|
   | clip | 0.83 | 0.61 (0.902) | 0.68 | yes |
   | **pe** | **0.94** | **0.89 (0.835)** | **0.78** | yes |
   | dinov2 | 0.83 | 0.06 (0.841) | 0.60 | no |
   | dinov3 | 0.94 | 0.94 (0.817) | 0.59 | no |
   Synthetic smoke test (same pixels) had favoured DINOv3; real cross-camera views do
   not. CLIP at the old 0.80 would accept almost anything (different objects average
   0.80). No embedder separates untaught look-alikes by appearance (PE needs 0.917 for
   12% recall): identity of look-alikes must come from place/recency or teaching.
   `pick_embedder` now prefers text-capable models unless image-only wins by >0.10 recall.
   **Replay done** (see Status): 0.90 holds stable objects together; lowering it (0.85 →
   14, 0.80 → 13 entities) barely helps, the fragments came from hands, now handled in code.
   Missed detections are vocabulary (SAM never matched "usb charger" to the white cube
   charger; blue pliers missed): tune `FM_CONCEPTS` to the demo objects. User footage so far:
   `sampledata.mp4` (handheld walk-through, narrated: names via Voice Transcribe) and
   `fixed cam.mp4` (overhead). Correction: the RGB keyboard never tripped the change gate on
   its own (0/69 frame pairs at 1 fps); the gate fired because hands moved almost all clip. Both git-excluded. Folder layout for
   `forgemind calibrate`:
   - 8–10 of the demo objects, **including look-alikes** (two screwdrivers, two hex keys).
   - One folder per object, named `<concept>-<which>` (e.g. `hex_key-2.5mm`,
     `screwdriver-red`); the part before `-` is what SAM is asked for.
   - In each: 3 phone photos (held close, different angles, like teaching) named
     `phone_1.jpg`… and 2–3 from the bench-camera position (above the bench, at its
     distance) named `bench_1.jpg`….
   - Run: `forgemind calibrate <folder> --embedders clip,pe,dinov2,dinov3`. It crops with
     SAM, prints per-embedder AUC, top-1 and the threshold for 90% precision, and writes
     `calibration.json`. Then set `FM_EMBEDDER` and both thresholds from its output.
   - **Smoke result** (synthetic: one workbench photo, 13 objects incl. 6 red pliers and 5
     screwdrivers, "bench" views = downscaled/rotated/relit crops; optimistic):
     | embedder | AUC | look-alike AUC | top-1 | recall @ P≥0.9 (thr) |
     |---|---|---|---|---|
     | clip (current) | 0.744 | 0.606 | 0.31 | 0.08 (0.898) |
     | pe | 0.863 | 0.749 | 0.58 | 0.19 (0.871) |
     | dinov2 | 0.904 | 0.746 | 0.65 | 0.27 (0.734) |
     | dinov3 | 0.897 | 0.767 | 0.77 | 0.54 (0.764) |
     Takeaways: CLIP B/32 at 0.80 would mis-identify often; DINO leads on identity but is
     image-only, so if it wins on real photos the registry needs two vectors per crop
     (DINO for identity, PE/CLIP for text search). Near-identical objects (the red
     pliers) can't be told apart from one crop by any of them: identity must lean on
     place and recency for those, and the demo should use visually distinct objects.
4. **Ego-Exo4D:** user requested access on Oct 27; credentials expected Oct 28. Then pull a
   few bike-repair takes (downscaled) for a larger cross-camera test with real tools.
5. **Glimmer server:** download the Q4 GGUF, serve with llama.cpp (`-c 32768`), point
   `FM_COMPAT_*` at it, confirm tool calls with `forgemind check`.
6. **Day 3 remaining:** bench region is built (`FM_BENCH_ROI`/`FM_BENCH_MASK`, suggested by
   `forgemind bench-region <video> [--roi] [--write]`; masks never cover detected bench
   objects; ROI crop is opt-in). Set it once the bench camera's final mount is fixed.
   **Concepts tuned** with `forgemind concepts` (6 desk frames + 8 web photos):
   - SAM 3.1 **never** answers to "soldering iron" (0 hits on 6 clear iron photos); irons
     come out as "soldering pen" (0.59–0.75; also fires on ballpoint pens, 0.81) or
     "screwdriver" (0.78–0.80). So the iron must be **taught as "soldering iron"**; the
     safety watch now matches taught names, not just detector concepts (tested).
   - "multimeter" fires on the user's digital stopwatch (0.94); no timer phrasing fires
     ("stopwatch", "kitchen timer", "timer", "digital clock", "egg timer"): teach the meter.
   - "soldering station" and "laptop" box whole scenes: dropped. Object boxes > 60% of the
     frame are now discarded; one label per object (cross-concept IoU ≥ 0.7 keeps the higher).
   - "lcd display"/"lcd screen" hit monitors and meter screens, not LCD modules; "circuit
     board" finds the modules (0.94–0.97). "wall charger" finds the cube charger (0.57),
     "copper coil" the coil (0.63). "sponge" (iron-stand sponge) is reliable (0.66–0.82).
   - Each concept costs ~37 ms/frame (14 concepts ≈ 780 ms on the 5090).
   Remaining: try teaching on the real phone over zrok. Then Day 4: `where_is` end to end.
   Note: "covered isn't missing" can delay a real "gone" while hands stay nearby; revisit if
   it hurts the safety watch.
7. **Added from the personal health agent paper** (PRD §7.4, Nature 2026), scheduled in §13:
   - F-AGT-8 explicit `missing` / `stale` fields from tools (Day 6, P0)
   - F-OUT-5 deterministic citation check on reports (Day 8, P0)
   - F-EXP-9 user notes memory: goals, obstacles, preferences per turn (Day 8, P1)
   - §15.1 evaluation harness with Glimmer as simulated user (Day 9)
   Do **not** add a multi-agent orchestrator to live turns: the paper's averaged 205 s/turn.

## Environment

- **Repo:** `F:\Forgemind` (Windows native, not WSL). Remote `shashanka300/Forgemind`.
- **venv:** `.venv`, Python 3.12 via uv. Installed: `.[vision,dev]`, torch 2.11 cu128,
  plus the `local` extra's packages (sam3 pinned, einops, pycocotools, psutil,
  triton-windows, open_clip_torch). Fresh setup:
  ```bash
  uv venv --python 3.12
  uv pip install -e ".[vision,dev]"
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  uv pip install -e ".[local]"
  ```
- **GPU:** RTX 5090 32 GB (compute 12.0), 64 GB RAM.
- **Dev server for the browser pane:** entry `forgemind` in `E:\challeenge\.claude\launch.json`
  (port 8770, `FM_TOKEN=devtoken`, data in `forgemind-data-dev/`). Open
  `http://localhost:8770/?t=devtoken`. First start downloads CLIP (~600 MB).
- **Ignored locally** (`.git/info/exclude`): `.claude/`, `forgemind-data-dev/`.
- **Commits:** no AI attribution lines; author `shashanka300 <shashanka300@gmail.com>`.

## Budget

**Hard cap: USD 150 of Meta Model API spend for the whole engagement** (user's limit).
Spent so far: about $0.65 (Oct 27: `forgemind check`, SAM API tests, Voice Transcribe
tests, ~9 agent-eval runs at $0.03–0.15 each). Before any API-heavy run, estimate the cost first.
Prices: Spark $1.25/$4.25 per 1M tokens in/out (~$0.003 per agent call at ~2.3k prompt
tokens), Wonder $4/$20 (~3x Spark), SAM API $2.50 per 1k images (one per concept),
Voice Transcribe $0.18 per audio hour. Bulk work (evaluation, simulated users,
calibration) runs on local models: SAM 3.1, Glimmer, embedders.

## Gotchas

- New objects need a second sighting within 5 s (`CONFIRM_WITHIN_S`); posting test frames
  by hand more slowly than that never confirms anything. The change gate also skips
  identical frames: vary the frame or wait 15 s.
- Phone frames with no AprilTag in view have zone None; the registry keeps the object's
  last known zone then (it used to "move" everything to nowhere).
- Keyframes: the bench camera saves one per processed frame (up to 1/s) and nothing deletes
  them yet. Fine for the demo; add a janitor (keep only referenced ones) before long runs.
- Not built from F-UI-3: an arrow toward an object that is in the zone but off-screen.

- `.env` holds the real `MODEL_API_KEY`; `Settings.from_env` loads it, so test fixtures
  must override `meta_api_key=""` (conftest does). A test once reached the real API.
- Voice Transcribe rejects audio slower than real time ("Ingress audio slower than
  real-time"); keep ~1–2 s of lead when replaying files. The relay should recover from it.
- SAM reads a digital stopwatch as "multimeter" (0.94): add decoy concepts or rely on
  taught identity for the multimeter.

- `sam3` on Windows needs `triton-windows`, `einops`, `pycocotools`, `psutil` (none are
  declared by the package).
- HF cache warns about symlinks on Windows; harmless (Developer Mode would silence it).
- Paths inside JSON launch configs: use forward slashes (`\f` became a form feed once).
- The SAM 3.1 API bills one image per concept per frame: never point `meta_sam` at the
  continuous camera stream (~$54/h at 12 concepts, one frame every 2 s).
- Contributor mode on Spark trains Meta's models on our bench frames: keep it off.

## Key docs

- [`PRD.md`](PRD.md): scope, milestones (§13), acceptance criteria (§14), model split (§7.3)
- [`META_INTEGRATION.md`](META_INTEGRATION.md): per-model setup, prices, status table
