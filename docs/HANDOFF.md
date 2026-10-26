# Handoff

Where the build stands and what to do next. Update this at the end of each work session.

**Last updated:** Oct 27, 2026 (Day 3 started). **Deadline:** Nov 5, 7:30 AM IST (Nov 4, 9 PM ET).
**Branch:** `main`. Check `git status -sb` for unpushed commits.

## Status

| Piece | State | Verified how |
|---|---|---|
| Full loop in mock mode (cameras → memory → agent → phone) | Works | 37 tests; phone page loads, chat round trip in browser |
| Object registry + teach identity (F-OBJ-1..5) | Existed in skeleton; teaching improved | Teach now takes up to 3 views from the last 4 s of phone frames, asks SAM for the named concept if the frame's detections missed it, and says when it fell back to the centre of the view. Tests only (colour-block fixtures) |
| Embedder calibration (`forgemind calibrate`) | Built | Synthetic smoke set only (see below). Real answer needs the user's photos |
| SAM 3.1 via Model API (`FM_SEGMENTER=meta_sam`) | Built | Fake endpoint only |
| Voice Transcribe relay (`/ws/asr`, phone AudioWorklet → server → Meta) | Built | Fake endpoint only; mic path not tried on a phone |
| LLM fallback to Glimmer (`FM_FALLBACK_PROVIDER=compat`) | Built | Stub models only; no Glimmer server yet |
| Perception Encoder embedder (`FM_EMBEDDER=pe`) | Built | Runs on the 5090: 16 crops ~60 ms, 2.7 GB VRAM. Thresholds not recalibrated |
| Local SAM 3 (`FM_SEGMENTER=sam3`) | Runs on the 5090 | Two bench photos: ~490 ms for 12 concepts, 80 ms for one, 4.1 GB VRAM. SAM 3.1 detector weights give the same detections. Misses: soldering iron read as "screwdriver" in one photo; irons, stations and pliers missed in the other |
| Muse Spark / Wonder via Model API | Code existed before | Not run: no `MODEL_API_KEY` yet |

## Blockers

1. ~~SAM access~~ Granted for both `facebook/sam3` and `facebook/sam3.1`.
2. **`MODEL_API_KEY`.** Not on this machine. The user adds it to `.env` themselves.

## Next steps (in order)

1. **SAM follow-ups:** the 3.1 checkpoint is now the default (`FM_SAM_CHECKPOINT=sam3.1`;
   the "missing convs.3 keys" message at load is harmless, see `Sam3Segmenter`). Tune concept names and the 0.5 cutoff on photos of
   the user's own bench ("soldering iron" vs "soldering station"). Benchmark script:
   `sam_bench.py` (scratchpad; move into the repo if kept). `Sam3Segmenter` now runs
   under bf16 autocast (required).
2. **When the key is in `.env`:** `forgemind check`; one `meta_sam` call on a real photo
   (confirm the reply format and that `c` scores come back); one `/ws/asr` utterance from
   a phone over zrok (confirm it works from India, the `endStream` message shape, and
   time to final transcript).
3. **Embedder crop test (Day 3), waiting on the user's photos.** Protocol:
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
4. **Glimmer server:** download the Q4 GGUF, serve with llama.cpp (`-c 32768`), point
   `FM_COMPAT_*` at it, confirm tool calls with `forgemind check`.
5. **Day 3 remaining:** decide embedder from the real crop test; if DINO wins, add a
   second (identity) vector to the registry; recalibrate thresholds; try teach on the
   real phone over zrok. Then Day 4: `where_is` end to end.
6. **Added from the personal health agent paper** (PRD §7.4, Nature 2026), scheduled in §13:
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

## Gotchas

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
