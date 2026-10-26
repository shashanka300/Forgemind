# Handoff

Where the build stands and what to do next. Update this at the end of each work session.

**Last updated:** Oct 27, 2026 (Day 2). **Deadline:** Nov 5, 7:30 AM IST (Nov 4, 9 PM ET).
**Branch:** `main`. Local commit `e2db319` is not pushed yet (`ac36304` is on GitHub).

## Status

| Piece | State | Verified how |
|---|---|---|
| Full loop in mock mode (cameras → memory → agent → phone) | Works | 29 tests; phone page loads, chat round trip in browser |
| SAM 3.1 via Model API (`FM_SEGMENTER=meta_sam`) | Built | Fake endpoint only |
| Voice Transcribe relay (`/ws/asr`, phone AudioWorklet → server → Meta) | Built | Fake endpoint only; mic path not tried on a phone |
| LLM fallback to Glimmer (`FM_FALLBACK_PROVIDER=compat`) | Built | Stub models only; no Glimmer server yet |
| Perception Encoder embedder (`FM_EMBEDDER=pe`) | Built | Runs on the 5090: 16 crops ~60 ms, 2.7 GB VRAM. Thresholds not recalibrated |
| Local SAM 3 (`FM_SEGMENTER=sam3`) | Imports on Windows | Not run: waiting on gated access |
| Muse Spark / Wonder via Model API | Code existed before | Not run: no `MODEL_API_KEY` yet |

## Blockers

1. **SAM access.** HF login works (`shashanka300`) but both `facebook/sam3` and
   `facebook/sam3.1` return GatedRepoError. Expected by the morning of Oct 28.
   Per-frame segmentation needs **`facebook/sam3`** (`build_sam3_image_model` loads
   `sam3.pt`); `sam3.1_multiplex.pt` is the video tracker only.
2. **`MODEL_API_KEY`.** Not on this machine. The user adds it to `.env` themselves.

## Next steps (in order)

1. **When SAM access lands:** load `build_sam3_image_model()` on the 5090, run
   `Sam3Segmenter` on a few bench photos, measure ms/frame for the full `FM_CONCEPTS`
   list, check the `set_text_prompt` output keys still match the wrapper.
2. **When the key is in `.env`:** `forgemind check`; one `meta_sam` call on a real photo
   (confirm the reply format and that `c` scores come back); one `/ws/asr` utterance from
   a phone over zrok (confirm it works from India, the `endStream` message shape, and
   time to final transcript).
3. **Embedder crop test (Day 3):** user to photograph ~50 crops of their own tools from
   both camera positions. Compare CLIP vs PE-Core vs DINOv2 on same-object matching, pick
   one, recalibrate `FM_TEACH_MIN_SIM` / `FM_MATCH_MIN_SIM`.
4. **Glimmer server:** download the Q4 GGUF, serve with llama.cpp (`-c 32768`), point
   `FM_COMPAT_*` at it, confirm tool calls with `forgemind check`.
5. Then back on the PRD milestones (§13): Day 3 registry + teach identity, Day 4
   `where_is` end to end.

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
