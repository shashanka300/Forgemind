# Plugging in Meta's models

Models are split by how often they run (PRD §7.3). Anything that runs on every frame, or
that the Model API doesn't offer, runs on the local GPU workstation (RTX 5090, 32 GB).
Reasoning and speech recognition go to the Meta Model API.

Everything model-specific sits behind the interfaces in `forgemind/providers/`, so moving
a slot between API and local is a `.env` change. Until you have keys, the system runs end
to end on stand-ins.

| Slot | Stand-in (no keys) | Target | Where | Switch | Status |
|---|---|---|---|---|---|
| Reasoning (agent + workers) | `mock` rule-based model | Muse Spark 1.3 (default), Muse Wonder (hard turns) | Model API | `FM_LLM_PROVIDER=meta` | Works |
| Speech-to-text | Browser speech recognition | Muse Voice Transcribe (streaming) | Model API | automatic when `MODEL_API_KEY` is set | Built, tested against a fake endpoint (section 3) |
| Segmentation, continuous | `clip_grid` (CLIP over tiles) | SAM 3 image model (open weights) | Local GPU | `FM_SEGMENTER=sam3` | Imports on Windows; waiting on `facebook/sam3` access |
| Segmentation, on demand | — | SAM 3.1 `sam-3.1` | Model API | `meta_sam` | Built, tested against a fake endpoint (section 2B) |
| Embeddings | CLIP ViT-B/32 (CPU) | Perception Encoder PE-Core-L14-336, if the crop test wins | Local GPU | `FM_EMBEDDER=pe` | Built; runs on the 5090 (16 crops in ~60 ms, 2.7 GB) |
| Fallback reasoning | — | Muse Glimmer 30B (Q4) | Local GPU | `FM_FALLBACK_PROVIDER=compat` | Switching built; Glimmer server not set up (section 4) |
| Text-to-speech | Browser `speechSynthesis` | Unchanged; Kokoro-82M if the voice is poor | Phone / local | — | Works |

The API has **no embeddings endpoint and no text-to-speech**, so those are always local.

## Prices and limits (Model API, checked Oct 26, 2026)

| Model | Price | RPM |
|---|---|---|
| `muse-wonder-1.0` | $4 in / $20 out per 1M tokens | 3,000 |
| `muse-spark-1.3` | $1.25 in / $4.25 out per 1M tokens | 3,000 |
| `muse-spark-1.3-contributor` | $0.10 / $0.20 per 1M tokens; **Meta trains on your prompts and images** | 150 |
| `muse-voice-transcribe-1.0` | $0.18 per audio hour | 256 concurrent streams |
| `sam-3.1` | $2.50 per 1k images; $0.20 per 1k video frames (min 50) | 3,000 |

Keep Contributor mode off: the images are frames of your bench.

## 1. Muse models via Meta Model API

The API is OpenAI-compatible (`https://api.meta.ai/v1`, key in `MODEL_API_KEY`,
`/chat/completions` with `tools`). `forgemind/providers/openai_compat.py` already
speaks it; the test suite runs the full agent loop against a fake Meta endpoint.

```bash
# .env
FM_LLM_PROVIDER=meta
MODEL_API_KEY=<your key>
FM_AGENT_MODEL=muse-spark-1.3      # switch to muse-wonder-1.0 for harder turns
FM_WORKER_MODEL=muse-spark-1.3

forgemind check      # lists models your key sees, then does a tool-call round trip
```

If `check` fails:
- **Model IDs differ** → copy the exact IDs from the list `check` prints into `.env`.
- **Region/access error from India** → raise it on the hackathon Discord; meanwhile use section 4.
- **`reasoning_effort` rejected** → leave `FM_REASONING_EFFORT` empty (Spark rejects `none`).
- **Images rejected** → set `FM_SEND_FRAMES=false`; the agent still gets the text situation summary.

Where the models are used:
- `agent/orchestrator.py`: every user turn (agent model), with the latest bench and phone
  frames attached when `FM_SEND_FRAMES=true`.
- `agent/tools.py::read_closeup`: the worker model reads instrument displays (JSON mode).
- `reports.py`: the worker model writes the summary paragraph, citing event ids only.

## 2. SAM 3.1

### A. Local GPU (continuous perception)

`Sam3Segmenter` in `providers/segmenters.py` runs SAM in-process. In the `sam3` package,
per-image segmentation (`build_sam3_image_model`) loads the **SAM 3** checkpoint
(`facebook/sam3`, `sam3.pt`); the SAM 3.1 checkpoint (`facebook/sam3.1`,
`sam3.1_multiplex.pt`) is the multiplex *video* tracker (`build_sam3_predictor`). Our
pipeline segments single frames, so it needs `facebook/sam3`; 3.1 matters only if we move
to video tracking. The RTX 5090 (Blackwell) needs PyTorch with CUDA 12.8 or newer.

```bash
# Both SAM repos are gated: request access on Hugging Face first
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[local]"     # sam3 + einops, pycocotools, psutil, triton-windows
hf auth login

# .env
FM_SEGMENTER=sam3
FM_SAM_DEVICE=cuda
FM_SAM_CHECKPOINT=          # empty = the package's default download
FM_SAM_MIN_SCORE=0.5
```

Things to verify on Day 2:
- `Sam3Segmenter` follows the SAM 3 README API (`build_sam3_image_model`,
  `Sam3Processor.set_image`, `set_text_prompt` → `masks/boxes/scores`). If SAM 3.1
  renames anything, that one class is the only code to change.
- Per-frame latency on the 5090 for the full `FM_CONCEPTS` list. Each concept is one
  prompt, so trim the list to what's actually on your bench.
- The `sam3` commit is pinned in `pyproject.toml` (`local` extra).

### B. Model API (on demand only)

`MetaApiSegmenter` (`FM_SEGMENTER=meta_sam`) sends one request per concept in parallel
(`POST /v1/responses`, model `sam-3.1`, the noun phrase plus the frame as a JPEG data URL,
`metadata.include_confidence: "true"`) and parses the special-token reply
(`<0f>0<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..;c=..|><|mask;...|>,1<|box;...`): pixel
xyxy boxes, a 0..1 confidence `c` compared with `FM_SAM_MIN_SCORE`, and a stable object
id kept as `track_id`. Masks are not decoded yet. `images_billed` counts billed images.

**Never use this per frame.** At one concept per request, 12 concepts on one bench frame
every 2 s is about 21,600 images, or ~$54, an hour. Use it to confirm a `where_is` result,
for close-ups and for teaching.

### C. Modal (fallback if the workstation is unavailable)

`deploy/modal_sam.py` packages the same `Sam3Segmenter` behind an HTTPS endpoint on
Modal's GPUs (hackathon credits).

```bash
pip install -e ".[deploy]" && modal setup
modal secret create huggingface HF_TOKEN=hf_xxx
modal secret create forgemind-sam FORGEMIND_SAM_TOKEN=$(openssl rand -hex 24)
modal deploy deploy/modal_sam.py                 # prints the endpoint URL
modal run deploy/modal_sam.py --image bench.jpg  # smoke test with a real photo

# .env
FM_SEGMENTER=remote_sam
FM_SAM_URL=https://<workspace>--forgemind-sam-sam-segment.modal.run
FM_SAM_TOKEN=<same token>
```

An L4 sleeps after 5 idle minutes; the first frame after that cold-starts in about
30–60 s. If the endpoint is down, frames return no detections, memory lookups, chat and
sessions keep working, and `/api/state` shows `segmenter_error`. Any other host can serve
the same protocol: `POST {image: base64 JPEG, concepts: [...], min_score}` →
`{detections: [{concept, box, score}]}`.

## 3. Speech-to-text: Muse Voice Transcribe

Today speech-to-text runs in the phone browser, which works well only in Chrome on
Android. The target is Muse Voice Transcribe (`muse-voice-transcribe-1.0`):

- Streaming: WebSocket `wss://api.meta.ai/v1/asr/realtime`
- Files: `POST /v1/asr/transcribe`
- Speaker labels and turn-level timestamps; no word-level timestamps.

Built (`server/asr.py`): while *Hold* is pressed the phone opens `/ws/asr` and streams
PCM16 16 kHz mono from an AudioWorklet; the server relays it to the realtime endpoint in
`PUSH_TO_TALK` mode, forwards cumulative partials to the text box, sends `endStream` on
release and hands the final transcript to the chat. `MODEL_API_KEY` never leaves the
server. Without a key the page uses the browser recognizer as before. Still to check
with a real key: that it works from India, the exact `endStream` message shape, and how
long the final transcript takes after release. Fallbacks: the browser
recognizer (current), or Parakeet TDT 0.6B v3 / Whisper large-v3-turbo on the local GPU.

## 4. Fallback reasoning: Meta open weights

Any OpenAI-compatible endpoint works through `FM_LLM_PROVIDER=compat`.

**Local: Muse Glimmer 30B** (Apache 2.0, text + image, built for tool calling). The 4-bit
build is about 17 GB and runs at roughly 75–84 tokens/s on an RTX 5090 with llama.cpp.
Cap the context near 32K so it fits beside SAM 3.1 (~4 GB) and the embedder (~1 GB).

```bash
# serve with llama.cpp's OpenAI-compatible server (or vLLM / SGLang)
llama-server -m <muse-glimmer Q4 gguf> -c 32768 -ngl 99 --jinja --port 8081

# .env
FM_LLM_PROVIDER=compat
FM_COMPAT_BASE_URL=http://localhost:8081/v1
FM_COMPAT_API_KEY=local
FM_COMPAT_AGENT_MODEL=<model id the server reports>
```

**Hosted** (partner credits: Baseten $100, Modal $100, Fireworks $25): point
`FM_COMPAT_BASE_URL` at the provider and copy the exact model id from its model page.

`forgemind check` confirms the endpoint returns tool calls before you rely on it.

**Automatic fallback** (PRD F-AGT-7): keep `FM_LLM_PROVIDER=meta` and set
`FM_FALLBACK_PROVIDER=compat` with the Glimmer endpoint in `FM_COMPAT_*`. When the Model API
errors, times out or rate-limits, the turn is retried on Glimmer and Glimmer stays in use
for `FM_FALLBACK_COOLDOWN` seconds (default 60) before the API is tried again.

## 5. Embeddings

`FM_EMBEDDER=clip` (CLIP ViT-B/32 via FastEmbed/ONNX, the gallery app's encoders) is the
default and is what `FM_TEACH_MIN_SIM=0.80` / `FM_MATCH_MIN_SIM=0.75` are tuned for.

Candidate upgrade: Meta Perception Encoder PE-Core-L14-336 (image + text,
`FM_EMBEDDER=pe`, OpenCLIP on the GPU, 1024-d, not gated). Decide with
the Day 3 crop test: CLIP vs PE-Core vs DINOv2 on ~50 crops of your own tools, measuring
same-object matching across both cameras. If the embedder changes, re-embed the `crops`
collection and recalibrate both thresholds.
