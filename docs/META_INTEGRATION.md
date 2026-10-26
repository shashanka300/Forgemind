# Plugging in Meta's models (day 1 checklist)

**Nothing runs on your machine.** Every model is a hosted API call. Your computer (or a
small cloud VM) only runs the ForgeMind app server: FastAPI, SQLite, and the CLIP
embeddings it uses for memory lookups, which run on the CPU.

Everything model-specific sits behind three interfaces in `forgemind/providers/`.
Until you have keys, the system runs end to end on stand-ins:

| Slot | Today (no keys) | With hackathon access | Switch |
|---|---|---|---|
| Reasoning (agent + workers) | `mock` rule-based model | Muse Wonder (agent), Muse Spark (workers) via Meta Model API | `FM_LLM_PROVIDER=meta` |
| Segmentation | `clip_grid` (CLIP over tiles, coarse boxes) | SAM 3.1 on a hosted GPU (Modal), or via Meta API if offered | `FM_SEGMENTER=remote_sam` / `meta_sam` |
| Embeddings | CLIP ViT-B/32 on the server CPU (gallery app's encoders) | unchanged | `FM_EMBEDDER=clip` |

## 1. Muse models via Meta Model API

The API is OpenAI-compatible (`https://api.meta.ai/v1`, key in `MODEL_API_KEY`,
`/chat/completions` with `tools`). `forgemind/providers/openai_compat.py` already
speaks it; the test suite runs the full agent loop against a fake Meta endpoint.

```bash
# .env
FM_LLM_PROVIDER=meta
MODEL_API_KEY=<your key>
FM_AGENT_MODEL=muse-wonder-1.0
FM_WORKER_MODEL=muse-spark-1.3

forgemind check      # lists models your key sees, then does a tool-call round trip
```

If `check` fails:
- **Model IDs differ** → copy the exact IDs from the list `check` prints into `.env`.
- **Region/access error from India** → raise it on the hackathon Discord; meanwhile use section 3.
- **`reasoning_effort` rejected** → leave `FM_REASONING_EFFORT` empty (Spark rejects `none`).
- **Images rejected** → set `FM_SEND_FRAMES=false`; the agent still gets the text situation summary.
- **Audio**: speech-to-text runs in the phone browser, so no audio goes to the API.
  Meta's docs say Spark 1.3 audio is degraded; try Wonder audio later if you want server-side STT.

Where the models are used:
- `agent/orchestrator.py`: every user turn (agent model), with the latest bench and phone
  frames attached when `FM_SEND_FRAMES=true`.
- `agent/tools.py::read_closeup`: the worker model reads instrument displays (JSON mode).
- `reports.py`: the worker model writes the summary paragraph, citing event ids only.

## 2. SAM 3.1 on a hosted GPU

### Option A: Modal (uses the hackathon's Modal credits)

`deploy/modal_sam.py` packages SAM 3 behind a small HTTPS endpoint. The GPU image is built
and run on Modal. It reuses the app's own `Sam3Segmenter`, so there's one copy of the SAM code.

```bash
pip install -e ".[deploy]" && modal setup
# SAM 3 checkpoints are gated: request access on Hugging Face first
modal secret create huggingface HF_TOKEN=hf_xxx
modal secret create forgemind-sam FORGEMIND_SAM_TOKEN=$(openssl rand -hex 24)
modal deploy deploy/modal_sam.py                 # prints the endpoint URL
modal run deploy/modal_sam.py --image bench.jpg  # smoke test with a real photo

# .env
FM_SEGMENTER=remote_sam
FM_SAM_URL=https://<workspace>--forgemind-sam-sam-segment.modal.run
FM_SAM_TOKEN=<same token>

forgemind check                                  # calls the endpoint and reports latency
```

Things to verify on day 1:
- `Sam3Segmenter` follows the SAM 3 README API (`build_sam3_image_model`,
  `Sam3Processor.set_image`, `set_text_prompt` → `masks/boxes/scores`). If SAM 3.1
  renames anything, that one class is the only code to change.
- Pin the `sam3` git commit in `modal_sam.py` once it works.

Cost and latency:
- An L4 sleeps after 5 idle minutes. The first frame after that cold-starts, which
  takes about 30–60 s. Set `MIN_CONTAINERS = 1` only while you're recording the demo.
- The bench camera is change-gated (SAM only runs when something moved), and the phone
  sends a frame only after the previous one returns, so a slow endpoint slows the
  overlay down rather than piling up requests.
- To make each call cheaper, trim `FM_CONCEPTS` to what's actually on your bench, since
  each concept is one prompt.
- If the endpoint is down, frames just return no detections. Memory lookups, chat and
  sessions keep working, and `/api/state` shows `segmenter_error`.

### Option B: Meta Model API, if SAM 3.1 is offered there

Fill in `MetaApiSegmenter.segment` in `providers/segmenters.py`, mapping the response to
`Detection(concept, box 0..1, score)`, then set `FM_SEGMENTER=meta_sam`.

### Any other host (Baseten, Replicate, your own)

Serve the same JSON protocol as `modal_sam.py` and point `FM_SAM_URL` at it:
`POST {image: base64 JPEG, concepts: [...], min_score}` → `{detections: [{concept, box, score}]}`.

## 3. Fallback reasoning: hosted Meta open weights, no Muse key needed

Any hosted OpenAI-compatible endpoint works, so you can test real reasoning before Muse
access arrives, using the partner credits:

```bash
# .env  (Fireworks shown; Baseten and Modal endpoints work the same way)
FM_LLM_PROVIDER=compat
FM_COMPAT_BASE_URL=https://api.fireworks.ai/inference/v1
FM_COMPAT_API_KEY=<fireworks key>
FM_COMPAT_AGENT_MODEL=<a Llama model id with tool calling, from the provider's model page>
```

`forgemind check` confirms the endpoint returns tool calls before you rely on it.
