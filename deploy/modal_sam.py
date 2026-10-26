"""SAM 3 / 3.1 as a hosted endpoint on Modal (serverless GPU, uses the
hackathon's Modal credits). Nothing runs on your machine.

One-time setup (from the Forgemind folder):
    pip install modal && modal setup
    # SAM 3 checkpoints are gated: request access on Hugging Face first
    modal secret create huggingface HF_TOKEN=hf_xxx
    modal secret create forgemind-sam FORGEMIND_SAM_TOKEN=<any long random string>

Deploy, then test with a photo of your bench:
    modal deploy deploy/modal_sam.py
    modal run deploy/modal_sam.py --image bench.jpg

Put the printed URL and the token in .env:
    FM_SEGMENTER=remote_sam
    FM_SAM_URL=https://<workspace>--forgemind-sam-sam-segment.modal.run
    FM_SAM_TOKEN=<the same random string>

Cost control: the container shuts down after SCALEDOWN_S idle seconds and
cold-starts on the next frame (expect ~30-60 s the first time). Set
MIN_CONTAINERS=1 only while recording the demo.
"""
import base64
import hmac
import io
import os
import time

import modal
from fastapi import HTTPException, Request

GPU = "L4"
SCALEDOWN_S = 300
MIN_CONTAINERS = 0

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch>=2.7", "torchvision", "numpy", "pillow", "huggingface_hub",
                 "fastapi[standard]")
    # TODO(day 1): pin to the SAM 3.1 release/commit once it's confirmed.
    .pip_install("git+https://github.com/facebookresearch/sam3.git")
    .add_local_python_source("forgemind")      # reuses Sam3Segmenter from the app
)

app = modal.App("forgemind-sam")
hf_cache = modal.Volume.from_name("forgemind-hf-cache", create_if_missing=True)


@app.cls(image=image, gpu=GPU, timeout=600, scaledown_window=SCALEDOWN_S,
         min_containers=MIN_CONTAINERS,
         secrets=[modal.Secret.from_name("huggingface"),
                  modal.Secret.from_name("forgemind-sam")],
         volumes={"/root/.cache/huggingface": hf_cache})
class Sam:
    @modal.enter()
    def load(self):
        from forgemind.providers.segmenters import Sam3Segmenter
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ.get("HF_TOKEN", ""))
        self.seg = Sam3Segmenter(device="cuda", min_score=0.0)
        self.token = os.environ.get("FORGEMIND_SAM_TOKEN", "")

    def _run(self, jpeg, concepts, min_score):
        from PIL import Image
        t0 = time.perf_counter()
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        dets = [d for d in self.seg.segment(img, concepts) if d.score >= min_score]
        return {"detections": [{"concept": d.concept, "box": [round(b, 4) for b in d.box],
                                "score": round(d.score, 4)} for d in dets],
                "ms": round((time.perf_counter() - t0) * 1000)}

    @modal.method()
    def run(self, jpeg: bytes, concepts: list[str], min_score: float = 0.5):
        return self._run(jpeg, concepts, min_score)

    @modal.fastapi_endpoint(method="POST")
    async def segment(self, request: Request):
        given = (request.headers.get("authorization") or "").removeprefix("Bearer ").strip()
        if not self.token or not hmac.compare_digest(given, self.token):
            raise HTTPException(401, "bad token")
        body = await request.json()
        concepts = [str(c) for c in body.get("concepts", [])][:32]
        if not concepts or "image" not in body:
            raise HTTPException(400, "need image and concepts")
        return self._run(base64.b64decode(body["image"]), concepts,
                         float(body.get("min_score", 0.5)))


@app.local_entrypoint()
def main(image: str, concepts: str = "multimeter,soldering iron,screwdriver,pliers,hand"):
    with open(image, "rb") as f:
        out = Sam().run.remote(f.read(), concepts.split(","))
    print(f"{len(out['detections'])} detections in {out['ms']} ms")
    for d in out["detections"]:
        print(f"  {d['concept']:<16} {d['score']:.2f}  {d['box']}")
