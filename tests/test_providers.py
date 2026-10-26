"""Hosted SAM client against a fake endpoint speaking deploy/modal_sam.py's protocol."""
import base64
import io
import json

import httpx
from PIL import Image

from forgemind.providers.segmenters import RemoteSamSegmenter


def test_remote_sam_roundtrip_and_auth():
    seen = {}

    def handler(request):
        body = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        img = Image.open(io.BytesIO(base64.b64decode(body["image"])))
        seen["size"] = img.size
        seen["concepts"] = body["concepts"]
        return httpx.Response(200, json={"ms": 42, "detections": [
            {"concept": "multimeter", "box": [0.1, 0.2, 0.3, 0.4], "score": 0.91},
            {"concept": "multimeter", "box": [0.11, 0.2, 0.3, 0.4], "score": 0.80},  # dup -> NMS
            {"concept": "unicorn", "box": [0, 0, 1, 1], "score": 0.99}]})        # not asked for

    seg = RemoteSamSegmenter("https://x.modal.run", "tok", 0.5,
                             http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    dets = seg.segment(Image.new("RGB", (2000, 1500)), ["multimeter", "hand"])
    assert seen["auth"] == "Bearer tok" and max(seen["size"]) == 1024
    assert seen["concepts"] == ["multimeter", "hand"]
    assert [(d.concept, d.box, d.score) for d in dets] == [("multimeter", (0.1, 0.2, 0.3, 0.4), 0.91)]
    assert seg.last_error is None and seg.last_ms is not None


def test_remote_sam_failure_degrades_to_no_detections():
    seg = RemoteSamSegmenter("https://x.modal.run", "tok", http_client=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"detail": "bad token"}))))
    assert seg.segment(Image.new("RGB", (64, 64)), ["multimeter"]) == []
    assert "401" in seg.last_error
