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


SAM_LINE = ("<0f>0<|box;x1=353;y1=382;x2=453;y2=427;w=537;h=561;c=0.91|>"
            "<|mask;x=0;y=0;c=0.9;data=46,101,~a,b<|box;|>,1<|box;x1=260;y1=382;x2=340;y2=427;"
            "w=537;h=561;c=0.42|><|mask;x=0;y=0;data=46,81,~xyz|>")


def test_parse_sam_text_boxes_ids_confidence():
    from forgemind.providers.segmenters import parse_sam_text
    recs = parse_sam_text(SAM_LINE + "\n<3f>2<|box;x1=1;y1=2;x2=3;y2=4;w=10;h=10|>")
    assert [(r["id"], r["frame"], r["c"]) for r in recs] == [(0, 0, 0.91), (1, 0, 0.42),
                                                              (2, 3, None)]
    assert (recs[0]["x1"], recs[0]["y2"], recs[0]["w"]) == (353, 427, 537)
    assert parse_sam_text("") == []


def test_meta_sam_one_request_per_concept_and_normalized_boxes():
    from forgemind.config import Settings
    from forgemind.providers.segmenters import MetaApiSegmenter
    seen = []

    def handler(request):
        body = json.loads(request.content)
        parts = body["input"][0]["content"]
        concept = parts[0]["text"]
        seen.append((request.url.path, request.headers["authorization"], body["model"],
                     concept, parts[1]["image_url"][:23],
                     body["metadata"]["include_confidence"]))
        text = SAM_LINE if concept == "multimeter" else ""
        return httpx.Response(200, json={"output": [{"type": "message", "content": [
            {"type": "output_text", "text": text}]}]})

    s = Settings.from_env(meta_api_key="k", sam_min_score=0.5)
    seg = MetaApiSegmenter(s, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    dets = seg.segment(Image.new("RGB", (537, 561)), ["multimeter", "hex key"])
    assert sorted(x[3] for x in seen) == ["hex key", "multimeter"]
    assert all(x[:3] == ("/v1/responses", "Bearer k", "sam-3.1") for x in seen)
    assert all(x[4] == "data:image/jpeg;base64," and x[5] == "true" for x in seen)
    assert len(dets) == 1                      # c=0.42 is under min_score
    d = dets[0]
    assert d.concept == "multimeter" and d.score == 0.91 and d.track_id == 0
    assert d.box == (353 / 537, 382 / 561, 453 / 537, 427 / 561)
    assert seg.images_billed == 2 and seg.last_error is None


def test_meta_sam_failure_degrades_to_no_detections():
    from forgemind.config import Settings
    from forgemind.providers.segmenters import MetaApiSegmenter
    seg = MetaApiSegmenter(Settings.from_env(meta_api_key="k"), http_client=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(429, json={"error": "slow"}))))
    assert seg.segment(Image.new("RGB", (64, 64)), ["multimeter"]) == []
    assert "429" in seg.last_error


class _Stub:
    supports_vision = True

    def __init__(self, name, fail=False):
        self.name, self.fail, self.calls = name, fail, []

    def model_for(self, role):
        return f"{self.name}-{role}"

    def chat(self, messages, tools=None, model=None, json_mode=False):
        self.calls.append(model)
        if self.fail:
            raise TimeoutError("api down")
        from forgemind.providers.base import ChatResult
        return ChatResult(content=f"from {self.name}")


def test_fallback_switches_on_error_and_maps_roles(monkeypatch):
    from forgemind.providers import fallback as fbm
    primary, local = _Stub("meta", fail=True), _Stub("glimmer")
    llm = fbm.FallbackModel(primary, local, cooldown=60)
    now = [1000.0]
    monkeypatch.setattr(fbm.time, "time", lambda: now[0])
    assert llm.chat([], model="meta-worker").content == "from glimmer"
    assert local.calls == ["glimmer-worker"] and "api down" in llm.last_error
    llm.chat([], model="meta-agent")                 # inside cooldown: primary not retried
    assert primary.calls == ["meta-worker"] and local.calls[-1] == "glimmer-agent"
    primary.fail, now[0] = False, 1061.0             # cooldown over, API back
    assert llm.chat([]).content == "from meta" and llm.active == "meta"


def test_build_llm_wraps_fallback():
    from forgemind.config import Settings
    from forgemind.providers import build_llm
    s = Settings.from_env(llm_provider="meta", meta_api_key="k", fallback_provider="compat",
                          compat_base_url="http://localhost:8081/v1", compat_api_key="local",
                          compat_agent_model="glimmer")
    llm = build_llm(s)
    assert llm.name == "meta+compat" and llm.model_for("agent") == s.agent_model
    assert llm.fallback.model_for("worker") == "glimmer"
