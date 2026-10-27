"""Spend ledger and the hard API budget cap."""
import json

import httpx
import pytest
from PIL import Image

from forgemind.config import Settings
from forgemind.providers.base import ChatResult
from forgemind.providers.budget import BudgetExceeded, Ledger, MeteredLLM, token_cost


class Paid:
    name, supports_vision = "meta", True

    def __init__(self):
        self.calls = 0

    def model_for(self, role):
        return "muse-spark-1.3"

    def chat(self, messages, tools=None, model=None, json_mode=False):
        self.calls += 1
        return ChatResult(content="ok", usage={"prompt_tokens": 3000, "completion_tokens": 100,
                                               "prompt_tokens_details": {"cached_tokens": 2000}})


def test_token_cost_uses_cached_rate():
    usd = token_cost("muse-spark-1.3", {"prompt_tokens": 3000, "completion_tokens": 100,
                                        "prompt_tokens_details": {"cached_tokens": 2000}})
    assert usd == pytest.approx((1000 * 1.25 + 2000 * 0.15 + 100 * 4.25) / 1e6)


def test_metered_llm_records_warns_and_stops_at_the_cap(tmp_path):
    warnings = []
    led = Ledger(tmp_path / "l.sqlite", cap_usd=0.0062, on_warn=warnings.append)
    llm = MeteredLLM(Paid(), led)
    llm.chat([])                                   # $0.001975 each
    assert led.total() == pytest.approx(0.001975) and warnings == []
    llm.chat([])                                   # crosses 1/3 of the cap
    assert len(warnings) == 1
    llm.chat([])                                   # crosses 2/3
    assert len(warnings) == 2
    llm.chat([])                                   # total 0.0079 >= cap 0.0062
    with pytest.raises(BudgetExceeded):
        llm.chat([])
    assert llm.llm.calls == 4
    assert led.summary()["by_kind"]["llm"]["calls"] == 4


def test_fallback_takes_over_when_the_budget_is_spent(tmp_path):
    from forgemind.providers.fallback import FallbackModel
    led = Ledger(tmp_path / "l.sqlite", cap_usd=0.0)
    local = Paid()
    local.name = "compat"
    llm = FallbackModel(MeteredLLM(Paid(), led), local)
    assert llm.chat([]).content == "ok" and local.calls == 1 and "budget" in llm.last_error


def test_sam_api_calls_are_billed_per_image(tmp_path, monkeypatch):
    from forgemind.providers.segmenters import MetaApiSegmenter
    led = Ledger(tmp_path / "l.sqlite", cap_usd=100)
    ok = httpx.Response(200, json={"output": [{"type": "message", "content": [
        {"type": "output_text", "text": ""}]}]})
    seg = MetaApiSegmenter(Settings.from_env(meta_api_key="k"), ledger=led,
                           http_client=httpx.Client(transport=httpx.MockTransport(lambda r: ok)))
    seg.segment(Image.new("RGB", (64, 64)), ["multimeter", "pliers", "hand"])
    assert led.total() == pytest.approx(3 * 2.5 / 1000)
    led.cap = 0.001
    assert seg.segment(Image.new("RGB", (64, 64)), ["multimeter"]) == []
    assert "budget" in seg.last_error


def test_asr_relay_bills_audio_seconds(tmp_path):
    from forgemind.server.asr import VoiceTranscribeRelay
    led = Ledger(tmp_path / "l.sqlite", cap_usd=100)
    relay = VoiceTranscribeRelay(Settings.from_env(meta_api_key="k"), ledger=led)
    relay._bytes = 32000 * 90 + 500               # 90.0 s of 16 kHz PCM
    relay._bill()
    assert led.total() == pytest.approx(90 / 3600 * 0.18)
