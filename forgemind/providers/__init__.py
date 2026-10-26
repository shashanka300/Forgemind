"""Builds providers from Settings. The rest of the code only sees the interfaces."""
from forgemind.providers.base import ChatResult, Detection, ToolCall  # noqa: F401


def build_embedder(settings):
    from forgemind.providers.embedders import ClipEmbedder, HashEmbedder
    if settings.embedder == "hash":
        return HashEmbedder()
    return ClipEmbedder(cache_dir=str(settings.data_dir / "models"))


def build_segmenter(settings, embedder):
    from forgemind.providers import segmenters as s
    kind = settings.segmenter
    if kind == "remote_sam":
        return s.RemoteSamSegmenter(settings.sam_url, settings.sam_token,
                                    settings.sam_min_score)
    if kind == "sam3":   # only inside a GPU worker; nothing runs on your machine
        return s.Sam3Segmenter(settings.sam_checkpoint, settings.sam_device,
                               settings.sam_min_score)
    if kind == "meta_sam":
        return s.MetaApiSegmenter(settings)
    if kind == "none":
        return s.NullSegmenter()
    if kind == "clip_grid":
        return s.ClipGridSegmenter(embedder)
    raise ValueError(f"unknown FM_SEGMENTER={kind}")


def build_llm(settings):
    kind = settings.llm_provider
    if kind == "mock":
        from forgemind.providers.mock_llm import MockModel
        return MockModel()
    from forgemind.providers.openai_compat import OpenAICompatModel
    if kind == "meta":
        if not settings.meta_api_key:
            raise ValueError("FM_LLM_PROVIDER=meta needs MODEL_API_KEY in .env")
        return OpenAICompatModel(settings.meta_base_url, settings.meta_api_key,
                                 settings.agent_model, settings.worker_model, name="meta",
                                 reasoning_effort=settings.reasoning_effort)
    if kind == "compat":
        return OpenAICompatModel(settings.compat_base_url, settings.compat_api_key,
                                 settings.compat_agent_model or settings.agent_model,
                                 settings.compat_worker_model or settings.compat_agent_model,
                                 name="compat", reasoning_effort=settings.reasoning_effort)
    raise ValueError(f"unknown FM_LLM_PROVIDER={kind}")
