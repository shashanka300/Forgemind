"""Settings, read once from the environment (and .env if present)."""
import json
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import find_dotenv, load_dotenv
    # Look for .env from the directory you run forgemind in (not the package's).
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:  # pragma: no cover
    pass


from forgemind.perception import bench as _bench  # noqa: E402


def _env(name, default=""):
    v = os.environ.get(name)
    return default if v is None or v.strip() == "" else v.strip()


def _bool(name, default):
    return _env(name, str(default)).lower() in ("1", "true", "yes", "on")


# "soldering pen", not "soldering iron": SAM 3.1 never answers to the latter.
# Teach the iron as "soldering iron" (the safety watch matches taught names).
DEFAULT_CONCEPTS = ("multimeter,soldering pen,screwdriver,hex key,pliers,breadboard,"
                    "circuit board,flux,person,hand,smoke")
DEFAULT_ZONES = '{"0": "bench", "1": "shelf", "2": "drawer", "3": "cabinet", "4": "kitchen"}'
# Detections of these concepts mean "someone is here"; they are never stored
# as objects in memory (no identities for people).
PRESENCE_CONCEPTS = {"person", "hand"}
HAZARD_CONCEPTS = {"smoke", "fire", "flame"}


@dataclass
class Settings:
    host: str = "0.0.0.0"
    port: int = 8770
    data_dir: Path = Path("forgemind-data")
    token: str | None = None

    llm_provider: str = "mock"
    meta_api_key: str = ""
    meta_base_url: str = "https://api.meta.ai/v1"
    agent_model: str = "muse-wonder-1.0"
    worker_model: str = "muse-spark-1.3"
    reasoning_effort: str = ""
    compat_base_url: str = "https://api.fireworks.ai/inference/v1"
    compat_api_key: str = "not-needed"
    compat_agent_model: str = ""
    compat_worker_model: str = ""
    send_frames: bool = True
    fallback_provider: str = ""
    fallback_cooldown: float = 60.0

    segmenter: str = "clip_grid"
    sam_url: str = ""
    sam_token: str = ""
    sam_checkpoint: str = "sam3.1"
    sam_device: str = "cuda"
    sam_min_score: float = 0.5
    embedder: str = "clip"
    vector_backend: str = "qdrant_edge"
    concepts: list[str] = field(default_factory=lambda: DEFAULT_CONCEPTS.split(","))
    zones: dict[int, str] = field(default_factory=dict)
    static_zone: str = "bench"
    bench_roi: tuple | None = None
    bench_masks: list = field(default_factory=list)

    default_watch: str = "soldering iron"
    unattended_seconds: float = 60

    teach_min_sim: float = 0.80
    confirm_new: bool = True
    notes: bool = True
    match_min_sim: float = 0.75
    confidence_half_life: float = 1800
    stale_confidence: float = 0.4

    @classmethod
    def from_env(cls, **overrides):
        zones_raw = json.loads(_env("FM_ZONES", DEFAULT_ZONES))
        s = cls(
            host=_env("FM_HOST", "0.0.0.0"),
            port=int(_env("FM_PORT", "8770")),
            data_dir=Path(_env("FM_DATA_DIR", "forgemind-data")),
            token=_env("FM_TOKEN", "auto"),
            llm_provider=_env("FM_LLM_PROVIDER", "mock").lower(),
            meta_api_key=_env("MODEL_API_KEY"),
            meta_base_url=_env("META_BASE_URL", "https://api.meta.ai/v1"),
            agent_model=_env("FM_AGENT_MODEL", "muse-wonder-1.0"),
            worker_model=_env("FM_WORKER_MODEL", "muse-spark-1.3"),
            reasoning_effort=_env("FM_REASONING_EFFORT"),
            compat_base_url=_env("FM_COMPAT_BASE_URL", "https://api.fireworks.ai/inference/v1"),
            compat_api_key=_env("FM_COMPAT_API_KEY", "not-needed"),
            compat_agent_model=_env("FM_COMPAT_AGENT_MODEL"),
            compat_worker_model=_env("FM_COMPAT_WORKER_MODEL"),
            send_frames=_bool("FM_SEND_FRAMES", True),
            fallback_provider=_env("FM_FALLBACK_PROVIDER").lower(),
            fallback_cooldown=float(_env("FM_FALLBACK_COOLDOWN", "60")),
            segmenter=_env("FM_SEGMENTER", "clip_grid").lower(),
            sam_url=_env("FM_SAM_URL"),
            sam_token=_env("FM_SAM_TOKEN"),
            sam_checkpoint=_env("FM_SAM_CHECKPOINT", "sam3.1"),
            sam_device=_env("FM_SAM_DEVICE", "cuda"),
            sam_min_score=float(_env("FM_SAM_MIN_SCORE", "0.5")),
            embedder=_env("FM_EMBEDDER", "clip").lower(),
            vector_backend=_env("FM_VECTOR_BACKEND", "qdrant_edge").lower(),
            concepts=[c.strip() for c in _env("FM_CONCEPTS", DEFAULT_CONCEPTS).split(",")
                      if c.strip()],
            zones={int(k): v for k, v in zones_raw.items()},
            static_zone=_env("FM_STATIC_ZONE", "bench"),
            bench_roi=_bench.parse_roi(_env("FM_BENCH_ROI")),
            bench_masks=_bench.parse_masks(_env("FM_BENCH_MASK")),
            default_watch=_env("FM_DEFAULT_WATCH", "soldering iron"),
            unattended_seconds=float(_env("FM_UNATTENDED_SECONDS", "60")),
            teach_min_sim=float(_env("FM_TEACH_MIN_SIM", "0.80")),
            confirm_new=_bool("FM_CONFIRM_NEW", True),
            notes=_bool("FM_NOTES", True),
            match_min_sim=float(_env("FM_MATCH_MIN_SIM", "0.75")),
            confidence_half_life=float(_env("FM_CONFIDENCE_HALF_LIFE", "1800")),
            stale_confidence=float(_env("FM_STALE_CONFIDENCE", "0.4")),
        )
        for k, v in overrides.items():
            setattr(s, k, v)
        if s.token is not None and s.token.lower() in ("", "auto"):
            # An empty value must never mean "no token".
            s.token = secrets.token_urlsafe(18)
        if s.token is not None and s.token.lower() == "off":
            s.token = None
        s.data_dir = Path(s.data_dir)
        return s

    @property
    def object_concepts(self):
        return [c for c in self.concepts
                if c not in PRESENCE_CONCEPTS and c not in HAZARD_CONCEPTS]
