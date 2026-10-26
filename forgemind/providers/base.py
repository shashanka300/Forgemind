"""Provider interfaces. Every model ForgeMind uses sits behind one of these,
so swapping mock -> Meta Model API -> hosted open weights is a config change.
No model runs on the user's machine; everything heavy is an API call.
"""
from dataclasses import dataclass, field
from typing import Any, Protocol

from PIL import Image


@dataclass
class Detection:
    """One segmented object. Boxes are normalized (0..1) x0, y0, x1, y1."""
    concept: str
    box: tuple[float, float, float, float]
    score: float
    mask: Any = None          # optional HxW bool array (SAM); not persisted
    track_id: int | None = None

    @property
    def center(self):
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)

    @property
    def area(self):
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])


class Segmenter(Protocol):
    name: str

    def segment(self, image: Image.Image, concepts: list[str]) -> list[Detection]:
        """Find every instance of each text concept in the image."""


class Embedder(Protocol):
    name: str
    dim: int

    def embed_image(self, image: Image.Image) -> list[float]: ...
    def embed_images(self, images: list[Image.Image]) -> list[list[float]]: ...
    def embed_text(self, text: str) -> list[float]: ...


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ChatResult:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None
    usage: dict | None = None


class ReasoningModel(Protocol):
    """OpenAI chat-completions shaped: messages in, text and/or tool calls out."""
    name: str
    supports_vision: bool

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             model: str | None = None, json_mode: bool = False) -> ChatResult: ...

    def model_for(self, role: str) -> str:
        """Model ID for a role: 'agent' (main turns) or 'worker' (subtasks)."""
