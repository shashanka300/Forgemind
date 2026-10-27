"""Test fixtures: no model downloads, no network.

ColorSegmenter stands in for SAM: solid colour blocks are "objects"
(red = soldering iron, blue = multimeter, green = hex key, yellow = person).
"""
import io

import numpy as np
import pytest
from PIL import Image

from forgemind.config import Settings
from forgemind.providers.base import Detection
from forgemind.providers.embedders import HashEmbedder

COLORS = {"soldering iron": (220, 30, 30), "multimeter": (30, 60, 220),
          "hex key": (30, 190, 60), "person": (240, 220, 40)}


class ColorSegmenter:
    name = "color"

    def segment(self, image, concepts):
        a = np.asarray(image.convert("RGB")).astype(int)
        h, w = a.shape[:2]
        out = []
        for concept, rgb in COLORS.items():
            if concept not in concepts:
                continue
            m = (np.abs(a - np.array(rgb)).sum(axis=2) < 60)
            if m.sum() < 0.005 * h * w:
                continue
            ys, xs = np.nonzero(m)
            out.append(Detection(concept, (xs.min() / w, ys.min() / h, (xs.max() + 1) / w,
                                           (ys.max() + 1) / h), 0.9))
        return out


def scene(blocks=(), tag=None, size=(640, 480)):
    """A grey frame with colour blocks [(concept, x0, y0, x1, y1) in 0..1] and an optional AprilTag."""
    img = Image.new("RGB", size, (200, 200, 200))
    a = np.asarray(img).copy()
    w, h = size
    for concept, x0, y0, x1, y1 in blocks:
        a[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = COLORS[concept]
    if tag is not None:
        import cv2
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        m = cv2.aruco.generateImageMarker(d, tag, 120)
        pad = np.full((160, 160), 255, np.uint8)
        pad[20:140, 20:140] = m
        a[10:170, w - 170:w - 10] = np.stack([pad] * 3, axis=2)
    return Image.fromarray(a)


def jpeg(img):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _isolated_ledger(tmp_path, monkeypatch):
    """Never write test calls into the real spend ledger (~/.forgemind)."""
    monkeypatch.setenv("FM_LEDGER", str(tmp_path / "ledger.sqlite"))


@pytest.fixture
def settings(tmp_path):
    # Never use the developer's real key or providers from .env in tests.
    return Settings.from_env(data_dir=tmp_path / "data", embedder="hash", meta_api_key="", confirm_new=False, notes=False, report_async=False,
                             fallback_provider="", sam_checkpoint="sam3.1",
                             vector_backend="numpy", llm_provider="mock", segmenter="none",
                             token="secret-token", send_frames=False,
                             concepts=["soldering iron", "multimeter", "hex key", "person",
                                       "smoke"],
                             zones={0: "bench", 1: "shelf", 2: "drawer"}, static_zone="bench",
                             unattended_seconds=60, default_watch="soldering iron")


@pytest.fixture
def rt(settings):
    from forgemind.runtime import Runtime
    r = Runtime(settings, segmenter=ColorSegmenter(), embedder=HashEmbedder(), warm=False)
    r.ctx.capture_timeout = 3.0
    yield r
    r.close()
