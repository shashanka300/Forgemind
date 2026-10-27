"""Image/text embedders.

`clip` is the gallery app's encoder pair (CLIP ViT-B/32 vision + text towers
via FastEmbed/ONNX, CPU, no PyTorch), so thresholds learned there carry over.
`pe` is Meta's Perception Encoder (PE-Core-L14-336 via OpenCLIP, PyTorch, GPU
if available): stronger image/text matching, but its similarities live on a
different scale, so FM_TEACH_MIN_SIM / FM_MATCH_MIN_SIM must be recalibrated.
`hash` is a tiny deterministic embedder for tests and offline smoke runs.
"""
import hashlib
import re
import threading
from functools import lru_cache

import numpy as np
from PIL import Image

CLIP_VISION_MODEL = "Qdrant/clip-ViT-B-32-vision"
CLIP_TEXT_MODEL = "Qdrant/clip-ViT-B-32-text"


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v)) or 1.0
    return (v / n).tolist()


class ClipEmbedder:
    name = "clip"
    dim = 512
    # Text->image cosine below this is noise for `where_is` by description.
    text_min_sim = 0.22

    def __init__(self, cache_dir=None, threads=None):
        self.cache_dir = cache_dir
        self.threads = threads
        self._lock = threading.Lock()

    @lru_cache(maxsize=1)
    def _vision(self):
        from fastembed import ImageEmbedding
        return ImageEmbedding(CLIP_VISION_MODEL, threads=self.threads, cache_dir=self.cache_dir)

    @lru_cache(maxsize=1)
    def _text(self):
        from fastembed import TextEmbedding
        return TextEmbedding(CLIP_TEXT_MODEL, threads=self.threads, cache_dir=self.cache_dir)

    def embed_images(self, images):
        if not images:
            return []
        with self._lock:
            return [_unit(v) for v in self._vision().embed([i.convert("RGB") for i in images])]

    def embed_image(self, image):
        return self.embed_images([image])[0]

    @lru_cache(maxsize=512)
    def _text_cached(self, text):
        with self._lock:
            return tuple(_unit(next(self._text().embed([text]))))

    def embed_text(self, text):
        return list(self._text_cached(text))

    def warm_up(self):
        self.embed_image(Image.new("RGB", (32, 32)))
        self.embed_text("a photo")


class PerceptionEncoderEmbedder:
    name = "pe"
    dim = 1024
    # PE's text->image cosines run lower than CLIP's (Assembly101 crops:
    # matching ~0.22, other ~0.17); 0.19 keeps most true matches.
    text_min_sim = 0.19
    MODEL = "PE-Core-L-14-336"

    def __init__(self, model=None, device=None, cache_dir=None):
        self.model_name = model or self.MODEL
        self.device = device
        self.cache_dir = cache_dir
        self._lock = threading.Lock()
        self._m = None

    def _load(self):
        if self._m is None:
            import open_clip
            import torch
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained="meta", device=device, cache_dir=self.cache_dir)
            model.eval()
            if device == "cuda":
                model = model.half()
            self._m = (model, preprocess, open_clip.get_tokenizer(self.model_name), device)
        return self._m

    def embed_images(self, images):
        if not images:
            return []
        import torch
        with self._lock:
            model, preprocess, _, device = self._load()
            x = torch.stack([preprocess(i.convert("RGB")) for i in images]).to(device)
            with torch.no_grad():
                v = model.encode_image(x.half() if device == "cuda" else x)
            return [_unit(r) for r in v.float().cpu().numpy()]

    def embed_image(self, image):
        return self.embed_images([image])[0]

    @lru_cache(maxsize=512)
    def _text_cached(self, text):
        import torch
        with self._lock:
            model, _, tokenizer, device = self._load()
            with torch.no_grad():
                v = model.encode_text(tokenizer([text]).to(device))
            return tuple(_unit(v.float().cpu().numpy()[0]))

    def embed_text(self, text):
        return list(self._text_cached(text))

    def warm_up(self):
        self.embed_image(Image.new("RGB", (32, 32)))
        self.embed_text("a photo")


class HashEmbedder:
    """Colour-histogram image vectors and bag-of-words text vectors.

    Not semantic across modalities; good enough to test identity matching
    (same object crop -> same vector) without downloading models.
    """
    name = "hash"
    dim = 64

    def embed_image(self, image):
        small = np.asarray(image.convert("RGB").resize((16, 16)), dtype=np.float32) / 255.0
        hist = []
        for c in range(3):
            h, _ = np.histogram(small[..., c], bins=16, range=(0, 1))
            hist.extend(h)
        hist.extend(small.mean(axis=(0, 1)).tolist() * 5 + [0.0])
        return _unit(np.asarray(hist[: self.dim], dtype=np.float32) + 1e-3)

    def embed_images(self, images):
        return [self.embed_image(i) for i in images]

    def embed_text(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        return _unit(v + 1e-3)

    def warm_up(self):
        pass
