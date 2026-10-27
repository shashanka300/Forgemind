"""Visual memory index: object crops and keyframes, searchable by text or image.

`QdrantEdgeIndex` is the gallery app's store (one Qdrant Edge shard, one
named cosine vector "image") with ForgeMind payloads: kind, entity_id,
camera, zone, ts, keyframe. `NumpyIndex` is a dependency-free fallback with
the same interface.
"""
import itertools
import json
import threading
import time
from pathlib import Path

import numpy as np


class NumpyIndex:
    name = "numpy"

    def __init__(self, path, dim):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._lock = threading.Lock()
        self.ids, self.vecs, self.payloads = [], np.zeros((0, dim), np.float32), []
        f = self.path / "index.npz"
        if f.exists():
            d = np.load(f, allow_pickle=False)
            self.ids = d["ids"].tolist()
            self.vecs = d["vecs"]
            self.payloads = json.loads((self.path / "payloads.json").read_text(encoding="utf-8"))
        self._next = itertools.count(time.time_ns())
        self._dirty = 0

    def upsert(self, vector, payload):
        pid = next(self._next)
        with self._lock:
            self.ids.append(pid)
            self.vecs = np.vstack([self.vecs, np.asarray(vector, np.float32)[None]])
            self.payloads.append(payload)
            self._dirty += 1
            if self._dirty >= 20:
                self._save()
        return pid

    def _save(self):
        np.savez(self.path / "index.npz", ids=np.asarray(self.ids, np.int64), vecs=self.vecs)
        (self.path / "payloads.json").write_text(json.dumps(self.payloads), encoding="utf-8")
        self._dirty = 0

    def search(self, vector, limit=10, where=None):
        with self._lock:
            if not self.ids:
                return []
            sims = self.vecs @ np.asarray(vector, np.float32)
            order = np.argsort(-sims)
            out = []
            for i in order:
                p = self.payloads[i]
                if where and any(p.get(k) != v for k, v in where.items()):
                    continue
                out.append((self.ids[i], float(sims[i]), p))
                if len(out) >= limit:
                    break
            return out

    def count(self):
        return len(self.ids)

    def close(self):
        with self._lock:
            self._save()


class QdrantEdgeIndex:
    name = "qdrant_edge"

    def __init__(self, path, dim):
        from qdrant_edge import (Distance, EdgeConfig, EdgeShard, EdgeVectorParams,
                                 PayloadSchemaType, UpdateOperation)
        self._q = __import__("qdrant_edge")
        self.dim = dim
        self._lock = threading.RLock()
        path = Path(path)
        if path.exists() and any(path.iterdir()):
            self.shard = EdgeShard.load(str(path))
        else:
            path.mkdir(parents=True, exist_ok=True)
            cfg = EdgeConfig(vectors={"image": EdgeVectorParams(size=dim,
                                                                distance=Distance.Cosine)})
            self.shard = EdgeShard.create(str(path), cfg)
            for key, schema in [("kind", PayloadSchemaType.Keyword),
                                ("camera", PayloadSchemaType.Keyword),
                                ("zone", PayloadSchemaType.Keyword),
                                ("entity_id", PayloadSchemaType.Integer),
                                ("ts", PayloadSchemaType.Float)]:
                self.shard.update(UpdateOperation.create_field_index(key, schema))
        self._ids = itertools.count(time.time_ns())

    def upsert(self, vector, payload):
        q = self._q
        pid = next(self._ids)
        with self._lock:
            self.shard.update(q.UpdateOperation.upsert_points([
                q.Point(id=pid, vector={"image": list(vector)}, payload=payload)]))
            self.shard.flush()
        return pid

    def search(self, vector, limit=10, where=None):
        q = self._q
        flt = None
        if where:
            flt = q.Filter(must=[q.FieldCondition(key=k, match=q.MatchValue(value=v))
                                 for k, v in where.items()])
        with self._lock:
            hits = self.shard.query(q.QueryRequest(
                query=q.Query.Nearest(list(vector), using="image"),
                filter=flt, limit=limit, with_payload=True))
        return [(h.id, float(h.score), dict(h.payload)) for h in hits]

    def count(self):
        with self._lock:
            return self.shard.count(self._q.CountRequest(exact=True))

    def close(self):
        with self._lock:
            self.shard.close()


def build_index(settings, dim):
    path = settings.data_dir / "visual"
    if settings.vector_backend == "qdrant_edge":
        try:
            return QdrantEdgeIndex(path / f"shard-{dim}", dim)
        except ImportError:
            print("qdrant-edge-py not installed; using the numpy index "
                  "(pip install 'forgemind[vision]' for the gallery store)")
    return NumpyIndex(path / f"numpy-{dim}", dim)
