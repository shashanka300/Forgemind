"""World memory facade: SQLite event store + visual index + registry + experiments."""
from forgemind.memory.db import DB
from forgemind.memory.experiments import Experiments
from forgemind.memory.notes import Notes
from forgemind.memory.registry import Registry
from forgemind.memory.vectors import build_index


class WorldMemory:
    def __init__(self, settings, embedder):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        (settings.data_dir / "keyframes").mkdir(exist_ok=True)
        (settings.data_dir / "reports").mkdir(exist_ok=True)
        self.settings = settings
        self.db = DB(settings.data_dir / "forgemind.sqlite")
        self.index = build_index(settings, embedder.dim)
        self.registry = Registry(self.db, self.index, embedder, settings)
        self.experiments = Experiments(self.db)
        self.notes = Notes(self.db)

    @property
    def keyframes_dir(self):
        return self.settings.data_dir / "keyframes"

    @property
    def reports_dir(self):
        return self.settings.data_dir / "reports"

    KEYFRAME_REFS = ("SELECT last_keyframe FROM entities", "SELECT keyframe FROM sightings",
                     "SELECT keyframe FROM entity_refs", "SELECT keyframe FROM steps",
                     "SELECT keyframe FROM measurements")

    def prune_keyframes(self, older_than_s=600, now=None):
        """Delete keyframes nothing refers to (the bench camera saves one per
        processed frame). Recent ones stay: the phone may still show them."""
        import time
        now = now or time.time()
        keep = set()
        for q in self.KEYFRAME_REFS:
            keep |= {v for r in self.db.all(q) for v in r.values() if v}
        removed = 0
        for p in self.keyframes_dir.glob("*.jpg"):
            try:
                if p.name not in keep and now - p.stat().st_mtime > older_than_s:
                    p.unlink()
                    removed += 1
            except OSError:
                pass
        return removed

    def close(self):
        self.index.close()
        self.db.close()
