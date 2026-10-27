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

    def close(self):
        self.index.close()
        self.db.close()
