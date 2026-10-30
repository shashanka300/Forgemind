"""The hand-written held-out test turns (defined in laya_zero_shot.py), without loading Laya."""
from pathlib import Path

_src = (Path(__file__).with_name("laya_zero_shot.py")).read_text(encoding="utf-8").split("agent = laya.load")[0]
_src = "\n".join(l for l in _src.splitlines() if not l.startswith(("import laya", "from forgemind")))
_ns = {"SPECS": {}, "AGENT_HIDDEN": set()}
exec(_src, _ns)
DATA, AMBIG, COMPOUND = _ns["DATA"], _ns["AMBIG"], _ns["COMPOUND"]
OBJECTS, OBJ_TESTS, SHORT = _ns["OBJECTS"], _ns["OBJ_TESTS"], _ns["SHORT"]


def held_out():
    return [u for us in DATA.values() for u in us] + [u for u, _ in AMBIG] + COMPOUND + [u for u, _ in OBJ_TESTS]
