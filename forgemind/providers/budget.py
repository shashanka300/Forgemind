"""API spend ledger and hard budget cap (USD 150 for this engagement).

Every Meta Model API call is recorded with its estimated cost in one ledger
per machine (default ~/.forgemind/ledger.sqlite, FM_LEDGER), so runs that use
throwaway data folders (evals, replays) count too. At FM_BUDGET_USD further
API calls raise BudgetExceeded: the fallback model (if configured) takes over,
or the agent says so. Prices: dev.meta.ai pricing page, checked Oct 26, 2026.
"""
import os
import sqlite3
import threading
import time
from pathlib import Path

# USD per 1M tokens: (input, cached input, output)
TOKEN_PRICES = {
    "muse-wonder-1.0": (4.00, 0.20, 20.00),
    "muse-spark-1.3": (1.25, 0.15, 4.25),
    "muse-spark-1.2": (1.25, 0.15, 4.25),
    "muse-spark-1.1": (1.25, 0.15, 4.25),
    "muse-spark-1.3-contributor": (0.10, 0.002, 0.20),
    "muse-spark-1.2-contributor": (0.10, 0.002, 0.20),
}
SAM_IMAGE_USD = 2.50 / 1000
ASR_HOUR_USD = 0.18
WARN_FRACTIONS = (1 / 3, 2 / 3)


class BudgetExceeded(RuntimeError):
    pass


def default_path():
    return Path(os.environ.get("FM_LEDGER") or Path.home() / ".forgemind" / "ledger.sqlite")


def token_cost(model, usage):
    pin, pcached, pout = TOKEN_PRICES.get(model, TOKEN_PRICES["muse-spark-1.3"])
    usage = usage or {}
    prompt = usage.get("prompt_tokens") or 0
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    out = usage.get("completion_tokens") or 0
    return ((prompt - cached) * pin + cached * pcached + out * pout) / 1e6


class Ledger:
    def __init__(self, path=None, cap_usd=140.0, on_warn=None):
        self.path = Path(path or default_path())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cap = float(cap_usd)
        self.on_warn = on_warn
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS spend(id INTEGER PRIMARY KEY, ts REAL, kind TEXT,"
                      " model TEXT, units REAL, usd REAL, note TEXT)")

    def _conn(self):
        return sqlite3.connect(self.path, timeout=10)

    def total(self):
        with self._conn() as c:
            return c.execute("SELECT COALESCE(SUM(usd), 0) FROM spend").fetchone()[0]

    def check(self):
        """Raise before a call once the cap is reached."""
        if self.total() >= self.cap:
            raise BudgetExceeded(f"API budget reached (${self.total():.2f} of ${self.cap:.0f}); "
                                 "not calling the Model API")

    def add(self, kind, model, units, usd, note=None):
        with self._lock:
            before = self.total()
            with self._conn() as c:
                c.execute("INSERT INTO spend(ts,kind,model,units,usd,note) VALUES(?,?,?,?,?,?)",
                          (time.time(), kind, model, units, usd, note))
            after = before + usd
        for f in WARN_FRACTIONS:
            if before < f * self.cap <= after and self.on_warn:
                self.on_warn(f"API spend passed ${f * self.cap:.0f} of the ${self.cap:.0f} budget.")
        return after

    def summary(self):
        with self._conn() as c:
            by_kind = c.execute("SELECT kind, COUNT(*), SUM(usd) FROM spend GROUP BY kind").fetchall()
            by_day = c.execute("SELECT date(ts, 'unixepoch', 'localtime'), SUM(usd) FROM spend"
                               " GROUP BY 1 ORDER BY 1").fetchall()
        return {"total": round(self.total(), 4), "cap": self.cap,
                "by_kind": {k: {"calls": n, "usd": round(u or 0, 4)} for k, n, u in by_kind},
                "by_day": {d: round(u or 0, 4) for d, u in by_day}}


class MeteredLLM:
    """Wraps a Model API reasoning model: refuses at the cap, records every call."""

    def __init__(self, llm, ledger):
        self.llm, self.ledger = llm, ledger
        self.name, self.supports_vision = llm.name, llm.supports_vision

    def model_for(self, role):
        return self.llm.model_for(role)

    def list_models(self):
        return self.llm.list_models()

    def chat(self, messages, tools=None, model=None, json_mode=False):
        self.ledger.check()
        model = model or self.llm.model_for("agent")
        res = self.llm.chat(messages, tools=tools, model=model, json_mode=json_mode)
        u = res.usage or {}
        self.ledger.add("llm", model, (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0),
                        token_cost(model, u))
        return res

    def __getattr__(self, name):            # reasoning_effort etc. on the wrapped model
        return getattr(self.llm, name)
