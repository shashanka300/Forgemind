"""User notes memory (PRD F-EXP-9): goals, obstacles and preferences the user
states, kept across sessions and shown to the agent.

Lesson from the personal health agent paper (PRD §7.4): remembering what the
user wants and what got in their way, not just the facts, kept multi-session
conversations coherent. Extraction runs after the reply (no added latency)
and only on turns that look like they contain such a statement.
"""
import json
import re
import threading
import time

KINDS = ("goal", "obstacle", "preference", "fact")
# Cheap gate: only these turns go to the worker model.
SIGNALS = re.compile(
    r"\b(i want|i'd like|i need|need to|my goal|trying to|aim|today i|by (tomorrow|tonight|friday)|"
    r"problem|issue|keeps?|stuck|can'?t|doesn'?t work|broke|annoying|"
    r"i (always|usually|never|prefer)|prefer|i keep|i store|i put|don'?t (remind|tell|ask)|"
    r"remember that|note that|fyi)\b", re.I)

EXTRACT_PROMPT = """From the user's message, extract lasting notes about the user's work:
goals (what they want to achieve), obstacles (problems they hit), preferences
(how they like to work, where they keep things, what not to do) and facts about
their setup. Ignore requests, questions and anything already logged as a step
or measurement. Answer JSON only: {"notes": [{"kind": "goal|obstacle|preference|fact",
"text": "short third-person note, e.g. 'keeps hex keys in the drawer'"}]}.
Return {"notes": []} if there is nothing lasting."""


class Notes:
    def __init__(self, db):
        self.db = db
        db.execute("CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, ts REAL, kind TEXT,"
                   " text TEXT NOT NULL, source_turn TEXT, session_id INTEGER, active INTEGER DEFAULT 1)")
        self._lock = threading.Lock()

    def add(self, kind, text, source_turn=None, session_id=None, ts=None):
        kind = kind if kind in KINDS else "fact"
        text = " ".join(text.split())[:200]
        if not text:
            return None
        with self._lock:
            same = self.db.one("SELECT id FROM notes WHERE active=1 AND lower(text)=lower(?)", (text,))
            if same:
                return same["id"]
            nid = self.db.execute(
                "INSERT INTO notes(ts,kind,text,source_turn,session_id) VALUES(?,?,?,?,?)",
                (ts or time.time(), kind, text, source_turn, session_id))
        self.db.log_event("note", "stated", {"kind": kind, "text": text}, session_id=session_id)
        return nid

    def recent(self, limit=8):
        return self.db.all("SELECT id, ts, kind, text, session_id FROM notes WHERE active=1"
                           " ORDER BY ts DESC LIMIT ?", (limit,))

    def forget(self, nid):
        self.db.execute("UPDATE notes SET active=0 WHERE id=?", (nid,))

    def extract(self, llm, text, session_id=None):
        """Worker-model extraction for one user turn; returns the stored notes."""
        if not SIGNALS.search(text or "") or getattr(llm, "name", "mock") == "mock":
            return []
        try:
            res = llm.chat([{"role": "system", "content": EXTRACT_PROMPT},
                            {"role": "user", "content": text}],
                           model=llm.model_for("worker"), json_mode=True)
            items = json.loads(res.content or "{}").get("notes", [])
        except Exception:
            return []
        out = []
        for it in items[:4]:
            if isinstance(it, dict) and it.get("text"):
                nid = self.add(it.get("kind", "fact"), it["text"], text, session_id)
                if nid:
                    out.append({"id": nid, "kind": it.get("kind"), "text": it["text"]})
        return out
