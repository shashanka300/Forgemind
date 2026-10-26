"""SQLite world memory: an append-only event log plus the tables derived from it.

Every fact carries provenance: `source` is one of
    observed    - seen by a camera (which one is in `camera`)
    stated      - the user said it
    instrument  - read from an instrument display (close-up frame)
    inferred    - concluded by a model
so reports and answers can say where each fact came from.
"""
import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY, ts REAL NOT NULL, type TEXT NOT NULL,
  source TEXT NOT NULL, camera TEXT, zone TEXT, entity_id INTEGER,
  session_id INTEGER, payload TEXT);
CREATE INDEX IF NOT EXISTS ev_ts ON events(ts);
CREATE INDEX IF NOT EXISTS ev_entity ON events(entity_id);

CREATE TABLE IF NOT EXISTS entities(
  id INTEGER PRIMARY KEY, concept TEXT NOT NULL, name TEXT,
  created_at REAL, last_seen REAL, last_camera TEXT, last_zone TEXT,
  last_region TEXT, last_box TEXT, last_keyframe TEXT, last_score REAL,
  sightings INTEGER DEFAULT 0, proto BLOB, taught INTEGER DEFAULT 0,
  deleted INTEGER DEFAULT 0);

CREATE TABLE IF NOT EXISTS entity_refs(
  id INTEGER PRIMARY KEY, entity_id INTEGER NOT NULL, vec BLOB NOT NULL,
  keyframe TEXT, created_at REAL);

CREATE TABLE IF NOT EXISTS sightings(
  id INTEGER PRIMARY KEY, entity_id INTEGER NOT NULL, ts REAL NOT NULL,
  camera TEXT, zone TEXT, region TEXT, box TEXT, score REAL, sim REAL,
  keyframe TEXT);
CREATE INDEX IF NOT EXISTS si_entity ON sightings(entity_id, ts);

CREATE TABLE IF NOT EXISTS sessions(
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, objective TEXT,
  started_at REAL, ended_at REAL, status TEXT DEFAULT 'active');

CREATE TABLE IF NOT EXISTS steps(
  id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, ord INTEGER,
  ts REAL, text TEXT NOT NULL, status TEXT NOT NULL, source TEXT,
  event_id INTEGER, keyframe TEXT);

CREATE TABLE IF NOT EXISTS measurements(
  id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, ts REAL,
  quantity TEXT NOT NULL, value REAL, unit TEXT, source TEXT NOT NULL,
  event_id INTEGER, keyframe TEXT, note TEXT);

CREATE TABLE IF NOT EXISTS configs(
  id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, ts REAL,
  key TEXT NOT NULL, value TEXT, event_id INTEGER);

CREATE TABLE IF NOT EXISTS watches(
  id INTEGER PRIMARY KEY, kind TEXT NOT NULL, concept TEXT NOT NULL,
  seconds REAL, active INTEGER DEFAULT 1, created_at REAL, last_fired REAL);

CREATE TABLE IF NOT EXISTS conversation(
  id INTEGER PRIMARY KEY, ts REAL, role TEXT, content TEXT);
"""


def vec_to_blob(v):
    return np.asarray(v, dtype=np.float32).tobytes()


def blob_to_vec(b):
    return np.frombuffer(b, dtype=np.float32) if b else None


class DB:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.lock = threading.RLock()
        with self.lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def execute(self, sql, params=()):
        with self.lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur.lastrowid

    def all(self, sql, params=()):
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def one(self, sql, params=()):
        rows = self.all(sql, params)
        return rows[0] if rows else None

    def log_event(self, type, source, payload=None, camera=None, zone=None,
                  entity_id=None, session_id=None, ts=None):
        return self.execute(
            "INSERT INTO events(ts,type,source,camera,zone,entity_id,session_id,payload)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (ts or time.time(), type, source, camera, zone, entity_id, session_id,
             json.dumps(payload or {})))

    def recent_events(self, limit=30, since=None, types=None):
        sql, params = "SELECT * FROM events", []
        where = []
        if since:
            where.append("ts >= ?")
            params.append(since)
        if types:
            where.append(f"type IN ({','.join('?' * len(types))})")
            params.extend(types)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts DESC, id DESC LIMIT ?"
        params.append(limit)
        rows = self.all(sql, params)
        for r in rows:
            r["payload"] = json.loads(r["payload"] or "{}")
        return rows

    def close(self):
        with self.lock:
            self.conn.close()
