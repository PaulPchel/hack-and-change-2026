"""Хранилище на SQLite: сессии, сообщения, заявки. Только стандартная библиотека."""

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

DB_PATH = os.getenv("DB_PATH", "storage/app.db")
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    has_lead INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    elapsed_ms INTEGER,
    sources TEXT
);
CREATE TABLE IF NOT EXISTS leads (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id),
    name TEXT, phone TEXT, company TEXT, summary TEXT,
    created_at TEXT NOT NULL
);
"""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run(sql, args=(), fetch=False):
    """Одна короткая транзакция на запрос: просто и безопасно для потоков."""
    with _lock:
        os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(sql, args)
            rows = [dict(r) for r in cur.fetchall()] if fetch else None
            conn.commit()
            return rows
        finally:
            conn.close()


def init():
    with _lock:
        os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.executescript(SCHEMA)
        conn.commit()
        conn.close()


def new_session():
    sid = uuid.uuid4().hex[:12]
    _run("INSERT INTO sessions (id, created_at) VALUES (?, ?)", (sid, _now()))
    return sid


def session_exists(sid):
    return bool(_run("SELECT 1 FROM sessions WHERE id = ?", (sid,), fetch=True))


def add_message(sid, role, content, elapsed_ms=None, sources=None):
    _run(
        "INSERT INTO messages (session_id, role, content, created_at, elapsed_ms, sources) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (sid, role, content, _now(), elapsed_ms, json.dumps(sources or [], ensure_ascii=False)),
    )


def get_messages(sid):
    rows = _run(
        "SELECT role, content, created_at, elapsed_ms, sources FROM messages "
        "WHERE session_id = ? ORDER BY id",
        (sid,),
        fetch=True,
    )
    for r in rows:
        r["sources"] = json.loads(r["sources"] or "[]")
    return rows


def list_sessions():
    return _run(
        "SELECT s.id, s.created_at, s.has_lead, "
        "(SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS messages, "
        "(SELECT content FROM messages m WHERE m.session_id = s.id AND m.role = 'user' "
        " ORDER BY m.id LIMIT 1) AS first_message "
        "FROM sessions s ORDER BY s.created_at DESC, s.rowid DESC",
        fetch=True,
    )


def save_lead(sid, lead):
    """Одна заявка на диалог: новая версия заменяет предыдущую."""
    _run(
        "INSERT OR REPLACE INTO leads (session_id, name, phone, company, summary, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (sid, lead.get("name"), lead.get("phone"), lead.get("company"), lead.get("summary"), _now()),
    )
    _run("UPDATE sessions SET has_lead = 1 WHERE id = ?", (sid,))


def list_leads():
    return _run("SELECT * FROM leads ORDER BY created_at DESC", fetch=True)
