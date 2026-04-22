"""
Session + trace persistence for the Streamlit chat UI.

We use a separate SQLite file (.smartfin/chat_ui.db) from the LangGraph
checkpointer (.smartfin/chatbot.db). The checkpointer flips to WAL mode;
keeping our tables in their own file avoids lock / journal-mode conflicts.

  sessions(thread_id TEXT PRIMARY KEY,
           title TEXT,
           created_at REAL,
           last_activity_at REAL)

  traces(thread_id TEXT,
         position INTEGER,
         role TEXT,
         agent TEXT,
         content TEXT,
         checkpoint_id TEXT,
         PRIMARY KEY (thread_id, position))

The `sessions` table is the sidebar's source of truth for "which
conversations exist". The `traces` table is the flat log of chat
bubbles we've rendered (user messages + per-node summaries), keyed by
thread_id and ordered by `position`.

`checkpoint_id` on a user-role row is the graph-state checkpoint id
that existed *before* that user message was submitted. The
edit-and-resend feature uses it to rewind the graph to that point,
invoke with the edited text, and re-stream new agent bubbles.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from app.orchestrator.checkpoints import DATA_DIR

DB_PATH: Path = DATA_DIR / "chat_ui.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    thread_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_activity_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS traces (
    thread_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    role TEXT NOT NULL,
    agent TEXT,
    content TEXT NOT NULL,
    checkpoint_id TEXT,
    extra TEXT,
    PRIMARY KEY (thread_id, position)
);

CREATE INDEX IF NOT EXISTS idx_sessions_last_activity
    ON sessions (last_activity_at DESC);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

def auto_title(first_message: str, max_len: int = 48) -> str:
    text = (first_message or "").strip().replace("\n", " ")
    if not text:
        return "New chat"
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def create_session(title: str | None = None) -> str:
    thread_id = f"ui-{uuid.uuid4().hex[:10]}"
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (thread_id, title, created_at, last_activity_at) "
            "VALUES (?, ?, ?, ?)",
            (thread_id, title or "New chat", now, now),
        )
    return thread_id


def list_sessions() -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT thread_id, title, created_at, last_activity_at "
            "FROM sessions ORDER BY last_activity_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_session(thread_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT thread_id, title, created_at, last_activity_at "
            "FROM sessions WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
    return dict(row) if row else None


def rename_session(thread_id: str, new_title: str) -> None:
    title = (new_title or "").strip() or "Untitled chat"
    with _connect() as conn:
        conn.execute("UPDATE sessions SET title = ? WHERE thread_id = ?", (title, thread_id))


def touch_session(thread_id: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE sessions SET last_activity_at = ? WHERE thread_id = ?",
            (time.time(), thread_id),
        )


def delete_session(thread_id: str) -> None:
    """
    Remove the session's sidebar entry and its trace. LangGraph checkpoints
    for this thread_id remain in the checkpoints tables but are unreachable
    from the UI — acceptable for now (SQLite file stays small enough).
    """
    with _connect() as conn:
        conn.execute("DELETE FROM traces WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM sessions WHERE thread_id = ?", (thread_id,))


# ---------------------------------------------------------------------------
# Trace persistence
# ---------------------------------------------------------------------------

def load_trace(thread_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, agent, content, checkpoint_id, extra "
            "FROM traces WHERE thread_id = ? ORDER BY position ASC",
            (thread_id,),
        ).fetchall()

    trace: list[dict[str, Any]] = []
    for r in rows:
        entry: dict[str, Any] = {"role": r["role"], "content": r["content"]}
        if r["agent"]:
            entry["agent"] = r["agent"]
        if r["checkpoint_id"]:
            entry["checkpoint_id"] = r["checkpoint_id"]
        if r["extra"]:
            try:
                entry.update(json.loads(r["extra"]))
            except json.JSONDecodeError:
                pass
        trace.append(entry)
    return trace


def save_trace(thread_id: str, trace: list[dict[str, Any]]) -> None:
    """Replace the stored trace for this thread_id wholesale."""
    with _connect() as conn:
        conn.execute("DELETE FROM traces WHERE thread_id = ?", (thread_id,))
        rows = []
        for i, entry in enumerate(trace):
            extras = {
                k: v
                for k, v in entry.items()
                if k not in {"role", "agent", "content", "checkpoint_id"}
            }
            rows.append(
                (
                    thread_id,
                    i,
                    entry.get("role", ""),
                    entry.get("agent"),
                    entry.get("content", ""),
                    entry.get("checkpoint_id"),
                    json.dumps(extras) if extras else None,
                )
            )
        if rows:
            conn.executemany(
                "INSERT INTO traces "
                "(thread_id, position, role, agent, content, checkpoint_id, extra) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )


# Initialise on import so callers don't need to remember to.
init_db()
