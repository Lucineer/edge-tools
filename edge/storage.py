"""
edge/storage.py — Persistent storage for the edge gateway.

SQLite-backed conversation history, usage tracking, and model metadata.
Thread-safe via connection-per-call pattern (SQLite handles this with WAL).

Usage:
    from edge.storage import EdgeStore
    db = EdgeStore()  # Uses default path
    conv_id = db.create_conversation(model="deepseek-r1:1.5b")
    db.add_message(conv_id, "user", "Hello")
    db.add_message(conv_id, "assistant", "Hi there!")
    history = db.get_conversation(conv_id)
    db.list_conversations(limit=10)
"""

import json
import os
import sqlite3
import time
from datetime import datetime

from .config import DATA_DIR

DEFAULT_DB_PATH = os.path.join(DATA_DIR, "edge-store.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    message_count INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('system', 'user', 'assistant', 'tool')),
    content TEXT NOT NULL,
    tokens_prompt INTEGER,
    tokens_completion INTEGER,
    created_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}',
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at);

CREATE TABLE IF NOT EXISTS usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    tokens_prompt INTEGER DEFAULT 0,
    tokens_completion INTEGER DEFAULT 0,
    latency_ms REAL,
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_usage_created ON usage_log(created_at);
"""


class EdgeStore:
    """SQLite-backed persistent storage for edge gateway."""

    def __init__(self, path=None):
        self.path = path or DEFAULT_DB_PATH
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._init_db()

    def _connect(self):
        """Get a new connection (SQLite thread-safe pattern)."""
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = self._connect()
        conn.executescript(_SCHEMA)
        conn.commit()
        conn.close()

    # ── Conversations ───────────────────────────────────────────

    def create_conversation(self, model="unknown", title=None, metadata=None):
        """Create a new conversation. Returns conversation ID."""
        import uuid
        conv_id = uuid.uuid4().hex[:12]
        now = datetime.now().isoformat()
        conn = self._connect()
        conn.execute(
            "INSERT INTO conversations (id, model, title, created_at, updated_at, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (conv_id, model, title, now, now, json.dumps(metadata or {}))
        )
        conn.commit()
        conn.close()
        return conv_id

    def get_conversation(self, conv_id):
        """Get a conversation with all messages."""
        conn = self._connect()
        row = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        if not row:
            conn.close()
            return None

        messages = conn.execute(
            "SELECT role, content, tokens_prompt, tokens_completion, created_at FROM messages WHERE conversation_id = ? ORDER BY id",
            (conv_id,)
        ).fetchall()
        conn.close()

        return {
            "id": row["id"],
            "model": row["model"],
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "message_count": row["message_count"],
            "messages": [
                {
                    "role": m["role"],
                    "content": m["content"],
                    "tokens_prompt": m["tokens_prompt"],
                    "tokens_completion": m["tokens_completion"],
                }
                for m in messages
            ],
        }

    def add_message(self, conv_id, role, content, tokens_prompt=0, tokens_completion=0):
        """Add a message to a conversation."""
        now = datetime.now().isoformat()
        conn = self._connect()
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, tokens_prompt, tokens_completion, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (conv_id, role, content, tokens_prompt, tokens_completion, now)
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ?, message_count = message_count + 1 WHERE id = ?",
            (now, conv_id)
        )
        conn.commit()
        conn.close()

    def list_conversations(self, limit=20, offset=0):
        """List conversations, most recent first."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT id, model, title, created_at, updated_at, message_count FROM conversations ORDER BY updated_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def delete_conversation(self, conv_id):
        """Delete a conversation and all its messages."""
        conn = self._connect()
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        conn.commit()
        conn.close()

    def search_conversations(self, query, limit=10):
        """Search conversation messages by content."""
        conn = self._connect()
        rows = conn.execute(
            """SELECT DISTINCT c.id, c.model, c.title, c.updated_at, c.message_count
               FROM conversations c
               JOIN messages m ON m.conversation_id = c.id
               WHERE m.content LIKE ?
               ORDER BY c.updated_at DESC LIMIT ?""",
            (f"%{query}%", limit)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ── Usage Tracking ──────────────────────────────────────────

    def log_usage(self, model, endpoint, tokens_prompt=0, tokens_completion=0,
                  latency_ms=0, error=None):
        """Log an API usage event."""
        now = datetime.now().isoformat()
        conn = self._connect()
        conn.execute(
            "INSERT INTO usage_log (model, endpoint, tokens_prompt, tokens_completion, latency_ms, error, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (model, endpoint, tokens_prompt, tokens_completion, latency_ms, error, now)
        )
        conn.commit()
        conn.close()

    def get_usage_stats(self, since=None):
        """Get aggregated usage stats."""
        conn = self._connect()
        if since:
            rows = conn.execute(
                """SELECT model, endpoint,
                          COUNT(*) as requests,
                          SUM(tokens_prompt) as total_prompt_tokens,
                          SUM(tokens_completion) as total_completion_tokens,
                          AVG(latency_ms) as avg_latency_ms,
                          SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as errors
                   FROM usage_log WHERE created_at >= ?
                   GROUP BY model, endpoint""",
                (since,)
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT model, endpoint,
                          COUNT(*) as requests,
                          SUM(tokens_prompt) as total_prompt_tokens,
                          SUM(tokens_completion) as total_completion_tokens,
                          AVG(latency_ms) as avg_latency_ms,
                          SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) as errors
                   FROM usage_log
                   GROUP BY model, endpoint"""
            ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ── Maintenance ─────────────────────────────────────────────

    def vacuum(self):
        """Vacuum the database to reclaim space."""
        conn = self._connect()
        conn.execute("VACUUM")
        conn.close()

    def get_db_size(self):
        """Get database file size in bytes."""
        if os.path.exists(self.path):
            return os.path.getsize(self.path)
        return 0
