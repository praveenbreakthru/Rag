"""
SQLite persistent chat history manager for Sakthi Finance RAG Assistant.
Stores conversations, messages, timestamps, and citation metadata permanently.
Survives application restarts and system reboots.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Stable absolute path to chat_history.db inside the UI/data folder
DB_PATH = Path(__file__).resolve().parent / "data" / "chat_history.db"


def get_db_connection() -> sqlite3.Connection:
    """Connect to SQLite database with foreign keys enabled."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_chat_history_db() -> None:
    """Initialize SQLite tables and indexes if they do not exist."""
    with get_db_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                citations_json TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);
        """)


def create_conversation(title: str = "New Financial Inquiry") -> str:
    """Create a new conversation record and return its UUID."""
    conv_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?);",
            (conv_id, title[:80], now, now),
        )
    return conv_id


def list_conversations() -> list[dict[str, Any]]:
    """List all saved conversations ordered by last updated descending."""
    init_chat_history_db()
    with get_db_connection() as conn:
        cursor = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC;"
        )
        return [dict(row) for row in cursor.fetchall()]


def get_conversation_messages(conversation_id: str) -> list[dict[str, Any]]:
    """Retrieve all messages for a conversation in chronological order."""
    init_chat_history_db()
    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            SELECT id, role, content, timestamp, citations_json
            FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp ASC;
            """,
            (conversation_id,),
        )
        messages = []
        for row in cursor.fetchall():
            m = dict(row)
            m["citations"] = json.loads(m["citations_json"]) if m.get("citations_json") else []
            messages.append(m)
        return messages


def add_message(
    conversation_id: str,
    role: str,
    content: str,
    citations: Optional[list[dict[str, Any]]] = None,
) -> str:
    """Add a message to SQLite and update the conversation timestamp."""
    init_chat_history_db()
    msg_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    cit_json = json.dumps(citations) if citations else None

    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO messages (id, conversation_id, role, content, timestamp, citations_json)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (msg_id, conversation_id, role, content, now, cit_json),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?;",
            (now, conversation_id),
        )
    return msg_id


def update_conversation_title(conversation_id: str, title: str) -> None:
    """Update title for a conversation (e.g. from the first user question)."""
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?;",
            (title[:80], conversation_id),
        )


def delete_conversation(conversation_id: str) -> None:
    """
    Delete a conversation and its messages.
    NOTE: NEVER touches documents, chunks, vectors, or knowledge bases.
    """
    with get_db_connection() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?;", (conversation_id,))
