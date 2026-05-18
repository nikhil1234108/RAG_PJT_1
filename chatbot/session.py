"""
session.py
PostgreSQL-backed session management.
Stores full conversation history per session_id.
Enables GPT-like memory across multiple turns and page reloads.
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgresql+psycopg://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+psycopg://", "postgresql://")

def get_conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def init_db():
    """
    Creates sessions and messages tables if not exist.
    Call once at startup.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id   TEXT PRIMARY KEY,
                    created_at   TIMESTAMP DEFAULT NOW(),
                    updated_at   TIMESTAMP DEFAULT NOW(),
                    mode         TEXT DEFAULT 'text',
                    metadata     JSONB DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id           SERIAL PRIMARY KEY,
                    session_id   TEXT REFERENCES sessions(session_id),
                    role         TEXT NOT NULL,
                    content      TEXT NOT NULL,
                    timestamp    TIMESTAMP DEFAULT NOW(),
                    metadata     JSONB DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id);
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp
                    ON messages(timestamp);
            """)
        conn.commit()
    print("Database tables initialized.")


def create_session(mode: str = "text") -> str:
    """Creates a new session and returns session_id."""
    session_id = str(uuid.uuid4())
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (session_id, mode) VALUES (%s, %s)",
                (session_id, mode)
            )
        conn.commit()
    return session_id


def get_session(session_id: str) -> Optional[Dict]:
    """Returns session metadata or None if not found."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM sessions WHERE session_id = %s",
                (session_id,)
            )
            return cur.fetchone()


def save_message(session_id: str, role: str,
                 content: str, metadata: dict = None):
    """
    Saves a message to PostgreSQL.
    role: 'user' | 'assistant' | 'system'
    metadata: optional dict (e.g. sources, voice_duration)
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO messages (session_id, role, content, metadata)
                   VALUES (%s, %s, %s, %s)""",
                (session_id, role, content,
                 json.dumps(metadata or {}))
            )
            cur.execute(
                "UPDATE sessions SET updated_at = NOW() WHERE session_id = %s",
                (session_id,)
            )
        conn.commit()


def get_history(session_id: str,
                last_n: int = 10) -> List[Dict[str, str]]:
    """
    Returns last N messages for a session.
    Format: [{"role": "user", "content": "..."}, ...]
    Used for memory injection into system prompt.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT role, content FROM messages
                   WHERE session_id = %s
                   ORDER BY timestamp DESC
                   LIMIT %s""",
                (session_id, last_n)
            )
            rows = cur.fetchall()
    return list(reversed(rows))


def format_memory_context(history: List[Dict[str, str]]) -> str:
    """
    Formats conversation history for injection into system prompt.
    Keeps last N turns visible to the LLM as memory context.
    """
    if not history:
        return "No previous conversation."

    lines = []
    for msg in history:
        role    = "User" if msg["role"] == "user" else "BlackBot"
        content = msg["content"][:300]   # truncate long messages
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def delete_session(session_id: str):
    """Deletes session and all its messages."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM messages WHERE session_id = %s", (session_id,)
            )
            cur.execute(
                "DELETE FROM sessions WHERE session_id = %s", (session_id,)
            )
        conn.commit()