"""
database/db_manager.py — SQLite persistence for Document RAG Q&A sessions.

This database saves:
- document Q&A sessions
- user questions
- assistant answers
- optional retrieved context for debugging
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


DB_PATH = Path("data/document_qa_sessions.db")


def get_connection() -> sqlite3.Connection:
    """Creates a SQLite connection and ensures the data folder exists."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    """Creates the database tables if they do not already exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS qa_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_name TEXT NOT NULL,
            chunk_count INTEGER NOT NULL,
            model_name TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS qa_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            retrieved_context TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES qa_sessions (id)
        )
        """
    )

    conn.commit()
    conn.close()


def create_qa_session(
    *,
    document_name: str,
    chunk_count: int,
    model_name: str,
) -> int:
    """Creates a new saved Q&A session and returns its session ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO qa_sessions (
            document_name,
            chunk_count,
            model_name,
            created_at
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            document_name,
            chunk_count,
            model_name,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )

    session_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return int(session_id)


def save_qa_message(
    *,
    session_id: int,
    role: str,
    content: str,
    retrieved_context: str = "",
) -> None:
    """Saves one user or assistant message for a Q&A session."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO qa_messages (
            session_id,
            role,
            content,
            retrieved_context,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            session_id,
            role,
            content,
            retrieved_context,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )

    conn.commit()
    conn.close()


def get_recent_sessions(limit: int = 10) -> list[tuple[Any, ...]]:
    """Returns recent saved Q&A sessions for the sidebar."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            qa_sessions.id,
            qa_sessions.document_name,
            qa_sessions.chunk_count,
            qa_sessions.model_name,
            qa_sessions.created_at,
            COUNT(qa_messages.id) AS message_count
        FROM qa_sessions
        LEFT JOIN qa_messages
            ON qa_sessions.id = qa_messages.session_id
        GROUP BY qa_sessions.id
        ORDER BY qa_sessions.created_at DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cursor.fetchall()
    conn.close()

    return rows


def get_messages_for_session(session_id: int) -> list[dict[str, str]]:
    """Loads all chat messages for a saved Q&A session."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT role, content
        FROM qa_messages
        WHERE session_id = ?
        ORDER BY id ASC
        """,
        (session_id,),
    )

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "role": row[0],
            "content": row[1],
        }
        for row in rows
    ]


def get_session_by_id(session_id: int) -> dict[str, Any] | None:
    """Returns one saved Q&A session by ID."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            document_name,
            chunk_count,
            model_name,
            created_at
        FROM qa_sessions
        WHERE id = ?
        """,
        (session_id,),
    )

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "id": row[0],
        "document_name": row[1],
        "chunk_count": row[2],
        "model_name": row[3],
        "created_at": row[4],
    }

def clear_chat_logs() -> None:
    """Deletes all saved question and answer records from the SQLite database."""
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM qa_messages")
        cursor.execute("DELETE FROM qa_sessions")
        connection.commit()
