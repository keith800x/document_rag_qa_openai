"""
database/db_manager.py — SQLite persistence for Document RAG Q&A sessions.

This stores:
- Q&A sessions
- One uploaded document per session
- User/assistant chat messages
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


def now_iso() -> str:
    """Returns the current timestamp as a readable ISO string."""
    return datetime.now().isoformat(timespec="seconds")


def init_db() -> None:
    """Creates all required database tables if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS qa_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS qa_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL UNIQUE,
            original_filename TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            chunk_count INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES qa_sessions (id)
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


def create_session(title: str = "New Session") -> int:
    """Creates a new Q&A session and returns the session ID."""
    conn = get_connection()
    cursor = conn.cursor()

    created_at = now_iso()

    cursor.execute(
        """
        INSERT INTO qa_sessions (title, created_at, updated_at)
        VALUES (?, ?, ?)
        """,
        (title, created_at, created_at),
    )

    session_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return int(session_id)


def update_session_title(session_id: int, title: str) -> None:
    """Updates the title of a saved Q&A session."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE qa_sessions
        SET title = ?, updated_at = ?
        WHERE id = ?
        """,
        (title, now_iso(), session_id),
    )

    conn.commit()
    conn.close()


def touch_session(session_id: int) -> None:
    """Updates a session's modified timestamp."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE qa_sessions
        SET updated_at = ?
        WHERE id = ?
        """,
        (now_iso(), session_id),
    )

    conn.commit()
    conn.close()


def save_session_document(
    *,
    session_id: int,
    original_filename: str,
    stored_path: str,
    chunk_count: int,
) -> None:
    """Saves or replaces the document attached to a session."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO qa_documents (
            session_id,
            original_filename,
            stored_path,
            chunk_count,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            original_filename = excluded.original_filename,
            stored_path = excluded.stored_path,
            chunk_count = excluded.chunk_count,
            created_at = excluded.created_at
        """,
        (
            session_id,
            original_filename,
            stored_path,
            chunk_count,
            now_iso(),
        ),
    )

    conn.commit()
    conn.close()

    update_session_title(session_id, original_filename)


def get_document_for_session(session_id: int) -> dict[str, Any] | None:
    """Returns the document attached to a session, if one exists."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT original_filename, stored_path, chunk_count
        FROM qa_documents
        WHERE session_id = ?
        """,
        (session_id,),
    )

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "original_filename": row[0],
        "stored_path": row[1],
        "chunk_count": row[2],
    }


def save_message(
    *,
    session_id: int,
    role: str,
    content: str,
    retrieved_context: str = "",
) -> None:
    """Saves one user or assistant message."""
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
            now_iso(),
        ),
    )

    conn.commit()
    conn.close()

    touch_session(session_id)


def get_messages_for_session(session_id: int) -> list[dict[str, str]]:
    """Loads all chat messages for a session."""
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

    return [{"role": row[0], "content": row[1]} for row in rows]


def get_recent_sessions(limit: int = 10) -> list[tuple[Any, ...]]:
    """Returns recent sessions for the sidebar."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            qa_sessions.id,
            qa_sessions.title,
            qa_sessions.updated_at,
            COALESCE(qa_documents.original_filename, 'No document'),
            COALESCE(qa_documents.chunk_count, 0),
            COUNT(qa_messages.id) AS message_count
        FROM qa_sessions
        LEFT JOIN qa_documents
            ON qa_sessions.id = qa_documents.session_id
        LEFT JOIN qa_messages
            ON qa_sessions.id = qa_messages.session_id
        GROUP BY qa_sessions.id
        ORDER BY qa_sessions.updated_at DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cursor.fetchall()
    conn.close()

    return rows


def delete_session(session_id: int) -> None:
    """Deletes one session and its saved messages/document record."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM qa_messages WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM qa_documents WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM qa_sessions WHERE id = ?", (session_id,))

    conn.commit()
    conn.close()