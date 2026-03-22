"""
database.py
Handles all PostgreSQL database operations: faces, events, visitor counts.
Uses psycopg2 with a connection-pool for thread-safe multi-threaded access.
"""

import pickle
import threading
from contextlib import contextmanager
from datetime import datetime

import psycopg2
import psycopg2.extras
from psycopg2 import pool


class Database:
    """
    Thread-safe PostgreSQL wrapper using a connection pool.
    All queries use %s placeholders (psycopg2 style).
    """

    def __init__(self, dsn: str, min_conn: int = 2, max_conn: int = 10):
        """
        Args:
            dsn: PostgreSQL DSN string, e.g.
                 "host=localhost port=5432 dbname=face_tracker user=postgres password=secret"
        """
        self._pool = pool.ThreadedConnectionPool(min_conn, max_conn, dsn=dsn)
        self._init_db()

    @contextmanager
    def _conn(self):
        """Yield a connection from the pool and return it on exit."""
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def _init_db(self):
        """Create all tables on first run (idempotent)."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS faces (
                        id          TEXT PRIMARY KEY,
                        first_seen  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        last_seen   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        visit_count INTEGER NOT NULL DEFAULT 1,
                        embedding   BYTEA
                    );

                    CREATE TABLE IF NOT EXISTS events (
                        id          SERIAL PRIMARY KEY,
                        face_id     TEXT NOT NULL REFERENCES faces(id),
                        event_type  TEXT NOT NULL,
                        timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        image_path  TEXT
                    );

                    CREATE TABLE IF NOT EXISTS visitor_summary (
                        date            DATE PRIMARY KEY DEFAULT CURRENT_DATE,
                        unique_visitors INTEGER NOT NULL DEFAULT 0,
                        total_entries   INTEGER NOT NULL DEFAULT 0,
                        total_exits     INTEGER NOT NULL DEFAULT 0
                    );

                    CREATE INDEX IF NOT EXISTS idx_events_face_id ON events(face_id);
                    CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);
                """)

    # ── Face operations ──────────────────────────────────────────────────────

    def register_face(self, face_id: str, embedding_bytes: bytes):
        """Insert a new face record (ignore if already present)."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO faces (id, embedding)
                    VALUES (%s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (face_id, psycopg2.Binary(embedding_bytes)),
                )

    def update_face_last_seen(self, face_id: str):
        """Bump last_seen and visit_count for a returning face."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE faces
                    SET last_seen = NOW(), visit_count = visit_count + 1
                    WHERE id = %s
                    """,
                    (face_id,),
                )

    def face_exists(self, face_id: str) -> bool:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM faces WHERE id = %s", (face_id,))
                return cur.fetchone() is not None

    def get_all_faces(self):
        """Return all face rows as list of dicts."""
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM faces")
                return cur.fetchall()

    # ── Event operations ─────────────────────────────────────────────────────

    def log_event(self, face_id: str, event_type: str, image_path: str = None):
        """Record an entry or exit event and update daily summary."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO events (face_id, event_type, image_path)
                    VALUES (%s, %s, %s)
                    """,
                    (face_id, event_type, image_path),
                )
                # Upsert daily summary
                cur.execute(
                    """
                    INSERT INTO visitor_summary (date) VALUES (CURRENT_DATE)
                    ON CONFLICT (date) DO NOTHING
                    """,
                )
                if event_type == "entry":
                    cur.execute(
                        "UPDATE visitor_summary SET total_entries = total_entries + 1 WHERE date = CURRENT_DATE"
                    )
                elif event_type == "exit":
                    cur.execute(
                        "UPDATE visitor_summary SET total_exits = total_exits + 1 WHERE date = CURRENT_DATE"
                    )

    def get_events_for_face(self, face_id: str):
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM events WHERE face_id = %s ORDER BY timestamp",
                    (face_id,),
                )
                return cur.fetchall()

    def get_recent_events(self, limit: int = 50):
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM events ORDER BY timestamp DESC LIMIT %s", (limit,)
                )
                return cur.fetchall()

    # ── Visitor summary ───────────────────────────────────────────────────────

    def increment_unique_visitor(self):
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO visitor_summary (date, unique_visitors)
                    VALUES (CURRENT_DATE, 1)
                    ON CONFLICT (date) DO UPDATE
                        SET unique_visitors = visitor_summary.unique_visitors + 1
                    """
                )

    def get_unique_visitor_count(self):
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(DISTINCT face_id) FROM events;")
                    return cur.fetchone()[0]
        except Exception:
            return 0
        return 0  # fallback if DB closed

    def get_summary(self):
        with self._conn() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM visitor_summary WHERE date = CURRENT_DATE"
                )
                result = cur.fetchone()
                if result:
                    return [result]
                # Return default if no record exists for today
                return [{"date": "today", "unique_visitors": 0, "total_entries": 0, "total_exits": 0}]

    def close(self):
        self._pool.closeall()
