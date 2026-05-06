"""SQLite persistence: hash postings + embedding blobs."""

import logging
import sqlite3
from pathlib import Path

import numpy as np

from audio_id.config import EMBEDDING_DIM

logger = logging.getLogger(__name__)


class FeatureStore:
    """SQLite-backed store for fingerprint postings and embedding vectors."""

    def __init__(self, db_path: Path) -> None:
        """Initialise the database and create tables if they do not exist."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()

    def _create_tables(self) -> None:
        """Create the songs, postings, and embeddings tables."""
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS songs (
                song_id TEXT PRIMARY KEY,
                genre TEXT NOT NULL,
                path TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS postings (
                hash_bucket INTEGER NOT NULL,
                song_id TEXT NOT NULL,
                time_offset REAL NOT NULL
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_postings_hash
            ON postings(hash_bucket)
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                song_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL
            )
        """)
        self._conn.commit()

    def upsert_song(self, song_id: str, genre: str, path: str) -> None:
        """Insert or replace a song record."""
        self._conn.execute(
            "INSERT OR REPLACE INTO songs (song_id, genre, path) VALUES (?, ?, ?)",
            (song_id, genre, path),
        )
        self._conn.commit()

    def insert_postings(self, song_id: str, pairs: list[tuple[int, float]]) -> None:
        """Batch insert fingerprint postings for a song."""
        # Delete existing postings for this song to allow re-indexing
        self._conn.execute("DELETE FROM postings WHERE song_id = ?", (song_id,))
        self._conn.executemany(
            "INSERT INTO postings (hash_bucket, song_id, time_offset) VALUES (?, ?, ?)",
            [(h, song_id, t) for h, t in pairs],
        )
        self._conn.commit()

    def insert_embedding(self, song_id: str, vector: np.ndarray) -> None:
        """Store an embedding vector as a binary blob."""
        blob = vector.astype(np.float32).tobytes()
        self._conn.execute(
            "INSERT OR REPLACE INTO embeddings (song_id, vector) VALUES (?, ?)",
            (song_id, blob),
        )
        self._conn.commit()

    def load_all_postings(self) -> dict[int, list[tuple[str, float]]]:
        """Load all postings into a dict: {hash_bucket: [(song_id, time_offset), ...]}."""
        cur = self._conn.execute("SELECT hash_bucket, song_id, time_offset FROM postings")
        result: dict[int, list[tuple[str, float]]] = {}
        for hash_bucket, song_id, time_offset in cur:
            result.setdefault(hash_bucket, []).append((song_id, time_offset))
        logger.info("Loaded %d hash buckets from store", len(result))
        return result

    def load_all_embeddings(self) -> tuple[list[str], np.ndarray]:
        """Load all embeddings as (song_ids_list, matrix [N, EMBEDDING_DIM])."""
        cur = self._conn.execute("SELECT song_id, vector FROM embeddings ORDER BY song_id")
        rows = cur.fetchall()
        if not rows:
            return [], np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        song_ids = []
        vectors = []
        for song_id, blob in rows:
            song_ids.append(song_id)
            vec = np.frombuffer(blob, dtype=np.float32).copy()
            vectors.append(vec)
        matrix = np.stack(vectors).astype(np.float32)
        logger.info("Loaded %d embeddings from store", len(song_ids))
        return song_ids, matrix

    def song_count(self) -> int:
        """Return the number of songs in the store."""
        cur = self._conn.execute("SELECT COUNT(*) FROM songs")
        return cur.fetchone()[0]

    def postings_count(self) -> int:
        """Return the total number of postings in the store."""
        cur = self._conn.execute("SELECT COUNT(*) FROM postings")
        return cur.fetchone()[0]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
