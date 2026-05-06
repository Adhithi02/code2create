"""In-memory index: hash dict + FAISS FlatIP; RLock; rebuild logic."""

import logging
import threading

import faiss
import numpy as np

from audio_id.config import EMBEDDING_DIM
from audio_id.store import FeatureStore

logger = logging.getLogger(__name__)


class AudioIndex:
    """Thread-safe in-memory index for hash lookups and embedding search."""

    def __init__(self) -> None:
        """Initialise an empty index."""
        self._hash_index: dict[int, list[tuple[str, float]]] = {}
        self._faiss_index: faiss.IndexFlatIP = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._song_ids: list[str] = []
        self._lock = threading.RLock()
        self._ready = False
        self._songs_count = 0
        self._postings_count = 0
        self._last_error: str | None = None

    @property
    def ready(self) -> bool:
        """Whether the index is built and ready for queries."""
        return self._ready

    @property
    def songs_count(self) -> int:
        """Number of songs in the index."""
        return self._songs_count

    @property
    def postings_count(self) -> int:
        """Total number of hash postings in the index."""
        return self._postings_count

    @property
    def embeddings_count(self) -> int:
        """Number of embedding vectors in the FAISS index."""
        with self._lock:
            return self._faiss_index.ntotal

    @property
    def last_error(self) -> str | None:
        """Last error message during index build, if any."""
        return self._last_error

    def build(self, store: FeatureStore) -> None:
        """Load postings and embeddings from the store and build in-memory indices."""
        with self._lock:
            try:
                self._ready = False
                logger.info("Building index from store...")

                # Load hash postings
                self._hash_index = store.load_all_postings()
                self._postings_count = sum(len(v) for v in self._hash_index.values())

                # Load embeddings
                song_ids, matrix = store.load_all_embeddings()
                self._song_ids = song_ids

                # Rebuild FAISS index
                self._faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
                if matrix.shape[0] > 0:
                    self._faiss_index.add(matrix.astype(np.float32))

                self._songs_count = store.song_count()
                self._ready = True
                self._last_error = None
                logger.info(
                    "Index built: %d songs, %d postings, %d embeddings",
                    self._songs_count, self._postings_count, self._faiss_index.ntotal,
                )
            except Exception as exc:
                self._last_error = str(exc)
                logger.error("Index build failed: %s", exc)
                raise

    def lookup_hashes(self, hashes: list[tuple[int, float]]) -> dict[str, dict[int, int]]:
        """Look up hash fingerprints and return per-song time-bin histograms."""
        with self._lock:
            result: dict[str, dict[int, int]] = {}
            for hash_int, query_offset in hashes:
                postings = self._hash_index.get(hash_int)
                if postings is None:
                    continue
                for song_id, db_offset in postings:
                    # Time alignment: bin by the difference in offsets (quantised to int seconds)
                    time_bin = int(round(db_offset - query_offset))
                    if song_id not in result:
                        result[song_id] = {}
                    result[song_id][time_bin] = result[song_id].get(time_bin, 0) + 1
            return result

    def lookup_embedding(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """FAISS search for the top-k most similar embeddings."""
        with self._lock:
            if self._faiss_index.ntotal == 0:
                return []
            query = query_vec.reshape(1, -1).astype(np.float32)
            k = min(top_k, self._faiss_index.ntotal)
            distances, indices = self._faiss_index.search(query, k)
            results: list[tuple[str, float]] = []
            for i in range(k):
                idx = int(indices[0][i])
                if 0 <= idx < len(self._song_ids):
                    results.append((self._song_ids[idx], float(distances[0][i])))
            return results
