"""Two-stage matching engine + logistic regression confidence scorer."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from audio_id.config import (
    CONFIDENCE_FLOOR,
    HASH_MATCH_THRESHOLD,
    TOP_K_CANDIDATES,
)
from audio_id.index import AudioIndex

logger = logging.getLogger(__name__)


class Matcher:
    """Two-stage matcher: hash histogram (exact) then embedding cosine (fuzzy)."""

    def __init__(self, index: AudioIndex, scorer_path: Path, catalog: pd.DataFrame) -> None:
        """Load the scorer model if available; otherwise use linear fallback."""
        self._index = index
        self._catalog = catalog
        self._genre_lookup: dict[str, str] = {}
        if not catalog.empty:
            self._genre_lookup = dict(zip(catalog["song_id"], catalog["genre"]))

        self._scorer = None
        if scorer_path.exists():
            try:
                import joblib
                self._scorer = joblib.load(scorer_path)
                logger.info("Loaded confidence scorer from %s", scorer_path)
            except Exception as exc:
                logger.warning("Failed to load scorer from %s: %s — using fallback", scorer_path, exc)
        else:
            logger.info("No scorer found at %s — using linear fallback", scorer_path)

    def _get_genre(self, song_id: str) -> str | None:
        """Look up genre from catalog."""
        return self._genre_lookup.get(song_id)

    def _compute_confidence(self, hit_ratio: float, cosine_sim: float) -> float:
        """Compute confidence score using the trained scorer or linear fallback."""
        if self._scorer is not None:
            try:
                features = np.array([[hit_ratio, cosine_sim]])
                proba = self._scorer.predict_proba(features)[0][1]
                return float(np.clip(proba, 0.0, 1.0))
            except Exception as exc:
                logger.warning("Scorer prediction failed: %s — using fallback", exc)
        # Linear fallback
        return float(np.clip(0.5 * hit_ratio + 0.5 * cosine_sim, 0.0, 1.0))

    def match(self, fingerprints: list[tuple[int, float]], embedding: np.ndarray) -> dict:
        """Run two-stage matching and return the result dict."""
        n_fingerprints = len(fingerprints)

        # ── Stage 1: Hash histogram ──────────────────────────────
        histograms = self._index.lookup_hashes(fingerprints)

        best_song_id = None
        best_hit_ratio = 0.0
        best_aligned_hits = 0

        for song_id, time_bins in histograms.items():
            # Max aligned hits across all time bins
            if time_bins:
                max_hits = max(time_bins.values())
            else:
                max_hits = 0
            hit_ratio = max_hits / max(n_fingerprints, 1)
            if hit_ratio > best_hit_ratio:
                best_hit_ratio = hit_ratio
                best_song_id = song_id
                best_aligned_hits = max_hits

        if best_hit_ratio >= HASH_MATCH_THRESHOLD and best_song_id is not None:
            # Stage 1 pass — exact match
            # Also get cosine sim for confidence calibration
            cosine_sim = 0.0
            emb_results = self._index.lookup_embedding(embedding, TOP_K_CANDIDATES)
            for sid, sim in emb_results:
                if sid == best_song_id:
                    cosine_sim = sim
                    break
            confidence = self._compute_confidence(best_hit_ratio, max(cosine_sim, best_hit_ratio))
            confidence = max(confidence, best_hit_ratio)  # floor at hit_ratio for exact matches
            confidence = float(np.clip(confidence, 0.0, 1.0))
            genre = self._get_genre(best_song_id)
            return {
                "song_id": best_song_id,
                "genre": genre,
                "confidence": round(confidence, 4),
                "match_type": "exact",
            }

        # ── Stage 2: Embedding cosine similarity via FAISS ───────
        candidates = self._index.lookup_embedding(embedding, TOP_K_CANDIDATES)

        if not candidates:
            return {
                "song_id": None,
                "genre": None,
                "confidence": 0.0,
                "match_type": "unknown",
            }

        best_emb_song_id, cosine_sim = candidates[0]
        confidence = self._compute_confidence(best_hit_ratio, cosine_sim)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        if confidence >= CONFIDENCE_FLOOR:
            genre = self._get_genre(best_emb_song_id)
            return {
                "song_id": best_emb_song_id,
                "genre": genre,
                "confidence": round(confidence, 4),
                "match_type": "fuzzy",
            }

        # Below confidence floor → unknown
        return {
            "song_id": None,
            "genre": None,
            "confidence": round(confidence, 4),
            "match_type": "unknown",
        }
