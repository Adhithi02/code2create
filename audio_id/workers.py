"""ProcessPoolExecutor pool for CPU-bound extraction during index build."""

import logging
from concurrent.futures import Future, ProcessPoolExecutor

import numpy as np

from audio_id.config import WORKER_COUNT

logger = logging.getLogger(__name__)

# Module-level singleton for worker processes
_worker_extractor = None


def _init_worker_extractor():
    """Lazily initialise a FeatureExtractor in the worker process."""
    global _worker_extractor
    if _worker_extractor is None:
        from audio_id.extractor import FeatureExtractor
        _worker_extractor = FeatureExtractor()
    return _worker_extractor


def _extract_in_worker(audio: np.ndarray, sr: int) -> dict:
    """Run feature extraction in a worker process."""
    extractor = _init_worker_extractor()
    return extractor.extract(audio, sr)


class ExtractionPool:
    """Pool of worker processes for CPU-bound feature extraction."""

    def __init__(self, worker_count: int = WORKER_COUNT) -> None:
        """Create a ProcessPoolExecutor with the given worker count."""
        self._pool = ProcessPoolExecutor(max_workers=worker_count)
        logger.info("Extraction pool started with %d workers", worker_count)

    def submit_extract(self, audio: np.ndarray, sr: int) -> Future:
        """Submit an extraction job to the pool."""
        return self._pool.submit(_extract_in_worker, audio, sr)

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the pool."""
        self._pool.shutdown(wait=wait)
        logger.info("Extraction pool shut down")
