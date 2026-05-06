"""All constants, paths, and thresholds for the Audio ID system."""

from pathlib import Path

# ── Dataset & persistence paths ──────────────────────────────
DATASET_ROOT = Path("data/raw/data/data")
DB_PATH = Path("db/fingerprints.db")
FAISS_PATH = Path("db/embeddings.index")
CATALOG_PATH = Path("db/catalog.csv")
SCORER_PATH = Path("db/scorer.pkl")

# ── Audio processing parameters ──────────────────────────────
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_PEAKS = 20          # spectral peaks per frame
FAN_VALUE = 10        # combinatorial pairs per anchor peak
TIME_DELTA_MAX = 200  # max frame distance between paired peaks

# ── Query validation thresholds ──────────────────────────────
MIN_QUERY_DURATION = 2.0    # seconds — reject shorter clips
MAX_QUERY_DURATION = 10.0   # seconds
SILENCE_THRESHOLD = 0.01    # RMS below this → reject as silent

# ── Matching thresholds ──────────────────────────────────────
HASH_MATCH_THRESHOLD = 0.12    # hit_ratio above this → exact match
CONFIDENCE_FLOOR = 0.40        # below this → return "unknown"
EMBEDDING_DIM = 2048           # CNN14 output dimension
TOP_K_CANDIDATES = 5           # FAISS top-k for Stage 2

# ── Concurrency ─────────────────────────────────────────────
WORKER_COUNT = 4

# ── Genre list ───────────────────────────────────────────────
GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]
