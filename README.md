# Audio Identification & Source Detection System

## Team Information
- **Team Name**: [Team Name]
- **Year**: [Year]
- **All-Female Team**: [Yes/No]

## Architecture Overview

#### Describe your approach here. Keep it short and clear.

    - We extract two complementary feature types: (1) Shazam-style combinatorial hash fingerprints from STFT spectral peaks, stored as integer hash buckets in SQLite with time-offset postings, and (2) 2048-dim CNN14 (PANNs) embeddings stored as binary blobs. This dual representation enables both exact and fuzzy matching.
    - We use a two-stage hybrid matching pipeline: Stage 1 performs hash-based histogram alignment (time-binned hit counting) for fast exact matches on clean audio. If the hit ratio is below threshold, Stage 2 falls back to FAISS IndexFlatIP cosine similarity search over CNN14 embeddings for noise-robust fuzzy matching. A logistic regression model calibrates confidence from both signals.
    - The architecture uses SQLite for persistent storage (O(1) hash lookups via indexed postings table), FAISS IndexFlatIP for embedding search (exact inner product, optimal for ≤10K vectors), and an in-memory hash dictionary rebuilt from SQLite on startup. ProcessPoolExecutor parallelises the one-time index build. FastAPI's async workers handle concurrent queries.
    - Low latency is achieved by running query-time extraction synchronously (avoiding process-pool overhead for short 3-10s clips), using pre-computed in-memory hash indices, and FAISS brute-force search (fast for 1K vectors). High accuracy under noise is ensured by the CNN14 embedding fallback (trained on AudioSet, robust to distortion), silero-VAD silence gating, RMS normalisation, and confidence calibration via a trained logistic regression scorer.

**Note:** Please do not change the format or spelling of anything in this README. The fields are extracted using a script, so any changes to the structure or formatting may break the extraction process.
