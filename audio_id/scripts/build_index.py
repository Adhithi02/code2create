"""Offline: run once to populate SQLite + FAISS from dataset."""

import logging
import sys
import time
from pathlib import Path

import faiss
import joblib
import numpy as np
import soundfile as sf
from sklearn.linear_model import LogisticRegression

# Ensure the project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audio_id.catalog import build_catalog
from audio_id.config import (
    CATALOG_PATH,
    DATASET_ROOT,
    DB_PATH,
    EMBEDDING_DIM,
    FAISS_PATH,
    SAMPLE_RATE,
    SCORER_PATH,
)
from audio_id.extractor import FeatureExtractor
from audio_id.store import FeatureStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Build the full index: catalog → extract → store → train scorer → save FAISS."""
    start_time = time.time()

    # 1. Build catalog
    logger.info("Building catalog from %s", DATASET_ROOT)
    catalog = build_catalog(DATASET_ROOT)
    ok_songs = catalog[catalog["status"] == "ok"]
    logger.info("Catalog: %d total songs, %d ok", len(catalog), len(ok_songs))

    if ok_songs.empty:
        logger.error("No valid songs found! Check dataset path: %s", DATASET_ROOT)
        sys.exit(1)

    # Save catalog
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(CATALOG_PATH, index=False)

    # 2. Initialise store
    store = FeatureStore(DB_PATH)

    # 3. Initialise extractor (loads CNN14 model)
    logger.info("Loading feature extractor...")
    extractor = FeatureExtractor()

    # 4. Process each song
    total_postings = 0
    total_embeddings = 0
    processed = 0
    errors = 0

    for idx, row in ok_songs.iterrows():
        song_id = row["song_id"]
        genre = row["genre"]
        path = row["path"]

        try:
            # Load audio
            audio, sr = sf.read(path, dtype="float32")

            # Handle stereo → mono
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            # Resample if needed
            if sr != SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                sr = SAMPLE_RATE

            # Extract features
            features = extractor.extract(audio, sr)

            # Store
            store.upsert_song(song_id, genre, path)
            store.insert_postings(song_id, features["fingerprints"])
            store.insert_embedding(song_id, features["embedding"])

            total_postings += len(features["fingerprints"])
            total_embeddings += 1
            processed += 1

            if processed % 50 == 0:
                logger.info("Processed %d / %d songs", processed, len(ok_songs))

        except Exception as exc:
            errors += 1
            logger.error("Failed to process %s: %s", song_id, exc)

    logger.info("Extraction complete: %d processed, %d errors", processed, errors)
    logger.info("Total postings: %d, Total embeddings: %d", total_postings, total_embeddings)

    # 5. Train logistic regression scorer
    logger.info("Training confidence scorer...")
    _train_scorer(processed)

    # 6. Save FAISS index
    logger.info("Saving FAISS index...")
    song_ids, matrix = store.load_all_embeddings()
    if matrix.shape[0] > 0:
        faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        faiss_index.add(matrix.astype(np.float32))
        FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(faiss_index, str(FAISS_PATH))
        logger.info("FAISS index saved: %d vectors", faiss_index.ntotal)

    # Summary
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("INDEX BUILD COMPLETE")
    logger.info("Songs indexed: %d", processed)
    logger.info("Postings inserted: %d", total_postings)
    logger.info("Embeddings stored: %d", total_embeddings)
    logger.info("Time elapsed: %.1f seconds", elapsed)
    logger.info("=" * 60)

    store.close()


def _train_scorer(n_songs: int) -> None:
    """Train a LogisticRegression scorer on synthetic (hit_ratio, cosine_sim) pairs."""
    rng = np.random.RandomState(42)

    # Generate synthetic training data
    n_pos = max(n_songs * 2, 200)
    n_neg = max(n_songs * 2, 200)

    # Positive pairs: high hit_ratio, high cosine_sim
    pos_hit_ratios = rng.uniform(0.3, 1.0, n_pos)
    pos_cosine_sims = rng.uniform(0.5, 1.0, n_pos)

    # Negative pairs: low hit_ratio, low cosine_sim
    neg_hit_ratios = rng.uniform(0.0, 0.15, n_neg)
    neg_cosine_sims = rng.uniform(0.0, 0.45, n_neg)

    X = np.vstack([
        np.column_stack([pos_hit_ratios, pos_cosine_sims]),
        np.column_stack([neg_hit_ratios, neg_cosine_sims]),
    ])
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    scorer = LogisticRegression(random_state=42, max_iter=1000)
    scorer.fit(X, y)

    SCORER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scorer, SCORER_PATH)
    logger.info("Scorer trained and saved to %s", SCORER_PATH)
    logger.info("Scorer coefficients: %s, intercept: %s", scorer.coef_, scorer.intercept_)


if __name__ == "__main__":
    main()
