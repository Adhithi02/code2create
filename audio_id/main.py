"""FastAPI app factory + lifespan startup."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from audio_id.catalog import build_catalog
from audio_id.config import CATALOG_PATH, DATASET_ROOT, DB_PATH, SCORER_PATH
from audio_id.extractor import FeatureExtractor
from audio_id.index import AudioIndex
from audio_id.matcher import Matcher
from audio_id.metrics import INDEX_POSTINGS, INDEX_SONGS
from audio_id.pipeline import QueryPipeline
from audio_id.store import FeatureStore

from audio_id.routes.query import router as query_router
from audio_id.routes.health import router as health_router
from audio_id.routes.catalog_routes import router as catalog_router
from audio_id.routes.index_routes import router as index_router
from audio_id.routes.metrics_routes import router as metrics_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_vad_model():
    """Load silero-VAD model via torch.hub."""
    try:
        import torch
        model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        logger.info("Silero-VAD model loaded")
        return model
    except Exception as exc:
        logger.warning("Failed to load silero-VAD: %s — VAD check will be skipped", exc)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown logic."""
    # 1. Build catalog
    logger.info("Building song catalog from %s", DATASET_ROOT)
    catalog = build_catalog(DATASET_ROOT)
    app.state.catalog = catalog

    # Save catalog to CSV for reference
    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(CATALOG_PATH, index=False)

    # 2. Initialise store
    store = FeatureStore(DB_PATH)
    app.state.store = store

    # 3. Initialise index (empty)
    index = AudioIndex()
    app.state.index = index

    # 4. Try to build index from existing DB (warm start)
    if DB_PATH.exists() and store.song_count() > 0:
        logger.info("Found existing database — building index (warm start)")
        try:
            index.build(store)
            INDEX_SONGS.set(index.songs_count)
            INDEX_POSTINGS.set(index.postings_count)
        except Exception as exc:
            logger.error("Failed to build index from DB: %s", exc)
    else:
        logger.warning("No index found — run 'python -m audio_id.scripts.build_index' first")

    # 5. Load feature extractor (CNN14 — slow on first run)
    logger.info("Loading feature extractor (CNN14)...")
    extractor = FeatureExtractor()
    app.state.extractor = extractor

    # 6. Load silero-VAD
    vad_model = _load_vad_model()
    app.state.vad = vad_model

    # 7. Initialise matcher
    matcher = Matcher(index, SCORER_PATH, catalog)
    app.state.matcher = matcher

    # 8. Initialise pipeline
    pipeline = QueryPipeline(extractor, matcher, index, vad_model=vad_model)
    app.state.pipeline = pipeline

    logger.info("Startup complete — server ready")
    yield

    # Shutdown
    store.close()
    logger.info("Shutdown complete")


app = FastAPI(title="Audio ID System", lifespan=lifespan)

# Register all routers with no prefix
app.include_router(query_router)
app.include_router(health_router)
app.include_router(catalog_router)
app.include_router(index_router)
app.include_router(metrics_router)
