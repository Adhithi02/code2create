"""Dataset scanner: builds song catalog from genre folders."""

import logging
from pathlib import Path

import pandas as pd
import soundfile as sf

from audio_id.config import GENRES

logger = logging.getLogger(__name__)


def build_catalog(dataset_root: Path) -> pd.DataFrame:
    """Walk genre folders and build a DataFrame of all songs."""
    records: list[dict] = []
    for genre in GENRES:
        genre_dir = dataset_root / genre
        if not genre_dir.is_dir():
            logger.warning("Genre directory not found: %s", genre_dir)
            continue
        for wav_path in sorted(genre_dir.glob("*.wav")):
            song_id = wav_path.stem
            status = "ok"
            try:
                info = sf.info(str(wav_path))
                if info.frames == 0:
                    status = "error"
                    logger.warning("Empty audio file: %s", wav_path)
            except Exception as exc:
                status = "error"
                logger.warning("Cannot read %s: %s", wav_path, exc)
            records.append({
                "song_id": song_id,
                "genre": genre,
                "path": str(wav_path.resolve()),
                "status": status,
            })
    df = pd.DataFrame(records, columns=["song_id", "genre", "path", "status"])
    logger.info("Catalog built: %d songs (%d ok, %d error)",
                len(df), (df["status"] == "ok").sum(), (df["status"] == "error").sum())
    return df
