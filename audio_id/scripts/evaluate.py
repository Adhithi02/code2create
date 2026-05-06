"""Local accuracy eval: SNR sweep, top-1 accuracy report."""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from audio_id.catalog import build_catalog
from audio_id.config import (
    DATASET_ROOT,
    DB_PATH,
    SAMPLE_RATE,
    SCORER_PATH,
)
from audio_id.extractor import FeatureExtractor
from audio_id.index import AudioIndex
from audio_id.matcher import Matcher
from audio_id.store import FeatureStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# SNR levels to evaluate
SNR_LEVELS = [0, 5, 10, 20]


def add_noise(audio: np.ndarray, snr_db: float, rng: np.random.RandomState) -> np.ndarray:
    """Add Gaussian noise at a given SNR level."""
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(max(noise_power, 1e-10)), len(audio))
    return (audio + noise).astype(np.float32)


def main() -> None:
    """Run evaluation: crop random windows, add noise, measure accuracy."""
    rng = np.random.RandomState(123)

    # Load components
    catalog = build_catalog(DATASET_ROOT)
    ok_songs = catalog[catalog["status"] == "ok"]

    if ok_songs.empty:
        logger.error("No songs found!")
        sys.exit(1)

    store = FeatureStore(DB_PATH)
    index = AudioIndex()
    index.build(store)

    extractor = FeatureExtractor()
    matcher = Matcher(index, SCORER_PATH, catalog)

    # Results tracking
    results: dict[int, dict[str, list[bool]]] = {snr: {} for snr in SNR_LEVELS}

    total = len(ok_songs)
    for idx, (_, row) in enumerate(ok_songs.iterrows()):
        song_id = row["song_id"]
        genre = row["genre"]
        path = row["path"]

        try:
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            # Resample
            if sr != SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
                sr = SAMPLE_RATE

            # Random crop: 5–8 seconds
            duration = len(audio) / sr
            crop_len = rng.uniform(5.0, min(8.0, duration))
            crop_samples = int(crop_len * sr)
            max_start = max(0, len(audio) - crop_samples)
            start_sample = rng.randint(0, max_start + 1)
            clip = audio[start_sample : start_sample + crop_samples]

            for snr_db in SNR_LEVELS:
                noisy_clip = add_noise(clip, snr_db, rng)
                features = extractor.extract(noisy_clip, sr)
                result = matcher.match(features["fingerprints"], features["embedding"])

                correct = result["song_id"] == song_id
                if genre not in results[snr_db]:
                    results[snr_db][genre] = []
                results[snr_db][genre].append(correct)

        except Exception as exc:
            logger.warning("Skipping %s: %s", song_id, exc)

        if (idx + 1) % 100 == 0:
            logger.info("Evaluated %d / %d songs", idx + 1, total)

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'Genre':<12}", end="")
    for snr in SNR_LEVELS:
        print(f"  {'SNR=' + str(snr) + 'dB':>10}", end="")
    print()
    print("-" * 70)

    for genre in sorted(set(g for snr_dict in results.values() for g in snr_dict)):
        print(f"{genre:<12}", end="")
        for snr in SNR_LEVELS:
            genre_results = results[snr].get(genre, [])
            if genre_results:
                acc = sum(genre_results) / len(genre_results)
                print(f"  {acc:>9.1%}", end="")
            else:
                print(f"  {'N/A':>9}", end="")
        print()

    print("-" * 70)
    print(f"{'OVERALL':<12}", end="")
    for snr in SNR_LEVELS:
        all_results = [v for genre_list in results[snr].values() for v in genre_list]
        if all_results:
            acc = sum(all_results) / len(all_results)
            print(f"  {acc:>9.1%}", end="")
        else:
            print(f"  {'N/A':>9}", end="")
    print()
    print("=" * 70)

    store.close()


if __name__ == "__main__":
    main()
