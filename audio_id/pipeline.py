"""Query pipeline: VAD → normalize → extract → match."""

import io
import logging

import librosa
import numpy as np
import soundfile as sf
from fastapi import HTTPException

from audio_id.config import (
    MAX_QUERY_DURATION,
    MIN_QUERY_DURATION,
    SAMPLE_RATE,
    SILENCE_THRESHOLD,
)
from audio_id.extractor import FeatureExtractor
from audio_id.index import AudioIndex
from audio_id.matcher import Matcher

logger = logging.getLogger(__name__)


class QueryPipeline:
    """End-to-end query processing: validate → extract → match."""

    def __init__(self, extractor: FeatureExtractor, matcher: Matcher, index: AudioIndex,
                 vad_model=None) -> None:
        """Initialise the pipeline with all required components."""
        self._extractor = extractor
        self._matcher = matcher
        self._index = index
        self._vad_model = vad_model

    def process(self, audio_bytes: bytes) -> dict:
        """Process raw audio bytes and return a match result."""
        # 1. Decode audio
        try:
            audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception as exc:
            logger.warning("Failed to decode audio: %s", exc)
            raise HTTPException(status_code=400, detail=f"corrupt or unsupported audio: {exc}")

        # Handle stereo → mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # 2. Validate duration
        duration = len(audio) / sr
        if duration < MIN_QUERY_DURATION:
            raise HTTPException(
                status_code=400,
                detail=f"audio too short: {duration:.2f}s (minimum {MIN_QUERY_DURATION}s)",
            )

        # Truncate if too long (don't reject, just clip)
        if duration > MAX_QUERY_DURATION:
            max_samples = int(MAX_QUERY_DURATION * sr)
            audio = audio[:max_samples]

        # 3. Resample to target sample rate if needed
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        # 4. RMS check — reject silence
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < SILENCE_THRESHOLD:
            raise HTTPException(status_code=400, detail="silent audio")

        # 5. Silero-VAD check
        if self._vad_model is not None:
            try:
                self._vad_check(audio, sr)
            except HTTPException:
                raise
            except Exception as exc:
                logger.warning("VAD check failed, proceeding anyway: %s", exc)

        # 6. RMS normalise
        audio = audio / (rms + 1e-9)

        # 7. Extract features
        features = self._extractor.extract(audio, sr)

        # 8. Match
        result = self._matcher.match(features["fingerprints"], features["embedding"])

        return result

    def _vad_check(self, audio: np.ndarray, sr: int) -> None:
        """Run silero-VAD to detect speech/sound activity."""
        import torch

        # Silero-VAD expects 16kHz
        if sr != 16000:
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio

        tensor = torch.FloatTensor(audio_16k)

        # Get speech probabilities for chunks
        window_size = 512  # samples at 16kHz (32ms windows)
        speech_detected = False
        n_chunks = len(tensor) // window_size
        for i in range(n_chunks):
            chunk = tensor[i * window_size : (i + 1) * window_size]
            if len(chunk) < window_size:
                break
            prob = self._vad_model(chunk, 16000).item()
            if prob > 0.5:
                speech_detected = True
                break

        if not speech_detected and n_chunks > 0:
            # Check overall — some music may not trigger speech VAD
            # For music identification, we rely more on RMS check
            # Only reject if truly silent according to both RMS and VAD
            logger.debug("VAD did not detect speech/sound — proceeding (music may not trigger speech VAD)")
