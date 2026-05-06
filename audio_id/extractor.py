"""Feature extraction: STFT peaks → hashes + CNN14 embeddings."""

import logging

import librosa
import numpy as np

from audio_id.config import (
    EMBEDDING_DIM,
    FAN_VALUE,
    HOP_LENGTH,
    N_FFT,
    N_PEAKS,
    SAMPLE_RATE,
    TIME_DELTA_MAX,
)

logger = logging.getLogger(__name__)

# CNN14 expected sample rate
_CNN14_SR = 32000


class FeatureExtractor:
    """Extracts Shazam-style fingerprints and CNN14 embeddings from audio."""

    def __init__(self) -> None:
        """Load the CNN14 model for embedding extraction."""
        from panns_inference import AudioTagging
        self._model = AudioTagging(checkpoint_path=None, device="cpu")
        logger.info("CNN14 model loaded for embedding extraction")

    def extract_fingerprint(self, audio: np.ndarray, sr: int) -> list[tuple[int, float]]:
        """Extract combinatorial hash fingerprints from audio."""
        # Compute STFT magnitude spectrogram
        stft = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
        n_freq_bins, n_frames = stft.shape

        # Find top N_PEAKS frequency indices per frame
        peaks_per_frame: list[list[int]] = []
        for frame_idx in range(n_frames):
            col = stft[:, frame_idx]
            if col.max() == 0:
                peaks_per_frame.append([])
                continue
            # Get indices of top N_PEAKS values
            n_actual = min(N_PEAKS, n_freq_bins)
            top_indices = np.argpartition(col, -n_actual)[-n_actual:]
            top_indices = top_indices[np.argsort(col[top_indices])[::-1]]
            peaks_per_frame.append(top_indices.tolist())

        # Combinatorial pairing: for each anchor peak, pair with peaks in future frames
        fingerprints: list[tuple[int, float]] = []
        for anchor_frame in range(n_frames):
            anchor_peaks = peaks_per_frame[anchor_frame]
            for f1 in anchor_peaks:
                pairs_made = 0
                for target_frame in range(anchor_frame + 1, min(anchor_frame + TIME_DELTA_MAX + 1, n_frames)):
                    if pairs_made >= FAN_VALUE:
                        break
                    target_peaks = peaks_per_frame[target_frame]
                    for f2 in target_peaks:
                        if pairs_made >= FAN_VALUE:
                            break
                        delta_t = target_frame - anchor_frame
                        # Hash formula: f1 << 20 | f2 << 10 | delta_t
                        hash_int = (int(f1) & 0x3FF) << 20 | (int(f2) & 0x3FF) << 10 | (int(delta_t) & 0x3FF)
                        time_offset = anchor_frame * HOP_LENGTH / sr
                        fingerprints.append((hash_int, time_offset))
                        pairs_made += 1

        return fingerprints

    def extract_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract a CNN14 embedding vector, L2-normalised."""
        # Resample to CNN14's expected sample rate
        if sr != _CNN14_SR:
            audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=_CNN14_SR)
        else:
            audio_resampled = audio

        # CNN14 expects (batch, samples) float32
        audio_input = audio_resampled[np.newaxis, :].astype(np.float32)
        _clipwise, embedding = self._model.inference(audio_input)

        # embedding shape is (1, EMBEDDING_DIM)
        vec = embedding[0].astype(np.float32)

        # L2 normalise
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        if vec.shape[0] != EMBEDDING_DIM:
            logger.warning("Embedding dim mismatch: expected %d, got %d", EMBEDDING_DIM, vec.shape[0])

        return vec

    def extract(self, audio: np.ndarray, sr: int) -> dict:
        """Single entry point: returns fingerprints and embedding."""
        fingerprints = self.extract_fingerprint(audio, sr)
        embedding = self.extract_embedding(audio, sr)
        return {"fingerprints": fingerprints, "embedding": embedding}
