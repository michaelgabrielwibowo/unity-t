"""
Incremental Feature Extraction with Ring-Buffer Caching
========================================================
Wraps each TRIBE v2 feature extractor (video, audio, text) so that
only *new* time slices are processed each step.  Previously extracted
features are cached in a ring buffer and concatenated to form the full
context window on demand.

This is the key to real-time performance: re‐extracting 40 s of features
every second would be infeasible, but extracting only the latest 1 s
chunk costs ~50–100 ms on a modern GPU.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ring Buffer for feature tensor slices
# ---------------------------------------------------------------------------

class FeatureRingBuffer:
    """Fixed-capacity ring buffer that stores 2-D numpy feature slices.

    Each slice has shape ``(feature_dim, time_steps)`` and represents
    the extracted features for one time chunk.  The buffer keeps the most
    recent *max_chunks* slices and can concatenate them along the time
    axis on demand.

    Thread-safe via a simple lock.
    """

    def __init__(self, max_chunks: int = 45):
        self._max_chunks = max_chunks
        self._buffer: Deque[np.ndarray] = deque(maxlen=max_chunks)
        self._lock = threading.Lock()
        self._total_appended: int = 0

    # -- mutators --------------------------------------------------------

    def append(self, chunk: np.ndarray) -> None:
        """Append a feature chunk.  Oldest chunk is dropped if full."""
        with self._lock:
            self._buffer.append(chunk)
            self._total_appended += 1

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._total_appended = 0

    # -- accessors -------------------------------------------------------

    def concatenated(self) -> Optional[np.ndarray]:
        """Return all cached slices concatenated along the time axis.

        Returns ``None`` if the buffer is empty.
        Resulting shape: ``(feature_dim, total_time_steps)``.
        """
        with self._lock:
            if len(self._buffer) == 0:
                return None
            return np.concatenate(list(self._buffer), axis=-1)

    @property
    def num_chunks(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def total_appended(self) -> int:
        with self._lock:
            return self._total_appended

    @property
    def is_full(self) -> bool:
        with self._lock:
            return len(self._buffer) >= self._max_chunks

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


# ---------------------------------------------------------------------------
# Incremental Feature Extractor
# ---------------------------------------------------------------------------

@dataclass
class IncrementalFeatureExtractor:
    """Wraps a TRIBE v2 feature extractor for incremental processing.

    Instead of re-processing the entire 40 s context window each step,
    this class:

    1. Accepts only the *new* raw data slice (1 s of video frames,
       audio samples, or text tokens).
    2. Runs the corresponding extractor on that slice.
    3. Caches the resulting feature tensor in a ``FeatureRingBuffer``.
    4. On request, returns the full window by concatenating cached slices.

    Parameters
    ----------
    name : str
        Modality name (``"video"``, ``"audio"``, ``"text"``).
    extractor : Any
        A TRIBE v2 ``neuralset`` extractor instance.
    max_chunks : int
        How many 1-s feature chunks to retain (should match
        ``window_sec / stride_sec``).
    device : str
        Torch device for extraction.
    """

    name: str
    extractor: Any  # ns.extractors.BaseExtractor
    max_chunks: int = 45
    device: str = "cuda"
    _cache: FeatureRingBuffer = field(init=False)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _prepared: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._cache = FeatureRingBuffer(max_chunks=self.max_chunks)

    # -- public API ------------------------------------------------------

    def prepare(self, sample_events) -> None:
        """One-time preparation (loads model weights, etc.)."""
        with self._lock:
            if not self._prepared:
                logger.info("Preparing extractor: %s", self.name)
                self.extractor.prepare(sample_events)
                self._prepared = True

    def extract_chunk(self, raw_data: Any, events_slice) -> np.ndarray:
        """Extract features for one time chunk and cache the result.

        Parameters
        ----------
        raw_data
            Raw input for this chunk (frames, audio samples, text tokens).
        events_slice : pd.DataFrame
            Events DataFrame slice covering this chunk's time span.

        Returns
        -------
        np.ndarray
            Extracted feature tensor for this chunk.
        """
        with self._lock:
            # Use the extractor's internal pipeline
            with torch.inference_mode():
                features = self._extract_impl(raw_data, events_slice)
            self._cache.append(features)
            return features

    def get_full_window(self) -> Optional[np.ndarray]:
        """Return the full feature window by concatenating cached chunks."""
        return self._cache.concatenated()

    def get_window_tensor(self) -> Optional[torch.Tensor]:
        """Return the full feature window as a GPU tensor."""
        arr = self._cache.concatenated()
        if arr is None:
            return None
        return torch.from_numpy(arr).to(self.device)

    @property
    def num_cached(self) -> int:
        return self._cache.num_chunks

    @property
    def is_warm(self) -> bool:
        """Whether the buffer has enough data for a full context window."""
        return self._cache.is_full

    def reset(self) -> None:
        """Clear all cached features."""
        self._cache.clear()

    # -- internal --------------------------------------------------------

    def _extract_impl(self, raw_data: Any, events_slice) -> np.ndarray:
        """Override-friendly extraction implementation.

        The default implementation calls ``self.extractor`` on the events.
        Subclasses may override for custom per-modality logic.
        """
        # For the generic case, the extractor operates on an events
        # DataFrame and returns features via its __call__ interface.
        # In production, each modality subclass would specialise this.
        try:
            result = self.extractor(events_slice)
            if isinstance(result, torch.Tensor):
                return result.detach().cpu().numpy()
            return np.asarray(result, dtype=np.float32)
        except Exception as exc:
            logger.warning(
                "Feature extraction failed for %s: %s — returning zeros",
                self.name,
                exc,
            )
            # Return a zero tensor of shape (feature_dim, expected_timesteps)
            # The caller should handle gracefully.
            return np.zeros((1, 1), dtype=np.float32)


# ---------------------------------------------------------------------------
# Modality-specific subclasses
# ---------------------------------------------------------------------------

class VideoFeatureExtractor(IncrementalFeatureExtractor):
    """Incremental V-JEPA2 feature extraction for video frames."""

    def _extract_impl(self, raw_data: Any, events_slice) -> np.ndarray:
        """
        raw_data: np.ndarray of shape (N, H, W, 3) — N frames
        Returns: np.ndarray of shape (num_layers, feature_dim, time_steps)
        """
        try:
            result = self.extractor(events_slice)
            if isinstance(result, torch.Tensor):
                return result.detach().cpu().numpy()
            return np.asarray(result, dtype=np.float32)
        except Exception as exc:
            logger.warning("Video extraction failed: %s", exc)
            return np.zeros((1, 1, 1), dtype=np.float32)


class AudioFeatureExtractor(IncrementalFeatureExtractor):
    """Incremental Wav2Vec-BERT feature extraction for audio."""

    def _extract_impl(self, raw_data: Any, events_slice) -> np.ndarray:
        """
        raw_data: np.ndarray of shape (num_samples,) — 16kHz mono audio
        Returns: np.ndarray of shape (num_layers, feature_dim, time_steps)
        """
        try:
            result = self.extractor(events_slice)
            if isinstance(result, torch.Tensor):
                return result.detach().cpu().numpy()
            return np.asarray(result, dtype=np.float32)
        except Exception as exc:
            logger.warning("Audio extraction failed: %s", exc)
            return np.zeros((1, 1, 1), dtype=np.float32)


class TextFeatureExtractor(IncrementalFeatureExtractor):
    """Incremental LLaMA 3.2 feature extraction for text/words."""

    def _extract_impl(self, raw_data: Any, events_slice) -> np.ndarray:
        """
        raw_data: list[str] — word tokens with timing
        Returns: np.ndarray of shape (num_layers, feature_dim, time_steps)
        """
        try:
            result = self.extractor(events_slice)
            if isinstance(result, torch.Tensor):
                return result.detach().cpu().numpy()
            return np.asarray(result, dtype=np.float32)
        except Exception as exc:
            logger.warning("Text extraction failed: %s", exc)
            return np.zeros((1, 1, 1), dtype=np.float32)
